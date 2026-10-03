"""Tests for plasticity.analysis (aggregate / plots / tables) and scripts/analyze.py on a synthetic results tree."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image
from scipy import stats as sps

from plasticity.analysis import aggregate as A
from plasticity.analysis import plots as P
from plasticity.analysis import tables as T
from plasticity.metrics.summary import summarize_run
from plasticity.utils import append_jsonl, dump_json

ROOT = Path(__file__).resolve().parents[1]
N_TASKS, SEEDS = 10, (0, 1, 2)
MECH = ("dead_frac/all", "inactive_frac/all", "w_abs_mean/all", "w_fro_norm/all", "stable_rank/last", "effective_rank/last",
        "stable_rank_c/last", "effective_rank_c/last", "jac_sq_mean/all", "w_over_A_mean/all", "sat_frac/all", "grad_norm_theta",
        "grad_norm_w", "fisher_trace_theta", "fisher_trace_w", "fisher_erank_theta", "fisher_erank_w", "jac_fro_ratio")
FRESH_TASKS = (0, 5, 9)


def _rows(label: str, seed: int):
    rng = np.random.default_rng(100 * seed + (1 if label == "sin" else 2))
    slope = 0.002 if label == "sin" else 0.02
    rows = []
    for t in range(N_TASKS):
        acc = 0.85 - slope * t + 0.01 * rng.standard_normal()
        r = {"task": t, "time": 0.5, "online_accuracy": float(acc), "online_loss": float(1 - acc), "n_train": 100,
             "step_grad_norm_theta": float(abs(rng.standard_normal())), "step_dw_norm": 0.01, "step_dtheta_norm": 0.01,
             "method/kappa": 2.0}
        if t % 2 == 0:  # mechanism metrics every 2 tasks
            for i, k in enumerate(MECH):
                r[k] = float(0.1 * i + 0.01 * t + 0.001 * rng.standard_normal())
        if t in FRESH_TASKS:
            r["fresh_accuracy"] = float(0.86 + 0.01 * rng.standard_normal())
        rows.append(r)
    return rows


def _cfg(label: str, seed: int, mode: str, lr: float | None = None):
    if lr is None:
        lr = 0.01 if label == "sin" else 0.03
    return {"seed": seed, "model": {"name": "mlp", "hidden_sizes": [10, 10], "reparam": {"hidden": mode, "output": mode}, "gamma": 1.5},
            "method": {"name": "baseline"}, "optimizer": {"name": "sgd", "lr": lr},
            "stream": {"name": "pmnist", "n_tasks": N_TASKS}}


def _write_run(d: Path, label: str, seed: int, mode: str, finished: bool = True, lr: float | None = None):
    d.mkdir(parents=True, exist_ok=True)
    dump_json(_cfg(label, seed, mode, lr=lr), d / "config.json")
    rows = _rows(label.split("/")[0], seed)
    for r in rows:
        append_jsonl(r, d / "tasks.jsonl")
    if finished:
        s = summarize_run(rows, "online_accuracy", window=0.2)
        s.update({"seed": seed, "method": "baseline", "stream": "pmnist", "wall_time": 12.5})
        dump_json(s, d / "summary.json")
    return rows


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("results") / "synthetic"
    for label, mode in (("sin", "sin"), ("baseline", "standard")):
        for s in SEEDS:
            _write_run(root / label / f"seed{s}", label, s, mode)
    _write_run(root / "unfinished" / "seed0", "unfinished", 0, "standard", finished=False)
    return root


@pytest.fixture(scope="module")
def loaded(suite):
    P.reset_palette()
    runs, curves = A.load_suite(suite, verbose=False)
    return runs, curves


SWEEP_LRS = (0.001, 0.01, 0.1)  # two decades -> log x-axis


@pytest.fixture(scope="module")
def sweep_suite(tmp_path_factory):
    """scripts/run_suite.py-style sweep layout: results/<suite>/<base>/lr=<v>/seed<k>."""
    root = tmp_path_factory.mktemp("results") / "sweep"
    for base, mode in (("sin", "sin"), ("baseline", "standard")):
        for lr in SWEEP_LRS:
            for s in (0, 1):
                _write_run(root / base / f"lr={lr}" / f"seed{s}", f"{base}/lr={lr}", s, mode, lr=lr)
    return root


# ---------------------------------------------------------------- aggregate
def test_discover_runs_nested(suite):
    found = A.discover_runs(suite)
    assert len(found) == 7
    assert [f["label"] for f in found] == ["baseline"] * 3 + ["sin"] * 3 + ["unfinished"]
    assert [f["seed"] for f in found] == [0, 1, 2, 0, 1, 2, 0]
    assert sum(f["finished"] for f in found) == 6


def test_load_suite_runs_and_curves(loaded):
    runs, curves = loaded
    assert runs.shape[0] == 6
    for c in ("label", "seed", "run_dir", "run_path", "auc_norm", "early_window_mean", "final_window_mean", "retention_ratio",
              "fresh_gap_final", "final_dead_frac/all", "cfg.optimizer.lr", "cfg.model.reparam.hidden", "cfg.seed", "wall_time"):
        assert c in runs.columns, c
    assert runs["cfg.model.hidden_sizes"].iloc[0] == "[10, 10]"  # lists are JSON strings (hashable, csv-stable)
    assert sorted(runs["seed"].tolist()) == [0, 0, 1, 1, 2, 2]
    assert runs.attrs["skipped"] == ["unfinished/seed0"]
    assert curves.shape[0] == 2 * len(SEEDS) * N_TASKS
    for c in ("label", "seed", "task", "online_accuracy", "online_loss", "dead_frac/all", "effective_rank/last", "method/kappa", "fresh_accuracy"):
        assert c in curves.columns, c
    assert curves["dead_frac/all"].isna().sum() == 2 * len(SEEDS) * N_TASKS // 2
    assert curves["fresh_accuracy"].notna().sum() == 2 * len(SEEDS) * len(FRESH_TASKS)
    assert (curves.groupby(["label", "seed"])["task"].count() == N_TASKS).all()
    runs2, curves2, skipped = A.load_suite(suite_root(runs), labels=["sin"], verbose=False, return_skipped=True)
    assert set(runs2["label"]) == {"sin"} and set(curves2["label"]) == {"sin"} and skipped == []


def suite_root(runs):
    return runs.attrs["root"]


def test_load_suite_flat_layout_and_unfinished(tmp_path):
    _write_run(tmp_path / "flat_run", "flat_run", 7, "tanh")
    _write_run(tmp_path / "partial", "partial", 3, "sin", finished=False)
    runs, curves = A.load_suite(tmp_path, verbose=False)
    assert runs["label"].tolist() == ["flat_run"] and runs["seed"].tolist() == [7]
    assert set(curves["label"]) == {"flat_run"}
    runs, curves = A.load_suite(tmp_path, verbose=False, include_unfinished=True)
    assert set(curves["label"]) == {"flat_run", "partial"} and len(runs) == 1
    assert A.load_suite(tmp_path / "flat_run", verbose=False)[0]["label"].tolist() == ["flat_run"]  # root itself is a run
    with pytest.raises(FileNotFoundError):
        A.load_suite(tmp_path / "nope")


def test_cache_roundtrip(loaded, tmp_path):
    runs, curves = loaded
    paths = A.save_tables(runs, curves, tmp_path / "cache", fmt="csv")
    assert paths["runs"].is_file() and paths["curves"].is_file()
    r2, c2 = A.load_suite(suite_root(runs), cache=tmp_path / "cache", verbose=False)
    assert len(r2) == len(runs) and len(c2) == len(curves)
    assert np.allclose(np.sort(r2["auc_norm"].to_numpy()), np.sort(runs["auc_norm"].to_numpy()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # falls back to csv (with a warning) when no parquet engine is installed
        p2 = A.save_tables(runs, curves, tmp_path / "cache2", fmt="parquet")
    assert p2["runs"].is_file() and p2["runs"].suffix in (".parquet", ".csv")
    with pytest.raises(ValueError):
        A.save_tables(runs, curves, tmp_path / "cache3", fmt="xlsx")


def test_mean_curves_t_ci_and_smoothing(loaded):
    runs, curves = loaded
    mc = A.mean_curves(curves, "online_accuracy")
    assert list(mc.columns) == ["label", "task", "n", "mean", "std", "ci_low", "ci_high"]
    assert len(mc) == 2 * N_TASKS and (mc["n"] == 3).all()
    x = curves[(curves.label == "baseline") & (curves.task == 4)]["online_accuracy"].to_numpy()
    row = mc[(mc.label == "baseline") & (mc.task == 4)].iloc[0]
    half = sps.t.ppf(0.975, 2) * x.std(ddof=1) / np.sqrt(3)
    assert row["mean"] == pytest.approx(x.mean()) and row["ci_low"] == pytest.approx(x.mean() - half) and row["ci_high"] == pytest.approx(x.mean() + half)
    # mechanism metric logged every 2 tasks -> 5 points per label, NaNs dropped
    mm = A.mean_curves(curves, "dead_frac/all")
    assert len(mm) == 2 * (N_TASKS // 2) and sorted(mm["task"].unique()) == [0, 2, 4, 6, 8]
    # smoothing: per-seed centred rolling mean before aggregation
    ms = A.mean_curves(curves, "online_accuracy", smooth=3)
    per_seed = []
    for s in SEEDS:
        v = curves[(curves.label == "sin") & (curves.seed == s)].sort_values("task")["online_accuracy"].to_numpy()
        per_seed.append(v[4:7].mean())
    assert ms[(ms.label == "sin") & (ms.task == 5)]["mean"].iloc[0] == pytest.approx(np.mean(per_seed))
    # ordering follows `labels`; missing metric -> empty frame with the right columns; single seed -> NaN CI
    assert A.mean_curves(curves, "online_accuracy", labels=["sin", "baseline"])["label"].iloc[0] == "sin"
    empty = A.mean_curves(curves, "test_accuracy")
    assert len(empty) == 0 and list(empty.columns) == list(mc.columns)
    one = A.mean_curves(curves[curves.seed == 0], "online_accuracy")
    assert one["ci_low"].isna().all() and (one["n"] == 1).all()


def test_base_label_and_label_matching(loaded, sweep_suite, tmp_path):
    runs, curves = loaded
    assert "base_label" in runs.columns and "base_label" in curves.columns and (runs["base_label"] == runs["label"]).all()
    assert A.base_label("sin/lr=0.01/kappa=2") == "sin" and A.base_label("sin") == "sin"
    present = ["baseline/lr=0.001", "baseline/lr=0.01", "sin/lr=0.001", "sin/lr=0.01", "cbp"]
    assert A.match_labels(["sin", "cbp", "ghost", "base"], present) == (["sin/lr=0.001", "sin/lr=0.01", "cbp"], ["ghost", "base"])
    assert A.match_labels(["sin/lr=0.01", "sin"], present)[0] == ["sin/lr=0.01", "sin/lr=0.001"]  # request order, no duplicates
    found = A.discover_runs(sweep_suite)
    assert len(found) == 12 and {f["base_label"] for f in found} == {"sin", "baseline"} and found[0]["label"] == "baseline/lr=0.001"
    r, c = A.load_suite(sweep_suite, verbose=False)
    assert len(r) == 12 and sorted(r["base_label"].unique()) == ["baseline", "sin"] and set(c["base_label"]) == {"baseline", "sin"}
    assert sorted(r.loc[r.base_label == "sin", "cfg.optimizer.lr"].unique()) == list(SWEEP_LRS)
    # a base label selects its sweep variants; unmatched requests are reported, never fatal
    r2, c2 = A.load_suite(sweep_suite, labels=["sin", "ghost"], verbose=False)
    assert sorted(r2["label"].unique()) == [f"sin/lr={lr}" for lr in SWEEP_LRS] and set(c2["base_label"]) == {"sin"}
    assert r2.attrs["unmatched_labels"] == ["ghost"]
    assert len(A.load_suite(sweep_suite, labels=["sin/lr=0.01"], verbose=False)[0]) == 2
    # the same matching on the cached tables (also tables cached without a base_label column)
    cache = tmp_path / "cache"
    A.save_tables(r.drop(columns=["base_label"]), c.drop(columns=["base_label"]), cache, fmt="csv")
    r3, c3 = A.load_suite(sweep_suite, labels=["baseline", "nope"], cache=cache, verbose=False)
    assert set(r3["base_label"]) == {"baseline"} == set(c3["base_label"]) and len(r3) == 6 and r3.attrs["unmatched_labels"] == ["nope"]


def test_auto_metric_and_fresh(loaded):
    runs, curves = loaded
    assert A.auto_metric(runs, curves) == "online_accuracy"
    assert A.auto_metric(runs.drop(columns=["metric"]), curves) == "online_accuracy"
    assert A.has_fresh_data(curves) and not A.has_fresh_data(curves.drop(columns=["fresh_accuracy"]))


# ---------------------------------------------------------------- plots: palette and text
def test_palette_pinned_and_stable(loaded):
    runs, _ = loaded
    P.reset_palette()
    pal = P.make_palette(["baseline", "cbp", "sin", "tanh"], runs=runs)
    assert pal["sin"] == P.PALETTE[0] and pal["tanh"] == P.PALETTE[1] and pal["baseline"] == P.PALETTE[2] and pal["cbp"] == P.PALETTE[3]
    pal2 = P.make_palette(["cbp", "sin"])  # a different call/order keeps the colours
    assert pal2 == {"cbp": pal["cbp"], "sin": pal["sin"]}
    assert P.order_labels(["cbp", "baseline", "tanh", "sin"], runs=runs) == ["sin", "tanh", "baseline", "cbp"]
    assert P.infer_modes(runs, ["sin", "baseline", "l2_init"]) == {"sin": "sin", "baseline": "standard", "l2_init": None}
    P.reset_palette()
    with pytest.warns(UserWarning):
        big = P.make_palette([f"m{i}" for i in range(10)])
    assert len(set(big.values())) == 9 and big["m8"] == P.OVERFLOW_COLOR == big["m9"]
    assert P._linestyle("m8") != P._linestyle("m9")
    P.reset_palette()
    assert P.make_palette(["x"], colors={"x": "#123456"}) == {"x": "#123456"}
    P.reset_palette()


def test_palette_reserves_pinned_slots_and_shares_with_base_label():
    P.reset_palette()
    pal = P.make_palette(["sin", "cbp", "l2_init", "upgd"])  # no tanh / baseline in this suite
    assert pal["sin"] == P.PALETTE[0]
    assert [pal["cbp"], pal["l2_init"], pal["upgd"]] == [P.PALETTE[3], P.PALETTE[4], P.PALETTE[5]]  # orange / aqua stay free
    assert P.PALETTE[1] not in pal.values() and P.PALETTE[2] not in pal.values()
    assert P.make_palette(["tanh", "baseline"]) == {"tanh": P.PALETTE[1], "baseline": P.PALETTE[2]}  # still available later
    P.reset_palette()
    pal = P.make_palette(["a", "b", "c", "d", "e", "f"])  # six unpinned labels: five free slots, then the reserved ones
    assert [pal[k] for k in "abcde"] == [P.PALETTE[i] for i in (3, 4, 5, 6, 7)] and pal["f"] == P.PALETTE[0]
    P.reset_palette()
    sweep = P.make_palette(["sin/lr=0.001", "sin/lr=0.01", "baseline/lr=0.001"])
    assert sweep["sin/lr=0.001"] == P.PALETTE[0] and sweep["sin/lr=0.01"] == P.PALETTE[3] and sweep["baseline/lr=0.001"] == P.PALETTE[2]
    base = P.make_palette(["sin", "baseline"])  # the base labels of the sweep share their variants' pinned colours
    assert base == {"sin": P.PALETTE[0], "baseline": P.PALETTE[2]}
    P.reset_palette()


def test_label_text_languages():
    assert P.label_text("task") == "Görev" and P.label_text("task", "en") == "Task"
    assert P.label_text("online_accuracy") == "Çevrimiçi doğruluk" and P.label_text("test_accuracy") == "Test doğruluğu"
    assert P.label_text("final_dead_frac/all").startswith("Son: ") and P.label_text("final_dead_frac/all", "en").startswith("Final: ")
    assert P.label_text("cfg.optimizer.lr") == "Öğrenme oranı" and P.label_text("method/kappa") == "kappa"
    assert P.label_text("unknown_key", "en") == "unknown_key"


# ---------------------------------------------------------------- plots: every figure function
def test_plot_functions_return_figures(loaded, tmp_path):
    runs, curves = loaded
    P.reset_palette()
    figs = {}
    figs["perf"] = P.plot_performance_curves(curves, "online_accuracy", runs=runs, smooth=3)
    figs["perf_en"] = P.plot_performance_curves(curves, "online_accuracy", labels=["sin"], lang="en", title="t", ylabel="y")
    figs["mech"] = P.plot_mechanism_panels(curves, keys=["dead_frac/all", "w_abs_mean/all", "effective_rank/last", "grad_norm_w"], runs=runs, ncols=2)
    figs["mech_default"] = P.plot_mechanism_panels(curves, runs=runs, smooth=2)
    figs["dots"] = P.plot_summary_dots(runs, "auc_norm", reference="baseline")
    figs["fresh"] = P.plot_fresh_gap(curves, runs=runs)
    figs["sweep"] = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm", group_key="cfg.model.reparam.hidden")
    figs["sweep_label"] = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm", lang="en")
    for name, fig in figs.items():
        assert isinstance(fig, Figure), name
    ax = figs["perf"].axes[0]
    assert len(ax.lines) == 2 and ax.get_xlabel() == "Görev" and ax.get_ylabel() == "Çevrimiçi doğruluk" and ax.get_legend() is not None
    assert {l.get_color() for l in ax.lines} == {P.PALETTE[0], P.PALETTE[2]}
    assert figs["perf_en"].axes[0].get_legend() is None  # single series: no legend box
    assert sum(a.get_visible() for a in figs["mech"].axes) == 4
    assert figs["dots"].axes[0].get_yticklabels()[0].get_text() in ("sin", "baseline")
    paths = P.save_figure(figs["perf"], tmp_path / "performance.png", close=True)
    assert [p.suffix for p in paths] == [".png", ".pdf"] and all(p.stat().st_size > 1000 for p in paths)
    # the same colour per label across figures
    fresh_lines = [l for l in figs["fresh"].axes[0].lines if l.get_linestyle() == "-" and l.get_label() in ("sin", "baseline")]
    assert {l.get_label(): l.get_color() for l in fresh_lines} == {"sin": P.PALETTE[0], "baseline": P.PALETTE[2]}
    for fig in figs.values():
        plt.close(fig)


def test_style_scoped_note_below_axes_and_outside_legend(loaded):
    runs, curves = loaded
    P.reset_palette()
    before = {k: matplotlib.rcParams[k] for k in P.STYLE}
    fig = P.plot_fresh_gap(curves, runs=runs)  # long Turkish title + the longest note
    after = {k: matplotlib.rcParams[k] for k in P.STYLE}
    assert after == before  # importing / plotting never mutates the global rcParams
    assert matplotlib.rcParams["savefig.bbox"] != "tight" or before["savefig.bbox"] == "tight"
    ax = fig.axes[0]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    note = [t for t in ax.texts if t.get_gid() == "ci_note"]
    assert len(note) == 1 and "n = 3" in note[0].get_text()
    nb, tb, xb = note[0].get_window_extent(rend), ax._left_title.get_window_extent(rend), ax.xaxis.label.get_window_extent(rend)
    assert not nb.overlaps(tb) and nb.y1 <= xb.y0  # below the x-axis label, never on the title
    assert ax.get_facecolor()[:3] == matplotlib.colors.to_rgb(P.SURFACE) and not ax.spines["top"].get_visible()
    # many series: the legend goes outside and the figure we created is widened to make room
    many = pd.concat([curves.assign(label=f"m{i}") for i in range(6)], ignore_index=True)
    fig2 = P.plot_performance_curves(many, "online_accuracy", figsize=(6.0, 3.6))
    assert fig2.get_figwidth() == pytest.approx(6.0 + P.LEGEND_EXTRA_WIDTH) and fig2.axes[0].get_legend() is not None
    fig3 = P.plot_performance_curves(curves, "online_accuracy", figsize=(6.0, 3.6))
    assert fig3.get_figwidth() == pytest.approx(6.0)  # two series: legend inside, width untouched
    # the fresh-gap figure: two legends inside for a few labels, one merged outside legend for many; every legend
    # (also one added with add_artist) must stay inside the saved tight bbox at any dpi
    import io
    legs_few = [a for a in fig.axes[0].get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legs_few) == 2 and all(not l.get_clip_on() or l is fig.axes[0].legend_ for l in legs_few)
    fig4 = P.plot_fresh_gap(many, runs=None)
    legs = [a for a in fig4.axes[0].get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legs) == 1 and fig4.get_figwidth() == pytest.approx(6.0 + P.LEGEND_EXTRA_WIDTH)
    assert [t.get_text() for t in legs[0].get_texts()][-2:] == ["sürekli", "taze model"] and len(legs[0].get_texts()) == 6 + 1 + 2

    def png_size(**kw):
        buf = io.BytesIO()
        fig4.savefig(buf, format="png", dpi=60, bbox_inches="tight", **kw)
        return Image.open(buf).size

    assert png_size()[0] == png_size(bbox_extra_artists=legs)[0]  # same width: the outside legend is inside the tight bbox
    for f in (fig, fig2, fig3, fig4):
        plt.close(f)
    P.reset_palette()


def test_plot_sweep_groups_by_base_label(sweep_suite):
    runs, _ = A.load_suite(sweep_suite, verbose=False)
    P.reset_palette()
    fig = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm")  # default: one sensitivity line per base label
    lines = {l.get_label(): l for l in fig.axes[0].lines}
    assert set(lines) == {"sin", "baseline"} and all(len(l.get_xdata()) == len(SWEEP_LRS) for l in lines.values())
    assert list(lines["sin"].get_xdata()) == list(SWEEP_LRS)
    assert lines["sin"].get_color() == P.PALETTE[0] and lines["baseline"].get_color() == P.PALETTE[2]
    assert fig.axes[0].get_legend() is not None and fig.axes[0].get_xscale() == "log"
    fig2 = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm", group_key="label")  # degenerate -> falls back to base_label
    assert sorted(l.get_label() for l in fig2.axes[0].lines) == ["baseline", "sin"] and all(len(l.get_xdata()) == 3 for l in fig2.axes[0].lines)
    fig3 = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm", group_key="cfg.model.reparam.hidden", labels=["sin"])
    assert [l.get_label() for l in fig3.axes[0].lines] == ["sin"]  # labels restrict the runs, grouping stays by the column
    fig4 = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm", labels=["baseline", "sin/lr=0.01"])
    assert [l.get_label() for l in fig4.axes[0].lines] == ["baseline", "sin"] and len(fig4.axes[0].lines[1].get_xdata()) == 1
    fig5 = P.plot_sweep(runs, "cfg.optimizer.lr", "auc_norm", group_key=None)
    assert [l.get_label() for l in fig5.axes[0].lines] == ["all"]
    for f in (fig, fig2, fig3, fig4, fig5):
        plt.close(f)
    P.reset_palette()


def test_plots_robust_to_missing_metrics(loaded):
    runs, curves = loaded
    fig = P.plot_performance_curves(curves, "test_accuracy", runs=runs)
    assert isinstance(fig, Figure) and len(fig.axes[0].lines) == 0
    fig2 = P.plot_mechanism_panels(curves, keys=["nope/all", "dead_frac/all"], runs=runs)
    assert sum(a.get_visible() for a in fig2.axes) == 1
    fig3 = P.plot_mechanism_panels(curves, keys=["nope/all"], runs=runs)
    fig4 = P.plot_summary_dots(runs, "final_test_accuracy")
    fig5 = P.plot_sweep(runs, "cfg.nothing", "auc_norm")
    with pytest.raises(ValueError):
        P.plot_fresh_gap(curves.drop(columns=["fresh_accuracy"]))
    for f in (fig, fig2, fig3, fig4, fig5):
        plt.close(f)


# ---------------------------------------------------------------- tables
def test_summary_table_and_markdown(loaded, tmp_path):
    runs, _ = loaded
    long = T.summary_table(runs, metrics=["auc_norm", "retention_ratio", "final_dead_frac/all", "not_a_metric"])
    assert set(long["metric"]) == {"auc_norm", "retention_ratio", "final_dead_frac/all"}
    row = long[(long.label == "sin") & (long.metric == "auc_norm")].iloc[0]
    x = runs[runs.label == "sin"]["auc_norm"].to_numpy()
    half = sps.t.ppf(0.975, 2) * x.std(ddof=1) / np.sqrt(3)
    assert row["n"] == 3 and row["mean"] == pytest.approx(x.mean()) and row["ci_high"] - row["ci_low"] == pytest.approx(2 * half)
    wide = T.format_summary(long)
    assert "Yöntem" in wide.columns and "Normalize AUC" in wide.columns and "±" in wide["Normalize AUC"].iloc[0]
    out = T.write_summary_tables(runs, tmp_path, lang="en")
    assert out["csv"].is_file() and out["md"].is_file()
    md = out["md"].read_text(encoding="utf-8")
    assert "| Method" in md and "sin" in md and "Normalised AUC" in md and "Retention ratio" in md
    assert len(pd.read_csv(out["csv"])) == len(T.summary_table(runs))
    head, sep = T.df_to_markdown(pd.DataFrame({"a|b": [1.23456, np.nan], "s": ["x", "y"]})).splitlines()[:2]
    assert head == "|  a\\|b | s |"  # the header pipe is escaped; numeric columns are right-aligned to the widest cell (1.235)
    assert sep == "|------:|---|"  # right-alignment marker for the numeric column only
    md2 = T.df_to_markdown(pd.DataFrame({"v": [1.23456, np.nan], "s": ["x|y", "z"]}))
    assert "1.235" in md2 and "—" in md2 and "x\\|y" in md2 and md2.splitlines()[1].startswith("|-")
    assert T.df_to_markdown(pd.DataFrame(columns=["a", "b"])).count("\n") == 2


def test_comparison_table_stats_and_fallback(loaded, tmp_path, monkeypatch):
    runs, _ = loaded
    cmp = T.comparison_table(runs, metrics=["auc_norm", "final_window_mean", "missing"], reference="baseline", n_boot=300)
    for c in ("label", "reference", "metric", "n_pairs", "mean_label", "mean_ref", "mean_diff", "ci_low", "ci_high", "p_perm", "p_holm", "cohen_dz"):
        assert c in cmp.columns, c
    assert set(cmp["metric"]) == {"auc_norm", "final_window_mean"} and set(cmp["label"]) == {"sin"} and (cmp["method"] == "stats").all()
    r = cmp[cmp.metric == "auc_norm"].iloc[0]
    a = runs[runs.label == "sin"].sort_values("seed")["auc_norm"].to_numpy()
    b = runs[runs.label == "baseline"].sort_values("seed")["auc_norm"].to_numpy()
    assert r["n_pairs"] == 3 and r["mean_diff"] == pytest.approx((a - b).mean()) and r["mean_diff"] > 0
    assert 0 <= r["p_perm"] <= 1 and 0 <= r["p_holm"] <= 1 and r["p_holm"] >= r["p_perm"] - 1e-12
    # fallback when plasticity.metrics.stats cannot be imported
    monkeypatch.setitem(sys.modules, "plasticity.metrics.stats", None)
    fb = T.comparison_table(runs, metrics=["auc_norm"], reference="baseline")
    assert (fb["method"] == "fallback_t").all() and fb["mean_diff"].iloc[0] == pytest.approx(r["mean_diff"])
    d = a - b
    assert fb["p_perm"].iloc[0] == pytest.approx(sps.ttest_1samp(d, 0).pvalue) and fb["cohen_dz"].iloc[0] == pytest.approx(d.mean() / d.std(ddof=1))
    monkeypatch.delitem(sys.modules, "plasticity.metrics.stats")
    out = T.write_comparison_tables(runs, tmp_path, reference="baseline", metrics=["auc_norm"], n_boot=300)
    md = out["md"].read_text(encoding="utf-8")
    assert "baseline" in md and "sin" in md and "p (Holm)" in md and out["csv"].is_file()
    with pytest.raises(ValueError):
        T.comparison_table(runs, reference="ghost")
    assert len(T.comparison_table(runs, metrics=["auc_norm"], reference="baseline", labels=["baseline"])) == 0


# ---------------------------------------------------------------- CLI
def test_comparison_fallback_on_internal_error_and_duplicates(loaded, monkeypatch):
    runs, _ = loaded
    import importlib
    S = importlib.import_module("plasticity.metrics.stats")  # the module object the lazy import in tables resolves to

    def boom(*a, **k):
        raise TypeError("data problem inside compare_to_reference")

    monkeypatch.setattr(S, "compare_to_reference", boom)
    with pytest.warns(UserWarning, match="using the fallback"):
        fb = T.comparison_table(runs, metrics=["auc_norm"], reference="baseline")  # must not escape
    assert (fb["method"] == "fallback_t").all() and len(fb) == 1
    monkeypatch.undo()
    # duplicate (label, seed) rows: both paths average them per unit
    dup = pd.concat([runs, runs[runs.label == "sin"].assign(auc_norm=lambda d: d["auc_norm"] + 0.1)], ignore_index=True)
    st = T.comparison_table(dup, metrics=["auc_norm"], reference="baseline", n_boot=200)
    assert (st["method"] == "stats").all()
    fb2 = T._compare_fallback(dup[["label", "seed", "auc_norm"]], "auc_norm", "baseline")
    assert fb2["mean_label"].iloc[0] == pytest.approx(st["mean_label"].iloc[0]) and fb2["mean_diff"].iloc[0] == pytest.approx(st["mean_diff"].iloc[0])
    assert fb2["mean_label"].iloc[0] == pytest.approx(runs.loc[runs.label == "sin", "auc_norm"].mean() + 0.05)


def test_format_comparison_blanks_single_seed_statistics(loaded, tmp_path):
    runs, _ = loaded
    one = runs[runs.seed == 0]
    cmp = T.comparison_table(one, metrics=["auc_norm"], reference="baseline", n_boot=100)
    assert (cmp["n_pairs"] == 1).all()
    pretty = T.format_comparison(cmp, lang="en")
    row = pretty.iloc[0]
    assert row["Pairs"] == 1 and row["p (permutation)"] == "—" and row["p (Holm)"] == "—" and row["Cohen's d_z"] == "—"
    assert row["Diff (method − reference) [95% CI]"].endswith(" —") and "[" not in row["Diff (method − reference) [95% CI]"]
    out = T.write_comparison_tables(one, tmp_path, reference="baseline", metrics=["auc_norm"], n_boot=100)
    md = out["md"].read_text(encoding="utf-8")
    assert "en az 2 tohum" in md and "| — " in md
    full = T.format_comparison(T.comparison_table(runs, metrics=["auc_norm"], reference="baseline", n_boot=100), lang="en")
    assert "[" in full.iloc[0]["Diff (method − reference) [95% CI]"] and full.iloc[0]["p (Holm)"] != "—"
    md_full = T.write_comparison_tables(runs, tmp_path / "full", reference="baseline", metrics=["auc_norm"], n_boot=100)["md"].read_text(encoding="utf-8")
    assert "en az 2 tohum" not in md_full


def test_analyze_main(suite, tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze  # scripts/analyze.py

    out = tmp_path / "report"
    rc = analyze.main(["--results", str(suite), "--out", str(out), "--reference", "baseline", "--smooth", "3", "--n-boot", "300", "--format", "csv", "--dpi", "60",
                        "--mechanism-keys", "dead_frac/all,w_abs_mean/all,effective_rank/last,missing/all"])
    assert rc == 0
    for name in ("runs.csv", "curves.csv", "summary.md", "summary.csv", "comparison.md", "comparison.csv", "performance.png", "performance.pdf",
                 "mechanism.png", "mechanism.pdf", "summary.png", "summary.pdf", "fresh_gap.png", "fresh_gap.pdf"):
        assert (out / name).is_file(), name
    printed = capsys.readouterr().out
    assert "Normalize AUC" in printed and "sin" in printed and "baseline" in printed
    assert len(pd.read_csv(out / "runs.csv")) == 6 and len(pd.read_csv(out / "curves.csv")) == 60
    cmp = pd.read_csv(out / "comparison.csv")
    assert set(cmp["reference"]) == {"baseline"} and "auc_norm" in set(cmp["metric"])
    # a missing metric, an unknown reference, a label subset and English must not crash
    out2 = tmp_path / "report_en"
    rc = analyze.main(["--results", str(suite), "--out", str(out2), "--metric", "test_accuracy", "--reference", "ghost", "--lang", "en",
                       "--labels", "sin,baseline", "--sweep", "cfg.optimizer.lr", "--n-boot", "200", "--dpi", "60", "--mechanism-keys", "dead_frac/all", "--quiet"])
    assert rc == 0 and (out2 / "summary.md").is_file() and (out2 / "performance.png").is_file() and (out2 / "sweep.png").is_file()
    assert "Normalised AUC" in (out2 / "summary.md").read_text(encoding="utf-8")
    # nothing finished -> exit code 1, no crash
    empty = tmp_path / "empty_suite"
    _write_run(empty / "x" / "seed0", "x", 0, "sin", finished=False)
    assert analyze.main(["--results", str(empty), "--out", str(tmp_path / "r3"), "--quiet"]) == 1
    plt.close("all")


def test_analyze_main_sweep_suite_and_metric_fallback(sweep_suite, tmp_path, loaded):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze

    out = tmp_path / "sweep_report"
    rc = analyze.main(["--results", str(sweep_suite), "--out", str(out), "--labels", "sin,baseline,ghost", "--reference", "baseline/lr=0.001",
                       "--sweep", "cfg.optimizer.lr", "--n-boot", "100", "--dpi", "50", "--quiet", "--figures", "sweep,summary"])
    assert rc == 0 and (out / "sweep.png").is_file() and (out / "summary.png").is_file() and (out / "comparison.md").is_file()
    assert not (out / "performance.png").exists()  # --figures selects the products
    runs_csv = pd.read_csv(out / "runs.csv")
    assert len(runs_csv) == 12 and sorted(runs_csv["base_label"].unique()) == ["baseline", "sin"]
    cmp = pd.read_csv(out / "comparison.csv")
    assert set(cmp["reference"]) == {"baseline/lr=0.001"} and len(set(cmp["label"])) == 5 and (cmp["n_pairs"] == 2).all()
    # --labels with base labels only, an ambiguous base reference and an explicit ':label' grouping must not crash
    out2 = tmp_path / "sweep_report2"
    assert analyze.main(["--results", str(sweep_suite), "--out", str(out2), "--labels", "sin", "--reference", "baseline", "--sweep",
                         "cfg.optimizer.lr:label", "--n-boot", "100", "--dpi", "50", "--quiet", "--no-figures"]) == 0
    assert len(pd.read_csv(out2 / "runs.csv")) == 6
    assert analyze.main(["--results", str(sweep_suite), "--out", str(tmp_path / "none"), "--labels", "ghost", "--quiet"]) == 1
    # run-level metric for the summary / sweep figures: never the per-task metric name when a run-level column exists
    runs, _ = loaded
    assert analyze._run_level_metric(runs, "online_accuracy") == "auc_norm"
    assert analyze._run_level_metric(runs.drop(columns=["auc_norm"]), "online_accuracy") == "final_window_mean"
    assert analyze._run_level_metric(runs.drop(columns=["auc_norm", "final_window_mean", "last_task_perf"]), "online_accuracy") == "online_accuracy"
    plt.close("all")
