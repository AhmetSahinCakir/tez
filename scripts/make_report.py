#!/usr/bin/env python
"""Generate every table and figure referenced by docs/RAPOR.md from the finished suites.

    python scripts/make_report.py [--results results] [--out reports] [--lang tr] [--smooth 5]

Protocol assets (``pmnist`` = pmnist_main + pmnist_ablation, ``cifar`` = cifar_main + cifar_ablation):
  <P>_summary_main.md/.csv      mean ± 95 % CI of the summary metrics, main suite labels
  <P>_vs_baseline.md/.csv       paired comparisons against the standard network (H1 + published methods)
  <P>_vs_sin.md/.csv            paired comparisons against the sin model (H2 + component analysis)
  <P>_ablation_summary.md/.csv  summary metrics of the component / secondary runs
  <P>_adam_unit.md/.csv         optimiser (Adam) and literal-scale (sin_unit) sensitivity runs
  <P>_performance_*.png/pdf     mean curves ± CI (core / methods / secondary / ablation)
  <P>_mechanism_*.png/pdf       mechanism panels (core / ablation)
  <P>_fresh_gap.png/pdf, <P>_dots_<metric>.png/pdf, <P>_gamma.png/pdf, <P>_scale_corrected.png/pdf ...
stationary_summary.md/.csv      i.i.d. capacity control (best learning rate per model, mean ± CI)
INDEX.md                        list of the generated files

Every asset is produced independently (a missing suite / label only skips that asset), so the script can be
re-run while suites are still being computed.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plasticity.analysis import plots  # noqa: E402
from plasticity.analysis.aggregate import load_suite, has_fresh_data  # noqa: E402
from plasticity.analysis.tables import (  # noqa: E402
    comparison_table, df_to_markdown, format_comparison, format_summary, summary_table,
)

CORE = ["baseline", "sin", "tanh"]
PUBLISHED = ["weight_clipping", "l2_init", "continual_backprop", "nap", "ln_wd"]
SECONDARY = ["shrink_perturb", "upgd", "parseval", "smooth_leaky", "sin_act"]
SUMMARY_METRICS = ["auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "fresh_gap_final",
                   "final_dead_frac/all", "final_w_abs_mean/all", "final_effective_rank_c/last"]
PRIMARY_METRICS = ["auc_norm", "final_window_mean", "retention_ratio", "fresh_gap_final"]
MECH_KEYS_CORE = ["dead_frac/all", "w_abs_mean/all", "effective_rank_c/last", "w_over_A_mean/all", "sat_frac/all",
                  "jac_sq_mean/all", "grad_norm_w", "fisher_trace_w", "step_dw_norm"]
HIGHER_IS_BETTER = {"fresh_gap_final": False, "final_dead_frac/all": False, "final_w_abs_mean/all": False}

_written: List[str] = []


def _log(msg: str) -> None:
    print(msg, flush=True)


def _try(name: str, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
        return res
    except Exception as exc:  # noqa: BLE001 - keep going, report at the end
        _log(f"  [skip] {name}: {exc.__class__.__name__}: {exc}")
        if kwargs.pop("_debug", False):
            traceback.print_exc()
        return None


def _load(results: Path, name: str):
    root = results / name
    if not root.exists():
        return None, None
    runs, curves = load_suite(root, verbose=False)
    if runs is None or len(runs) == 0:
        return None, None
    runs = runs.copy()
    runs["suite"] = name
    curves = curves.copy()
    curves["suite"] = name
    return runs, curves


def _combine(results: Path, names: Sequence[str]):
    rs, cs = [], []
    for n in names:
        r, c = _load(results, n)
        if r is not None:
            rs.append(r)
            cs.append(c)
    if not rs:
        return None, None
    runs = pd.concat(rs, ignore_index=True, sort=False)
    curves = pd.concat(cs, ignore_index=True, sort=False)
    return runs, curves


def _present(runs: pd.DataFrame, labels: Sequence[str]) -> List[str]:
    have = set(runs["label"].unique())
    return [l for l in labels if l in have]


def _labels_starting(runs: pd.DataFrame, prefix: str) -> List[str]:
    return sorted(l for l in runs["label"].unique() if l == prefix or l.startswith(prefix + "/"))


def _write_tables(df_long: pd.DataFrame, out: Path, stem: str, lang: str, title: str, fmt_fn) -> None:
    df_long.to_csv(out / f"{stem}.csv", index=False)
    md = f"# {title}\n\n" + df_to_markdown(fmt_fn(df_long, lang=lang)) + "\n"
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    _written.extend([f"{stem}.csv", f"{stem}.md"])


def _save(fig, out: Path, stem: str) -> None:
    if fig is None:
        return
    plots.save_figure(fig, out / stem, close=True)
    _written.extend([f"{stem}.png", f"{stem}.pdf"])


def protocol_assets(P: str, results: Path, out: Path, lang: str, smooth: Optional[int]) -> None:
    runs, curves = _combine(results, [f"{P}_main", f"{P}_ablation"])
    if runs is None:
        _log(f"[{P}] no results yet")
        return
    main = runs[runs["suite"] == f"{P}_main"]
    metric = "online_accuracy" if P == "pmnist" else "test_accuracy"
    n_seeds = runs.groupby("label")["seed"].nunique()
    _log(f"[{P}] {len(runs)} runs, labels: " + ", ".join(f"{l}({n})" for l, n in n_seeds.items()))

    core = _present(main, CORE)
    published = _present(main, PUBLISHED)
    secondary = _present(main, SECONDARY)
    main_labels = core + published + secondary

    # ---- tables ------------------------------------------------------------------------------------
    if main_labels:
        _try("summary_main", lambda: _write_tables(summary_table(main, SUMMARY_METRICS, labels=main_labels), out,
                                                   f"{P}_summary_main", lang, f"{P}: özet ölçütler (ana paket)", format_summary))
    if "baseline" in main_labels and len(main_labels) > 1:
        def _vs_base():
            cmp = comparison_table(main, PRIMARY_METRICS + ["final_dead_frac/all", "final_w_abs_mean/all",
                                                            "final_effective_rank_c/last"],
                                   reference="baseline", labels=main_labels, higher_is_better=HIGHER_IS_BETTER)
            _write_tables(cmp, out, f"{P}_vs_baseline", lang, f"{P}: standart ağa karşı eşleştirilmiş karşılaştırma",
                          format_comparison)
        _try("vs_baseline", _vs_base)
    ablation_labels = [l for l in runs["label"].unique() if l.startswith(("sin/", "tanh/"))]
    if "sin" in main_labels:
        def _vs_sin():
            labs = ["sin"] + [l for l in ["tanh"] if l in main_labels] + sorted(ablation_labels) + \
                   [l for l in published if l in main_labels]
            cmp = comparison_table(runs, PRIMARY_METRICS + ["final_dead_frac/all", "final_sat_frac/all",
                                                            "final_effective_rank_c/last"],
                                   reference="sin", labels=labs, higher_is_better=HIGHER_IS_BETTER)
            _write_tables(cmp, out, f"{P}_vs_sin", lang, f"{P}: sinüs modeline karşı eşleştirilmiş karşılaştırma",
                          format_comparison)
        _try("vs_sin", _vs_sin)
    if ablation_labels:
        _try("ablation_summary", lambda: _write_tables(
            summary_table(runs, SUMMARY_METRICS + ["final_sat_frac/all", "final_w_over_A_mean/all", "final_jac_sq_mean/all"],
                          labels=["sin"] + sorted(ablation_labels)),
            out, f"{P}_ablation_summary", lang, f"{P}: bileşen ve ikincil deneyler", format_summary))
    extra = [l for l in runs["label"].unique() if l.startswith(("adam_", "sin_unit"))]
    if extra:
        _try("adam_unit", lambda: _write_tables(summary_table(runs, SUMMARY_METRICS, labels=["baseline", "sin", "tanh"] + sorted(extra)),
                                                out, f"{P}_adam_unit", lang, f"{P}: eniyileyici ve ölçek duyarlılığı", format_summary))

    # ---- figures -------------------------------------------------------------------------------------
    T = plots.label_text
    if core:
        _try("perf_core", lambda: _save(plots.plot_performance_curves(curves, metric, labels=core, smooth=smooth, lang=lang, runs=runs,
                                                                        title=T("performance_title", lang)), out, f"{P}_performance_core"))
        _try("mech_core", lambda: _save(plots.plot_mechanism_panels(curves, keys=MECH_KEYS_CORE, labels=core, smooth=smooth, lang=lang, runs=runs),
                                        out, f"{P}_mechanism_core"))
        if has_fresh_data(curves[curves["label"].isin(core)]):
            _try("fresh_gap", lambda: _save(plots.plot_fresh_gap(curves, metric=metric, labels=core, lang=lang, runs=runs), out, f"{P}_fresh_gap"))
    if published:
        labs = [l for l in ["baseline", "sin"] if l in main_labels] + published
        _try("perf_methods", lambda: _save(plots.plot_performance_curves(curves, metric, labels=labs, smooth=smooth, lang=lang, runs=runs),
                                           out, f"{P}_performance_methods"))
        _try("mech_methods", lambda: _save(plots.plot_mechanism_panels(curves, keys=["dead_frac/all", "w_abs_mean/all", "effective_rank_c/last",
                                                                                      "grad_norm_w", "fisher_trace_w", "step_dw_norm"],
                                                                        labels=labs, smooth=smooth, lang=lang, runs=runs), out, f"{P}_mechanism_methods"))
    if secondary:
        labs = [l for l in ["baseline", "sin"] if l in main_labels] + secondary
        _try("perf_secondary", lambda: _save(plots.plot_performance_curves(curves, metric, labels=labs, smooth=smooth, lang=lang, runs=runs),
                                             out, f"{P}_performance_secondary"))
    for m in ["auc_norm", "final_window_mean", "retention_ratio", "fresh_gap_final"]:
        if main_labels and m in main.columns and main[m].notna().any():
            _try(f"dots_{m}", lambda m=m: _save(plots.plot_summary_dots(main, m, labels=main_labels, reference="baseline" if "baseline" in main_labels else None,
                                                                        lang=lang, title=T(m, lang)), out, f"{P}_dots_{m.replace('/', '_')}"))
    comp = [l for l in ["sin", "sin/bias_std", "sin/hidden_only", "sin/output_only", "tanh", "tanh/bias_std", "sin/learn_amplitude", "sin/cbp"]
            if l in set(runs["label"])]
    if len(comp) > 1:
        _try("perf_ablation", lambda: _save(plots.plot_performance_curves(curves, metric, labels=comp, smooth=smooth, lang=lang, runs=runs),
                                            out, f"{P}_performance_ablation"))
        _try("mech_ablation", lambda: _save(plots.plot_mechanism_panels(curves, keys=MECH_KEYS_CORE, labels=comp, smooth=smooth, lang=lang, runs=runs),
                                            out, f"{P}_mechanism_ablation"))
    gamma_labels = ["sin"] + _labels_starting(runs, "sin/gamma")
    if len(gamma_labels) > 1 and "cfg.model.gamma" in runs.columns:
        sub = runs[runs["label"].isin(gamma_labels)].copy()
        sub["grp"] = "sin"
        _try("gamma", lambda: _save(plots.plot_sweep(sub, "cfg.model.gamma", "auc_norm", group_key="grp", lang=lang,
                                                      title="Genlik duyarlılığı (γ)" if lang == "tr" else "Amplitude sensitivity (γ)",
                                                      xlabel="γ"), out, f"{P}_gamma"))
        _try("gamma_curves", lambda: _save(plots.plot_performance_curves(curves, metric, labels=gamma_labels, smooth=smooth, lang=lang, runs=runs),
                                           out, f"{P}_gamma_curves"))
    sc_labels = ["sin"] + _labels_starting(runs, "sin/scale_corrected")
    if len(sc_labels) > 1:
        _try("scale_corrected", lambda: _save(plots.plot_performance_curves(curves, metric, labels=sc_labels, smooth=smooth, lang=lang, runs=runs),
                                              out, f"{P}_scale_corrected"))
        _try("scale_corrected_mech", lambda: _save(plots.plot_mechanism_panels(curves, keys=["sat_frac/all", "jac_sq_mean/all", "w_over_A_mean/all",
                                                                                              "dead_frac/all", "effective_rank_c/last", "step_dw_norm"],
                                                                                labels=sc_labels, smooth=smooth, lang=lang, runs=runs), out, f"{P}_scale_corrected_mechanism"))
    unit_labels = ["sin"] + _labels_starting(runs, "sin_unit")
    if len(unit_labels) > 1:
        _try("sin_unit", lambda: _save(plots.plot_performance_curves(curves, metric, labels=unit_labels, smooth=smooth, lang=lang, runs=runs),
                                       out, f"{P}_sin_unit"))
    adam_labels = [l for l in runs["label"].unique() if l.startswith("adam_")]
    if adam_labels:
        _try("adam", lambda: _save(plots.plot_performance_curves(curves, metric, labels=sorted(adam_labels), smooth=smooth, lang=lang, runs=runs),
                                   out, f"{P}_adam"))


def stationary_assets(results: Path, out: Path, lang: str) -> None:
    runs, _ = _load(results, "stationary")
    if runs is None:
        _log("[stationary] no results yet")
        return
    runs = runs.copy()
    runs["model"] = runs["base_label"]  # e.g. mnist/standard -> base_label 'mnist'? keep full stem instead
    runs["model"] = runs["label"].str.replace(r"/lr=.*$", "", regex=True)
    best_rows = []
    for model, g in runs.groupby("model"):
        per_lr = g.groupby("cfg.optimizer.lr")["final_test_accuracy"].mean()
        lr_best = per_lr.idxmax()
        gb = g[g["cfg.optimizer.lr"] == lr_best]
        best_rows.append(gb.assign(best_lr=lr_best))
    best = pd.concat(best_rows)
    long = summary_table(best.assign(label=best["model"]), ["final_test_accuracy", "best_test_accuracy", "final_train_accuracy",
                                                            "final_dead_frac/all", "final_effective_rank_c/last"])
    lrs = best.groupby("model")["best_lr"].first()
    wide = format_summary(long, lang=lang)
    wide.insert(1, "lr", [lrs.get(l, np.nan) for l in wide.iloc[:, 0]])
    (out / "stationary_summary.md").write_text("# Durağan kontrol: en iyi öğrenme oranında test doğruluğu\n\n" + df_to_markdown(wide) + "\n", encoding="utf-8")
    long.to_csv(out / "stationary_summary.csv", index=False)
    _written.extend(["stationary_summary.md", "stationary_summary.csv"])
    all_long = summary_table(runs.assign(label=runs["label"]), ["final_test_accuracy"])
    (out / "stationary_all.md").write_text("# Durağan kontrol: bütün öğrenme oranları\n\n" + df_to_markdown(format_summary(all_long, lang=lang)) + "\n", encoding="utf-8")
    _written.append("stationary_all.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args(argv)
    results = (PROJECT_ROOT / args.results) if not Path(args.results).is_absolute() else Path(args.results)
    out = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for P in ("pmnist", "cifar"):
        protocol_assets(P, results, out, args.lang, args.smooth)
    stationary_assets(results, out, args.lang)
    (out / "INDEX.md").write_text("# Üretilen dosyalar\n\n" + "\n".join(f"- `{f}`" for f in sorted(set(_written))) + "\n", encoding="utf-8")
    _log(f"wrote {len(set(_written))} files to {out}")


if __name__ == "__main__":
    main()
