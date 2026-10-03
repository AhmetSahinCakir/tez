"""Markdown / CSV tables: per-label summary (mean ± 95 % CI over seeds) and paired comparison vs a reference.

* :func:`summary_table` -- long frame ``label, metric, n, mean, std, ci_low, ci_high`` (Student-t CI, the
  seed being the statistical unit); :func:`format_summary` turns it into the wide "mean ± half-width" text
  table used in the thesis.
* :func:`comparison_table` -- every label vs ``reference`` on one or more metrics, paired by seed, through
  :func:`plasticity.metrics.stats.compare_to_reference` (percentile-bootstrap CI, sign-flip permutation p,
  Holm correction, Cohen's d_z).  The import is lazy; if that module is unavailable *or raises*, a minimal
  internal fallback is used (mean difference, t CI, paired t-test p as ``p_perm``, Holm) and the ``method``
  column says so.  Both paths average duplicate ``(label, seed)`` rows.  :func:`format_comparison` blanks
  the CI, p-values and d_z of comparisons with fewer than two pairs (a paired test needs >= 2 seeds).
* :func:`df_to_markdown` -- dependency-free GitHub-flavoured markdown writer (no ``tabulate`` needed).
"""
from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .aggregate import t_half_width

__all__ = ["DEFAULT_SUMMARY_METRICS", "DEFAULT_COMPARISON_METRICS", "summary_table", "format_summary", "comparison_table",
           "format_comparison", "df_to_markdown", "write_summary_tables", "write_comparison_tables"]

DEFAULT_SUMMARY_METRICS: Sequence[str] = (
    "auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "drop", "slope_per_100", "fresh_gap_final",
    "final_dead_frac/all", "final_inactive_frac/all", "final_w_abs_mean/all", "final_w_fro_norm/all",
    "final_effective_rank/last", "final_stable_rank/last", "final_grad_norm_w", "final_fisher_trace_w",
    "final_w_over_A_mean/all", "final_sat_frac/all", "final_test_accuracy", "wall_time",
)
DEFAULT_COMPARISON_METRICS: Sequence[str] = ("auc_norm", "final_window_mean", "retention_ratio", "fresh_gap_final",
                                             "final_dead_frac/all", "final_effective_rank/last", "final_w_abs_mean/all")


def _present(df: pd.DataFrame, metrics: Optional[Iterable[str]], default: Sequence[str]) -> List[str]:
    cand = list(metrics) if metrics is not None else list(default)
    out = []
    for m in cand:
        if m in df.columns and pd.to_numeric(df[m], errors="coerce").notna().any():
            out.append(m)
    return out


def _label_order(df: pd.DataFrame, labels: Optional[Iterable[str]]) -> List[str]:
    present = [str(l) for l in pd.unique(df["label"])]
    if labels is None:
        try:
            from .plots import order_labels
            return order_labels(present, runs=df)
        except Exception:  # pragma: no cover
            return sorted(present)
    return [str(l) for l in labels if str(l) in present]


# ------------------------------------------------------------------------------------------------
# summary
# ------------------------------------------------------------------------------------------------
def summary_table(runs: pd.DataFrame, metrics: Optional[Iterable[str]] = None, labels: Optional[Iterable[str]] = None,
                  alpha: float = 0.05) -> pd.DataFrame:
    """Long table: one row per (label, metric) with ``n, mean, std, ci_low, ci_high`` (t-based CI over seeds).

    Metrics missing from ``runs`` are silently dropped; a metric with a single seed has NaN CI bounds.
    """
    cols = ["label", "metric", "n", "mean", "std", "ci_low", "ci_high"]
    if len(runs) == 0 or "label" not in runs.columns:
        return pd.DataFrame(columns=cols)
    metrics = _present(runs, metrics, DEFAULT_SUMMARY_METRICS)
    order = _label_order(runs, labels)
    rows: List[Dict[str, Any]] = []
    for lab in order:
        sub = runs[runs["label"] == lab]
        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy(dtype=float)
            n = int(x.size)
            mean = float(x.mean()) if n else float("nan")
            sd = float(x.std(ddof=1)) if n > 1 else float("nan")
            half = float(t_half_width(np.array([n]), np.array([sd]), alpha)[0]) if n else float("nan")
            rows.append({"label": lab, "metric": m, "n": n, "mean": mean, "std": sd, "ci_low": mean - half, "ci_high": mean + half})
    return pd.DataFrame(rows, columns=cols)


def _fmt(v: float, digits: int) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{digits}f}"


def format_summary(long: pd.DataFrame, digits: int = 3, lang: str = "tr", translate: bool = True) -> pd.DataFrame:
    """Wide text table: rows = labels, columns = ``n`` then one ``mean ± half`` column per metric."""
    from .plots import label_text
    T = (lambda k: label_text(k, lang)) if translate else (lambda k: k)
    if len(long) == 0:
        return pd.DataFrame()
    labels = list(dict.fromkeys(long["label"]))
    metrics = list(dict.fromkeys(long["metric"]))
    rows = []
    for lab in labels:
        sub = long[long["label"] == lab].set_index("metric")
        row: Dict[str, Any] = {T("label"): lab, T("n_seeds"): int(sub["n"].max()) if len(sub) else 0}
        for m in metrics:
            if m in sub.index:
                r = sub.loc[m]
                half = (r["ci_high"] - r["ci_low"]) / 2 if np.isfinite(r["ci_high"]) else float("nan")
                cell = _fmt(r["mean"], digits) + (f" ± {_fmt(half, digits)}" if np.isfinite(half) else "")
            else:
                cell = "—"
            row[T(m)] = cell
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# paired comparison vs reference
# ------------------------------------------------------------------------------------------------
_CMP_COLS = ["label", "reference", "metric", "n_pairs", "mean_label", "mean_ref", "mean_diff", "ci_low", "ci_high",
             "p_perm", "p_holm", "cohen_dz", "method"]


def _holm(p: np.ndarray) -> np.ndarray:
    adj = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.size
    if m:
        pv = p[valid]
        order = np.argsort(pv, kind="stable")
        stepped = np.maximum.accumulate((m - np.arange(m)) * pv[order])
        out = np.empty(m)
        out[order] = np.minimum(stepped, 1.0)
        adj[valid] = out
    return adj


def _compare_fallback(df: pd.DataFrame, metric: str, reference_label: str, label_col: str = "label", unit_col: str = "seed",
                      alpha: float = 0.05, labels: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Minimal stand-in for ``plasticity.metrics.stats.compare_to_reference``: paired by unit, mean diff,
    Student-t CI, paired t-test p (reported as ``p_perm``), Holm correction, Cohen's d_z.  Several rows of
    one unit within a label are averaged (the same as ``compare_to_reference(..., duplicates='mean')``)."""
    vals = pd.to_numeric(df[metric], errors="coerce")

    def per_unit(lab: str) -> pd.Series:
        m = df[label_col] == lab
        s = pd.Series(vals[m].to_numpy(dtype=float), index=pd.Index(df.loc[m, unit_col].to_numpy()))
        return s.groupby(level=0, sort=False).mean().dropna()  # NaN-skipping mean of duplicate units

    ref = per_unit(reference_label)
    order = [l for l in (list(labels) if labels is not None else list(pd.unique(df[label_col]))) if l != reference_label]
    rows = []
    for lab in order:
        s = per_unit(lab)
        j = pd.concat([s.rename("x"), ref.rename("r")], axis=1, join="inner")
        n = int(len(j))
        row: Dict[str, Any] = {"label": lab, "reference": reference_label, "metric": metric, "n_pairs": n}
        if n == 0:
            row.update({k: float("nan") for k in ("mean_label", "mean_ref", "mean_diff", "ci_low", "ci_high", "p_perm", "cohen_dz")})
        else:
            d = (j["x"] - j["r"]).to_numpy(dtype=float)
            sd = float(d.std(ddof=1)) if n > 1 else float("nan")
            half = float(t_half_width(np.array([n]), np.array([sd]), alpha)[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(sps.ttest_1samp(d, 0.0).pvalue) if n > 1 and sd > 0 else float("nan")
            row.update({"mean_label": float(j["x"].mean()), "mean_ref": float(j["r"].mean()), "mean_diff": float(d.mean()),
                        "ci_low": float(d.mean() - half), "ci_high": float(d.mean() + half), "p_perm": p,
                        "cohen_dz": float(d.mean() / sd) if n > 1 and sd > 0 else float("nan")})
        rows.append(row)
    out = pd.DataFrame(rows, columns=[c for c in _CMP_COLS if c not in ("p_holm", "method")])
    out["p_holm"] = _holm(out["p_perm"].to_numpy(dtype=float)) if len(out) else np.array([])
    out["method"] = "fallback_t"
    return out[_CMP_COLS]


def _load_compare_to_reference(**optional_kw):
    """Lazily import ``plasticity.metrics.stats.compare_to_reference`` (None when unavailable) and keep, of the
    optional keywords given, those its signature accepts (``alpha``, ``higher_is_better``, ``labels``,
    ``duplicates`` are not part of the minimal documented signature)."""
    try:
        from ..metrics.stats import compare_to_reference  # lazy: written by another component
    except Exception:  # pragma: no cover - exercised through tests via monkeypatching
        return None, {}
    try:
        params = inspect.signature(compare_to_reference).parameters
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        kw = {k: v for k, v in optional_kw.items() if accepts_any or k in params}
        if "higher_is_better" in params or accepts_any:
            kw["higher_is_better"] = True  # placeholder, set per metric by the caller
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables without a signature
        kw = {}
    return compare_to_reference, kw


def comparison_table(runs: pd.DataFrame, metrics: Optional[Iterable[str]] = None, reference: Optional[str] = None,
                     labels: Optional[Iterable[str]] = None, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05,
                     unit_col: str = "seed", higher_is_better: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
    """Paired comparison of every label with ``reference`` (default: first label in display order) for each metric.

    Columns: ``label, reference, metric, n_pairs, mean_label, mean_ref, mean_diff, ci_low, ci_high, p_perm, p_holm,
    cohen_dz, method`` (``method`` is ``'stats'`` when :mod:`plasticity.metrics.stats` was used, ``'fallback_t'`` otherwise).
    Holm correction is applied within each metric's family of comparisons (as ``compare_to_reference`` does).
    """
    if len(runs) == 0 or "label" not in runs.columns:
        return pd.DataFrame(columns=_CMP_COLS)
    order = _label_order(runs, labels)
    if reference is None:
        reference = order[0] if order else None
    if reference is None or reference not in order and reference not in set(runs["label"].astype(str)):
        raise ValueError(f"reference label {reference!r} not among {order}")
    metrics = _present(runs, metrics, DEFAULT_COMPARISON_METRICS)
    others = [l for l in order if l != reference]
    frames: List[pd.DataFrame] = []
    compare_to_reference, extra_kw = _load_compare_to_reference(alpha=alpha, labels=order, duplicates="mean")
    for m in metrics:
        sub = runs[["label", unit_col, m]].copy()
        sub[m] = pd.to_numeric(sub[m], errors="coerce")
        hib = True if higher_is_better is None else bool(higher_is_better.get(m, True))
        res = None
        if compare_to_reference is not None:
            kw = dict(extra_kw)
            if "higher_is_better" in kw:
                kw["higher_is_better"] = hib
            try:
                res = compare_to_reference(sub, m, reference, label_col="label", unit_col=unit_col, n_boot=n_boot, seed=seed, **kw).copy()
                res["method"] = "stats"
            except Exception as e:  # noqa: BLE001 - any failure (data or signature) routes to the fallback
                warnings.warn(f"compare_to_reference failed for {m!r} ({type(e).__name__}: {e}); using the fallback")
                res = None
        if res is None:
            res = _compare_fallback(sub, m, reference, unit_col=unit_col, alpha=alpha, labels=order)
        for c in _CMP_COLS:
            if c not in res.columns:
                res[c] = float("nan") if c not in ("label", "reference", "metric", "method") else (m if c == "metric" else reference)
        res = res[res["label"].isin(others)] if others else res.iloc[0:0]
        frames.append(res[_CMP_COLS + [c for c in ("p_wilcoxon", "t_p", "better", "worse") if c in res.columns]])
    if not frames:
        return pd.DataFrame(columns=_CMP_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


MIN_PAIRS = 2  # a paired test / CI needs at least this many (label, reference) pairs = seeds


def format_comparison(cmp: pd.DataFrame, digits: int = 3, lang: str = "tr", translate: bool = True) -> pd.DataFrame:
    """Readable text table: one row per (metric, label) with ``Δ [CI]``, p-values with stars and d_z.

    Comparisons with fewer than :data:`MIN_PAIRS` pairs (single-seed suites) show the mean difference only;
    their CI, p-values and d_z are blanked (``—``) because a paired test needs at least two seeds."""
    from .plots import label_text
    T = (lambda k: label_text(k, lang)) if translate else (lambda k: k)
    if len(cmp) == 0:
        return pd.DataFrame()
    rows = []
    for _, r in cmp.iterrows():
        n = int(r["n_pairs"]) if np.isfinite(r["n_pairs"]) else 0
        degenerate = n < MIN_PAIRS
        ci = "—" if degenerate else f"[{_fmt(r['ci_low'], digits)}, {_fmt(r['ci_high'], digits)}]"
        p_holm = "—" if degenerate else _fmt(r["p_holm"], 4) + (" " + _stars(r["p_holm"]) if _stars(r["p_holm"]) else "")
        rows.append({
            T("metric"): T(r["metric"]), T("label"): r["label"], T("reference"): r["reference"], T("n_pairs"): n,
            T("mean_label"): _fmt(r["mean_label"], digits), T("mean_ref"): _fmt(r["mean_ref"], digits),
            T("mean_diff") + " [" + T("ci95") + "]": f"{_fmt(r['mean_diff'], digits)} {ci}",
            T("p_perm"): "—" if degenerate else _fmt(r["p_perm"], 4), T("p_holm"): p_holm,
            T("cohen_dz"): "—" if degenerate else _fmt(r["cohen_dz"], 2),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# markdown + writers
# ------------------------------------------------------------------------------------------------
def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False, na_rep: str = "—") -> str:
    """GitHub-flavoured markdown table (pipes escaped), numeric cells formatted with ``floatfmt``."""
    if index:
        df = df.reset_index()

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return na_rep
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt)
        if isinstance(v, (bool, np.bool_)):
            return "✓" if v else ""
        return str(v).replace("|", "\\|").replace("\n", " ")

    cols = [cell(str(c)) for c in df.columns]  # header cells are escaped like body cells
    body = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(c), *(len(r[i]) for r in body)) if body else len(c) for i, c in enumerate(cols)]
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.rjust(w) if num else c.ljust(w) for c, w, num in zip(cells, widths, numeric)) + " |"

    sep = "|" + "|".join(("-" * (w + 1) + ":") if num else ("-" * (w + 2)) for w, num in zip(widths, numeric)) + "|"
    out = [line(cols), sep] + [line(r) for r in body]
    return "\n".join(out) + "\n"


def write_summary_tables(runs: pd.DataFrame, out_dir: str | os.PathLike, metrics: Optional[Iterable[str]] = None,
                         labels: Optional[Iterable[str]] = None, lang: str = "tr", digits: int = 3, stem: str = "summary",
                         title: Optional[str] = None) -> Dict[str, Path]:
    """Write ``<stem>.csv`` (long, numeric) and ``<stem>.md`` (wide, mean ± 95 % CI).  Returns the paths."""
    from .plots import label_text
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long = summary_table(runs, metrics=metrics, labels=labels)
    csv_path = out / f"{stem}.csv"
    long.to_csv(csv_path, index=False)
    wide = format_summary(long, digits=digits, lang=lang)
    md_path = out / f"{stem}.md"
    head = title if title is not None else label_text("summary_title", lang)
    note = label_text("mean_ci_note", lang)
    text = f"## {head}\n\n_{note}_\n\n" + (df_to_markdown(wide) if len(wide) else "_(no finished runs)_\n")
    md_path.write_text(text, encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "long": long, "wide": wide}  # type: ignore[dict-item]


def write_comparison_tables(runs: pd.DataFrame, out_dir: str | os.PathLike, reference: Optional[str] = None,
                            metrics: Optional[Iterable[str]] = None, labels: Optional[Iterable[str]] = None, lang: str = "tr",
                            digits: int = 3, n_boot: int = 10000, seed: int = 0, stem: str = "comparison",
                            title: Optional[str] = None) -> Dict[str, Path]:
    """Write ``<stem>.csv`` (numeric) and ``<stem>.md`` (formatted) paired comparison tables.  Returns the paths."""
    from .plots import label_text
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmp = comparison_table(runs, metrics=metrics, reference=reference, labels=labels, n_boot=n_boot, seed=seed)
    csv_path = out / f"{stem}.csv"
    cmp.to_csv(csv_path, index=False)
    pretty = format_comparison(cmp, digits=digits, lang=lang)
    md_path = out / f"{stem}.md"
    ref = cmp["reference"].iloc[0] if len(cmp) else reference
    head = title if title is not None else f"{label_text('reference', lang)}: {ref}"
    method = cmp["method"].iloc[0] if len(cmp) and "method" in cmp.columns else "—"
    note = ("eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi"
            if lang == "tr" else "paired difference (method − reference), 95 % bootstrap CI, sign-flip permutation test, Holm correction")
    if method == "fallback_t":
        note = ("eşleştirilmiş fark, t-GA ve eşleştirilmiş t-testi (yedek uygulama)" if lang == "tr"
                else "paired difference, t CI and paired t-test (fallback implementation)")
    text = f"## {head}\n\n_{note}; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_\n\n" + (df_to_markdown(pretty) if len(pretty) else "_(nothing to compare)_\n")
    if len(cmp) and (pd.to_numeric(cmp["n_pairs"], errors="coerce").fillna(0) < MIN_PAIRS).any():
        foot = (f"_Eşleştirilmiş testler en az {MIN_PAIRS} tohum gerektirir; tek tohumlu karşılaştırmalarda GA, p değerleri ve d_z "
                f"verilmemiştir (—)._" if lang == "tr" else
                f"_Paired tests need at least {MIN_PAIRS} seeds; CI, p-values and d_z are left blank (—) for single-seed comparisons._")
        text += "\n" + foot + "\n"
    md_path.write_text(text, encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "table": cmp, "pretty": pretty}  # type: ignore[dict-item]
