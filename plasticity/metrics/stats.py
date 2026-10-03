"""Statistics for the paired experimental design of the thesis ("Tekrar Sayısı ve İstatistiksel Analiz").

Design
------
The statistical unit is the *independent repetition*: one seed, i.e. one task stream plus one
initialisation (Dohare et al. 2024, "Loss of plasticity in deep continual learning", Nature 632,
report means over independent runs the same way).  Every method is run on the *same* seeds, so a
comparison of two methods is a **paired** comparison: for seed ``i`` we observe ``a_i`` (method A)
and ``b_i`` (method B) and analyse the differences ``d_i = a_i - b_i``.  The number of seeds is small
(typically 5-30), so the module relies on distribution-free procedures:

* **Percentile bootstrap CI of the mean paired difference** (Efron & Tibshirani 1993, ch. 13):
  ``n_boot`` resamples of ``d`` with replacement, CI = ``[alpha/2, 1-alpha/2]`` percentiles of the
  resampled means.  No bias/acceleration correction (plain percentile interval, as pre-registered).
* **Two-sided paired permutation (sign-flip) test** on the mean difference (Fisher 1935; Good 2005):
  under H0 the sign of every ``d_i`` is exchangeable, so the null distribution of ``|mean(d)|`` is
  obtained by flipping signs.  When the number of *non-zero* differences ``m`` satisfies ``m <= exact_max_n``
  (default 14) all ``2^m`` sign vectors are enumerated and the p-value is exact
  (``p = #{|T_s| >= |T_obs|} / 2^m``; zero differences are invariant under flipping, so this equals the
  enumeration over all ``2^n`` flips).  Otherwise ``n_perm`` random sign vectors are drawn and the
  p-value uses the +1 correction of Phipson & Smyth (2010), ``p = (1 + #{|T_s| >= |T_obs|}) / (n_perm + 1)``,
  which never returns 0.
* **Wilcoxon signed-rank test** (Wilcoxon 1945) via :func:`scipy.stats.wilcoxon` (``method='auto'``:
  exact for small samples, normal approximation otherwise); NaN when all differences are zero.
* **Cohen's d_z** for paired data (Cohen 1988): ``mean(d) / sd(d)`` with ``ddof=1``.
* **Paired t-test** p-value (reference only; normality is not assumed for the conclusions).
* **Holm-Bonferroni step-down** (Holm 1979) over the family of comparisons made in one call, to control
  the family-wise error rate when several methods are compared against the same reference.

Everything is numpy/scipy/pandas only and deterministic given the ``seed`` argument
(``numpy.random.default_rng``).  Users import ``plasticity.metrics.stats`` directly.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sps

__all__ = [
    "paired_difference",
    "sign_flip_test",
    "mean_ci",
    "t_ci",
    "holm_correction",
    "compare_to_reference",
    "summary_table",
    "pairwise_comparisons",
    "pairwise_matrix",
]

DEFAULT_N_BOOT = 10000
DEFAULT_N_PERM = 20000
DEFAULT_EXACT_MAX_N = 14
DEFAULT_ALPHA = 0.05

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


# ------------------------------------------------------------------------------------------------
# low-level helpers
# ------------------------------------------------------------------------------------------------
def _as_1d(x: ArrayLike, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr


def _bootstrap_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``n_boot`` means of resamples (with replacement) of ``x`` -- vectorised."""
    n = x.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    return x[idx].mean(axis=1)


def mean_ci(x: ArrayLike, n_boot: int = DEFAULT_N_BOOT, seed: int = 0, alpha: float = DEFAULT_ALPHA) -> Tuple[float, float, float]:
    """Percentile-bootstrap confidence interval of the mean.  Returns ``(mean, low, high)``.

    NaN entries are dropped.  With a single observation the interval collapses to the point; with no
    observation everything is NaN.
    """
    x = _as_1d(x)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n == 1:
        return m, m, m
    rng = np.random.default_rng(seed)
    means = _bootstrap_means(x, n_boot, rng)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return m, float(low), float(high)


def t_ci(x: ArrayLike, alpha: float = DEFAULT_ALPHA) -> Tuple[float, float, float]:
    """Normal-theory (Student t) confidence interval of the mean: ``mean -/+ t_{1-alpha/2, n-1} * s / sqrt(n)``."""
    x = _as_1d(x)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n == 1:
        return m, float("nan"), float("nan")
    half = float(sps.t.ppf(1 - alpha / 2, df=n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - half, m + half


def _sign_matrix(m: int) -> np.ndarray:
    """All ``2^m`` sign vectors in {-1, +1}^m as a ``(2^m, m)`` int8 matrix (row k = binary digits of k)."""
    k = np.arange(2 ** m, dtype=np.int64)[:, None]
    bits = (k >> np.arange(m, dtype=np.int64)[None, :]) & 1
    return (2 * bits - 1).astype(np.int8)


def sign_flip_test(d: ArrayLike, n_perm: int = DEFAULT_N_PERM, seed: int = 0, exact_max_n: int = DEFAULT_EXACT_MAX_N,
                   force_monte_carlo: bool = False) -> Dict[str, Any]:
    """Two-sided paired permutation (sign-flip) test of H0: mean(d) = 0.

    Statistic ``T = |mean(d)|``.  Exact enumeration over the ``m`` non-zero differences when
    ``m <= exact_max_n`` (identical to enumerating all ``2^n`` flips), otherwise Monte-Carlo with
    ``n_perm`` draws and the +1 correction (Phipson & Smyth 2010).

    Returns ``{'p': float, 'method': 'exact'|'monte_carlo', 'n_perm': int, 'stat': float}``.
    """
    d = _as_1d(d)
    d = d[~np.isnan(d)]
    n = d.size
    if n == 0:
        return {"p": float("nan"), "method": "none", "n_perm": 0, "stat": float("nan")}
    t_obs = abs(float(d.mean()))
    tol = 1e-12 * max(1.0, float(np.abs(d).max()))  # guard against summation-order round-off
    nz = d[d != 0]
    m = nz.size
    if m == 0:  # all differences zero: every sign flip reproduces the statistic
        return {"p": 1.0, "method": "exact", "n_perm": 1, "stat": 0.0}
    if m <= exact_max_n and not force_monte_carlo:
        signs = _sign_matrix(m)
        stat = np.abs(signs @ nz) / n
        k = int(np.count_nonzero(stat >= t_obs - tol))
        return {"p": k / signs.shape[0], "method": "exact", "n_perm": int(signs.shape[0]), "stat": t_obs}
    rng = np.random.default_rng(seed)
    n_perm = int(n_perm)
    # draw in chunks to bound memory for large n
    k = 0
    chunk = max(1, min(n_perm, 4_000_000 // max(m, 1)))
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(b, m), dtype=np.int8) * 2 - 1
        stat = np.abs(signs @ nz) / n
        k += int(np.count_nonzero(stat >= t_obs - tol))
        done += b
    return {"p": (1 + k) / (n_perm + 1), "method": "monte_carlo", "n_perm": n_perm, "stat": t_obs}


def _wilcoxon_p(d: np.ndarray) -> float:
    if d.size < 1 or not np.any(d != 0):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("auto", "approx"):
            try:
                return float(sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method=method).pvalue)
            except (ValueError, TypeError):
                continue
    return float("nan")


def _ttest_p(d: np.ndarray) -> float:
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sps.ttest_1samp(d, 0.0).pvalue)


def _cohen_dz(d: np.ndarray) -> float:
    if d.size < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    m = float(d.mean())
    if sd > 0:
        return m / sd
    return float("nan") if m == 0 else float(np.sign(m) * np.inf)


# ------------------------------------------------------------------------------------------------
# paired comparison of two methods on the same seeds
# ------------------------------------------------------------------------------------------------
def paired_difference(a: ArrayLike, b: ArrayLike, n_boot: int = DEFAULT_N_BOOT, seed: int = 0, alpha: float = DEFAULT_ALPHA,
                      n_perm: int = DEFAULT_N_PERM, exact_max_n: int = DEFAULT_EXACT_MAX_N,
                      force_monte_carlo: bool = False) -> Dict[str, Any]:
    """Paired comparison of ``a`` and ``b`` (same length, ``a[i]`` and ``b[i]`` from the same seed).

    Returns a dict with ``n, mean_a, mean_b, mean_diff (a-b), std_diff, ci_low, ci_high`` (percentile
    bootstrap of the mean difference), ``p_perm`` (sign-flip test), ``p_wilcoxon``, ``cohen_dz``, ``t_p``
    and bookkeeping (``perm_method, n_perm, n_boot, alpha``).  Pairs with a NaN in either sample are dropped.
    """
    a = _as_1d(a, "a")
    b = _as_1d(b, "b")
    if a.size != b.size:
        raise ValueError(f"paired samples must have the same length, got {a.size} and {b.size}")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = int(a.size)
    d = a - b
    out: Dict[str, Any] = {"n": n, "n_boot": int(n_boot), "alpha": float(alpha)}
    if n == 0:
        out.update({k: float("nan") for k in ("mean_a", "mean_b", "mean_diff", "std_diff", "ci_low", "ci_high",
                                              "p_perm", "p_wilcoxon", "cohen_dz", "t_p")})
        out.update({"perm_method": "none", "n_perm": 0})
        return out
    mean_diff, ci_low, ci_high = mean_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)
    perm = sign_flip_test(d, n_perm=n_perm, seed=seed, exact_max_n=exact_max_n, force_monte_carlo=force_monte_carlo)
    out.update({
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "std_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_perm": perm["p"],
        "perm_method": perm["method"],
        "n_perm": perm["n_perm"],
        "p_wilcoxon": _wilcoxon_p(d),
        "cohen_dz": _cohen_dz(d),
        "t_p": _ttest_p(d),
    })
    return out


# ------------------------------------------------------------------------------------------------
# multiple comparisons
# ------------------------------------------------------------------------------------------------
def holm_correction(pvals: ArrayLike):
    """Holm-Bonferroni step-down adjusted p-values (Holm 1979).

    ``p_(k)`` sorted ascending over the ``m`` non-NaN entries: ``adj_(k) = max_{j<=k} (m - j + 1) * p_(j)``,
    clipped at 1 (monotone).  The output keeps the input order; NaN entries stay NaN and do not count
    towards ``m``.  A :class:`pandas.Series` input returns a Series with the same index.
    """
    is_series = isinstance(pvals, pd.Series)
    index = pvals.index if is_series else None
    p = np.asarray(pvals, dtype=float).ravel()
    adj = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.size
    if m > 0:
        pv = p[valid]
        if np.any((pv < 0) | (pv > 1)):
            raise ValueError("p-values must lie in [0, 1]")
        order = np.argsort(pv, kind="stable")
        factors = m - np.arange(m)  # m, m-1, ..., 1
        stepped = np.maximum.accumulate(factors * pv[order])
        adj_sorted = np.minimum(stepped, 1.0)
        out = np.empty(m)
        out[order] = adj_sorted
        adj[valid] = out
    if is_series:
        return pd.Series(adj, index=index, name=getattr(pvals, "name", None))
    return adj


# ------------------------------------------------------------------------------------------------
# DataFrame-level helpers (one row per run: label, seed, metrics...)
# ------------------------------------------------------------------------------------------------
def _per_unit(df: pd.DataFrame, metric: str, label_col: str, unit_col: Optional[str], duplicates: str) -> Dict[Any, pd.Series]:
    """``{label: Series(metric values indexed by unit)}``, handling duplicate units per label."""
    if metric not in df.columns:
        raise KeyError(f"metric column {metric!r} not in DataFrame (columns: {list(df.columns)})")
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not in DataFrame")
    out: Dict[Any, pd.Series] = {}
    for label in pd.unique(df[label_col]):
        sub = df.loc[df[label_col] == label]
        if unit_col is None:
            s = pd.Series(sub[metric].to_numpy(dtype=float), index=pd.RangeIndex(len(sub)))
        else:
            if unit_col not in df.columns:
                raise KeyError(f"unit column {unit_col!r} not in DataFrame")
            s = pd.Series(sub[metric].to_numpy(dtype=float), index=pd.Index(sub[unit_col].to_numpy()))
            if s.index.has_duplicates:
                if duplicates == "mean":
                    s = s.groupby(level=0, sort=False).mean()
                else:
                    dup = s.index[s.index.duplicated()].unique().tolist()
                    raise ValueError(f"label {label!r} has several rows for unit(s) {dup}; the statistical unit must be "
                                     f"unique per label (pass duplicates='mean' to average them)")
        out[label] = s
    return out


def _labels_in_order(df: pd.DataFrame, label_col: str, labels: Optional[Iterable[Any]]) -> List[Any]:
    present = list(pd.unique(df[label_col]))
    if labels is None:
        return present
    labels = list(labels)
    missing = [l for l in labels if l not in present]
    if missing:
        raise ValueError(f"labels {missing} not present in column {label_col!r}")
    return labels


def compare_to_reference(df: pd.DataFrame, metric: str, reference_label: Any, label_col: str = "label", unit_col: str = "seed",
                         n_boot: int = DEFAULT_N_BOOT, seed: int = 0, alpha: float = DEFAULT_ALPHA, n_perm: int = DEFAULT_N_PERM,
                         higher_is_better: bool = True, labels: Optional[Iterable[Any]] = None,
                         duplicates: str = "raise") -> pd.DataFrame:
    """Compare every non-reference label with ``reference_label`` on ``metric``, paired by ``unit_col``.

    Rows are paired by an inner join on the units (seeds) present in both labels.  One output row per
    non-reference label (the reference itself is absent) with ``label, reference, metric, n_pairs,
    mean_label, mean_ref, mean_diff (label - reference), ci_low, ci_high, p_perm, p_holm, p_wilcoxon,
    cohen_dz, t_p, better, worse``.  ``p_holm`` is the Holm correction over the family of comparisons
    made in this call; ``better = mean_diff > 0 and p_holm < alpha`` (sign reversed when
    ``higher_is_better=False``), ``worse`` analogously.
    """
    per = _per_unit(df, metric, label_col, unit_col, duplicates)
    if reference_label not in per:
        raise ValueError(f"reference label {reference_label!r} not found in column {label_col!r} (have {list(per)})")
    ref = per[reference_label].dropna()
    order = [l for l in _labels_in_order(df, label_col, labels) if l != reference_label]
    rows: List[Dict[str, Any]] = []
    nan_keys = ("mean_diff", "ci_low", "ci_high", "p_perm", "p_wilcoxon", "cohen_dz", "t_p")
    for label in order:
        s = per[label].dropna()
        joined = pd.concat([s.rename("x"), ref.rename("r")], axis=1, join="inner")
        row: Dict[str, Any] = {"label": label, "reference": reference_label, "metric": metric, "n_pairs": int(len(joined))}
        if len(joined) == 0:
            row.update({"mean_label": float("nan"), "mean_ref": float("nan")})
            row.update({k: float("nan") for k in nan_keys})
        else:
            res = paired_difference(joined["x"].to_numpy(), joined["r"].to_numpy(), n_boot=n_boot, seed=seed, alpha=alpha, n_perm=n_perm)
            row.update({"mean_label": res["mean_a"], "mean_ref": res["mean_b"]})
            row.update({k: res[k] for k in nan_keys})
        rows.append(row)
    cols = ["label", "reference", "metric", "n_pairs", "mean_label", "mean_ref", "mean_diff", "ci_low", "ci_high",
            "p_perm", "p_holm", "p_wilcoxon", "cohen_dz", "t_p", "better", "worse"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_perm"].to_numpy())
    sign = 1.0 if higher_is_better else -1.0
    signif = out["p_holm"].to_numpy() < alpha  # NaN -> False
    diff = sign * out["mean_diff"].to_numpy()
    out["better"] = (diff > 0) & signif
    out["worse"] = (diff < 0) & signif
    return out[cols].reset_index(drop=True)


def summary_table(df: pd.DataFrame, metrics: Union[str, Sequence[str]], label_col: str = "label", unit_col: Optional[str] = "seed",
                  n_boot: int = DEFAULT_N_BOOT, seed: int = 0, alpha: float = DEFAULT_ALPHA, labels: Optional[Iterable[Any]] = None,
                  duplicates: str = "raise") -> pd.DataFrame:
    """Per-label, per-metric summary: ``n, mean, ci_low, ci_high (bootstrap), std (ddof=1), sem``.

    Long format with one row per ``(label, metric)``; ``n`` counts the units with a non-NaN value.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    order = _labels_in_order(df, label_col, labels)
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        per = _per_unit(df, metric, label_col, unit_col, duplicates)
        for label in order:
            x = per[label].dropna().to_numpy(dtype=float)
            m, lo, hi = mean_ci(x, n_boot=n_boot, seed=seed, alpha=alpha)
            sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
            rows.append({"label": label, "metric": metric, "n": int(x.size), "mean": m, "ci_low": lo, "ci_high": hi,
                         "std": sd, "sem": sd / np.sqrt(x.size) if x.size > 1 else float("nan")})
    return pd.DataFrame(rows, columns=["label", "metric", "n", "mean", "ci_low", "ci_high", "std", "sem"])


def pairwise_comparisons(df: pd.DataFrame, metric: str, labels: Optional[Iterable[Any]] = None, label_col: str = "label",
                         unit_col: str = "seed", n_boot: int = DEFAULT_N_BOOT, seed: int = 0, alpha: float = DEFAULT_ALPHA,
                         n_perm: int = DEFAULT_N_PERM, duplicates: str = "raise") -> pd.DataFrame:
    """All unordered pairs of labels: ``label_a, label_b, n_pairs, mean_diff (a-b), ci_low, ci_high, p_perm, p_holm, cohen_dz``.

    ``p_holm`` is Holm-corrected over all pairs produced by this call.
    """
    per = _per_unit(df, metric, label_col, unit_col, duplicates)
    order = _labels_in_order(df, label_col, labels)
    rows: List[Dict[str, Any]] = []
    for la, lb in itertools.combinations(order, 2):
        joined = pd.concat([per[la].dropna().rename("a"), per[lb].dropna().rename("b")], axis=1, join="inner")
        row: Dict[str, Any] = {"label_a": la, "label_b": lb, "metric": metric, "n_pairs": int(len(joined))}
        if len(joined) == 0:
            row.update({k: float("nan") for k in ("mean_diff", "ci_low", "ci_high", "p_perm", "p_wilcoxon", "cohen_dz")})
        else:
            res = paired_difference(joined["a"].to_numpy(), joined["b"].to_numpy(), n_boot=n_boot, seed=seed, alpha=alpha, n_perm=n_perm)
            row.update({k: res[k] for k in ("mean_diff", "ci_low", "ci_high", "p_perm", "p_wilcoxon", "cohen_dz")})
        rows.append(row)
    cols = ["label_a", "label_b", "metric", "n_pairs", "mean_diff", "ci_low", "ci_high", "p_perm", "p_holm", "p_wilcoxon", "cohen_dz"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p_perm"].to_numpy())
    return out[cols]


def pairwise_matrix(df: pd.DataFrame, metric: str, labels: Optional[Iterable[Any]] = None, value: str = "p_holm",
                    label_col: str = "label", unit_col: str = "seed", n_boot: int = DEFAULT_N_BOOT, seed: int = 0,
                    alpha: float = DEFAULT_ALPHA, n_perm: int = DEFAULT_N_PERM, duplicates: str = "raise") -> pd.DataFrame:
    """Square ``labels x labels`` DataFrame of ``value`` for all pairs (optional helper).

    ``value`` is one of ``p_holm`` (default), ``p_perm``, ``p_wilcoxon``, ``cohen_dz``, ``mean_diff``, ``n_pairs``.
    p-values are symmetric; ``mean_diff`` / ``cohen_dz`` are *row minus column* (antisymmetric).  The diagonal
    is NaN.
    """
    pairs = pairwise_comparisons(df, metric, labels=labels, label_col=label_col, unit_col=unit_col, n_boot=n_boot, seed=seed,
                                 alpha=alpha, n_perm=n_perm, duplicates=duplicates)
    order = _labels_in_order(df, label_col, labels)
    if value not in ("p_holm", "p_perm", "p_wilcoxon", "cohen_dz", "mean_diff", "n_pairs"):
        raise ValueError(f"unknown value {value!r}")
    antisym = value in ("mean_diff", "cohen_dz")
    mat = pd.DataFrame(np.nan, index=pd.Index(order, name=label_col), columns=pd.Index(order, name=label_col), dtype=float)
    for _, r in pairs.iterrows():
        v = float(r[value])
        mat.loc[r["label_a"], r["label_b"]] = v
        mat.loc[r["label_b"], r["label_a"]] = -v if antisym else v
    return mat
