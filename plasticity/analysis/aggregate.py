"""Load a results suite into two DataFrames and aggregate per-task curves over seeds.

Result layout (written by ``scripts/run_suite.py`` / ``plasticity.run``)::

    results/<suite>/<label>/seed<k>/{config.json, tasks.jsonl, summary.json}      # nested (label may be
    results/<suite>/<label>/<sub>/seed<k>/...                                     #  several levels deep)
    results/<suite>/<run_dir>/{config.json, tasks.jsonl, summary.json}            # flat (label = run_dir)

A *run directory* is any directory below the suite root that contains ``config.json``.  Its label is
its path relative to the suite root with a trailing ``seed<k>`` component removed; the seed comes from
that component, else from ``config.json['seed']``, else from ``summary.json['seed']``.  Sweep suites
(``scripts/run_suite.py``) label their runs ``<base>/<key>=<value>[/...]`` (e.g. ``sin/lr=0.01``); the
*base label* is the first path component (``sin``) and is exposed as the ``base_label`` column, which the
sensitivity figure (``plots.plot_sweep``) groups by.

``load_suite`` returns ``(runs, curves)``:

* ``runs``   -- one row per *finished* run (``summary.json`` present): ``label, base_label, seed, run_dir,
  run_path``, every ``summary.json`` key, and the flattened config as ``cfg.<dotted.key>`` columns.
* ``curves`` -- long table, one row per task of every run: ``label, base_label, seed, task`` and every key
  of the ``tasks.jsonl`` row (mechanism keys are NaN on the tasks where they were not logged).

``load_suite(labels=[...])`` matches a requested label against the full label *or* its leading path
components (``'sin'`` selects ``sin``, ``sin/lr=0.01``, ``sin/lr=0.01/kappa=2`` ...); requested labels that
match nothing are reported (``runs.attrs['unmatched_labels']``).

Unfinished runs (no ``summary.json``) are skipped; they are listed in ``runs.attrs['skipped']`` (and
printed when ``verbose=True``).  ``mean_curves`` aggregates a metric over seeds with a Student-t 95 %
confidence interval (the statistical unit is the seed, as in the thesis' paired design).

Cache (``load_suite(cache=dir)``): the tables of the **whole** suite are written to ``dir`` together with a
sidecar ``suite_manifest.json`` that records every finished run directory and the mtime / size of its
``summary.json`` and ``tasks.jsonl``.  The cached tables are reused only when that manifest matches the
suite on disk (same run directories, same files), so runs that finish, are re-run or are removed after
the cache was written invalidate it automatically; ``labels`` is applied to the *returned* frames only and
never to what is cached.  ``refresh=True`` forces a rebuild.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..utils import flatten_dict, read_jsonl

__all__ = ["discover_runs", "load_run", "load_suite", "mean_curves", "auto_metric", "has_fresh_data",
           "save_tables", "load_tables", "t_half_width", "base_label", "match_labels", "MECHANISM_KEYS", "PERFORMANCE_KEYS"]

_SEED_DIR = re.compile(r"^seed[_-]?(\d+)$")
_MANIFEST = "config.json"
_CACHE_MANIFEST = "suite_manifest.json"  # sidecar of the cached tables (see load_suite)
_CACHE_VERSION = 1

PERFORMANCE_KEYS = ("online_accuracy", "test_accuracy", "online_loss", "fresh_accuracy")
MECHANISM_KEYS = ("dead_frac/all", "inactive_frac/all", "w_abs_mean/all", "w_fro_norm/all", "stable_rank/last",
                  "effective_rank/last", "stable_rank_c/last", "effective_rank_c/last", "jac_sq_mean/all",
                  "w_over_A_mean/all", "sat_frac/all", "grad_norm_theta", "grad_norm_w", "fisher_trace_theta",
                  "fisher_trace_w", "fisher_erank_theta", "fisher_erank_w", "jac_fro_ratio",
                  "step_grad_norm_theta", "step_dw_norm", "step_dtheta_norm")


# ------------------------------------------------------------------------------------------------
# discovery
# ------------------------------------------------------------------------------------------------
def base_label(label: Any) -> str:
    """The label without its sweep segments: the first ``/``-separated component (``'sin/lr=0.01' -> 'sin'``)."""
    return str(label).split("/")[0]


def match_labels(requested: Iterable[str], present: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Resolve requested labels against the labels of a suite.

    A requested label ``r`` matches a present label ``p`` when ``p == r`` or ``p`` starts with ``r + '/'``
    (i.e. ``r`` is the base label or a leading path of a sweep label).  Returns ``(matched, unmatched)``:
    the matched present labels in the order of the requests (sorted within one request, no duplicates),
    and the requested labels that matched nothing.
    """
    present = [str(p) for p in dict.fromkeys(present)]
    matched: List[str] = []
    unmatched: List[str] = []
    for r in requested:
        r = str(r).strip().strip("/")
        hits = [p for p in present if p == r or p.startswith(r + "/")]
        if not hits:
            unmatched.append(r)
            continue
        for p in sorted(hits):
            if p not in matched:
                matched.append(p)
    return matched, unmatched


def _ensure_base_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``base_label`` column (after ``label``) when it is missing, e.g. in tables cached by an older version."""
    if "label" in df.columns and "base_label" not in df.columns:
        df = df.copy()
        df.insert(list(df.columns).index("label") + 1, "base_label", df["label"].map(base_label))
    return df


def _label_and_seed(rel: Path) -> Tuple[str, Optional[int]]:
    parts = list(rel.parts)
    seed: Optional[int] = None
    if parts:
        m = _SEED_DIR.match(parts[-1])
        if m:
            seed = int(m.group(1))
            parts = parts[:-1]
    label = "/".join(parts) if parts else rel.name
    return label, seed


def discover_runs(root: str | os.PathLike) -> List[Dict[str, Any]]:
    """Find run directories below ``root``.

    Returns a list of dicts ``{label, base_label, seed, run_dir (relative), run_path (absolute), finished (bool)}``
    sorted by ``(label, seed)``.  ``seed`` is None when it cannot be determined from the directory name
    (``load_run`` then looks into ``config.json`` / ``summary.json``).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"results root {root} is not a directory")
    found: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if _MANIFEST in filenames or "tasks.jsonl" in filenames:
            p = Path(dirpath)
            rel = p.relative_to(root)
            if rel == Path("."):  # the root itself is a run -> label = root name
                rel = Path(root.name)
            label, seed = _label_and_seed(rel)
            found.append({"label": label, "base_label": base_label(label), "seed": seed, "run_dir": str(rel),
                          "run_path": str(p.resolve()), "finished": (p / "summary.json").is_file()})
            dirnames[:] = []  # do not descend into a run directory
    found.sort(key=lambda r: (r["label"], -1 if r["seed"] is None else r["seed"]))
    return found


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------
def _jsonable_scalar(v: Any) -> Any:
    """Lists / dicts become JSON strings so that the resulting columns are hashable and CSV-stable."""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run(run_path: str | os.PathLike, label: Optional[str] = None, seed: Optional[int] = None,
             cfg_prefix: str = "cfg.") -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load one run directory.

    Returns ``(run_row, task_rows)``.  ``run_row`` is None when ``summary.json`` is missing (unfinished run);
    ``task_rows`` holds the rows of ``tasks.jsonl`` (possibly empty) with ``label``, ``base_label`` and ``seed`` added.
    """
    p = Path(run_path)
    cfg: Dict[str, Any] = {}
    if (p / _MANIFEST).is_file():
        try:
            cfg = _read_json(p / _MANIFEST)
        except (OSError, json.JSONDecodeError) as e:  # pragma: no cover - defensive
            warnings.warn(f"{p}: cannot read config.json ({e})")
    if seed is None:
        seed = cfg.get("seed")
    summary: Optional[Dict[str, Any]] = None
    if (p / "summary.json").is_file():
        try:
            summary = _read_json(p / "summary.json")
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"{p}: cannot read summary.json ({e}); treated as unfinished")
            summary = None
    if seed is None and summary is not None:
        seed = summary.get("seed")
    if label is None:
        label = p.name
    rows: List[Dict[str, Any]] = []
    if (p / "tasks.jsonl").is_file():
        try:
            rows = read_jsonl(p / "tasks.jsonl")
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"{p}: cannot read tasks.jsonl ({e}); curves dropped")
            rows = []
    task_rows = []
    for r in rows:
        if "task" not in r:
            continue
        tr = {"label": label, "base_label": base_label(label), "seed": seed}
        tr.update({k: _jsonable_scalar(v) for k, v in r.items()})
        task_rows.append(tr)
    run_row: Optional[Dict[str, Any]] = None
    if summary is not None:
        run_row = {"label": label, "base_label": base_label(label), "seed": seed, "run_dir": str(p.name), "run_path": str(p.resolve())}
        run_row.update({k: _jsonable_scalar(v) for k, v in summary.items()})
        run_row["seed"] = seed  # the directory / config seed wins over the summary's
        run_row["n_logged_tasks"] = len(task_rows)
        run_row.update({f"{cfg_prefix}{k}": _jsonable_scalar(v) for k, v in flatten_dict(cfg).items()})
    return run_row, task_rows


def _load_runs(found: List[Dict[str, Any]], include_unfinished: bool, verbose: bool) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Read the run directories in ``found`` into ``(runs, curves, skipped)`` (see :func:`load_suite`)."""
    run_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for info in found:
        run_row, task_rows = load_run(info["run_path"], label=info["label"], seed=info["seed"])
        if run_row is None:
            skipped.append(info["run_dir"])
            if include_unfinished:
                curve_rows.extend(task_rows)
            continue
        run_row["run_dir"] = info["run_dir"]
        run_rows.append(run_row)
        curve_rows.extend(task_rows)
    if verbose and skipped:
        print(f"[load_suite] skipped {len(skipped)} unfinished run(s) (no summary.json): " + ", ".join(skipped))
    runs = pd.DataFrame(run_rows)
    curves = pd.DataFrame(curve_rows)
    if len(runs):
        runs = runs.sort_values(["label", "seed"], kind="stable").reset_index(drop=True)
    else:
        runs = pd.DataFrame(columns=["label", "base_label", "seed", "run_dir", "run_path"])
    if len(curves):
        curves = curves.sort_values(["label", "seed", "task"], kind="stable").reset_index(drop=True)
    else:
        curves = pd.DataFrame(columns=["label", "base_label", "seed", "task"])
    return runs, curves, skipped


def _suite_fingerprint(root: Path, found: List[Dict[str, Any]], include_unfinished: bool) -> Dict[str, Any]:
    """Everything the cached tables depend on: the run directories and the mtime / size of the files they were
    built from (``summary.json`` + ``tasks.jsonl`` of finished runs; ``tasks.jsonl`` of unfinished runs only when
    ``include_unfinished``).  One ``stat`` per file, so checking it is far cheaper than re-reading the suite."""
    entries: List[List[Any]] = []
    for info in found:
        p = Path(info["run_path"])
        files = ["summary.json", "tasks.jsonl"] if info["finished"] else (["tasks.jsonl"] if include_unfinished else [])
        for name in files:
            f = p / name
            if f.is_file():
                st = f.stat()
                entries.append([info["run_dir"], name, int(st.st_mtime_ns), int(st.st_size)])
    return {"version": _CACHE_VERSION, "root": str(root.resolve()), "include_unfinished": bool(include_unfinished), "runs": entries}


def _read_cache(cache_dir: Path, fingerprint: Dict[str, Any], verbose: bool):
    """The cached ``(runs, curves, skipped)`` when the manifest in ``cache_dir`` matches ``fingerprint``, else None."""
    mp = cache_dir / _CACHE_MANIFEST
    if not mp.is_file():
        return None
    try:
        man = _read_json(mp)
    except (OSError, json.JSONDecodeError):
        return None
    if man.get("fingerprint") != fingerprint:
        if verbose:
            print(f"[load_suite] cache {cache_dir} is stale (suite changed on disk); rebuilding")
        return None
    frames = []
    try:
        for name in ("runs", "curves"):
            fp = cache_dir / str(man["files"][name])
            frames.append(pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp))
    except (KeyError, TypeError, OSError, ValueError) as e:
        if verbose:
            print(f"[load_suite] cache {cache_dir} unreadable ({e}); rebuilding")
        return None
    runs, curves = _ensure_base_label(frames[0]), _ensure_base_label(frames[1])
    return runs, curves, [str(s) for s in man.get("skipped", [])]


def _write_cache(cache_dir: Path, runs: pd.DataFrame, curves: pd.DataFrame, fingerprint: Dict[str, Any], skipped: List[str]) -> None:
    paths = save_tables(runs, curves, cache_dir)  # tables first: a crash before the manifest leaves a (stale) cache that is rebuilt
    man = {"fingerprint": fingerprint, "files": {k: p.name for k, p in paths.items()}, "skipped": list(skipped)}
    with open(cache_dir / _CACHE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)


def load_suite(path: str | os.PathLike, labels: Optional[Iterable[str]] = None, include_unfinished: bool = False,
               verbose: bool = True, cache: Optional[str | os.PathLike] = None, refresh: bool = False,
               return_skipped: bool = False):
    """Load every run below ``path``.  Returns ``(runs, curves)`` (plus ``skipped`` if ``return_skipped``).

    ``labels``            keep only these labels (None = all); a requested label also selects every label
                          below it in the sweep hierarchy (``'sin'`` -> ``sin/lr=0.01`` ...), see :func:`match_labels`.
                          Requested labels that match nothing are stored in ``runs.attrs['unmatched_labels']``.
    ``include_unfinished`` also put the task rows of runs without ``summary.json`` into ``curves``
                          (they never appear in ``runs``).
    ``cache``             directory holding the tables of the *whole* suite plus ``suite_manifest.json``.  The
                          tables are reused only when the manifest matches the suite on disk (same run
                          directories, same ``summary.json`` / ``tasks.jsonl`` mtimes and sizes) and
                          ``refresh`` is False; otherwise the suite is re-read and the cache rewritten.  The
                          ``labels`` filter is applied to the returned frames only, never to the cache.
    ``runs.attrs['skipped']`` lists the skipped (unfinished / unreadable) run directories (of the kept labels).
    """
    root = Path(path)
    found = discover_runs(root)
    keep_labels: Optional[set] = None
    unmatched: List[str] = []
    if labels is not None:
        keep, unmatched = match_labels(labels, [f["label"] for f in found])
        keep_labels = set(keep)
        if verbose and unmatched:
            print(f"[load_suite] requested label(s) matching nothing under {root}: {unmatched}")
    loaded = None
    fingerprint: Optional[Dict[str, Any]] = None
    if cache is not None:
        cache_dir = Path(cache)
        fingerprint = _suite_fingerprint(root, found, include_unfinished)
        if not refresh:
            loaded = _read_cache(cache_dir, fingerprint, verbose)
    if loaded is None:
        # without a cache only the requested labels are read; with one the whole suite is read so that the
        # cached tables never depend on `labels`
        subset = found if (keep_labels is None or cache is not None) else [f for f in found if f["label"] in keep_labels]
        runs, curves, skipped = _load_runs(subset, include_unfinished, verbose)
        if cache is not None:
            _write_cache(cache_dir, runs, curves, fingerprint, skipped)
    else:
        runs, curves, skipped = loaded
    if keep_labels is not None:
        runs = runs[runs["label"].astype(str).isin(keep_labels)].reset_index(drop=True)
        curves = curves[curves["label"].astype(str).isin(keep_labels)].reset_index(drop=True)
        label_of = {f["run_dir"]: f["label"] for f in found}
        skipped = [s for s in skipped if label_of.get(s, s) in keep_labels]
    runs.attrs["skipped"] = skipped
    runs.attrs["unmatched_labels"] = unmatched
    runs.attrs["root"] = str(root.resolve())
    return (runs, curves, skipped) if return_skipped else (runs, curves)


# ------------------------------------------------------------------------------------------------
# cache helpers (parquet when an engine is installed, csv otherwise)
# ------------------------------------------------------------------------------------------------
def _parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def save_tables(runs: pd.DataFrame, curves: pd.DataFrame, out_dir: str | os.PathLike, fmt: str = "auto") -> Dict[str, Path]:
    """Write ``runs`` and ``curves`` to ``out_dir`` as parquet (if available / requested) or csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "auto":
        fmt = "parquet" if _parquet_available() else "csv"
    elif fmt == "parquet" and not _parquet_available():
        warnings.warn("no parquet engine (pyarrow / fastparquet) installed; writing csv instead")
        fmt = "csv"
    elif fmt not in ("parquet", "csv"):
        raise ValueError(f"fmt must be 'auto', 'parquet' or 'csv', got {fmt!r}")
    paths: Dict[str, Path] = {}
    for name, df in (("runs", runs), ("curves", curves)):
        p = out / f"{name}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        paths[name] = p
    return paths


def load_tables(cache_dir: str | os.PathLike) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Read cached tables written by :func:`save_tables`; None when absent."""
    d = Path(cache_dir)
    for fmt in ("parquet", "csv"):
        rp, cp = d / f"runs.{fmt}", d / f"curves.{fmt}"
        if rp.is_file() and cp.is_file():
            if fmt == "parquet":
                return pd.read_parquet(rp), pd.read_parquet(cp)
            return pd.read_csv(rp), pd.read_csv(cp)
    return None


# ------------------------------------------------------------------------------------------------
# aggregation over seeds
# ------------------------------------------------------------------------------------------------
def t_half_width(n: np.ndarray, sd: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Half width of the Student-t ``(1-alpha)`` CI of a mean: ``t_{1-alpha/2, n-1} * sd / sqrt(n)`` (NaN for n < 2)."""
    n = np.asarray(n, dtype=float)
    sd = np.asarray(sd, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = sps.t.ppf(1 - alpha / 2, df=np.maximum(n - 1, 1))
        half = q * sd / np.sqrt(n)
    return np.where(n >= 2, half, np.nan)


def mean_curves(curves: pd.DataFrame, metric: str, smooth: Optional[int] = None, labels: Optional[Sequence[str]] = None,
                alpha: float = 0.05, label_col: str = "label", unit_col: str = "seed", x_col: str = "task") -> pd.DataFrame:
    """Per label and task: ``n, mean, std, ci_low, ci_high`` of ``metric`` over seeds (Student-t CI).

    ``smooth`` (int > 1) applies a centred moving average of that window to every seed's series *before*
    aggregating (over the tasks where the metric is logged), so the CI describes the smoothed quantity.
    Tasks where the metric is NaN (e.g. mechanism metrics logged every n tasks) are dropped.
    Returns an empty frame with the right columns when the metric is absent.
    """
    cols = [label_col, x_col, "n", "mean", "std", "ci_low", "ci_high"]
    if metric not in curves.columns or len(curves) == 0:
        return pd.DataFrame(columns=cols)
    df = curves[[label_col, unit_col, x_col, metric]].copy()
    if labels is not None:
        df = df[df[label_col].isin(list(labels))]
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = df.sort_values([label_col, unit_col, x_col], kind="stable")
    if smooth is not None and int(smooth) > 1:
        w = int(smooth)
        df[metric] = df.groupby([label_col, unit_col], sort=False)[metric].transform(
            lambda s: s.rolling(w, center=True, min_periods=1).mean())
    g = df.groupby([label_col, x_col], sort=True)[metric]
    out = g.agg(n="count", mean="mean", std=lambda s: s.std(ddof=1)).reset_index()
    half = t_half_width(out["n"].to_numpy(), out["std"].to_numpy(), alpha)
    out["ci_low"] = out["mean"] - half
    out["ci_high"] = out["mean"] + half
    if labels is not None:  # keep the requested order
        order = {l: i for i, l in enumerate(labels)}
        out = out.assign(_o=out[label_col].map(order)).sort_values(["_o", x_col], kind="stable").drop(columns="_o")
    return out[cols].reset_index(drop=True)


def auto_metric(runs: pd.DataFrame, curves: Optional[pd.DataFrame] = None) -> str:
    """Pick the performance metric of a suite: ``summary['metric']`` (majority) else by column presence."""
    if "metric" in runs.columns and runs["metric"].notna().any():
        m = runs["metric"].dropna().mode()
        if len(m):
            return str(m.iloc[0])
    if curves is not None:
        for m in ("test_accuracy", "online_accuracy"):
            if m in curves.columns and curves[m].notna().any():
                return m
    for m in ("final_test_accuracy", "auc_norm"):
        if m in runs.columns:
            return "test_accuracy" if m == "final_test_accuracy" else "online_accuracy"
    return "online_accuracy"


def has_fresh_data(curves: pd.DataFrame, fresh_key: str = "fresh_accuracy") -> bool:
    return fresh_key in curves.columns and bool(pd.to_numeric(curves[fresh_key], errors="coerce").notna().any())
