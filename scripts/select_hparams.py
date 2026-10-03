#!/usr/bin/env python
"""Pick the best hyper-parameter variant per method from a dev-suite results tree.

Dev suites (``suites/*_dev.yaml``) expand sweeps into labels such as ``weight_clipping/lr=0.01/kappa=2.0``;
each label holds ``seed<k>/{config.json, summary.json}`` run directories.  This script

1. groups the variants by their *base label* (the text before the first ``/``; ``--group sweep`` strips only
   the trailing ``k=v`` segments instead, for labels such as ``mnist/sin/lr=0.01``),
2. averages a summary metric (default ``auc_norm``; ``--lower-is-better`` for losses) over seeds, ignoring
   variants with fewer than ``--min-seeds`` finished seeds,
3. picks the best variant per base label and recovers its overrides from ``config.json`` -- the dotted
   config keys that vary across the group's variants (restricted to the sweep / override keys recorded in
   ``<results>/suite.json`` when that file exists), plus the keys named by the label's ``k=v`` segments --
   which is more robust than parsing the label alone.  A label segment ``lr=0.01`` is mapped to a dotted
   config key in this order: the sweep key recorded for that label in ``<results>/suite.json`` (written by
   ``run_suite.py``), else the unique config key ending in ``.lr`` that *varies* across the group, else the
   unique one whose value equals the label value, else the only candidate; when this is still ambiguous
   (e.g. ``optimizer.lr`` and ``method.lr`` both equal 0.01 and nothing varies) the segment is reported as a
   warning and skipped -- it is never guessed, and a bare non-dotted key is never emitted,
4. writes a YAML file ``{metric, results, selected: {base: {dotted.key: value}}, details: {...}}`` that
   ``scripts/run_suite.py`` consumes through the suite key ``selected_hparams``, plus a markdown table of
   every variant (mean +/- std over seeds, n, chosen marker).

Usage::

    python scripts/select_hparams.py --results results/pmnist_dev --out suites/selected_pmnist.yaml \
        [--metric auc_norm] [--min-seeds 1] [--table reports/pmnist_dev_hparams.md] [--lower-is-better]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plasticity.utils import _parse_scalar, flatten_dict  # noqa: E402

__all__ = ["Variant", "discover_variants", "base_label", "select_hparams", "markdown_table", "write_selection", "main"]

_SEED_DIR = re.compile(r"^seed[_-]?(\d+)$")
_KV = re.compile(r"^([^=/]+)=(.*)$")
# config keys that legitimately differ between seeds / runs but are not hyper-parameters
_IGNORED_KEYS = {"seed", "stream_seed", "stream_seed_offset", "threads", "log_every", "save_model", "out", "quiet"}


@dataclass
class Variant:
    base: str                       # base label (group)
    label: str                      # full variant label, e.g. 'weight_clipping/lr=0.01/kappa=2.0'
    seeds: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    configs: List[Dict[str, Any]] = field(default_factory=list)   # flattened config.json per seed
    chosen: bool = False

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0

    @property
    def config(self) -> Dict[str, Any]:
        return self.configs[0] if self.configs else {}


def base_label(label: str, group: str = "first") -> str:
    """``'wc/lr=0.01/kappa=2' -> 'wc'`` (``group='first'``: text before the first '/';
    ``group='sweep'``: strip trailing ``k=v`` segments, so ``'mnist/sin/lr=0.01' -> 'mnist/sin'``)."""
    parts = label.split("/")
    if group == "sweep":
        while len(parts) > 1 and _KV.match(parts[-1]):
            parts = parts[:-1]
        return "/".join(parts)
    return parts[0]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def discover_variants(results: str | os.PathLike, metric: str = "auc_norm", group: str = "first") -> Tuple[List[Variant], List[str]]:
    """Walk ``results`` for finished runs (``summary.json``) and group them into variants.

    Returns ``(variants, problems)`` where ``problems`` lists runs without the metric / unreadable files.
    """
    root = Path(results)
    if not root.is_dir():
        raise FileNotFoundError(f"results dir {root} not found")
    variants: Dict[str, Variant] = {}
    problems: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if "summary.json" not in filenames:
            continue
        run = Path(dirpath)
        rel = run.relative_to(root)
        parts = list(rel.parts)
        seed: Optional[int] = None
        if parts and _SEED_DIR.match(parts[-1]):
            seed = int(_SEED_DIR.match(parts[-1]).group(1))
            parts = parts[:-1]
        label = "/".join(parts) if parts else root.name
        summary = _read_json(run / "summary.json")
        cfg = _read_json(run / "config.json") or {}
        if seed is None:
            seed = int(cfg.get("seed", summary.get("seed", -1) if summary else -1))
        if not summary or metric not in summary or summary[metric] is None:
            problems.append(f"{rel}: no '{metric}' in summary.json")
            continue
        try:
            score = float(summary[metric])
        except (TypeError, ValueError):
            problems.append(f"{rel}: '{metric}' is not numeric ({summary[metric]!r})")
            continue
        if not np.isfinite(score):
            problems.append(f"{rel}: '{metric}' is {score}")
            continue
        var = variants.setdefault(label, Variant(base=base_label(label, group), label=label))
        var.seeds.append(seed)
        var.scores.append(score)
        var.configs.append(flatten_dict(cfg))
    return sorted(variants.values(), key=lambda v: (v.base, v.label)), problems


_SCI_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def _parse_label_value(raw: str) -> Any:
    """Parse the value of a ``k=v`` label segment (``run_suite._label_value`` rendering) back to Python.

    ``'1e-05'`` is a *string* for YAML 1.1 (:func:`_parse_scalar`), so scientific-notation floats are coerced."""
    val = _parse_scalar(raw)
    if isinstance(val, str) and _SCI_NUMBER.fullmatch(val.strip()):
        return float(val)
    return val


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
    return a == b


def _last(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def _suite_jobs(results: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``label -> {'sweep': {dotted: value}, 'overrides': {dotted: value}}`` from ``<results>/suite.json`` (if any)."""
    doc = _read_json(results / "suite.json")
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not doc:
        return out
    for job in doc.get("jobs") or []:
        label = job.get("label") if isinstance(job, dict) else None
        if not label or label in out:
            continue
        out[str(label)] = {"sweep": flatten_dict(job.get("sweep") or {}), "overrides": flatten_dict(job.get("overrides") or {})}
    return out


def _resolve_label_key(short: str, val: Any, cfg: Dict[str, Any], varying: Sequence[str] = (),
                       job: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[Optional[str], List[str]]:
    """Dotted config key named by a label segment ``short=val``; returns ``(key or None, candidates)``."""
    if job:  # the sweep keys recorded by run_suite.py are authoritative
        for source in (job.get("sweep") or {}, job.get("overrides") or {}):
            cands = [k for k in source if _last(k) == short]
            if len(cands) == 1:
                return cands[0], cands
            by_val = [k for k in cands if _values_equal(source[k], val)]
            if len(by_val) == 1:
                return by_val[0], cands
    cands = [k for k in cfg if k == short or k.endswith("." + short)]
    if not cands:
        return (short if "." in short else None), cands   # a dotted key from the label is still a valid override
    pref = [k for k in cands if k in varying]
    if len(pref) == 1:
        return pref[0], cands
    by_val = [k for k in cands if _values_equal(cfg[k], val)]
    if len(by_val) == 1:
        return by_val[0], cands
    if len(cands) == 1:   # value differs (e.g. label rounding) -> trust the config
        return cands[0], cands
    return None, cands


def _label_overrides(label: str, base: str, cfg: Dict[str, Any], varying: Sequence[str] = (),
                     job: Optional[Dict[str, Dict[str, Any]]] = None, problems: Optional[List[str]] = None) -> Dict[str, Any]:
    """Overrides named by the ``k=v`` label segments, resolved to dotted config keys (see :func:`_resolve_label_key`).

    Segments that cannot be mapped to a unique key are appended to ``problems`` and skipped."""
    out: Dict[str, Any] = {}
    rest = label[len(base):].strip("/") if label.startswith(base) else label
    for seg in [s for s in rest.split("/") if s]:
        m = _KV.match(seg)
        if not m:
            continue
        short, val = m.group(1), _parse_label_value(m.group(2))
        key, cands = _resolve_label_key(short, val, cfg, varying, job)
        if key is None:
            if problems is not None:
                what = f"ambiguous config keys {sorted(cands)}" if cands else "no matching dotted config key"
                problems.append(f"{label}: label segment '{seg}' not selected ({what}); "
                                f"run select_hparams on a results root with suite.json or add the key to the final suite by hand")
            continue
        out[key] = cfg[key] if key in cfg else val
    return out


def _group_overrides(variants: Sequence[Variant], chosen: Variant,
                     allowed: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Dotted config keys that vary across the group's variants, with the chosen variant's values.

    With ``allowed`` (the sweep / override keys recorded for the chosen label in ``suite.json``) only those
    keys are considered, so config values that merely *co-vary* with a sweep (derived settings) are not
    turned into overrides."""
    keys = set()
    for v in variants:
        keys.update(v.config.keys())
    if allowed is not None:
        keys &= set(allowed)
    varying = []
    for k in sorted(keys):
        if k in _IGNORED_KEYS:
            continue
        vals = []
        for v in variants:
            vals.append(json.dumps(v.config.get(k, "<missing>"), sort_keys=True, default=str))
        if len(set(vals)) > 1:
            varying.append(k)
    return {k: chosen.config[k] for k in varying if k in chosen.config}


def select_hparams(results: str | os.PathLike, metric: str = "auc_norm", min_seeds: int = 1, higher_is_better: bool = True,
                   group: str = "first") -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[Variant], List[str]]:
    """Return ``(selected, details, variants, problems)``.

    ``selected``  base label -> chosen dotted overrides
    ``details``   base label -> {variant, mean, std, n, seeds, n_variants}
    ``variants``  every variant (``chosen`` flag set on the winners), sorted by base then score
    """
    variants, problems = discover_variants(results, metric=metric, group=group)
    jobs = _suite_jobs(Path(results))   # sweep keys per label, when the tree was written by run_suite.py
    selected: Dict[str, Dict[str, Any]] = {}
    details: Dict[str, Dict[str, Any]] = {}
    bases = sorted({v.base for v in variants})
    for base in bases:
        members = [v for v in variants if v.base == base]
        eligible = [v for v in members if v.n >= int(min_seeds)]
        if not eligible:
            problems.append(f"{base}: no variant has >= {min_seeds} finished seeds")
            continue
        key = (lambda v: (v.mean, -v.std)) if higher_is_better else (lambda v: (-v.mean, -v.std))
        best = max(eligible, key=key)   # ties -> first in label order (stable)
        best.chosen = True
        job = jobs.get(best.label)
        allowed = (list(job["sweep"]) + list(job["overrides"])) if job else None
        overrides = _group_overrides(members, best, allowed=allowed)
        label_over = _label_overrides(best.label, base, best.config, varying=list(overrides), job=job, problems=problems)
        for k, val in label_over.items():
            overrides.setdefault(k, val)
        selected[base] = overrides
        details[base] = {"variant": best.label, "mean": best.mean, "std": best.std, "n": best.n,
                         "seeds": sorted(best.seeds), "n_variants": len(members)}
    sign = -1.0 if higher_is_better else 1.0
    variants.sort(key=lambda v: (v.base, sign * v.mean, v.label))
    return selected, details, variants, problems


def markdown_table(variants: Sequence[Variant], metric: str) -> str:
    lines = [f"| base | variant | {metric} (mean ± std) | n | seeds | chosen |", "|---|---|---|---|---|---|"]
    for v in variants:
        seeds = ",".join(str(s) for s in sorted(v.seeds))
        lines.append(f"| {v.base} | {v.label} | {v.mean:.4f} ± {v.std:.4f} | {v.n} | {seeds} | {'*' if v.chosen else ''} |")
    return "\n".join(lines) + "\n"


def _plain(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def write_selection(out: str | os.PathLike, selected: Dict[str, Dict[str, Any]], details: Dict[str, Dict[str, Any]],
                    variants: Sequence[Variant], metric: str, results: str | os.PathLike, higher_is_better: bool = True,
                    table: Optional[str | os.PathLike] = None, min_seeds: int = 1) -> Tuple[Path, Path]:
    """Write the YAML selection and the markdown table; returns ``(yaml_path, table_path)``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"metric": metric, "direction": "max" if higher_is_better else "min", "results": str(results),
           "min_seeds": int(min_seeds), "selected": _plain(selected), "details": _plain(details)}
    header = (f"# Hyper-parameters selected by scripts/select_hparams.py from {results}\n"
              f"# metric: {metric} ({'higher' if higher_is_better else 'lower'} is better), mean over seeds; "
              f"'selected' maps base label -> dotted overrides (consumed by run_suite.py via 'selected_hparams').\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None, allow_unicode=True, width=120)
    table_path = Path(table) if table is not None else out.with_suffix(".md")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# Hyper-parameter search: {results}\n\nMetric: `{metric}` ({'higher' if higher_is_better else 'lower'} is better), "
                f"mean ± std over seeds.\n\n")
        f.write(markdown_table(variants, metric))
    return out, table_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="dev-suite results dir, e.g. results/pmnist_dev")
    ap.add_argument("--out", default=None, help="output YAML (default: <results>/selected.yaml)")
    ap.add_argument("--metric", default="auc_norm", help="summary.json key to maximise (default auc_norm)")
    ap.add_argument("--min-seeds", type=int, default=1, help="ignore variants with fewer finished seeds (default 1)")
    ap.add_argument("--lower-is-better", action="store_true", help="minimise the metric instead")
    ap.add_argument("--group", choices=["first", "sweep"], default="first",
                    help="base label = text before the first '/' (default) or label without trailing k=v segments")
    ap.add_argument("--table", default=None, help="markdown table path (default: <out>.md)")
    args = ap.parse_args(argv)
    out = args.out or str(Path(args.results) / "selected.yaml")
    selected, details, variants, problems = select_hparams(args.results, metric=args.metric, min_seeds=args.min_seeds,
                                                           higher_is_better=not args.lower_is_better, group=args.group)
    yaml_path, table_path = write_selection(out, selected, details, variants, args.metric, args.results,
                                            higher_is_better=not args.lower_is_better, table=args.table, min_seeds=args.min_seeds)
    print(markdown_table(variants, args.metric))
    for base, d in details.items():
        print(f"{base}: {d['variant']}  {args.metric}={d['mean']:.4f} (n={d['n']})  -> {selected[base]}")
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)
    print(f"wrote {yaml_path} and {table_path}")
    return 0 if selected else 1


if __name__ == "__main__":
    sys.exit(main())
