#!/usr/bin/env python3
"""Companion tables for the scale-up sections of docs/RAPOR.md (§14.2 / §14.3) that scripts/analyze.py does
not print: online accuracy in task blocks, mechanism trajectories at chosen tasks, and paired differences
for arbitrary label pairs (e.g. tri − weight_clipping), all as mean over seeds with the paired bootstrap CI
and t-test of plasticity.metrics.stats.

    python3 scripts/scaleup_tables.py results/pmnist_fullstream --block 100 \
        --pairs sin:baseline,tanh:baseline,weight_clipping:baseline,tri:baseline,tri:weight_clipping \
        --mech-tasks 0,200,400,600,799
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plasticity.metrics.stats import paired_difference  # noqa: E402
from plasticity.utils import read_jsonl  # noqa: E402

SUMMARY_KEYS = ["auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "fresh_gap_final",
                "final_dead_frac/all", "final_effective_rank/last", "final_w_abs_mean/all", "final_step_dw_norm",
                "final_sat_frac/all", "final_jac_sq_mean/all", "final_w_over_A_mean/all"]
MECH_KEYS = ["online_accuracy", "sat_frac/all", "sat_frac/L0", "jac_sq_mean/all", "w_over_A_mean/all", "w_abs_mean/all",
             "dead_frac/all", "effective_rank/last", "step_dw_norm"]


def load(results: Path):
    runs = {}
    for f in sorted(results.glob("*/seed*/summary.json")):
        label, seed = f.parent.parent.name, int(f.parent.name[4:])
        runs.setdefault(label, {})[seed] = {"summary": json.load(open(f)), "rows": read_jsonl(f.parent / "tasks.jsonl")}
    return runs


def fmt(m, s=None):
    return f"{m:.3f}" if s is None else f"{m:.3f} ± {s:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--metric", default="online_accuracy")
    ap.add_argument("--pairs", default="", help="comma-separated a:b pairs for paired differences (a − b)")
    ap.add_argument("--mech-tasks", default="0,200,400,600,799")
    ap.add_argument("--labels", default=None, help="comma-separated labels (display order)")
    args = ap.parse_args()
    runs = load(Path(args.results))
    labels = args.labels.split(",") if args.labels else list(runs)
    print(f"# {args.results}: " + ", ".join(f"{l} (n={len(runs[l])})" for l in labels))

    print(f"\n## Summary metrics, mean ± sd over seeds")
    print("| label | n | " + " | ".join(SUMMARY_KEYS) + " |")
    for l in labels:
        cells = []
        for k in SUMMARY_KEYS:
            v = [r["summary"][k] for r in runs[l].values() if k in r["summary"]]
            cells.append(fmt(np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else "—")
        print(f"| {l} | {len(runs[l])} | " + " | ".join(cells) + " |")

    print(f"\n## {args.metric} in blocks of {args.block} tasks, mean over seeds (sd in brackets)")
    for l in labels:
        per_seed = []
        for r in runs[l].values():
            acc = np.array([row[args.metric] for row in r["rows"]])
            per_seed.append([acc[i:i + args.block].mean() for i in range(0, len(acc), args.block)])
        a = np.array(per_seed)
        print(f"- {l}: " + " → ".join(f"{m:.3f}" + (f" ({s:.3f})" if a.shape[0] > 1 else "") for m, s in zip(a.mean(0), a.std(0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1]))))

    tasks = [int(t) for t in args.mech_tasks.split(",")]
    print(f"\n## Mechanism trajectories at tasks {tasks}, mean over seeds")
    for k in MECH_KEYS:
        print(f"- {k}:")
        for l in labels:
            vals = []
            for t in tasks:
                v = [row[k] for r in runs[l].values() for row in r["rows"] if int(row["task"]) == t and k in row]
                vals.append(np.mean(v) if v else np.nan)
            if np.all(np.isnan(vals)):
                continue
            print(f"    {l:<16} " + " → ".join("—" if np.isnan(v) else f"{v:.3f}" for v in vals))

    if args.pairs:
        print("\n## Paired differences (a − b, same seeds): mean [95% bootstrap CI], t-test p, Cohen d_z")
        for pair in args.pairs.split(","):
            a, b = pair.split(":")
            seeds = sorted(set(runs[a]) & set(runs[b]))
            for k in ["auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "fresh_gap_final",
                      "final_dead_frac/all", "final_effective_rank/last"]:
                xa = [runs[a][s]["summary"][k] for s in seeds]
                xb = [runs[b][s]["summary"][k] for s in seeds]
                d = paired_difference(xa, xb)
                print(f"- {a} − {b} | {k:<26} n={len(seeds)} | {d['mean_diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
                      f"p_t={d.get('t_p', float('nan')):.3f} d_z={d.get('cohen_dz', float('nan')):+.2f} | per seed: "
                      + ", ".join(f"{u - v:+.4f}" for u, v in zip(xa, xb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
