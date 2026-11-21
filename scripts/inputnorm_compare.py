"""Paired comparison of standardized-input runs (results/pmnist_inputnorm) with the unit-input runs of the same
seeds (results/pmnist_adam10 for Adam, results/pmnist_main for SGD).  Prints a markdown table of per-condition
means (seeds 0-4), the paired difference standardized - unit with a 95% bootstrap interval, the first-hidden-layer
dead fraction, and how many seeds meet the three-part retention criterion."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
SEEDS = [0, 1, 2, 3, 4]
PAIRS = [  # (standardized label, unit-input directory, unit-input label)
    ("adam_std", "pmnist_adam10", "adam_std"), ("adam_sin", "pmnist_adam10", "adam_sin"),
    ("adam_tanh", "pmnist_adam10", "adam_tanh"), ("adam_tri", "pmnist_adam10", "adam_tri"),
    ("adam_weight_clipping", "pmnist_adam10", "adam_weight_clipping"),
    ("sgd_std", "pmnist_main", "baseline"), ("sgd_tri", "pmnist_main", "tri"),
    ("sgd_weight_clipping", "pmnist_main", "weight_clipping"),
]

def load(d, label, seed):
    p = ROOT / d / label / f"seed{seed}"
    s = json.load(open(p / "summary.json"))
    rows = [json.loads(l) for l in open(p / "tasks.jsonl") if l.strip()]
    return s, rows

def last_with(rows, key):
    for r in reversed(rows):
        if r.get(key) is not None:
            return r[key]
    return float("nan")

def at_task(rows, key, t):
    for r in rows:
        if r.get("task") == t and r.get(key) is not None:
            return r[key]
    return float("nan")

def crit(s):
    return (s["retention_ratio"] >= 0.99 and s["slope_per_100"] >= -0.002 and s["fresh_gap_final"] <= 0.01)

def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def main():
    keys = ["auc_norm", "final_window_mean", "retention_ratio", "fresh_gap_final"]
    print("| model | input | AUC | final window | retention | fresh gap | dead L0 (task 10) | dead L0 (final) | dead all (final) | criterion |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    diffs = {}
    for lab, ud, ul in PAIRS:
        for name, d, l in (("unit", ud, ul), ("standardized", "pmnist_inputnorm", lab)):
            S, R = zip(*[load(d, l, s) for s in SEEDS])
            vals = {k: np.array([s[k] for s in S]) for k in keys}
            d0_10 = np.array([at_task(r, "dead_frac/L0", 10) for r in R])
            d0_f = np.array([last_with(r, "dead_frac/L0") for r in R])
            da_f = np.array([s["final_dead_frac/all"] for s in S])
            nc = sum(crit(s) for s in S)
            diffs[(lab, name)] = dict(vals, d0_f=d0_f, da_f=da_f)
            print(f"| {lab} | {name} | " + " | ".join(f"{vals[k].mean():.3f} ± {vals[k].std(ddof=1):.3f}" for k in keys)
                  + f" | {d0_10.mean():.2f} | {d0_f.mean():.2f} | {da_f.mean():.2f} | {nc}/{len(SEEDS)} |")
    print("\nPaired differences, standardized − unit (same seeds), mean [95% bootstrap interval], per seed:")
    for lab, _, _ in PAIRS:
        for k in keys + ["d0_f"]:
            d = diffs[(lab, "standardized")][k] - diffs[(lab, "unit")][k]
            m, lo, hi = boot(d)
            print(f"- {lab} | {k}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | " + ", ".join(f"{x:+.4f}" for x in d))
    return 0

if __name__ == "__main__":
    sys.exit(main())
