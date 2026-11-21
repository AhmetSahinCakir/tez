"""Random Label MNIST control analysis.

(a) Positive control: L2 Init (results/rlmnist_ctrl_main, ten seeds) against the five models of
    results/rlmnist_main on the same seeds and streams (paired differences, 95% bootstrap interval).
(b) Diagnostic on the development stream (one seed): standardized inputs for the standard network and the
    triangle wave (results/rlmnist_ctrl_dev) against the [0,1]-input development runs (results/rlmnist_dev).
Prints markdown."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
TASKS = [0, 2, 4, 9, 19, 29, 49]

def load(run_dir):
    s = json.load(open(run_dir / "summary.json"))
    rows = [json.loads(l) for l in open(run_dir / "tasks.jsonl") if l.strip()]
    return s, rows

def last_with(rows, key):
    for r in reversed(rows):
        if r.get(key) is not None:
            return r[key]
    return float("nan")

def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"

def crit(s):
    return s["retention_ratio"] >= 0.99 and s["slope_per_100"] >= -0.002 and s.get("fresh_gap_final", 9) <= 0.01

def main():
    keys = ["auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "fresh_gap_final"]
    models = [("l2_init", "rlmnist_ctrl_main"), ("tri", "rlmnist_main"), ("weight_clipping", "rlmnist_main"),
              ("tanh", "rlmnist_main"), ("sin", "rlmnist_main"), ("baseline", "rlmnist_main")]
    data = {}
    print("## (a) Evaluation streams, ten paired seeds (mean ± sd)\n")
    print("| model | n | AUC | early window | final window | retention | fresh gap | dead L0/L1/L2 | |w| mean | criterion |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for lab, d in models:
        runs = sorted((ROOT / d / lab).glob("seed*"), key=lambda p: int(p.name[4:]))
        runs = [r for r in runs if (r / "summary.json").exists()]
        S, R = zip(*[load(r) for r in runs])
        seeds = [int(r.name[4:]) for r in runs]
        vals = {k: np.array([s.get(k, np.nan) for s in S]) for k in keys}
        acc = np.array([[r["task_accuracy"] for r in rows] for rows in R])
        dead = [np.mean([last_with(rows, f"dead_frac/L{i}") for rows in R]) for i in range(3)]
        w = np.mean([s["final_w_abs_mean/all"] for s in S])
        data[lab] = dict(vals, seeds=seeds, acc=acc)
        print(f"| {lab} | {len(S)} | " + " | ".join(ms(vals[k]) for k in keys)
              + f" | {dead[0]:.2f}/{dead[1]:.2f}/{dead[2]:.2f} | {w:.3f} | {sum(crit(s) for s in S)}/{len(S)} |")
    print("\nMean task accuracy at tasks " + ", ".join(str(t + 1) for t in TASKS) + ":")
    for lab, _ in models:
        a = data[lab]["acc"].mean(0)
        print(f"- {lab}: " + " → ".join(f"{a[t]:.2f}" for t in TASKS)
              + f" | min over seeds of final-window mean: {data[lab]['final_window_mean'].min():.3f}")
    print("\nPaired differences l2_init − model (same seeds), mean [95% bootstrap interval]:")
    for lab, _ in models[1:]:
        common = [s for s in data["l2_init"]["seeds"] if s in data[lab]["seeds"]]
        for k in ["auc_norm", "final_window_mean"]:
            a = np.array([data["l2_init"][k][data["l2_init"]["seeds"].index(s)] for s in common])
            b = np.array([data[lab][k][data[lab]["seeds"].index(s)] for s in common])
            m, lo, hi = boot(a - b)
            print(f"- l2_init − {lab} | {k} (n={len(common)}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | min per-seed {np.min(a-b):+.3f}")

    print("\n## (b) Development stream, one seed: standardized inputs against [0,1] inputs\n")
    print("| run | AUC | early | final | retention | dead L0/L1/L2 | task accuracy at tasks " + ", ".join(str(t + 1) for t in TASKS) + " |")
    print("|---|---|---|---|---|---|---|")
    for d, pat in [("rlmnist_dev", "baseline"), ("rlmnist_ctrl_dev", "stdin_baseline"), ("rlmnist_dev", "tri"),
                   ("rlmnist_ctrl_dev", "stdin_tri"), ("rlmnist_ctrl_dev", "l2_init")]:
        for sj in sorted((ROOT / d / pat).rglob("summary.json")):
            s, rows = load(sj.parent)
            name = f"{d}/" + str(sj.parent.relative_to(ROOT / d))
            a = [r["task_accuracy"] for r in rows]
            dead = [last_with(rows, f"dead_frac/L{i}") for i in range(3)]
            print(f"| {name} | {s['auc_norm']:.3f} | {s['early_window_mean']:.3f} | {s['final_window_mean']:.3f} | "
                  f"{s['retention_ratio']:.3f} | {dead[0]:.2f}/{dead[1]:.2f}/{dead[2]:.2f} | "
                  + " → ".join(f"{a[t]:.2f}" for t in TASKS if t < len(a)) + " |")
    return 0

if __name__ == "__main__":
    sys.exit(main())
