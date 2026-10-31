#!/usr/bin/env python3
"""Equivalence (TOST) and paired-difference analysis of two run labels on the same seeds (advisor review, 4.4).

Usage:
  python3 scripts/equivalence.py results/pmnist_fullstream tri weight_clipping --delta 0.005 0.01
  python3 scripts/equivalence.py results/pmnist_main tri weight_clipping --metrics auc_norm,final_window_mean,retention_ratio
Pairs seeds present for BOTH labels, prints a markdown table per margin: mean diff, 95% bootstrap CI, 90% t-CI
(the TOST interval), TOST p and the verdict.  Equivalence is declared only when p_tost < 0.05, i.e. the 90% CI
lies inside (-delta, +delta).  With n = 3 the test has almost no power; a non-significant result then means
"undecided", not "different".
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plasticity.metrics.stats import paired_difference, tost_paired

def load(root, label, metric):
    out = {}
    for p in sorted(glob.glob(f"{root}/{label}/seed*/summary.json")):
        seed = int(os.path.basename(os.path.dirname(p))[4:])
        out[seed] = json.load(open(p)).get(metric, float("nan"))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--metrics", default="auc_norm,final_window_mean,retention_ratio,fresh_gap_final")
    ap.add_argument("--delta", type=float, nargs="+", default=[0.005, 0.01])
    args = ap.parse_args()
    for delta in args.delta:
        print(f"\n### {args.root}: {args.a} − {args.b}, eşdeğerlik marjı Δ = {delta}\n")
        print("| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |")
        print("|---|---:|---|---|---|---|---|")
        for m in args.metrics.split(","):
            A, B = load(args.root, args.a, m), load(args.root, args.b, m)
            seeds = sorted(set(A) & set(B))
            a = np.array([A[s] for s in seeds]); b = np.array([B[s] for s in seeds])
            pd_ = paired_difference(a, b, n_boot=5000, seed=0); t = tost_paired(a, b, delta=delta)
            verdict = "eşdeğer (Δ içinde)" if t["equivalent"] else ("farklı (GA Δ dışında)" if (t["ci_low"] > delta or t["ci_high"] < -delta) else "karar verilemez")
            print(f"| {m} | {len(seeds)} | {pd_['mean_diff']:+.4f} | [{pd_['ci_low']:+.4f}, {pd_['ci_high']:+.4f}] | "
                  f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | {t['p_tost']:.3f} | {verdict} |")

if __name__ == "__main__":
    main()
