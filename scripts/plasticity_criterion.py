#!/usr/bin/env python3
"""Operational criterion for "plasticity retained" (advisor review, item 4.3), applied to finished runs.

A run *retains plasticity* when all three hold on its summary.json:
  (1) retention_ratio = final window / early window >= 1 - TOL_RET         (default TOL_RET = 0.01)
  (2) slope_per_100   = linear trend of online accuracy per 100 tasks >= -TOL_SLOPE   (default 0.002)
  (3) fresh_gap_final = fresh model - continual learner at the last probe <= TOL_FRESH  (default 0.01)
Usage: python3 scripts/plasticity_criterion.py results/pmnist_fullstream results/pmnist_widenet results/pmnist_main
Prints one markdown table per results root (mean of each quantity over seeds and the number of seeds passing).
"""
import argparse, glob, json, os, statistics as st

TOL_RET, TOL_SLOPE, TOL_FRESH = 0.01, 0.002, 0.01

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--labels", default=None, help="comma-separated run labels (default: all)")
    args = ap.parse_args()
    for root in args.roots:
        labels = args.labels.split(",") if args.labels else sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and glob.glob(f"{root}/{d}/seed*/summary.json"))
        print(f"\n### {root}: koruma ölçütü (koruma ≥ {1-TOL_RET:.2f}, eğim/100 ≥ {-TOL_SLOPE:+.3f}, taze model farkı ≤ {TOL_FRESH:+.2f})\n")
        print("| Model | n | Koruma oranı | Eğim / 100 görev | Taze model farkı (son) | Ölçüt (1)(2)(3) geçen tohum | Karar |")
        print("|---|---:|---|---|---|---|---|")
        for lab in labels:
            rows = [json.load(open(p)) for p in sorted(glob.glob(f"{root}/{lab}/seed*/summary.json"))]
            if not rows:
                continue
            ret = [r["retention_ratio"] for r in rows]; slo = [r["slope_per_100"] for r in rows]
            fg = [r.get("fresh_gap_final", float("nan")) for r in rows]
            c1 = sum(x >= 1 - TOL_RET for x in ret); c2 = sum(x >= -TOL_SLOPE for x in slo)
            c3 = sum((x == x) and x <= TOL_FRESH for x in fg)
            allok = sum((a >= 1 - TOL_RET) and (b >= -TOL_SLOPE) and (c == c) and (c <= TOL_FRESH) for a, b, c in zip(ret, slo, fg))
            n = len(rows)
            verdict = "korunmuş" if allok == n else ("kısmen" if allok > 0 else "kaybediyor")
            def ms(v): return f"{st.mean(v):.3f}" + (f" ± {st.pstdev(v):.3f}" if n > 1 else "")
            print(f"| {lab} | {n} | {ms(ret)} | {st.mean(slo):+.4f} | {ms(fg)} | {c1}/{c2}/{c3} ({allok}/{n} tümü) | **{verdict}** |")

if __name__ == "__main__":
    main()
