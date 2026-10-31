#!/usr/bin/env bash
# B package of docs/YAPILACAKLAR.md (advisor review): cifar_tri then pmnist_tri_gamma, each followed by analysis.
# Resumable (finished runs are skipped, interrupted ones resume from checkpoint.pt); writes "B CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/b_chain.log
for s in cifar_tri pmnist_tri_gamma; do
  echo "[$(date -u)] starting suite $s" | tee -a "$LOG"
  python3 scripts/run_suite.py suites/$s.yaml --workers 4 --quiet > results/suite_$s.log 2>&1
  echo "[$(date -u)] suite $s exit code $?" | tee -a "$LOG"
done
python3 scripts/analyze.py --results results/cifar_tri --out reports/cifar_tri --smooth 5 > /dev/null 2>&1 || true
python3 scripts/analyze.py --results results/pmnist_tri_gamma --out reports/pmnist_tri_gamma --smooth 5 > /dev/null 2>&1 || true
echo "[$(date -u)] B CHAIN FINISHED" | tee -a "$LOG"
