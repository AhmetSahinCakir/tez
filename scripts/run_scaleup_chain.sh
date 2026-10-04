#!/usr/bin/env bash
# Sequential chain of the follow-up suites (resumable: completed runs are skipped), each followed by analysis.
cd "$(dirname "$0")/.."
LOG=results/scaleup_chain.log
for s in pmnist_adam_methods pmnist_fullstream pmnist_widenet; do
  echo "[$(date -u)] starting suite $s" | tee -a "$LOG"
  python3 scripts/run_suite.py suites/$s.yaml --workers 4 --quiet > results/suite_$s.log 2>&1
  echo "[$(date -u)] suite $s exit code $?" | tee -a "$LOG"
  python3 scripts/analyze.py --results results/$s --out reports/$s --reference baseline --smooth 5 > /dev/null 2>&1 || true
done
echo "[$(date -u)] SCALEUP CHAIN FINISHED" | tee -a "$LOG"
