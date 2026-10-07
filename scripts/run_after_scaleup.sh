#!/usr/bin/env bash
# Follow-on to scripts/run_scaleup_chain.sh: once the chain has finished, re-run every suite of the chain until
# each is complete (the runner skips finished runs and resumes interrupted ones from checkpoint.pt), so that
# runs skipped by one pass (e.g. "locked" after a container replacement, or added to a suite later such as
# adam_tri) are completed.  Then refresh the analyses.  Resumable: if the chain is already finished it starts
# immediately.
cd "$(dirname "$0")/.."
LOG=results/scaleup_chain.log
until grep -q "SCALEUP CHAIN FINISHED" "$LOG" 2>/dev/null; do sleep 600; done
for pass in 1 2 3; do
  all_ok=1
  for s in pmnist_adam_methods pmnist_fullstream pmnist_widenet; do
    python3 scripts/run_suite.py suites/$s.yaml --workers 4 --quiet > results/suite_${s}_followon${pass}.log 2>&1
    rc=$?
    echo "[$(date -u)] follow-on pass $pass: suite $s exit code $rc" | tee -a "$LOG"
    [ "$rc" -eq 0 ] || all_ok=0
    python3 scripts/analyze.py --results results/$s --out reports/$s --reference baseline --smooth 5 > /dev/null 2>&1 || true
  done
  [ "$all_ok" -eq 1 ] && break
done
echo "[$(date -u)] FOLLOW-ON FINISHED" | tee -a "$LOG"
