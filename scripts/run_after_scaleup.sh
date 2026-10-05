#!/usr/bin/env bash
# Follow-on to scripts/run_scaleup_chain.sh: waits until the chain has finished, then re-runs the Adam suite so
# the run added after the chain had started (adam_tri) is completed (the runner skips finished runs), and
# refreshes that suite's analysis.  Resumable: if the chain is already finished it starts immediately.
cd "$(dirname "$0")/.."
LOG=results/scaleup_chain.log
until grep -q "SCALEUP CHAIN FINISHED" "$LOG" 2>/dev/null; do sleep 600; done
echo "[$(date -u)] follow-on: starting suite pmnist_adam_methods (runs added later)" | tee -a "$LOG"
python3 scripts/run_suite.py suites/pmnist_adam_methods.yaml --workers 4 --quiet > results/suite_pmnist_adam_methods_followon.log 2>&1
echo "[$(date -u)] follow-on: suite pmnist_adam_methods exit code $?" | tee -a "$LOG"
python3 scripts/analyze.py --results results/pmnist_adam_methods --out reports/pmnist_adam_methods --reference baseline --smooth 5 > /dev/null 2>&1 || true
echo "[$(date -u)] FOLLOW-ON FINISHED" | tee -a "$LOG"
