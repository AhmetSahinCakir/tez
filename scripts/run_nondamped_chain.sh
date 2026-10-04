#!/usr/bin/env bash
# dev search for the non-damped maps -> hyper-parameter selection -> main runs (10 seeds) -> report assets.
cd "$(dirname "$0")/.."
LOG=results/nondamped_chain.log
while pgrep -f "run_suite.py suites/pmnist_dev_nondamped.yaml" > /dev/null; do sleep 60; done
echo "[$(date -u)] dev sweep finished; selecting" | tee -a "$LOG"
python3 scripts/select_hparams.py --results results/pmnist_dev --out suites/selected_pmnist.yaml --table reports/pmnist_dev_hparams.md >> "$LOG" 2>&1
echo "[$(date -u)] starting main runs" | tee -a "$LOG"
python3 scripts/run_suite.py suites/pmnist_main_nondamped.yaml --workers 4 --quiet > results/suite_pmnist_main_nondamped.log 2>&1
echo "[$(date -u)] main runs exit code $?" | tee -a "$LOG"
python3 scripts/make_report.py --results results --out reports >> "$LOG" 2>&1
echo "[$(date -u)] NONDAMPED CHAIN FINISHED" | tee -a "$LOG"
