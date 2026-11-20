#!/usr/bin/env bash
# Controls for the second benchmark (20 November 2026): L2 Init as a positive control and a standardized-input
# diagnostic on the development stream, then L2 Init with ten paired seeds.  Starts after the F chain
# (Random Label MNIST main runs), with two workers.  Resumable; writes "G CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/g_chain.log
until grep -q "F CHAIN FINISHED" results/f_chain.log results/chain_markers.txt 2>/dev/null; do sleep 300; done
echo "[$(date -u)] starting rlmnist_ctrl_dev" | tee -a "$LOG"
nice -n 5 python3 scripts/run_suite.py suites/rlmnist_ctrl_dev.yaml --workers 2 --quiet > results/suite_rlmnist_ctrl_dev.log 2>&1
echo "[$(date -u)] rlmnist_ctrl_dev exit code $?" | tee -a "$LOG"
python3 scripts/select_hparams.py --results results/rlmnist_ctrl_dev --out suites/selected_rlmnist_ctrl.yaml --table reports/rlmnist_ctrl_dev_hparams.md >> "$LOG" 2>&1
echo "[$(date -u)] selection exit code $?" | tee -a "$LOG"
if [ -f suites/selected_rlmnist_ctrl.yaml ]; then
  echo "[$(date -u)] starting rlmnist_ctrl_main" | tee -a "$LOG"
  nice -n 5 python3 scripts/run_suite.py suites/rlmnist_ctrl_main.yaml --workers 2 --quiet > results/suite_rlmnist_ctrl_main.log 2>&1
  echo "[$(date -u)] rlmnist_ctrl_main exit code $?" | tee -a "$LOG"
fi
echo "[$(date -u)] G CHAIN FINISHED" | tee -a "$LOG"
