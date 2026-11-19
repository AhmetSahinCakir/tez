#!/usr/bin/env bash
# Second benchmark (19 November 2026): Random Label MNIST.  Starts after the ten-seed extension of the 200-task
# suite (scripts/run_e_chain.sh), with two workers: hyper-parameter search on the development stream, selection
# by AUC, then ten paired seeds on the evaluation streams.  Resumable; writes "F CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/f_chain.log
until grep -q "pmnist_main extension exit code" results/e_chain.log 2>/dev/null || grep -q "E CHAIN FINISHED" results/chain_markers.txt 2>/dev/null; do sleep 120; done
echo "[$(date -u)] starting rlmnist_dev" | tee -a "$LOG"
nice -n 5 python3 scripts/run_suite.py suites/rlmnist_dev.yaml --workers 2 --quiet > results/suite_rlmnist_dev.log 2>&1
echo "[$(date -u)] rlmnist_dev exit code $?" | tee -a "$LOG"
python3 scripts/select_hparams.py --results results/rlmnist_dev --out suites/selected_rlmnist.yaml --table reports/rlmnist_dev_hparams.md >> "$LOG" 2>&1
echo "[$(date -u)] selection exit code $?" | tee -a "$LOG"
if [ -f suites/selected_rlmnist.yaml ]; then
  echo "[$(date -u)] starting rlmnist_main" | tee -a "$LOG"
  nice -n 5 python3 scripts/run_suite.py suites/rlmnist_main.yaml --workers 2 --quiet > results/suite_rlmnist_main.log 2>&1
  echo "[$(date -u)] rlmnist_main exit code $?" | tee -a "$LOG"
  python3 scripts/analyze.py --results results/rlmnist_main --out reports/rlmnist_main --smooth 3 > /dev/null 2>&1 || true
fi
echo "[$(date -u)] F CHAIN FINISHED" | tee -a "$LOG"
