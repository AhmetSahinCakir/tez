#!/usr/bin/env bash
# Extra runs requested on 19 November 2026 (two workers while the other chains use the machine):
#   1. pilot for a second benchmark, Random Label MNIST (suites/rlmnist_pilot.yaml);
#   2. all comparison methods of the 200-task suite brought to ten seeds (suites/pmnist_main.yaml);
#   3. input standardization test (suites/pmnist_inputnorm.yaml), after the Adam ten-seed chain.
# Resumable (finished runs are skipped); writes "E CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/e_chain.log
echo "[$(date -u)] starting rlmnist_pilot" | tee -a "$LOG"
nice -n 5 python3 scripts/run_suite.py suites/rlmnist_pilot.yaml --workers 2 --quiet > results/suite_rlmnist_pilot.log 2>&1
echo "[$(date -u)] rlmnist_pilot exit code $?" | tee -a "$LOG"
# never run two suite runners on the same results directory
while ps -eo args | grep -q "^python3 scripts/run_suite.py suites/pmnist_main.yaml --only weight_clipping"; do sleep 60; done
echo "[$(date -u)] starting pmnist_main ten-seed extension" | tee -a "$LOG"
nice -n 5 python3 scripts/run_suite.py suites/pmnist_main.yaml --workers 2 --quiet > results/suite_pmnist_main10.log 2>&1
echo "[$(date -u)] pmnist_main extension exit code $?" | tee -a "$LOG"
until grep -q "ADAM10 CHAIN FINISHED" results/adam10_chain.log results/chain_markers.txt 2>/dev/null; do sleep 300; done
echo "[$(date -u)] starting pmnist_inputnorm" | tee -a "$LOG"
python3 scripts/run_suite.py suites/pmnist_inputnorm.yaml --workers 4 --quiet > results/suite_pmnist_inputnorm.log 2>&1
echo "[$(date -u)] pmnist_inputnorm exit code $?" | tee -a "$LOG"
python3 scripts/analyze.py --results results/pmnist_inputnorm --out reports/pmnist_inputnorm --smooth 5 > /dev/null 2>&1 || true
echo "[$(date -u)] E CHAIN FINISHED" | tee -a "$LOG"
