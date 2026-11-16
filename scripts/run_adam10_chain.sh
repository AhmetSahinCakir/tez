#!/usr/bin/env bash
# Adam regime with ten paired seeds (suites/pmnist_adam10.yaml).  Starts only after the single-factor causal
# runs have finished, so that the two never compete for the four cores.  Resumable; writes "ADAM10 CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/adam10_chain.log
until grep -q "CAUSAL CHAIN FINISHED" results/causal_chain.log results/chain_markers.txt 2>/dev/null; do sleep 300; done
echo "[$(date -u)] starting suite pmnist_adam10" | tee -a "$LOG"
python3 scripts/run_suite.py suites/pmnist_adam10.yaml --workers 4 --quiet > results/suite_pmnist_adam10.log 2>&1
echo "[$(date -u)] suite pmnist_adam10 exit code $?" | tee -a "$LOG"
python3 scripts/analyze.py --results results/pmnist_adam10 --out reports/pmnist_adam10 --smooth 5 > /dev/null 2>&1 || true
echo "[$(date -u)] ADAM10 CHAIN FINISHED" | tee -a "$LOG"
