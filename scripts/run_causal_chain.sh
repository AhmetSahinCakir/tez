#!/usr/bin/env bash
# Advisor report of 16 November 2026, item 2.1: single-factor reflected-sine runs (label sin_refl_matched in
# suites/pmnist_fullstream.yaml; 800 tasks, 10 paired seeds).  Resumable; writes "CAUSAL CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/causal_chain.log
echo "[$(date -u)] starting sin_refl_matched (single-factor causal test)" | tee -a "$LOG"
python3 scripts/run_suite.py suites/pmnist_fullstream.yaml --workers 4 --quiet --only sin_refl_matched > results/suite_pmnist_causal.log 2>&1
echo "[$(date -u)] sin_refl_matched exit code $?" | tee -a "$LOG"
python3 scripts/analyze.py --results results/pmnist_fullstream --out reports/pmnist_fullstream --smooth 5 > /dev/null 2>&1 || true
echo "[$(date -u)] CAUSAL CHAIN FINISHED" | tee -a "$LOG"
