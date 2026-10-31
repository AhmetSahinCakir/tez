#!/usr/bin/env bash
# C package of docs/YAPILACAKLAR.md (advisor review): the 800-task x 60,000 PMNIST stream with 10 paired seeds
# (C1), the LN + weight-decay baseline (C2) and the reflected sine (C3), all in suites/pmnist_fullstream.yaml.
# Resumable (finished runs are skipped, interrupted ones resume from checkpoint.pt); writes "C CHAIN FINISHED".
cd "$(dirname "$0")/.."
LOG=results/c_chain.log
echo "[$(date -u)] starting suite pmnist_fullstream (C package)" | tee -a "$LOG"
python3 scripts/run_suite.py suites/pmnist_fullstream.yaml --workers 4 --quiet > results/suite_pmnist_fullstream_c.log 2>&1
echo "[$(date -u)] suite pmnist_fullstream exit code $?" | tee -a "$LOG"
python3 scripts/analyze.py --results results/pmnist_fullstream --out reports/pmnist_fullstream --smooth 5 > /dev/null 2>&1 || true
echo "[$(date -u)] C CHAIN FINISHED" | tee -a "$LOG"
