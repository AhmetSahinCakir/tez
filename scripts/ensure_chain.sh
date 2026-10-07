#!/usr/bin/env bash
# Idempotent (re)starter for the long-running experiment chain.  Safe to run at any time: starts only the
# pieces that are not running and not finished.  Used by the SessionStart hook so that a replaced container
# resumes the scale-up suites (runs continue from their checkpoint.pt) without waiting for a manual check-in.
cd "$(dirname "$0")/.."
LOG=results/scaleup_chain.log
mkdir -p results
running() { ps -eo args | grep -v grep | grep -q "$1"; }
started=""
if ! grep -q "SCALEUP CHAIN FINISHED" "$LOG" 2>/dev/null && ! running "scripts/run_scaleup_chain.sh"; then
  setsid nohup bash scripts/run_scaleup_chain.sh > /dev/null 2>&1 < /dev/null &
  started="$started run_scaleup_chain"
fi
if ! grep -q "FOLLOW-ON FINISHED" "$LOG" 2>/dev/null && ! running "scripts/run_after_scaleup.sh"; then
  setsid nohup bash scripts/run_after_scaleup.sh > /dev/null 2>&1 < /dev/null &
  started="$started run_after_scaleup"
fi
if ! grep -q "FOLLOW-ON FINISHED" "$LOG" 2>/dev/null && ! running "scripts/autocommit_results.sh"; then
  setsid nohup bash scripts/autocommit_results.sh 1800 > /dev/null 2>&1 < /dev/null &
  started="$started autocommit_results"
fi
echo "[ensure_chain $(date -u +%FT%TZ)] started:${started:- nothing (all running or finished)}" | tee -a results/ensure_chain.log
