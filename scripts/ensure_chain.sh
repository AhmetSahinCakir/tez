#!/usr/bin/env bash
# Idempotent (re)starter for the long-running experiment chain.  Safe to run at any time: starts only the
# pieces that are not running and not finished.  Used by the SessionStart hook so that a replaced container
# resumes the scale-up suites (runs continue from their checkpoint.pt) without waiting for a manual check-in.
# Two extras for a container that is NOT the one that ran the chain so far (fresh clone / take-over):
#   * checkpoints synced through git (checkpoint.sync.pt) are restored as checkpoint.pt where none exists;
#   * nothing is started while ANOTHER container is still pushing autosaves (scripts/other_container_active.sh),
#     so two containers never work on, and push, the same runs.
cd "$(dirname "$0")/.."
LOG=results/scaleup_chain.log
mkdir -p results
running() { ps -eo args | grep -v grep | grep -q "$1"; }
restored=0
for f in $(find results -name checkpoint.sync.pt 2>/dev/null); do
  d=$(dirname "$f")
  if [ ! -f "$d/checkpoint.pt" ] && [ ! -f "$d/summary.json" ]; then cp -p "$f" "$d/checkpoint.pt" && restored=$((restored + 1)); fi
done
if other=$(bash scripts/other_container_active.sh); then
  echo "[ensure_chain $(date -u +%FT%TZ)] skipped: another container is active ($other); restored $restored checkpoint(s)" | tee -a results/ensure_chain.log
  exit 0
fi
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
echo "[ensure_chain $(date -u +%FT%TZ)] started:${started:- nothing (all running or finished)}; restored $restored checkpoint(s)" | tee -a results/ensure_chain.log
