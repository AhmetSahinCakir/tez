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
# results/*.log is git-ignored, so a fresh clone would not see the FINISHED markers and would restart the
# finished chains; results/chain_markers.txt is tracked and carries the same markers.
MARK=results/chain_markers.txt
done_marker() { grep -q "$1" "$2" 2>/dev/null || grep -q "$1" "$MARK" 2>/dev/null; }
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
if ! done_marker "SCALEUP CHAIN FINISHED" "$LOG" && ! running "scripts/run_scaleup_chain.sh"; then
  setsid nohup bash scripts/run_scaleup_chain.sh > /dev/null 2>&1 < /dev/null &
  started="$started run_scaleup_chain"
fi
if ! done_marker "FOLLOW-ON FINISHED" "$LOG" && ! running "scripts/run_after_scaleup.sh"; then
  setsid nohup bash scripts/run_after_scaleup.sh > /dev/null 2>&1 < /dev/null &
  started="$started run_after_scaleup"
fi
CLOG=results/c_chain.log
if ! done_marker "C CHAIN FINISHED" "$CLOG" && ! running "scripts/run_c_chain.sh"; then
  setsid nohup bash scripts/run_c_chain.sh > /dev/null 2>&1 < /dev/null &
  started="$started run_c_chain"
fi
if { ! done_marker "FOLLOW-ON FINISHED" "$LOG" || ! done_marker "C CHAIN FINISHED" "$CLOG"; } && ! running "scripts/autocommit_results.sh"; then
  setsid nohup bash scripts/autocommit_results.sh 1800 > /dev/null 2>&1 < /dev/null &
  started="$started autocommit_results"
fi
echo "[ensure_chain $(date -u +%FT%TZ)] started:${started:- nothing (all running or finished)}; restored $restored checkpoint(s)" | tee -a results/ensure_chain.log
