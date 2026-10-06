#!/usr/bin/env bash
# Session keep-alive wait: blocks until something notable happens to the scale-up chain or MAX_MIN pass.
# Exits (and thereby wakes the agent) when a suite finishes, a run fails, the chain/follow-on finishes, or
# the chain process is gone.  Prints a one-line reason.
cd "$(dirname "$0")/.."
MAX_MIN=${1:-110}
LOG=results/scaleup_chain.log
start_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)
start=$(date +%s)
while true; do
  now=$(date +%s); el=$(( (now - start) / 60 ))
  if [ "$el" -ge "$MAX_MIN" ]; then echo "timeout after ${el} min"; exit 0; fi
  lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  if [ "$lines" -ne "$start_lines" ]; then echo "chain log changed: $(tail -1 "$LOG")"; exit 0; fi
  if grep -q "FOLLOW-ON FINISHED" "$LOG" 2>/dev/null; then echo "follow-on finished"; exit 0; fi
  if ! ps -eo args | grep -v grep | grep -q "run_scaleup_chain.sh\|run_after_scaleup.sh"; then echo "no chain process running"; exit 0; fi
  for f in results/suite_*.log; do
    if grep -q "failed (exit code\|Traceback" "$f" 2>/dev/null; then echo "failure in $f: $(grep -m1 'failed (exit code\|Traceback' "$f")"; exit 0; fi
  done
  sleep 120
done
