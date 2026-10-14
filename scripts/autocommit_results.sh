#!/usr/bin/env bash
# Commit and push run outputs every N seconds so that long suites survive a lost container
# (scripts/run_suite.py resumes from committed summary.json files).  Every SYNC_EVERY-th iteration the live
# checkpoints of the small-network suites are copied to checkpoint.sync.pt (tracked, see
# scripts/sync_checkpoints.sh) so that a *different* container can continue a half-finished run.  Autosave
# commits carry this container's disk id ("[disk xxxxxxxx]"), which scripts/other_container_active.sh uses
# to tell whether another container is still working.  Usage: scripts/autocommit_results.sh [interval_s] [sync_every]
cd "$(dirname "$0")/.."
INTERVAL="${1:-1800}"
SYNC_EVERY="${2:-2}"
DISK_ID=$(bash scripts/disk_id.sh)
i=0
while true; do
  sleep "$INTERVAL"
  i=$((i + 1))
  if [ $((i % SYNC_EVERY)) -eq 0 ]; then bash scripts/sync_checkpoints.sh > /dev/null 2>&1 || true; fi
  git add -A results reports suites 2>/dev/null
  if ! git diff --cached --quiet; then
    if git -c user.name="Ahmet Şahin Çakır" -c user.email="cakirahmetsahin1956@gmail.com" \
         commit -q -m "results: autosave $(date -u +%Y-%m-%dT%H:%MZ) [disk $DISK_ID]"; then
      git push -q origin HEAD 2>/dev/null || echo "[autocommit $(date -u +%FT%TZ)] push failed" >> results/autocommit.log
    fi
  fi
done
