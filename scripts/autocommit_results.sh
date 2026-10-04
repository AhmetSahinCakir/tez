#!/usr/bin/env bash
# Commit and push finished run outputs every N seconds so that long suites survive a lost container
# (scripts/run_suite.py resumes from committed summary.json files).  Usage: scripts/autocommit_results.sh [interval_s]
cd "$(dirname "$0")/.."
INTERVAL="${1:-1800}"
while true; do
  sleep "$INTERVAL"
  git add -A results reports suites 2>/dev/null
  if ! git diff --cached --quiet; then
    git -c user.name="Ahmet Şahin Çakır" -c user.email="cakirahmetsahin1956@gmail.com" commit -q -m "results: autosave $(date -u +%Y-%m-%dT%H:%MZ)" \
      && git push -q origin HEAD 2>/dev/null || echo "[autocommit] push failed at $(date)"
  fi
done
