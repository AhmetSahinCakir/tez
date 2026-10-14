#!/usr/bin/env bash
# Fallback watchdog for a FRESH session/container (see docs/RUNBOOK.md).  The last line printed is the decision:
#   ALL DONE   - every run of the scale-up chain has a summary.json: nothing left to compute
#   NOT NEEDED - another container pushed an autosave less than STALE_MIN (150) minutes ago, i.e. it is alive
#   TAKEOVER   - nobody is alive: environment prepared, synced checkpoints restored, chain started HERE
set -uo pipefail
cd "$(dirname "$0")/.."
BRANCH=claude/tez-projesi-tamamla-m2x560
timeout 120 git fetch -q origin "$BRANCH" 2>/dev/null || true
git checkout -q "$BRANCH" 2>/dev/null || true
git pull -q --ff-only origin "$BRANCH" 2>/dev/null || true
if python3 -c "import yaml" > /dev/null 2>&1 && python3 scripts/suite_status.py --check-done > /dev/null 2>&1; then
  echo "ALL DONE"; exit 0
fi
if other=$(bash scripts/other_container_active.sh); then
  echo "another container is alive: $other"; echo "NOT NEEDED"; exit 0
fi
if ! python3 -c "import torch, numpy, yaml, matplotlib, scipy, pandas" > /dev/null 2>&1; then
  pip install -q -r requirements.txt > /dev/null 2>&1 || echo "pip install failed"
fi
if [ ! -f data/processed/mnist.npz ]; then
  bash scripts/download_data.sh > /dev/null 2>&1 || echo "data download failed"
fi
bash scripts/ensure_chain.sh
python3 scripts/suite_status.py 2>/dev/null || true
echo "TAKEOVER"
