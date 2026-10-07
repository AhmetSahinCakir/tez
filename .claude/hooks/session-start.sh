#!/bin/bash
# SessionStart hook (cloud sessions only): make sure the experiment chain is running after a container
# replacement.  Idempotent; finishes in seconds unless the data cache or torch is missing.
set -uo pipefail
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
if ! python3 -c "import torch, numpy, yaml, matplotlib, scipy, pandas" > /dev/null 2>&1; then
  pip install -q -r requirements.txt > /dev/null 2>&1 || true
fi
if [ ! -f data/processed/mnist.npz ]; then
  bash scripts/download_data.sh > /dev/null 2>&1 || true
fi
bash scripts/ensure_chain.sh || true
exit 0
