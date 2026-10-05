#!/usr/bin/env bash
# Start the scale-up chain only after the non-damped chain has finished (never run both at once on 4 cores).
cd "$(dirname "$0")/.."
until grep -q "NONDAMPED CHAIN FINISHED" results/nondamped_chain.log 2>/dev/null; do sleep 120; done
exec scripts/run_scaleup_chain.sh
