#!/usr/bin/env bash
# Copies the live checkpoint.pt of every unfinished run of the small-network suites to checkpoint.sync.pt
# (tracked in git, ~0.4 MB each) and removes the copies of finished runs.  checkpoint.pt itself stays
# untracked; scripts/ensure_chain.sh restores checkpoint.pt from checkpoint.sync.pt on a container that has
# no live checkpoint (a fresh clone), so another container can continue a run at most one sync behind.
# The wide-network suite is excluded on purpose (its checkpoints are ~40 MB).
cd "$(dirname "$0")/.."
SUITES="${SYNC_SUITES:-pmnist_fullstream pmnist_adam_methods}"
for s in $SUITES; do
  [ -d "results/$s" ] || continue
  find "results/$s" -name summary.json -printf '%h\n' 2>/dev/null | while read -r d; do rm -f "$d/checkpoint.sync.pt"; done
  find "results/$s" -name checkpoint.pt 2>/dev/null | while read -r f; do
    d=$(dirname "$f")
    [ -f "$d/summary.json" ] && continue
    cp -p "$f" "$d/checkpoint.sync.pt.tmp" && mv -f "$d/checkpoint.sync.pt.tmp" "$d/checkpoint.sync.pt"
  done
done
