#!/usr/bin/env bash
# Prints a stable 8-hex id of this container's disk.  Created on first use and kept in results/.disk_id
# (untracked), so it survives a container restart that keeps the disk but differs on a fresh clone.
cd "$(dirname "$0")/.."
mkdir -p results
f=results/.disk_id
if [ ! -s "$f" ]; then
  (cat /proc/sys/kernel/random/uuid 2>/dev/null || date +%s%N) | tr -d '-' | cut -c1-8 > "$f"
fi
cat "$f"
