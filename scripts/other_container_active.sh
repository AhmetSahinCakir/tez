#!/usr/bin/env bash
# Is ANOTHER container running the experiment chain on this branch?  Autosave commits made by
# scripts/autocommit_results.sh carry the disk id of the container that made them ("[disk xxxxxxxx]").
# Exit 0 (and print "active <id> <age>min") when the newest autosave on the remote is younger than
# STALE_MIN minutes and was made by a different disk; exit 1 otherwise (ours, old, untagged, or no network).
cd "$(dirname "$0")/.."
STALE_MIN="${STALE_MIN:-150}"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
[ -n "$BRANCH" ] && [ "$BRANCH" != "HEAD" ] || exit 1
timeout 90 git fetch -q origin "$BRANCH" 2>/dev/null || exit 1
line=$(git log -1 --format='%ct %s' --grep='^results: autosave' "origin/$BRANCH" 2>/dev/null)
[ -n "$line" ] || exit 1
ts=${line%% *}
id=$(printf '%s' "${line#* }" | sed -n 's/.*\[disk \([0-9a-f]*\)\].*/\1/p')
[ -n "$id" ] || exit 1
[ "$id" != "$(bash scripts/disk_id.sh)" ] || exit 1
age=$(( ($(date +%s) - ts) / 60 ))
[ "$age" -lt "$STALE_MIN" ] || exit 1
echo "active $id ${age}min"
exit 0
