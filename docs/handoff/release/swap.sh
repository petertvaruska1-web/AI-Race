#!/usr/bin/env bash
# The release's swap of the data folders (plan C, Task 7): training text 2 (data2/) becomes the
# main game's data (data/), and text 1 becomes the measuring stick (data/yardstick/).
#
#   bash docs/handoff/release/swap.sh          # checks only, moves nothing
#   bash docs/handoff/release/swap.sh --move   # checks, backs up data/server, then moves
#
# Run from the repository root with Git Bash, with no game server, helper or build running.
# Every check runs before anything moves; any surprise stops the script. data/rivals/v1 and v2
# and data/server are never moved or changed (data/server is copied to a dated backup first).
set -euo pipefail

MOVE=0
[ "${1:-}" = "--move" ] && MOVE=1
D=data
T=data2
STAMP=2026-11-24
fail() { echo "STOP: $*" >&2; exit 1; }
say() { echo "-- $*"; }

[ -f CLAUDE.md ] && [ -d "$D" ] && [ -d "$T" ] || fail "run this from the repository root"

say "1. nothing runs on the folders"
for lock in "$D/server/server.lock" "$T/server/server.lock"; do
  if [ -f "$lock" ]; then
    pid=$(cut -d' ' -f1 "$lock")
    if tasklist //FI "PID eq $pid" 2>/dev/null | grep -q " $pid "; then
      fail "a game server (process $pid) holds $lock; stop it first"
    fi
  fi
done
# A worker, build or gate keeps files open.
if powershell -NoProfile -Command "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine" 2>/dev/null \
  | grep -E "airace-ml|airace-content" | grep -v -e grep -e Get-CimInstance | grep -q .; then
  fail "an airace-ml or airace-content process is running; let it finish first"
fi

say "2. what is where"
for p in "$T/corpus/v2/manifest.json" "$T/judge/v1/model.safetensors" "$T/novelty/v1/index.npy" \
  "$T/catalog/v1/catalog.json" "$T/rivals/v3/rivals.json" "$D/corpus/v1/manifest.json" \
  "$D/judge/v1/model.safetensors" "$D/novelty/v1/index.npy" "$D/catalog/v1/catalog.json" \
  "$D/rivals/v1/rivals.json" "$D/rivals/v2/rivals.json" "$D/server/db"; do
  [ -e "$p" ] || fail "missing $p"
done
[ ! -e "$D/yardstick" ] || fail "$D/yardstick already exists: the swap was started before"
[ ! -e "$D/rivals/v3" ] || fail "$D/rivals/v3 already exists"
[ ! -e "$D/corpus/v2" ] || fail "$D/corpus/v2 already exists"
grep -q '"version": "rivals-v3"' "$T/rivals/v3/rivals.json" || fail "$T/rivals/v3 is not pack rivals-v3"
if grep -q '"quick": true' "$T/rivals/v3/rivals.json"; then fail "$T/rivals/v3 is a quick pack"; fi
cmp -s "$D/tokenizer/tok-v1/tokenizer.json" "$T/tokenizer/tok-v1/tokenizer.json" \
  || fail "the two roots hold different tokenizers"

say "3. the identities of both texts"
ids=$(cd ml && python -m uv run python -c "
from pathlib import Path
from airace_ml.data.identity import data_identity
print(data_identity(Path('../$D')), data_identity(Path('../$T')))")
ID1=${ids% *}
ID2=${ids#* }
echo "   text 1 (data/):  $ID1"
echo "   text 2 (data2/): $ID2"
grep -q "$ID2" "$T/catalog/v1/catalog.json" || fail "data2's catalog is not of data2's data"
grep -q "$ID2" "$T/rivals/v3/rivals.json" || fail "data2's rival pack 3 is not of data2's data"

if [ "$MOVE" -eq 0 ]; then
  echo "checks passed; nothing moved (run with --move to swap)"
  exit 0
fi

say "4. back up the owner's worlds to $D/server-backup-$STAMP"
[ ! -e "$D/server-backup-$STAMP" ] || fail "$D/server-backup-$STAMP already exists"
cp -a "$D/server" "$D/server-backup-$STAMP"
a=$(find "$D/server" -type f | wc -l); b=$(find "$D/server-backup-$STAMP" -type f | wc -l)
sa=$(du -sb "$D/server" | cut -f1); sb=$(du -sb "$D/server-backup-$STAMP" | cut -f1)
[ "$a" = "$b" ] && [ "$sa" = "$sb" ] || fail "the backup differs ($a/$b files, $sa/$sb bytes)"
echo "   $a files, $sa bytes"

say "5. text 1 into $D/yardstick"
mkdir "$D/yardstick"
for d in corpus judge novelty catalog; do mv "$D/$d" "$D/yardstick/$d"; done
cp -a "$D/tokenizer" "$D/yardstick/tokenizer"

say "6. text 2 into $D"
for d in corpus judge novelty catalog; do mv "$T/$d" "$D/$d"; done
mv "$T/rivals/v3" "$D/rivals/v3"

say "7. check"
ids=$(cd ml && python -m uv run python -c "
from pathlib import Path
from airace_ml.data.identity import data_identity
print(data_identity(Path('../$D')), data_identity(Path('../$D/yardstick')))")
[ "${ids% *}" = "$ID2" ] || fail "data/ is not text 2 after the move"
[ "${ids#* }" = "$ID1" ] || fail "data/yardstick is not text 1 after the move"
for p in data-check/corpus data-check/catalog data-check/rivals/v3; do
  [ -e "$p" ] || echo "   note: $p does not resolve"
done
echo "swap done: data/ is text 2 ($ID2), data/yardstick/ is text 1 ($ID1)"
