#!/usr/bin/env bash
# Task 15, a later round: the scorecard on text 2 as it is now, and the small-talk replies of its
# first AIs and Small for the blind reading. Run from ml/ with Git Bash:
#   bash ../docs/handoff/next-level/text_round.sh ROUND > runs/text-round-ROUND.log 2>&1
# It waits for a running growth lab first.
set -u
R="$1"
LINE="You are Toby, an AI made by Sunrise Studio."  # the line of round 1
until [ ! -f runs/growth-lab-long.log ] || grep -q "^exit " runs/growth-lab-long.log; do sleep 30; done
echo "== scorecard, text 2, round $R $(date +%T)"
AIRACE_DATA=../data2 python -m uv run airace-ml scorecard --out "runs/scorecard-e3-text2-r$R" \
  --yardstick-root ../data > "runs/scorecard-e3-text2-r$R.log" 2>&1
echo "exit $?"
T2="runs/scorecard-e3-text2-r$R/runs"
for lines in standard held; do
  echo "== replies to the $lines lines $(date +%T)"
  python -m uv run python ../docs/handoff/next-level/held_talk.py --lines "$lines" --system "$LINE" \
    --out "runs/talk-t2r$R-$lines.json" t2s1=$T2/first-s1 t2s2=$T2/first-s2 t2small=$T2/small
done
echo "== done $(date +%T)"
