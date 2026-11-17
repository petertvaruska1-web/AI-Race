#!/usr/bin/env bash
# Task 15, round 1: the scorecard on text 1 and on text 2 (engine 3), then the small-talk replies
# the blind reading compares. Run from ml/ with Git Bash; it waits for a running growth lab first.
#   bash ../docs/handoff/next-level/text_ab.sh > runs/text-ab.log 2>&1
set -u
LINE="You are Toby, an AI made by Sunrise Studio."  # a name and company no training text uses

until [ ! -f runs/growth-lab-2.log ] || grep -q "^exit " runs/growth-lab-2.log; do sleep 30; done
echo "== scorecard, text 1 $(date +%T)"
python -m uv run airace-ml scorecard --out runs/scorecard-e3-text1 > runs/scorecard-e3-text1.log 2>&1
echo "exit $?"
echo "== scorecard, text 2 $(date +%T)"
AIRACE_DATA=../data2 python -m uv run airace-ml scorecard --out runs/scorecard-e3-text2 \
  --yardstick-root ../data > runs/scorecard-e3-text2.log 2>&1
echo "exit $?"
T1=runs/scorecard-e3-text1/runs
T2=runs/scorecard-e3-text2/runs
for lines in standard held; do
  echo "== replies to the $lines lines $(date +%T)"
  python -m uv run python ../docs/handoff/next-level/held_talk.py --lines "$lines" \
    --out "runs/talk-t1-$lines.json" t1s1=$T1/first-s1 t1s2=$T1/first-s2 t1small=$T1/small
  python -m uv run python ../docs/handoff/next-level/held_talk.py --lines "$lines" --system "$LINE" \
    --out "runs/talk-t2-$lines.json" t2s1=$T2/first-s1 t2s2=$T2/first-s2 t2small=$T2/small
done
echo "== done $(date +%T)"
