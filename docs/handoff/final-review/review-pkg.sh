#!/usr/bin/env bash
# Review package limited to code paths (ml/ and anything else passed after HEAD), so the
# controller's docs/handoff sync commits stay out of the reviewer's diff.
# Usage: review-pkg.sh BASE HEAD [PATHSPEC...]   (default pathspec: ml)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
base=$1; head=$2; shift 2
paths=("$@"); [ ${#paths[@]} -eq 0 ] && paths=(ml)
git merge-base --is-ancestor "$base" "$head" || { echo "HEAD not descendant of BASE" >&2; exit 3; }
out=".superpowers/sdd/2026-10-06-m1-it-learns/review-$(git rev-parse --short "$base")..$(git rev-parse --short "$head").diff"
{
  echo "# Review package: ${base}..${head} (paths: ${paths[*]})"
  echo
  echo "## Commits"
  git log --oneline "${base}..${head}" -- "${paths[@]}"
  echo
  echo "## Files changed"
  git diff --stat "${base}..${head}" -- "${paths[@]}"
  echo
  echo "## Diff"
  git diff -U10 "${base}..${head}" -- "${paths[@]}"
} > "$out"
echo "wrote $(pwd)/${out}: $(git rev-list --count "${base}..${head}" -- "${paths[@]}") commit(s), $(wc -c < "$out" | tr -d ' ') bytes"
