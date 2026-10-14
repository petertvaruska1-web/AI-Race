#!/usr/bin/env bash
# Template saved from the 2026-10-13/14 cloud session. Usage: COMMIT_TRAILER=$'Co-Authored-By: ...\nClaude-Session: ...' SYNC_BRANCH=<branch> bash cloud-sync.sh "message"
# Same steps as tools/sync-handoff.sh, plus final-review artefacts; commit trailer and branch come from the environment.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.superpowers/sdd/2026-10-06-m1-it-learns
mkdir -p docs/handoff/sdd
[ -f "$W/progress.md" ] && cp "$W/progress.md" docs/handoff/sdd/ledger.md
for f in "$W"/task-*-report.md "$W"/task-*-brief.md "$W"/*-instructions.md "$W"/global-constraints.md "$W"/deferred-minors.md "$W"/plan-path; do
  [ -f "$f" ] && cp "$f" docs/handoff/sdd/
done
# Final review: report (once synthesized) and every finished review agent's result.
mkdir -p docs/handoff/final-review
[ -f "$W/final-review.md" ] && cp "$W/final-review.md" docs/handoff/final-review/final-review.md
J="${REVIEW_JOURNAL:-}"  # journal.jsonl of the current final-review workflow run, if any
[ -f "$J" ] && python3 docs/handoff/final-review/export-results.py "$J" docs/handoff/final-review/agent-results.json >/dev/null
git add docs/handoff
if ! git diff --cached --quiet -- docs/handoff; then
  git commit -q -m "${1:-docs(handoff): sync ledger and task reports}

${COMMIT_TRAILER:?set COMMIT_TRAILER to the commit trailer lines of this session}" -- docs/handoff
fi
for d in 2 4 8 16; do
  git push -q -u origin "${SYNC_BRANCH:?set SYNC_BRANCH}" 2>&1 && break || sleep $d
done
echo "sync: done ($(git rev-parse --short HEAD))"
