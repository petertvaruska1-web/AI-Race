#!/usr/bin/env bash
# Copy the SDD workspace ledger and task reports into docs/handoff/sdd, commit, and push,
# so the latest progress survives a session cutoff. Safe to run repeatedly.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.superpowers/sdd/2026-10-06-m1-it-learns
mkdir -p docs/handoff/sdd
[ -f "$W/progress.md" ] && cp "$W/progress.md" docs/handoff/sdd/ledger.md
for f in "$W"/task-*-report.md "$W"/*-instructions.md "$W"/global-constraints.md; do
  [ -f "$f" ] && cp "$f" docs/handoff/sdd/
done
git add docs/handoff
if ! git diff --cached --quiet; then
  git commit -q -m "docs(handoff): sync ledger and task reports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
fi
git push -q origin HEAD 2>&1 || echo "sync-handoff: push failed (will retry next sync)"
echo "sync-handoff: done ($(git rev-parse --short HEAD))"
