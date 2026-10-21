# Fix wave: state and how to resume

**Status (2026-10-21): complete.** All 43 findings of groups A–F are FIXED (commits `40381f0..9d803cd`), plus controller follow-ups `ceee83f` (F13) and `a2837b4` (F2). No blocking review issue remains; the leftover notes have ledger rulings. Nothing below needs running again.

- **Briefs:**
  - `group-A.md` … `group-F.md` hold the binding rulings per finding.
  - `fix-wave-rules.md` holds the implementer rules.
  - The rulings are also in the ledger, as `FINAL REVIEW: Ruling Fxx` lines.
- **Run:** workflow `wf_046f12a6-b98`, launched 2026-10-20 ~08:55 UTC. Its script copy is `fix-wave-workflow.js`. Groups run strictly in sequence, A → F:
  1. a Sonnet implementer
  2. an adversarial Sonnet review
  3. if anything is blocking, one fix round and a narrow re-check
- **Progress lives in git:** every fix is its own commit, with subject `fix(<G>): Fxx …` or `fix(<G>): review — …`. Implementer reports are `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-<G>-report.md`, synced to `docs/handoff/sdd/`.
- **Resume after a usage-limit stop:**
  - Same session: `Workflow({scriptPath: <session>/workflows/scripts/m1-fix-wave-wf_046f12a6-b98.js, resumeFromRunId: "wf_046f12a6-b98"})`. Finished agents replay from cache, and an interrupted implementer continues from the commits and working tree it left (the rules tell it to).
  - Fresh session: run `fix-wave-workflow.js` as a new workflow with `args: {groups: [<groups not yet finished>]}`. A group is finished when its commits exist and its report holds a review section, which `git log --oneline | grep 'fix(<G>)'` shows. Fix the `SP` scratch path in the script first.
- **After all groups:**
  1. The controller rules on residual issues and NOT_REPRODUCED findings.
  2. The controller writes the ADR text: ADR 0001 addenda for F1, F5 and F6, and a new ADR 0002 for the gate rules F9, F10 and F13, plus the G1 single-seed rule.
  3. The controller adds the M2/M4 notes (F8, F32, F49, F50) and updates `docs/progress.md` and `docs/handoff/README.md`.
  4. Sync and push, then the final owner message listing every ruling.
