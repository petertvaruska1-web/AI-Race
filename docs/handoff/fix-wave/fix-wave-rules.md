# Fix wave: implementer rules (read fully before starting)

You are one implementer in the **final-review fix wave** of AI Race Milestone 1, a strategy game in which each player trains a real miniature language model. Its Python ML core lives in `ml/`.
- Your **group dispatch notes** (path given in your prompt) list the confirmed findings you must fix, each with the controller's ruling. The ruling is binding: implement what it says.
- Each finding's full text (scenario, evidence, suggested fix) is in `docs/handoff/final-review/verify/<group>.json`. The verifier's verdict, including its fix recommendation, is in `docs/handoff/final-review/verify/verify-results.json` under `verify3:<group>`.
- Where a ruling and a recommendation differ, the ruling wins.

## Environment
- **Repo:** `/home/user/AI-Race`, branch `claude/upbeat-franklin-k1192h`, already checked out.
  - Never switch branches, never push (the controller pushes), never rewrite history, never force anything.
- **Linux, CPU only.** Run everything from `ml/` with `uv run --no-sync ...`, e.g. `cd ml && uv run --no-sync pytest tests/test_x.py -v`.
  - Never run `uv sync`, `uv add` or `uv lock`. Never edit `pyproject.toml` or `uv.lock` unless your notes say so.
- All file I/O uses explicit `encoding="utf-8"`.
- The default suite must stay CPU-only and offline and well under 3 minutes. Mark heavy tests `@pytest.mark.slow` and GPU tests `@pytest.mark.gpu`.
- **Bindings:**
  - `CLAUDE.md` (project rules R1–R8; deterministic code; no scripted model output)
  - `.superpowers/sdd/2026-10-06-m1-it-learns/global-constraints.md`
- Background only, if needed: the spec `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md` and ADR `docs/decisions/0001-bench-v1-measurement-rules.md`.
  - Never read the whole plan file.
- Probe scripts and scratch files go under `/tmp/claude-0/-home-user-AI-Race/475d8ca9-df1d-54e7-9466-322920536cbb/scratchpad/fixwave/<group>/`, never in the repo.

## If an earlier attempt of your group was interrupted
Start with `git status` and `git log --oneline -20`. A usage limit may have stopped an earlier attempt of this same group, leaving commits whose subject starts with `fix(<group letter>)` or uncommitted changes. In that case:
- audit what is there
- keep what is correct
- finish the rest

Do not redo finished findings.

## How to work, per finding
1. **Reproduce first.** Write a test that fails because of the defect, and run it to see it fail (RED).
   - For a pure docs or config finding, a check that shows the old text is enough.
   - If you **cannot reproduce** a finding, do not change code for it. Record `NOT_REPRODUCED` with the exact evidence (command and output) and move on.
2. **Fix it minimally,** as the ruling says. Run the focused test to see it pass (GREEN).
3. **Commit:** one commit per finding, or per small cluster named in your notes.
   - Subject: `fix(<group letter>): <Fxx> <short summary>`, e.g. `fix(A): F1 bench function prompts end before the body's newline`.
   - Stage only the files you changed. Never `git add -A`; nothing from the scratchpad, `data/` or `runs/` may enter git.
   - The message must end with exactly these two lines:
     ```
     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     Claude-Session: https://claude.ai/code/session_0151XfjeP3Lq29ohwpUneXb2
     ```
4. **Never weaken an existing assertion** to make a test pass. If an existing test encodes the old (defective) behaviour that the ruling changes, update it to the ruled behaviour, and say so in your report.

## Scope limits
- Do **not** edit `docs/decisions/*`, the spec, the plan, or anything under `docs/handoff/` except your report. The controller writes ADR text and handoff docs from your report.
- Edit `CLAUDE.md` only where your notes explicitly say to.
- Do not fix things outside your notes. Put anything else you notice under "Other observations" in the report.
- Do not dispatch subagents.

## Before you report
1. Run the full default suite: `cd ml && uv run --no-sync pytest -q`. Record passed, skipped and seconds. The output must be clean, with no new warnings.
2. Run `cd ml && uv run --no-sync ruff check .`; it must be clean.
3. Run any slow tests your notes name, e.g. `uv run --no-sync pytest -m slow tests/test_gate.py -q`.
4. Re-read your whole diff adversarially: what would make a reviewer reject this? Fix what you find.

## Report
Write the full report to `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-<group letter>-report.md`. It contains:
- **Per finding:** status (FIXED | NOT_REPRODUCED | PARTIAL), the RED command and its failing output, the GREEN command and its passing output, files changed, and commit SHAs.
- **Suite:** the full-suite result (passed/skipped/seconds), the ruff result, and any slow-test results.
- **Self-review findings,** concerns, and other observations.

Then return the structured result your prompt asks for.
