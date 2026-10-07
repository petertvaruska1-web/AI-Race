# Standard implementer instructions (read fully before starting)

## Project
AI Race is a multiplayer strategy game in which each player trains a **real miniature language model**. This milestone (M1) builds the Python ML core in `ml/`. The repo is at `C:\Users\petko\Desktop\AI Race`, branch `m1-it-learns`, on Windows 11. Use the Bash tool (Git Bash) or PowerShell.

## Environment rules
- uv is NOT on PATH. Run every uv command as `python -m uv ...` from `ml/`, e.g. `cd ml && python -m uv run pytest tests/test_x.py -v`. Wherever your brief says `uv run`, use `python -m uv run`.
- CUDA torch 2.14.1+cu130 is installed in `ml/.venv`. The default test suite must stay CPU-only and offline; mark heavy tests `@pytest.mark.slow` or `@pytest.mark.gpu`.
- All file I/O uses explicit `encoding="utf-8"`.
- The global constraints are in `.superpowers/sdd/2026-10-06-m1-it-learns/global-constraints.md`. Read them; they bind your task.
- Project instructions: `CLAUDE.md`. Architecture spec (for background only, if needed): `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`.
- Never read the whole plan file. Your brief is your requirements.

## Before you begin
If anything in the brief is unclear, ask now (reply NEEDS_CONTEXT with your questions).

## Your job
1. Implement exactly what the brief specifies, following TDD: write the failing test, run it to see it fail, implement, run it to see it pass.
2. While iterating, run the focused test. Run the full suite (`python -m uv run pytest`) and `python -m uv run ruff check .` once before committing. Test output must be pristine (no warnings).
3. Commit with the brief's commit message, ending with a `Co-Authored-By:` trailer line. Stage only the files you created or changed.
4. Self-review your diff: completeness, quality, YAGNI, real-behavior tests.
5. Write the report and reply (format below).

Exact values in the brief (numbers, names, signatures, test assertions) are binding. If brief test code fails `ruff check` only for formatting or import-order reasons, apply `ruff check --fix` / `ruff format` to it and mention that in the report. Never weaken an assertion. If an assertion seems wrong, report it as a concern instead of changing it.

## You do not dispatch subagents
Do all the work yourself. Never spawn a subagent, and never spawn a reviewer. The controller schedules review after you report.

## Code organization
Follow the plan's file structure. Each file has one clear responsibility. If a file grows well beyond the brief's intent, report DONE_WITH_CONCERNS; don't restructure on your own.

## When you're in over your head
Stop and report BLOCKED or NEEDS_CONTEXT with specifics. Bad work is worse than no work.

## Report
Write the full report to the report-file path given in your dispatch. Include:
- what you implemented
- tests and results
- TDD evidence: RED (command + failing output, and why it was expected) and GREEN (command + passing output)
- files changed
- self-review findings
- concerns

Then reply with ONLY (under 15 lines):
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- Commits created (short SHA + subject)
- One-line test summary
- Concerns, if any
- The report file path

## After review findings
If you are resumed with findings: fix them, re-run the covering tests, and append a fix report to the same report file (what changed, the tests run, the command, the output). Then reply with the same short format.
