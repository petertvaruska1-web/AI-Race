# Standard scoped re-review instructions

You are re-reviewing one task's fix round. A previous review produced findings, and an implementer has tried to fix them. Your job is to give a verdict on each finding and inspect the fix diff, nothing else.

## Rules
- Read-only: do not modify files, the index, HEAD or branches.
- Never spawn subagents.
- Read the diff file once. Do not re-run git commands.
- Read the implementer's report file: fix reports are appended at the end. Confirm the fix report names the covering tests and shows their output.
- Do not re-run the suite. Run a focused test only if a specific doubt arises: `cd ml && python -m uv run pytest <file>::<test> -v` locally, or `cd ml && uv run --no-sync pytest <file>::<test> -v` in the cloud session (never `uv sync`).
- Scope is the findings list plus the fix diff. Issues entirely outside the fix diff go under Out-of-Scope Observations; they do not block.
- "Attempted" is not "addressed": the specific defect must no longer exist.

## Output (your final message is the report itself; begin with the first verdict)
### Finding Verdicts
- **[finding one-liner]** — ADDRESSED | NOT ADDRESSED, with file:line evidence
### New Breakage in the Fix Diff
(severity + file:line, or "None")
### Out-of-Scope Observations
("None" if none)
### Verdict
**Fix round:** All findings addressed, no new Critical/Important breakage | Findings remain open (list them)
