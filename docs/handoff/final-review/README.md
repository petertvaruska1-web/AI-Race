# Final whole-branch review of Milestone 1: state and how to resume

Saved 2026-10-14 by the cloud session (branch `claude/upbeat-franklin-k1192h`). Usage limits interrupted the review repeatedly. Everything a fresh session needs to finish it without redoing finished work is in this folder.

## What the review is
Script: `final-review.js` in this folder. It is the portable copy of the workflow run `wf_c9a5295d-a19`. It reviews the whole branch (`c28acd2..HEAD`, code at `93f2737`) in six phases:
1. **Find**: 18 finders.
   - 5 subsystems × 3 lenses. The subsystems are core, skills, content, evals and gate_cli. The lenses are correctness, robustness and integration.
   - 3 specialists: triage-deferred, tests-health and spec-coverage.
2. **Critic**: a completeness critic proposes up to 6 extra finders.
3. **Find more**: those extra finders run.
4. **Dedup**: duplicate findings are merged.
5. **Verify**:
   - Critical and Important findings each get 3 skeptics (reproduce, intent, impact). A finding is confirmed if at least 2 of 3 uphold it.
   - Minor findings get 1 skeptic.
6. **Synthesize**: writes `.superpowers/sdd/2026-10-06-m1-it-learns/final-review.md`. The sync script copies it here as `final-review.md`.

The script stops and returns `{incomplete: <phase>}` instead of synthesizing from partial data whenever an agent fails, e.g. on a usage limit.

## State at save time (2026-10-14 10:15 UTC)
- **Find: 16 of 18 finders done.** Their full results are in `agent-results.json` (label → result).
- **Still to run:**
  - `find:tests-health` and `find:spec-coverage`
  - then Critic, Find more, Dedup, Verify and Synthesize
- **A relaunch was running when this was saved.** If the same session continues, it may finish on its own. Check `final-review.md` here and the ledger first.

## How to resume
**Same session** (the workflow cache still exists): `Workflow({scriptPath: <session scratchpad>/final-review.js, resumeFromRunId: "wf_c9a5295d-a19"})`. Repeat after each limit reset until it returns a report.

**Fresh session or container** (the cache is gone):
1. Recreate the SDD workspace (docs/handoff/README.md §3). The script reads `progress.md`, `global-constraints.md` and `deferred-minors.md` from it.
2. Read `agent-results.json`, and list its keys except `_failed`.
3. Run `Workflow({scriptPath: "docs/handoff/final-review/final-review.js", args: {cachedLabels: [<those keys>]}})`. Use the absolute path.
   - Cached labels are replayed by a cheap loader agent, which prints the saved JSON.
   - Everything else runs for real.
   - Optional args: `root`, `workspace`, `scratch`, `cacheFile`.
4. After any interruption:
   - Export again with `python3 docs/handoff/final-review/export-results.py <transcripts>/subagents/workflows/<run id>/journal.jsonl docs/handoff/final-review/agent-results.json`.
   - Relaunch with the new label list, or with `resumeFromRunId` if you are still in the same session.

## After the report exists (the rest of the plan)
1. Read `final-review.md`. Decide each confirmed finding: fix, or accept with a ledger `Ruling:` line that says why and what it costs if wrong. Measurement-validity rules (scores, chance levels, gate criteria) also go into `docs/decisions/0001-bench-v1-measurement-rules.md` or a new ADR.
2. Send **one** fix implementer (opus) the confirmed Critical and Important findings, the cheap Minors, and the carried minors:
   - G4–G6 report-table rounding near thresholds
   - the duplicate CPU-time formatting in the non-CUDA G1 detail
3. Run exactly **one** scoped re-review as a workflow with skeptics. Adjudicate any residuals with ledger rulings.
4. Update `docs/progress.md` and `docs/handoff/README.md` with the new commit, test counts and the local GPU checklist. Then sync and push.
5. Final message to the owner: every ruling, each with its cost if wrong, plus what their local session must do next. Do not merge to `main`, and open no PR unless asked.

## Early signal from the 16 finished finders (not yet verified)
These are claims, not confirmed defects. Verification decides.
- **Coding bench:**
  - Function-completion prompts end on a bare newline token that training never follows with an indented line, so all 50 items may be unanswerable. Reported by 4 finders, including one Critical.
  - Knowledge multiple-choice and creativity can be gamed by question-blind or constant output, scoring about 15–19 and 21–52.
  - "Bandar Seri Begawan" may not fit the 8-token answer budget.
  - bench-v1 has no pinned content digest.
- **Content build:**
  - A full build silently reuses a tokenizer left by an earlier tiny build. Reported by 3 finders.
  - The build gives no progress output and no pre-flight check of its sources.
  - CLAUDE.md's build command lacks the `content` extra.
  - `airace-content` doesn't switch its output to UTF-8.
- **Trainer:**
  - Each batch goes through one forward/backward pass with no micro-batching, so the judge recipe may not fit in 8 GB of VRAM.
  - Replaying a parent run that trained on notebook or coaching data fails.
  - A `batch_tokens` value that is not a multiple of `ctx_len` overshoots the token budget.
- **Evals and gate:**
  - The judge and the novelty index are not tied to the corpus and tokenizer they were built from.
  - G2, G5, G6 and G7 are decided on seed 1 alone, though the spec says 3 seeds.
  - G4's 0.05 distance threshold sits below sampling noise.
  - G5's continuation runs repeat the starter's exact token stream.
  - G2 and G7(a) can be gamed by degenerate output.
  - The gate's JSON writes don't tolerate a refused rename on Windows.
