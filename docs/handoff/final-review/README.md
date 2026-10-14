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
5. **Verify**: v2, as described under "State at save time".
6. **Synthesize**: the controller writes `.superpowers/sdd/2026-10-06-m1-it-learns/final-review.md` from the journal results. The sync script copies it here as `final-review.md`.

The script stops and returns `{incomplete: <phase>}` instead of synthesizing from partial data whenever an agent fails, e.g. on a usage limit.

## State at save time (2026-10-14 20:30 UTC)
- **Find, Critic, Find more and Dedup are done.** Dedup produced **56 findings**: 1 Critical, 14 Important and 41 Minor. They are numbered F1–F56 in the order of `agent-results.json` → `dedup.merged`.
- **Verification (v3) runs as 13 Sonnet agents** started with the Agent tool. The owner asked to use lighter models where the strongest one isn't needed, and the workflow run was stopped before any verifier finished.
  - The shared brief is `verify/brief.md`.
  - The group files are `verify/<group>.json`: six groups of serious findings (`s-*`) and seven groups of minors (`m-*`).
  - F51 (stale docs) is skipped, because it was fixed on 2026-10-14.
  - Each agent writes `verdicts-<group>.json` (one verdict per finding) to the session scratchpad. The sync script copies these to `verify/`.
- **Synthesis is done by the controller** from the verdicts. It writes `final-review.md` together with its rulings.

## How to resume
**Same session**: wait for the 13 verifier notifications. Re-dispatch any verifier whose verdict file is missing after a usage-limit stop.

**Fresh session or container**:
1. Recreate the SDD workspace (docs/handoff/README.md §3).
2. For every `verify/<group>.json` that has no `verify/verdicts-<group>.json` yet, start one Sonnet agent (Agent tool, `model: "sonnet"`) with this prompt: "You are an adversarial verifier in the final review of the AI Race repository. First read the brief at <abs path>/docs/handoff/final-review/verify/brief.md and follow it exactly. Your group is '<group>'; its findings are in <abs path>/verify/<group>.json. Write your verdicts to <scratch>/verdicts-<group>.json."
   - The brief's scratch paths point at the old session's scratchpad. Replace them with your own scratchpad, e.g. with sed on a copy of the brief.
3. Copy the verdict files into `verify/`, then continue with "After the report exists" below.

(`final-review.js` is kept for reference and for re-running finders. With `args.cachedLabels` it replays saved results through a Haiku loader agent.)

## After the verdicts (the rest of the plan)
Follow **`fix-wave-plan.md`** in this folder.
- It holds the tentative ruling for every finding, six implementer groups A–F in order, the model tiers, and the process: TDD, one re-review, ADRs by the controller, docs, and the final owner message.
- Update a ruling wherever a verdict contradicts it.
- Do not merge to `main`, and open no PR unless asked.

## Early signal from the first 16 finders (superseded by the 56 de-duplicated findings in `verify/*.json`)
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
