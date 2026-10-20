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

## State at save time (2026-10-20 08:20 UTC)
- **Find, Critic, Find more and Dedup are done.** Dedup produced **56 findings** (1 Critical, 14 Important, 41 Minor), numbered F1–F56 in the order of `agent-results.json` → `dedup.merged`.
- **Verification v4 runs as a workflow** (`verify/verify-workflow.js`, run `wf_1d6b0acd-0c8`): 13 Sonnet skeptics, one per group file `verify/<group>.json`.
  - All 13 v3 verifiers (Agent tool) had died at the usage limit within minutes, with no verdicts.
  - Each skeptic returns structured verdicts, which the workflow journal caches, and also writes `verdicts-<group>.json`.
  - The sync script exports the journal to `verify/verify-results.json` (label `verify3:<group>` → verdicts).
  - F51 is skipped (fixed on 2026-10-14).
- **Synthesis is done by the controller** from the verdicts. It writes `final-review.md` together with its rulings.

## How to resume
**Same session**: resume the verification workflow after a limit with `Workflow({scriptPath: <session>/workflows/scripts/m1-final-review-verify-wf_1d6b0acd-0c8.js, resumeFromRunId: "wf_1d6b0acd-0c8"})`. Finished groups replay from the cache.

**Fresh session or container**:
0. Run `verify/verify-workflow.js` as a new workflow, after fixing its `SP` path to your scratchpad and copying `verify/brief.md` and the group files there. Skip groups whose verdicts are already in `verify/verify-results.json`. Alternatively, use the Agent-tool route below.
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
