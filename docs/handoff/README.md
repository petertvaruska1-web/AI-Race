# Handoff: resuming AI Race development

Written 2026-10-07 by the local Claude Code session (Windows desktop) before a usage-limit pause. A cloud Claude Code session continues from here, then the local session resumes after the cloud session. **Read this whole file first.**

## 1. Who you're working for (persistent memory, not otherwise in the repo)
- The product owner is **not technical**. Claude has full technical authority. **Never ask technical questions or present technical option menus.** Decide, record the rationale (spec, `docs/decisions/` ADR, or the ledger), and proceed.
- Ask only **creative/product** questions, and only when they materially change the product and can't be inferred from the design PDFs. Otherwise make a documented temporary assumption.
- Don't stop between tasks asking whether to continue.
- Creative decisions so far: **C1** shared session clock; **C2** small Notebook dataset. Pending: **D1** visual direction (ask at the start of Milestone 2 with 2–3 mockups), **D2** dataset source naming.
- Project rules: `CLAUDE.md`. Architecture: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`. Plan being executed: `docs/superpowers/plans/2026-10-06-m1-it-learns.md`.

## 2. Where things stand
- Branch **`m1-it-learns`**, executing Milestone 1 ("It learns": the real tiny-LM core) with the **superpowers:subagent-driven-development** workflow: fresh implementer subagent per task → task reviewer → fix rounds → scoped re-review.
- **Tasks 1–11: complete and reviewed.** 699 tests pass on CPU, and ruff is clean.
- **Task 12** (code, instruction and grammar generators) is **implemented at `f4eaa83` but not reviewed**. The implementer was stopped mid-refinement and wrote no report. See **`docs/handoff/task-12-dispatch-notes.md`** for the exact state and next step.
- **Tasks 13–20:** not started.
- The full execution record, with every ruling, deferred minor finding and carry-forward note per task, is **`docs/handoff/sdd/ledger.md`**. The task reports (`task-N-report.md`) hold the implementers' details and fix histories.

## 3. How to resume the workflow
1. `git checkout m1-it-learns && git pull`.
2. Recreate the SDD workspace (it is gitignored scratch):
   ```bash
   W=.superpowers/sdd/2026-10-06-m1-it-learns
   mkdir -p "$W"
   cp docs/handoff/sdd/ledger.md "$W/progress.md"
   cp docs/handoff/sdd/*-instructions.md docs/handoff/sdd/global-constraints.md docs/handoff/sdd/task-*-report.md "$W/"
   echo "docs/superpowers/plans/2026-10-06-m1-it-learns.md" > "$W/plan-path"
   ```
3. Invoke the `superpowers:subagent-driven-development` skill and use its scripts:
   - `scripts/task-brief PLAN N` generates a task brief.
   - `scripts/review-package PLAN BASE HEAD` generates a review diff package.
4. The dispatch prompts reference `implementer-instructions.md`, `reviewer-instructions.md`, `rereview-instructions.md` and `global-constraints.md` in the workspace. Keep using them.
   - **Edit the uv note in `implementer-instructions.md`:** on Linux, `uv` may be on PATH. Use `uv` if it works, else `python -m uv`.
5. Model choices used so far:
   - Implementers: sonnet; opus for the trainer.
   - Reviewers: sonnet; opus for security-sensitive or complex diffs.
   - Small re-reviews: haiku or sonnet.

## 4. Immediate next step: finish and review Task 12
Follow `docs/handoff/task-12-dispatch-notes.md`:
1. Dispatch a fresh implementer to audit and finish `f4eaa83` and write `task-12-report.md`.
2. Run the task review (base `f1a092b`).
3. Run fix rounds as needed, then a scoped re-review.
4. Mark Task 12 complete in the ledger and continue with Task 13.

(Task 11 finished on 2026-10-07 after two fix rounds. Its residual shortcut rulings are in the ledger.)

## 5. What to do in the cloud vs. leave for the local GPU machine
The local machine has an RTX 3050 and will hold the generated `data/`, which is gitignored and large, so it never travels through git.
- **Do in the cloud:** all *code* tasks: 11 (round 2), 12, 13, 14, 16, 17 (code and tests only), 18, 19, and Task 20 steps 1–5 (code and tests). These are CPU/offline by design.
- **Leave for the local session** (record them as pending in the ledger):
  - Task 15, the real dataset download and build.
  - Task 17's real judge training (`build-judge`, about 1 h on GPU).
  - Task 20 steps 6–9 (real gate runs on GPU and the report).
  - The final whole-branch review can run in either place once all code tasks are done.

## 6. Sync protocol between sessions
- Commit after every task and fix round (each commit message ends with the `Co-Authored-By` trailer).
- After each completed task, update **`docs/handoff/sdd/ledger.md`** (copy the workspace `progress.md` back over it) and the relevant `task-N-report.md` copies. Add a dated entry to **`docs/progress.md`**.
- Push to `origin m1-it-learns` after each completed task. **Never force-push. Never rewrite history.** Don't merge to `main`; merging is the owner-facing finish step.
- **`tools/sync-handoff.sh`** does the copy, commit and push in one step. Run it after every task and fix round.
- The local Windows checkout also has a `.git/hooks/post-commit` hook that auto-pushes every commit on `m1-it-learns`. Hooks aren't versioned, so recreate one in the cloud if you want the same safety net. That hook is not a substitute for running the sync script: the ledger lives in the gitignored workspace until synced.
- Before stopping, write the exact resume point in the ledger (a "NEXT STEP ON RESUME" line) and in this file's §2/§4, then push.

## 7. Carry-forward notes for upcoming tasks
These are already recorded per task in the ledger, and collected here for convenience.
- **Task 12:** generated MiniPy programs must run within MiniPy's limits. It has a 200k cumulative-element budget, nesting 250, and `max_seconds` with the `"TimeLimit"` error string. Keep programs tiny.
- **Task 14:**
  - Never hold out a dup-cluster's canonical doc. Hold out whole clusters, because `eligible_docs` excludes held-out docs before dedup.
  - Assert `tok.vocab_size == 4096` after training the real tokenizer.
  - `plan_false_facts(300)` consumes all of day_after, planet_order, month_after and shape_sides. That is fine.
  - `ExactItem.answers` may hold several accepted forms.
- **Task 16:**
  - Add a minimum-n (≈10) rule to `tag_breakdown` so tiny tags (e.g. `fam:double`) aren't reported.
  - Exact matching: case-insensitive, punctuation stripped, compared against every entry of `answers`.
  - Size `batch_size` for logits memory (`[B, W, V]`).
  - GPU bf16 scores aren't bit-comparable with CPU scores.
- **Task 19:** CLI must reconfigure stdout/stderr to UTF-8, because untrained models emit U+FFFD.
- **Task 20:**
  - `TorchLM.complete()` reserves no reply room on context-length inputs, so use short prefixes.
  - Extend the `tiny_data_root` fixture with `known_vocab.txt`, `false_facts.json`, and lazily built tiny judge and novelty files.
- **M2 (later):**
  - `TrainRunConfig` JSON rejects integer fields sent as floats.
  - Events can re-emit after a crash-resume, so consumers dedupe by step.
  - The MiniPy parse lock is process-wide.
