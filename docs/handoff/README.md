# Handoff: resuming AI Race development

First written 2026-10-07 by the local Claude Code session (Windows desktop). **Updated 2026-10-14 by the cloud session**, which did Tasks 12–20 (code) and is finishing the final whole-branch review. **Read this whole file first.**

## 1. Who you're working for (persistent memory, not otherwise in the repo)
- The product owner is **not technical**. Claude has full technical authority. **Never ask technical questions or present technical option menus.** Decide, record the rationale (spec, `docs/decisions/` ADR, or the ledger), and proceed.
- Ask only **creative/product** questions, and only when they materially change the product and can't be inferred from the design PDFs. Otherwise make a documented temporary assumption.
- Don't stop between tasks asking whether to continue.
- Creative decisions so far: **C1** shared session clock; **C2** small Notebook dataset. Pending: **D1** visual direction (ask at the start of Milestone 2 with 2–3 mockups), **D2** dataset source naming.
- Project rules: `CLAUDE.md`. Architecture: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`. Plan being executed: `docs/superpowers/plans/2026-10-06-m1-it-learns.md`.

## 2. Where things stand (2026-10-14)
- **Work branch:** **`claude/upbeat-franklin-k1192h`**, the cloud session's branch.
  - It contains all of `m1-it-learns` plus the cloud work, so `m1-it-learns` fast-forwards to it.
  - The local session runs: `git fetch origin && git checkout m1-it-learns && git merge --ff-only origin/claude/upbeat-franklin-k1192h`.
- **Milestone 1 code is complete and reviewed task by task.** Tasks 1–14 and 16–20 are done, with Task 20 limited to its code steps 1–5.
  - Ledger completion lines: Task 12 `f1a092b..3e705b4`, 13 `16e9dab..bf70e88`, 14 `5aeb149..832dddb`, 16 `e34d2f9..10b1edf`, 17 `2815283..435946c`, 18 `e2e7c7e..1f0ab4f`, 19 `9f3832b..5070a03`, 20 `0243951..93f2737`.
  - Latest code commit: `93f2737`.
  - Default suite: **1247 passed, 1 skipped** (~110 s on CPU); ruff is clean.
  - The slow quick-gate test passes in ~96 s on CPU.
- **In progress: the final whole-branch review.**
  - Usage limits interrupted it repeatedly. 16 of 18 finders are done, and their results are saved in `docs/handoff/final-review/agent-results.json`.
  - **Resume guide: `docs/handoff/final-review/README.md`.** It covers how to finish the review without redoing finished work, then the single fix round, the scoped re-review, and the closing docs.
  - That README also lists the early unverified signals, some serious. For example, the coding-bench function prompts may be unanswerable, and a full build may silently reuse a tiny-build tokenizer.
- **Left for the local GPU machine** (see §5): Task 15, the real judge build, and Task 20 steps 6–9.
- **Execution record:**
  - `docs/handoff/sdd/ledger.md` has every ruling, deferred minor and carry-forward note.
  - `docs/handoff/sdd/deferred-minors.md` lists the deferred minors for the final review.
  - The `task-N-report.md` and `task-N-brief.md` files hold per-task details.
  - `docs/decisions/0001-bench-v1-measurement-rules.md` records the measurement rules beyond the plan.

## 3. How to resume the workflow
1. Check out the work branch: `git fetch origin && git checkout claude/upbeat-franklin-k1192h`. In the local session, use `m1-it-learns` after the fast-forward in §2.
2. Recreate the SDD workspace. It is gitignored scratch.
   ```bash
   W=.superpowers/sdd/2026-10-06-m1-it-learns
   mkdir -p "$W"
   cp docs/handoff/sdd/ledger.md "$W/progress.md"
   cp docs/handoff/sdd/*-instructions.md docs/handoff/sdd/global-constraints.md docs/handoff/sdd/deferred-minors.md \
      docs/handoff/sdd/task-*-report.md docs/handoff/sdd/task-*-brief.md "$W/"
   cp docs/handoff/task-*-dispatch-notes.md "$W/"
   echo "docs/superpowers/plans/2026-10-06-m1-it-learns.md" > "$W/plan-path"
   [ -f docs/handoff/final-review/final-review.md ] && cp docs/handoff/final-review/final-review.md "$W/"
   ```
3. Use the `superpowers:subagent-driven-development` skill and its scripts (`task-brief`, `review-package`).
   - Dispatch prompts reference `implementer-instructions.md`, `reviewer-instructions.md`, `rereview-instructions.md` and `global-constraints.md` in the workspace.
   - Model choices so far:
     - Implementers: sonnet, or opus for complex or measurement-critical work.
     - Reviewers: sonnet or opus; re-reviews ran as workflows with adversarial skeptics.
4. **Commands:**
   - Windows: `cd ml && python -m uv run pytest`.
   - Cloud Linux: `cd ml && uv run --no-sync pytest`. Never `uv sync` there, because the CUDA wheel host is blocked; the venv has CPU torch from PyPI.
5. **Syncing the ledger to git:**
   - Windows: `tools/sync-handoff.sh`.
   - Cloud: the template `docs/handoff/final-review/cloud-sync.sh`. It also copies briefs, deferred minors and final-review artefacts, and takes the commit trailer and branch from the environment.

## 4. Immediate next step
**Finish the final review.** Follow `docs/handoff/final-review/README.md` exactly:
1. Finish the review run: the 2 remaining finders, then the critic, dedup, verification and synthesis.
2. Run one fix round over the confirmed findings, then one scoped re-review.
3. Rule on any residuals in the ledger.
4. Update `docs/progress.md` and this file, then sync and push.

After that, the branch is ready for the local GPU session (§5). Do not merge to `main`; merging is the owner-facing finish step.

## 5. Local GPU session checklist (Windows 11 + RTX 3050)
`data/` is gitignored and large, so it never travels through git.
1. **Task 15: real dataset build.** It needs huggingface.co, which is blocked in the cloud.
   - Re-verify every source license. simplewiki is cc-by-sa-3.0, which is outside the plan's permissive list: decide, and record the decision.
   - Run `airace-content build --scale full`. Check the exact command in CLAUDE.md "Commands"; the final review flags that it may need the `content` extra.
   - Then check:
     - the manifest's `exhausted` component warnings
     - that Gutenberg's children's-book subjects are actually found
     - educational text complexity
     - the conversations' Thorough pass rate (~10% of clean tiny chats failed it)
     - the quality histograms per dataset
   - Write `docs/data-sources.md`.
   - Run the tok-v1 reply-budget test that is skipped without a tokenizer (`tests/test_bench.py`). Every answer must fit its `max_new_tokens`.
2. **After any corpus rebuild,** delete `data/judge/v1` and the novelty index before rebuilding them. They are not yet tied to the corpus they were built from, unless the final-review fix round adds that.
3. **Build the evaluators:**
   - `airace-ml build-novelty-index`
   - `airace-ml build-judge` (~1 h on GPU)
4. **Task 20 steps 6–9: the real feasibility gate.**
   - Run `airace-ml gate --out runs/gate-1 [--cpu-speed]`, then read `runs/gate-1/report.md`.
   - Iterate as the plan's step 8 says, and record the iterations in the report.
   - A rerun reuses every run and measurement whose data and code are unchanged.
5. Write the dated entry in `docs/progress.md`, then the final whole-milestone wrap-up.

## 6. Sync protocol between sessions
- Commit after every task and fix round. Each commit message ends with the session's trailer lines.
- After each task or fix round:
  - Copy the workspace `progress.md` over `docs/handoff/sdd/ledger.md`, plus the reports and briefs.
  - Add a dated entry to `docs/progress.md` at milestones.
  - Push.
- **Never force-push. Never rewrite history.** Don't merge to `main`.
- Before stopping:
  - Write the exact resume point in the ledger (a "NEXT STEP ON RESUME" line) and in this file's §2/§4.
  - Save anything that lives only in a session scratchpad (scripts, review results) under `docs/handoff/`.
  - Push.

## 7. Carry-forward notes
Each task's notes are in the ledger. These still matter after M1's code:
- **Real runs:**
  - `TorchLM.complete()` reserves no reply room on context-length inputs.
  - GPU bf16 scores are not bit-comparable with CPU scores.
  - Gate run reuse covers code changes through source digests.
- **M2 (later):**
  - `TrainRunConfig` JSON rejects integer fields sent as floats.
  - Events can re-emit after a crash-resume, so consumers dedupe by step.
  - The MiniPy parse lock is process-wide.
