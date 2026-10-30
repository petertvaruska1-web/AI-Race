# Handoff: resuming AI Race development

First written 2026-10-07 by the local Claude Code session (Windows desktop). **Updated 2026-10-14 by the cloud session**, which did Tasks 12–20 (code) and is finishing the final whole-branch review. **Read this whole file first.**

## 1. Who you're working for (persistent memory, not otherwise in the repo)
- The product owner is **not technical**. Claude has full technical authority. **Never ask technical questions or present technical option menus.** Decide, record the rationale (spec, `docs/decisions/` ADR, or the ledger), and proceed.
- Ask only **creative/product** questions, and only when they materially change the product and can't be inferred from the design PDFs. Otherwise make a documented temporary assumption.
- Don't stop between tasks asking whether to continue.
- Creative decisions so far: **C1** shared session clock; **C2** small Notebook dataset. Pending: **D1** visual direction (ask at the start of Milestone 2 with 2–3 mockups), **D2** dataset source naming.
- Project rules: `CLAUDE.md`. Architecture: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`. Plan being executed: `docs/superpowers/plans/2026-10-06-m1-it-learns.md`.

## 2. Where things stand (2026-10-21)
- **Work branch:** **`claude/upbeat-franklin-k1192h`**, the cloud session's branch.
  - It contains all of `m1-it-learns` plus the cloud work, so `m1-it-learns` fast-forwards to it.
  - The local session runs: `git fetch origin && git checkout m1-it-learns && git merge --ff-only origin/claude/upbeat-franklin-k1192h`.
- **Milestone 1 code is complete, reviewed task by task, and has passed the final whole-branch review and its fix wave.**
  - Tasks 1–14 and 16–20 are done, with Task 20 limited to its code steps 1–5.
  - Task completion ranges are in the ledger.
  - **Final review:** 18 finders, a completeness critic and 6 extra finders produced 56 findings; 13 Sonnet skeptics verified them. The report is `docs/handoff/final-review/final-review.md`.
  - **Rulings:** 43 fixes, 10 refutations accepted, F6's refutation overruled (it is among the 43), 2 deferred, 1 already done. All are in the ledger.
  - **Fix wave:** six sequential Sonnet groups A–F (`docs/handoff/fix-wave/`), each with an adversarial review; all 43 fixes landed, and nothing blocking remains.
  - **Controller follow-ups:** `ceee83f` (G7 also counts swapped-letter typos, and its data warning leads the detail) and `a2837b4` (the GPU memory test runs on the real vocabulary with a 4 GiB bound).
  - **New decision records:** `docs/decisions/0002-feasibility-gate-measurement-rules.md`, `docs/decisions/0003-corpus-build-rules.md`, and an addendum to ADR 0001.
  - **Tests:** default suite **1561 passed, 1 skipped** (166–232 s on cloud CPUs, depending on the machine); slow suite (including the quick feasibility gate) 5 passed (280 s); ruff clean.
- **Next: the local GPU session (§5).** It covers Task 15, the evaluator builds and Task 20 steps 6–9. Nothing else is left in the cloud.
- **Model tiers** (the owner's suggestion; a ledger ruling):
  - the top model for rulings and measurement-rule decisions only
  - Sonnet for verification, implementation and review
  - Haiku for copy jobs
- **Execution record:**
  - `docs/handoff/sdd/ledger.md` has every ruling.
  - `docs/handoff/sdd/task-*-report.md` holds per-task reports, including `task-fixwave-<G>-report.md`.
  - The decision records in `docs/decisions/`.

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
**Milestone 1 is closed** (2026-10-22, owner decision C4): merged into `main` and tagged `m1-complete`. The gate stands at 5 of 7 with two accepted gaps (`docs/reports/m1-feasibility.md`).

**Milestone 2 starts on branch `m2-two-friends`.** Its plan is `docs/superpowers/plans/2026-10-22-m2-two-friends.md` (28 tasks).
- Stage 1, Tasks 1–24: the single-player first playable version (C6). Stop and report to the owner after Task 24.
- Stage 2, Tasks 25–28: friends.
- New owner decisions: C5 (the company palette themes the dashboard; resolves D1) and C6 (single player first, multiplayer designed in).
- Executed inline (`superpowers:executing-plans`, no subagents) to save usage, committing and pushing after every task.
- **Done:** Task 1 (scaffold, ADR 0008), Task 2 (protocol), Task 3 (sim foundations), Task 4 (worlds, companies, clock), Task 5 (training economics), Task 6 (projection and properties). The sim package is complete for Stage 1. Task 7 (server scaffold, database, test harness), Task 8 (guest accounts), Task 9 (worlds and the world actor), Task 10 (realtime gateway), Task 11 (blob store), Task 12 (dataset catalog), Task 13 (job broker and worker gateway), Task 14 (training service), Task 15 (chat service). The Stage 1 server is complete. Task 16 (catalog export), Task 17 (worker agent), Task 18 (worker jobs). The server and the ML worker are complete for Stage 1. Task 19 (client scaffold), Task 20 (home and founding), Task 21 (dashboard and AI Lab planning), Task 22 (live training view), Task 23 (playground and benchmarks). All Stage 1 screens exist. Task 24 (end-to-end test, dev runner). `npm test` and `npm run check` pass.
- **Stage 1 is complete** (2026-10-27): `npm run dev` plays the single-player game; `npm run e2e` passes. Stop and report to the owner before Stage 2.
- **The single-player game layer** (owner request, 2026-10-27): design `docs/superpowers/specs/2026-10-27-game-layer-design.md` (decisions C7–C29, four phases).
  - **Phase 1 "A real company" is complete** (2026-10-28; plan `docs/superpowers/plans/2026-10-27-game-layer-phase1.md`, 15 tasks): launch day, plans and prices, a national market of six customer groups, real-reply customer posts, news, reputation and followers, the money ledger, the new look with the orb, and the Company, Money and Market screens. Record: `docs/handoff/sdd/ledger-game-layer-phase1.md`.
  - **Owner feedback after playing Phase 1 (2026-10-28, decision C30):** make the game engaging and immersive, not "looking at graphs": motion, transitions, windows that open (detail pop-ups), things to click, charts that expand, more visuals.
  - **The immersion pass is complete** (2026-10-28; design `docs/superpowers/specs/2026-10-28-immersion-design.md`): windows for every number (customers, groups, products, the AI, money, months, money lines, posts), day and month charts, 120 days of history per company, motion (screen slides, a day bar, floating changes, toasts with bursts, customers streaming into the orb, a crowd of dots per group), expanding training charts. Every new screen follows its rules.
  - **Phase 2 "A growing business" is complete** (2026-10-29; plan `docs/superpowers/plans/2026-10-28-game-layer-phase2.md`, record `docs/handoff/sdd/ledger-game-layer-phase2.md`): sim-3 with 9 regions, ad campaigns, cloud capacity and overload, data centers and the share for training, monthly report cards, rescue by loan or investor; their screens and windows.
  - **Phase 3 "Rivals" is built** (2026-10-30; record `docs/handoff/sdd/ledger-game-layer-phase3.md`): the rival pack builder, rivals as owner-less companies (sim-4, sim-3 worlds upgrade), releases and business habits, league tables, battles with ranking points, trash-talk, the rival and battle services, rival chats, the Rivals and Arena screens. The real rival pack is built (`data/rivals/v1`, about an hour on the RTX 3050) and was played in the browser. Next: Phase 4 on the owner's go.
  - **Phases 3 and 4 were planned** (2026-10-29). The owner asked for the plans without starting work; Phase 3 started on their go (2026-10-30).
    - Phase 3 "Rivals": `docs/superpowers/plans/2026-10-29-game-layer-phase3.md` (16 tasks).
    - Phase 4 "Surprises and goals": `docs/superpowers/plans/2026-10-29-game-layer-phase4.md` (15 tasks).
  - **How to resume them:**
    1. **Wait for the owner's go** to start Phase 3. When it comes, tell them first:
       - Their current Phase 2 world carries over (the sim-3 → sim-4 upgrade in Phase 3 Task 4), and rivals join it once the rival AIs are built.
       - Building the rival AIs needs about two quiet hours on the graphics card (Phase 3 Task 16). Ask when that suits them.
    2. **Execute inline** (`superpowers:executing-plans`), as Phases 1 and 2 were:
       - Open a ledger `docs/handoff/sdd/ledger-game-layer-phase3.md` in the Phase 2 ledger's format: plan line, pre-flight rows, one line per task, rulings.
       - Commit and push after every task.
       - End with one whole-branch self-review (the owner prefers no helper agents this milestone), then report.
    3. **Phase 3 order:** Part A, the ML (battle challenges, `build-rivals`, the battle job), comes first. Parts B–D use quick packs and fixtures, so no long GPU run is needed until Task 16.
    4. **GPU runs:** never run a second `npm run dev` or server on `data/` while the owner's game runs. The save folder is now locked (`data/server/server.lock`); for checks, use a second game with its own data root (CLAUDE.md). Keep a Monitor heartbeat on the long `build-rivals` run.
    5. **After Phase 3:** report it, with the plan's "Decisions made in this plan" list plus the rulings and deferred minors from the ledger, and let the owner play. Start Phase 4 on their go, unless they ask to go straight through. Phase 4 upgrades sim-4 worlds to sim-5, so their world carries over again.
    6. **After Phase 4:** M2 Stage 2 (friends, Tasks 25–28 of `docs/superpowers/plans/2026-10-22-m2-two-friends.md`), on the owner's go.
  - **If a plan step proves wrong while executing:** rule on it against the design (`docs/superpowers/specs/2026-10-27-game-layer-design.md`), record the ruling in the phase ledger, and keep going (CLAUDE.md: decide and proceed; ask the owner only creative questions).

## 5. Local GPU session checklist (Windows 11 + RTX 3050)
`data/` is gitignored and large, so it never travels through git. Commands are in CLAUDE.md "Commands"; every corpus build needs `--extra content`.
0. **Setup and GPU tests first.**
   - Run `cd ml && python -m uv sync --extra content`, then `python -m uv run pytest`, then `python -m uv run pytest -m gpu`.
   - The GPU memory test (`test_a_judge_recipe_step_fits_the_8gb_reference_card`, below 4 GiB on the real vocabulary) and the bf16 parity test (`test_cuda_bf16_path_tracks_the_cpu_path`, run with tolerances) have never run on CUDA. They must pass before the 1-hour judge build.
   - Optionally run `python -m uv run --extra content hf auth login`, because the Hub allows more traffic when logged in.
   - **Time the default suite.** If it takes more than about 170 s, mark the four heaviest audits `@pytest.mark.slow`, which saves about 20 s:
     - `test_code_instr_grammar.py::test_training_replies_are_right_as_well_as_well_formed`
     - `test_code_instr_grammar.py::test_every_body_style_of_every_header_is_right`
     - `test_kb_facts.py::test_no_benchmark_phrasing_contains_or_is_contained_in_a_training_phrasing`
     - `test_kb_facts.py::test_no_option_set_offers_two_confusable_objects_across_300_seeds`
     They then run with `-m slow`. The suite was 166–232 s on cloud CPUs.
1. **Task 15: the real dataset build.** It needs huggingface.co, which is blocked in the cloud.
   - Re-verify every source license. simplewiki is cc-by-sa-3.0, which is outside the plan's permissive list: decide, and record the decision.
   - **Calibrate the simplicity thresholds on real data before the full build (F15).** The current thresholds keep about 97–100% of real web, educational and Wikipedia text.
     - On the first ~2000 rows of fineweb, fineweb_edu, simplewiki and gutenberg, measure per-source simplicity quantiles with the real simple_vocab. (A first tiny or partial build into its own `--out` produces simple_vocab.)
     - Pick a keep-rate: about 0.6 keeps the simpler ~90% of FineWeb, and about 0.7 keeps ~80%. Set `simplicity_min` in `ml/src/airace_content/assemble.py` RECIPES and record the numbers.
     - Alternatively, record that the filter stays deliberately mild.
   - Run `python -m uv run --extra content airace-content build --scale full`, into the default root. A tiny check needs its own `--out`. Progress prints on stderr, and the build waits out dropped connections.
   - The manifest records each source's pinned Hub commit (F29). Copy the verified shas into `SourceSpec.revision`, so later rebuilds read the same data.
   - **Then check:**
     - the summary's warnings: exhausted components, and components below 90% of plan (if the templated generators fact_prose, fact_chat or instructions fall short, give them their own bytes-per-token constant in `BYTES_PER_TOKEN_BY_COMPONENT`)
     - that Gutenberg's children's-book subjects are found, and that passages read as whole paragraphs
     - that no mojibake appears outside the deliberately garbled documents
     - educational text complexity
     - the conversations' Thorough pass rate
     - the quality histograms per dataset
   - Write `docs/data-sources.md`.
   - Run the tok-v1 reply-budget test, skipped until a tokenizer exists: `python -m uv run pytest tests/test_bench.py -k real_tokenizer`. Every answer must fit its `max_new_tokens`; for example, the 12-token budget must hold "Bandar Seri Begawan".
2. **Build the evaluators:** `airace-ml build-novelty-index`, then `airace-ml build-judge` (~1 h on GPU).
   - Both record the data they were built from; `bench` and the gate refuse stale ones, and `build-judge` retrains when the data changed.
   - After any later corpus rebuild, run both again.
3. **Task 20 steps 6–9: the real feasibility gate.**
   - Run `airace-ml gate --out runs/gate-1 [--cpu-speed]`, then read `runs/gate-1/report.md`. The run takes about 30–40 minutes longer than first estimated, because every criterion but G1 now runs 3 seeds (ADR 0002).
   - A quick check needs its own folder (`--out runs/gate-quick --quick`).
   - **Read the G7 detail first.** If it starts with "Check the training data first", the data-level garble ratio is above 0.7, so strengthen the web noise or the cleaning thresholds before trusting G7(a).
   - Check creativity on the real judge: a constant reply should score near 0, and real models clearly higher (ADR 0001 rule 11).
   - Iterate as the plan's step 8 says, recording each iteration in the report's Iterations section, which reruns now keep.
   - A rerun reuses every run and measurement whose data and code are unchanged.
4. Write the dated entry in `docs/progress.md`, then the final whole-milestone wrap-up and the merge to `main`.

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
  - A continuation run must never reuse its parent's sampler stream. The gate's G5 already avoids it; the trainer still seeds the sampler with `cfg.seed` only (F8).
  - The live-training view needs a mini-eval event during training (spec §4.7 "live charts"). M1's trainer emits held-out losses and samples only (F50).
  - `MICRO_BATCH_TOKENS` (8192) can be lowered if the largest legal shapes (about 16 × 640) run out of memory; it changes only summation order (F2).
- **M4 (Notebook, Coaching, discoveries):**
  - Creativity scoring must count a model's own notebook and coaching text as already seen (`NoveltyIndex.novelty(seen=)`).
  - Notebook and coaching entries must be screened against bench-v1 prompts before training once discovery cards can show real prompts (F32).
  - Replay currently reviews base datasets only (F7).
  - The full topic × skill × format tag space for discoveries can be added by item id without re-measuring any score (F49).
