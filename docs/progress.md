# Progress log

Newest entries first. Each entry: date, what changed, what's next, open issues.

## 2026-10-22 — Milestone 1 closed; Milestone 2 planned
- **Milestone 1 closed** at 5 of 7 (owner decision C4): merged into `main` and tagged `m1-complete`.
- **Owner decisions:**
  - C5: each company's palette colors its logo and the player's whole dashboard, over a neutral dark base. This resolves D1; the base style is a temporary assumption.
  - C6: the first playable version may be single-player, with multiplayer designed in.
- **Milestone 2 plan:** `docs/superpowers/plans/2026-10-22-m2-two-friends.md`, 28 tasks.
  - Stage 1 (single player, Tasks 1–24):
    - TS monorepo, protocol and deterministic sim
    - server with guest accounts, world actors with input log and snapshots, realtime gateway, job broker and blob store
    - Python catalog export and dial-out worker (train, eval, chat)
    - React client (company founding, AI Lab, live training, playground, benchmarks)
  - Stage 2 (Tasks 25–28): invites, sharing AIs with friends, hosting from the PC, the two-friends end-to-end test.
- **Next:** execute Task 1 on branch `m2-two-friends`.

## 2026-10-22 — Local GPU session: real corpus, judge and feasibility gate (5 of 7 pass)
- **Setup (2026-10-21):** default suite 1559 passed in 153 s on the owner's PC. pytest -m gpu passed 4 tests on the RTX 3050. One Windows-only test was fixed.
- **Task 15, corpus v1** (ADR 0004; `docs/data-sources.md`):
  - 139.4M tokens, 936k documents, tokenizer tok-v1 frozen.
  - All licences re-verified; Simple Wikipedia dropped (CC-BY-SA, share-alike).
  - Simplicity thresholds calibrated on real data; sources pinned at the commits read; the TinyStories mojibake repair extended.
- **Evaluators:** novelty index (16 s) and judge (about 36 min on GPU), rebuilt after each corpus change.
- **Feasibility gate, four runs and three iterations** (ADR 0005–0007; report: `docs/reports/m1-feasibility.md`):
  - Run 1: 4 of 7. Run 2: 3 of 7 (the PC slept mid-run). Run 3: 4 of 7. **Run 4: 5 of 7.**
  - Content fixes:
    - Stories told on request (`tinystories_requests`): creativity 0 → 20.
    - Quiz sheets in Reference Facts (`fact_qa`): knowledge 5 → 77.
    - Denser planted false facts and a stricter quality score.
  - Training fixes: the first model trains on 12.6M tokens, and the default batch is 8,192 tokens.
  - Measurement fixes:
    - G2 pools 180 replies.
    - G4 fingerprints sample 8 replies per probe.
    - G7(a) prompts with clean pages.
    - G7(b) fact arms train on web alone.
  - **Passing:** G1 speed (68 s GPU, 17 min CPU), G3 differentiation, G4 seed variation, G5 growth, G6 forgetting.
- **Owner decision C3 (a babbling start):** the first AI's legibility bar is 60% (was 70%, which no first model reached).
- **Open, for the owner:**
  - **G2:** the first model is 59% legible (107 of 180), one reply short of 60%. Model size, budget, learning rate, batch and corpus mix were all swept; none moves it reliably.
  - **G7(a):** thorough cleaning lowers garbled output only slightly (2.9% → 2.4%), and not on every seed. Small models drop random typos by themselves, and neither more samples nor more training changes that. Fact-checking works (67% → 48%).
- **Spec updated:**
  - §0: C3.
  - §4.11: the first model trains on about 12.6M tokens, with a CPU fallback of at most 20 minutes.
  - §4.12: G2 is judged at 60% on pooled replies.
- **Next:** the owner's decisions on G2 and G7(a), then the milestone wrap-up and the merge to `main`. M2 starts with the D1 visual-direction question.

## 2026-10-21 — Milestone 1 code finished: final review and fix wave complete
- **Final whole-branch review:**
  - 18 finders, a completeness critic, and 6 extra finders that checked real Hugging Face rows.
  - De-duplication into 56 findings, verified by 13 Sonnet skeptics.
  - 43 confirmed findings were fixed; F6's refutation was overruled, and it is among the 43. 10 refutations were accepted, 2 items were deferred and 1 was already done.
  - Report: `docs/handoff/final-review/final-review.md`.
- **Fix wave:** six Sonnet implementer groups (benchmark items, data quality, build operations, trainer, evals, gate and CLI), each with an adversarial Sonnet review. All 43 fixes landed (`40381f0..9d803cd`), plus two controller follow-ups (`ceee83f`, `a2837b4`).
- **Highlights:**
  - The coding benchmark's 50 function items are answerable again.
  - Knowledge answers are even, and creativity multiplies by variety, so models that know nothing score near 0.
  - The judge trains in micro-batches, so it should fit an 8 GB card.
  - The hour-long build retries and resumes, pins its sources, and refuses a tiny build's tokenizer.
  - Real Gutenberg and TinyStories text arrive clean.
  - The gate runs 3 seeds, measures its own noise floor, and counts only garble-like words.
- **Decision records:** ADR 0001 addendum, ADR 0002 (gate measurement rules), ADR 0003 (corpus build rules).
- **Tests:** default suite **1561 passed, 1 skipped** (166–232 s on cloud CPUs, depending on the machine); slow suite 5 passed (280 s); ruff clean.
- **Model tiers** (the owner's suggestion): the top model for rulings only; Sonnet for verification, implementation and review.
- **Next: the local GPU session** (`docs/handoff/README.md` §5):
  - setup and GPU tests
  - Task 15 (license decision, simplicity calibration, full build)
  - build-novelty-index and build-judge
  - the real feasibility gate

## 2026-10-14 — Milestone 1 code complete (Tasks 12–20); final review in progress
- Cloud session, branch `claude/upbeat-franklin-k1192h`. `m1-it-learns` fast-forwards to it.
- **Tasks 12–14 and 16–20 (code and tests) are complete and reviewed.** Each task had its own review and fix rounds (1–3 per task), and every ruling is in `docs/handoff/sdd/ledger.md`.
  - What was built: code, instruction and grammar generators; content text processing; the offline build pipeline; the bench-v1 suite; the novelty index, judge and creativity scoring; the personality fingerprint; the CLI; the feasibility gate code.
  - Latest code commit: `93f2737`. Default suite: **1247 passed, 1 skipped**. Ruff is clean, and the quick feasibility gate passes on CPU.
- Measurement rules beyond the plan are recorded in `docs/decisions/0001-bench-v1-measurement-rules.md`. They keep scores of untrained, constant, looping and parroting models near 0.
- **In progress: the final whole-branch review.** Usage limits interrupted it repeatedly.
  - Finding and de-duplication are complete: 56 findings (1 Critical, 14 Important, 41 Minor).
  - Verification runs on Sonnet agents, after the owner suggested using lighter models where possible.
  - A fix plan with tentative rulings is drafted: `docs/handoff/final-review/fix-wave-plan.md`.
  - Next: decide the findings, then one fix wave of six Sonnet implementers, one re-review, and the handoff.
- **Pending on the local GPU machine:**
  - Task 15 (license check and the real dataset build)
  - `build-novelty-index` and `build-judge`
  - the real feasibility gate runs (Task 20 steps 6–9)
  - The checklist is in `docs/handoff/README.md` §5.

## 2026-10-07 — Milestone 1 in progress; handoff to cloud session
- Branch `m1-it-learns`. **Tasks 1–11 complete and reviewed** (scaffold, tokenizer, transformer, growth, inference, corpus/sampler, training config, trainer, MiniPy sandbox, knowledge base, reasoning/pattern benchmarks). 699 tests pass.
- Task 12 (code/instruction/grammar generators) is implemented at `f4eaa83` but not yet reviewed. Next step: `docs/handoff/task-12-dispatch-notes.md`.
- **Resume guide: `docs/handoff/README.md`.** The full execution ledger with every ruling: `docs/handoff/sdd/ledger.md`.
- Cloud session does the code tasks; the local GPU machine does the dataset build, judge training and feasibility gate runs.

## 2026-10-06 — Project start
- Repository initialized (greenfield; only the two design PDFs existed).
- Environment: Windows 11, Node 24.20, npm 11.19, Python 3.14 (system), git 2.55, RTX 3050 8 GB (CUDA cc 8.6), Ryzen 5 5500, 32 GB RAM. No Docker or Postgres; dev setup must not need them.
- Creative decisions: C1 shared session clock; C2 small Notebook dataset (spec §0).
- Architecture spec written: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`.
