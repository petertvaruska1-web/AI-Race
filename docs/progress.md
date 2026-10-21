# Progress log

Newest entries first. Each entry: date, what changed, what's next, open issues.

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
