# Progress log

Newest entries first. Each entry: date, what changed, what's next, open issues.

## 2026-10-07 — Milestone 1 in progress; handoff to cloud session
- Branch `m1-it-learns`. **Tasks 1–10 complete and reviewed** (scaffold, tokenizer, transformer, growth, inference, corpus/sampler, training config, trainer, MiniPy sandbox, knowledge base). 631 tests pass.
- Task 11 (reasoning/pattern generators) awaits fix round 2 (two shortcut findings).
- **Resume guide: `docs/handoff/README.md`.** The full execution ledger with every ruling: `docs/handoff/sdd/ledger.md`.
- Cloud session does the code tasks; the local GPU machine does the dataset build, judge training and feasibility gate runs.

## 2026-10-06 — Project start
- Repository initialized (greenfield; only the two design PDFs existed).
- Environment: Windows 11, Node 24.20, npm 11.19, Python 3.14 (system), git 2.55, RTX 3050 8 GB (CUDA cc 8.6), Ryzen 5 5500, 32 GB RAM. No Docker or Postgres; dev setup must not need them.
- Creative decisions: C1 shared session clock; C2 small Notebook dataset (spec §0).
- Architecture spec written: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`.
