# Progress log

Newest entries first. Each entry: date, what changed, what's next, open issues.

## 2026-10-06 — Milestone 1 in progress (paused at usage limit)
- Branch `m1-it-learns`. Tasks 1–7 of the M1 plan are implemented, reviewed and fixed: scaffold, tokenizer, transformer, function-preserving growth, inference, corpus/sampler, training config.
- Task 8 (trainer) is implemented at `a0ea8e5`, but its review has **not run yet**. Known fix needed: allow `ctx_len` to shrink (spec §4.3).
- Execution ledger with every ruling: `.superpowers/sdd/2026-10-06-m1-it-learns/progress.md` (gitignored). Resume from the "NEXT STEP ON RESUME" line.
- uv is not on PATH: use `python -m uv`.
- **Next:** Task 8 review → Tasks 9–20 → final branch review → feasibility gate report.

## 2026-10-06 — Project start
- Repository initialized (greenfield; only the two design PDFs existed).
- Environment: Windows 11, Node 24.20, npm 11.19, Python 3.14 (system), git 2.55, RTX 3050 8 GB (CUDA cc 8.6), Ryzen 5 5500, 32 GB RAM. No Docker or Postgres; dev setup must not need them.
- Creative decisions: C1 shared session clock; C2 small Notebook dataset (spec §0).
- Architecture spec written: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`.
