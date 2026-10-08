# AI Race — project instructions

A multiplayer strategy sandbox where each player runs an AI company and trains a **real miniature language model**.

**Design authority:** `AI_Race_Game_Design_Document_Updated.pdf` and `Fun_Features_to_Add.pdf` (repo root).
**Architecture spec:** `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`. Read it before any significant change.
**Progress log:** `docs/progress.md`. Update it at every meaningful milestone or handoff.
**Cross-session handoff:** `docs/handoff/README.md` (resume guide) and `docs/handoff/sdd/ledger.md` (execution ledger with every ruling). Read both when picking up work.

## Working agreement with the product owner
- The owner is not technical. **Never ask technical questions or present technical option menus.** Decide, record the rationale (spec or `docs/decisions/` ADR), and proceed.
- Ask only creative/product questions, and only when they materially change the product and can't be inferred from the design docs. Otherwise make a documented temporary assumption (spec §0 "Pending creative decision points").
- Don't stop between implementation tasks to ask whether to continue.

## Non-negotiables (spec §1)
- R1 Real model: model output is never scripted, templated or rewritten. Filters may only block.
- R2 Opacity: weights, data, mix, prep, design, training logs, notebook and coaching never reach another player. All state goes to clients through `projectView` only.
- R3 Fair compute: training depends on in-game compute, never player hardware. Training and inference run server-side.
- R4 Server authority: clients send intents only.
- R5 Emergence: no unlock lists, tiers, personality menus or level caps. Scores, traits and discoveries are measurements.
- R6 Continuous lineage: runs continue from the parent checkpoint; growth is function-preserving.
- R7 Plain language: player controls use no ML jargon but map to real parameters.
- R8 Friends-first multiplayer, designed in from the start.

## Layout
- `ml/`: Python (uv). `airace_ml` (model/train/infer/evals/worker), `airace_content` (dataset build).
- `apps/client`, `apps/server`, `packages/protocol`, `packages/sim`: TypeScript (npm workspaces), from Milestone 2.
- `data/`: generated build outputs, gitignored. Never commit corpora or checkpoints.

## Commands
- On the owner's Windows PC, `uv` is not on PATH, so run it as `python -m uv`. Where `uv` is on PATH (e.g. cloud Linux), plain `uv` works.
- ML setup (installs Python 3.13 and CUDA torch): `cd ml && python -m uv sync`
- ML tests: `cd ml && python -m uv run pytest`
- ML lint: `cd ml && python -m uv run ruff check .`
- Cloud Linux (no GPU): `cd ml && uv run --no-sync pytest`. Always `--no-sync` there; never `uv sync`.
- Slow tests (incl. the quick feasibility gate): `cd ml && python -m uv run pytest -m slow`
- Build corpora + tokenizer: `cd ml && python -m uv run airace-content build --scale {tiny,full}`
- ML CLI (`--help` on each): `cd ml && python -m uv run airace-ml train|chat|bench|fingerprint|build-novelty-index|build-judge|gate`
- Feasibility gate: `airace-ml gate --out runs/gate-1 [--cpu-speed] [--quick]` (needs `build-novelty-index` and `build-judge` first). Rerunning reuses finished runs; after a training-code change use a fresh `--out`.
- (More commands are added here as they come to exist.)

## Conventions
- TDD for logic. Deterministic sim: no `Date.now()` or `Math.random()` in `packages/sim`.
- Dataset sources must be permissively licensed and recorded in the content manifest.
