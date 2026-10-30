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
- ML setup (installs Python 3.13, CUDA torch and the `datasets` package the corpus build needs): `cd ml && python -m uv sync --extra content`. Keep `--extra content` on every later `sync`: a plain `sync` removes `datasets` again.
- ML tests: `cd ml && python -m uv run pytest`
- ML lint: `cd ml && python -m uv run ruff check .`
- Cloud Linux (no GPU): `cd ml && uv run --no-sync pytest`. Always `--no-sync` there; never `uv sync`.
- Slow tests (incl. the quick feasibility gate): `cd ml && python -m uv run pytest -m slow`
- GPU tests (need CUDA): `cd ml && python -m uv run pytest -m gpu`
- Build corpora + tokenizer: `cd ml && python -m uv run --extra content airace-content build --scale {tiny,full}`. It prints progress on stderr and, if the connection drops, waits and carries on; if it still ends with a network problem, run the same command again. A tiny build (a quick check of the setup) needs its own `--out` (e.g. `--out ../data/tiny`): the tokenizer is frozen as tok-v1, so a full build refuses a tokenizer or shared files that a tiny build left, and a tiny build refuses a root that holds a full one.
- Optional, before the full build (the Hugging Face hub allows more when logged in): `cd ml && python -m uv run --extra content hf auth login`
- ML CLI (`--help` on each): `cd ml && python -m uv run airace-ml train|chat|bench|fingerprint|build-novelty-index|build-judge|gate`
- Feasibility gate: `airace-ml gate --out runs/gate-1 [--cpu-speed]` (needs `build-novelty-index` and `build-judge` first). Rerunning it reuses every run and measurement whose data and code are unchanged, and redoes the rest. A quick check of the setup (`--quick`) needs a folder of its own, e.g. `airace-ml gate --out runs/gate-quick --quick`: a gate folder remembers its setup (quick or full, device, seeds) in `gate_setup.json`, and the gate refuses to run another setup into it (exit 2), because the runs are named alike and would be trained over.
- After any corpus or tokenizer rebuild (`airace-content build`), run `airace-ml build-novelty-index`, then `airace-ml build-judge`, then `airace-ml build-rivals` again: each records the data it was built from, and `bench` and the gate refuse ones built from older data (and a judge built with `--budget-tokens`, which is for quick tests only). `build-judge` trains the judge again when the data changed.
- Chat with a trained model (the reference judge is the best one): `cd ml && python -m uv run airace-ml chat --model ../data/judge/v1` (`/raw`, `/reset`, `/quit`; `--temperature` sets the answer style). It only reads the model, but it shares the GPU and CPU, so don't run it while a gate measures speed (G1).
- Long GPU runs (judge, gate) on the owner's PC: the app's keep-awake lets go after about 5 idle minutes and the PC then sleeps mid-run (gate run 2 lost hours this way). Keep a Monitor with a heartbeat of a few minutes on any run that outlasts the turn.
- Gate evidence scripts (each states its own command): `docs/handoff/calibration/` (`g2_diagnosis.py`, `g4_k_probe.py`, `g7_*.py`, `creativity_probe.py`).
- Dataset catalog for the game server (after any data rebuild, and once before the server's first start): `cd ml && python -m uv run airace-ml export-catalog --population runs/gate-1/runs` writes `data/catalog/v1/catalog.json`. Without it the server runs but refuses training ("The training data is not installed yet.").
- Rival AIs for the game (once, and after any data rebuild): `cd ml && python -m uv run airace-ml build-rivals` trains, measures and pre-answers 8 rival lineages into `data/rivals/v1/` (about 75-90 minutes on the RTX 3050; rerunning reuses finished versions; keep a Monitor heartbeat on it). Build the judge first, or rivals can't battle in creativity. `--quick` makes tiny rivals in about a minute for checks. Without a pack the game runs without rivals; `npm run dev` never builds it.
- ML worker for the game server: `cd ml && python -m uv run airace-ml worker [--server ws://localhost:8787/worker] [--token T] [--device cuda|cpu]`. It dials out to the server and runs its train, eval and chat jobs; on the owner's PC it reads the token the local server wrote to `data/server/worker-token`. Stop it with Ctrl+C.
- TypeScript (repo root, Milestone 2): `npm install`, then `npm test` (Vitest over every workspace) and `npm run check` (Biome plus a type check of every workspace).
- Play the game on this PC: `npm run dev` starts the game server (port 8787), an ML worker on the GPU and the web app; open http://localhost:5173. It writes the dataset catalog first if it is missing. Ctrl+C stops everything. Game data (accounts, worlds, trained AIs) lives in `data/server/`.
- A second game beside the first (for checks while the owner plays): `AIRACE_DATA=<another data root> AIRACE_SERVER_PORT=8788 AIRACE_WEB_PORT=5175 npm run dev`. The other root needs its own `server/` folder; link `catalog`, `corpus`, `judge`, `novelty` and `tokenizer` from `data/` (directory junctions on Windows).
- End-to-end test (slow, about a minute; not part of `npm test`): `npm run e2e`. It builds a tiny practice dataset in a temp folder and plays a whole single-player game in the Edge that Windows has: train, chat, launch, customers, real-reply posts and money. `E2E_ECHO=1` prints the processes' output.
- (More commands are added here as they come to exist.)

## Conventions
- TDD for logic. Deterministic sim: no `Date.now()` or `Math.random()` in `packages/sim`.
- Dataset sources must be permissively licensed and recorded in the content manifest.
