# AI Race — project instructions

A multiplayer strategy sandbox where each player runs an AI company and trains a **real miniature language model**.

**Design authority:** `AI_Race_Game_Design_Document_Updated.pdf` and `Fun_Features_to_Add.pdf` (repo root).
**Architecture spec:** `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`. Read it before any significant change.
**Progress log:** `docs/progress.md`. Update it at every meaningful milestone or handoff.
**Start here for what is next:** `docs/handoff/next-steps.md` (the release, then the ultimate game document).
**Cross-session handoff:** `docs/handoff/README.md` (resume guide) and `docs/handoff/sdd/ledger.md` (execution ledger with every ruling). Read both when picking up work.

## Working agreement with the product owner
- The owner is not technical. **Never ask technical questions or present technical option menus.** Decide, record the rationale (spec or `docs/decisions/` ADR), and proceed.
- Ask only creative/product questions, and only when they materially change the product and can't be inferred from the design docs. Otherwise make a documented temporary assumption (spec §0 "Pending creative decision points").
- Don't stop between implementation tasks to ask whether to continue.

## Non-negotiables (spec §1)
- R1 Real model: model output is never scripted, templated or rewritten. Filters may only block.
- R2 Opacity: weights, data, mix, prep, design, training logs, notebook and coaching never reach another player. All state goes to clients through `projectView` only.
- R3 Fair compute: in-game compute sets every run's game time and cost; real work runs on the players' own helper cards (Fair mode pools them equally); never in the browser.
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
- Build corpora + tokenizer: `cd ml && python -m uv run --extra content airace-content build --scale {tiny,full}`. A root that holds no text gets training text 2 (the game's since the release, ADR 0010 §8); `--text 1` builds text 1, the measuring stick's (`data/yardstick`), into its own `--out`. It prints progress on stderr and, if the connection drops, waits and carries on; if it still ends with a network problem, run the same command again. A tiny build (a quick check of the setup) needs its own `--out` (e.g. `--out ../data/tiny`): the tokenizer is frozen as tok-v1, so a full build refuses a tokenizer or shared files that a tiny build left, and a tiny build refuses a root that holds a full one.
- Optional, before the full build (the Hugging Face hub allows more when logged in): `cd ml && python -m uv run --extra content hf auth login`
- ML CLI (`--help` on each): `cd ml && python -m uv run airace-ml train|chat|bench|fingerprint|build-novelty-index|build-judge|gate`
- Feasibility gate: `airace-ml gate --out runs/gate-1 [--cpu-speed]` (needs `build-novelty-index` and `build-judge` first). Rerunning it reuses every run and measurement whose data and code are unchanged, and redoes the rest. A quick check of the setup (`--quick`) needs a folder of its own, e.g. `airace-ml gate --out runs/gate-quick --quick`: a gate folder remembers its setup (quick or full, device, seeds) in `gate_setup.json`, and the gate refuses to run another setup into it (exit 2), because the runs are named alike and would be trained over.
- After any corpus or tokenizer rebuild (`airace-content build`), run `airace-ml build-novelty-index`, then `airace-ml build-judge`, then `airace-ml build-rivals` again: each records the data it was built from, and `bench` and the gate refuse ones built from older data (and a judge built with `--budget-tokens`, which is for quick tests only). `build-judge` trains the judge again when the data changed.
- Chat with a trained model (the reference judge is the best one): `cd ml && python -m uv run airace-ml chat --model ../data/judge/v1` (`/raw`, `/reset`, `/quit`; `--temperature` sets the answer style). It only reads the model, but it shares the GPU and CPU, so don't run it while a gate measures speed (G1).
- Long GPU runs (judge, gate, rivals) on the owner's PC: `airace-ml train|bench|build-novelty-index|build-judge|build-rivals|gate` and a helper at work keep the PC awake themselves (`airace_ml/awake.py`; gate run 2 lost hours to sleep before this). Anything else long (a measuring script) still needs a Monitor with a heartbeat of a few minutes.
- The measuring stick (ADR 0010 §3): `cd ml && python -m uv run airace-ml scorecard --out runs/<folder> [--baseline ../docs/reports/2026-11-scorecard-engine2.json]` measures against `data/yardstick` (text 1, its judge and word index, never changed) by default. Between texts the blind reading decides (`docs/handoff/next-level/blind_read.py`); the judge-based talk rules are guards within one text. The lab scripts' notes on folders: `docs/handoff/next-level/README.md`.
- Gate evidence scripts (each states its own command): `docs/handoff/calibration/` (`g2_diagnosis.py`, `g4_k_probe.py`, `g7_*.py`, `creativity_probe.py`).
- Dataset catalog for the game server (after any data rebuild, and once before the server's first start): `cd ml && python -m uv run airace-ml export-catalog --population runs/gate-3/runs` writes `data/catalog/v1/catalog.json` (gate 3 is the gate on text 2; its models are the personality reference). Without it the server runs but refuses training ("The training data is not installed yet.").
- Rival AIs for the game (once, and after any data rebuild): `cd ml && python -m uv run airace-ml build-rivals` trains, measures and pre-answers 8 rival lineages into `data/rivals/v3/` (pack `rivals-v3`, training text 2, keys `-r3-`; about 75 minutes on the RTX 3050 with the PC otherwise quiet; rerunning reuses finished versions). `data/rivals/v1/` (`balance-v2` worlds) and `data/rivals/v2/` (`balance-v3` worlds, text 1) are earlier packs: the server keeps them loaded whatever data they were built on, for the rivals already seated from them, so never build into them or delete them (`build-rivals` refuses a folder holding another pack). New worlds are seated only from the pack built on the current data; a world of older rules whose own pack is missing draws from the current one. Build the judge first, or rivals can't battle in creativity. `--quick` makes tiny rivals in about a minute for checks. Without a pack the game runs without rivals; `npm run dev` never builds it.
- ML worker for the game server: `cd ml && python -m uv run airace-ml worker [--server ws://localhost:8787/worker] [--token T] [--device cuda|cpu]`. It dials out to the server and runs its train, eval and chat jobs; on the owner's PC it reads the token the local server wrote to `data/server/worker-token`. Stop it with Ctrl+C.
- New worlds use the rules `balance-v4`: `balance-v3` (Large and Huge brains, the Marathon length) with a first reading of conversations 30% and the other seven datasets 10% each (ADR 0010 §5.3). `AIRACE_BALANCE=balance-v3` or `balance-v2` makes new worlds with earlier rules. A world keeps the rules it was made with.
- Training engine: `ml/src/airace_ml/engine.py` and `TRAINING_ENGINE` in `packages/protocol/src/worker.ts` must name the same number (the shared `whello` example ties them); raise both when training changes what a run produces, and friends then run the helper's setup file again.
- TypeScript (repo root, Milestone 2): `npm install`, then `npm test` (Vitest over every workspace) and `npm run check` (Biome plus a type check of every workspace).
- Play the game on this PC: `npm run dev` starts the game server (port 8787), an ML worker on the GPU and the web app; open http://localhost:5173. It writes the dataset catalog first if it is missing or older than the server reads. Ctrl+C stops everything. Game data (accounts, worlds, trained AIs) lives in `data/server/`.
- A second game beside the first (for checks while the owner plays): `AIRACE_DATA=<another data root> AIRACE_SERVER_PORT=8788 AIRACE_WEB_PORT=5175 npm run dev`. The other root needs its own `server/` folder; link `catalog`, `corpus`, `judge`, `novelty` and `tokenizer` from `data/` (directory junctions on Windows).
- Play with friends from this PC (Stage 2, `docs/hosting.md`): `npm run host` builds the web app, packs the helper into `data/helper/ml.zip`, starts the server (serving the web app, listening on the network) and this PC's card, and opens a free Cloudflare quick tunnel when `cloudflared` is installed (it prints the friends' link, new every session). It never downloads anything. Checks beside a game in progress: `AIRACE_DATA=data-check AIRACE_SERVER_PORT=8790 AIRACE_NO_TUNNEL=1 HOST=127.0.0.1 npm run host`.
- A friend's helper: the game's "Connect your computer" → "Get the helper" gives a personal `AI Race Helper setup.cmd` (it installs uv, the helper and `airace-ml fetch-data`, and registers the `airace-helper://` link). By hand: `cd ml && python -m uv run airace-ml fetch-data --server <game link or https address> [--key K] --data-root DIR` (exit 3: the game's data changed), then `airace-ml worker --server <game link> --data-root DIR --device auto`.
- Two-player end-to-end test (slow, about 30 minutes; not part of `npm test`): `npm run e2e:friends`. Two browsers, two CPU helpers with players' own keys, a Fair-off and a Fair-on world: joining by link, training on each player's card, trying a shared AI, a friend battle, the world chat, an opacity check and a server restart.
- End-to-end tests (slow, about 5 minutes; not part of `npm test`): `npm run e2e` runs every browser test: the single-player game (`npx playwright test tests/e2e/solo.spec.ts`, about 3 minutes: it builds a tiny practice dataset in a temp folder and plays a whole game in the Edge that Windows has: train, chat, launch, customers, real-reply posts, money and coaching) and the two-player tests. `E2E_ECHO=1` prints the processes' output.
- (More commands are added here as they come to exist.)

## Conventions
- TDD for logic. Deterministic sim: no `Date.now()` or `Math.random()` in `packages/sim`.
- Dataset sources must be permissively licensed and recorded in the content manifest.
