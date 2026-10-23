# ADR 0008: Milestone 2 technical decisions

- **Date:** 2026-10-23
- **Status:** Accepted (engineering decisions; no creative impact)
- **Context:** Milestone 2 builds the first playable version (plan `docs/superpowers/plans/2026-10-22-m2-two-friends.md`). It must run on the owner's Windows PC with no Docker or Postgres install, stay multiplayer-shaped (C6) and reuse Milestone 1's Python code as the ML engine.

## Decisions
1. **Database: PGlite on disk for dev and self-hosting; in-memory PGlite in tests.**
   - Postgres compiled to WebAssembly, installed by `npm install`, so nothing else needs installing. `embedded-postgres` for Postgres 18 is still beta.
   - The schema is plain Postgres via Drizzle, so a hosted Postgres later is a driver swap, not a rewrite.
2. **Training data stays with the worker.**
   - The corpus, tokenizer, judge and novelty index are installed alongside the worker (they are 100s of MB and already built by `airace-content build`).
   - A worker advertises their `dataIdentity`. The server leases jobs only to workers whose identity matches the catalog, so a stale worker never trains on different data than the game describes.
   - Checkpoints travel as one `checkpoint.tar` blob through short-lived signed URLs. Players never receive blob URLs (R2).
3. **The server's dataset facts come from `catalog.json`**, exported by the ML project (plan Task 16). One source of truth for dataset names, sizes and the data identity.
4. **Sharing with friends (`publishVersion`) is a minimal stand-in** for Milestone 3's release flow: a version is either shared with the world or not.
5. **Milestone 2 economics are cloud rental only.** No data centers, hardware purchases, finishing focus or answer-style setting; those are Milestone 3.
6. **Training telemetry is revealed at game pace** (spec §7.3). The worker may finish early; results are held and shown as game days pass, so faster hardware never shows a player more or sooner (R3).
7. **TypeScript 7 for type checking; workspace packages are consumed from source** (`"exports": {".": "./src/index.ts"}`), ESM only. No package build step: the server runs under `tsx`, the client bundles with Vite.
8. **A train job interrupted by a lost worker restarts from its parent**, unless the same worker returns with local resume state. Restarting is simple and correct; the player is never charged twice (the charge belongs to the run, not the job).
9. **The sim's determinism rule is enforced by a test** (`packages/sim/test/determinism-rules.test.ts`): the sim source may not mention the wall clock, ambient randomness, `crypto` or `node:` modules. Server-made IDs travel inside the logged input so replay reproduces them.
10. **Biome checks code only**, not Markdown: its Markdown formatter would rewrite the hand-kept docs.
11. **Two balance numbers in the plan were off by a factor and are corrected** (plan Task 3 updated):
    - `cloudPricePerPflop` is **400,000** dollars, not 400. The plan's stated intent is a first model costing about $42k of the $250k founding cash; at 400 it would cost $42. At 400,000 it costs $41,596 over 13 game days (104 s at 1×).
    - `BALANCE_TEST.cloudPflopsPerDay` is **0.000005**, not 0.0000005. The test model (2×64, 65,536 tokens) then trains in 30 game days (6 s at 1×) instead of 291 (58 s), which keeps the end-to-end tests quick.
12. **A shared AI's public entry carries no brain shape** (plan Task 2 updated). The plan put `shape` (layers, width, attention span) in `PublicVersion`, but brain size is a design choice, which R2 keeps private, and spec §6.4 lists what is public: name, version, release date, public scores and personality words. Others learn an AI's size only by using it.
13. **Run and version ids are unique across the world**, not only within a company. The server makes them as UUIDs; the reducer refuses a reused one, since workers' reports name a run or version without naming its company.
