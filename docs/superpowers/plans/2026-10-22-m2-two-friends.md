# Milestone 2 "Two Friends, Two AIs" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The first playable AI Race. **Stage 1 (Tasks 1–24)** is a single-player version: a player founds a company in their own colors, designs and trains a real first AI while watching it learn, chats with it and sees its scores. **Stage 2 (Tasks 25–28)** adds friends: joining by invite, trying each other's AIs, and hosting from the owner's PC. This meets the spec's M2 exit criteria.

**Architecture:**
- A TypeScript npm-workspaces monorepo:
  - `packages/protocol`: zod schemas, wire messages, view types and JSON patches.
  - `packages/sim`: the pure deterministic reducer, the economy and the per-viewer projection.
  - `apps/server`: Fastify. Guest auth; one single-writer actor per world with an input log and snapshots in PGlite; a WebSocket gateway; a job broker for ML workers; the blob store.
  - `apps/client`: React and Vite.
- The Python `ml/` project gains a catalog exporter, streaming chat and a **worker agent**. The worker dials out to the server, runs training, eval and chat jobs on the existing M1 code, and moves checkpoints through signed blob URLs.
- Everything is multiplayer-shaped from Task 1 (C6): server authority, worlds and memberships, and `projectView` as the only path to the wire. Stage 1 simply has one member per world.

**Tech Stack:**
- Node 24, npm 11 workspaces, TypeScript 7, Biome 2, Vitest 5, fast-check 4
- zod 4, Fastify 5 (+ @fastify/websocket, @fastify/cookie, @fastify/static), pino 10
- Drizzle ORM 0.45 + drizzle-kit 0.31 on PGlite 0.5
- React 19, Vite 8, Zustand 5, Tailwind 4, Radix primitives, Motion 14, uPlot 1.6
- Playwright 1.64
- Python 3.13 (existing uv project), plus `websockets`

**Spec:** `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`. §0 (C1–C6), §1 (R1–R8), §3, §4.5–4.8, §5.1–5.3, §6, §7, §9–§11 and §13–§15 apply. Read it first. Milestone 1's Python code is the ML engine; its plan is `docs/superpowers/plans/2026-10-06-m1-it-learns.md`.

## Global Constraints

**Toolchain**
- Node `>=24`, npm workspaces `apps/*`, `packages/*`. ESM only (`"type": "module"`).
- Workspace packages are consumed from TypeScript source (`"exports": { ".": "./src/index.ts" }`). There is no package build step.
- The server runs with `tsx`; the client builds with Vite.
- Pins (caret ranges): `typescript ^7.0.2`, `zod ^4.6.7`, `vitest ^5.0.5`, `fast-check ^4.10.2`, `@biomejs/biome ^2.6.2`, `tsx ^4.23.15`, `fastify ^5.12.7`, `@fastify/websocket ^11.3.4`, `@fastify/cookie ^11.1.3`, `@fastify/static ^10.1.7`, `ws ^8.22.0`, `pino ^10.4.0`, `drizzle-orm ^0.45.4`, `drizzle-kit ^0.31.11`, `@electric-sql/pglite ^0.5.11`, `react ^19.3.1`, `react-dom ^19.3.1`, `vite ^8.3.5`, `@vitejs/plugin-react ^6.1.2`, `zustand ^5.0.16`, `tailwindcss ^4.3.3`, `@tailwindcss/vite ^4.3.3`, `@radix-ui/react-slider ^1.6.1`, `motion ^14.6.0`, `uplot ^1.6.32`, `@playwright/test ^1.64.1`.
- Client tests use `@testing-library/react` with `happy-dom`. Python adds `websockets` (current release) to `ml/pyproject.toml` main dependencies.

**Commands (root)**
- `npm install`
- `npm test`: Vitest over every workspace
- `npm run check`: `biome check .` plus `tsc --noEmit` in every workspace
- `npm run dev`: server, worker and Vite together
- `npm run e2e`: Playwright, slow
- Python commands are unchanged (CLAUDE.md).

**Determinism (spec §5.2)**
- `packages/sim` never calls `Date`, `Date.now`, `Math.random` or `performance`, and imports nothing from `node:` or the DOM. A test in Task 1 scans the source and fails on any of them.
- All randomness comes from the PRNG stored in `WorldState.rng`.
- Server-assigned IDs (`crypto.randomUUID()`) travel inside the logged input (`Input.newId`), so replay reproduces them.

**Privacy (R2, spec §6.4)**
- `projectView(state, viewerUserId)` is the only path from world state to any client.
- Another company's entry may hold only: `id`, `name`, `logo`, `palette`, `ownerUserId` and `publishedVersions`.
- Never sent to a non-owner: cash, runs, plans, telemetry, unpublished versions, chat transcripts, or training data.
- Never sent to anyone, the owner included: weights, checkpoint keys or blob URLs.

**Units**
- Money is in integer dollars. Game time is an integer `day`; day 0 is 2027-01-01.
- Token budgets are integers. FLOPs are computed as `6 × params × tokens`.

**Text from players and models is untrusted** (spec §11)
- Validated lengths, after trimming and removing control characters (U+0000–U+001F except `\n` in chat, and U+007F): company name 2–32, display name 1–24, world name 2–40, probe prompt 0–120, chat message 1–500.
- The client renders all such text as text: never `dangerouslySetInnerHTML`, and model output goes in `white-space: pre-wrap` elements.

**Player-facing copy (R7)**
- Sentence case and plain language, no ML jargon: say "brain size", not "parameters"; "learning style", not "learning rate".
- Every control shows what it does and, where it costs money or time, its cost.
- The UI never predicts model quality (spec §10).

**Python** keeps M1's rules (CLAUDE.md): TDD, `ruff check` clean, the default pytest suite under 3 minutes, tests offline and CPU-only by default.

**Ports and data**
- The server listens on 8787; Vite dev runs on 5173 and proxies `/api`, `/ws` and `/blobs`.
- Server data: `data/server/db` (PGlite) and `data/server/blobs`. The catalog: `data/catalog/v1/catalog.json`. All gitignored.

**Stages**
- Stage 1 ends at Task 24 with a playable single-player build and green tests. Stop and report there before Stage 2.

## Review Focus

1. **Reloading or closing the tab mid-training.** On return the live view shows everything revealed so far and keeps streaming; nothing restarts and nothing is charged twice. Owner: Tasks 14 and 22.
2. **A command sent twice** (double click, retry after a dropped socket, two tabs). Exactly one effect: one training run and one charge, and the second send is acked as a duplicate. Owner: Task 10.
3. **A server restart while a model trains.** The world resumes at its day from the snapshot plus replay. The run finishes and its version appears exactly once, cash is unchanged by the restart, and the worker re-attaches its job. Owner: Tasks 9, 13 and 14.
4. **Hostile or odd text** in names, probe prompts, chat and model output: HTML, `<|end|>` literals, 10,000 characters, emoji, control characters. It is rejected or trimmed by the rules above and always rendered as text; a `<|end|>` typed by a player is plain text to the tokenizer. Owner: Tasks 4, 15 and 23.
5. **A training request that cannot run.** Not enough cash, a run already active, a smaller brain than the parent, an all-zero mix, or no worker online. The first four are rejected before any job with a plain reason. No worker online leaves the run waiting ("waiting for compute"), and it starts when one connects. Owner: Tasks 5 and 14.

---

# Stage 1: single-player first playable

### Task 1: Monorepo scaffold and M2 technical decisions

**Files:**
- Create:
  - root `package.json`, `biome.json`, `tsconfig.base.json`, `vitest.config.ts` (projects = workspaces)
  - `packages/protocol/{package.json,tsconfig.json,src/index.ts}`, `packages/sim/{package.json,tsconfig.json,src/index.ts}`
  - `packages/sim/test/determinism-rules.test.ts`
  - `docs/decisions/0008-m2-technical-decisions.md`
- Modify: `.gitignore` (`node_modules/`, `dist/`, `playwright-report/`, `test-results/`), `CLAUDE.md` (Commands: the root npm commands above)

**Interfaces:**
- Produces: the workspaces `@airace/protocol` and `@airace/sim` (empty `index.ts` exports), and the root scripts `test`, `check` and `typecheck`.

- [ ] **Step 1: Write the failing test** `packages/sim/test/determinism-rules.test.ts`. It reads every `packages/sim/src/**/*.ts` file and asserts that none matches `/\bDate\b|Math\.random|performance\.|from ['"]node:/`. Add a dummy `src/bad.ts` containing `Date.now()` and see it fail; then delete the dummy.
- [ ] **Step 2: Write ADR 0008** recording these decisions with rationale:
  1. PGlite on disk is the database for dev and self-hosting (`embedded-postgres` for PG 18 is beta only); in-memory PGlite is used for tests. The schema is plain Postgres, so production is a driver swap.
  2. The corpus, tokenizer, judge and novelty index are installed with the worker. A worker advertises their `dataIdentity`, and the server leases only to workers whose identity matches the catalog. Checkpoints travel as one `checkpoint.tar` blob through signed URLs.
  3. The server's dataset facts come from `catalog.json`, exported by the ML project (Task 16).
  4. Stage 2's "Share with friends" (`publishVersion`) is a minimal stand-in for M3's release flow.
  5. M2 economics: cloud rental only (no data centers, purchases, finishing focus or answer-style setting). Those are M3.
  6. Telemetry is revealed at game pace (§7.3).
  7. TypeScript 7 for type checking; packages are consumed from source.
  8. A train job interrupted by a lost worker restarts from its parent, unless the same worker comes back with local resume state.
- [ ] **Step 3:** Run `npm install`, then `npm test` (PASS) and `npm run check` (PASS).
- [ ] **Step 4: Commit** `chore: TS monorepo scaffold; ADR 0008 (M2 technical decisions)`.

---

### Task 2: Protocol schemas and JSON patches

**Files:**
- Create in `packages/protocol/src/`:
  - `text.ts`, `world.ts` (palette, logo, plan, commands), `view.ts`, `wire.ts` (client and server messages), `patch.ts`
  - exports from `index.ts`
- Test: `packages/protocol/test/{schemas,patch}.test.ts`

**Interfaces:**
- Produces:
  - **Text and IDs:**
    - `cleanText(s: string, opts: {allowNewlines?: boolean}): string`: trims and removes control characters
    - `zName(min: number, max: number)`: a zod string that applies `cleanText` and checks the length
  - **Datasets:**
    - `DATASET_IDS = ["web","books","educational","conversations","code","reasoning","facts","creative"] as const`; `DatasetId`
  - **Palette and logo:**
    - `Palette = {primary: Hex, secondary: Hex}`, where `Hex` matches `/^#[0-9a-f]{6}$/i`
    - `LOGO_SHAPES = ["circle","rounded-square","hexagon","shield","diamond"]`
    - `LOGO_SYMBOLS = ["spark","atom","leaf","bolt","star","wave","eye","flame","rocket","brain","cube","moon","sun","anchor","crown","feather"]`
    - `Logo = {shape, symbol: LogoSymbol | null, initials: string}`, where initials match `/^[A-Z0-9]{0,2}$/`
  - **The player-facing training plan:**
    - `TrainingPlan = {mix: Record<DatasetId, number>, cleaning: "light"|"standard"|"thorough", removeDuplicates: boolean, factCheck: boolean, variety: "natural"|"balanced", brainSize: string, attentionSpan: number, length: string, learningStyle: number, review: number, probePrompt: string, parentVersionId: string | null}`
    - mix weights are integers 0–100; `learningStyle` is in [0, 1]; `review` is in [0, 0.5]
  - **Commands (`Command`, a discriminated union on `type`):**
    - `createCompany{name, logo, palette}`, `startWorld{}`, `setSpeed{speed: 1|2|4}`, `setPaused{paused}`, `startTraining{plan}`, `publishVersion{versionId, published}` (sim support lands in Task 26)
  - **`WorldView`:**
    - `world: {id, name, phase: "setup"|"running", day, date: string, speed, paused, hostUserId, effectiveHostUserId, inviteCode}`
    - `members: {userId, displayName, online, companyId: string|null}[]`
    - `companies: PublicCompany[]`, each `{id, name, logo, palette, ownerUserId, publishedVersions: PublicVersion[]}`
    - `me: {userId, company: PrivateCompany | null}`
    - `PrivateCompany = {id, cash, runs: RunView[], versions: VersionView[]}`
    - `RunView = {id, status: "training"|"finalizing"|"completed"|"failed", computeStarted: boolean, plan, preview: TrainingPreview, startDay, endDay, progress: number, failure: string | null, versionId: string | null}`
    - `VersionView = {id, label, runId, parentId, shape: {nLayer, dModel, ctxLen}, createdDay, trainedTokens, scores: Record<string, number|null> | null, overall: number|null, personalityWords: string[], published: boolean}`
    - `PublicVersion = {id, label, createdDay, scores, overall, personalityWords}` (no shape: brain size is private, R2 and spec §6.4; ADR 0008 item 12)
    - `TrainingPreview = {params, tokens, flops, cost, durationDays}`
  - **Client messages (`ClientMessage`; every message has `v: 1`):**
    - `hello{token}`, `watch{worldId}`, `unwatch{worldId}`, `resync{worldId}`
    - `command{cid, worldId, cmd}`, `chat{cid, worldId, threadId: string|null, versionId, text}`
    - `query{id, q}`, `subscribe{sub}`, `unsubscribe{sub}`
    - `cid` is a UUID; `q` is a union defined in later tasks (start with `{kind:"ping"}`)
  - **Server messages (`ServerMessage`):**
    - `welcome{userId}`, `snapshot{worldId, version, view}`, `patch{worldId, from, to, ops}`
    - `ack{cid, duplicate: boolean, data?}`, `reject{cid, reason}`
    - `event{worldId, kind, text}`, `stream{sub, data}`, `queryResult{id, ok, data?, error?}`, `error{reason}`
  - **Patches:**
    - `PatchOp = {op:"set", path: (string|number)[], value} | {op:"del", path}`
    - `diff(a: unknown, b: unknown): PatchOp[]`
    - `applyPatch<T>(doc: T, ops: PatchOp[]): T`, immutable

- [ ] **Step 1: Write the failing tests.**
  - `schemas.test.ts`:
    - `cleanText("  Ac\u0007me‮ ", {})` gives `"Acme‮"`: only C0 and DEL are stripped; the bidi mark stays, and the client renders it as text.
    - `zName(2,32)` rejects `"a"` and a 33-character string, and accepts `"<b>Acme</b>"` as text.
    - A `TrainingPlan` with mix weight `101`, `learningStyle: 1.5` or `review: 0.6` is rejected.
    - `ClientMessage.parse({v:1,type:"command",cid:"not-a-uuid",...})` throws.
    - Every server message type round-trips through `JSON.stringify`/`parse`.
  - `patch.test.ts`, a fast-check property: for arbitrary JSON values `a` and `b`, `applyPatch(a, diff(a,b))` deep-equals `b`, and `diff(a,a)` is `[]`.
  - Unit tests: arrays of different length give one `set` of the whole array; arrays of equal length diff element-wise; a removed key gives `del`.
- [ ] **Step 2:** Run `npx vitest run packages/protocol`: FAIL.
- [ ] **Step 3: Implement the schemas and `diff`/`applyPatch` as specified.**
  - `diff` recurses into plain objects key by key, and into arrays only when the lengths are equal. Otherwise it sets the value.
- [ ] **Step 4:** Run the tests: PASS. `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(protocol): commands, plan, view and wire schemas; JSON patches`.

---

### Task 3: Sim foundations: PRNG, hashing, balance, model math, calendar

**Files:**
- Create in `packages/sim/src/`: `rng.ts`, `hash.ts`, `balance.ts`, `shape.ts`, `calendar.ts`
- Test: `packages/sim/test/foundations.test.ts`

**Interfaces:**
- Produces:
  - **PRNG:**
    - `RngState = {a: number, b: number, c: number, d: number}`
    - `seedRng(seed: number): RngState`: splitmix32 expanding into sfc32 state
    - `rngNext(s: RngState): {value: number, next: RngState}`: a value in [0, 1)
    - `rngInt(s, n): {value: number, next}`: in [0, n)
  - **Hashing:**
    - `stableStringify(v: unknown): string`: sorted keys
    - `stateHash(v: unknown): string`: 16 hex chars, cyrb53-style over `stableStringify`
  - **Model math:**
    - `paramCount(nLayer: number, dModel: number, vocab = 4096): number`: exactly the Python `ModelShape.param_count`, with ffn hidden = `64 * ceil(8*d/3/64)`
    - `trainingFlops(params: number, tokens: number): number`: `6 × params × tokens`
  - **Calendar:**
    - `dateOfDay(day: number): string`: an ISO `YYYY-MM-DD` from 2027-01-01, using days-from-civil arithmetic (no `Date`)
  - **Balance:**
    - `Balance` type, plus `BALANCE_V1` and `BALANCE_TEST` (below)
    - `getBalance(version: "balance-v1" | "balance-test"): Balance`
  - **`BALANCE_V1`:**
    - `{version:"balance-v1", tickMsAt1x: 8000, speeds:[1,2,4], foundingCash: 250000, cloudPflopsPerDay: 0.008, cloudPricePerPflop: 400000, factCheckCost: 5000, maxMembers: 8}`
    - `brainSizes`: `[{id:"tiny",nLayer:4,dModel:128},{id:"small",nLayer:6,dModel:192},{id:"medium",nLayer:8,dModel:256}]`
    - `attentionSpans`: `[128, 256, 512]`
    - `lengths`: `[{id:"short",tokens:6291456},{id:"standard",tokens:12582912},{id:"long",tokens:25165824}]`
    - `defaultPlan`: balanced mix (12 each), standard cleaning, no dedup, no fact-check, natural variety, tiny, 256, standard, learning style 0.5, review 0, empty probe, no parent
  - **`BALANCE_TEST`** (E2E and fast tests): as V1 except
    - `tickMsAt1x: 200`
    - `brainSizes: [{id:"tiny",nLayer:2,dModel:64}]`, `attentionSpans: [64]`
    - `lengths: [{id:"short",tokens:65536}]`
    - `cloudPflopsPerDay: 0.000005`

- [ ] **Step 1: Write the failing tests.**
  - `paramCount`: (4,128) gives 1,377,408; (6,192) gives 3,443,136; (8,256) gives 7,475,456; (10,384) gives 19,275,648; (12,512) gives 40,645,120. These are the Python values.
  - `trainingFlops(1377408, 12582912)` equals `6*1377408*12582912`.
  - `dateOfDay(0)` is `"2027-01-01"`; `dateOfDay(58)` is `"2027-02-28"`; `dateOfDay(365)` is `"2028-01-01"`; `dateOfDay(424)` is `"2028-02-29"`.
  - The same seed gives the same 1000 values; different seeds give different first values; all values are in [0, 1).
  - `stateHash({a:1,b:[2]})` equals `stateHash({b:[2],a:1})`.
  - The first model at V1: `ceil(trainingFlops(1377408,12582912)/1e15/0.008)` is 13 days and costs `round(0.104 × 400000)` dollars ($41,596; ADR 0008 item 11). Assert against the exact formula, so the first model costs about $42k of the $250k founding cash and lasts about 13 game days (104 s at 1×).
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): PRNG, stable hash, balance v1, model math, calendar`.

---

### Task 4: Sim world, companies and clock

**Files:**
- Create in `packages/sim/src/`: `state.ts`, `palette.ts`, `reducer.ts`, `world.ts`
- Test: `packages/sim/test/world.test.ts`

**Interfaces:**
- Consumes: Task 2 schemas (`Command`, `Palette`, `Logo`, `zName`) and Task 3 (`Balance`, rng).
- Produces:
  - **State:**
    - `WorldState = {worldId, name, balance: string, rng: RngState, phase, day, speed, paused, hostUserId, members: Record<string, Member>, companies: Record<string, CompanyState>, inviteCode, seq: number}`
    - `Member = {userId, displayName, joinedSeq, online, companyId: string | null}`
    - `CompanyState = {id, ownerUserId, name, logo, palette, cash, runs: Record<string, RunState>, versions: Record<string, VersionState>, versionSeq: number}`
  - **Inputs:**
    - `Input = {kind:"command", userId, cmd: Command, newId?: string} | {kind:"tick"} | {kind:"system", event: SystemEvent}`
    - `SystemEvent`, part 1: `memberJoined{userId, displayName}`, `presence{userId, online}`
  - **Steps:**
    - `createWorld(p: {worldId, name, hostUserId, hostName, seed: number, balance: "balance-v1"|"balance-test", inviteCode: string}): WorldState`
    - `step(state: WorldState, input: Input): {ok: true, state: WorldState} | {ok: false, reason: string}`: pure; every accepted input increments `seq`
    - `effectiveHost(state): string | null`: the host if online, else the earliest-joined online member
  - **Palettes:**
    - `BASE_BG = "#0f1320"`
    - `contrastRatio(a: Hex, b: Hex): number`: WCAG relative luminance
    - `paletteProblem(p: Palette): string | null`: needs each color ≥ 3:1 against `BASE_BG`, and the two colors not identical
    - `PALETTE_PRESETS: {id, label, palette}[]`: 12 presets, each passing `paletteProblem`

- [ ] **Step 1: Write the failing tests.**
  - `createCompany`:
    - It gives the company `foundingCash` and sets `members[u].companyId`, with `newId` as the company id.
    - A second company for the same member is rejected ("you already have a company").
    - A name equal to another company's (case-insensitive) is rejected.
    - Palette `{primary:"#101010",…}` is rejected with a contrast reason.
    - A `newId` missing from the input is rejected.
  - `startWorld`:
    - Only the effective host may start, only in `setup`, and only once the host has a company. The phase becomes `running`.
  - Clock:
    - `tick` in `setup` or while paused is rejected ("not running"); otherwise it increments `day`.
    - `setSpeed{3}` fails zod in the reducer's command parse; a non-host `setSpeed` is rejected.
  - Presence:
    - `presence{host,false}` makes `effectiveHost` the next-joined online member, or `null` when nobody is online.
  - Every `PALETTE_PRESETS` entry passes `paletteProblem`.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement `step` as a switch over input kinds.**
  - Commands are re-validated with the Task 2 zod schemas inside the reducer, so replayed or hostile input is checked the same way.
  - Domain rules go in the per-command handlers.
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): worlds, companies, palettes and the clock`.

---

### Task 5: Sim training economics, runs and versions

**Files:**
- Create: `packages/sim/src/training.ts`
- Modify: `reducer.ts`, `state.ts`
- Test: `packages/sim/test/training.test.ts`

**Interfaces:**
- Consumes: Tasks 3 and 4.
- Produces:
  - **Preview:**
    - `previewTraining(state, companyId, plan): {ok: true, preview: TrainingPreview} | {ok: false, reason: string}`
    - `cost = round(flops/1e15 × cloudPricePerPflop) + (factCheck ? factCheckCost : 0)`
    - `durationDays = max(1, ceil(flops/1e15 / cloudPflopsPerDay))`
  - **State:**
    - `RunState = {id, plan, preview, seed, status: "training"|"finalizing"|"completed"|"failed", computeStarted: boolean, startDay, endDay, held: TrainingResult | null, failure: string | null, versionId: string | null}`
    - `VersionState = {id, label, runId, parentId, nLayer, dModel, ctxLen, createdDay, trainedTokens, scores, overall, personalityWords: string[], published: boolean}`
  - **System events**, part 2:
    - `trainingCompleted{runId, versionId, result: {steps, finalLoss: number|null, heldoutLosses: Record<string, number|null>, unstable: boolean}}`
    - `trainingFailed{runId, reason}`
    - `computeStarted{runId}`: a worker took the job (sets `computeStarted`; idempotent)
    - `evalCompleted{versionId, scores: Record<string, number|null>, overall: number|null, personalityWords: string[]}`
  - **Commands:** `startTraining{plan}`, with `newId` as the run id.
  - **Helpers:**
    - `activeRun(company): RunState | null`: a run that is training or finalizing
    - `runProgress(run, day): number`: `clamp((day - startDay)/(endDay - startDay), 0, 1)`

- [ ] **Step 1: Write the failing tests.**
  - `startTraining`:
    - With the default plan it charges exactly `preview.cost`, sets `status "training"`, `startDay = day` and `endDay = day + durationDays`, and draws `seed` from the world rng (rng state advances).
  - Rejections, each with no state change and a plain reason:
    - a second run while one is active
    - `cash < cost`
    - an all-zero mix
    - an unknown `brainSize`, `length` or `attentionSpan`
    - a `parentVersionId` of another company
    - a brain size smaller than the parent's in layers or width
    - the world not running
  - Completion held until its game time:
    - `trainingCompleted` arriving at `day < endDay` sets `held`, and the status stays `training`.
    - The tick that reaches `endDay` applies it: version `v1`, `trainedTokens = tokens + parent's`, run `completed`, `versionId` set.
    - A tick reaching `endDay` with no result sets `finalizing`; a later `trainingCompleted` applies at once.
    - A duplicate `trainingCompleted` for a completed run is rejected ("already applied").
  - Failure and evals:
    - `trainingFailed` refunds the cost and sets `failed`.
    - `evalCompleted` fills `scores`, `overall` and `personalityWords`.
  - Version labels count per company: `v1`, `v2`, ….
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
  - The tick handler applies held results for every company with `day >= endDay`, in company-id order, so it stays deterministic.
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): training economics, game-paced runs, versions and evals`.

---

### Task 6: Sim projection, privacy and determinism properties

**Files:**
- Create: `packages/sim/src/project.ts`, `packages/sim/test/arbitraries.ts`
- Test: `packages/sim/test/{project,properties}.test.ts`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `projectView(state: WorldState, viewerUserId: string): WorldView`: validated by `WorldView.parse` in tests
  - Arbitraries:
    - `arbInputs(n): fc.Arbitrary<Input[]>`: random command, tick and system sequences across 1–4 users, some invalid
    - `runInputs(state, inputs)`: applies `step` and skips rejects

- [ ] **Step 1: Write the failing tests.**
  - `project.test.ts`:
    - The owner sees `me.company` with cash, runs and versions.
    - For any other viewer, that company's entry has only the allowed public keys, and `publishedVersions` holds only `published` versions.
    - `inviteCode` is present for members and the view of a non-member throws.
  - `properties.test.ts` (fast-check, 200 runs each):
    - **(a) privacy:** for random worlds, for every viewer and every other company, `JSON.stringify(view)` contains none of that company's cash value-key pairs, run ids, plan probe prompts or unpublished version ids.
    - **(b) determinism:** the same inputs give the same `stateHash`.
    - **(c) replay:** `runInputs(createWorld, inputs)` equals `runInputs(runInputs(createWorld, inputs[:k]), inputs[k:])` for any `k`.
    - **(d) money:** a company's cash = founding cash − charged runs + refunds, and is never negative.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement `projectView`** by building the public shape explicitly, field by field. Never spread or filter the state object.
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): per-viewer projection with privacy, determinism and replay properties`.

---

### Task 7: Server scaffold, database and test harness

**Files:**
- Create in `apps/server/`:
  - `package.json`, `tsconfig.json`, `drizzle.config.ts`, `drizzle/` (generated SQL)
  - `src/{config.ts,app.ts,main.ts,log.ts}`, `src/db/{schema.ts,index.ts}`, `test/harness.ts`
- Test: `apps/server/test/app.test.ts`

**Interfaces:**
- Produces:
  - **Options:**
    - `ServerOptions = {dbDir: string | null /* null = in-memory */, blobDir: string, catalogPath: string | null, secret: string, workerToken: string, balance: "balance-v1"|"balance-test", scheduler: Scheduler, publicUrl: string}`
    - `Scheduler = {now(): number, setTimeout(fn, ms): Handle, clearTimeout(h): void}`, with `realScheduler` and the test `fakeScheduler()` (`advance(ms)`)
  - `buildApp(opts): Promise<{app: FastifyInstance, ctx: AppContext, close(): Promise<void>}>`
  - **Tables** (drizzle; snake_case columns):
    - `users(id uuid pk, display_name, created_at)`
    - `auth_identities(id, user_id, kind 'guest', secret_hash)`
    - `sessions(id, user_id, token_hash unique, created_at, last_seen_at)`
    - `worlds(id, name, host_user_id, invite_code unique, balance, seed int, created_at, owner_instance, lease_until)`
    - `memberships(world_id, user_id, display_name, joined_at, pk(world_id,user_id))`
    - `world_inputs(world_id, seq int, kind, payload jsonb, cid uuid null, user_id null, at, pk(world_id,seq), unique(world_id,user_id,cid))`
    - `world_snapshots(world_id, seq, sim_version, state jsonb, at, pk(world_id,seq))`
    - `jobs(id uuid pk, type, priority int, world_id, payload jsonb, status, lease_worker, lease_until, attempts, result jsonb, error, created_at, updated_at)`
    - `training_runs(id pk, world_id, company_id, job_id, status, created_at)`
    - `training_events(run_id, idx int, step int, event jsonb, pk(run_id,idx))`
    - `lineages(id, world_id, company_id)`
    - `model_versions(id pk, lineage_id, world_id, company_id, parent_id, run_id, blob_key, shape jsonb, created_at)`
    - `eval_results(version_id pk, report jsonb, fingerprint jsonb)`
    - `chat_threads(id, world_id, user_id, version_id, created_at)`
    - `chat_messages(thread_id, idx, role, text, at, pk(thread_id,idx))`
  - `SIM_VERSION = "sim-1"`
  - `harness.ts`: `startTestServer(over?: Partial<ServerOptions>): Promise<TestServer>`, with `url`, `wsUrl`, `ctx`, `scheduler` and `close()`; in-memory PGlite and a temp blob dir

- [ ] **Step 1: Write the failing test** `app.test.ts`.
  - `GET /api/health` returns `{ok:true, sim:"sim-1"}`.
  - Migrations create every table: query `information_schema.tables`.
  - A restart on the same `dbDir` (temp dir) keeps a row written before.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
  - pino with `worldId`, `userId`, `cid`, `jobId` and `runId` bindings where known.
  - Migrations are generated with drizzle-kit, committed, and applied at startup with `drizzle-orm/pglite/migrator`.
  - `main.ts` reads options from the environment (`AIRACE_DATA`, `AIRACE_SECRET` generated into `data/server/secret` on first start, `AIRACE_WORKER_TOKEN`, `AIRACE_BALANCE`, `PORT`).
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): Fastify scaffold, PGlite + Drizzle schema, test harness`.

---

### Task 8: Guest accounts and sessions

**Files:**
- Create: `apps/server/src/auth.ts`
- Test: `apps/server/test/auth.test.ts`

**Interfaces:**
- Consumes: Task 7.
- Produces:
  - **Routes:**
    - `POST /api/guest {displayName}` creates a user and session and sets the httpOnly `airace_session` cookie (SameSite=Lax; `Secure` when `publicUrl` is https). It returns `{userId, displayName}`.
    - `GET /api/me` returns `{userId, displayName, wsToken}` or 401.
  - **Functions:**
    - `wsToken = base64url(userId.exp.hmac)` with a 10-minute expiry
    - `verifyWsToken(ctx, token): string | null`: the userId
    - `requireUser(req): Promise<{userId}>`
  - Session tokens: 32 random bytes; only the sha256 is stored.

- [ ] **Step 1: Write the failing tests.**
  - Guest creation sets a cookie, and `/api/me` with it returns the name.
  - The token is not stored in plain text: no column equals the cookie value.
  - A display name of 25 characters or containing only spaces gets 400.
  - A tampered or expired ws token fails `verifyWsToken`; advance the fake scheduler 11 minutes.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): guest accounts, hashed sessions, ws tokens`.

---

### Task 9: Worlds and the world actor

**Files:**
- Create: `apps/server/src/worlds/{routes.ts,actor.ts,registry.ts,inviteCode.ts}`
- Test: `apps/server/test/{worlds,actor}.test.ts`

**Interfaces:**
- Consumes: Tasks 4–8.
- Produces:
  - **Routes:**
    - `POST /api/worlds {name}` creates the world (`createWorld`, seeded with a random 31-bit int stored in `worlds.seed`) and the creator's membership, and logs `memberJoined` as input 1. It returns `{worldId}`.
    - `GET /api/worlds` returns `{worlds: {id, name, phase, day, memberCount}[]}` for the user.
  - **`WorldActor`:**
    - `load(ctx, worldId): Promise<WorldActor>`: the latest snapshot plus replay of later inputs; refuses a snapshot whose `sim_version` differs
    - `submit(input: Input, meta?: {cid?, userId?}): Promise<{ok: true, seq, duplicate: boolean} | {ok: false, reason}>`
    - `state(): WorldState`
    - `onChange(listener: (state, seq) => void): () => void`
    - `snapshot(reason): Promise<void>`
    - `unload(): Promise<void>`
  - `WorldRegistry.get(worldId): Promise<WorldActor>` (loads lazily; one actor per world) and `shutdown()`.
  - **Order inside `submit`, sequential per world via a promise queue:**
    1. A duplicate `(userId, cid)` returns `{ok:true, duplicate:true}` without applying.
    2. `step()`; a reject returns at once with nothing logged.
    3. `INSERT world_inputs`.
    4. Swap the state.
    5. Notify listeners.
    6. Resolve.
  - **Ticks:** a `tick` is submitted every `tickMsAt1x / speed` ms while `phase=running`, not paused, and at least one member is online. It uses `ctx.scheduler`.
  - **Snapshots:** every 15 s while running, on pause, when the last member goes offline (freeze), and on `shutdown`. After freezing, the actor unloads after 60 s idle.

- [ ] **Step 1: Write the failing tests.**
  - Creating a world logs `memberJoined`, and `GET /api/worlds` lists it.
  - Tick and freeze timing:
    - With the fake scheduler, a running world with an online member advances 3 days after `3 × 8000` ms at 1×, and 6 days at 2×.
    - It advances 0 days while paused or with everyone offline.
  - The same `cid` submitted twice gives one input row and `duplicate:true`.
  - **Recovery:**
    - Write inputs, snapshot at seq 3, write more inputs, drop the actor, reload.
    - The state hash equals the live hash, and the day continues from it.
  - A snapshot with another `sim_version` makes `load` throw with a clear message.
  - A rejected command logs nothing.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
  - Invite codes: 6 characters from `ABCDEFGHJKMNPQRSTUVWXYZ23456789`, unique (retry on collision).
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): worlds, single-writer actors, input log, snapshots, recovery, ticks`.

---

### Task 10: Realtime gateway

**Files:**
- Create: `apps/server/src/realtime/{gateway.ts,session.ts,broadcast.ts}`
- Test: `apps/server/test/gateway.test.ts`, plus `test/wsClient.ts` (a typed test client)

**Interfaces:**
- Consumes: Tasks 2, 6, 8 and 9.
- Produces:
  - **`/ws`:**
    - The first message must be `hello{token}`, within 5 s; otherwise close with 4401.
    - Then `welcome{userId}`.
  - **`watch{worldId}`** (members only, else `error`):
    - submits `presence{userId, online:true}` if the user was offline
    - sends `snapshot{worldId, version: seq, view: projectView(...)}`
  - **Broadcast:**
    - Per watching socket the server keeps `lastView` and `lastVersion`.
    - At most every 250 ms (coalesced) it sends `patch{from: lastVersion, to: seq, ops: diff(lastView, newView)}`, skipping empty diffs.
  - `resync{worldId}` sends a fresh snapshot.
  - `command{cid, worldId, cmd}` goes to `actor.submit({kind:"command", userId, cmd, newId: needsId(cmd) ? randomUUID() : undefined}, {cid, userId})`, then `ack{cid, duplicate}` or `reject{cid, reason}`.
  - **Disconnect:**
    - When a user's last socket on a world closes, `presence{online:false}` is submitted after a 5 s grace.
    - A reconnect within the grace cancels it.
  - **Limits:** 20 commands per 10 s per user per world; above that, `reject{reason:"slow down"}`.
  - `subscribe`/`unsubscribe`: `Gateway.registerStream(prefix, authorize: (userId, sub) => Promise<boolean>)` with `publish(sub, data)` (used by Tasks 14 and 15).
  - `query`: `Gateway.registerQuery(kind, handler: (userId, q) => Promise<unknown>)`.

- [ ] **Step 1: Write the failing tests** (two `wsClient`s, fake scheduler):
  - A bad or absent hello is closed with 4401.
  - `watch` gives a snapshot whose `view` passes `WorldView.parse`.
  - Commands and patches:
    - `createCompany` gives an ack.
    - Within 250 ms one patch arrives whose `applyPatch(snapshot.view, ops)` equals a fresh `projectView`.
  - Resending the same `cid` gives `ack{duplicate:true}` and no second company.
  - A client that drops a patch, so the versions don't chain, sends `resync` and gets a snapshot.
  - **Presence:**
    - Close and reconnect within 5 s: no presence-offline input is logged.
    - After 6 s: `presence false` is logged, and the world freezes (no ticks).
  - 25 rapid commands: the last ones are rejected with "slow down".
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): realtime gateway: snapshots, coalesced patches, acks, resync, presence`.

---

### Task 11: Blob store and signed URLs

**Files:**
- Create: `apps/server/src/blobs.ts`
- Test: `apps/server/test/blobs.test.ts`

**Interfaces:**
- Produces:
  - `BlobStore = {put(key, stream): Promise<void>, get(key): Promise<Readable | null>, exists(key): Promise<boolean>}`
  - `LocalFsBlobStore(dir)`: keys match `/^[a-z0-9][a-z0-9/_.-]{0,200}$/`, no `..`, and every path stays inside `dir`
  - `signBlobUrl(ctx, key, method: "GET"|"PUT", ttlMs = 3_600_000): string`: `${publicUrl}/blobs/${key}?exp=<ms>&sig=<hmac(method key exp)>`
  - Routes `GET /blobs/*` and `PUT /blobs/*` check the signature and expiry (403 otherwise). PUT streams the body up to 512 MB and writes atomically (temp file, then rename).

- [ ] **Step 1: Write the failing tests.**
  - A PUT then a GET round-trips 3 MB.
  - A GET signature used for PUT gets 403; an expired URL gets 403 (fake scheduler); `../x` and `a//..` keys are rejected.
  - A PUT interrupted halfway leaves no blob at the key.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): local blob store with signed GET/PUT URLs`.

---

### Task 12: Catalog of datasets

**Files:**
- Create: `packages/protocol/src/catalog.ts`, `apps/server/src/catalog.ts`, `packages/protocol/fixtures/catalog.sample.json`
- Test: `packages/protocol/test/catalog.test.ts`, `apps/server/test/catalog.test.ts`

**Interfaces:**
- Produces:
  - **`Catalog`** (zod):
    - `{version: "catalog-v1", dataIdentity: string, tokenizer: "tok-v1", suite: "bench-v1", datasets: CatalogDataset[], referenceTraits: Record<string, number>[]}`
    - `CatalogDataset = {id: DatasetId, title, blurb, tokens, docs, pools: Record<PrepKey, {docs, tokens}>, samples: string[] /* 5, ≤600 chars */, removedExamples: {cleaning: Record<"light"|"standard"|"thorough", string[]>, duplicates: string[], factCheck: string[]}}`
    - `PrepKey = "${cleaning}|${dedup ? 1 : 0}|${factCheck ? 1 : 0}"`: 12 keys
  - **Datasets' player-facing titles** (D2's temporary assumption): Web Crawl, Books, Educational, Conversations, Code, Reasoning, Reference Facts, Creative Writing.
  - `loadCatalog(path): Catalog | null`. A missing file logs a warning; `startTraining` is then rejected server-side with "the training data is not installed yet".
  - Query `catalog` returns the catalog without `referenceTraits`.

- [ ] **Step 1: Write the failing tests.**
  - The sample fixture parses.
  - A catalog with 11 pool keys fails.
  - The server's `catalog` query returns 8 datasets in `DATASET_IDS` order.
  - Training with no catalog is rejected with the message above.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat: dataset catalog schema, loading and query`.

---

### Task 13: Job broker and worker gateway

**Files:**
- Create: `packages/protocol/src/worker.ts`, `packages/protocol/fixtures/worker/*.json`, `apps/server/src/jobs/{broker.ts,workerGateway.ts}`, `apps/server/test/fakeWorker.ts`
- Test: `apps/server/test/broker.test.ts`

**Interfaces:**
- Produces:
  - **Worker messages** (zod, `v:1`).
    - Worker to server:
      - `whello{token, workerId, capabilities: {device: "cuda"|"cpu", vramMb: number, maxParams: number, jobTypes: ("train"|"eval"|"generate")[], slots: {train: number, eval: number, generate: number}, dataIdentity: string}, running: string[] /* job ids it still runs */}`
      - `heartbeat{}`
      - `progress{jobId, event: unknown}`
      - `chunk{jobId, text}`
      - `done{jobId, result: unknown}`
      - `failed{jobId, error: string, retryable: boolean}`
    - Server to worker:
      - `wwelcome{}`
      - `lease{jobId, type, spec: unknown, leaseMs: 30000}`
      - `cancel{jobId}`
  - **Job specs and results:**
    - `TrainJobSpec = {runId, config: object /* Python TrainRunConfig JSON, snake_case */, parentTarUrl: string|null, outTarUrl: string}`
    - `TrainJobResult = {steps, final_loss, heldout_losses, status: "completed"|"unstable_stopped"}`
    - `EvalJobSpec = {versionId, modelTarUrl, population: Record<string, number>[]}`
    - `EvalJobResult = {report: object /* BenchReport.to_dict */, traits: Record<string,number>, personalityWords: string[]}`
    - `GenerateJobSpec = {versionId, modelTarUrl, history: [ "user"|"ai", string ][], maxNewTokens: 96, temperature: 0.8, topP: 0.95, seed: number}`
    - `GenerateJobResult = {text}`
  - **`JobBroker`:**
    - `enqueue(type, payload, worldId): Promise<jobId>`. Priority: generate 0, eval 1, train 2; lower runs first.
    - `onProgress(type, fn)`, `onChunk(fn)`, `onDone(type, fn)`, `onFailed(type, fn)`
    - leasing: whenever a worker has a free slot of a type, it gets the oldest highest-priority queued job of a type it supports, if its `dataIdentity` equals the catalog's
    - heartbeat every 5 s; a lease expires after 30 s without one, and the job is requeued (`attempts+1`; failed after 3)
    - `running` in `whello` re-attaches leases to the reconnecting worker
    - `reconcile()` at startup requeues `leased` jobs whose lease has expired
  - `/worker`: authenticates `whello.token === workerToken` (else close 4401).
  - **`fakeWorker.ts`**, a test helper only (never shipped): it connects as a worker, records leases, and lets the test send `progress`, `chunk`, `done` and `failed`.

- [ ] **Step 1: Write the failing tests.**
  - Leasing and priority:
    - An enqueued train job is leased to a fake worker with a train slot.
    - A generate job enqueued afterwards is leased before a queued second train job.
    - A worker with another `dataIdentity` gets nothing.
  - Lease loss and re-attach:
    - No heartbeat for 31 s requeues the job, and a second worker gets it.
    - A worker reconnecting with `running:[jobId]` keeps its lease, and the job is not given to another worker.
  - Results and restarts:
    - `done` marks the job done, stores the result, and fires `onDone`.
    - A server restart with a leased job, then `reconcile` after the expiry, requeues it.
  - Cross-language fixtures:
    - Every fixture in `fixtures/worker/` parses with the zod schemas.
    - The Python suite (Task 17) loads the same files.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): job broker with leases, priorities and worker gateway`.

---

### Task 14: Training service: from command to live view to new version

**Files:**
- Create: `apps/server/src/training/{service.ts,config.ts,reveal.ts}`
- Test: `apps/server/test/training.test.ts`

**Interfaces:**
- Consumes: Tasks 5, 9, 10, 11, 12 and 13.
- Produces:
  - **`toTrainRunConfig(plan, {runId, seed, balance, parent: {dir: "parent"} | null}): object`**: the exact Python `TrainRunConfig` JSON.
    - `mixture` = the plan's mix with zero weights dropped
    - `prep` = `{cleaning, dedup: removeDuplicates, fact_check: factCheck, variety}`
    - `shape` from the brain size and attention span
    - `token_budget` from the length
    - `boldness = learningStyle`, `replay = review`
    - `probe_prompts = probePrompt ? [probePrompt] : []`
    - `batch_tokens: 8192`, `parent_dir: "parent"` when there is a parent
  - **`TrainingService`:**
    - It listens for accepted `startTraining` inputs (actor `onChange` plus a check of the input log) and creates the `training_runs` row and a train job.
    - Signed URLs: `parentTarUrl` is a GET of `checkpoints/<parentVersionId>.tar`; `outTarUrl` is a PUT of `checkpoints/<newVersionId>.tar`, with `newVersionId = randomUUID()` stored on the run row.
    - `progress` events are stored in `training_events`.
    - On the job's first lease it submits `computeStarted{runId}`. On `done` it submits `trainingCompleted{runId, versionId, result}`, mapping the worker's snake_case result (`final_loss`, `heldout_losses`, `status === "unstable_stopped"` → `unstable`); on `failed` (not retryable, or after 3 attempts) it submits `trainingFailed`.
    - When the actor applies a completion, it inserts `lineages`/`model_versions` rows (blob key, shape) and enqueues an eval job. The population is the world's other versions' traits plus the catalog's `referenceTraits`.
    - On an eval `done` it stores `eval_results` and submits `evalCompleted`.
  - **Game-pace reveal:**
    - `revealedCount(run, day, intraDayFraction, events): number`: events whose `step ≤ totalSteps × progress`, where `progress = clamp((day + fraction - startDay)/(endDay - startDay), 0, 1)`
    - Stream `training:<worldId>:<runId>` (owner only) publishes newly revealed events every 500 ms.
    - Query `runTelemetry{worldId, runId}` (owner only) returns all events revealed so far.
  - **Startup reconciliation:** runs that are done in `jobs` but not applied in the world get their `trainingCompleted` again (the reducer rejects duplicates).

- [ ] **Step 1: Write the failing tests** (fake worker, `balance-test`, fake scheduler):
  - `toTrainRunConfig(defaultPlan, …)` deep-equals the committed fixture `fixtures/train-config.default.json`. Task 17's pytest also loads that fixture into Python `TrainRunConfig` and validates it.
  - **Full flow:**
    1. `startTraining` leads to a train job with the signed URLs.
    2. Ten `progress` events are sent.
    3. At 50% of game time the owner's `runTelemetry` returns about 5, and a non-owner `subscribe` is refused.
    4. `done` at 60% leads to `held`; at `endDay` version v1 appears in the owner's view, with `model_versions` row count 1.
    5. An eval job is enqueued; its `done` fills the version's scores.
  - **Restart mid-run:** stop the server after `progress`, start it on the same `dbDir`, and the fake worker reconnects with `running:[jobId]`. Then `done` leads to exactly one version, and cash equals founding cash − cost.
  - **No worker online:**
    - The run's view status is `training`, and the job stays queued.
    - At `endDay` the run becomes `finalizing`.
    - A worker connects, gets the lease, and its `done` applies at once.
  - `failed` with `retryable:false` refunds the cost.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): training service: jobs, game-paced telemetry, versions, evals, recovery`.

---

### Task 15: Chat service (playground)

**Files:**
- Create: `apps/server/src/chat/service.ts`
- Test: `apps/server/test/chat.test.ts`

**Interfaces:**
- Consumes: Tasks 10, 11 and 13.
- Produces:
  - **`chat{cid, worldId, threadId|null, versionId, text}`:**
    - Allowed when the version belongs to the user's company. Stage 2 extends this to published versions of other companies.
    - A new thread is created when `threadId` is null, and the user message is stored.
    - A generate job is enqueued: `history` = the thread's last 12 turns, `modelTarUrl` = signed GET, `seed` = random.
    - The ack is `ack{cid, data:{threadId}}`.
  - **Streaming:** `chunk`s go out on stream `chat:<worldId>:<threadId>` (asker only) as `{kind:"chunk", text}`. On `done` the AI message is stored and `{kind:"done", text}` published; on `failed`, `{kind:"failed", reason:"The AI couldn't answer right now. Try again."}`.
  - Queries `chatThreads{worldId}` (the asker's threads) and `chatThread{worldId, threadId}` (messages, asker only).
  - **Limits:** one in-flight reply per thread; 30 messages per 10 minutes per user. Text is cleaned (1–500 characters, newlines allowed).
  - Literal `<|end|>` in user text is passed through unchanged as text: the Python tokenizer treats it as plain text (M1 Review Focus 1).

- [ ] **Step 1: Write the failing tests.**
  - Sending to an own version gives a generate job whose spec parses as `GenerateJobSpec`.
  - Fake-worker chunks `"Hel"` and `"lo"` arrive in order on the stream, and `done` stores `"Hello"`.
  - Reconnecting and querying `chatThread` returns both messages.
  - Another user's `chatThread` query is refused.
  - Another company's unpublished version is rejected.
  - A 501-character message is rejected, and a second send while a reply is in flight gets "wait for the reply".
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): playground chat over generate jobs with streaming and history`.

---

### Task 16: Catalog export (Python)

**Files:**
- Create: `ml/src/airace_ml/catalog.py`
- Modify: `ml/src/airace_ml/cli.py` (the `export-catalog` command), `CLAUDE.md` (the command)
- Test: `ml/tests/test_catalog.py`

**Interfaces:**
- Consumes: M1 `Corpus`, `prep.eligible_docs`, `data_identity`, `Tok`, and the gate fingerprint caches.
- Produces:
  - `build_catalog(root: Path, population_dirs: list[Path] = []) -> dict` matching Task 12's `Catalog`, with `dataIdentity = data_identity(root)`.
  - Per dataset:
    - `pools` covers all 12 prep keys (docs and tokens of the eligible pool).
    - `samples`: 5 clean documents chosen with a fixed seed and cut at a word boundary at 600 characters.
    - `removedExamples`: 3 per option, taken from documents that option removes. For cleaning, those below the threshold; for duplicates, non-canonical cluster members; for fact-check, `false_fact` documents.
  - `referenceTraits`: the `traits` of every `gate_fingerprint.json` found under `population_dirs`.
  - CLI: `airace-ml export-catalog [--data-root] [--population ml/runs/gate-1/runs]` writes `data/catalog/v1/catalog.json`.

- [ ] **Step 1: Write the failing tests** on `tiny_data_root`:
  - All 8 datasets are present with 12 pool keys each, and thorough pools are never larger than light ones.
  - Samples are ≤ 600 characters and come from clean documents.
  - The output validates against `packages/protocol/fixtures/catalog.sample.json`'s key structure: the same top-level and per-dataset keys.
  - Running twice gives byte-identical JSON.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS. Then run it for real on `data/` and commit nothing from `data/`.
- [ ] **Step 5: Commit** `feat(ml): export-catalog for the game server`.

---

### Task 17: Streaming chat and the worker agent (Python)

**Files:**
- Create: `ml/src/airace_ml/worker/{__init__.py,protocol.py,agent.py,blobs.py}`
- Modify: `ml/src/airace_ml/infer/lm.py`, `ml/src/airace_ml/cli.py` (`worker` command), `ml/pyproject.toml` (`websockets`)
- Test: `ml/tests/test_worker_protocol.py`, `ml/tests/test_worker_agent.py`, `ml/tests/test_infer.py` (streaming)

**Interfaces:**
- Consumes: Task 13's message shapes, through the shared fixtures in `packages/protocol/fixtures/worker/`.
- Produces:
  - **`TorchLM.stream_chat_reply(history, *, max_new_tokens=96, temperature=0.8, top_p=0.95, seed=0, on_text: Callable[[str], None]) -> str`.**
    - It reproduces `chat_reply` with the same seed: same text, same sampling.
    - It calls `on_text` with each newly decodable piece (decode the full reply each step and emit the new suffix, so multi-byte characters never split).
  - **Protocol:** `parse_server_message(raw: str) -> ServerMsg` and `dumps(msg) -> str`, with dataclasses mirroring Task 13. Unknown types raise `ProtocolError`.
  - **`WorkerAgent(server_url, token, data_root, device, slots)`:**
    - `async run_forever()`: connect, `whello` (capabilities: `maxParams` 85,000,000 or the VRAM-scaled limit, `dataIdentity`), heartbeat every 5 s, and a reconnect backoff of 1, 2, 4 … 30 s
    - each `lease` runs the registered handler for its type in a thread, sending `progress`, `chunk`, `done` or `failed`
    - after a reconnect, `whello.running` lists the jobs still in flight
  - **Blobs:**
    - `download_tar(url, dest_dir) -> Path`: stream, then extract with path-traversal checks
    - `upload_dir_tar(src_dir, url) -> None`: a streamed PUT with retries
    - both use `httpx` (already present via huggingface) or `urllib`; no new dependency except `websockets`
  - CLI: `airace-ml worker --server ws://localhost:8787/worker --token T [--data-root] [--device]`

- [ ] **Step 1: Write the failing tests.**
  - The streaming reply equals `chat_reply` with the same seed, and the concatenated `on_text` pieces equal the reply.
  - Every fixture in `packages/protocol/fixtures/worker/` round-trips through `parse_server_message`/`dumps` (worker-to-server fixtures through the encoders).
  - `fixtures/train-config.default.json` loads as `TrainRunConfig` and validates.
  - An agent against an in-process `websockets` fake server:
    - it sends `whello` with the right `dataIdentity`, then heartbeats
    - on a fake `lease` of type `echo` (test-only handler) it sends `done`
    - after the server drops the connection, it reconnects with `running` listing an unfinished job
  - `download_tar` refuses a tar containing `../evil`.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS; ruff clean.
- [ ] **Step 5: Commit** `feat(ml): streaming chat replies and the dial-out worker agent`.

---

### Task 18: Worker job handlers (Python)

**Files:**
- Create: `ml/src/airace_ml/worker/jobs.py`, `ml/src/airace_ml/worker/models.py`
- Test: `ml/tests/test_worker_jobs.py`

**Interfaces:**
- Consumes: Task 17 and M1's `train_run`, `run_benchmarks`, `measure_fingerprint`, `describe`, `Judge`, `NoveltyIndex` and `load_checkpoint`.
- Produces:
  - **`handle_train(spec, emit) -> TrainJobResult`:**
    1. Download the parent tar (if any) into `work/<runId>/parent`.
    2. `train_run(TrainRunConfig.from_json(spec.config with parent_dir), out_dir=work/<runId>/out, on_event=lambda e: emit("progress", event_to_dict(e)), resume=has_resume_state(out_dir))`.
    3. Tar `out` without `resume/`, upload it, and return the result fields.
    - Validation errors are `failed(retryable=False)`; I/O and network errors are `retryable=True`.
  - **`handle_eval(spec) -> EvalJobResult`:**
    - Load the model from the tar (cached by version).
    - `run_benchmarks(lm, tok, judge=…, novelty=…)` with the installed judge and novelty index. When either is missing or refused, run without them and report `creativity` as `None` ("not measured").
    - Fingerprint at the default k (8).
    - `personalityWords = describe(fp, population fingerprints built from spec.population)`.
    - Return `report.to_dict()`, traits and words.
  - **`handle_generate(spec, emit) -> GenerateJobResult`:** an LRU-cached model (`ModelCache(capacity=4)`, keyed by version id; it evicts from GPU to CPU to dropped), then `stream_chat_reply`, emitting `chunk`s.

- [ ] **Step 1: Write the failing tests** (CPU, `tiny_data_root`, quick shapes):
  - **Train:** `handle_train` on a `(2,64,64)` config with a local fake blob server (an http.server fixture):
    - It emits Progress, HeldoutEval, Sample and Done dicts.
    - It uploads a tar that `load_checkpoint` reads.
    - A second train with that tar as parent continues the lineage (meta parent id).
  - **Eval:** `handle_eval` returns all 8 category keys. Creativity is `None` without a judge, and the words are a list.
  - **Generate:**
    - The chunks join to the returned text.
    - Two generates on the same version download once (cache hit).
  - A config whose shape is smaller than the parent's gives `failed(retryable=False)`.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS. The default suite stays under 3 minutes; mark anything over 10 s `@pytest.mark.slow`.
- [ ] **Step 5: Commit** `feat(ml): worker train, eval and generate jobs`.

---

### Task 19: Client scaffold: theme, net layer, store

**Files:**
- Create in `apps/client/`: `package.json`, `vite.config.ts` (proxy to 8787), `index.html`
- Create in `apps/client/src/`: `main.tsx`, `app.tsx` (router), `theme/{tokens.css,palette.ts}`, `net/{socket.ts,api.ts}`, `store/{world.ts,session.ts}`, `components/{Button.tsx,Card.tsx,Field.tsx}`
- Test: `apps/client/test/{palette,socket,store}.test.ts`

**Interfaces:**
- Consumes: Task 2 (`ClientMessage`, `ServerMessage`, `applyPatch`) and Task 4 (`contrastRatio`, `BASE_BG`).
- Produces:
  - **Base theme** (C5): CSS variables in `tokens.css`: `--bg #0f1320`, `--surface #171c2c`, `--surface-2 #1f2538`, `--text #e8ebf5`, `--muted #9aa3c2`, `--border #2a3150`, `--danger #ff6b6b`.
    - Type: "Space Grotesk" for headings, the system UI font for body, loaded from Google Fonts.
    - Motion respects `prefers-reduced-motion`.
  - **Company theme:**
    - `applyCompanyTheme(p: Palette | null): void` sets `--brand`, `--brand-2`, `--on-brand` (`#000` or `#fff`, whichever contrasts more) and `--brand-soft` (primary at 16% over `--bg`). `null` gives the neutral defaults.
  - **`GameSocket`:**
    - `connect(): Promise<void>` fetches `/api/me`, then `hello` with the ws token.
    - `watch(worldId)`
    - `command(worldId, cmd): Promise<{ok: true, data?} | {ok: false, reason}>`: generates a cid and resends it on reconnect until an ack
    - `chat(...)`, `query(q)`, `subscribe(sub, onData): () => void`
    - Reconnect backoff 0.5, 1, 2 … 10 s; on reconnect it re-watches, resubscribes and resends un-acked commands with the same cids.
  - **`useWorld` store** (Zustand): `{worldId, version, view, status: "connecting"|"live"|"offline"}`, with `onSnapshot` and `onPatch`. On `patch.from !== version` it calls `resync`.
  - `api.ts`: `guest(displayName)`, `me()`, `listWorlds()`, `createWorld(name)`.

- [ ] **Step 1: Write the failing tests.**
  - Every `PALETTE_PRESETS` entry gives an `--on-brand` with contrast ≥ 4.5 against the primary.
  - **Socket** (a mock WebSocket):
    - A command un-acked when the socket drops is resent once with the same cid after reconnect.
    - A patch with a version gap triggers `resync`.
    - Patches apply in order to give the expected view.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS. `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(client): theme with company palettes, game socket, world store`.

---

### Task 20: Home, world creation and company founding

**Files:**
- Create: `apps/client/src/screens/{Home.tsx,Lobby.tsx}`, `apps/client/src/components/{LogoBuilder.tsx,Logo.tsx,PalettePicker.tsx}`
- Test: `apps/client/test/{home,lobby}.test.tsx`

**Interfaces:**
- Consumes: Tasks 19 and 4 (`paletteProblem`, `PALETTE_PRESETS`, `LOGO_*`).
- Produces:
  - **Home:**
    - On first visit it asks for a display name ("What should we call you?"), then `guest`.
    - It lists "Your worlds" with Open buttons, and offers "Start a new world" (name).
  - **Lobby (setup phase):**
    - "Found your company": name, logo builder (shape, symbol or initials, shown live in the palette) and palette picker (12 presets, plus "Custom" with two color inputs and a plain contrast warning: "This color is hard to read on the dark background. Pick a brighter one.").
    - "Found company" sends `createCompany`, then "Open for business" sends `startWorld`. In Stage 1 the founder is the host.
  - `Logo` renders SVG from `Logo` and `Palette`. The initials are text nodes, never HTML.
  - After founding, `applyCompanyTheme` runs with the company's palette, so every later screen takes its colors.

- [ ] **Step 1: Write the failing tests** (Testing Library, mock socket):
  - Picking a preset and a symbol shows them in the preview.
  - A custom primary `#202020` shows the contrast warning and disables nothing (the server also rejects it).
  - Submitting sends `createCompany` with exactly the chosen name, logo and palette.
  - A company name `<img src=x onerror=alert(1)>` renders as literal text.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): home, world creation, company founding with logo and palette`.

---

### Task 21: Dashboard and AI Lab planning (Data, Prepare, Design)

**Files:**
- Create: `apps/client/src/screens/{Dashboard.tsx,lab/LabLayout.tsx,lab/DataStep.tsx,lab/PrepareStep.tsx,lab/DesignStep.tsx}`, `apps/client/src/store/plan.ts`
- Test: `apps/client/test/lab.test.tsx`

**Interfaces:**
- Consumes: Tasks 5 (`previewTraining`), 12 (the `catalog` query) and 19.
- Produces:
  - **Dashboard:**
    - It shows the company header (logo, name), cash, the date with speed and pause controls (host only), the current run status and the latest version.
    - Navigation: AI Lab, Playground, Benchmarks.
  - `usePlan` store: a `TrainingPlan` starting from `balance.defaultPlan`, with the parent set to the latest version.
  - **Data step:**
    - A card per dataset: title, blurb, size ("about 32 million words"), and a "See examples" drawer with the catalog samples.
    - A slider for each mix weight, with shares shown as percentages.
  - **Prepare step:**
    - Cleaning as a segmented control (Light, Standard, Thorough), plus toggles for Remove duplicates, Fact-checking (with its $ cost) and Variety.
    - Each shows "what this removes" with real removed examples, and the pool size after preparation from `pools[prepKey]`.
  - **Design step:**
    - Controls: brain size (shows cost and time; options smaller than the parent are disabled with "Your AI can only grow"), attention span, training length, learning style (Careful to Bold), review what it knows (0–50%), and an optional "Your test question" (the probe prompt, ≤120 characters).
  - **Summary bar:** a live `previewTraining` cost and game-time duration, and "Start training" (disabled with a reason when the preview fails). It sends `startTraining`, then goes to the live view.
  - No control text uses ML jargon (R7); each has a one-line "what this does".

- [ ] **Step 1: Write the failing tests.**
  - Moving the brain size from tiny to small changes the shown cost to `previewTraining`'s value.
  - With cash below the cost, "Start training" shows "Not enough cash".
  - Data step: the cards show the catalog samples as text.
  - Prepare step: Thorough shows the catalog's thorough removed examples and the thorough pool size.
  - "Start training" sends the exact plan.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): dashboard and AI Lab data, prepare and design steps`.

---

### Task 22: Live training view and versions

**Files:**
- Create: `apps/client/src/screens/lab/{TrainStep.tsx,VersionsStep.tsx}`, `apps/client/src/components/{LossChart.tsx,LearningChart.tsx,SampleFeed.tsx}`
- Test: `apps/client/test/train.test.tsx`

**Interfaces:**
- Consumes: Task 14 (the `runTelemetry` query, the `training:` stream, `RunView`).
- Produces:
  - **TrainStep:**
    - On mount it queries `runTelemetry`, then subscribes to the training stream.
    - It shows: game-time progress; "How wrong its guesses are" (a uPlot of training loss, falling); "What it's learning" (held-out loss per dataset as lines in the company palette family); and "What it says" (a feed of Sample events, the probe prompt and the player's test question, newest first, text only).
    - States: "Waiting for compute" (`computeStarted` false: "No lab computer is free yet; your run starts as soon as one is."), "Training", "Finishing up" (finalizing: "The lab computers are still working"), "Done: meet <label>", and "Failed" with the reason and "Your cash was refunded".
  - **VersionsStep:** the list of versions (label, brain size, trained words, date, overall score once evaluated) with "Chat" and "Train further from this" (sets the parent).

- [ ] **Step 1: Write the failing tests.**
  - The backlog then live events render the right number of chart points and samples.
  - A remount (as after a reload) shows the same backlog without duplicates.
  - The status `finalizing` shows "Finishing up".
  - A sample text containing `<script>` renders as text.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): live training view and versions`.

---

### Task 23: Playground and benchmarks

**Files:**
- Create: `apps/client/src/screens/{Playground.tsx,Benchmarks.tsx}`, `apps/client/src/components/ChatLog.tsx`
- Test: `apps/client/test/{playground,benchmarks}.test.tsx`

**Interfaces:**
- Consumes: Tasks 15 and 22.
- Produces:
  - **Playground:**
    - A version picker (own versions), a thread list, a chat log (`white-space: pre-wrap`; streamed text appears as it arrives) and an input (1–500 characters).
    - "Thinking…" shows while waiting.
    - A failure message offers "Try again".
    - A note for a model never trained on conversations: "This AI never saw a conversation, so it may just continue your text." The note is driven by `plan.mix.conversations === 0` of its training runs.
  - **Benchmarks:**
    - Per version, eight category scores (0–100, "0 = guessing") with plain explanations: Language, Reasoning, Patterns, Knowledge, Coding, Creativity, Consistency, Following instructions.
    - "Not measured" for null, the overall score, the personality words, and a comparison with the parent version (up or down).

- [ ] **Step 1: Write the failing tests.**
  - Typing and sending calls `socket.chat`. Streamed chunks `"Hi"` and `" there"` render "Hi there", then "done" finalizes.
  - A 501-character draft shows "Keep it under 500 characters".
  - Benchmarks render the eight categories with `creativity: null` shown as "Not measured".
  - Model output `<b>x</b>` renders as literal text.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): playground chat and benchmark scores`.

---

### Task 24: Stage 1 end-to-end, dev runner, docs

**Files:**
- Create: `tools/dev.ts`, `playwright.config.ts`, `tests/e2e/solo.spec.ts`, `tests/e2e/fixtures.ts`
- Modify: `CLAUDE.md` (dev and E2E commands), `docs/progress.md`, `docs/handoff/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - **`npm run dev`** (`tools/dev.ts`):
    - starts the server (`tsx apps/server/src/main.ts`), the Python worker (`python -m uv run airace-ml worker …`, with `uv` resolved as in CLAUDE.md) with the shared worker token, and Vite
    - exports the catalog first if it is missing
    - prints the URL
  - **`tests/e2e/fixtures.ts`:**
    - builds a tiny corpus into a temp root (`airace-content build --scale tiny --fixtures ml/tests/fixtures/sources --out <tmp>`), exports its catalog, and starts the server with `AIRACE_BALANCE=balance-test` and a real CPU worker
    - this is the slow E2E path, not part of `npm test`

- [ ] **Step 1: Write the E2E test** `solo.spec.ts`:
  1. Open the app and enter a name.
  2. Create a world and found a company with a preset palette; assert the dashboard's `--brand` equals the palette primary.
  3. Open for business and start training with the defaults.
  4. Assert that the loss chart gains points and a sample appears.
  5. Reload the page mid-run: the chart shows the backlog and keeps growing.
  6. Restart the server process mid-run (via the fixture): the page reconnects and the run completes.
  7. Assert version v1 appears once and the cash equals founding cash minus one cost.
  8. Open the Playground, send "Hello", and assert a non-empty streamed reply.
  9. Open Benchmarks: eight categories are shown.
- [ ] **Step 2:** Run `npm run e2e -- solo`: FAIL, then make it pass (fix only real bugs, with tests in their owning tasks).
- [ ] **Step 3: Run everything.**
  - `npm test`, `npm run check`, `cd ml && python -m uv run pytest`, `cd ml && python -m uv run ruff check .`: all green.
  - Run `npm run dev` by hand on the full corpus, with one real first model on the GPU.
- [ ] **Step 4: Update the docs.** CLAUDE.md commands; a dated `docs/progress.md` entry ("Stage 1 playable"); the handoff resume point.
- [ ] **Step 5: Commit** `test(e2e): single-player flow; dev runner; Stage 1 complete`. **Stop and report to the owner before Stage 2.**

---

# Stage 2: friends

### Task 25: Joining friends

**Files:**
- Modify: `apps/server/src/worlds/routes.ts`, `packages/sim/src/reducer.ts`, `apps/client/src/screens/{Home.tsx,Lobby.tsx}`, plus a `Join.tsx` route
- Test: `apps/server/test/join.test.ts`, `apps/client/test/join.test.tsx`

**Interfaces:**
- Produces:
  - `POST /api/worlds/join {code}`: a membership plus `memberJoined`. The 9th human gets "This world is full (8 players)". Joining twice is idempotent.
  - Invite link `/join/<code>`: a guest who is not signed in is asked for a name first.
  - **Lobby:** a members list with online dots, and "Invite friends" (code and copy-link).
  - **Late joiners** found their company in a running world; a "Founder grant" scaled to world age (`foundingCash + 2000 × day`, capped at 2× founding) is applied in `createCompany` when `day > 0`.
  - **Host controls** follow `effectiveHost`. When nobody is online the world freezes (Tasks 9 and 10 already do this); test it with two users.

- [ ] **Step 1: Write the failing tests.**
  - Two users: B joins by code and sees A's company in `companies` without cash or runs.
  - A goes offline, so B becomes the effective host and can pause.
  - Both go offline, so no ticks run; B returns and the day continues.
  - A late joiner on day 30 gets `250000 + 60000` cash.
  - A full world refuses the 9th user.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat: join worlds by invite, presence-aware host, late founders`.

---

### Task 26: Sharing AIs with friends (AI Directory)

**Files:**
- Modify: `packages/sim/src/{training.ts,project.ts}`, `apps/server/src/chat/service.ts`
- Create: `apps/client/src/screens/Directory.tsx`
- Test: `packages/sim/test/publish.test.ts`, `apps/server/test/directory.test.ts`, `apps/client/test/directory.test.tsx`

**Interfaces:**
- Produces:
  - **Sim:** `publishVersion{versionId, published}` (owner only; the version must exist). `PublicCompany.publishedVersions` lists the published ones with `PublicVersion` fields only.
  - **Chat:** extends to other companies' published versions ("Try it"). The thread is private to the asker.
  - **Counts:** the owner sees `tries7d` per published version, through query `versionTries{worldId}`, with counts only and never transcripts. Generation limits are 10 per 10 minutes per asker per version.
  - **Directory screen:** every company's published AIs with logo, name, label, brain size, scores and personality words, plus "Try it". Owners get "Share with friends" and "Stop sharing" in Versions.
  - **Opacity tests:** the property test from Task 6 extended with `publishVersion`; a server test asserting that a non-owner can never query `runTelemetry`, `chatThread` (of others) or an unpublished version's chat.

- [ ] **Step 1: Write the failing tests.**
  - Unpublished versions are absent from other viewers' views; published ones carry only public fields.
  - B chats with A's published v1 and gets a streamed reply. A's `chatThreads` doesn't include B's thread, and A's `versionTries` shows 1.
  - After A stops sharing, B's next message is rejected.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat: share AIs with friends: directory, try-it chat, opacity tests`.

---

### Task 27: Host from your PC

**Files:**
- Create: `tools/host.ts`, `docs/hosting.md`
- Modify: `apps/server/src/app.ts` (serve `apps/client/dist` with @fastify/static and SPA fallback), root `package.json` (`npm run host`)

**Interfaces:**
- Produces: `npm run host`:
  1. Builds the client.
  2. Exports the catalog if it is missing.
  3. Starts the server on 8787 (serving the client) and the GPU worker.
  4. If `cloudflared` is installed, runs `cloudflared tunnel --url http://localhost:8787` and prints the public `https://…trycloudflare.com` link, with `publicUrl` set to it so cookies are `Secure` and blob URLs correct. Otherwise it prints the LAN link and how to install cloudflared.
  - `docs/hosting.md` is a plain-language guide for the owner: start, share the link, stop, and what friends need (only a browser).

- [ ] **Step 1: Write the failing test** `apps/server/test/static.test.ts`: with a built `dist` fixture, `GET /` and `GET /join/ABC123` return `index.html`, and `/api/health` still works.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.** The script detects `cloudflared` on PATH and never downloads anything itself.
- [ ] **Step 4:** Run: PASS, then a manual run of `npm run host` with and without cloudflared.
- [ ] **Step 5: Commit** `feat: host from your PC with an optional Cloudflare quick tunnel`.

---

### Task 28: Two-friends end-to-end and milestone close

**Files:**
- Create: `tests/e2e/friends.spec.ts`
- Modify: `docs/progress.md`, `docs/handoff/README.md`, `CLAUDE.md`

- [ ] **Step 1: Write the E2E test** `friends.spec.ts` (two browser contexts = two networks' worth of cookies; `balance-test`; real CPU worker):
  1. A creates a world; B joins by link. Both found companies.
  2. Both train in parallel, and both live views progress.
  3. A shares v1; B tries it and gets a reply.
  4. **Opacity:** B's page, its `WorldView` JSON over the socket, and every network response never contain A's cash, plan or probe prompt, or any blob URL.
  5. B disconnects (context offline) for 10 s and comes back: B's view resyncs, and A's world kept ticking.
  6. A server restart: both reconnect; days, versions and cash are preserved.
- [ ] **Step 2:** Run `npm run e2e`: PASS.
- [ ] **Step 3: Run everything** (as in Task 24), plus a real two-person test through the tunnel with the owner.
- [ ] **Step 4: Update the docs**: progress entry "Milestone 2 complete" and the handoff resume point (M3 next).
- [ ] **Step 5: Commit** `test(e2e): two friends; Milestone 2 complete`.
