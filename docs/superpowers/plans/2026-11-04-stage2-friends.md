# Stage 2 "Friends" Implementation Plan (own hardware, Fair mode)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The owner prefers inline execution (`superpowers:executing-plans`), committing and pushing after every task, with one whole-branch self-review at the end.

**Goal:** Friends play one world together, each training and running their own AI on their own computer through a helper program, with an optional Fair mode that pools everyone's graphics cards so hardware doesn't decide.

**Architecture:**
- The existing ML worker becomes each player's **helper**: personal keys bind a card to a player, and capabilities report the card's name, memory and measured speed.
- Every job gets an **owner** and a **need**. The broker routes by the world's Fair mode: own cards only, or a shared pool served least-used company first, on the fastest card that fits.
- The sim (sim-7) gains Fair mode, the founder grant, autopilot and battles against players' AIs.
- The server gains joining, the AI Directory's Try-it, two-sided battles with votes, world chat, hosting through a quick tunnel and the helper downloads.

**Tech Stack:** As Phase 4. Python 3.13 worker (`ml/`), TypeScript (protocol, sim, Fastify server, React client), PGlite with drizzle migrations, Playwright for end to end. Windows PowerShell for the helper setup.

**Spec:** `docs/superpowers/specs/2026-11-04-remaining-features-design.md` §0–§2 and §8–§9 (decisions C36–C41). Also the architecture spec §6 (multiplayer) and §7 (job system). R1, R2 and R4–R8 apply; R3 is amended by spec §1.

**Replaces:** Tasks 25–28 of `docs/superpowers/plans/2026-10-22-m2-two-friends.md` (never built). Read them for their tests; this plan supersedes their interfaces.

## Global Constraints

- **Versions:** `SIM_VERSION = "sim-7"`; `UPGRADABLE` gains `"sim-6"`; `upgradeState` adds the new fields with their starting values (Fair mode false).
- **Fair mode** is fixed at world creation; the default is off.
- **Notice copy (exact):** "Everyone's graphics cards are shared. Your AI, its reading list and your notebook may be trained on a friend's computer for a while. Play with people you trust."
- **Players:** 8 humans at most ("This world is full (8 players)."); the four rival companies stay.
- **Founder grant:** `foundingCash + 2000 × day`, capped at `2 × foundingCash`, when a company is founded on day > 0.
- **Fair-mode share:** least card time over the last 30 game days goes first; then priority; then age.
- **Stale customer questions:** feedback and appReview jobs older than 14 game days are dropped.
- **Run progress in Fair mode:** saved every 2 minutes.
- **Speed reference:** 30,000 training tokens/s = 1.0×.
- **Try-it:** 10 messages per 10 minutes per asker per AI; owner sees counts only.
- **Story-battle votes:** 5 real minutes, or until every other online player has voted; a tie or no voters falls back to the judge.
- **World chat:** 1–300 characters, at most 5 per 10 s per player, the last 200 kept, the notebook filter.
- **Offline copy:**
  - "Its maker's computer is offline, so it can't answer right now."
  - "Its maker's computer is offline, so it can't battle right now."
  - "Connect your computer to do this."
- **Plain words (R7):** "helper", "your computer", "graphics card", "Fair mode", "card", "pool"; never "worker", "lease", "job", "GPU" in player copy ("graphics card" instead).
- **Tests:** `npm test` and `npm run check` are green after every task; ML tasks also need the ML suite and ruff green; `npm run e2e` (single player) and the new `npm run e2e:friends` are green at the end.

## Review Focus

1. **The host's card serves only the host's company with Fair mode off.** A friend's queued training must wait for the friend's card even while the house card is idle. Owner: Task 6, test "a friend's job never goes to the house card with Fair mode off".
2. **A card that leaves mid-job:**
   - Fair off: the job waits, and the same owner's card resumes it locally.
   - Fair on: another card continues from the last uploaded progress.
   - Nothing is lost or counted twice.
   - Owners: Tasks 6 and 13.
3. **A player with no card at all.** They can join and found a company; training says "Connect your computer to do this."; their customers' questions are dropped after 14 days, not stacked up. Owners: Tasks 7, 8 and 16.
4. **A stale tunnel link.** The helper never keeps the link; "Connect your computer" always passes today's link, and a helper started from an old shortcut asks for it. Owners: Tasks 13 and 14.
5. **Opacity with two players.** No other player's cash, plan, notebook, lessons, helper key or blob URL in any view, query, stream or chat message. The house card's token never reaches a client. Owners: Tasks 9, 10 and 18.

---

## Part A: the game rules (sim-7)

### Task 1: sim-7: Fair mode, the founder grant and autopilot

**Files:**
- Modify:
  - `packages/sim/src/world.ts` (`createWorld({..., fairMode})`)
  - `packages/sim/src/state.ts` (`WorldState.fairMode: boolean`)
  - `packages/sim/src/reducer.ts` (`createCompany` on day > 0)
  - `packages/sim/src/campaigns.ts`, `packages/sim/src/rescue.ts`, `packages/sim/src/upgrade.ts`, `packages/sim/src/project.ts`
  - `packages/protocol/src/view.ts` (`world.fairMode`)
  - `apps/server/src/config.ts` (`SIM_VERSION`)
- Test: `packages/sim/test/friends.test.ts`, `packages/sim/test/upgrade.test.ts`, `apps/server/test/actor.test.ts`

**Interfaces:**
- Produces:
  - `WorldState.fairMode`; `projectView(...).world.fairMode`.
  - `foundingCashOn(b: Balance, day: number): number` (exported from `world.ts`).
  - **Autopilot** (a company whose owner `members[owner].online === false`):
    - `campaignsDay` skips its campaign spend for the day when its cash is below 0 (the campaign stays, paused).
    - `rescueDay` moves its `rescue.deadlineDay` one day later instead of taking the loan.

- [ ] **Step 1: Write the failing tests:**
  - `createWorld({fairMode: true})` shows `world.fairMode === true`.
  - A company founded on day 30 has `250000 + 60000` cash and `startCash` equal; on day 200, `500000`.
  - An offline owner's campaign spends nothing on a day its cash is negative, and spends again once they're back.
  - An offline owner's rescue deadline doesn't pass; online, the loan is taken on the deadline as today.
  - A sim-6 world upgrades with `fairMode: false` and replays.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): sim-7: Fair mode per world, founder grants and autopilot`.

### Task 2: Battles against friends' AIs

**Files:**
- Modify: `packages/sim/src/battles.ts`, `packages/sim/src/state.ts` (`BattleState.opponent`), `packages/sim/src/project.ts`, `packages/protocol/src/world.ts` (`startBattle.rivalProductId` keeps its name; any company's live product), `packages/protocol/src/view.ts`
- Test: `packages/sim/test/battles.test.ts`

**Interfaces:**
- Produces:
  - `startBattle` accepts a live product of **any other company** (rival or player). Refusals:
    - "That AI isn't on sale."
    - "You can't battle your own AI."
    - The cooldown is per opponent company ("You battled {Company} recently. Try again on {date}.").
  - `BattleState.opponent: {companyId; kind: "rival" | "player"}`. For a player opponent, `theirs.packKey` is `null` and `theirs.versionId` is their version.
  - `battleResolved` applies to both companies:
    - The defender gets a mirrored `BattleState` (`defending: true`, mine and theirs swapped) in its `battles`.
    - Ranking points move for both.
    - Each gets its own news.
  - The defender pays no fee.

- [ ] **Step 1: Write the failing tests:**
  - Two player companies: A battles B's live product.
  - The result mirrors into B's battles with the scores swapped, and Elo moves both ways.
  - Battling your own product is refused.
  - Rival battles are unchanged (existing tests still pass).
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): battle a friend's AI; both companies see the result`.

---

## Part B: helpers, cards and routing (server and ML)

### Task 3: Helper keys and card capabilities

**Files:**
- Create: migration `0009_helpers.sql` (via `npm run db:generate -w @airace/server`), `apps/server/src/helpers/keys.ts`
- Modify:
  - `apps/server/src/db/schema.ts` (`helperKeys`: `userId uuid primary key`, `keyHash text unique`, `createdAt`)
  - `apps/server/src/jobs/workerGateway.ts` (bind a connection to a user, or to the house)
  - `apps/server/src/jobs/broker.ts` (`WorkerConn.ownerUserId: string | null`, null = house)
  - `packages/protocol/src/worker.ts` (`WorkerCapabilities.gpuName: string` default `"unknown"`, `speed: number` default `0`)
  - `packages/protocol/fixtures/worker/whello.json`
- Test: `apps/server/test/helpers.test.ts`, `packages/protocol/test/worker.test.ts`

**Interfaces:**
- Produces:
  - `helperKeyFor(ctx, userId): Promise<string>`: creates a key once (32 random bytes, base64url) and stores only its sha256. Reading it again needs the stored key, so the plaintext is returned on creation **and** on a "new key" request (`rotate: true`, which replaces the old one).
  - `GET /api/helper/key` (signed-in): `{key}`, creating it on first call. `POST /api/helper/key/rotate`.
  - Gateway hello:
    - the server's worker token → `ownerUserId: null` (house card)
    - a helper key → its user
    - anything else → close 4401

- [ ] **Step 1: Write the failing tests:**
  - A helper key connects as its user; a rotated key stops working; the house token still connects as house.
  - An older `whello` without `gpuName`/`speed` parses with the defaults.
  - The key's plaintext is never stored (the table holds a hash).
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): personal helper keys bind a graphics card to its player`.

### Task 4: The helper measures and names its card

**Files:**
- Modify: `ml/src/airace_ml/worker/agent.py` (capabilities), `ml/src/airace_ml/worker/protocol.py`, `ml/src/airace_ml/cli.py` (`worker --device auto`; `--server` may be an `airace-helper://connect?...` link)
- Create: `ml/src/airace_ml/worker/speed.py`
- Test: `ml/tests/test_worker_speed.py`, `ml/tests/test_worker_protocol.py`

**Interfaces:**
- Produces:
  - `measure_speed(device: torch.device, seconds: float = 5.0) -> float`: training tokens per second on the reference shape (2 layers of 64, context 64, batch 8192 tokens, random tokens), cached in `<data_root>/helper/speed.json` keyed by the device name.
  - `gpu_name(device) -> str`: `torch.cuda.get_device_name` or `"CPU"`.
  - `parse_connect_link(link: str) -> tuple[str, str]`: `airace-helper://connect?url=<https://x>&key=<k>` → `("wss://x/worker", k)`. Plain `ws(s)://` URLs pass through.
  - `--device auto`: cuda when available, else cpu.

- [ ] **Step 1: Write the failing tests:**
  - `measure_speed` on the CPU returns a positive finite number and caches it.
  - `parse_connect_link` gives wss for an https link, refuses other schemes, and keeps the key.
  - The hello carries `gpuName` and `speed`.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the ML suite and ruff: PASS.
- [ ] **Step 5: Commit** `feat(ml): the helper measures its card and connects from a game link`.

### Task 5: Cards in the world

**Files:**
- Create: `apps/server/src/helpers/cards.ts`
- Modify: `packages/protocol/src/wire.ts` (Query `cards {worldId}`), `apps/server/src/app.ts`, `apps/server/src/jobs/broker.ts` (connect and disconnect events)
- Test: `apps/server/test/cards.test.ts`

**Interfaces:**
- Produces:
  - `CardView = {userId, playerName, gpuName, vramMb, speed (× of 30,000), online: boolean, house: boolean}`.
  - Query `cards {worldId}` (members only): one entry per member, with the house card shown as the world host's. A member without a card has an entry with `online: false` and `gpuName: null`.
  - Stream `cards:<worldId>` pushes the list when a member's card connects or leaves.
  - Never the key, worker id or IP.

- [ ] **Step 1: Write the failing tests:**
  - Two members: A's helper connects → B's stream gets A's card online; it disconnects → offline.
  - A non-member's query is refused.
  - The JSON never contains the key.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): every player sees the world's graphics cards`.

### Task 6: Job owners, needs and routing by Fair mode

**Files:**
- Create: migration `0010_job_owners.sql`
- Modify:
  - `apps/server/src/db/schema.ts` (`jobs.ownerUserId text`, `jobs.minParams integer not null default 0`, `jobs.companyId text`)
  - `apps/server/src/jobs/broker.ts` (`enqueue(type, payload, worldId, {priority?, owner?, companyId?, minParams?})`; dispatch)
  - every enqueue site: `training/service.ts`, `chat/service.ts`, `feedback/service.ts`, `battles/service.ts`, `minigames/service.ts`, `keynotes/service.ts`, `apps/service.ts`, `interviews/service.ts`, `requests.ts`
- Create: `apps/server/src/jobs/fairShare.ts`
- Test: `apps/server/test/routing.test.ts`, `apps/server/test/broker.test.ts`

**Interfaces:**
- Consumes: Task 3's `WorkerConn.ownerUserId`; Task 1's `fairMode`.
- Produces:
  - **Owners** per spec §2.2's table. `minParams` is the model's parameter count (`paramCount(nLayer, dModel)` from the sim) for train, eval, generate, battle and minigame; 0 for interview.
  - **`canRun(job, conn, world): boolean`:**
    - a world-less job: any card
    - Fair off: `conn.ownerUserId === job.ownerUserId`, or the house card when `job.ownerUserId === world.hostUserId`
    - Fair on: the card's owner is a member of the job's world, or it is the house card
    - always `conn.caps.maxParams >= job.minParams`
  - **Dispatch order:**
    - Fair off: as today (priority, then seq) among jobs the card can run.
    - Fair on: among a world's runnable jobs, the company with the least `usageMs` goes first, then priority, then seq. A free card takes the job only if no faster free card can run it.
  - **`FairShare`:** `add(worldId, companyId, ms)`; `usage(worldId, companyId, day)` decays by half every 15 game days; in memory (restarts start even).

- [ ] **Step 1: Write the failing tests:**
  - Fair off: a friend's train job stays queued while only the house card is connected; it leases as soon as the friend's helper connects.
  - Fair on: the house card takes the friend's job.
  - Fair on, two companies: the one that used less card time is served first.
  - A job too big for a card waits for a bigger one.
  - Single-player worlds lease exactly as before (existing broker tests pass).
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): jobs run on their owner's card, or on a fair shared pool`.

### Task 7: Training limits, real-time estimates and runs that move

**Files:**
- Modify:
  - `apps/server/src/realtime/gateway.ts` (a command guard registry: `setCommandGuard(type, (userId, worldId, cmd) => Promise<string | null>)`, checked before `actor.submit`)
  - `apps/server/src/training/service.ts` (guard for `startTraining`; resume URLs)
  - `packages/protocol/src/worker.ts` (`TrainJobSpec.resumeUrl: Url | null` default null, `resumeFromUrl: Url | null` default null)
  - `ml/src/airace_ml/worker/jobs.py` (`handle_train` uploads `resume/` every 120 s when `resumeUrl` is set, and restores from `resumeFromUrl`)
- Test: `apps/server/test/training.test.ts`, `ml/tests/test_worker_jobs.py`

**Interfaces:**
- Produces:
  - **Guard:**
    - Fair off: refuse when no card of the player is online ("Connect your computer to do this.") or when the brain's parameters exceed their best card's `maxParams` ("Your graphics card can train up to the {Brain} brain.").
    - Fair on: the same against the pool's biggest card.
    - A refused command is never logged.
  - **Query `trainingLimit {worldId}`:** `{maxBrain: string | null, tokensPerSecond: number}` (the player's card or the pool, by mode). The Lab estimates real minutes as `tokens / tokensPerSecond / 60`.
  - **Moving runs (Fair on only):**
    - The worker tars `out/resume` and PUTs it every 120 s.
    - A re-lease of the same run passes `resumeFromUrl`, and the worker unpacks it before `train_run(resume=True)`.
    - The blob key is `resume/<runId>.tar`, deleted when the run completes.

- [ ] **Step 1: Write the failing tests:**
  - The guard refuses a too-big brain (Fair off) with the exact copy and lets a fitting one through.
  - With no card online, training is refused.
  - Fair on: a run whose first card vanished re-leases with `resumeFromUrl`.
  - ML: a run interrupted after its first upload resumes from the tar to the same final checkpoint as an uninterrupted run (same seed).
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the ML suite, ruff and `npm test`: PASS.
- [ ] **Step 5: Commit** `feat: training fits the player's card, shows real minutes, and moves between cards in Fair mode`.

### Task 8: Customer questions don't pile up while a card is off

**Files:**
- Modify: `apps/server/src/feedback/service.ts`, `apps/server/src/apps/service.ts` (payloads gain `enqueuedDay`), `apps/server/src/jobs/broker.ts` (a prepare hook may return `"drop"`)
- Test: `apps/server/test/feedback.test.ts`

**Interfaces:**
- Produces: at lease time, a feedback or appReview job whose `enqueuedDay` is more than 14 game days behind its world's day is cancelled (status `cancelled`, error "stale"). No input reaches the world; the service's request record is marked done, so it is never asked again.

- [ ] **Step 1: Write the failing test:** with the owner's card offline for 20 game days, then back: no feedback batch from the first week is applied, and the next week's batch is.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): unanswered customer questions expire instead of piling up`.

---

## Part C: playing together (server)

### Task 9: Joining friends

**Files:**
- Modify: `apps/server/src/worlds/routes.ts` (`POST /api/worlds` takes `fairMode: boolean` default false; `POST /api/worlds/join {code}`), `apps/client/src/net/api.ts`
- Test: `apps/server/test/join.test.ts`

**Interfaces:**
- Produces:
  - `api.createWorld(name, goal?, fairMode?)`; `api.joinWorld(code): Promise<string>`.
  - Join is idempotent; it adds a membership and submits `memberJoined`.
  - Refusals:
    - "That invite code doesn't match a world."
    - "This world is full (8 players)."
  - Joining a running world is allowed (the founder grant is Task 1's).

- [ ] **Step 1: Write the failing tests:**
  - B joins by code and sees A's company without its cash or runs.
  - A goes offline → B is the effective host and can pause.
  - Both offline → no ticks; B returns → the day continues.
  - The 9th human is refused.
  - A world created with `fairMode: true` shows it.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): join a friend's world by invite, with Fair mode chosen at creation`.

### Task 10: The AI Directory's Try it

**Files:**
- Modify: `apps/server/src/chat/service.ts` (friends' shared versions; owner = maker; offline refusal; limits), `packages/protocol/src/wire.ts` (Query `versionTries {worldId}`)
- Test: `apps/server/test/directory.test.ts`

**Interfaces:**
- Consumes: the sim's `publishVersion` and `PublicCompany.publishedVersions` (exist).
- Produces:
  - `chat` to another company's **published** version is allowed. Its thread is the asker's; the job's owner is the maker.
  - Refused when:
    - unpublished: "That AI isn't shared."
    - over the limit: "You've tried this AI a lot. Take a short break."
    - Fair off and the maker's card is offline: the offline copy
  - Query `versionTries {worldId}`: `{versionId, tries7d}` for your own published versions; counts only.

- [ ] **Step 1: Write the failing tests:**
  - B chats with A's published v1: the generate job's owner is A, and the reply streams to B only.
  - A's `chatThreads` lacks B's thread; A's `versionTries` is 1.
  - After A unshares, B's next message is refused.
  - With A's helper offline (Fair off), B gets the offline copy.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): try a friend's shared AI, answered by its maker's card`.

### Task 11: Two-sided battles and story votes

**Files:**
- Modify:
  - `apps/server/src/battles/service.ts` (friend battles as two side jobs; votes)
  - `packages/protocol/src/worker.ts` (`BattleJobSpec.sides: ("a" | "b")[]`, default `["a", "b"]`)
  - `ml/src/airace_ml/worker/jobs.py` (`handle_battle` answers only the listed sides)
  - `packages/protocol/src/wire.ts` (client message `battleVote {cid, worldId, battleId, round, pick: "left" | "right"}`; stream `battlevote:<worldId>`)
- Test: `apps/server/test/battles.test.ts`, `ml/tests/test_worker_jobs.py`

**Interfaces:**
- Consumes: Task 2's player battles, Task 6's owners.
- Produces:
  - A friend battle enqueues side "a" (owner: the challenger) and side "b" (owner: the defender), with the same challenge seed.
  - The server joins their rounds. With Fair off, if the defender's card is offline, the battle is called off at once (fee refunded) with the offline copy as its reason.
  - **Story votes:**
    - The other online members get a `battlevote` stream entry with the two stories in a random left/right order (seeded by battle id), unlabelled.
    - Votes are kept server-side.
    - Voting closes after 300,000 ms of real time (the scheduler), or when every other online member has voted.
    - A round's winner is the story with more votes; a tie or no votes falls back to the judge's scores.
    - Only totals are published.

- [ ] **Step 1: Write the failing tests:**
  - A friend battle produces two side jobs with different owners and one `battleResolved`.
  - A story round with votes 2–1 goes to the voted story, whatever the judge said.
  - With no voters, the judge decides.
  - A player in the battle can't vote ("You can't vote on your own battle.").
  - ML: `sides: ["b"]` answers only side b.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the ML suite, ruff and `npm test`: PASS.
- [ ] **Step 5: Commit** `feat: battles between friends' AIs, each side on its own card, story rounds voted by the others`.

### Task 12: World chat

**Files:**
- Create: `apps/server/src/worldchat/service.ts`, migration `0011_world_chat.sql`
- Modify: `apps/server/src/notebook/filter.ts` (export `textProblem(text): string | null` for links, phones, `<|` and blocked words, shared with the notebook), `packages/protocol/src/wire.ts` (client message `worldChat {cid, worldId, text}`; Query `worldChat {worldId}`; stream `worldchat:<worldId>`)
- Test: `apps/server/test/worldchat.test.ts`

**Interfaces:**
- Produces:
  - Table `world_chat (id serial, world_id uuid, user_id uuid, name text, text text, at bigint)`, the last 200 per world.
  - Members only.
  - Refusals:
    - "Messages are 1–300 characters."
    - the filter's messages
    - "Slow down a little." (more than 5 in 10 s)
  - The stream pushes `{name, companyId | null, text, at}`.

- [ ] **Step 1: Write the failing tests:**
  - A posts → B's stream gets it; a non-member can't read.
  - A link is refused with the filter's copy.
  - The sixth message in 10 s is refused.
  - Only 200 are kept.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): a world chat for the players`.

### Task 13: Hosting from your PC and the helper downloads

**Files:**
- Create: `tools/host.ts`, `docs/hosting.md`, `apps/server/src/helpers/downloads.ts`, `tools/helper/setup.ps1`, `tools/helper/setup.cmd.template`
- Modify: `apps/server/src/app.ts` (serve `apps/client/dist` with `@fastify/static` and an SPA fallback for `/`, `/join/*`, `/w/*`), root `package.json` (`"host": "tsx tools/host.ts"`)
- Test: `apps/server/test/static.test.ts`, `apps/server/test/downloads.test.ts`

**Interfaces:**
- Produces:
  - **`npm run host`:**
    1. Builds the client.
    2. Writes the catalog if missing or old (as `tools/dev.ts`).
    3. Zips `ml/` (src, pyproject, uv.lock) into `data/helper/ml.zip`.
    4. Writes the data manifest.
    5. Starts the server (serving the client) and the house card.
    6. If `cloudflared` is on PATH, runs `cloudflared tunnel --url http://localhost:8787`, reads the `https://…trycloudflare.com` link, and restarts the server with `publicUrl` set to it; otherwise prints the LAN link and `winget install Cloudflare.cloudflared`.
    7. Never downloads anything itself.
  - **Downloads** (signed-in, or a helper key in `Authorization: Bearer <key>`):
    - `GET /helper/setup.cmd`: the template filled with `publicUrl`, the player's key and the setup script.
    - `GET /helper/ml.zip`.
    - `GET /helper/data/manifest.json`: `{dataIdentity, files: [{path, size, sha256}]}` for `corpus/`, `tokenizer/`, `judge/`, `novelty/` and `rivals/`.
    - `GET /helper/data/<path>` with HTTP range support.
  - **`setup.ps1`** (run by `setup.cmd` with `-ExecutionPolicy Bypass`):
    1. Installs uv if missing (the official installer).
    2. Unzips `ml.zip` to `%LOCALAPPDATA%\AIRaceHelper\ml` and runs `uv sync`.
    3. Runs `uv run airace-ml fetch-data` (Task 14).
    4. Registers `airace-helper://` under `HKCU\Software\Classes` to run `uv run airace-ml worker --server "%1" --data-root ... --device auto`.
    5. Adds a Start-menu shortcut that asks for today's link.
    6. Prints each step in plain words.

- [ ] **Step 1: Write the failing tests:**
  - With a built `dist` fixture, `GET /` and `GET /join/ABC123` return `index.html`, and `/api/health` still works.
  - `GET /helper/setup.cmd` without a session or key is 401; with a session it contains the player's key and the public URL.
  - The manifest lists each data file with a sha256 matching the file.
  - A range request returns the right bytes.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS, then a manual `npm run host` with and without cloudflared.
- [ ] **Step 5: Commit** `feat: host from your PC with a quick tunnel, and download the helper from the game`.

### Task 14: `airace-ml fetch-data`

**Files:**
- Create: `ml/src/airace_ml/worker/fetch.py`
- Modify: `ml/src/airace_ml/cli.py`
- Test: `ml/tests/test_fetch_data.py` (against a local HTTP server fixture)

**Interfaces:**
- Produces: `airace-ml fetch-data --server <https URL or airace-helper link> --key K --data-root DIR`:
  - downloads every manifest file whose sha256 differs (resuming partial files by range), and verifies each
  - checks `data_identity(DIR) == manifest.dataIdentity`
  - prints progress in plain words
  - exit 0 when complete; exit 3 with "The game's data changed; run the helper setup again." on an identity mismatch

- [ ] **Step 1: Write the failing tests:**
  - A fresh folder downloads everything and matches the identity.
  - A second run downloads nothing.
  - A truncated file resumes.
  - A wrong sha256 is re-downloaded.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the ML suite and ruff: PASS.
- [ ] **Step 5: Commit** `feat(ml): the helper fetches the game's data from the host and checks it`.

---

## Part D: the screens

### Task 15: Starting and joining worlds

**Files:**
- Create: `apps/client/src/screens/Join.tsx`
- Modify: `apps/client/src/screens/Home.tsx` (Fair mode choice with the notice; "Join with a code"), `apps/client/src/screens/Lobby.tsx` (members, online dots, cards, "Invite friends": code and copy-link), `apps/client/src/route.ts` (`/join/<code>`)
- Test: `apps/client/test/join.test.tsx`, `apps/client/test/home.test.tsx`

**Interfaces:**
- Consumes: Task 9's API, Task 5's cards.
- Produces:
  - Home's "Start a new world" has a **"Fair mode"** switch, showing the notice when on, and sends `fairMode`.
  - `/join/<code>` asks a guest for a name first, then joins. A Fair world shows the notice before joining, with "Join" and "Back".

- [ ] **Step 1: Write the failing tests:**
  - The switch sends `fairMode: true` and shows the notice text exactly.
  - The join link signs up a new guest, then posts `/api/worlds/join`.
  - A full world shows its refusal.
  - The lobby lists both members and copies the link.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): start a world with Fair mode, join friends by link`.

### Task 16: "Connect your computer" and the cards

**Files:**
- Create: `apps/client/src/cards/Cards.tsx` (a top-bar chip, a Cards window and the Connect panel), `apps/client/src/store/cards.ts`
- Modify: `apps/client/src/components/TopBar.tsx`, `apps/client/src/screens/lab/DesignStep.tsx` (brain sizes over the limit are disabled with the reason), `apps/client/src/screens/lab/LabLayout.tsx` (real-minute estimate), `apps/client/src/net/api.ts` (`helperKey()`)
- Test: `apps/client/test/cards.test.tsx`

**Interfaces:**
- Consumes: Tasks 3, 5 and 7.
- Produces:
  - **The top-bar chip:** "Your computer: RTX 3050 · online", or "Connect your computer"; it opens the Cards window.
  - **The Cards window:**
    - every player's card(s): name, memory, speed, online
    - in Fair mode, "Shared pool: N cards"
    - your own row's buttons: "Get the helper" (downloads `/helper/setup.cmd`) and "Connect your computer" (opens `airace-helper://connect?url=<location.origin>&key=<key>`)
  - **The Lab:**
    - "Your graphics card can train up to the {Brain} brain." (or "The pool can …")
    - "About {n} minutes on your card." (or "… in the pool.")
    - "Connect your computer to do this." on Start when no card

- [ ] **Step 1: Write the failing tests:**
  - With no card, the chip says "Connect your computer" and Start training is disabled with the copy.
  - With a card limit of small, the larger brains are disabled.
  - The connect link carries the page's origin and the key.
  - Another player's card shows name and speed, never a key.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): connect your computer, see everyone's cards, train within your card's limits`.

### Task 17: The AI Directory

**Files:**
- Create: `apps/client/src/screens/Directory.tsx`
- Modify: `apps/client/src/components/SideNav.tsx` ("AI Directory"), `apps/client/src/screens/lab/VersionsStep.tsx` ("Share with friends" and "Stop sharing", with the tries count), `apps/client/src/screens/Playground.tsx` (talk to a shared AI, with the line "Its maker's computer answers your messages.")
- Test: `apps/client/test/directory.test.tsx`

**Interfaces:**
- Produces:
  - Every company's published versions (logo, name, label, brain, scores, personality words) plus the rivals' AIs on sale, each with "Try it".
  - Offline makers' AIs show "Its maker's computer is offline" and a disabled Try it (Fair off).

- [ ] **Step 1: Write the failing tests:**
  - A friend's shared AI is listed and Try it opens the Playground on it.
  - Sharing sends `publishVersion`.
  - The owner sees "tried 3 times this week".
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): the AI Directory and sharing with friends`.

### Task 18: Friend battles and votes on screen

**Files:**
- Modify: `apps/client/src/screens/Arena.tsx` and `apps/client/src/windows/BattleWindows.tsx` (opponents: rivals and players; the defender's view), `apps/client/src/components/Toasts.tsx` ("Vote on a story battle")
- Create: `apps/client/src/windows/VoteWindow.tsx`
- Test: `apps/client/test/battles.test.tsx`

**Interfaces:**
- Consumes: Tasks 2 and 11.
- Produces:
  - The battle picker lists players' AIs on sale with the rivals' (offline makers disabled, Fair off).
  - A vote toast opens `VoteWindow`: two stories, left and right, unlabelled, "This one" under each, and a countdown. It sends `battleVote`.
  - Defenders see incoming battles in the Arena.
  - Story replies render as text (R1).

- [ ] **Step 1: Write the failing tests:**
  - Picking a friend's AI sends `startBattle` with their product.
  - The vote window shows both stories without names and sends the pick.
  - A battle you defended shows in your Arena with your side first.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): battle friends and vote on their story battles`.

### Task 19: World chat on screen

**Files:**
- Create: `apps/client/src/components/WorldChat.tsx`
- Modify: `apps/client/src/components/AppShell.tsx`
- Test: `apps/client/test/worldchat.test.tsx`

**Interfaces:**
- Produces:
  - A collapsible panel (bottom left) with the last messages, each in its company's color, and a box.
  - Refusals show inline.
  - An unread count when collapsed.
  - Messages render as text.

- [ ] **Step 1: Write the failing tests:**
  - Sending posts `worldChat`.
  - A streamed message appears with its name.
  - The unread count rises while collapsed.
  - Markup renders as text.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): a world chat panel`.

---

## Part E: end to end

### Task 20: Two players, two helpers, both modes

**Files:**
- Create: `tests/e2e/friends.spec.ts`, `tests/e2e/friends.fixtures.ts` (a tiny data root, the server on `balance-test`, and **two** CPU helpers with two players' keys)
- Modify: root `package.json` (`"e2e:friends": "playwright test tests/e2e/friends.spec.ts"`), `CLAUDE.md` (R3 per spec §1; the new commands), `docs/progress.md`, `docs/handoff/README.md`, `docs/handoff/sdd/ledger-stage2-friends.md`

- [ ] **Step 1: Write the E2E test**, run once with Fair off and once with Fair on (two browser contexts):
  1. A creates the world; B joins by link. Both found companies, and both connect a helper.
  2. Both train at once; each run is leased to its owner's helper (Fair off), or to either (Fair on); both live views progress.
  3. A shares v1; B tries it and gets a reply.
  4. A friend battle in knowledge, then a story round voted by a third context.
  5. World chat both ways.
  6. **Opacity:** B's page, its `WorldView` JSON and every network response never contain A's cash, plan, notebook, helper key or any blob URL.
  7. B's helper stops (Fair off): B's next training waits; B's customers' old questions are dropped after 14 days.
  8. A server restart: both reconnect; days, versions and cash are preserved.
- [ ] **Step 2:** Run `npm run e2e:friends`: PASS; `npm run e2e`: PASS.
- [ ] **Step 3: Run everything:** `npm test`, `npm run check`, the ML suite with ruff.
- [ ] **Step 4: A real session:** the owner and one friend, each on their own PC, through `npm run host`'s link. Fair off first, then a Fair world. Ledger what happened.
- [ ] **Step 5: Final review** of the branch (self-review, per the owner's preference), one fix pass for Critical and Important findings, each with a failing test first.
- [ ] **Step 6: Docs and report:**
  - CLAUDE.md's R3 line becomes: "R3 Fair compute: in-game compute sets every run's game time and cost; real work runs on the players' own helper cards (Fair mode pools them equally); never in the browser."
  - The progress entry "Milestone 2 complete".
  - The handoff resume point: the coaching plan.
  - Commit, push and report to the owner, with the rulings and the deferred minors.

## Decisions made in this plan (for the owner report)

1. **The house card** (the PC running the world) counts as each world's host's card, so single-player worlds behave exactly as today. If wrong: a "use this PC's card" setting.
2. **Fair-mode share** is kept in memory, so it starts even after a restart. If wrong: a small table.
3. **A friend battle with an offline defender** (Fair off) is called off at once with the fee refunded, rather than waiting. If wrong: a waiting state.
4. **The helper never stores the game link** (quick-tunnel links change every session), so the link arrives with "Connect your computer". If wrong: a named tunnel later.
5. **Commands over a card's limit are refused before they are logged**, because cards are real-time facts outside the deterministic world. If wrong: none for play.
