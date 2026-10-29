# Game Layer Phase 3 "Rivals" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The player has competition. Every world gets an arch-rival (a surprise type each game) and three rivals:
- Their AIs are real models, built once on this PC.
- Over the game year they launch those AIs, set prices, advertise and build data centers, each in its own style.
- The arch-rival trash-talks in the news.

The player can:
- try any rival AI that is on sale
- battle it skill by skill, with both replies streaming side by side
- climb league tables and earn ranking points

**Architecture:**
- **ML:** `airace-ml build-rivals` trains a rival pack once: 8 lineages, 3 versions each, about 75–90 minutes on the RTX 3050, and resumable. It measures each version (scores, traits, personality words, tag breakdown) and writes its answers to the customer questions ahead of time. The worker gains a `battle` job.
- **Sim (`sim-4`):** rivals are ordinary companies with a `rival` field and no owner, so the market, products, fame, compute and ledger already treat them as competitors.
  - New modules: `rivalCast` (names, palettes, lines), `rivals` (joining, releases, business habits, trash-talk), `league` and `battles` (with ranking points).
  - One logged input, `rivalsJoined`, carries the pack's measurements. Releases then happen on game days, deterministically.
- **Server:**
  - A **rival service** loads the pack and seats rivals with `rivalsJoined`. It answers rival feedback from the stored replies and places rival models in the blob store for chats and battles.
  - A **battle service** runs and streams battles.
  - **Chat** accepts rivals' AIs that are on sale, at the lowest priority.
- **Client:** Rivals and Arena screens; rival colors on the map; battle windows that stream both replies; rival news and trash-talk in the feed.

**Tech Stack:** As Phase 2 (TypeScript, zod, Vitest, fast-check, Fastify, PGlite, React 19, Vite, Tailwind 4, Zustand, motion), plus Python 3.13 with uv (torch) for the pack and the battle job.

**Spec:** `docs/superpowers/specs/2026-10-27-game-layer-design.md`:
- §9 (rivals), §10 (battles), §12 (hardware priority), §13–§15 (Phase 3 row)
- §8's report-card rank changes, deferred from Phase 2

Also `docs/superpowers/specs/2026-10-28-immersion-design.md`. R1–R8 always apply.

## Before you start (read once)

- **The owner's world carries over.**
  - Task 4 adds a bridge from sim-3 to sim-4. The Phase 2 world the owner is playing keeps going.
  - Rivals join it on its next load once the pack exists ("New rivals enter the AI race").
  - Tell the owner this when you start.
- **The real pack needs about two quiet hours on the graphics card** (Task 16):
  - Ask the owner for a time when they are not playing.
  - Keep a Monitor heartbeat of a few minutes on the build so the PC doesn't sleep (CLAUDE.md).
  - Everything before Task 16 uses quick packs and fixtures.

## Global Constraints

- **Versions:**
  - `SIM_VERSION = "sim-4"`. Worlds saved by `sim-3` are upgraded on load by `upgradeState` (Task 4), then saved as sim-4 at once. Any older version is refused as today.
  - Balance ids stay `"balance-v2"` and `"balance-test"`.
- **Pack:**
  - Version `"rivals-v1"`, at `data/rivals/v1/`, holding:
    - `rivals.json` (the index)
    - `models/<key>.tar` (a checkpoint tar exactly like a player version's blob)
    - `replies/<key>.json`
    - `measured/` and `work/` (resume state)
  - Version keys are `<lineageId>-v<k>` with k from 1.
  - 8 lineages, exactly 2 per style.
- **Ids** (all valid `zId`):
  - rival companies `rival-1`…`rival-4` (`rival-1` is the arch-rival)
  - their versions `<companyId>-v<k>`, products `<companyId>-p<k>`, campaigns `<companyId>-c<n>`, data centers `<companyId>-d<n>`
  - a battle takes the command's `newId`
- **Styles and archetypes:**
  - `RIVAL_STYLES = ["premium","budget","specialist","consumer"] as const`
  - `ARCHETYPES = ["giant","upstart","lab"] as const`
  - Words (R7):
    - archetypes: "an arrogant tech giant", "a cheeky upstart", "a mysterious lab"
    - styles: premium "Premium: big, pricey AIs", budget "Budget: cheap and quick", specialist "Specialist: great at one thing", consumer "Crowd-pleaser: an AI for everyone"
  - The arch-rival's style follows its archetype: giant→premium, upstart→consumer, lab→specialist.
- **Battle skills:** `BATTLE_SKILLS = ["knowledge","coding","instruction","pattern","creativity"] as const`. These are the skills whose replies are written text that a check or the judge can score.
- **Job priority** (lower runs first): `generate` 0 (the player's chat), `battle` 1, `train` 2, `eval` 3, `feedback` 4, and rival chats 5 (a `generate` job with a priority override).
- **News topics:** `NEWS_TOPICS` gains `"battle"`, `"ranking"` and `"trashTalk"`.
- **R1:**
  - Battle replies and rival posts quote real model output verbatim. Rival replies were generated at build time and are stored unchanged.
  - Trash-talk is game text. It is shown as a quote from the rival's founder, labelled as game commentary, never as an AI's words.
- **R2:**
  - Rivals are seen only through `PublicCompany` plus `rival: {style, archetype}`.
  - A rival's cash, ledger, campaigns, data centers, pack keys, lineage and upcoming releases never reach a view.
  - Revenue league rows show other companies' ranks, never their amounts.
- **R3/R5:** rival AIs never train during play. Their scores are real measurements. League places and ranking points come from measurements and battle results.
- **R7** (plain words): "rival", "arch-rival", "battle", "round", "league table", "ranking points" (for Elo), "Try it".
- **Immersion (C30):**
  - Every rival, battle and league row opens a window.
  - Rival launches, battle results and trash-talk slide into the feed.
  - Toasts appear only for stories that name the player, for the arch-rival's launches and for battle results.
  - Everything is instant under reduced motion.
- **Tests:**
  - `npm test` and `npm run check` are green after every task.
  - ML tasks also need `cd ml && python -m uv run pytest` and `python -m uv run ruff check .` green.
  - `npm run e2e` is green at the end.

## Review Focus

1. **No pack, a stale pack, a quick pack, a pack missing a style, a pack built after the world started:**
   - With no usable pack, the world runs without rivals and the Rivals screen says why in plain words. A later pack makes rivals join on the next load.
   - Owners: Tasks 4, 10 and 13.
2. **A battle whose job fails, whose rival product retires mid-battle, or that resolves after a server restart:** refunded exactly once or applied exactly once, with no point change on failure. Owners: Tasks 7 and 12.
3. **Rival finances over years:** never NaN, never negative customers, never more than 3 products on sale, cash below zero answered by an immediate loan, prices always inside the ranges. Owners: Tasks 5 and 9.
4. **Odd text in pack and battle replies** (markup, `<|end|>`, very long, empty): stored and shown verbatim as text; an empty reply scores 0. Owners: Tasks 3, 7 and 14.
5. **Rival internals in views:** a fast-check property over random worlds with rivals finds no pack key, lineage id, upcoming release, or rival cash or ledger in any view. Owner: Task 4 (extended in Tasks 5–7).

---

## Part A: the rival AIs (ML)

### Task 1: Battle challenges and single-story scoring

**Files:**
- Create: `ml/src/airace_ml/battles.py`
- Modify: `ml/src/airace_ml/evals/creativity.py` (extract `score_story`)
- Test: create `ml/tests/test_battles.py`; extend `ml/tests/test_creativity.py`

**Interfaces:**
- Produces:
  - `BATTLE_SKILLS = ("knowledge", "coding", "instruction", "pattern", "creativity")`.
  - `@dataclass(frozen=True) class StoryPrompt: id: str; prompt: str`, and `type Challenge = ExactItem | CheckItem | StoryPrompt`.
  - **`battle_challenges(skill: str, n: int, seed: int) -> list[Challenge]`:** `n` different challenges, the same ones for the same seed.
    - Checked skills come from the bench generators on `skill_rng(skill, "battles")`. Like `airace_ml.feedback`, only items a chat can ask are used, renamed `"<skill>-battle-<k:04d>"`.
    - Creativity comes from `story_prompts()`: "Write a short story about {subject} {situation}." over at least 20 subjects × 20 situations, on its own `skill_rng("creativity", "battles")` stream, never one of `STORY_PROMPTS`.
    - Unknown skills raise `ValueError`.
  - **`score_story(story: str, prompt: str, tok: Tok, judge: Judge, novelty: NoveltyIndex) -> float`:** the per-story score `score_creativity` computes today. `score_creativity` now calls it.
  - **`score_challenge(item: Challenge, reply: str, *, tok=None, judge=None, novelty=None) -> float`** (0..1): `score_reply` for checked items, `score_story` for stories. Raises `ValueError("creativity needs the judge and the novelty index")` without them.

- [ ] **Step 1: Write the failing tests:**
  - the same seed gives the same challenges; seeds 1 and 2 differ; ids are distinct
  - no challenge prompt equals a bench-v1 item prompt (`build_suite()`), a customer-pool prompt (`customer_pool`) or a `STORY_PROMPTS` entry
  - an `ExactItem` scores 1 with its accepted answer and 0 with `""`
  - `score_creativity` gives the same numbers as before on the existing fake judge test
  - an unknown skill raises; creativity without a judge raises
- [ ] **Step 2:** Run `cd ml && python -m uv run pytest tests/test_battles.py tests/test_creativity.py`: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the ML suite and ruff: PASS.
- [ ] **Step 5: Commit** `feat(ml): fresh battle challenges per skill, and scoring for one story`.

### Task 2: The rival pack format and `build-rivals`

**Files:**
- Create:
  - protocol: `packages/protocol/src/rivals.ts` (export it from `index.ts`) and `packages/protocol/fixtures/rivals/rivals.json` (a valid pack: 8 lineages × 2 versions, made-up numbers)
  - ML: `ml/src/airace_ml/rivals/__init__.py`, `ml/src/airace_ml/rivals/recipe.py` and `ml/src/airace_ml/rivals/build.py`
- Modify: `ml/src/airace_ml/paths.py` (`rivals_dir(root) = root / "rivals" / "v1"`), `ml/src/airace_ml/cli.py` (`build-rivals`), `CLAUDE.md` (the command, and "after a corpus rebuild, also `build-rivals`")
- Test: create `packages/protocol/test/rivals.test.ts` and `ml/tests/test_rivals.py`

**Interfaces:**
- Produces (protocol):
  ```ts
  export const RIVAL_STYLES = ["premium", "budget", "specialist", "consumer"] as const;
  export const ARCHETYPES = ["giant", "upstart", "lab"] as const;
  export const ARCHETYPE_WORDS: Record<Archetype, string>; // see Global Constraints
  export const STYLE_WORDS: Record<RivalStyle, string>;
  export const RivalPackVersion = z.strictObject({
    key: zId, nLayer: z.number().int().positive(), dModel: z.number().int().positive(),
    ctxLen: z.number().int().positive(), trainedTokens: z.number().int().min(0),
    scores: z.record(z.string(), z.number().nullable()), overall: z.number().nullable(),
    traits: z.record(z.string(), z.number()), personalityWords: z.array(z.string()),
    /** "<category>/<tag>" -> [mean item score x 100, item count] (BenchReport.tag_breakdown). */
    breakdown: z.record(z.string(), z.tuple([z.number(), z.number().int().min(0)])),
  });
  export const RivalPackLineage = z.strictObject({
    id: zId, style: z.enum(RIVAL_STYLES), focus: GroupId,
    versions: z.array(RivalPackVersion).min(1).max(5),
  });
  export const RivalPack = z.strictObject({
    version: z.literal("rivals-v1"), dataIdentity: z.string().min(1),
    tokenizer: z.literal("tok-v1"), suite: z.literal("bench-v1"), quick: z.boolean(),
    lineages: z.array(RivalPackLineage).length(8),
  }); // refined: exactly 2 lineages per style; version keys unique across the pack
  export const RivalReply = z.strictObject({
    group: GroupId, questionId: z.string().min(1), question: z.string(),
    reply: z.string(), score: z.number().min(0).max(1),
  });
  export const RivalReplies = z.array(RivalReply);
  ```
- Produces (ML):
  - **`recipe.py`:**
    - `VersionStep(shape: ModelShape, tokens: int)` and `LineageRecipe(id: str, style: str, focus: str, mix: dict[str, float], steps: tuple[VersionStep, ...])`.
    - `LINEAGES`: the 8 rows below, and `recipe_digest(recipe) -> str` (sha256 of its JSON).
    - Shapes are the balance's brain sizes at attention span 256: tiny `ModelShape(4,128,256)`, small `ModelShape(6,192,256)`, medium `ModelShape(8,256,256)`.
    - Each step adds `12_582_912` tokens. Mixes use `mix_heavy` from `airace_ml.experiments.configs` where named; "rest even" splits what remains over the other datasets.

    | id | style | focus | mix | steps |
    |----|-------|-------|-----|-------|
    | premium-a | premium | businesses | all eight even | small, medium, medium |
    | premium-b | premium | researchers | facts 0.35, educational 0.35, rest even | small, medium, medium |
    | budget-a | budget | everyday | web 0.4, conversations 0.4, rest even | tiny, tiny, small |
    | budget-b | budget | students | educational 0.4, conversations 0.3, rest even | tiny, tiny, small |
    | specialist-code | specialist | developers | `mix_heavy("code")` | tiny, small, medium |
    | specialist-stories | specialist | creators | `mix_heavy("creative")` | tiny, small, medium |
    | consumer-a | consumer | everyday | `mix_heavy("conversations")` | tiny, small, small |
    | consumer-b | consumer | creators | books 0.3, creative 0.3, conversations 0.3, rest even | tiny, small, small |

  - **`build_rival_pack(data_root: Path, out: Path, device: torch.device, *, quick: bool = False, on_status: Callable[[str], None] = print) -> Path`:**
    - **Training:**
      - Each lineage's versions train in order.
      - Version k continues from version k−1's checkpoint (`parent_dir`; growth when the shape grows, R6).
      - Seed = lineage index + 1; standard preparation (cleaning `standard`, no dedup, no fact-check).
      - A version is reused, and `"reused <key>"` reported, when `models/<key>.tar` and `measured/<key>.json` exist with the same recipe digest and data identity.
    - **Measuring each version:**
      - `run_benchmarks`, with the judge and novelty index when both are installed and valid; otherwise creativity is missing and a warning is printed once.
      - `measure_fingerprint` and `report.tag_breakdown()`.
      - Personality words come after every version is measured: `describe(fp, population)`, where the population is every other pack version plus the catalog's `referenceTraits` when `data/catalog/v1/catalog.json` exists.
    - **Replies:** for each group, the first `REPLIES_PER_GROUP = 24` questions of `customer_pool(group)` in pool order. Each reply is `lm.chat_reply([("user", q.prompt)], max_new_tokens=96, temperature=0.8, seed=k)` (k = its place in the pool), scored with `score_reply`. They are written to `replies/<key>.json`.
    - **Writing:** `rivals.json` is written last and atomically, so a half-built pack never loads.
    - **Quick build:**
      - shapes `QUICK_SHAPE`, then `QUICK_GROWN_SHAPE`; 2 versions per lineage of `QUICK_BUDGET` tokens each
      - bench limited to 8 items per category, fingerprint k=1, 2 replies per group
      - `"quick": true`
  - **CLI:** `airace-ml build-rivals [--data-root DIR] [--out DIR] [--quick] [--device cuda|cpu]`.
    - Like `build-judge`, it needs the GPU unless `--device cpu` or `--quick` is given.
    - It prints the expected time (about 75–90 minutes on the RTX 3050) and per-version progress.
    - Exit code 2 when the tokenizer or corpus is missing, with the existing plain messages.

- [ ] **Step 1: Write the failing tests:**
  - protocol:
    - the fixture parses
    - a pack with 3 premium lineages, 7 lineages, or a repeated version key is refused
  - ML (fast):
    - `LINEAGES` has 8 rows, 2 per style; shapes never shrink inside a lineage; keys are unique
    - the writer's JSON for faked measurements has exactly the fixture's field names at every level (read `packages/protocol/fixtures/rivals/rivals.json` and compare key sets)
  - ML (`@pytest.mark.slow`):
    - a quick build on the tiny corpus the slow gate test uses writes `rivals.json` with 8 × 2 versions
    - it writes a tar for each key and replies for 6 groups × 2 per key (non-empty `question`, string `reply`), plus a breakdown
    - a second run reports `reused` for all 16 and trains nothing
- [ ] **Step 2:** Run `npx vitest run packages/protocol` and `cd ml && python -m uv run pytest tests/test_rivals.py -m "slow or not slow"`: FAIL.
- [ ] **Step 3: Implement.** Use `make_tar` from `airace_ml.worker.blobs` for the model tars.
- [ ] **Step 4:** Run: PASS (including the slow test once), and ruff.
- [ ] **Step 5: Commit** `feat(ml): build-rivals trains, measures and pre-answers a pack of 8 rival lineages`.

### Task 3: The worker's battle job; rival models leave the cache first

**Files:**
- Modify:
  - protocol and fixtures: `packages/protocol/src/world.ts` (`BATTLE_SKILLS`, `BattleSkill`), `packages/protocol/src/worker.ts`; `packages/protocol/fixtures/worker/` (add `lease.battle.json`, `done.battle.json`, `progress.battle.json`)
  - ML worker: `ml/src/airace_ml/worker/jobs.py` (`handle_battle`), `ml/src/airace_ml/worker/models.py` (`transient`), `ml/src/airace_ml/worker/protocol.py`, `ml/src/airace_ml/cli.py` (worker slots: `battle` 1)
  - server: `apps/server/src/jobs/broker.ts`
- Test: `ml/tests/test_worker_jobs.py`, `ml/tests/test_worker_protocol.py`, `apps/server/test/broker.test.ts`, the protocol fixtures test

**Interfaces:**
- Produces:
  - **Protocol job types and slots:**
    - `JOB_TYPES = ["train","eval","generate","feedback","battle"]`.
    - `WorkerCapabilities.slots.battle: Slots.default(0)`, so an older worker gets no battles.
    - `GenerateJobSpec.versionId` becomes `zId`: rival chats name a pack key.
  - **`BattleJobSpec`:**
    - `battleId: zId`, `skill: z.enum(BATTLE_SKILLS)`, `rounds: int 1..5`, `seed: int ≥ 0`
    - `a: {versionId: zId, modelTarUrl: Url}`, `b: {versionId: zId, modelTarUrl: Url}` (`b` is always the rival)
  - **Progress event:** `{kind: "battleText", round: number, side: "a" | "b", text: string}`, where `text` is the newly generated piece.
  - **`BattleJobResult`:** `{rounds: [{prompt: string, a: {reply: string, score: 0..1}, b: {reply, score}}]}`.
  - **`JobRunner.handle_battle`:**
    - Challenges are `battle_challenges(skill, rounds, seed)`.
    - Each round streams side a, then side b, with `stream_chat_reply`: `max_new_tokens=96`, `seed=seed+round`, temperature 0.9 and top_p 0.95 for creativity, else 0.8 and 0.95.
    - Each reply is scored with `score_challenge` (creativity uses the worker's judge and novelty index).
    - The rival model is loaded with `transient=True`.
  - **`ModelCache.get(version_id, url, *, transient: bool = False)`:** transient models are evicted before any other, on the device and in CPU memory.
  - **Broker:**
    - `JOB_PRIORITY = {generate: 0, battle: 1, train: 2, eval: 3, feedback: 4}`.
    - `enqueue(type, payload, worldId, opts?: {priority?: number})` stores `opts.priority` when given.

- [ ] **Step 1: Write the failing tests:**
  - the battle fixtures round-trip in Python and parse in zod; a `whello` without a battle slot parses with 0
  - `handle_battle` with two tiny test models:
    - progress comes for a then b in every round
    - 3 rounds come back with scores in [0, 1]
    - an empty reply scores 0
    - creativity without a judge raises `ValueError`
  - at capacity 1, a transient model is evicted before a normal one
  - broker: queue `feedback`, then a priority-5 `generate`, then `battle`; one free worker leases battle, feedback, then the rival chat
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `npm test`, `npm run check`, the ML suite and ruff: PASS.
- [ ] **Step 5: Commit** `feat(worker): battle jobs stream both AIs' replies and score them; rival models leave the cache first`.

---

## Part B: the game rules (sim)

### Task 4: sim-4: a cast of rivals joins the world

**Files:**
- Create: `packages/sim/src/rivalCast.ts`, `packages/sim/src/rivals.ts`, `packages/sim/src/upgrade.ts`, `packages/sim/test/pack.ts` (`SYNTHETIC_PACK`, a valid `RivalPack` whose version scores grow per version and tilt by style), `packages/sim/test/rivals.test.ts`
- Modify:
  - protocol: `packages/protocol/src/world.ts` (the three news topics), `packages/protocol/src/view.ts`
  - sim: `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts` (system `rivalsJoined`; `createCompany` sets `startCash`), `packages/sim/src/world.ts`, `packages/sim/src/balance.ts` (rivals block), `packages/sim/src/project.ts`, `packages/sim/src/index.ts`, `packages/sim/test/arbitraries.ts` (worlds that may get rivals on day 1), the money identity in `properties.test.ts` (per-company `startCash`)
  - server: `apps/server/src/config.ts` (`SIM_VERSION`), `apps/server/src/worlds/actor.ts` (upgrade a sim-3 snapshot, then snapshot at once)
- Test: `rivals.test.ts`, `properties.test.ts`, `project.test.ts`, `apps/server/test/actor.test.ts`

**Interfaces:**
- Consumes: Task 2's `RivalPack`.
- Produces:
  - **State:**
    - `WorldState.rivals: {joinedDay: number; archId: string} | null` (null until rivals join)
    - `CompanyState.ownerUserId: string | null`
    - `CompanyState.startCash: number`
    - `CompanyState.rival: RivalState | null`
    - `VersionState.packKey: string | null` (rival versions use `runId: "pack"`)
    - `RivalState = {lineageId: string; style: RivalStyle; focus: GroupId; archetype: Archetype | null; upcoming: PackRelease[]; released: number; lastReviewDay: number; lastTrashTalkDay: number | null; counters: {campaigns: number; dataCenters: number}}`
    - `PackRelease = RivalPackVersion & {releaseDay: number}` (the sim's copy of the pack's numbers)
  - **System event:** `{type: "rivalsJoined"; pack: RivalPack}`, handled by `joinRivals(s, pack): Handled` in `rivals.ts`:
    - **Refusals:** the world isn't running ("The world is not running yet."); `s.rivals !== null` ("Rivals have already joined."); `RivalPack.safeParse` fails ("The rival pack isn't valid.").
    - **Draws, from the world rng in this order:**
      1. the archetype
      2. the arch-rival's lineage, one of the two of its style
      3. the style left out (one of the four)
      4. for each kept style in `RIVAL_STYLES` order, one lineage of that style the arch-rival didn't take
      5. names, logos and palettes, per company
      6. release jitter, per version
    - **Companies:** ids `rival-1` (the arch-rival) to `rival-4`, `ownerUserId: null`, the usual new business, compute and ledger.
    - **Cash:** `startCash = foundingCash × startCash[style] × (arch ? archBonus : 1)`.
    - **Release days:** version k (from 0) of company i (0 = the arch-rival) releases on `joinedDay + firstRelease[i] + k × releaseGapDays + jitter`, with jitter in `[−jitterDays, jitterDays]`. Days are at least `joinedDay + 1` and strictly increasing within a company.
    - **News** (topic `trashTalk`, sentiment 0, about the arch-rival):
      - joining on day ≤ 7: `{Arch}, {archetype words}, enters the AI race`
      - joining later: `New rivals enter the AI race, led by {Arch}`
  - **Cast** (`rivalCast.ts`):
    - `RIVAL_NAME_FIRST`: 24 invented words, none a real AI company: Nova, Quanta, Lumen, Vertex, Cinder, Mosaic, Orbit, Pike, Harbor, Flux, Ember, Zephyr, Atlas, Kestrel, Juniper, Onyx, Prism, Tidal, Sable, Corvid, Halo, Ridge, Aurora, Basalt.
    - `RIVAL_NAME_ENDINGS: Record<Archetype | RivalStyle, string[]>`:
      - giant: Global, Dynamics, Systems
      - upstart: Labs, AI, Works
      - lab: Institute, Research, Foundry
      - premium: Intelligence, Systems, Analytics
      - budget: AI, Go, Mind
      - specialist: Labs, Logic, Studio
      - consumer: AI, Chat, Pal
    - A name is `${first} ${ending}` (the arch-rival uses its archetype's endings). It is unique case-insensitively and never a player company's name.
    - `RIVAL_PALETTES`: 12 palettes that each pass `paletteProblem`. No two companies in a world share a primary color.
    - `PRODUCT_WORDS: Record<RivalStyle, string[]>`: premium Titan, Apex, Sage; budget Lite, Go, Spark; specialist Forge, Scholar, Muse; consumer Buddy, Chat, Pal.
    - A rival's product k is named by its word, then `"{word} {k}"` from the second release on (e.g. "Titan", "Titan 2").
    - Logos come from `LOGO_SHAPES` and `LOGO_SYMBOLS`.
  - **Balance** `rivals` (balance-v2):
    - `startCash: {premium: 3, budget: 1, specialist: 1.5, consumer: 2}`, `archBonus: 1.5`
    - `firstRelease: [12, 25, 40, 55]`, `releaseGapDays: 110`, `jitterDays: 10`
    - `countdownDays: {premium: 10, budget: 3, specialist: 7, consumer: 5}`
    - `plans: {premium: ["personal","business"], budget: ["free","personal"], specialist: ["personal","business"], consumer: ["free","personal","business"]}`
    - `priceFactor: {premium: 1.5, budget: 0.5, specialist: 1.1, consumer: 0.8}`
    - `reviewEveryDays: 30`, `priceStep: 0.1`
    - `ad: {shareOfCash: 0.01, maxPerDay: {premium: 3000, budget: 500, specialist: 1000, consumer: 4000}, days: 30, minCash: 200_000}`
    - `dataCenterAtServingPerMonth: 60_000`, `trashTalkGapDays: 21`
    - balance-test overrides: `firstRelease: [1, 2, 3, 4]`, `releaseGapDays: 20`, `jitterDays: 0`, every countdown 1, `trashTalkGapDays: 2`
  - **View:**
    - `PublicCompany.ownerUserId: zId.nullable()`
    - `PublicCompany.rival: {style: RivalStyle; archetype: Archetype | null} | null` (archetype only for the arch-rival)
    - `world.rivals: "joined" | "none"`
  - **`upgradeState(state: unknown, from: "sim-3"): WorldState`** (`upgrade.ts`), adding the defaults:
    - `rivals: null`; every company gets `rival: null`, `startCash = foundingCash` and its versions `packKey: null`
    - the empty fields Tasks 5–7 add, with each later task extending the bridge and its test
    - the actor upgrades a sim-3 snapshot, replays the later inputs, and writes a sim-4 snapshot at once

- [ ] **Step 1: Write the failing tests:**
  - **Cast:**
    - the same pack and seed give the same cast (names, styles, archetype, release days)
    - over 20 seeds, at least two archetypes appear
    - the arch-rival's style matches its archetype
    - the four lineages are distinct; the three other rivals' styles are distinct
    - names are unique and never the player's (a player called "Nova Global" makes that name skipped)
    - palettes pass `paletteProblem`, and no two companies share a primary color
  - **Refusals:** a second join; a world in setup; a pack with a style missing. Each leaves the state unchanged.
  - **Release days and news:**
    - release days increase per company and are ≥ `joinedDay + 1`
    - a join on day 40 dates the first releases from day 40 and uses the "New rivals" headline
  - **View:** rivals appear as companies with `ownerUserId` null and `rival` set; `world.rivals` is `"joined"`.
  - **Privacy property:** no view contains any pack key, lineage id, `upcoming`, or a rival's cash.
  - **Money property:** `cash = startCash + Σincome − Σcosts + Σloans + Σinvestors − Σrepaid` holds for every company, rivals included.
  - **Upgrade:**
    - a sim-3 state (taken from a Phase 2 test world) upgrades to a state `step` accepts
    - the actor loads a sim-3 snapshot, replays its inputs, and writes a sim-4 snapshot
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.** Set `SIM_VERSION = "sim-4"`.
- [ ] **Step 4:** Run `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): an arch-rival and three rivals join each world from the rival pack; sim-4`.

### Task 5: Rivals release their AIs and run their businesses

**Files:**
- Modify: `packages/sim/src/rivals.ts` (`rivalsDay`), `packages/sim/src/reducer.ts` (tick: `rivalsDay` right after `settleRuns`), `packages/sim/src/feed.ts` (a rival batch posts once), `packages/sim/src/upgrade.ts`
- Test: `packages/sim/test/rivals.test.ts`; extend the property arbitraries

**Interfaces:**
- Consumes: Task 4, `productCommand`, `campaignCommand`, `computeCommand` and `rescueCommand` (called with the rival company and the ids from the Global Constraints).
- Produces `rivalsDay(s)`, which runs for each rival (ids sorted):
  1. **Release:** each `upcoming` entry whose `releaseDay` is today:
     - becomes the version `{id: "<companyId>-v<k>", label: "v<k>", runId: "pack", packKey: key, parentId: <previous rival version or null>, createdDay: s.day, published: false, …the pack's numbers}`
     - launches as product `<companyId>-p<k>`, named per Task 4, on the style's plans with the style's countdown
     - when 3 products are already on sale, first retires the oldest one that powers no plan; if all three power plans, the release waits a day
  2. **Tidy the lineup:** a live product that powered no plan yesterday and today is retired.
  3. **Set prices:**
     - On the first day the rival has a live product, both prices become `round(default × priceFactor[style])`, clamped to the range.
     - Every `reviewEveryDays` days after joining:
       - customers fell against the history sample from `reviewEveryDays` ago: both prices × `(1 − priceStep)`
       - `compute.served < 0.95`: both prices × `(1 + priceStep)`
       - prices are rounded and clamped
  4. **Advertise** (at a review), when no campaign is running and cash ≥ `ad.minCash`:
     - `budgetPerDay = min(ad.maxPerDay[style], floor(cash × ad.shareOfCash))`; skipped below the campaign minimum
     - runs for `ad.days` days, aimed at the focus group in the region with the most adoptable people of that group
  5. **Build** (at a review): when last closed month's serving cost ≥ `dataCenterAtServingPerMonth`, cash ≥ 2 × the data center's cost and it is below the maximum, build in the region where it has the most customers. `trainingShare` stays 0.
  6. **Rescue:** an offer is answered with the loan the same day.
- **Feedback:** `applyFeedback` for a rival product makes **one** post: the reply whose `|score − 0.5|` is largest (ties: the first). It never makes a viral story. A player's batch is unchanged (best and worst).

- [ ] **Step 1: Write the failing tests:**
  - the arch-rival's first product goes live on its release day plus countdown, with the pack version's scores, and wins customers after it
  - on balance-v2, every rival releases all its versions within 365 days of joining
  - prices stay in range; at most 3 products are on sale; a product that powers no plan is retired the next day
  - a rival below zero at a month end takes the loan that day
  - a review starts a campaign aimed at the focus group's biggest region when cash allows, and builds a data center when serving is costly
  - a rival batch makes exactly one post (the most extreme) and no viral story; a player batch still makes two
  - property: random worlds with rivals over 400 days keep customers ≥ 0, cash finite, and prices in range
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): rivals launch their AIs on schedule and price, advertise and build by their style`.

### Task 6: League tables, #1 news and rank changes on the report card

**Files:**
- Create: `packages/sim/src/league.ts`, `packages/sim/test/league.test.ts`
- Modify: `packages/protocol/src/world.ts` (`LEAGUE_TABLES`), `packages/protocol/src/view.ts`, `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts` (tick: `leagueDay` after `newsDay`), `packages/sim/src/reports.ts`, `packages/sim/src/project.ts`, `packages/sim/src/balance.ts` (`league.newsGapDays: 14`), `packages/sim/src/upgrade.ts`

**Interfaces:**
- Produces:
  - `LEAGUE_TABLES = ["customers", "revenue", "overall", ...BENCH_CATEGORIES] as const` (11 tables), `LeagueTable`.
  - **`leagueTables(s): Record<LeagueTable, {companyId: string; value: number | null}[]>`:** best first; ties go to the lower company id; null values come last.
    - customers: the total
    - revenue: the income of the last closed month (0 without one)
    - overall: the best `overall` of a live product (null without one)
    - each skill: the best score in that category of a live product
  - **State:** `WorldState.leaders: Partial<Record<LeagueTable, string>>`, `WorldState.lastRankingNews: Partial<Record<LeagueTable, number>>`, `CompanyState.lastRanks: {customers: number; revenue: number; overall: number} | null`.
  - **`leagueDay(s)`:**
    - For customers, overall and each skill, a new leader (with a value above 0) is recorded in `leaders`.
    - When the world has rivals and the table's last ranking story is at least `newsGapDays` old, the change makes a story (topic `ranking`, sentiment 0.4, about the new leader):
      - skills: "{Company} takes #1 in {skill words}"
      - customers: "{Company} now has the most customers"
      - overall: "{Company} now has the best AI overall"
  - **Report card:** `ReportCard.ranks: {table: "customers" | "revenue" | "overall"; from: number | null; to: number}[] | null`. It is null in a world without rivals. `lastRanks` updates at every close.
  - **View:**
    - `world.league: {tables: Record<LeagueTable, LeagueRowView[]>; points: Record<BattleSkill, PointsRowView[]>}`
    - `LeagueRowView = {companyId, rank, value: number | null}`
    - Revenue values are null except the viewer's own. Other companies' customer values use three significant figures.
    - `points` is filled by Task 7 (empty until then).
    - `ReportCardView.ranks`.

- [ ] **Step 1: Write the failing tests:**
  - table order and ties; a company with nothing on sale is last with null
  - privacy property: no other company's revenue amount appears in any view
  - a change at #1 makes one story; a flip-flop within 14 days makes no second one; a world without rivals tells none
  - a report card shows customers rank `from 3 to 1` after the player overtakes; it is null without rivals
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): league tables, news of a new #1, and rank changes on the report card`.

### Task 7: Battles and ranking points

**Files:**
- Create: `packages/sim/src/battles.ts`, `packages/sim/test/battles.test.ts`
- Modify:
  - protocol: `packages/protocol/src/world.ts` (the `startBattle` command), `packages/protocol/src/view.ts`
  - sim: `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts` (the command and the `battleResolved` system event), `packages/sim/src/balance.ts`, `packages/sim/src/project.ts`, `packages/sim/src/upgrade.ts`
  - sim tests: `packages/sim/test/arbitraries.ts`, `properties.test.ts`
  - server: `apps/server/src/realtime/gateway.ts` (`startBattle` in `NEEDS_ID`)

**Interfaces:**
- Produces:
  - **Command:** `startBattle {rivalProductId: zId, skill: z.enum(BATTLE_SKILLS)}`. It needs `newId`.
  - **Balance** `battles`:
    - `fee: 5_000`, `rounds: 5`, `cooldownDays: 7`, `drawMargin: 0.25`
    - `pointsStart: 1000`, `pointsK: 32`
    - `sentiment: {win: 0.4, draw: 0.1, loss: -0.3}`, `followers: {win: 0.05, loss: -0.02}`
    - `kept: 20`
    - balance-test: `rounds: 3`, `cooldownDays: 2`
  - **State:**
    - `BattleState = {id; day; skill: BattleSkill; rivalId: string; mine: {productId; versionId; productName}; theirs: {productId; versionId; productName; packKey: string}; seed: number; status: "running" | "done" | "calledOff"; rounds: BattleRound[] | null; outcome: "win" | "draw" | "loss" | null}`
    - `BattleRound = {prompt: string; mine: {reply: string; score: number}; theirs: {reply: string; score: number}}`
    - `CompanyState.battles: BattleState[]` (the last `kept`, newest last)
    - `WorldState.points: Partial<Record<BattleSkill, Record<string, number>>>`
  - **`startBattle` refusals, in this order:**
    1. "There are no rivals in this world."
    2. "That AI isn't on sale." (not a live rival product)
    3. "Launch an AI before battling."
    4. "Your last battle is still going."
    5. "You battled {Rival} recently. Try again on {date}." (within `cooldownDays` of the last battle with that rival; the date in words)
    6. "Creativity can't be judged on this computer yet." (either side's creativity score is null)
    7. "A battle costs $5,000. You have $X."
  - **Starting a battle:**
    - books `marketing` = fee
    - your fighter is your live product with the best score in that skill (ties: the newest)
    - draws `seed` from the world rng; status `running`
  - **System event:** `{type: "battleResolved"; battleId; rounds: BattleRound[]} | {type: "battleResolved"; battleId; failed: true}`.
    - **Refusals:** unknown, or not running ("This battle is over."). Rounds that aren't the balance's count, a score outside 0..1, or a non-string reply give "That battle result isn't valid."
    - **Failed:** status `calledOff`; the fee is refunded by booking `marketing` −fee. Nothing else changes.
    - **Rounds:**
      - each round goes to the higher score (equal: shared); totals are the score sums
      - outcome: a win when `mine − theirs > drawMargin`, a loss below `−drawMargin`, else a draw
    - **Points** (Elo, both sides start at `pointsStart`):
      - `E = 1 / (1 + 10^((Rtheirs − Rmine) / 400))` and `R' = round(R + K × (S − E))`, with S = 1, 0.5 or 0
      - the rival's side is symmetric
    - **Fame:** followers × (1 + `followers.win`) for the winner and × (1 + `followers.loss`) for the loser; no change on a draw.
    - **News** (topic `battle`, sentiment by your outcome):
      - "{Winner}'s {Product} beats {Loser}'s {Product} in a {skill words} battle, {w}–{l}"
      - "{A}'s {Product} and {B}'s {Product} draw in a {skill words} battle, {w}–{l}"
  - **View:**
    - `PrivateCompany.battles: BattleView[]` (the state without `seed` and `packKey`)
    - `world.league.points[skill]`: `{companyId, rank, points}` rows for companies that have points

- [ ] **Step 1: Write the failing tests:**
  - each refusal leaves the state unchanged
  - starting books the fee once; a failed result refunds it once (the money property with battles in the arbitraries)
  - a 4–1 win between two 1000s gives exactly +16 and −16, raises followers and tells one story; a draw inside the margin moves points by 0
  - replies are kept verbatim (a hostile string stays byte-identical)
  - the cooldown date; creativity is refused when a score is null
  - a result for a rival product that retired meanwhile still applies
  - privacy: battles appear once (own) or not at all; `packKey` and `seed` never appear
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): battles against rivals, scored by real replies, with ranking points`.

### Task 8: The arch-rival's trash-talk

**Files:**
- Modify: `packages/sim/src/rivalCast.ts` (`TRASH_TALK`), `packages/sim/src/rivals.ts` (`trashTalk` and its triggers), `packages/sim/src/battles.ts` and `packages/sim/src/rescue.ts` (call the trigger)
- Test: `packages/sim/test/rivals.test.ts`

**Interfaces:**
- Produces:
  - `TrashEvent = "playerLaunch" | "playerWinsBattle" | "playerPassesArch" | "playerRescued" | "archLaunch"`.
  - `TRASH_TALK: Record<Archetype, Record<TrashEvent, [string, string]>>` uses the placeholders `{arch}`, `{player}` and `{product}`. Write the 30 lines in the archetype's voice; three of them are fixed:
    - giant/playerLaunch: `Cute launch, {player}. Call us when {product} can count.`
    - upstart/playerLaunch: `Welcome to the party, {player}. We brought snacks; you brought {product}.`
    - lab/playerLaunch: `{product}. Interesting. We have notes.`
  - **`trashTalk(s, event, {player?, product?})`:**
    - **Triggers** (each a real change that day):
      - a player product goes live
      - the player wins a battle against the arch-rival
      - the player's total customers pass the arch-rival's for the first time (once per world)
      - the player takes a rescue
      - the arch-rival launches
    - **Story:** headline `{Arch}'s founder: "{line}"`; topic `trashTalk`; sentiment 0; `companyId` = the arch-rival; the line drawn from the world rng.
    - **Rare:** at most one story per `trashTalkGapDays` (`rival.lastTrashTalkDay`); never without an arch-rival.

- [ ] **Step 1: Write the failing tests:**
  - each trigger tells one story starting with `{Arch}'s founder: "`
  - the gap holds when two triggers come close together
  - a world without rivals tells none; the same seed gives the same lines
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): the arch-rival comments on what the player really did, as labelled game talk`.

### Task 9: Balance with rivals

**Files:**
- Modify: `packages/sim/test/bots.ts` (bot worlds can include `SYNTHETIC_PACK`; scripts for premium, budget and specialist players), `packages/sim/test/balance.test.ts`, `packages/sim/src/balance.ts` (retuned values, one-line reason each)

**Interfaces:**
- Consumes: Tasks 4–8.
- Produces these targets on `balance-v2`, for a year with the synthetic pack and one bot player:
  1. **Earlier targets still hold:** Phase 1's and Phase 2's targets run again with rivals present. Any that now fails is retuned or, if it no longer makes sense, rewritten with a recorded reason.
  2. **Rivals neither run away nor die:** at day 365 each rival has between 0.5% and 45% of all AI customers, and each has launched at least 3 versions.
  3. **Three ways to win:** a premium bot (high prices, its best model), a budget bot (free plus a low personal price) and a specialist bot (business plan, a coding-heavy model) each have positive profit in at least one of the last three months of the first year.
  4. **Competition comes early:** the arch-rival's first product is live before day 30.
  5. **Ads with competition** (the deferred Phase 2 minor): record whether the $350,000 blitz still takes two months or more to pay back with rivals advertising. Retune `costPerReach` only if it now pays back faster.

- [ ] **Step 1:** Write the targets as tests.
- [ ] **Step 2:** Run: some FAIL.
- [ ] **Step 3:** Retune until all pass; record each change with its reason in `balance.ts` and the ledger.
- [ ] **Step 4:** Run `npm test`: PASS.
- [ ] **Step 5: Commit** `test(sim): balance targets with an arch-rival and three rivals`.

---

## Part C: the server

### Task 10: The rival service

**Files:**
- Create: `apps/server/src/rivals/pack.ts`, `apps/server/src/rivals/service.ts`, `apps/server/test/rivalPack.ts` (writes a fixture pack into a temp folder: the protocol fixture's `rivals.json`, replies for every key, small dummy `.tar` files), `apps/server/test/rivals.test.ts`
- Modify: `apps/server/src/app.ts` (`rivalsDir` option; load the pack after the catalog; build the service), `apps/server/src/main.ts` (`rivalsDir: join(data, "rivals", "v1")`), `apps/server/src/feedback/service.ts`, `apps/server/src/training/service.ts` (eval population), `apps/server/src/worlds/routes.ts` (`GET /api/setup`), `apps/client/src/net/api.ts` (`setup()`)

**Interfaces:**
- Consumes: Tasks 2 and 4.
- Produces:
  - **`loadRivalPack(dir: string | null, catalog: Catalog | null, log): LoadedPack | null`**, where `LoadedPack = {pack: RivalPack; dir: string}`:
    - a missing `rivals.json` warns "no rival pack; worlds run without rivals until it is built" and returns null
    - no catalog, or a `dataIdentity` different from the catalog's, warns "the rival pack was built from other training data; build it again" and returns null
    - a malformed file throws, like the catalog
    - a quick pack only warns
  - **`RivalService`:**
    - It attaches to every world like `FeedbackService` (serial, on load and on change).
    - When `s.phase === "running"`, `s.rivals === null` and a pack is loaded, it submits `rivalsJoined {pack}` once per world (an in-memory memo; the sim refuses repeats).
    - `replies(packKey): RivalReply[]` reads `replies/<key>.json` once and caches it.
    - `feedbackFor(productId: string, packKey: string, key: string, perGroup: number): FeedbackResult[]` picks `perGroup` replies per group with `seedOf(productId, key)` (exported from the feedback service), deterministically.
    - `modelUrl(packKey): Promise<string>`: the first time, it copies `models/<key>.tar` into the blob store at `rivals/<first 12 hex of sha256(dataIdentity)>/<key>.tar` with `ctx.blobs.put`; it returns a GET URL signed for 12 hours.
  - **FeedbackService:** a live product of a rival company records its request row as today, then submits `feedbackScored` with `feedbackFor(...)` directly, with no job.
  - **Eval population:** it adds the traits of every rival version in the world's state.
  - **`GET /api/setup`:** `{trainingData: boolean, rivals: boolean}`, signed-in users only.

- [ ] **Step 1: Write the failing tests:**
  - with the fixture pack, a running world gets four rivals once, and a restart doesn't add more
  - a world running without a pack gets rivals after a restart with one (the late-join path)
  - no pack, or a stale identity: no rivals; `/api/setup` says `rivals: false`
  - a live rival product's weekly batches arrive with no job queued, and the same replies come for the same week after a restart
  - `modelUrl` copies the tar once, and the blob route serves it at the signed URL
  - the eval spec's population includes rival traits
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(server): the rival service seats rivals, answers their feedback from the pack and serves their models`.

### Task 11: Try a rival's AI

**Files:**
- Modify: `apps/server/src/chat/service.ts`
- Test: `apps/server/test/chat.test.ts`

**Interfaces:**
- Consumes: Tasks 3 and 10.
- Produces:
  - **Which AIs a chat may name:** a version of your own company, or the version behind a live product of a rival company. Anything else gives "You can only talk to your own AIs and rivals' AIs on sale."
  - **A rival chat:**
    - is enqueued with `{priority: 5}`
    - its spec says `versionId: packKey` and `modelTarUrl: await rivals.modelUrl(packKey)`
    - its thread stores the rival's version id, so threads list per AI as today
    - your own chats are unchanged
  - Rate limits are shared.

- [ ] **Step 1: Write the failing tests:**
  - a rival chat streams through the fake worker; its job has priority 5
  - a rival product retired since the thread started refuses new messages
  - your own chat is unchanged (priority 0)
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): chat with rivals' AIs on sale, at the lowest priority`.

### Task 12: The battle service

**Files:**
- Create: `apps/server/src/battles/service.ts`, a migration adding `battle_requests` (`battle_id text primary key`, `world_id uuid not null`, `job_id uuid`), `apps/server/test/battles.test.ts`
- Modify: `apps/server/src/db/schema.ts`, `apps/server/src/app.ts`, `apps/server/test/fakeWorker.ts` (battle jobs: progress for both sides, then done), `packages/protocol/src/wire.ts` (`Query {kind: "battle", worldId, battleId}`), `apps/server/src/realtime/gateway.ts` (the `battle` stream prefix)

**Interfaces:**
- Consumes: Tasks 3, 7 and 10.
- Produces:
  - **Sync** (on load and on change, serial): a battle with status `running` and no request row gets a row inserted, then `enqueue("battle", {worldId, companyId, battleId}, worldId, {priority: 1})`.
  - **The spec, built at lease:** `skill`; `rounds` from the balance; `seed` from the battle; `a` = your version's checkpoint URL (`checkpointKey`); `b` = `{versionId: packKey, modelTarUrl: rivals.modelUrl(packKey)}`.
  - **Progress:** a `battleText` event is published on `battle:<worldId>:<battleId>` as `{kind: "text", round, side: "mine" | "theirs", text}` (a → mine, b → theirs) and appended to an in-memory transcript.
  - **Done:** submits `battleResolved` with the rounds mapped a → mine, b → theirs; publishes `{kind: "done"}`.
  - **Final failure:** submits `battleResolved {failed: true}`; publishes `{kind: "calledOff"}`.
  - **The stream and the query:** the `battle` stream and the `battle` query (the transcript so far) are allowed for the battle's owner only.
  - **Restart:** a battle whose job finished while the world wasn't applied is applied on load, like training's reconciliation.

- [ ] **Step 1: Write the failing tests:**
  - a battle streams both sides in round order and resolves; the view shows the outcome and the points
  - a failed job calls the battle off and refunds once
  - a restart mid-battle applies the result exactly once
  - another user can't follow the stream or query it
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): battles run as jobs, stream both replies live and come back as results`.

---

## Part D: the screens (client)

Every new window plugs into the window stack (`WindowSpec` gains `rival`, `newBattle` and `battle`) and follows the immersion design.

### Task 13: The Rivals screen and Try it

**Files:**
- Create: `apps/client/src/screens/Rivals.tsx`, `apps/client/src/windows/RivalWindow.tsx`, `apps/client/test/rivals.test.tsx`
- Modify: `apps/client/src/route.ts` (`TABS` gains `"rivals"` and `"arena"`), `apps/client/src/components/SideNav.tsx` (Rivals and Arena with icons), `apps/client/src/screens/World.tsx`, `apps/client/src/screens/Playground.tsx` and `apps/client/src/store/playground.ts` (a "Rivals' AIs" group in the AI picker), the window store and host

**Interfaces:**
- Consumes: Tasks 4, 6, 10 and 11.
- Produces:
  - **The Rivals screen:**
    - The arch-rival comes first and largest: "Arch-rival · {archetype words}". The three rivals follow.
    - Each card shows the logo, name, style words, products (public scores, personality words), prices, customers and share of the willing people, reputation, followers, and the latest news about them.
    - Each live product has a "Try it" button and a "Battle" button (Task 14's window).
    - Each card opens `RivalWindow`.
  - **No rivals:** "This world has no rivals yet." When `/api/setup` says `rivals: false`, it adds: "Their AIs aren't built on this computer yet. Building them takes about an hour and a half on the graphics card, and they join your worlds after that."
  - **`RivalWindow({companyId})`:** everything on the card, plus:
    - their customers by region on the `RegionMap` in their color
    - their AIs' scores against yours as paired bars per skill
    - their news
  - **Try it:** goes to the Playground with that rival AI chosen. Chat works as with your own; the rival's logo is on its bubbles.

- [ ] **Step 1: Write the failing tests:**
  - four cards with the arch-rival first and its archetype words
  - Try it routes to the Playground with that AI chosen and sends `chat` with its version id
  - the no-rivals message, with the build line when setup says so
  - every value comes from the view
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): a Rivals screen with their public profiles, and Try it chats with their AIs`.

### Task 14: The Arena: league tables and battles

**Files:**
- Create: `apps/client/src/screens/Arena.tsx`, `apps/client/src/windows/NewBattleWindow.tsx`, `apps/client/src/windows/BattleWindow.tsx`, `apps/client/test/arena.test.tsx`
- Modify: the window store and host, `apps/client/src/components/Toasts.tsx` (a battle result toast)

**Interfaces:**
- Consumes: Tasks 6, 7 and 12.
- Produces:
  - **The Arena:**
    - League tables as tabs: Customers, Revenue, Overall, the eight skills, and Ranking points (one table per battle skill).
    - Rows show the rank, logo, name and value ("—" where it is private). Your row is in your colors. Rows open the rival window.
    - "Start a battle" opens the new battle window.
  - **`NewBattleWindow({rivalProductId?})`:**
    - Pick a rival AI on sale and a skill. Skills that can't be used show why: the cooldown date, or "Creativity can't be judged on this computer yet".
    - It shows the fee. Start sends `startBattle`; a refusal shows in the window.
  - **`BattleWindow({battleId})`:**
    - Two columns: your AI in your colors and theirs in theirs.
    - Each round shows the prompt and both replies, streaming live. It queries the `battle` transcript first, then follows `battle:<world>:<id>`.
    - Each round's two scores show as bars. At the end comes a stamp (Win / Draw / Loss with "4–1") and the change in ranking points.
    - Finished battles open from the Arena's "Past battles" list.
  - **Toast:** "You won the coding battle against {Rival}, 4–1" (or drew / lost); it opens the battle.

- [ ] **Step 1: Write the failing tests:**
  - Start sends the exact command, and a refusal shows
  - streamed text lands in the right column and stays text (hostile markup)
  - the done stamp and the points come from the view
  - tables show ranks and hide other companies' revenue
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): the Arena with league tables and live side-by-side battles`.

### Task 15: Rivals everywhere else

**Files:**
- Modify: `apps/client/src/components/RegionMap.tsx` and `apps/client/src/screens/Market.tsx`, `apps/client/src/components/Feed.tsx` and `NewsCard.tsx`, `apps/client/src/screens/Company.tsx`, `apps/client/src/windows/ReportWindow.tsx`, `apps/client/src/components/Toasts.tsx`
- Test: extend `market.test.tsx`, `company.test.tsx`, `reports.test.tsx`, `shell.test.tsx`

**Interfaces:**
- Consumes: Tasks 4–8.
- Produces:
  - **Market map:** a "You / Leaders" switch. "Leaders" tints each region in the color of the company with the most customers there, by its share. Spec §3: "rivals in theirs".
  - **Feed:**
    - A filter, "Everything / About you".
    - A rival's story shows the rival's logo.
    - A trash-talk story is styled as a quote with a small "Game commentary" tag, never as an AI's words.
  - **Company:** a "Rivals" strip with the arch-rival's logo and customers against yours. It opens the Rivals screen.
  - **Report card:** rank changes per table with up and down arrows.
  - **Toasts:** the arch-rival's launches ("{Arch} launches {Product}"), and trash-talk that names you.

- [ ] **Step 1: Write the failing tests:**
  - the switch tints regions by the leader
  - the filter hides other companies' stories
  - trash-talk shows the tag
  - the report card shows "Customers: 3rd → 1st"
  - toasts appear only for the cases listed
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): rivals on the map, in the feed, on the Company screen and on report cards`.

---

## Part E: end to end

### Task 16: End-to-end, the real pack, docs and the final review

**Files:**
- Modify: `tests/e2e/fixtures.ts`, `tests/e2e/solo.spec.ts`, `CLAUDE.md`, `docs/progress.md`, `docs/handoff/README.md`, `docs/handoff/sdd/ledger.md`

- [ ] **Step 1: Extend the E2E.**
  - In the fixtures, after `export-catalog`, run `airace-ml build-rivals --data-root <data> --quick --device cpu`. Measure how long the setup takes; keep it under 3 minutes.
  - After the launch, the test checks:
    1. the Rivals screen shows four rivals with the arch-rival first
    2. a rival's launch story appears in the feed
    3. Try it: a rival's AI answers in the Playground, streamed
    4. start a knowledge battle from the Arena; both columns stream; the stamp and the ranking points show
    5. after a month closes, the report card shows a rank line
- [ ] **Step 2:** Run `npm run e2e`: FAIL, then fix real bugs, each with a unit test in the task that owns the code.
- [ ] **Step 3:** Run `npm test`, `npm run check`, the ML suite with ruff, and `npm run e2e` (twice).
- [ ] **Step 4: Build the real pack** in a GPU window the owner agreed to: `cd ml && python -m uv run airace-ml build-rivals`, with a Monitor heartbeat. Then play in the browser on the GPU (beside the owner's game if theirs runs):
  - rivals join; a rival launches; Try it; a battle in coding and one in creativity; the league tables
  - the owner's upgraded Phase 2 world gets its rivals
- [ ] **Step 5: Final review** of the whole branch range; one fix pass for Critical and Important findings, each with a failing test first.
- [ ] **Step 6: Docs and report:**
  - CLAUDE.md: the `build-rivals` command; after any corpus rebuild, `build-rivals` runs again after `build-judge`; `npm run dev` never builds it.
  - The progress entry, the handoff resume point (Phase 4), and the ledger.
  - Commit and push, then **report to the owner** with the rulings and deferred minors.

## Decisions made in this plan (for the owner report)

These are recorded here because the owner asked for the plan without a review stop. Each costs little if wrong.

1. **Rivals are ordinary companies without an owner.** The market, money, fame and computers treat them exactly like the player. If wrong: a separate rival model would mean rewriting the market.
2. **One logged `rivalsJoined` input carries the pack's measurements; releases then happen on game days inside the sim.** The spec's per-release `rivalReleased` input isn't needed: nothing to schedule on the server, and replay is exact. If wrong: one more input type.
3. **Rivals join any running world that has none once the pack exists, including the owner's Phase 2 world after its upgrade** (the spec only says worlds without a pack start without rivals). If wrong: a late joiner starts small against an established player.
4. **A battle costs $5,000 of marketing**, refunded if it is called off (spec §13.5 implies a fee). If wrong: one number.
5. **Battle skills:** knowledge, coding, instruction, pattern and creativity, the five with written replies. Creativity needs both AIs to have a creativity score. If wrong: other skills would need replies made from multiple choice.
6. **Rival feedback makes one post per batch and never goes viral,** so four rivals don't flood the feed. If wrong: a quieter rival presence.
7. **The revenue league shows other companies' places, not their money (R2).** If wrong: a less informative table.
8. **The pack is 8 lineages with 3 versions each, about 75–90 minutes on the RTX 3050** (the spec said about an hour), resumable. If wrong: a longer one-time wait, or fewer versions.
9. **Elo is called "ranking points" (R7).**

## Deferred to Phase 4

Interviews, discoveries, the notebook, milestones (including "first battle won against the arch-rival"), and goals with the victory screen.
