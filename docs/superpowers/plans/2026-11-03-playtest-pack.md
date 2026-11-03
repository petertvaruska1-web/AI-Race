# Playtest Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fair battles, a posts pause while in the Playground, six real-AI minigames with showcases, and eight new things to buy (spec §2–§5).

**Architecture:** ML adds the minigame content, the run loop and three job handlers. The sim (sim-5) adds showcases, sponsorships, looks, apps, investments, overhead, the keynote and the training extras. The server adds the minigame, keynote and app-review services, the loved-replies store and the presence pause. The client adds Playground games and the spending screens.

**Tech Stack:** Python (uv, torch), TypeScript (zod, Fastify, PGlite/Drizzle, React 19, Tailwind 4, Zustand, motion).

**Spec:** `docs/superpowers/specs/2026-11-03-playtest-pack-design.md`, plus `2026-10-27-game-layer-design.md` and the architecture spec.

**Ledger:** `docs/handoff/sdd/ledger-playtest-pack.md` (committed; one line per finished task, every ruling).

## Global Constraints
- R1–R8 (CLAUDE.md). Model output is never rewritten; a minigame choice is the model's own likelihood pick or its own typed reply.
- No `Date.now()` or `Math.random()` in `packages/sim`; every new random draw uses `rngInt` or `rngFloat` from `rng.ts`.
- Private company data (loved replies, fund, apps' reviews before posting, minigame runs) reaches clients only through `projectView`, for the owner only (R2).
- Plain language in every player-facing string (R7). Never "softmax", "log-probability" or "token".
- Balance values live in `balance.ts` (`ECONOMY_V2` and the test balance). Spec §5 gives the starting numbers.
- `SIM_VERSION = "sim-5"`; sim-4 snapshots upgrade on load.
- Commit per task, trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, branch `m2-two-friends`, and push.
- Never start a server on `data/` while the owner's game runs (check `data/server/.lock`).

## Review Focus
1. A minigame against a model that always answers the same option: the run must end within the step and life limits, and nothing may loop forever.
2. A rival product that leaves sale mid-duel: the job finishes with what it has, and the client shows "ended early".
3. A sim-4 save with live campaigns, data centers and a loan upgrades to sim-5 with every new field defaulted, and replays to the same cash.
4. Withdrawing more than the fund holds, or depositing more cash than exists, is refused with a plain reason.
5. Learn from customers with fewer than 10 loved conversations is refused before any money moves.

---

### Task 1: Fair battles
**Files:**
- `ml/src/airace_ml/battles.py`, `ml/src/airace_ml/worker/jobs.py`, `ml/tests/test_battles.py`
- `packages/protocol/src/worker.ts` (round `expected: string | null`), `packages/protocol/src/view.ts` (`BattleView.rounds[].expected`)
- `packages/sim/src/battles.ts`, `packages/sim/src/balance.ts` (`drawMargin` 0.1)
- `apps/server/src/battles/service.ts`, `apps/client/src/windows/BattleWindows.tsx`, plus the tests beside each

**Interfaces:**
- `battle_prompt(item) -> (prompt: str, chat: bool, max_new_tokens: int, temperature: float)`
- `closeness(item, reply) -> float` in [0, 1]
- `score_round(item, reply, ...) -> float`, where a pass gives 1 and otherwise `0.5 * closeness`
- `expected_answer(item) -> str | None`

**Steps:**
- [ ] **Tests first.**
  - `test_score_round_partial_credit_orders_replies`: for a numeric exact item with answer 14, the reply "13" scores above "pigmpton." and "14" scores 1.
  - `test_code_shaped_reply_beats_chatter`: for the coding CheckItem "def front(seq)", the reply `return seq[1]` scores above `Great!`.
  - `test_battle_asks_like_the_benchmark`: an ExactItem with `chat=False` is asked without the chat wrapper, greedily, at the item's length.
  - Sim test: totals 2.3 vs 2.25 → win at margin 0.1... no, they differ by 0.05, so the result is a draw; totals 2.3 vs 2.1 → win.
  - Client test: the window shows "Right answer: 14".
- [ ] Implement, run `cd ml && python -m uv run pytest tests/test_battles.py` and `npm test`, commit.

### Task 2: Posts pause while in the Playground
**Files:**
- `packages/protocol/src/wire.ts` (Intent `playgroundPresence {worldId}`)
- `apps/server/src/jobs/broker.ts` (`holdWhile(worldId, types)`, a presence map with a TTL of 30 s on the scheduler clock)
- `apps/server/src/app.ts` or the realtime handler
- `apps/client/src/screens/Playground.tsx` (15 s heartbeat while mounted; the note; the "New chat" hint)
- Tests

**Interfaces:**
- `broker.setPresence(worldId, now)`
- `broker` lease skips `feedback` and `appReview` jobs whose world has a presence newer than 30 s

**Steps:**
- [ ] **Tests:**
  - `feedback jobs wait while the player is in the Playground`: a lease returns a generate job but no feedback job; 31 s later the feedback job leases.
  - A client test shows the note.
- [ ] Implement, test, commit.

### Task 3: Minigame content (ML)
**Files:** `ml/src/airace_ml/minigames/__init__.py`, `content.py`; `ml/tests/test_minigame_content.py`

**Interfaces:**
- `GAMES = ("maze", "bridge", "quiz", "story", "simon")`, `DIFFICULTIES = ("easy", "medium", "hard")`
- `@dataclass GameItem`:
  - `id`, `band: int`, `prompt: str`, `chat: bool`
  - `options: list[str] | None`, `answer_index: int | None`, `check: ExactItem | CheckItem | None`
  - `typed: bool`, `display: dict`
- `game_items(game, difficulty, seed, settings) -> list[GameItem]`
  - `settings`: quiz `{topic}` from `quiz_topics()`; story `{dice: [hero, situation, word|None]] x3 or "roll"}`

**Steps:**
- [ ] **Tests:**
  - `test_items_never_repeat_benchmark_prompts`
  - `test_bands_rise_with_step` (non-decreasing band)
  - `test_option_counts_by_difficulty` (2, 3, 4; the quiz always 4 with `display.struck`)
  - `test_pattern_options_include_answer_once`
  - `test_display_pictures` (a cycle of colors gives `display.terms` with `{kind: "color", value: "red"}`)
  - `test_same_seed_same_items`
  - `test_story_dice_prompt_and_rule`
- [ ] Implement from `skills/*` generators on `skill_rng(skill, "battles")`, filtered by `battles._taken_prompts()`. Commit.

### Task 4: The minigame run loop, and the keynote and app-review jobs (ML + worker protocol)
**Files:**
- `ml/src/airace_ml/minigames/run.py`, `ml/src/airace_ml/worker/jobs.py`, `ml/src/airace_ml/worker/protocol.py`, `cli.py` (slots)
- `packages/protocol/src/worker.ts` and fixtures `lease-minigame`, `result-minigame`, `progress-minigame`, `result-keynote`, `result-appreview`
- Tests

**Interfaces:**
- `choose(lm, tok, prompt, chat, options) -> (index, confidences: list[float])`
- `play(game, items, side_models, on_step) -> {side: {steps, score, maxScore}}`, with lives 3; maxScore 10, 12 for the bridge and quiz, 39 for the story
- `handle_minigame(spec, ctx)`
- `handle_keynote(spec) -> {prompt, reply, score}`: the question id comes from the customer pools
- `handle_app_review(spec) -> {reviews: [{prompt, reply, score}]}`: the skill pool
- Slots: `minigame: 1`, `keynote: 1`, `appReview: 1`

**Steps:**
- [ ] **Tests:**
  - `test_run_ends_on_three_misses` (a fake model always picks the option that sorts first alphabetically)
  - `test_confidences_sum_to_one`
  - `test_duel_same_items_both_sides`
  - `test_typed_answers_on_hard_late_steps`
  - `test_story_scores_with_judge` (fake judge)
  - Protocol fixture round-trips
  - Slow: a tiny real model plays a Pattern Bridge
- [ ] Implement, test, commit.

### Task 5: Game-layer protocol
**Files:** `packages/protocol/src/world.ts`, `view.ts`, `wire.ts`, `minigames.ts` (new: game ids, difficulty, settings schemas, the `MinigameStepEvent` and `MinigameRunView` types)

**Interfaces:**
- **Commands:**
  - `startShowcase {game, difficulty, versionId, settings}`
  - `sponsor {region, kind}`
  - `buyLook {lookId}`, `useLook {lookId}`
  - `fundApp {kind, productId}`, `closeApp {appId}`
  - `invest {amount}`, `withdraw {amount}`
  - `launchProduct` gains `keynote: {questionId} | null`
  - `TrainingPlan` gains `tutoring: {dataset, packs: 0..5} | null`, `learnFromCustomers: boolean`, `cleaning` gains "expert"
- **System inputs:** `showcaseResolved`, `showcaseFailed`, `keynoteResolved`, `appReviewed`
- **Intents:** `startMinigame {worldId, game, difficulty, settings, versionId, opponent: productId | null}`, `playgroundPresence`
- **Query:** `minigameRuns`

**Steps:**
- [ ] Schema tests (each new command parses, and bad amounts are refused).
- [ ] Implement, `npm test -w @airace/protocol`, commit.

### Task 6: sim-5, the upgrade and showcases
**Files:** `packages/sim/src/state.ts`, `upgrade.ts`, `world.ts` (`SIM_VERSION`), `showcases.ts`, `reducer.ts`, `balance.ts` (`showcases: {fee, cooldownDays, followers}`), `project.ts`; tests

**Steps:**
- [ ] **Tests:**
  - A sim-4 state upgrades with defaults (`looks`, `apps`, `sponsorships`, `fund`, `showcases`, `lastShowcaseDay`) and replays to the same cash.
  - A showcase charges the fee, refuses during cooldown, and its resolution moves followers and posts news with the right wording.
  - A failed showcase is refunded.
- [ ] Implement, commit.

### Task 7: Investments and growing bills
**Files:** `packages/sim/src/investments.ts`, `overhead.ts`, `ledger.ts` (new categories), `balance.ts`, `reports.ts`; tests; the balance harness

**Steps:**
- [ ] **Tests:**
  - Deposits move cash to the fund (the money identity holds with the fund counted).
  - Month-end returns are seeded and booked as gains or losses.
  - A withdrawal over the balance is refused.
  - Overhead is 0 with no customers and about 15% of revenue at 50k.
  - The balance harness still meets the game spec §14 targets.
- [ ] Implement, retune the balance if the harness asks for it (ledger a ruling), commit.

### Task 8: Sponsorships and looks
**Files:** `packages/sim/src/sponsorships.ts`, `looks.ts` (catalog `LOOKS`), `market.ts` hook, `balance.ts`, `project.ts` (`PublicCompany.logoMotion`, `world.sponsorships` markers); tests

**Steps:**
- [ ] **Tests:**
  - A sponsorship charges its price, raises its groups' awareness in its region daily until it ends, posts news on day 1, and shows as a marker.
  - Buying a look charges once, using it switches, using one not owned is refused, and the free ones are owned at start.
- [ ] Implement, commit.

### Task 9: Partner apps
**Files:** `packages/sim/src/apps.ts`, `balance.ts` (`apps.kinds`), `reducer.ts` (`appsDay`), `feed.ts` (app reviews as posts), `project.ts`; tests

**Steps:**
- [ ] **Tests:**
  - Funding charges setup.
  - Users grow toward capacity × fit from the product version's measured score.
  - A version without that score grows at fit 0.1.
  - The monthly revenue share and upkeep are booked.
  - `appReviewed` posts reviews with stars and moves the rating.
  - Closing stops upkeep.
  - An app whose product is retired pauses.
- [ ] Implement, commit.

### Task 10: The launch keynote and the training extras (sim)
**Files:** `packages/sim/src/products.ts` (keynote), `training.ts` (`previewPlan`: tutoring tokens, cost, duration; expert cost; learn cost from `lovedCount` passed in the command), `balance.ts`; tests

**Steps:**
- [ ] **Tests:**
  - A keynote charges its fee.
  - `keynoteResolved` before launch day is applied on launch day, and after launch day at once.
  - The awareness lift is scaled by 0.5 + s, news quotes the reply, and s < 0.3 is a flop story.
  - Tutoring adds 10% of tokens per pack, cost and days.
  - Expert costs $4 per 1k tokens.
  - Learn from customers with fewer than 10 is refused.
- [ ] Implement, commit.

### Task 11: The training extras (ML)
**Files:** `ml/src/airace_ml/data/prep.py` (cleaning "expert": 0.80, forces dedup and fact check), `catalog.py` (expert pools; catalog version bump), `train/config.py` (accept "expert"); tests

**Steps:**
- [ ] **Tests:**
  - Expert keeps a subset of thorough.
  - The catalog lists expert pools.
  - A run with coaching pairs and a `coaching` mixture share trains (the existing support, now exercised).
- [ ] Check the share of documents kept per dataset on the real corpus; ledger it, and lower the threshold if a dataset keeps under 20%.
- [ ] Implement, commit.

### Task 12: The minigame service and showcases (server)
**Files:** `apps/server/src/minigames/service.ts`, `db/schema.ts`, migration `0006_minigames`, `app.ts` wiring, the realtime stream; tests with the fake worker

**Steps:**
- [ ] **Tests:**
  - A practice run streams steps and stores the best.
  - A duel loads the rival's pack model.
  - One run per player.
  - A showcase result becomes `showcaseResolved`, and a failure becomes `showcaseFailed`.
- [ ] Implement, commit.

### Task 13: Keynote, app reviews, loved replies and the training config (server)
**Files:** `apps/server/src/keynotes/service.ts`, `apps/server/src/apps/service.ts`, `apps/server/src/feedback/service.ts` (record loved replies), the `loved_replies` table (migration 0007), `apps/server/src/training/service.ts` (coaching pairs, tutoring mix and token budget, expert prep); tests

**Steps:**
- [ ] **Tests:**
  - Feedback results scoring at least 0.75 are stored, keeping the newest 400.
  - A training job with `learnFromCustomers` carries them as coaching at 5%.
  - Tutoring changes the mixture and the budget exactly as the preview says.
  - The keynote job is enqueued at launch and its result is logged.
  - App reviews are enqueued weekly per live app.
- [ ] Implement, commit.

### Task 14: Playground games (client)
**Files:**
- `apps/client/src/screens/Playground.tsx` (modes)
- `apps/client/src/games/` (`GameGallery.tsx`, `GameSetup.tsx`, `MazeStage.tsx`, `BridgeStage.tsx`, `QuizStage.tsx`, `StoryDiceStage.tsx`, `SimonStage.tsx`, `DuelView.tsx`, `useMinigame.ts`, `pictures.ts`)
- Tests fed recorded `minigameStep` events

**Steps:**
- [ ] **Tests:**
  - Each stage renders its steps: the orb at the chosen door, tiles laid or crumbled, the ladder rung, the dice and the rule chip, the green or red cards.
  - The duel shows two lanes.
  - The showcase button shows the fee and the cooldown.
  - The personal best updates.
- [ ] Implement with SVG and `motion`, in the company palette. Commit.

### Task 15: Spending screens (client)
**Files:**
- `apps/client/src/windows/CustomizeWindow.tsx`, `components/Orb.tsx` (styles), `components/Logo.tsx` (motions)
- `screens/Company.tsx` (Apps section), `windows/AppWindow.tsx`, `windows/SponsorWindow.tsx`, `screens/Market.tsx` (markers)
- `screens/Money.tsx` (fund, overhead, new categories), `money.ts`
- The AI Lab plan step (tutoring, learn, expert), the Launch dialog (keynote)
- Tests

**Steps:**
- [ ] Tests per window (buying, refusing, showing).
- [ ] Implement, commit.

### Task 16: End to end, docs and the final review
**Steps:**
- [ ] Extend `tests/e2e/solo.spec.ts`: a practice Pattern Bridge, buying a look, depositing into the fund, a sponsorship, a training run with one tutoring pack.
- [ ] Run `npm test`, `npm run check`, ML `pytest` and `ruff`, and `npm run e2e`.
- [ ] Play it in the browser on a second data root.
- [ ] Update `docs/progress.md`, the handoff README §4 (Phase 4 now targets sim-6), CLAUDE.md commands (catalog rewrite), and the ledger.
- [ ] Do the final whole-branch review and fix pass, then push.
