# Game Layer Phase 1 "A Real Company" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the M2 demo into a real single-player company game. The player:
- trains an AI and launches it as a named product on free, personal and business plans
- watches customers in six groups arrive and pay
- reads social posts that quote the AI's real replies, plus the news
- follows reputation, followers and money (in and out)
- does all this on a new futuristic Company screen with the AI as a glowing orb

**Architecture:**
- New game rules go into the deterministic sim as focused modules: `products`, `market`, `ledger`, `feed` and `fame`, all under `SIM_VERSION` `sim-2` and `balance-v2`.
- The server gains a feedback service. It turns weekly sampling into `feedback` jobs, and the Python worker answers them with real replies scored by benchmark-style checks; the results come back as logged inputs.
- The client gets a new app shell, the orb, and the Company, Market and Money screens. Everything still flows through `projectView`, snapshots and patches.

**Tech Stack:** As M2 (TypeScript 7, zod 4, Vitest 5, fast-check 4, Fastify 5, PGlite and Drizzle, React 19, Vite 8, Tailwind 4, Zustand 5, uPlot; Python 3.13 with the M1 engine).

**Spec:** `docs/superpowers/specs/2026-10-27-game-layer-design.md`. §1–§8, §12–§15 apply to Phase 1; R1–R8 from the architecture spec always apply.

## Global Constraints

- **Versions:**
  - `SIM_VERSION` becomes `"sim-2"`. Worlds saved by `sim-1` are refused by `WorldActor.load` (already) and listed as outdated (Task 8).
  - The balance ids become `"balance-v2"` and `"balance-test"`. `BALANCE_V1` is renamed `BALANCE_V2`.
- **Groups:** `GROUP_IDS = ["students","creators","developers","businesses","everyday","researchers"] as const`.
- **Plans:** `PLAN_IDS = ["free","personal","business"] as const`.
- **Region:** Phase 1 has one region, `"national"`. Market state is keyed by region, so Phase 2 can add the 9 divisions without changing shapes.
- **Lineup:** at most **3** products in `countdown` or `live`. Product names: `zName(2, 24)`. Countdown: 3–14 game days.
- **Prices** (integer dollars per month):
  - personal: default 10, allowed 1–200
  - business: per seat, default 30, allowed 5–500
- **Money:** integer dollars. Every change to `cash` is also booked in the ledger in the same step. Fractional amounts are rounded once per booking.
- **R1:** posts and news may quote a reply only **verbatim**. Game text (headlines, frames) never puts words in an AI's mouth. The client renders all player and model text as text.
- **R2:** `projectView` builds public fields explicitly. Another company's cash, ledger, satisfaction, awareness and spend never leave the server.
- **R7:** no ML jargon in any new copy. Say "customers", "plans", "your AI", "launch".
- **Job priority** (lower runs first): generate (chat) 0, train 2, eval 3, feedback 4. This replaces generate 0, eval 1, train 2, per spec §12.
- **Tests:** `npm test` and `npm run check` green after every task. `cd ml && python -m uv run pytest` under 3 minutes, and `ruff check` clean. Python commands follow CLAUDE.md (`--extra content` / `--no-sync`).

## Review Focus

1. **Launching an AI that has no scores yet** (scoring runs a minute or two after training). Expect a plain refusal ("Wait until your AI has been scored before launching it") and no state change. Owner: Task 2.
2. **Hostile or odd text in product names and in model replies quoted by posts:** HTML, `<|end|>`, 2,000 characters, emoji, empty replies. Names are cleaned or rejected. Replies are shown whole and verbatim as text (long ones fold behind "Show more", never cut in the data). Owners: Tasks 2 and 12.
3. **Lineup and plan edge cases:** a fourth product, retiring a product that powers a plan, an out-of-range price, a plan pointed at a retired or countdown product. Each is refused with a plain reason and leaves cash and customers unchanged. Owner: Task 2.
4. **A feedback result delivered twice, or a server restart mid-countdown.** Satisfaction and posts apply once, and the launch happens exactly once on its day. Owners: Tasks 5 and 8.
5. **Money edge cases:** a training refund after a month closes, revenue with zero customers, a company with no live product. Cash always equals founding cash plus every ledger booking, and never drifts by rounding. Owner: Task 4.

---

## Part A: the game rules (sim)

### Task 1: Balance v2, groups, plans and sim-2

**Files:**
- Modify: `packages/sim/src/balance.ts`, `packages/protocol/src/world.ts`, every reference to `BALANCE_V1` and `"balance-v1"` (sim, server, client, tests), `apps/server/src/config.ts` (`SIM_VERSION = "sim-2"`), `apps/server/src/main.ts` (default `AIRACE_BALANCE` is `balance-v2`)
- Test: `packages/sim/test/foundations.test.ts` (extend)

**Interfaces:**
- Produces:
  - **Protocol:**
    - `GROUP_IDS`, `GroupId`, `PLAN_IDS`, `PlanId`
    - `REGION_IDS = ["national"] as const`, `RegionId`
  - **Balance:**
    - `BalanceVersion = "balance-v2" | "balance-test"`
    - `Balance.economy: Economy`
  - **`Economy`** (all numbers; initial values are the implementer's choice, then tuned by Task 7):
    - `population: Record<RegionId, number>`: national = 335_000_000
    - `adoption: {start, end, midYear, steepness}`: willing share = `start + (end-start)/(1+exp(-steepness*(day/365-midYear))))`, with start 0.10 and end 0.60
    - `groupMix: Record<RegionId, Record<GroupId, number>>`: sums to 1
    - `groups: Record<GroupId, {weights: Record<string, number> /* over the 8 bench categories */, priceSensitivity, speedPreference /* -1 smarts … +1 speed */, churnPerDay, buysBusiness: boolean, seatsPerAccount}>`
    - `prices: {personal: {default: 10, min: 1, max: 200}, business: {default: 30, min: 5, max: 500}}`
    - `noAiUtility: Record<GroupId, number>`
    - `utility: {score, price, speed, reputation, satisfaction}`: coefficients
    - `diffusion: {adoptRate, switchRate, wordOfMouth, awarenessDecay, launchBoost}`
    - `servingCostPerCustomerDayPerMParam`
    - `fame: {reputationStart: 50, reputationRate, newsWeight, followersPerHype, followersDecayPerDay, viralScore, viralFollowers}`
    - `feedback: {perGroup: 1, satisfactionAlpha, satisfactionStart: 0.5}`
    - `feedCap: 200`
    - `hypePerDay`
    - `maxProducts: 3`
    - `countdown: {min: 3, max: 14}`
  - **`BALANCE_TEST`:** its economy makes customers arrive within a few game days of a launch (for E2E).

- [ ] **Step 1: Write the failing tests.**
  - `getBalance("balance-v2")` is `BALANCE_V2`.
  - `getBalance("balance-test")` is `BALANCE_TEST`.
  - Every `groupMix` row sums to 1 (±1e-9).
  - Every group's weights name only bench categories.
  - `adoption` at day 0 is 0.10 ±0.01 and tends to 0.60.
  - Price defaults lie within their bounds.
- [ ] **Step 2:** Run `npx vitest run packages/sim`: FAIL.
- [ ] **Step 3: Implement.** Rename `BALANCE_V1` to `BALANCE_V2` across the repo. Update the server default and `SIM_VERSION`. Server tests that create worlds use `balance-test`.
- [ ] **Step 4:** `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): balance v2 economy constants, customer groups, plans; sim-2`.

### Task 2: Products, plans and launch day

**Files:**
- Create: `packages/sim/src/products.ts`
- Modify: `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts`, `packages/protocol/src/world.ts`, `apps/server/src/realtime/gateway.ts` (`needsId`)
- Test: `packages/sim/test/products.test.ts`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - **State:**
    - `ProductState = {id, name, versionId, status: "countdown"|"live"|"retired", announcedDay, launchDay, hype: number, feedbackKeys: string[]}`
    - `PlansState = {free: {productId: string|null}, personal: {productId: string|null, price: number}, business: {productId: string|null, price: number}}`
    - `CompanyState` gains `products: Record<string, ProductState>` and `plans: PlansState` (all plans closed, default prices)
  - **Commands** (protocol `Command`):
    - `launchProduct{versionId, name: zName(2,24), plans: PlanId[], countdownDays: int 3–14}`, with `newId` as the product id. `needsId` in the gateway adds `launchProduct`.
    - `setPlan{plan: PlanId, productId: zId | null}`
    - `setPrice{plan: "personal"|"business", price: int}`
    - `retireProduct{productId}`
  - `liveProductFor(company, plan): ProductState | null`
- Rules (each rejection has a plain reason and leaves state unchanged):
  - **launchProduct:**
    - the version is the company's own and has `scores !== null` ("Wait until your AI has been scored before launching it")
    - fewer than `maxProducts` products in countdown or live
    - the name is unique among the company's non-retired products (case-insensitive)
    - the world is running
  - **Countdown:** each tick adds `hype += hypePerDay × (reputation/100 + followers/(followers+1000))`. On `day == launchDay` the tick sets the status to `live` and points every listed plan at it.
  - **setPlan:** the product must be `live`, or `null` (closes the plan).
  - **setPrice:** within the bounds.
  - **retireProduct:** refused while any plan points at it ("Move its plans to another AI first").

- [ ] **Step 1: Write the failing tests** in `products.test.ts`:
  - an unscored version is refused
  - a countdown of 3 goes live on `announcedDay+3` with its plans set
  - hype grows during the countdown
  - a fourth product is refused
  - a duplicate name in any case is refused
  - `name: "<b>Swift</b>"` is kept as text
  - a name of 25 characters is refused
  - setPlan to a countdown product is refused
  - a price of 0 or 201 is refused
  - retiring a plan's product is refused; after `setPlan(null)` it succeeds
  - every rejection leaves `stateHash` unchanged
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement** in `products.ts` (handlers and tick hook), wired from `reducer.ts`.
- [ ] **Step 4:** Run: PASS. `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): products, plans, prices and launch countdowns`.

### Task 3: The market: customers, choice and spotlights

**Files:**
- Create: `packages/sim/src/market.ts`, `packages/sim/src/spotlights.ts`
- Modify: `packages/sim/src/state.ts`, `packages/sim/src/world.ts`, `packages/sim/src/reducer.ts` (tick)
- Test: `packages/sim/test/market.test.ts`

**Interfaces:**
- Consumes: Tasks 1–2; `VersionState.scores` and `paramCount`.
- Produces:
  - **State:**
    - `MarketState = {customers: Record<companyId, Record<RegionId, Record<GroupId, Record<PlanId, number>>>>, awareness: Record<companyId, Record<RegionId, Record<GroupId, number>>>, satisfaction: Record<productId, Record<GroupId, number>>}`
    - `WorldState.market`
    - `WorldState.spotlights: Spotlight[]`, where `Spotlight = {id, name, handle, group, region, bio}`, six (one per group) drawn from the world rng in `createWorld`
  - **Pure functions:**
    - `adoptionShare(economy, day): number`
    - `productAppeal(economy, group, version, product, plan, price, reputation, satisfaction): number`: utility per spec §5. Scores weighted by group, minus price × sensitivity (`ln(1+price/10)`), plus speed × preference (`speed = 1 − params/(params+5e6)`), plus reputation and satisfaction terms.
    - `choiceShares(utilities: number[], noAi: number): number[]`: a logit with the outside option
    - `marketDay(state): void`: the daily step. Target customers = adoptable × share. Customers move toward the target (up at `adoptRate×awareness`, down at `switchRate`), lose churn × (1−satisfaction), and get word of mouth into awareness.
  - **Launch boost:** when a product goes live, awareness in every group rises by `launchBoost × (0.5 + hype)` (capped at 1).
- Algorithm note: iterate companies, regions, groups and plans in sorted key order, so the step is deterministic.

- [ ] **Step 1: Write the failing tests:**
  - `choiceShares([0,0], 0)` gives `[1/3, 1/3]`
  - shares plus the no-AI share sum to 1
  - a higher score in a group's weighted categories raises that group's share
  - a higher price lowers it, more for students than for businesses
  - a company with no live product has 0 customers after 30 ticks
  - after a launch with scores `{overall ~40}`, customers grow day over day and stay ≤ adoptable
  - closing the personal plan moves its customers off over the following days, never below 0
  - six spotlights, one per group, are the same for the same seed
  - `marketDay` leaves `stateHash` equal across two runs of the same inputs
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): national market with six customer groups, choice, diffusion and spotlights`.

### Task 4: The ledger: money in and out

**Files:**
- Create: `packages/sim/src/ledger.ts`
- Modify: `packages/sim/src/reducer.ts` (startTraining, trainingFailed, tick), `packages/sim/src/state.ts`
- Test: `packages/sim/test/ledger.test.ts`; update the money property in `packages/sim/test/properties.test.ts`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces:
  - **Categories:**
    - `INCOME_CATEGORIES = ["personal","business"]`
    - `COST_CATEGORIES = ["training","serving","dataChecks"]`; marketing, data centers and interest come in Phase 2
  - **State:**
    - `MonthLedger = {month: string /* "2027-03" */, income: Record<IncomeCategory, number>, costs: Record<CostCategory, number>, revenueByGroup: Record<GroupId, number>, revenueByPlan: Record<PlanId, number>, customersEnd: number}`
    - `CompanyState.ledger = {current: MonthLedger, months: MonthLedger[] /* last 24, newest last */}`
  - **Functions:**
    - `book(company, category, amount: number, detail?: {group?, plan?}): void`: rounds once, then changes `cash` and the current month together
    - `ledgerDay(state): void`: daily revenue (`personal customers × price / 30`; `business accounts × seatsPerAccount × price / 30`) and serving costs (all customers × `servingCostPerCustomerDayPerMParam` × the params of the product behind their plan, in millions)
    - `closeMonth(company, month, customers): void`, called by the tick when `dateOfDay` changes month
  - **Existing money paths now book:**
    - startTraining: the training cost under `training`, fact-checking under `dataChecks`
    - trainingFailed: the refund, as a negative cost under the same categories

- [ ] **Step 1: Write the failing tests:**
  - 30 days with N personal customers at $10 books about N×$10 of personal income
  - serving cost scales with product params
  - a training run books its cost under `training` (and `dataChecks` when fact-checking)
  - a refund books it back
  - a month change closes `current` into `months`
  - a refund arriving after its month closed books into the current month
  - a day with zero customers books zero income and zero serving
  - after any sequence: `cash === foundingCash + Σ(income) − Σ(costs)` over all months, exactly (integers)
- [ ] **Step 2:** Update the property (d) money in `properties.test.ts` to the identity above, over the extended arbitraries of Task 6.
- [ ] **Step 3:** Run: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5:** Run: PASS.
- [ ] **Step 6: Commit** `feat(sim): ledger with income, costs and monthly summaries`.

### Task 5: Feedback, posts, news and fame

**Files:**
- Create: `packages/sim/src/feed.ts`, `packages/sim/src/fame.ts`
- Modify: `packages/sim/src/state.ts` (SystemEvent), `packages/sim/src/reducer.ts`
- Test: `packages/sim/test/feed.test.ts`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - **System event:** `feedbackScored{productId, key: string /* "w12" weekly, "launch" burst */, results: {group: GroupId, questionId: string, question: string, reply: string, score: number /* 0..1 */}[]}`. A repeated `(productId, key)` is refused ("already applied"), using `ProductState.feedbackKeys` (the last 20 keys).
  - **Effects:**
    - satisfaction becomes an EMA with `satisfactionAlpha` per group
    - the highest- and lowest-scoring results become **posts**
  - **State:**
    - `FeedItem = {id: string, day: number, companyId: string|null, sentiment: number /* -1..1 */} & ({kind: "post", author: {name, handle, group: GroupId, spotlightId: string|null}, question: string, reply: string, tone: "praise"|"joke"|"roast"} | {kind: "news", topic: NewsTopic, headline: string})`
    - `WorldState.feed: FeedItem[]`, capped at `feedCap`, oldest dropped
  - **Posts:**
    - authored by the group's spotlight half the time (world rng), otherwise a generated handle
    - tone from score: ≥0.75 praise, ≤0.25 roast, else joke
    - `reply` is stored exactly as received
  - **News topics in Phase 1:** `launch`, `customers` (1k, 10k, 100k and 1M thresholds), `firstPaying`, `viral` (a post whose score is ≥ `viralScore` or ≤ 1−`viralScore` while followers ≥ `viralFollowers`).
    - Headlines are templates over real numbers and names, for example "Acme launches Swift: #… in coding" without rival ranks in Phase 1, and "Acme passes 10,000 customers".
  - **Fame:** `CompanyState.fame = {reputation: number /* 0..100 */, followers: number}`
    - Reputation moves `reputationRate` per day toward `100 × (customer-weighted satisfaction)` blended with the EMA of the company's news sentiment (`newsWeight`).
    - Followers gain `followersPerHype × hype` at launch and `viralFollowers/10` per viral post, and decay by `followersDecayPerDay`.

- [ ] **Step 1: Write the failing tests:**
  - a `feedbackScored` with scores 0.9 and 0.1 creates exactly two posts
  - the praise post's `reply` equals the input reply byte for byte, including `"<b>x</b> <|end|>"`
  - satisfaction for the praised group rises and for the roasted group falls
  - the same `(productId, key)` twice is refused the second time
  - a launch creates one `launch` news item
  - crossing 1,000 customers creates one `customers` item, and only once
  - the feed never exceeds `feedCap`
  - reputation stays within 0..100
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): feedback satisfaction, real-reply posts, news and fame`.

### Task 6: The view, privacy and properties for the new state

**Files:**
- Modify: `packages/protocol/src/view.ts`, `packages/sim/src/project.ts`, `packages/sim/test/arbitraries.ts`, `packages/sim/test/{project,properties}.test.ts`, `packages/protocol/test/schemas.test.ts`
- Test: those

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces (strict zod schemas):
  - **`WorldView.world.market`** = `{adoption: number, adoptable: number}`
  - **`WorldView.feed`:** `FeedItemView[]` (the last 50 public items; posts and news about any company are public)
  - **`WorldView.spotlights`**
  - **`PublicCompany` gains:**
    - `products: {id, name, status, launchDay, version: PublicVersion}[]` (countdown and live only)
    - `prices: {free: boolean, personal: number|null, business: number|null}`
    - `customers: number` (total, rounded to 3 significant figures)
    - `reputation: number` (an integer)
    - `followers: number` (an integer)
  - **`PrivateCompany` gains:**
    - `products: ProductView[]`, where `ProductView = {id, name, versionId, status, announcedDay, launchDay, hype}`
    - `plans`
    - `customers: {total, byGroup: Record<GroupId, Record<PlanId, number>>}` (rounded integers)
    - `satisfaction: Record<productId, Record<GroupId, number>>`
    - `ledger: {current: MonthLedgerView, months: MonthLedgerView[] /* last 12 */}`
    - `fame`
  - **`VersionView.traits: Record<string, number> | null`** (own company only). `evalCompleted` gains `traits`. The training service passes `EvalJobResult.traits` (modify `apps/server/src/training/service.ts`). `PublicVersion` keeps personality words only.
- **Privacy:** another company's entry adds only the public fields above.

- [ ] **Step 1: Extend the arbitraries:** launchProduct (scored and unscored versions), setPlan, setPrice, retireProduct, `feedbackScored` with hostile replies, and evalCompleted so that scores exist.
- [ ] **Step 2: Write the failing tests:**
  - `WorldView.parse` accepts every viewer's view
  - **privacy:** for every other company, the view contains none of its exact `cash`, ledger numbers, satisfaction values or awareness, and no unannounced product names
  - determinism and JSON-snapshot replay still hold
  - money (Task 4) holds
  - the generator guard also requires at least one launch and one post across 200 worlds
- [ ] **Step 3:** Run: FAIL.
- [ ] **Step 4: Implement** `projectView` additions field by field.
- [ ] **Step 5:** Run: PASS, ten repeated runs.
- [ ] **Step 6: Commit** `feat(sim): project products, customers, ledger, feed and fame with privacy properties`.

### Task 7: Balance harness

**Files:**
- Create: `packages/sim/test/balance.test.ts`, `packages/sim/test/bots.ts`
- Modify: `packages/sim/src/balance.ts` (tuned constants)

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - `typicalFirstModel` scores, taken from the M1 gate's first models (for example `{language: 60, reasoning: 8, pattern: 80, knowledge: 44, coding: 38, creativity: 2, consistency: 37, instruction: 63}`, overall 42)
  - `runBot(balance, script, days): Report`, which feeds `step` the inputs of a scripted company: found, start, train (fake `trainingCompleted` and `evalCompleted`), launch with default plans, retrain later
  - `Report`: daily customers, cash, income and costs

- [ ] **Step 1: Write the tests on `BALANCE_V2`:**
  - with the typical first model launched with default prices, the first paying customer comes within 7 game days of launch
  - by 60 days after launch, monthly income exceeds monthly serving cost
  - customers stay below 2% of the adoptable people for the first year (a small startup)
  - a model 2× better on every score gains at least 1.5× the customers by day 90
  - doubling the personal price lowers student customers by more than business customers
  - on `BALANCE_TEST`: the first paying customer within 3 game days of launch
- [ ] **Step 2:** Run: FAIL or PASS. Then tune the `Economy` constants until all pass. Record the tuned values with one line of rationale each in `balance.ts`.
- [ ] **Step 3: Commit** `test(sim): balance harness and tuned balance-v2 economy`.

---

## Part B: server and worker

### Task 8: Feedback jobs, priorities and outdated worlds (server)

**Files:**
- Create: `apps/server/src/feedback/service.ts`
- Modify:
  - `packages/protocol/src/worker.ts` (`JOB_TYPES` adds `"feedback"`; `FeedbackJobSpec`; `FeedbackJobResult`)
  - `packages/protocol/fixtures/worker/` (`lease-feedback.json`, `result-feedback.json`)
  - `apps/server/src/jobs/broker.ts` (`JOB_PRIORITY`)
  - `apps/server/src/db/schema.ts` (+ migration `feedback_requests(product_id, key, job_id, pk(product_id,key))`)
  - `apps/server/src/app.ts`
  - `apps/server/src/worlds/routes.ts` (outdated worlds)
- Test: `apps/server/test/feedback.test.ts`; extend `apps/server/test/worlds.test.ts`

**Interfaces:**
- Consumes: Tasks 2 and 5; the broker (`onPrepare`, `onDone`, `onFailed`, `enqueue`); the registry `onLoad` hook.
- Produces:
  - **Specs and results:**
    - `FeedbackJobSpec = {productId: string, versionId: string, modelTarUrl: string, groups: GroupId[], perGroup: number, seed: number}`
    - `FeedbackJobResult = {results: {group, questionId, question, reply, score}[]}`
  - **`FeedbackService`:**
    - On each actor change, a live product of a player company gets one feedback request per game week (`key = "w" + floor(day/7)`), plus `key = "launch"` on the day it goes live.
    - Requests are deduplicated by a `feedback_requests` insert (on conflict, nothing happens).
    - The model URL is signed at lease time.
    - `done` submits `feedbackScored`. A final failure is logged and the week is skipped.
  - **`JOB_PRIORITY = {generate: 0, train: 2, eval: 3, feedback: 4}`**
  - **`GET /api/worlds`:** an entry whose actor fails to load with a sim-version error becomes `{id, name, outdated: true}`.

- [ ] **Step 1: Write the failing tests** (fake worker, `balance-test`):
  - after a launch goes live, a `feedback` lease arrives whose spec parses as `FeedbackJobSpec` with all six groups
  - the fake worker's result produces two posts in the owner's view
  - the same week does not enqueue twice across a server restart on the same `dbDir`
  - a restart mid-countdown still launches once: one `live` product and one `launch` news item
  - a chat job queued after a feedback job is leased first
  - a world snapshot rewritten to `sim-1` lists as `outdated: true` and does not break the list
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.** Add the fixtures, then check the protocol fixture test parses them.
- [ ] **Step 4:** Run: PASS (`npm test`).
- [ ] **Step 5: Commit** `feat(server): weekly customer feedback jobs, job priorities, outdated worlds`.

### Task 9: Customer questions and the feedback job (Python)

**Files:**
- Create: `ml/src/airace_ml/feedback.py`
- Modify:
  - `ml/src/airace_ml/evals/suite.py` (public `score_reply(item, reply) -> float`, wrapping the existing `_judge_reply`)
  - `ml/src/airace_ml/worker/jobs.py` (`handle_feedback`, plus `"feedback"` in `handlers()`)
  - `ml/src/airace_ml/worker/protocol.py` (`JOB_TYPES`)
- Test: `ml/tests/test_feedback.py`; extend `ml/tests/test_worker_jobs.py` and `ml/tests/test_worker_protocol.py` (the new fixtures)

**Interfaces:**
- Consumes: Task 8's spec and result shapes; the bench builders in `airace_ml.evals`.
- Produces:
  - `GROUP_CATEGORIES`:
    - students: (knowledge, reasoning)
    - creators: (instruction,)
    - developers: (coding,)
    - businesses: (instruction, knowledge)
    - everyday: (knowledge, instruction)
    - researchers: (reasoning, pattern)
  - `customer_pool(group: str) -> list[BenchItem]`: generative items only (`ExactItem`, `CheckItem`) from those categories' builders with `skill_rng(category, "customers")`, so they are distinct from the scored suite; cached
  - `pick_questions(group, n, seed) -> list[BenchItem]`: deterministic
  - `JobRunner.handle_feedback(spec, ctx) -> {"results": [...]}`:
    - the cached model answers each question with `chat_reply` (temperature 0.8, 96 tokens, seed from the spec)
    - `score_reply` scores it
    - the reply is returned exactly as generated

- [ ] **Step 1: Write the failing tests:**
  - each group's pool has at least 50 items, all generative, and none shares an `item_id` with `build_suite()`
  - `pick_questions` is stable for a seed
  - `handle_feedback` on the `trained` fixture model returns six results with scores in [0,1]
  - each result's `reply` equals `lm.chat_reply` for the same question and seed
  - the result parses against `result-feedback.json`'s schema
- [ ] **Step 2:** Run `python -m uv run --no-sync pytest tests/test_feedback.py`: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS. Full suite under 3 minutes; `ruff check` clean.
- [ ] **Step 5: Commit** `feat(ml): customer question pools and the feedback job`.

---

## Part C: the client

### Task 10: Design system and app shell

**Files:**
- Create: `apps/client/src/components/{AppShell.tsx,TopBar.tsx,SideNav.tsx,Panel.tsx,Stat.tsx,CountUp.tsx}`
- Modify: `apps/client/src/screens/Home.tsx` and `apps/client/src/net/api.ts` (a `WorldSummary.outdated` entry shows "Made with an older version of the game" and no Open button), `apps/client/src/theme/tokens.css`, `apps/client/src/route.ts` (tabs: `company`, `market`, `money`, `lab`, `play`, `bench`; `dashboard` redirects to `company`), `apps/client/src/screens/World.tsx`
- Delete: `WorldLayout` in `screens/Dashboard.tsx`; the old Dashboard is replaced in Task 12
- Test: `apps/client/test/shell.test.tsx`

**Interfaces:**
- Produces:
  - **Tokens:**
    - `--glass` (a panel fill with alpha)
    - `--glow` (`0 0 24px color-mix(in srgb, var(--brand) 35%, transparent)`)
    - `--line` (a luminous hairline in brand)
    - radius scale
  - **Panels:** `.panel` is glass, with a 1px `--line` top border and a soft glow on hover.
  - **Components:**
    - `Panel({title, hint, actions, children})`
    - `Stat({label, value, delta?, format})`: a large number with an animated count-up (`CountUp`, off under reduced motion) and a green or red delta
    - `TopBar` (logo, company name, cash, profit this month, date, speed and pause for the host)
    - `SideNav` (six screens; Arena and Rivals are added in Phase 3)
    - `AppShell({view, tab, children})`

- [ ] **Step 1: Write the failing tests:**
  - the shell shows the company name in `--brand` and cash formatted
  - the nav has the six screens, and the current one is `aria-current="page"`
  - `#/w/w1/dashboard` routes to `company`
  - Home shows an outdated world's note and no Open button
  - `CountUp` renders the final value immediately when `matchMedia('(prefers-reduced-motion: reduce)')` matches
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.** Restyle the Lab, Playground and Benchmarks screens into `Panel` (no behavior change; their tests stay green).
- [ ] **Step 4:** Run: PASS. `npm run build -w @airace/client` succeeds.
- [ ] **Step 5: Commit** `feat(client): futuristic design system and app shell`.

### Task 11: The orb

**Files:**
- Create: `apps/client/src/orb/params.ts`, `apps/client/src/orb/Orb.tsx`
- Modify: `apps/client/src/screens/lab/TrainStep.tsx` (the orb above the charts)
- Test: `apps/client/test/orb.test.ts`

**Interfaces:**
- Produces:
  - `OrbInput = {overall: number|null, scores: Record<string, number|null>|null, traits: Record<string, number>|null, trainedTokens: number, palette: Palette, training: {progress: number, lossDrop: number /* 0..1 */, mix: Record<DatasetId, number>} | null}`
  - `orbParams(input) -> {radius, brightness, rings, ringDetail, swirl, turbulence, smoothness, sparkRate, glowSoftness, flicker, colors: string[], intake: {rate, colors} | null}`
    - every field is a continuous function of the inputs (spec §4)
    - `rings = 1 + count(categories ≥ 20) + floor(log10(max(1, trainedTokens/1e6)))`
    - `colors` is the primary → secondary gradient
  - `Orb({input, size})`: a canvas render loop via requestAnimationFrame. A static frame under reduced motion, or when the canvas is unavailable.
  - `traits` comes from `VersionView.traits` (Task 6). Before the first scoring it is null, and the orb uses neutral motion.

- [ ] **Step 1: Write the failing tests:**
  - `orbParams` with `overall: null` and no version is the faint spark (`radius` < 0.3 of max, `rings` = 0)
  - radius and brightness rise strictly with `overall`
  - `rings` counts categories ≥ 20
  - higher boldness gives higher `swirl`
  - higher steadiness gives higher `smoothness`
  - `colors[0]` is the palette primary
  - `intake` is non-null exactly while training
  - a 1-point score change moves no parameter by more than 10% (continuity)
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): the AI orb, driven by real measurements`.

### Task 12: The Company screen and the feed

**Files:**
- Create: `apps/client/src/screens/Company.tsx`, `apps/client/src/components/{Feed.tsx,PostCard.tsx,NewsCard.tsx}`
- Modify: `apps/client/src/screens/World.tsx`
- Test: `apps/client/test/company.test.tsx`

**Interfaces:**
- Consumes: Task 6's view; Tasks 10–11.
- Produces:
  - **Layout** (spec §3):
    - centre: the orb with the newest version's label and personality words, and buttons Train (to the Lab), Chat (to the Playground) and Launch (opens Task 13's flow)
    - left: `Stat` for customers (total, weekly delta), a group breakdown bar, `Stat` for cash and profit this month, and a profit sparkline from `ledger.months`
    - right: the `Feed`
  - **`PostCard`:**
    - the author (spotlight name, or the handle), the group, the question, and the AI's reply in a quote block (`player-text`)
    - a reply over 280 characters folds behind "Show more" (the full text stays in the DOM)
    - the tone sets the accent
  - **`NewsCard`:** the headline, the day, and the sentiment color.

- [ ] **Step 1: Write the failing tests** (views from the sim):
  - a running world with a launched product shows its customers total, and cash equal to `formatMoney(cash)`
  - a post's reply `"<b>x</b> <|end|>"` renders as literal text, with no `<b>` element
  - a 2,000-character reply shows "Show more" and contains all 2,000 characters
  - with no versions, the orb is the spark and the centre says "Train your first AI"
  - Launch is disabled with the reason "Wait until your AI has been scored" for an unscored newest version
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): Company screen with orb, customers, money and the live feed`.

### Task 13: Launch flow, lineup and the Money screen

**Files:**
- Create: `apps/client/src/screens/{Money.tsx,LaunchDialog.tsx}`
- Modify: `apps/client/src/screens/lab/VersionsStep.tsx` (a "Launch this AI" button)
- Test: `apps/client/test/money.test.tsx`

**Interfaces:**
- Consumes: Tasks 2, 4 and 6; `GameSocket.command`.
- Produces:
  - **`LaunchDialog({view, versionId?})`:**
    - choose a scored version, a product name (2–24), which plans it powers (checkboxes), and a countdown (3–14 days, default 5)
    - sends `launchProduct`
    - shows the server's reason on reject
  - **Money screen:**
    - **Coming in:** this month and last, by plan and by group, as bars
    - **Going out:** by category (training, serving, data checks)
    - **Profit:** a chart over months
    - **Lineup:** products with status, countdown days left, and Retire
    - **Plans:** each plan's AI (a select of live products, or Closed) and its price input (bounds shown)
    - every control states what it does and its effect on customers, in plain words

- [ ] **Step 1: Write the failing tests:**
  - LaunchDialog sends exactly `{type:"launchProduct", versionId, name, plans:["free","personal"], countdownDays:5}`
  - a name of 25 characters disables Launch with "2–24 characters"
  - the Money screen shows this month's personal income from the view's ledger
  - setting the personal price sends `setPrice`; out-of-range input shows the bounds and sends nothing
  - Retire on a plan's product shows the server's "Move its plans to another AI first"
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): launch day, model lineup, plans and the Money screen`.

### Task 14: The Market screen (national, Phase 1)

**Files:**
- Create: `apps/client/src/screens/Market.tsx`
- Test: `apps/client/test/market.test.tsx`

**Interfaces:**
- Consumes: Task 6's view.
- Produces:
  - **One national card:** people willing to use AI (`world.market.adoptable`, formatted "about 33.5 million"), your customers and share, and a growth sparkline.
  - **Six group cards:** what each group cares about (plain words from the economy weights), your customers by plan, satisfaction as a face (happy, neutral, unhappy) with its percentage, and the group's spotlight person (name, handle, bio).
  - A note under the national card: "Regions arrive soon: you'll see your customers spread across the country."

- [ ] **Step 1: Write the failing tests:**
  - six group cards
  - each shows its spotlight's name as text (a hostile spotlight bio stays text)
  - satisfaction 0.8 shows the happy face and "80%"
  - the national card shows the adoptable people from the view
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): Market screen with customer groups and spotlights`.

---

## Part D: end to end

### Task 15: End-to-end, dev run and docs

**Files:**
- Modify: `tests/e2e/solo.spec.ts`, `CLAUDE.md`, `docs/progress.md`, `docs/handoff/README.md`, `docs/handoff/sdd/ledger.md`

- [ ] **Step 1: Extend the E2E** after v1 is scored:
  1. open Launch from the Company screen, name it "Spark", put it on the free and personal plans with a 3-day countdown, and launch
  2. let the clock run until it is live
  3. assert the Company customers stat becomes > 0
  4. assert the Money screen's personal income becomes > $0
  5. assert a post appears in the feed whose quoted reply is non-empty
  6. assert a `launch` news item exists
  7. reload, and the feed and customers persist
- [ ] **Step 2:** Run `npm run e2e`: FAIL, then fix real bugs, each with a unit test in the task that owns the code.
- [ ] **Step 3: Run everything:** `npm test`, `npm run check`, `npm run e2e` (twice), `cd ml && python -m uv run pytest`, `ruff check .`.
- [ ] **Step 4: Run `npm run dev` on the full corpus by hand:**
  - train a first model on the GPU, launch it, watch customers and posts arrive
  - check that posts quote real replies
  - check that chats stay snappy while feedback jobs run
- [ ] **Step 5: Update the docs:**
  - CLAUDE.md (any new command)
  - a dated `docs/progress.md` entry ("Game layer Phase 1: a real company")
  - the handoff resume point (Phase 2)
- [ ] **Step 6: Commit** `test(e2e): launch, customers, money and real-reply posts; game layer phase 1`. **Report to the owner** before Phase 2.
