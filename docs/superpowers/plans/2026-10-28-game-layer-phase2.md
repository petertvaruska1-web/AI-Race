# Game Layer Phase 2 "A Growing Business" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The company grows across the country and has to manage it. The player:
- sees customers spread over 9 US regions on a map, and regions diverge by the player's choices
- runs ad campaigns by region and group
- serves customers on rented cloud capacity that can overload after a spike
- builds data centers that make answers faster nearby, cut serving costs and lend compute to training
- gets a monthly report card with a grade, and a rescue (loan or investor) when cash runs out

**Architecture:**
- New sim modules: `campaigns`, `compute` (serving capacity and data centers), `reports` and `rescue`; `market`, `ledger`, `products`, `training` and `feed` grow. `SIM_VERSION` becomes `sim-3`.
- The server needs no new service: every new action is a command, every new effect happens in the tick, and `projectView` carries the new state (R2).
- The client gets a tile map of the 9 regions and new windows (region, campaign, data center, computers, report card, rescue), all following the immersion design.

**Tech Stack:** As Phase 1 (TypeScript, zod, Vitest, fast-check, Fastify, PGlite, React 19, Vite, Tailwind 4, Zustand, motion).

**Spec:** `docs/superpowers/specs/2026-10-27-game-layer-design.md` §5, §8, §13–§15 (Phase 2 row), plus the immersion design `docs/superpowers/specs/2026-10-28-immersion-design.md`. R1–R8 always apply.

## Global Constraints

- **Versions:** `SIM_VERSION = "sim-3"`. Worlds saved by `sim-2` are refused and listed as outdated (the existing path). `upgradeState` (a sim-2 bridge) is removed with its tests. The balance ids stay `"balance-v2"` and `"balance-test"`; their constants grow.
- **Regions:** `REGION_IDS = ["newEngland","midAtlantic","greatLakes","plains","southAtlantic","midSouth","southCentral","mountain","pacific"] as const`, with `REGION_NAMES` (protocol): New England, Mid-Atlantic, Great Lakes, Great Plains, South Atlantic, Mid-South, South Central, Mountain West, Pacific.
- **Money:** integer dollars; every cash change is booked in the same step, as income, a cost, or financing (loans, investors, repayments). Identity: `cash = foundingCash + Σincome − Σcosts + Σloans + Σinvestors − Σrepaid`.
- **Cost categories:** `["training","serving","dataChecks","marketing","dataCenterBuild","dataCenterUpkeep","interest"]`.
- **Compute unit:** PFLOP per game day. Serving one user for a day on an AI of M million parameters takes `M × servingCostPerCustomerDayPerMParam / cloudPricePerPflop` PFLOP, so cloud serving costs exactly what Phase 1 charged.
- **R1:** report cards quote posts verbatim; all other report and news text is game commentary.
- **R2:** campaigns, data centers, compute, loans, rescue offers, report cards and customers by group and region stay private. Public additions: customers by region (three significant figures) and the investors' share.
- **R7:** plain words on every control: "ad campaign", "people reached", "data center", "answered", "share for training", "report card", "emergency loan", "investor".
- **Immersion (C30):** every new number opens a window; the map, gauges and grade stamp move; new state is announced by toasts where the immersion design says so; everything is instant under reduced motion.
- **Tests:** `npm test` and `npm run check` green after every task; `npm run e2e` green at the end.

## Review Focus

1. **Spending what the company doesn't have:** a campaign or data center with too little cash, a budget or duration out of range, more than 5 campaigns, stopping an unknown campaign. Each is refused in plain words with no state change. Owners: Tasks 3 and 5.
2. **Rescue edge cases:** a month end with cash below zero while an offer is already pending, cash recovering before the deadline, choosing twice, a second rescue while a loan runs. One offer at a time; a lapsed offer leaves no financing. Owner: Task 7.
3. **Serving with nothing to serve:** zero customers, a retired product, a closed plan, owned capacity larger than demand. `served` stays 1 and nothing divides by zero. Owner: Task 4.
4. **A report card from a thin month:** no posts, no news, the first month after launch (no last month), zero customers at the start. The card exists, its grade follows the formula, and nothing crashes. Owner: Task 6.
5. **Money identity across rescues, refunds, data-center builds, upkeep and interest:** exact integers over random games. Owners: Tasks 2 and 7.

---

## Part A: the game rules (sim)

### Task 1: Nine regions and sim-3

**Files:**
- Modify: `packages/protocol/src/world.ts`, `packages/protocol/src/view.ts`, `packages/sim/src/balance.ts`, `packages/sim/src/market.ts`, `packages/sim/src/spotlights.ts`, `packages/sim/src/history.ts`, `packages/sim/src/project.ts`, `apps/server/src/config.ts`, `apps/server/src/worlds/actor.ts` (drop `upgradeState`)
- Test: create `packages/sim/test/regions.test.ts`; update `history.test.ts`, `project.test.ts`, `properties.test.ts`, `apps/server/test/actor.test.ts` (drop the sim-2 bridge test), protocol `schemas.test.ts`

**Interfaces:**
- Produces:
  - **Protocol:** `REGION_IDS` as above, `RegionId`, `REGION_NAMES: Record<RegionId, string>`.
  - **Economy:**
    - `population` (people): newEngland 15_100_000, midAtlantic 42_000_000, greatLakes 47_200_000, plains 21_800_000, southAtlantic 68_700_000, midSouth 19_800_000, southCentral 43_000_000, mountain 25_500_000, pacific 53_000_000.
    - `groupMix[region]`: the national mix of Phase 1 times these multipliers, normalised to sum 1 (unlisted groups ×1): newEngland students 1.4, researchers 1.8, developers 1.1; midAtlantic businesses 1.5, creators 1.2, researchers 1.2; greatLakes businesses 1.1; plains everyday 1.15, developers 0.7, researchers 0.7; southAtlantic businesses 1.1; midSouth everyday 1.2, creators 0.9, developers 0.6, researchers 0.6; southCentral businesses 1.2; mountain developers 1.2; pacific developers 1.9, creators 1.4, researchers 1.3, everyday 0.85.
    - `neighbors: Record<RegionId, RegionId[]>` (symmetric): newEngland–midAtlantic; midAtlantic–greatLakes, southAtlantic; greatLakes–plains, midSouth, southAtlantic; plains–midSouth, southCentral, mountain; southAtlantic–midSouth; midSouth–southCentral; southCentral–mountain; mountain–pacific.
    - `diffusion.neighborSpill = 0.15`: word of mouth also reaches each neighbouring region's same group at this fraction.
  - **Market:** `customersByRegion(m, companyId): Record<RegionId, number>`; word of mouth spills to neighbours.
  - **Spotlights:** each spotlight's region is drawn from the world rng, weighted by the region's people in its group.
  - **History:** `DaySample.regions: Record<RegionId, number>` (whole customers).
  - **View:** `world.market.regions: Record<RegionId, {adoptable: Count}>`; `PublicCompany.customersByRegion: Record<RegionId, Count>` (three significant figures); `PrivateCompany.customers.byRegion: Record<RegionId, Record<GroupId, Count>>`; `DaySampleView.regions`.

- [ ] **Step 1: Write the failing tests:**
  - populations sum to 336.1 million; each region's mix sums to 1 (±1e-9); `neighbors` is symmetric and every region has one
  - `adoptable(e, day, "pacific", "developers")` equals population × adoption × mix
  - a launch gives customers in all nine regions, and the Pacific's developer share of its customers is above the Mid-South's
  - customers seeded only in the Pacific raise awareness in Mountain West after a day, and not in New England
  - spotlight regions are region ids, the same for the same seed
  - the view: `world.market.regions` sum to `world.market.adoptable` (±1 per region); public `customersByRegion` is rounded to three figures; private `byRegion` and history `regions` sum to the totals
  - privacy property: `byRegion` appears once (own company) or not at all
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement**; set `SIM_VERSION = "sim-3"`; remove `upgradeState` and its two tests.
- [ ] **Step 4:** Run `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): nine US regions with their own people, mixes and neighbours; sim-3`.

### Task 2: The ledger grows

**Files:**
- Modify: `packages/protocol/src/world.ts` (cost categories), `packages/protocol/src/view.ts`, `packages/sim/src/ledger.ts`, `packages/sim/src/state.ts`, `packages/sim/src/project.ts`
- Test: `packages/sim/test/ledger.test.ts`, the money property in `properties.test.ts`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `COST_CATEGORIES` as in the Global Constraints.
  - `MonthLedger` gains `revenueByRegion: Record<RegionId, number>`, `customersEndByGroup: Record<GroupId, number>` (whole customers at the close), `financing: {loans: number, investors: number, repaid: number}`.
  - `book(company, category, amount, detail?: {group?, plan?, region?})`; `ledgerDay` books revenue per (region, group, plan).
  - `finance(company, kind: "loans" | "investors" | "repaid", amount: number): void`: rounds once; loans and investors add to cash, repaid takes from it; books into `current.financing`.
  - `MonthLedgerView` mirrors the three new fields.

- [ ] **Step 1: Write the failing tests:**
  - a month's `revenueByRegion` sums to its income, and so does `revenueByPlan`
  - `customersEndByGroup` at the close matches the market, and sums to `customersEnd` (±6 for rounding)
  - `finance` moves cash and the month's financing together; repaid takes cash away
  - the property: `cash = founding + Σincome − Σcosts + Σloans + Σinvestors − Σrepaid` over random games
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): revenue by region, customers by group at month end, financing lines`.

### Task 3: Ad campaigns

**Files:**
- Create: `packages/sim/src/campaigns.ts`
- Modify: `packages/protocol/src/commands.ts` (or wherever `Command` lives), `packages/sim/src/balance.ts`, `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts` (commands; tick calls `campaignsDay` before `marketDay`), `packages/sim/src/project.ts`, `packages/protocol/src/view.ts`, `packages/sim/test/arbitraries.ts`
- Test: create `packages/sim/test/campaigns.test.ts`; privacy property

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
  - **Commands:** `startCampaign {region: RegionId | "all", group: GroupId | "all", budgetPerDay: int, days: int}` (needs `newId`); `stopCampaign {campaignId}`.
  - **Balance:** `campaigns = {budget: {min: 100, max: 100_000}, days: {min: 7, max: 90}, maxRunning: 5, costPerReach: 0.05, awarenessPerReach: 0.5, hypePerDollar: 0.000002}` (tuned in Task 8).
  - **State:** `CompanyState.campaigns: Record<string, CampaignState>`, `CampaignState = {id, region: RegionId | "all", group: GroupId | "all", budgetPerDay, startDay, endDay /* startDay + days */, spent}`.
  - **`campaignsDay(s)`:** for each running campaign (ids sorted): books `marketing` = `budgetPerDay`; reaches `budgetPerDay / costPerReach` people, split over its target (region, group) cells by their adoptable people; each cell's awareness `a += (1 − a) × min(1, cellReach / cellPeople) × awarenessPerReach`; each product in countdown gains `budgetPerDay × hypePerDollar` hype; after spending on `day === endDay` the campaign is removed. A campaign spends on exactly `days` ticks.
  - **Refusals:** no product in countdown or live ("Launch an AI before advertising it."); budget or days out of range (the range in words); `maxRunning` reached ("You can run up to 5 campaigns at once."); cash below one day's budget ("You need at least $X for a day of this campaign."); unknown campaign ("That campaign isn't running.").
  - **View:** `PrivateCompany.campaigns: CampaignView[]` (the state's fields), oldest first.

- [ ] **Step 1: Write the failing tests:**
  - each refusal, with no state change
  - a 10-day $1,000 campaign books $10,000 of marketing over exactly 10 ticks, then is gone
  - a Pacific developers campaign raises awareness only in that cell, and after 30 days the Pacific has more developer customers than a twin world without it
  - "all"/"all" splits its reach over every cell by people
  - `stopCampaign` stops the spending at once
  - a campaign during a countdown adds hype
  - privacy: `campaigns` appears once or not at all
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement**; add both commands to the property arbitraries. **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): ad campaigns by region and group`.

### Task 4: Serving capacity: the cloud ramp and overload

**Files:**
- Create: `packages/sim/src/compute.ts`
- Modify: `packages/sim/src/balance.ts`, `packages/sim/src/state.ts`, `packages/sim/src/market.ts` (`productAppeal` takes `served`), `packages/sim/src/ledger.ts` (serving moves out of `ledgerDay`), `packages/sim/src/feed.ts` (export `tellNews`; topic `overload`), `packages/protocol/src/world.ts` (`NEWS_TOPICS` gains `overload`, `dataCenter`, `rescue`), `packages/sim/src/reducer.ts` (tick: `computeDay` right after `marketDay`), view and projection
- Test: create `packages/sim/test/compute.test.ts`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - **Balance:** `compute = {cloudStartPflops: 0.00025, cloudGrowthPerDay: 0.2, ownedServingPriceShare: 0.3, overloadSatisfactionHit: 0.05, overloadNewsBelow: 0.9, overloadNewsGapDays: 30}`; `utility.reliability = 3`.
  - **State:** `CompanyState.compute = {cloudPflops, trainingShare /* 0..1, Task 5 */, loadPflops, served /* 0..1 */, lastOverloadNews: number | null}`; new companies start with `cloudPflops = cloudStartPflops`, `trainingShare = 0`, `served = 1`.
  - **`servingLoad(s, companyId): number`:** PFLOP needed today (users × params of the product behind their plan, business seats counted, per the unit rule).
  - **`ownedPflops(company, day): number`:** 0 until Task 5 adds data centers.
  - **`computeDay(s)`:** owned for serving `O = ownedPflops × (1 − trainingShare)`; `need = max(0, load − O)`; `cloud = min(need, max(cloudStartPflops, cloud × (1 + cloudGrowthPerDay)))`; `served = load > 0 ? min(1, (O + cloud) / load) : 1`; books `serving` = `min(cloud, need) × cloudPricePerPflop + min(O, load) × cloudPricePerPflop × ownedServingPriceShare`; when `served < 1`, every satisfaction of the company's products falls by `(1 − served) × overloadSatisfactionHit` (not below 0); news (topic `overload`, sentiment −0.5) "{Company}'s AI buckles under demand: only {n}% of questions answered" when `served < overloadNewsBelow` and no overload story in the last 30 days.
  - **Appeal:** `productAppeal(…, served)` adds `utility.reliability × (served − 1)`; the market uses the company's `served` from yesterday.
  - **`tellNews(s, company, topic, headline, sentiment)`** exported from `feed.ts`.
  - **View:** `PrivateCompany.compute = {cloudPflops, ownedPflops, trainingShare, loadPflops, served}`.

- [ ] **Step 1: Write the failing tests:**
  - with demand under the cloud start, `served` is 1 and serving costs what Phase 1 charged (to the dollar)
  - customers tripled overnight: `served < 1` next day, satisfaction falls, one overload story, then the cloud grows 20% a day until `served` is 1 again
  - an overloaded company wins fewer new customers than its un-overloaded twin
  - zero customers, a retired product or a closed plan: `served` stays 1, no NaN, no cost
  - privacy: `compute` appears once or not at all
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): cloud serving capacity that ramps up and overloads after a spike`.

### Task 5: Data centers and the share for training

**Files:**
- Modify: `packages/sim/src/compute.ts`, `packages/sim/src/balance.ts`, `packages/sim/src/state.ts`, `packages/sim/src/market.ts` (nearby term), `packages/sim/src/training.ts` (`previewPlan` takes owned compute), `packages/sim/src/reducer.ts` (commands; startTraining passes the owned compute), commands, view and projection, arbitraries
- Test: `packages/sim/test/compute.test.ts`, `packages/sim/test/training.test.ts`

**Interfaces:**
- Consumes: Task 4.
- Produces:
  - **Commands:** `buildDataCenter {region}` (needs `newId`); `setTrainingShare {share: number}` (0..1, stored rounded to 2 decimals).
  - **Balance:** `dataCenter = {cost: 1_500_000, buildDays: 60, pflopsPerDay: 0.05, upkeepPerMonth: 30_000, maxPerCompany: 6}`; `utility.nearby = 0.6`; `nearby = {region: 1, neighbor: 0.4}`.
  - **State:** `CompanyState.dataCenters: Record<string, {id, region, startDay, openDay /* startDay + buildDays */}>`.
  - **Rules:**
    - building books `dataCenterBuild` = cost at once; refused when cash < cost ("A data center costs $1,500,000. You have $X.") or at `maxPerCompany`
    - on `openDay`: news (topic `dataCenter`, sentiment 0.5) "{Company} opens a data center in {Region}"
    - each day an open data center books `upkeepPerMonth / 30` as `dataCenterUpkeep`
    - `ownedPflops` = open centers × `pflopsPerDay`; serving uses `(1 − trainingShare)` of it (Task 4)
    - `closeness(company, region)`: 1 with an open center in the region, 0.4 in a neighbour, else 0; `productAppeal` adds `utility.nearby × closeness × (1 + max(0, speedPreference))`
    - `previewPlan(b, parent, plan, ownedPflopsPerDay = 0)`: days `= ceil(pflop / (cloudPflopsPerDay + owned))`; cost `= round(pflop × cloudPricePerPflop × cloud / (cloud + owned))` plus fact-checking; `startTraining` passes `ownedPflops × trainingShare`
  - **View:** `PrivateCompany.dataCenters: {id, region, startDay, openDay, open}[]`.

- [ ] **Step 1: Write the failing tests:**
  - build refusals; a build books the cost and opens on day `startDay + 60`, with one news story
  - upkeep is booked every day once open (30 days ≈ $30,000)
  - with an open center, serving costs less than its twin's for the same customers
  - an open center in the Pacific raises Pacific (and Mountain West) customers against a twin
  - share 0.5 with one open center: the next run's preview is shorter and cheaper (exact numbers from the formula), and serving headroom halves
  - `setTrainingShare` refuses −0.1, 1.5 and NaN
  - privacy: `dataCenters` appears once or not at all
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): data centers that serve faster and cheaper nearby, and lend compute to training`.

### Task 6: The monthly report card

**Files:**
- Create: `packages/sim/src/reports.ts`
- Modify: `packages/sim/src/ledger.ts` (`closeMonths` calls `reportMonth` after closing), `packages/sim/src/state.ts`, view and projection
- Test: create `packages/sim/test/reports.test.ts`

**Interfaces:**
- Consumes: Tasks 2 and 4.
- Produces:
  - **State:** `CompanyState.reports: ReportCard[]` (last 24, newest last); `ReportCard = {month, revenue, costs, profit, lastProfit: number | null, customersStart, customersEnd, growthByGroup: Record<GroupId, number>, satisfaction: number | null, best: MomentPost | null, worst: MomentPost | null, headline: string | null, grade: "A" | "B" | "C" | "D" | "F"}`; `MomentPost = {author: {name, handle, group}, productName, question, reply, tone}` (copied verbatim, R1).
  - **`reportMonth(s, company, closed: MonthLedger, previous: MonthLedger | undefined)`:** makes a card only once the company has launched a product before the close. `customersStart` is the previous month's `customersEnd` (0 without one); growth by group from `customersEndByGroup`. Satisfaction: the customer-weighted mean over the company's live products and groups (null without customers). Best and worst: the company's posts from that month with the highest and lowest sentiment (ties: earliest; worst is null when it would be the best). Headline: the company's story of that month with the largest |sentiment|.
  - **Grade:** points = profit (1 if profit > 0, plus 1 if profit > lastProfit, or if lastProfit is null and profit > 0) + growth (1 if customers grew 10% or more, or from 0 to more; 0.5 if they grew at all) + mood (1 if satisfaction ≥ 0.6, 0.5 if ≥ 0.45). A ≥ 3.5, B ≥ 2.5, C ≥ 1.5, D ≥ 0.5, else F.
  - **View:** `PrivateCompany.reports: ReportCardView[]` (last 12).

- [ ] **Step 1: Write the failing tests:**
  - no card before the first launch; a card at the first close after it, with revenue, costs and profit equal to the closed month's
  - grade cases for A, C and F from constructed months
  - best and worst quote their replies exactly (a hostile reply stays byte-identical); a month without posts or news gives nulls
  - the headline is the strongest story; cards are capped at 24
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): a monthly report card with a grade and the month's best and worst moments`.

### Task 7: The rescue

**Files:**
- Create: `packages/sim/src/rescue.ts`
- Modify: `packages/sim/src/balance.ts`, `packages/sim/src/state.ts`, `packages/sim/src/ledger.ts` (month close: interest and repayment, then the offer), `packages/sim/src/reducer.ts` (command; tick calls `rescueDay` after the ledger), commands, view and projection, arbitraries
- Test: create `packages/sim/test/rescue.test.ts`; the money and privacy properties

**Interfaces:**
- Consumes: Tasks 2 and 6.
- Produces:
  - **Command:** `chooseRescue {kind: "loan" | "investor"}`.
  - **Balance:** `rescue = {cushion: 100_000, loanMonthlyRate: 0.04, loanTermMonths: 24, investorStake: 0.25, decideDays: 14}`.
  - **State:** `rescue: {offeredDay, deadlineDay} | null`, `loan: {balance, principalPerMonth} | null`, `ownership: number` (1 at founding).
  - **Rules:**
    - a month close runs in this order: close the ledger month, make the report card, take the loan's monthly payment (booked in the new month), then check for an offer
    - with cash < 0 and no offer pending, an offer opens for `decideDays`; news (topic `rescue`, −0.6) "{Company} runs out of cash"
    - `chooseRescue`: refused without an offer ("There is no rescue to choose."); amount = `cushion − cash`; a loan books `finance("loans")`, adds to `loan.balance` and sets `principalPerMonth = ceil(balance / loanTermMonths)` (news −0.3 "{Company} takes an emergency loan"); an investor books `finance("investors")` and multiplies `ownership` by `1 − investorStake`, rounded to 4 decimals (news −0.2 "Investors take a 25% stake in {Company}"); the offer closes
    - `rescueDay`: at the deadline, the loan is taken if cash is still below zero; otherwise the offer lapses
    - each month close with a loan: interest `round(balance × loanMonthlyRate)` booked as `interest`, then `min(balance, principalPerMonth)` repaid through `finance("repaid")`; the loan ends at 0
  - **View:** private `rescue: {deadlineDay} | null`, `loan: {balance, principalPerMonth, monthlyRate} | null`; public `ownership`.

- [ ] **Step 1: Write the failing tests:**
  - an offer only when cash < 0 at a month end, one at a time
  - each choice restores cash to the cushion and books it once; choosing twice is refused
  - the deadline takes the loan, or lets the offer lapse if cash recovered
  - monthly interest and repayment until the loan ends; a second loan adds to the balance
  - the properties: money identity with financing; `0 < ownership ≤ 1`; a company is never removed
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): rescue by emergency loan or investor when cash runs out`.

### Task 8: Balance harness for a growing business

**Files:**
- Modify: `packages/sim/test/bots.ts` (scripts can start campaigns, build data centers and set the training share), `packages/sim/test/balance.test.ts`, `packages/sim/src/balance.ts` (retuned values, one-line reason each)

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces these targets on `balance-v2`:
  1. Phase 1's targets still hold.
  2. **Regions diverge:** a $2,000-a-day, 30-day Pacific campaign from launch day raises the Pacific's share of the company's customers 1.3 times or more against the same bot without it, 90 days after launch.
  3. **A data center pays at scale:** with 500,000 customers seeded, serving per customer costs at least 40% less with one open center than without, and the center's region gains share.
  4. **Overload recovers:** customers tripled overnight give `served < 1` the next day and `served = 1` within 10 days.
  5. **Grades spread:** the bots' report cards over 12 months show at least four different letters.
  6. **Rescue works:** a spendthrift bot (five $10,000-a-day campaigns) gets an offer, the loan is taken at the deadline, cash is back above zero, and the loan shrinks month by month.

- [ ] **Step 1:** Write the six targets as tests. **Step 2:** Run: some FAIL. **Step 3:** Retune until all pass; record each change with its reason. **Step 4:** Run `npm test`: PASS.
- [ ] **Step 5: Commit** `test(sim): balance targets for regions, campaigns, data centers, report cards and rescue`.

---

## Part B: the screens (client)

Every new window plugs into the window stack (`WindowSpec` gains `region`, `campaign`, `dataCenter`, `computers`, `report`, `rescue`) and follows the immersion design.

### Task 9: The region map and the Market screen

**Files:**
- Create: `apps/client/src/regions.ts` (tile layout), `apps/client/src/components/RegionMap.tsx`, `apps/client/src/windows/RegionWindow.tsx`
- Modify: `apps/client/src/screens/Market.tsx`, `apps/client/src/screens/Company.tsx` (a small map in the customers panel), `apps/client/src/store/windows.ts`, `apps/client/src/windows/WindowHost.tsx`
- Test: create `apps/client/test/regions.test.tsx`; extend `market.test.tsx` and `company.test.tsx`

**Interfaces:**
- Consumes: Task 1's view.
- Produces:
  - **Tiles:** `STATE_TILES: {state: string, col: number, row: number, region: RegionId}[]`, the 50 states and DC on a 12 × 8 grid: AK 0,0; ME 11,0; WI 6,1; VT 10,1; NH 11,1; WA 1,2; ID 2,2; MT 3,2; ND 4,2; MN 5,2; IL 6,2; MI 7,2; NY 9,2; MA 10,2; OR 1,3; NV 2,3; WY 3,3; SD 4,3; IA 5,3; IN 6,3; OH 7,3; PA 8,3; NJ 9,3; CT 10,3; RI 11,3; CA 1,4; UT 2,4; CO 3,4; NE 4,4; MO 5,4; KY 6,4; WV 7,4; VA 8,4; MD 9,4; DE 10,4; AZ 2,5; NM 3,5; KS 4,5; AR 5,5; TN 6,5; NC 7,5; SC 8,5; DC 9,5; OK 4,6; LA 5,6; MS 6,6; AL 7,6; GA 8,6; HI 0,7; TX 4,7; FL 9,7 (census divisions for regions).
  - **`RegionMap({values: Record<RegionId, number> /* 0..1 */, color, onOpen?, marks?: Partial<Record<RegionId, {campaign?: boolean; dataCenter?: "building" | "open"}>>, small?: boolean})`:** rounded tiles tinted by value; a region lifts and glows under the pointer with a readout; one button per region (`Open {Region name}`); campaigns pulse; data centers show a glowing mark (a ring while building).
  - **Market:** the map leads "Across the country", tinted by your share of each region's willing people; the region readout shows customers and share.
  - **`RegionWindow`:** people and willing people, your customers and share, customers by group, this month's revenue from the region, data centers and campaigns there, closeness in words ("Fast answers: a data center is here" / "nearby" / "Answers come from the cloud"), the region's customers day by day, and an "Advertise here" button (Task 10's window).
  - **Company:** a small map under the customer count; a region opens its window.

- [ ] **Step 1: Write the failing tests:** 51 tiles in 9 regions; nine region buttons; tint follows share; the region window shows that region's numbers from the view; the Market and Company maps open region windows.
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): a map of the nine regions on the Market and Company screens, and region windows`.

### Task 10: Ad campaigns on screen

**Files:**
- Create: `apps/client/src/windows/CampaignWindow.tsx`
- Modify: `apps/client/src/screens/Market.tsx` (an "Ad campaigns" panel), the window store and host
- Test: create `apps/client/test/campaigns.test.tsx`

**Interfaces:**
- Consumes: Task 3's commands and view; Task 9's map.
- Produces:
  - **Panel:** each running campaign with region, group, dollars a day, a bar of days left and the amount spent, and a Stop button; "New ad campaign" opens the campaign window. Campaign regions pulse on the map.
  - **`CampaignWindow({region?})`:** region (or everywhere), group (or everyone), budget a day (a slider with $500, $2,000 and $10,000 marks), days (7–90); live words: "Reaches about N people a day · $X in total"; Start sends `startCampaign`; a refusal shows its reason in the window.

- [ ] **Step 1: Write the failing tests:** the form sends the exact command; a refusal shows; Stop sends `stopCampaign`; the panel shows days left and spend.
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): run ad campaigns from the Market screen`.

### Task 11: Computers and data centers

**Files:**
- Create: `apps/client/src/components/Computers.tsx` (the panel), `apps/client/src/windows/ComputersWindow.tsx`, `apps/client/src/windows/DataCenterWindow.tsx`
- Modify: `apps/client/src/screens/Money.tsx`, `apps/client/src/money.ts` (names and explanations of the new categories), `apps/client/src/windows/MoneyWindows.tsx`, `apps/client/src/screens/lab/LabLayout.tsx` and `DesignStep.tsx` (previews pass the owned compute)
- Test: extend `money.test.tsx`; create `apps/client/test/computers.test.tsx`

**Interfaces:**
- Consumes: Tasks 2, 4 and 5.
- Produces:
  - **Money:** coming in by region (rows open region windows); the new cost rows with plain explanations; financing shown under the month ("Loans", "Investors", "Repaid").
  - **`Computers` panel:** an animated ring of the share of questions answered; capacity as owned and cloud against today's demand, in customers of your biggest AI ("room for about N more customers"); the share-for-training slider (sends `setTrainingShare` when released); data centers, each building one with a bar and days left; "Build a data center" opens its window.
  - **`DataCenterWindow`:** region, cost, build time, upkeep and capacity in plain words, the closeness effect, Build (sends `buildDataCenter`); refusals show.
  - **Lab:** the training preview counts the share for training ("Your data centers make this run N days faster and $X cheaper").

- [ ] **Step 1: Write the failing tests:** the new rows; the slider sends `setTrainingShare`; the build window sends `buildDataCenter` and shows a refusal; a building center shows days left; a preview with owned compute is shorter and cheaper.
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): computers, data centers and the share for training`.

### Task 12: Report cards on screen

**Files:**
- Create: `apps/client/src/windows/ReportWindow.tsx`
- Modify: `apps/client/src/screens/Money.tsx` (a "Report cards" panel), `apps/client/src/screens/Company.tsx` (the last grade), `apps/client/src/components/Toasts.tsx` (a new card is announced), the window store and host
- Test: create `apps/client/test/reports.test.tsx`

**Interfaces:**
- Consumes: Task 6's view.
- Produces:
  - **Panel:** one tile per card with its month and grade, newest first; a tile opens the card.
  - **`ReportWindow({month})`:** the grade as a stamp that lands; revenue, costs and profit against last month; customers from start to end and growth by group; the best and worst moments as quoted posts (verbatim, as text); the headline; a word on what drove the grade.
  - **Toast:** a new card pops up as "Your {Month} report card: {grade}" and opens the card.
  - **Company:** the latest grade beside the money, opening the card.

- [ ] **Step 1: Write the failing tests:** the window shows the grade and quotes replies exactly (hostile text stays text); a toast only for a new card; tiles and the Company grade open the window.
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): monthly report cards with a grade stamp and the month's moments`.

### Task 13: Rescue and overload in the shell

**Files:**
- Create: `apps/client/src/windows/RescueWindow.tsx`
- Modify: `apps/client/src/components/TopBar.tsx` (overload chip), `apps/client/src/components/AppShell.tsx` (rescue banner), `apps/client/src/screens/Money.tsx` (loan and ownership), `apps/client/src/screens/Company.tsx` (ownership), the window store and host
- Test: extend `shell.test.tsx`; create `apps/client/test/rescue.test.tsx`

**Interfaces:**
- Consumes: Tasks 4 and 7.
- Produces:
  - **Banner:** while an offer is pending, "Your cash ran out. Choose a rescue within N days" under the top bar, opening the rescue window.
  - **`RescueWindow`:** two cards, the emergency loan (amount, monthly interest, repayment a month) and the investor (amount, their 25% share, what you keep); choosing sends `chooseRescue`.
  - **Overload chip:** "Servers overloaded · N% answered" in the top bar while `served < 1`, pulsing, opening the computers window.
  - **Money and Company:** the loan's balance and repayment; "You own N%" once investors hold a share.

- [ ] **Step 1: Write the failing tests:** the banner and chip appear and disappear with the view; the rescue window sends each choice; ownership shows only below 100%.
- [ ] **Step 2:** Run: FAIL. **Step 3: Implement.** **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): rescue offers and overload warnings in the shell`.

---

## Part C: end to end

### Task 14: End-to-end, docs and the final review

**Files:**
- Modify: `tests/e2e/solo.spec.ts`, `CLAUDE.md`, `docs/progress.md`, `docs/handoff/README.md`, `docs/handoff/sdd/ledger.md`

- [ ] **Step 1: Extend the E2E** after the launch:
  1. the Market map shows customers in at least two regions
  2. start a 7-day $500 campaign in the Pacific from the campaign window; the panel lists it
  3. the Money screen's marketing cost becomes more than $0
  4. the month closes (the test balance runs a month in seconds): a report card toast appears and its window shows a grade
  5. the computers panel shows 100% answered
- [ ] **Step 2:** Run `npm run e2e`: FAIL, then fix real bugs, each with a unit test in the task that owns the code.
- [ ] **Step 3: Run everything:** `npm test`, `npm run check`, `npm run e2e` (twice).
- [ ] **Step 4: Play it in the browser on the GPU** (a second game beside the owner's if theirs is running): campaign, map, report card, and a data center with seeded cash if needed.
- [ ] **Step 5: Final review** of the whole branch range; one fix pass for Critical and Important findings, each with a failing test first.
- [ ] **Step 6: Docs:** progress entry, handoff resume point (Phase 3 "Rivals"), ledger; commit and push; **report to the owner** with the rulings and deferred minors.

## Deferred to later phases

- Report-card rank changes against rivals (Phase 3, when rivals exist).
- Data-quality scandals and other consequence events beyond overload (Phase 4 with discoveries).
