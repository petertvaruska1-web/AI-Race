# Game Layer Phase 4 "Surprises and Goals" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The player's AI surprises them, and the game gives them something to aim for. The player can:
- interview their AI in front of the press, choosing which real answers to publish
- see discoveries the scoring finds in their AI ("Surprisingly good at counting"), with the real question and reply as proof, and publish them as news
- write a small notebook of their own that joins training as a ninth dataset
- collect milestones on a trophy shelf, each one earned by a measurement
- optionally pick a goal when starting a world, follow it on the Company screen, and get a victory screen with the company's story

**Architecture:**
- **ML:**
  - The eval job also reports the signals milestones need, the tag breakdown, and discoveries. Discoveries compare this version with the one before and with the other AIs in the world.
  - A new, cheap `interview` job scores published answers: a check where the prompt card has one, otherwise how fluent the answer reads.
  - Prompt cards are exported in the catalog.
- **Sim (`sim-5`):**
  - New modules: `milestones`, `discoveries`, `interviews` and `goal`.
  - The training plan gains a `notebook` share; versions remember whether they learned from unchecked web data, which the data-quality scandal needs (deferred from Phase 2).
- **Server:**
  - The training service sends the previous version and the world's AIs to each eval, and passes discoveries into the world.
  - An interview service: interviews are chats kept on the server, so a player can only publish what the AI really said.
  - A notebook service: private text, filtered, stored by hash.
  - World creation takes a goal.
- **Client:**
  - the Playground's Interview mode
  - discoveries in the AI window
  - the notebook in the Data step
  - the trophy shelf, the goal picker, goal progress and the victory screen

**Tech Stack:** As Phase 3.

**Spec:** `docs/superpowers/specs/2026-10-27-game-layer-design.md`:
- §10 (interviews, discoveries, notebook) and §11 (milestones, goals)
- §7 (consequence events: the data-quality scandal) and §13–§15 (Phase 4 row)

Also architecture spec §4.9 (discoveries) and the immersion design. R1–R8 always apply.

**Requires:** Phase 3 complete. The ranking tables, battles and rivals are used by the milestones and goals here.

## Global Constraints

- **Versions:** `SIM_VERSION = "sim-5"`. `upgradeState` gains the step from sim-4, and sim-3 worlds chain through it. Older versions are refused as today.
- **Mix:** `MIX_IDS = [...DATASET_IDS, "notebook"] as const`. `TrainingPlan.mix` is keyed by `MIX_IDS`; `notebook` defaults to 0.
- **Jobs:** `JOB_TYPES` gains `"interview"` (priority 1; `slots.interview` defaults to 0).
- **News topics:** `NEWS_TOPICS` gains `"interview"`, `"discovery"`, `"goal"` and `"scandal"`.
- **Milestones:** `MILESTONE_IDS` with `MILESTONE_NAMES`:

  | id | name |
  |----|------|
  | firstSentence | "First full sentence" |
  | firstFact | "First correct fact" |
  | firstProgram | "First working program" |
  | customers1k | "First 1,000 customers" |
  | customers100k | "100,000 customers" |
  | customers1m | "1,000,000 customers" |
  | firstPaying | "First paying customer" |
  | firstProfit | "First profitable month" |
  | numberOne | "#1 in a skill" |
  | beatArch | "Beat the arch-rival in a battle" |
  | dataCenter | "A data center opened" |
  | gradeA | "An A on the report card" |

- **Goals:**
  - kinds: `["none","customers","profit","bestAi","beatArch","value"]`
  - default targets: 1,000,000 customers; $1,000,000 profit in a month; $100,000,000 company value
  - deadlines: none, or 1, 2 or 5 game years of 365 days
  - company value counts only the player's share (spec §8)
- **R1:** interview answers and discovery evidence are quoted verbatim inside game-written frames. Headlines are game text.
- **R2:**
  - The notebook's text reaches only this company's training jobs. The sim logs only its hash and word count, and no log prints it.
  - Discoveries and unpublished interviews are their owner's only.
- **R5:** milestones fire only from measurements or state. Discoveries come from measured differences, with no fixed list. The trophy shelf shows earned milestones only: no list of things to unlock.
- **R7** (plain words): "interview", "prompt card", "publish", "discovery", "notebook", "milestone", "trophy shelf", "goal", "company value".
- **Immersion:**
  - Discoveries, interviews, milestones and goal progress open windows.
  - A new milestone bursts like the existing celebrations. A new discovery arrives as a toast.
  - The victory screen animates its story.
- **Tests:**
  - `npm test` and `npm run check` are green after every task.
  - ML tasks also need the ML suite and ruff green.
  - `npm run e2e` is green at the end.

## Review Focus

1. **Missing measurements:**
   - Cases: no judge (legibility unknown, fluency without the judge), a model that gets nothing right, an eval with no previous version or too few peers.
   - Expected: no milestone from a missing measurement, no discovery from too little data, no crash.
   - Owners: Tasks 1, 2 and 4.
2. **Interview edge cases:**
   - Cases: publishing 0 or 6 answers, an index that points at the player's own message, another player's thread, the cooldown, the product retired before scoring, the scoring job failing.
   - Expected: each refused in plain words, or nothing changes.
   - Owners: Tasks 6 and 10.
3. **Notebook abuse:**
   - Cases: 5,001 words; links, emails or phone numbers; `<|end|>`; banned words; an empty notebook with a share above 0.
   - Expected: refused in plain words; the text never appears in a view or a log.
   - Owners: Tasks 3 and 11.
4. **Goal edge cases:**
   - Cases: a rival goal in a world without rivals, the deadline day being the day it is reached, reaching it twice, company value with negative cash or after an investor.
   - Owner: Task 7.
5. **Upgraded worlds (sim-4 → sim-5):**
   - Old plans gain notebook 0.
   - State-based milestones catch up the next day, but measurement milestones wait for the next eval (no invented history).
   - Replay is exact.
   - Owners: Tasks 3 and 4.

---

## Part A: measurements and content (ML)

### Task 1: The eval reports milestone signals, the tag breakdown and discoveries

**Files:**
- Create: `ml/src/airace_ml/discoveries.py`, `ml/tests/test_discoveries.py`
- Modify: `packages/protocol/src/worker.ts` (eval spec and result), `packages/protocol/fixtures/worker/` (eval lease and done fixtures), `ml/src/airace_ml/worker/jobs.py` (`handle_eval`)
- Test: `ml/tests/test_worker_jobs.py`, the protocol fixtures test

**Interfaces:**
- Produces:
  - **`EvalJobSpec` gains** (each with a default, so older specs still parse):
    - `previous: {label: string, breakdown: Breakdown, traits: Traits} | null` (default null)
    - `peers: {breakdown: Breakdown, traits: Traits}[]` (default `[]`)
    - `thresholds: {minPoints: number, minZ: number, minPeers: int, perEval: int}` (default `{25, 2, 3, 3}`)
    - `Breakdown = record<string, [number, int]>`
  - **`EvalJobResult` gains:**
    - `signals: {legibleReplies: int | null, correctFacts: int, passingPrograms: int}`
    - `breakdown: Breakdown`
    - `discoveries: Discovery[]`
  - **`Discovery`:** `{key: string, kind: "strength" | "weakness" | "better" | "worse" | "changed", about: string, title: string, category: string, tag: string, value: number, compared: number, evidence: {prompt: string, reply: string} | null}`.
  - **Signals:**
    - `legibleReplies`: the fingerprint's sample replies that pass `is_well_formed` under the installed judge; null without one.
    - `correctFacts`: knowledge results of free-answer (`ExactItem`) items scoring 1.
    - `passingPrograms`: coding results of written-function (`CheckItem`) items scoring 1.
  - **`find_discoveries(report: BenchReport, fingerprint: Fingerprint, previous: dict | None, peers: list[dict], thresholds: dict) -> list[Discovery]`**, over each tag in `report.tag_breakdown()` with at least the existing minimum item count, and over each trait:
    - **Against the previous version** (when it has the tag): `d = value − previous`. `d ≥ minPoints` is a "better"; `d ≤ −minPoints` is a "worse"; `compared` is the previous value.
    - **Against peers** (when at least `minPeers` peers have the tag): mean m and spread sd, with `z = (value − m) / max(sd, 5)`.
      - `value − m ≥ minPoints` and `z ≥ minZ` is a "strength"
      - `m − value ≥ minPoints` and `z ≤ −minZ` is a "weakness"
      - `compared` is m
    - **Traits against the previous version** (spec §10: "the tag-level breakdown and traits"):
      - a trait whose change is at least `minZ` times the peers' spread for that trait (with at least `minPeers` peers) is a "changed"
      - `category` is `"personality"` and `tag` is the trait name
      - `about` is the trait's word for the direction it moved, from `TRAIT_WORDS` in `personality/fingerprint.py` (e.g. "talkative" or "terse")
      - its title is "More {about} than {previous label}"
      - its evidence is one of this version's fingerprint sample replies with its probe prompt (null if none)
    - **Keeping:** one finding per tag (on a tie, the previous-version finding wins); the `perEval` largest `|value − compared|`, where traits count in units of the peers' spread × 10.
    - **`about = tag_words(key)`:** plain words for the tag.
      - `topic:x` → x with underscores as spaces
      - `fmt:story` → "stories"
      - relation tags from a small table
      - anything else → "{category words} ({tag value})"
    - **Titles:**
      - strength "Surprisingly good at {about}"
      - weakness "Struggles with {about}"
      - better "Much better at {about} than {previous label}"
      - worse "Worse at {about} than {previous label}"
    - **Evidence:**
      - a result with that tag: for strength or better, the first with score 1 (else the highest); for weakness or worse, the first with score 0 (else the lowest)
      - `prompt` is the suite item's prompt (stories: `STORY_PROMPTS`; consistency: the group's first member); `reply` is the result's output, verbatim
      - null when no result has the tag

- [ ] **Step 1: Write the failing tests:**
  - **Signals:** from a hand-made report, the counts; `legibleReplies` is null without a judge.
  - **`find_discoveries`:**
    - +30 on a tag against the previous version is a "better"; +20 is nothing
    - a trait that jumps by 3 peer spreads is a "changed" titled with its word
    - peers below `minPeers` give no strength
    - only the top 3 are kept
    - evidence is quoted byte-for-byte
    - unknown tags get the fallback words
  - **The eval job:** it returns the new fields; an old-style spec still works.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run the ML suite, ruff and `npm test`: PASS.
- [ ] **Step 5: Commit** `feat(ml): evals report milestone signals, the tag breakdown and discoveries with real evidence`.

### Task 2: Interview prompt cards and the interview job

**Files:**
- Create: `ml/src/airace_ml/interviews.py`, `ml/tests/test_interviews.py`
- Modify:
  - protocol and fixtures: `packages/protocol/src/worker.ts` (job type, slot, spec and result), `packages/protocol/fixtures/worker/` (interview fixtures), `packages/protocol/src/catalog.ts` (`interviewCards`)
  - ML: `ml/src/airace_ml/worker/jobs.py` (`handle_interview`), `ml/src/airace_ml/cli.py` (worker slot `interview` 1), `ml/src/airace_ml/catalog.py` (export the cards)
  - server: `apps/server/src/catalog.ts` (`publicCatalog` includes the cards), `apps/server/src/jobs/broker.ts` (`interview: 1`)

**Interfaces:**
- Produces:
  - **`InterviewCard(id: str, prompt: str, checked: bool)`.** `interview_cards() -> list[InterviewCard]` returns 24 cards:
    - **12 checked cards:** 3 each from the knowledge, pattern, instruction and coding generators on `skill_rng(category, "interviews")`. Only items a chat can ask; ids `interview-<category>-<k>`.
    - **12 open cards,** ids `interview-open-<k:02d>`, with this copy:
      1. "What would you like people to know about you?"
      2. "Tell us a joke."
      3. "Describe your perfect day."
      4. "What is your favorite word, and why?"
      5. "Write a short poem about the sea."
      6. "Explain rain to a five-year-old."
      7. "What makes a good friend?"
      8. "Tell us a short story about a robot."
      9. "What do you think about cats?"
      10. "Give one tip for learning faster."
      11. "What would you do with a million dollars?"
      12. "Describe the color blue."
  - **`score_answer(card_id: str | None, reply: str, judge: Judge | None) -> tuple[float, bool]`:**
    - a checked card: `(score_reply(item, reply), True)`
    - anything else is scored for fluency, `checked = False`:
      - with the judge: `_coherence(nll) × (1 − repetitiveness(reply))` when the reply has reply words, else 0
      - without the judge: 1.0 when it has reply words, else 0.0
    - an unknown card id counts as open
  - **The job:**
    - `InterviewJobSpec = {interviewId: zId, answers: [{cardId: string | null, question: string, reply: string}]}` with 1–5 answers
    - `InterviewJobResult = {answers: [{score: 0..1, checked: boolean}]}`, in the same order
    - it loads no model
  - **The catalog:** `interviewCards: [{id, prompt, checked}]`. The zod `Catalog` gives it `.default([])`, so an older catalog still loads, and `PublicCatalog` carries it.

- [ ] **Step 1: Write the failing tests:**
  - 24 cards with unique ids; no checked prompt equals a bench prompt
  - a checked card's accepted answer scores 1 (checked); `""` scores 0; open fluency without a judge
  - the job round-trips through the fixtures
  - the export includes the cards; an older catalog parses with none
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(ml): interview prompt cards and a job that scores published answers`.

---

## Part B: the game rules (sim)

### Task 3: sim-5, the upgrade, and the notebook in the training plan

**Files:**
- Modify:
  - protocol: `packages/protocol/src/world.ts` (`MIX_IDS`, the plan's mix, `saveNotebook`), `packages/protocol/src/view.ts`
  - sim: `packages/sim/src/balance.ts` (default plans with notebook 0; `notebook.maxWords: 5000`), `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts`, `packages/sim/src/training.ts`, `packages/sim/src/project.ts`, `packages/sim/src/upgrade.ts`
  - server: `apps/server/src/config.ts`, `apps/server/src/realtime/gateway.ts` (`SERVER_ONLY` commands: players may not send `saveNotebook`)
- Test: `packages/sim/test/training.test.ts`, `project.test.ts`, `properties.test.ts`, `apps/server/test/actor.test.ts`, `apps/server/test/gateway.test.ts`

**Interfaces:**
- Produces:
  - **Command:** `saveNotebook {hash: /^[0-9a-f]{64}$/, words: int 0..5000}`, sent only by the server's notebook service (Task 11).
  - **State:**
    - `CompanyState.notebook: {hash: string; words: number; savedDay: number} | null`
    - `RunState.notebookHash: string | null`: set at `startTraining` when `plan.mix.notebook > 0`
  - **Rules:** `startTraining` refuses a notebook share above 0 while the notebook is null or has 0 words: "Write something in your notebook first."
  - **View:** `PrivateCompany.notebook: {words, savedDay} | null`, without the hash.
  - **`upgradeState` from sim-4:**
    - every stored plan's mix gains `notebook: 0`; `notebook: null`; runs get `notebookHash: null`
    - every field Tasks 4–8 add, with each task extending this bridge and its test
    - `SIM_VERSION = "sim-5"`
  - **Gateway:** a player's `saveNotebook` is rejected with "That request isn't valid."

- [ ] **Step 1: Write the failing tests:**
  - the refusal
  - `saveNotebook` stores the hash and words; a run records the hash
  - no view contains the hash (privacy property)
  - a sim-4 world (from a Phase 3 test world) upgrades and replays
  - a player's `saveNotebook` is rejected at the gateway
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): the notebook joins the training plan as a ninth share; sim-5`.

### Task 4: Milestones and the trophy shelf

**Files:**
- Create: `packages/sim/src/milestones.ts`, `packages/sim/test/milestones.test.ts`
- Modify: `packages/protocol/src/world.ts` (`MILESTONE_IDS` and `MILESTONE_NAMES`), `packages/protocol/src/view.ts`, `packages/sim/src/state.ts` (`evalCompleted` gains optional `signals`), `packages/sim/src/reducer.ts` (tick: `milestonesDay` after `leagueDay`), `packages/sim/src/ledger.ts` (month close), `packages/sim/src/battles.ts`, `packages/sim/src/project.ts`, `packages/sim/src/upgrade.ts`

**Interfaces:**
- Produces:
  - **State:** `CompanyState.milestones: {id: MilestoneId; day: number; detail: string | null}[]`.
  - **`noteMilestone(s, company, id, detail)`:** each id once per company, players only (`rival === null`).
  - **Checks:**
    - **On `evalCompleted` with `signals`:**
      - `legibleReplies ≥ 1` → firstSentence (detail: the version label)
      - `correctFacts ≥ 1` → firstFact
      - `passingPrograms ≥ 1` → firstProgram
      - no `signals` (older inputs), or null legibility → nothing
    - **Daily (`milestonesDay`):**
      - total customers ≥ 1,000 / 100,000 / 1,000,000
      - a paying customer → firstPaying
      - #1 in any skill table in a world with rivals → numberOne (detail: the skill words)
      - an open data center → dataCenter
    - **At a month close:** profit above 0 → firstProfit; an A → gradeA.
    - **On `battleResolved`:** a win against the arch-rival → beatArch.
  - **View:** `PrivateCompany.milestones`, oldest first.

- [ ] **Step 1: Write the failing tests:**
  - each milestone fires once, from its own measurement
  - none for rivals
  - signals with `legibleReplies: null` never give firstSentence
  - an upgraded world gets its state-based milestones the next day and no measurement milestones until an eval
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): milestones earned only from measurements, for a trophy shelf`.

### Task 5: Discoveries

**Files:**
- Create: `packages/sim/src/discoveries.ts`, `packages/sim/test/discoveries.test.ts`
- Modify: `packages/protocol/src/world.ts` (`publishDiscovery`), `packages/protocol/src/view.ts`, `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts`, `packages/sim/src/balance.ts`, `packages/sim/src/project.ts`, `packages/sim/test/arbitraries.ts`, `packages/sim/src/upgrade.ts`

**Interfaces:**
- Consumes: Task 1's `Discovery`.
- Produces:
  - **Balance** `discoveries`:
    - balance-v2: `{minPoints: 25, minZ: 2, minPeers: 3, perEval: 3, publishFollowers: 0.03, kept: 30}`
    - balance-test: `{minPoints: 1, minZ: 0, minPeers: 1, perEval: 3, publishFollowers: 0.03, kept: 30}`
    - the server passes these to the eval (Task 9)
  - **System event:** `{type: "discoveryFound"; versionId; discoveries: Discovery[]}`.
    - Refused: an unknown or rival version; a version that already has findings ("Already found."); more than `perEval` items or an invalid shape.
  - **State:** `CompanyState.discoveries: DiscoveryState[]` (the last `kept`), where `DiscoveryState = Discovery & {id: "<versionId>-d<k>"; versionId; day; published: boolean}`.
  - **Command `publishDiscovery {discoveryId}`:**
    - refusals: "That discovery isn't yours." / "You've already published that."
    - news (topic `discovery`; sentiment 0.4 for strength or better, 0.2 for changed, 0.1 for weakness or worse):
      - strength: "{Company}'s {label} turns out to be surprisingly good at {about}"
      - weakness: "{Company} admits its {label} struggles with {about}"
      - better: "{Company}'s {label} got much better at {about}"
      - worse: "{Company}'s {label} got worse at {about}"
      - changed: "{Company}'s {label} has become more {about}"
    - with evidence, a post by a drawn researcher that quotes the prompt and the reply verbatim (tone praise or roast by kind)
    - followers × (1 + `publishFollowers`)
  - **View:** `PrivateCompany.discoveries`, with their evidence.

- [ ] **Step 1: Write the failing tests:**
  - findings are stored and the cap holds
  - a second finding for the same version is refused
  - publishing makes one story and one verbatim post (a hostile reply stays byte-identical); publishing twice is refused
  - privacy: discoveries appear once (own) or not at all
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): discoveries about an AI, published as news with real evidence`.

### Task 6: Interviews

**Files:**
- Create: `packages/sim/src/interviews.ts`, `packages/sim/test/interviews.test.ts`
- Modify: `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts`, `packages/sim/src/balance.ts`, `packages/sim/src/feed.ts` (export the post helper), `packages/sim/src/project.ts`, `packages/protocol/src/view.ts`, `packages/sim/src/upgrade.ts`

**Interfaces:**
- Produces:
  - **Balance** `interviews`: `{cooldownDays: 14, maxAnswers: 5, followersPerPoint: 0.1, viralBelow: 0.1, viralFollowers: 1000, kept: 10}`; balance-test `cooldownDays: 1`.
  - **System event:** `{type: "interviewScored"; interviewId; productId; answers: {question: string; reply: string; score: number; checked: boolean}[]}`.
  - **Refusals:**
    - "That AI isn't one of yours." (unknown product, or a rival's)
    - "That AI isn't on sale."
    - "That interview isn't valid." (0 or more than 5 answers, a bad shape)
    - "You gave an interview recently. Next one on {date}." (within `cooldownDays` of `lastInterviewDay`)
    - "Already published." (a known `interviewId`)
  - **Reaction:**
    - **Score:** `r = clamp(mean score + traitBonus, 0, 1)`. `traitBonus` is +0.05 for each of the version's personality words containing "warm", "bold" or "inventive", and −0.05 for each containing "slip" or "timid".
    - **Sentiment:** `clamp(2r − 1, −0.8, 0.8)`.
    - **Followers:** × (1 + `followersPerPoint` × (2r − 1)).
    - **Outlet:** drawn from the world rng: "The Daily Byte" (@dailybyte), "Circuit Weekly" (@circuitweekly), "TechTalk Radio" (@techtalkradio), "The Model Times" (@modeltimes).
    - **News** (topic `interview`):
      - r ≥ 0.7: "{Product} wows in its interview with {outlet}"
      - r ≥ 0.4: "{Product} gives a so-so interview to {outlet}"
      - otherwise: "{Product} stumbles through an interview with {outlet}"
    - **Posts** by the outlet:
      - the best answer (praise)
      - when r < 0.4 or an answer scores ≤ `viralBelow`, also the worst (roast)
      - a worst answer ≤ `viralBelow` with followers ≥ `viralFollowers` also makes the existing viral story
  - **State:**
    - `CompanyState.interviews: {id; day; productId; productName; outlet; mean: number; answers}[]` (the last `kept`)
    - `CompanyState.lastInterviewDay: number | null`
  - **View:** `PrivateCompany.interviews` (answers verbatim) and `nextInterviewDay`.

- [ ] **Step 1: Write the failing tests:**
  - each refusal
  - a strong interview gains followers and a "wows" story with a praise post quoting the answer exactly
  - a weak one with followers ≥ 1,000 goes viral
  - the trait bonus moves r
  - the cooldown date
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): published interviews move followers and make news from the AI's real answers`.

### Task 7: The goal and the victory

**Files:**
- Create: `packages/sim/src/goal.ts`, `packages/sim/test/goal.test.ts`
- Modify: `packages/protocol/src/world.ts` (`Goal`, `GoalSpec`), `packages/protocol/src/view.ts`, `packages/sim/src/world.ts` (`createWorld` takes `goal`), `packages/sim/src/reducer.ts` (`startWorld` sets the deadline; tick: `goalDay` after `milestonesDay`), `packages/sim/src/state.ts`, `packages/sim/src/balance.ts` (`goal.valueMonths: 24`), `packages/sim/src/project.ts`, `packages/sim/src/upgrade.ts` (goal none)

**Interfaces:**
- Produces:
  - **`Goal`** (zod, discriminated on `kind`):
    - `{kind: "none"}`
    - `{kind: "customers", target: int 1_000..1_000_000_000}`
    - `{kind: "profit", target: int 1_000..1_000_000_000}`
    - `{kind: "bestAi"}`, `{kind: "beatArch"}`
    - `{kind: "value", target: int 100_000..1_000_000_000_000}`
    - `GoalSpec = {goal: Goal, deadlineYears: 1 | 2 | 5 | null}`
  - **State:** `WorldState.goal: {spec: GoalSpec; deadlineDay: number | null; reachedDay: number | null; reachedBy: string | null; missedDay: number | null}`. `deadlineDay = start day + 365 × years` is set by `startWorld`.
  - **`companyValue(s, company)`:** `round(ownership × max(0, cash − loanBalance + valueMonths × max(0, mean profit of the last 3 closed months)))`.
  - **`goalProgress(s, company): {value: number; target: number | null; done: boolean}`:**
    - customers: the total
    - profit: the last closed month's profit
    - bestAi: rank in the overall table; done at rank 1 with rivals present and a live product
    - beatArch: your customers against the arch-rival's; done when above
    - value: `companyValue`
  - **`goalDay(s)`:**
    - The first player company to be done sets `reachedDay` and `reachedBy`, and tells "{Company} reaches its goal: {goal words}" (topic `goal`, sentiment 0.8).
    - When the deadline passes first: `missedDay` and "{Company} misses its goal deadline" (sentiment −0.2), told once.
    - Play always continues, and a goal is reached at most once.
  - **View:**
    - `world.goal: {kind, target: number | null, deadlineDay, reachedDay, missedDay, reachedBy} | null` (null for no goal)
    - `me.company.goalProgress: {value, target, done} | null`

- [ ] **Step 1: Write the failing tests:**
  - each kind's progress
  - reached exactly once, with one story
  - reaching it on the deadline day counts as reached
  - a missed deadline is told once
  - company value with investors (the player's share) and with negative cash (0)
  - a rival goal in a world without rivals never completes
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): an optional goal with progress, a deadline and a once-only victory`.

### Task 8: The data-quality scandal

**Files:**
- Modify: `packages/sim/src/state.ts`, `packages/sim/src/training.ts` (`applyCompletion` sets `unchecked`), `packages/sim/src/feed.ts` (`applyFeedback`), `packages/sim/src/balance.ts`, `packages/sim/src/upgrade.ts`
- Test: `packages/sim/test/feed.test.ts`

**Interfaces:**
- Produces:
  - `VersionState.unchecked: boolean` = the parent's `unchecked`, or (`plan.mix.web > 0` and not `plan.factCheck`). Rival versions are false.
  - `ProductState.scandal: boolean`.
  - **Balance** `scandal`: `{minCustomers: 100_000, satisfactionHit: 0.05, sentiment: -0.7}`; balance-test `minCustomers: 100`.
  - **In `applyFeedback`,** for a player's product when all of these hold:
    - its version is `unchecked`, and the product has no scandal yet
    - the company has at least `minCustomers` customers
    - a result with a `questionId` starting `"knowledge-"` scores 0

    Then:
    - the product's `scandal` becomes true
    - news (topic `scandal`): "Customers catch {Product} stating false facts: {Company} trained it on unchecked web data"
    - every group's satisfaction for that product falls by `satisfactionHit` (not below 0)

- [ ] **Step 1: Write the failing tests:**
  - the scandal fires once
  - it doesn't fire with fact-checking, with a fact-checked lineage from the start, below the customer mark, or for rivals
  - satisfaction falls
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): a data-quality scandal when an AI trained on unchecked web data is caught at scale`.

---

## Part C: the server

### Task 9: Evals carry signals and discoveries into the world

**Files:**
- Modify: `apps/server/src/training/service.ts`, `apps/server/src/rivals/service.ts` (`breakdownOf(packKey)` and `traitsOf(packKey)` from the pack), `apps/server/test/fakeWorker.ts` (eval results with signals, breakdown and discoveries)
- Test: `apps/server/test/training.test.ts`

**Interfaces:**
- Consumes: Tasks 1, 4 and 5.
- Produces:
  - **The eval spec:**
    - `previous`: the parent version's stored breakdown and traits, with its label
    - `peers`: every other evaluated version in the world, plus each released rival version from the pack
    - `thresholds`: the world balance's `discoveries`
  - **Storage:** `eval_results.report` keeps the breakdown and the signals.
  - **Results:** `evalCompleted` carries `signals`. A non-empty finding list then submits `discoveryFound`. On a restart, the stored report applies again and a repeated finding list is refused by the sim.

- [ ] **Step 1: Write the failing tests:**
  - the spec carries the parent's breakdown and the peers
  - signals reach the sim and give a milestone
  - findings arrive once, including after a restart
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): evals compare each AI with its parent and the world's AIs, and report discoveries`.

### Task 10: The interview service

**Files:**
- Create: `apps/server/src/interviews/service.ts`, a migration (`chat_threads.kind text not null default 'chat'`, `chat_threads.product_id uuid`, `chat_messages.card_id text`, table `interview_requests (id uuid primary key, world_id uuid, company_id uuid, product_id uuid, thread_id uuid, idxs jsonb, job_id uuid, created_at)`), `apps/server/test/interviews.test.ts`
- Modify: `packages/protocol/src/wire.ts` (the `chat` message's optional `interview`; the new `interviewPublish` message), `apps/server/src/chat/service.ts` (interview threads; priority 1), `apps/server/src/db/schema.ts`, `apps/server/src/app.ts`, `apps/server/src/realtime/gateway.ts`, `apps/server/test/fakeWorker.ts`

**Interfaces:**
- Consumes: Tasks 2 and 6.
- Produces:
  - **Wire:**
    - `chat` gains optional `interview: {productId: zId, cardId: string | null}`. A new thread with it becomes an interview thread for that product's AI (your own and live only). Each user message stores its `cardId`. Its generate jobs run at priority 1.
    - `{v, type: "interviewPublish", cid: uuid, worldId: zId, threadId: zId, idxs: int[] (1–5)}`.
  - **`InterviewService.publish(userId, msg)`:**
    - **Checks:** your own interview thread; every index points at an AI message; the product is still live; the cooldown, from the state.
    - **Refusals** use the sim's words, plus "Pick answers from this interview."
    - **Then:**
      - inserts a request row and acks `{interviewId}`
      - enqueues `interview` (priority 1); the spec is built at lease from the stored messages, with each question taken from the message before its reply and the `cardId` kept
    - **When scored:** submits `interviewScored`.
    - **On a final failure:** sends the user an event "The interview couldn't be scored. Try publishing again." and leaves the state unchanged.

- [ ] **Step 1: Write the failing tests:**
  - an interview thread generates at priority 1
  - publishing 2 answers produces `interviewScored` with the exact stored text
  - an index of a user message is refused; another user's thread is refused; the cooldown is refused
  - a failed job changes nothing and tells the user
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): interviews held on the server and published as scored answers`.

### Task 11: The notebook service and goals at world creation

**Files:**
- Create: `apps/server/src/notebook/service.ts`, `apps/server/src/notebook/filter.ts`, `apps/server/src/notebook/wordlist.ts`, a migration (`notebooks (company_id uuid primary key, world_id uuid, hash text)`, `notebook_texts (hash text primary key, text text)`), `apps/server/test/notebook.test.ts`
- Modify: `packages/protocol/src/wire.ts` (`notebookSave`; the `notebook` query), `apps/server/src/training/config.ts` and `service.ts` (the notebook in the run's config), `apps/server/src/worlds/routes.ts` (`POST /api/worlds` takes `goal`), `apps/client/src/net/api.ts`
- Test: also `apps/server/test/training.test.ts` and `worlds.test.ts`

**Interfaces:**
- Consumes: Tasks 3 and 7.
- Produces:
  - **Saving:** `{v, type: "notebookSave", cid, worldId, text: string (at most 40,000 characters)}`.
    - **`notebookProblem(text): string | null`:**
      - more than 5,000 words: "Notebooks hold up to 5,000 words."
      - links, email addresses or phone numbers: "Notebooks can't hold links, email addresses or phone numbers."
      - `<|`: "That text has characters the AI uses for itself."
      - a listed word: "That notebook has words that aren't allowed."
      - empty text is allowed (it clears the notebook)
    - **Storing:** the text goes into `notebook_texts` by sha256; the company's row is upserted; the server submits `saveNotebook {hash, words}` as that user.
  - **Query** `notebook {worldId}`: the owner's text (`""` when none).
  - **Training:** a run with a `notebookHash` gets `notebook` = that text split into paragraphs, and mixture `notebook` = the plan's share. Logs never include the text.
  - **World creation:** `POST /api/worlds {name, goal?: GoalSpec}`.
    - A rival goal without a pack: "Rival goals need the rival AIs, which aren't built yet."
    - `api.createWorld(name, goal?)`.

- [ ] **Step 1: Write the failing tests:**
  - each filter refusal; a save stores the text by hash and the sim gets the hash and words
  - a run trains with exactly the text saved when it started, even if the notebook changed after
  - the query is owner-only
  - captured logs never contain the text
  - a world created with a goal shows it; a rival goal without a pack is refused
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): a private, filtered notebook for training, and goals chosen at world creation`.

---

## Part D: the screens (client)

Every new window plugs into the window stack (`WindowSpec` gains `discovery`, `interview`, `notebook`, `milestones` and `goal`) and follows the immersion design.

### Task 12: Interview mode in the Playground

**Files:**
- Create: `apps/client/src/screens/Interview.tsx`, `apps/client/src/windows/InterviewWindow.tsx`, `apps/client/test/interview.test.tsx`
- Modify: `apps/client/src/screens/Playground.tsx` (a "Chat / Interview" switch), `apps/client/src/net/socket.ts` (`interviewPublish`), the window store and host

**Interfaces:**
- Consumes: Tasks 2, 6 and 10.
- Produces:
  - **Interview mode:**
    - Pick one of your AIs on sale.
    - Prompt cards appear as chips, from the catalog's `interviewCards`. Checked cards carry "has a right answer". You can also type your own question.
    - Answers stream like chat. Each answer has a "Publish this" tick (up to 5).
    - "Publish interview" sends `interviewPublish`; a refusal shows inline.
    - The cooldown shows as "Your next interview: {date}".
  - **Afterwards:** when `me.company.interviews` gains the interview, a card shows the outlet, the headline from the feed, the followers change, and each published answer with its score bar. It opens `InterviewWindow`.

- [ ] **Step 1: Write the failing tests:**
  - a card sends `chat` with `interview.cardId`
  - ticking 2 answers and publishing sends the exact message
  - a refusal shows; the result card appears from the view
  - answers render as text (hostile markup)
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): interview your AI with prompt cards and publish the answers you choose`.

### Task 13: Discoveries, the notebook and the trophy shelf

**Files:**
- Create: `apps/client/src/windows/DiscoveryWindow.tsx`, `apps/client/src/windows/NotebookWindow.tsx`, `apps/client/src/components/TrophyShelf.tsx`, `apps/client/src/windows/MilestonesWindow.tsx`, `apps/client/test/surprises.test.tsx`
- Modify: `apps/client/src/windows/AiWindow.tsx` and `apps/client/src/screens/lab/VersionsStep.tsx` (discoveries), `apps/client/src/screens/lab/DataStep.tsx` (a ninth card, Notebook), `apps/client/src/store/plan.ts` (the notebook share), `apps/client/src/screens/Company.tsx` (the shelf), `apps/client/src/components/Toasts.tsx`, the window store and host

**Interfaces:**
- Consumes: Tasks 3, 4, 5 and 11.
- Produces:
  - **Discoveries:**
    - The AI window lists its findings with a kind badge (Strength, Weakness, Better, Worse, Changed). A finding opens `DiscoveryWindow`: the title, the value against the comparison as two bars, the evidence prompt and reply verbatim, and "Publish as news" (`publishDiscovery`). Published findings show "Published".
    - A new finding arrives as the toast "Your AI has a surprise: {title}".
  - **Notebook card** in the Data step:
    - It shows the word count (n / 5,000), a share slider, and "Write", which opens `NotebookWindow`.
    - The slider is disabled until the notebook has words.
    - A share above 20% shows: "Your notebook is small: a big share makes the AI repeat it word for word."
  - **`NotebookWindow`:**
    - a text area with a live word count
    - Save sends `notebookSave`; a refusal shows; on success it says "Saved. Only your company's AI reads this."
    - it loads the text with the `notebook` query
  - **Trophy shelf:**
    - A row on the Company screen with the earned milestones only: an icon, the name and the date. It opens `MilestonesWindow` (all earned, with the day and the detail).
    - A new milestone bursts like the existing celebrations.

- [ ] **Step 1: Write the failing tests:**
  - a finding opens with verbatim evidence; Publish sends the command
  - the notebook saves, refusals show, and the slider is locked at 0 words
  - the shelf shows only earned milestones, and the burst plays once per new one
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): discoveries with real evidence, the notebook, and a trophy shelf`.

### Task 14: Goals: the picker, the progress and the victory screen

**Files:**
- Create: `apps/client/src/components/GoalPicker.tsx`, `apps/client/src/components/GoalProgress.tsx`, `apps/client/src/screens/Victory.tsx`, `apps/client/test/goal.test.tsx`
- Modify: `apps/client/src/screens/Home.tsx` (the picker in "Start a new world"), `apps/client/src/screens/Company.tsx` (progress along the bottom), `apps/client/src/screens/World.tsx` (shows the victory screen), `apps/client/src/net/api.ts` (`setup()` hides rival goals)

**Interfaces:**
- Consumes: Tasks 7 and 11, and Phase 3's `/api/setup`.
- Produces:
  - **`GoalPicker`:** cards, each with an editable target where it has one, plus a deadline choice: "No deadline / 1 year / 2 years / 5 years". The rival goals are hidden when `setup().rivals` is false.
    - "No goal: play as long as you like"
    - "Reach {1,000,000} customers"
    - "Make {$1,000,000} profit in a month"
    - "Build the best AI overall"
    - "Beat the arch-rival on customers"
    - "Grow the company's value to {$100,000,000}"
  - **`GoalProgress`:** a bar with "value / target", a deadline countdown, and "Rivals haven't joined yet" where that applies. It opens a goal window with the progress over time from the history.
  - **The victory screen,** shown once per world per browser (local storage, wrapped in try/catch), when `world.goal.reachedDay` is set and `reachedBy` is you:
    - the company's story, drawn in: customers and cash day charts, profit month bars, the milestones with their days, the best report-card moments (verbatim), battles won, and the biggest headline
    - "Keep playing"
  - **Missed deadline:** a calm card on the Company screen.

- [ ] **Step 1: Write the failing tests:**
  - the picker sends the exact goal, and rival goals are hidden without rivals
  - progress shows value and target
  - the victory screen shows once with the story from the view, and "Keep playing" closes it
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): pick a goal, follow it, and celebrate it with the company's story`.

---

## Part E: end to end

### Task 15: End-to-end, docs and the final review

**Files:**
- Modify: `tests/e2e/solo.spec.ts`, `CLAUDE.md`, `docs/progress.md`, `docs/handoff/README.md`, `docs/handoff/sdd/ledger.md`

- [ ] **Step 1: Extend the E2E:**
  1. create the world with the goal "Reach 1,000 customers" (the test balance gets there quickly)
  2. write and save a notebook line; the Data step shows its word count
  3. after the first version is scored, its AI window lists at least one discovery (the test balance's thresholds); publish it, and its story appears in the feed
  4. interview: pick a prompt card, wait for the answer, tick it, publish; the interview story appears
  5. the trophy shelf shows "First paying customer"
  6. the victory screen appears; "Keep playing" returns to the Company screen
- [ ] **Step 2:** Run `npm run e2e`: FAIL, then fix real bugs, each with a unit test in the task that owns the code.
- [ ] **Step 3:** Run `npm test`, `npm run check`, the ML suite with ruff, and `npm run e2e` (twice).
- [ ] **Step 4: Play it on the GPU:** an interview, a notebook-trained version (watch it parrot at a big share), a discovery, a milestone, a goal.
- [ ] **Step 5: Final review** of the whole branch range; one fix pass for Critical and Important findings, each with a failing test first.
- [ ] **Step 6: Docs and report:**
  - progress entry; the handoff resume point is M2 Stage 2 (friends, plan `docs/superpowers/plans/2026-10-22-m2-two-friends.md` Tasks 25–28) on the owner's go; the ledger
  - commit and push, then **report to the owner** with the rulings and deferred minors

## Decisions made in this plan (for the owner report)

1. **The goal is chosen in the "start a new world" request**, not by a `setGoal` command; it is part of the world from its first moment. If wrong: a command to add later.
2. **Interviews are published through a server service that holds the AI's answers** (rather than a `publishInterview` command carrying text). A player can't put words in the AI's mouth (R1). The world receives `interviewScored`. If wrong: none for play.
3. **Discoveries compare a version with the one before and with the world's other AIs** (the game-layer spec), not with the data-mix regression of architecture spec §4.9. If wrong: fewer, but still real, surprises.
4. **"First correct fact" counts only free-answer questions; "first working program" counts written functions that pass hidden tests.** Multiple choice can be right by luck (R5). If wrong: these milestones come a little later.
5. **The trophy shelf shows earned milestones only:** a list of what is still to earn would be an unlock list (R5). If wrong: a list to add.
6. **A notebook is stored by its fingerprint (hash), so a training run uses exactly the text saved when it started.** If wrong: a little more storage.
7. **The discovery thresholds live in the balance;** the test balance makes the first eval show one, so the E2E can check it. If wrong: one number to tune.
8. **"Plus traits" in interview scoring is a small bonus from the AI's measured personality words** (warm, bold, inventive up; slips, timid down). If wrong: one rule to change.
9. **The data-quality scandal** (deferred from Phase 2) lands here, with the other consequence events.
10. **Worlds from Phase 3 carry over** (sim-4 → sim-5). Measurement milestones are not backfilled; they come with the next scoring.
