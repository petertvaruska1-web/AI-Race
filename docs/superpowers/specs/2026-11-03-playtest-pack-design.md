# AI Race: Playtest Pack Design (fair battles, Playground games, things to buy)

**Date:** 2026-11-03. **Status:** approved by the owner through the interview below; it comes before Phase 4.
**Builds on:** `2026-10-27-game-layer-design.md` (the game layer) and the architecture spec. Every R1–R8 rule applies.

## 0. Why

These are the owner's playtest requests after Phase 3 (2026-10-30):

1. Battles call a draw when there is a clear winner.
2. The Playground feels weaker than the customers' posts, and the owner wants the posts paused while they chat.
3. Playground minigames, which are the top priority. Each must really use the player's AI, be visual where it fits, and not be too hard for a home-made AI.
4. Money piles up with nothing worth buying.

## 1. Interview decisions (2026-11-03)

- **C31 Minigames:**
  - The games are Puzzle Maze (reasoning), Pattern Bridge (patterns), Quiz Show (knowledge), Story Dice (creativity plus a rule to follow), Simon Says (instructions) and Duel mode (any game against a rival's AI).
  - The Code Machine and the Obstacle Course were not chosen.
- **C32 Results:**
  - Practice is free, private and keeps personal bests.
  - A showcase is a paid live game for the press: a great run wins fans and good news, a bad one gets roasted.
- **C33 Difficulty:** the player picks Easy, Medium or Hard, and the puzzles also get harder the further the AI gets in a run.
- **C34 Spending:**
  - The owner chose Launch events (a keynote with a live demo), Sponsorships, Looks and Partner apps.
  - They also chose three ways to improve the AI directly: Learn from customers, Tutoring packs and Fact-check plus.
  - For finances they chose Investments.
  - Rejected: headquarters, buying out a rival, faster chips, customer programs, crisis PR, AI-written ads, Stump the AI, tournaments, focus groups, gadgets, awards, a schools program, Think harder, efficiency research, business deals and prepaid compute.
- **C35 Bills:** a bigger company pays bigger bills, so money stays meaningful; a well-run company still grows rich.

## 2. Fair battles

**Diagnosis.** Battles asked every challenge as a chat at temperature 0.8 with up to 96 tokens. The benchmark asks the same kinds of item in the format the AI was trained on (`item.chat`), greedily, with the item's short reply length and its extract rule. So the AIs rambled. Then each check scored pass/fail, so two wrong answers tied 0–0.

**Fix (ML, `battles.py` + `handle_battle`):**
- **Asking.** Checked skills ask each item the way the benchmark does: its own chat flag, greedy decoding, the item's `max_new_tokens` and its extract rule for scoring. Stories stay as they are (temperature 0.9, judge-scored).
- **Graded score per round.** A right answer scores 1. Otherwise the score is `0.5 × closeness`. Closeness is the best of:
  - **Numbers:** if both the reply's answer and the right one are numbers, `max(0, 1 − |a − b| / max(1, |b|))`.
  - **Words:** the token-overlap F1 between the extracted reply and the right answer (or the checker's reference reply), after lower-casing and stripping punctuation.
  - **Checks with parts:** the share of a check's parts that pass, where the checker has parts.
- **The right answer travels with each round.** It is `expected` (the answer, the reference reply, or null for stories), and the Battle window shows it under the two replies.
- **Draws.** The draw margin drops from 0.25 to 0.1, so a draw needs the totals to be almost equal.

This scoring never rewrites a reply (R1). The partial credit is a measurement against the right answer (R5).

## 3. The Playground versus the posts, and the pause

**Diagnosis** (told to the owner in plain words):
- Sharing the GPU makes answers slower, not worse.
- The posts are better for three reasons:
  1. Each feedback batch posts only its **best and worst** replies.
  2. Customers ask the kinds of question the AI was trained on.
  3. A Playground chat sends up to 12 earlier turns as context, which confuses a tiny AI. Fix: the Playground's "New chat" button sits next to the box, and a hint explains that a fresh chat answers best.

**The pause:**
- While a player has the Playground open (any mode), the client sends `playgroundPresence {worldId}` every 15 s.
- While that world has a presence less than 30 s old, the broker leases no `feedback` or `appReview` jobs for it. The jobs stay queued and run once the player leaves.
- The Playground shows a small note: "Customer posts are paused while you're here, so your AI answers you faster."

## 4. Playground minigames

### 4.1 Rules
- **Every decision is the player's AI's own (R1):**
  - A choice is the option the model finds most likely: the mean log-probability of each option's tokens after the prompt, as the benchmark scores MC items.
  - Confidence is the softmax of those means, shown as bars.
  - On Hard, the AI types its answer: greedy generation, checked like the benchmark.
- **The content** comes from the skill generators on their `battles` random stream, from held-out keys only, never a scored benchmark prompt. It is the same pool rule as battles.
- **Difficulty:**
  - Easy offers 2 options, Medium 3 and Hard 4. The Quiz Show always shows 4, with 2 struck out on Easy and 1 on Medium.
  - On Hard, the Puzzle Maze, Pattern Bridge and Quiz Show also have the AI type its answer from step 7 on.
  - Within a run, item `k` comes from the difficulty band `band(k)` (each game's families are ordered easy to hard), so a weak AI can always clear the first steps.
- **A run:**
  - 3 lives and up to 10 steps (Pattern Bridge 12, Quiz Show 12 rungs).
  - A wrong answer costs a life; the run ends when the lives or the steps run out.
  - Score = steps cleared; Story Dice instead scores audience points, below.
  - A run is deterministic for its seed and model.
- **Duel:**
  - Both AIs face the same item list, side by side.
  - Each keeps its own lives; the one that clears more steps wins, and equal steps is a tie.
  - The rival's AI is loaded as a transient model, like rival chat.

### 4.2 The games (what the player sees)

| Game | Skill | Item | Visual |
|---|---|---|---|
| **Puzzle Maze** | reasoning | reasoning MC items (`compare`, `syllogism`, `count`, `word_problem`; bands: syllogism (yes or no), compare, count, word problem) | A seeded grid maze in SVG. At each junction one door per option is labelled with the option; the orb walks to the AI's door. Right goes on deeper; wrong shows a dead end, the orb walks back, and a life is lost. |
| **Pattern Bridge** | pattern | pattern "what comes next" with generated options (the right term plus distractors from the same family: the cycle's other members, off-by-one steps, wrong ratios); bands: cycle, step, countdown, letters, double | A canyon. Each term shows as a picture: colors as colored discs, animals as emoji, numbers and letters as tiles. The AI's pick drops in as the next tile; a right tile holds and the orb hops on, a wrong one crumbles. |
| **Quiz Show** | knowledge | knowledge MC items, by KB topic (the player picks a topic or "Mixed") | A game-show stage: four lit answers, confidence bars, a 12-rung prize ladder. A lock-in animation, then the right answer glows. |
| **Story Dice** | creativity + instruction | prompt `Write a short story about {hero} {situation}.`, plus an optional rule `Use the word "{word}".` | Three dice (hero, situation, word) with emoji faces that the player rolls or sets. The story types out; an audience meter fills from the judge's story score; a rule chip lights when the word is used. A run is 3 stories, scoring `Σ(10 × story score + 3 × rule met)`, max 39. |
| **Simon Says** | instruction | instruction CheckItems; bands: one word, yes or no, list n, starts with, all caps | Command cards flip up; the AI's reply types under each; the card turns green or red with the check's reason. |
| **Duel** | any of the above but Story Dice | the same items for both | Two lanes side by side, one in each company's colors. |

The player sets, per game: difficulty; the Quiz Show topic; Story Dice dice (roll or pick, and the rule on or off); Duel's opponent (a rival product on sale) and its game.

### 4.3 Showcases (sim)
- `startShowcase {game, difficulty, versionId}` charges the fee ($20,000 in balance-v2).
  - Only one showcase runs at a time, with a 7-day cooldown (2 on the test balance).
  - The version must be live in a product.
- The result comes back as `showcaseResolved {showcaseId, stepsCleared, maxSteps}`. Its performance is `p = stepsCleared / maxSteps`, or `points / 39` for Story Dice.
  - **News:** "<Company>'s <Product> reaches step 7 of the Puzzle Maze live on stage", or "... stumbles live on stage" when p < 0.3. Sentiment is `2p − 1`.
  - **Followers:** × (1 + 0.08 · (2p − 1)).
  - **Reputation** is noted through `noteNews`.
- If the job fails, the fee is refunded.

### 4.4 Server
- `MinigameService` handles practice and duel runs.
  - It stores them in `minigame_runs (id, world_id, user_id, company_id, game, difficulty, settings json, status, steps json, score, max_score, created_at)`.
  - Personal bests are kept per company, game and difficulty.
  - It streams each step as a `minigameStep` event on the run's stream.
  - Priority 0, like chat. One run per player at a time, rate-limited like chat.
- A showcase is a minigame job whose result becomes the logged `showcaseResolved` input, as a battle does.

### 4.5 Worker
A `minigame` job takes this spec:

```
{ game, difficulty, seed, settings, sides: [{side, versionId, modelTarUrl}] }
```

It streams `{kind: "minigameStep", side, step, ...display}` and returns `{sides: {a: {steps:[...], score, maxScore}, b?: ...}}`.

## 5. Things to buy

| Feature | Where | What it does | Real? |
|---|---|---|---|
| **Launch keynote** | Launch dialog toggle, $30,000 | The player picks one of 3 demo questions (customer-pool questions for the product's plans). At launch a `keynote` job asks the product's AI and checks the reply. Score s: the launch's awareness lift × (0.5 + s), and news quotes the real reply ("Live on stage, Spark said: '…'"). s < 0.3 makes a flop story. | Real reply (R1) |
| **Sponsorships** | Market screen, next to ads | Kinds: Stadium (everyday, students; $150k, 90 days), Tech conference (developers, businesses, researchers; $80k, 30 days), Arts festival (creators, everyday; $60k, 60 days). Each day in the region, the target groups' awareness rises (`a += (1 − a) × rate`), news on the first day, a marker on the map. | Sim |
| **Looks** | Company screen, "Customize" | Orb styles: Classic (free), Nebula $50k, Crystal $120k, Plasma $250k, Aurora $500k, Supernova $1M. Logo motions: Still (free), Pulse $25k, Shimmer $75k, Orbit $200k. Bought once, switch any time; the logo motion shows wherever the logo does. | Show only |
| **Partner apps** | New "Apps" section on the Company screen | Kinds: Homework Helper (knowledge, students), Story Time (creativity, everyday), Code Buddy (coding, developers), Puzzle Pal (reasoning, students), Office Assistant (instruction, businesses). Setup costs $120k–$250k plus monthly upkeep, and each app runs on one of the player's products. App users grow toward `capacity × fit`, where `fit = (score/100)^1.5` from the version's **measured** score in the app's skill, times the app's rating. They pay a monthly revenue share. Weekly `appReview` jobs ask 2 skill questions and post real-reply reviews with stars = round(1 + 4 × score). | Measured score, real replies |
| **Learn from customers** | AI Lab, training plan | Server keeps each company's **loved conversations** (feedback replies scored ≥ 0.75, newest 400; private, R2). The plan option "Learn from customers" ($40 per conversation, at least 10) sends them as the run's `coaching` pairs at a 5% share. | Real training |
| **Tutoring packs** | AI Lab, training plan | Choose a dataset and 1–5 packs. Each pack adds 10% of the run's tokens, all from that dataset, on top of the length. Cost: $15,000 per pack plus the compute of the extra tokens; the run takes longer too. | Real training |
| **Fact-check plus** | AI Lab, data step | A cleaning level "Expert": quality ≥ 0.80, with duplicate removal and fact-check forced on. Cost: $4 per thousand tokens checked. | Real filter |
| **Investments** | Money screen | Deposit or withdraw. Month end: return = N(0.8%, 3%), with a 1-in-12 dip of −8% to −15% (seeded rng). Gains and losses are booked in the ledger; the fund balance counts toward company value. | Sim |
| **Showcases** | Playground games | §4.3 | Real AI |
| **Growing bills** (C35) | Money screen, "Office and support" | Monthly overhead = `0.0407 × R^1.1` for monthly revenue R, booked daily from the day's revenue: 12% of a $60k month, 15% of $465k, 17% of $1.7M, 21% of $14M. (First drafted on customers; free users made the budget strategy lose money, so it follows revenue.) | Sim |

## 6. Architecture changes

- **Sim:** `SIM_VERSION` becomes `sim-5`, with `upgradeState` from sim-4: new company fields get defaults. Phase 4's plan moves to sim-6.
  - New modules: `showcases.ts`, `sponsorships.ts`, `looks.ts`, `apps.ts`, `investments.ts` and `overhead.ts`.
  - The training plan gains `tutoring`, `learnFromCustomers` and cleaning "expert".
  - The launch gains `keynote`.
  - Ledger categories: `sponsorships`, `looks`, `apps` (out), `appRevenue` (in), `investmentGains` (in), `investmentLosses` (out), `overhead` (out) and `showcases` (out).
- **Protocol:**
  - New commands and inputs.
  - New job types `minigame`, `keynote` and `appReview`.
  - View fields: `me.company.{looks, apps, sponsorships, fund, showcases}`; `PublicCompany.logoMotion`, sponsorship markers and apps' public stats.
- **Server:**
  - New services: `MinigameService`, `KeynoteService`, `AppReviewService` and `LovedStore`, the last as a new `loved_replies` table filled from feedback results.
  - The broker gets the presence pause.
  - The training service adds coaching, tutoring and expert prep to the job config.
- **ML:**
  - `minigames/` holds the content, options and the run loop.
  - New job handlers: `minigame`, `keynote` (one question, check-scored) and `appReview`.
  - The `expert` cleaning level.
  - The catalog adds expert pools: rerun `export-catalog` (`npm run dev` writes it only when it is missing, so the catalog version moves on, which forces the rewrite).
- **Client:**
  - Playground modes: Chat, Games.
  - The game stages are SVG plus `motion`.
  - Also: the Customize window, the Apps section, the sponsorship window, the Money screen's fund, and the AI Lab extras.

## 7. Testing
- TDD everywhere.
- **ML:** content (held-out and free of benchmark prompts), option generation, the band order, the run loop with a fake model, and a slow test with a tiny real model.
- **Sim:** each feature's reducer tests plus the balance harness with the new bills, where the targets in §14 of the game spec still hold.
- **Server:** service tests with the fake worker, and the presence pause.
- **Client:** component tests per game stage, fed recorded step events.
- **E2E:** play a practice Pattern Bridge, buy a look, deposit into the fund, start a sponsorship, and train with a tutoring pack.

## 8. Temporary assumptions
All the prices and rates above are balance values the harness may tune. The KB topics offered by the Quiz Show are the KB's relation families. Story Dice word lists are fixed lists of simple nouns.
