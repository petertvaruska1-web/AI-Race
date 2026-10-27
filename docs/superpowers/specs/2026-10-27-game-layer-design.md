# AI Race — Single-Player Game Layer Design ("Open for Business")

- **Date:** 2026-10-27
- **Status:** Approved by the owner section by section in the design interview (2026-10-27); this document awaits their review.
- **Relation to other documents:**
  - Extends the architecture spec (`2026-10-06-ai-race-architecture-design.md`). Its non-negotiables R1–R8 apply unchanged.
  - Pulls Milestone 3 ("Open for business") forward and adds parts of M4 (battles, interviews, discoveries, notebook) and M5 (look and feel), all **single player** (C6).
  - The M2 plan's Stage 2 (Tasks 25–28, friends) moves after this work.
- **Delivery:** four playable phases (§15). Each phase gets its own implementation plan.

---

## 1. The owner's vision (interview decisions)

| ID | Area | Decision |
|----|------|----------|
| C7 | Feel | A **playful tycoon** (Game Dev Tycoon-like) with a **clean, smart, futuristic** look; every screen takes the company's colors (C5). |
| C8 | Main screen | The player's **AI is a glowing orb** at the centre of the Company screen: customers and money on the left, news and social posts on the right, Train / Chat / Launch actions under the orb. |
| C9 | What grows | Customers on a map, money and finances, the AI's abilities, the company's fame. All four get visible space. |
| C10 | Orb | The orb's look is computed from real measurements only (scores, traits, words read). No preset levels (R5). It starts as a faint spark. |
| C11 | Training | Keep today's live charts and graphs. Add the orb evolving during training. (Mid-training choices, milestone pop-ups during training and "then vs now" were not chosen for this layer.) |
| C12 | Team | **No staff** for now; may come later. |
| C13 | Events | **Rare and meaningful**, grounded in what the player and their AI actually did. |
| C14 | Customers | Groups (Students, Creators, Developers, Businesses, Everyday users, Researchers) with a few **spotlight** people. Acquisition by **word of mouth plus optional ad campaigns**. |
| C15 | Feedback | **Social posts** quoting real conversations, and **headline moments**. |
| C16 | Money | **Simple, but not too simple:** where money comes from, where it goes, a monthly report card. |
| C17 | Pricing | **Several plans** (Free, Personal, Business). |
| C18 | Model lineup | Up to **three AIs on sale at once** under names the player chooses (like a small/fast and a big/smart model). Each plan is powered by one of them. |
| C19 | Bankruptcy | **Rescue with a cost:** an emergency loan with heavy interest, or an investor who takes a share of the company. |
| C20 | Launch | Each release is a **launch day**: countdown, hype, first posts, news. |
| C21 | Fame | **Reputation meter, followers, rankings against rivals.** |
| C22 | Rivals | **An arch-rival (a surprise type each game) plus three rivals.** Their AIs are real. **The player's AI quality has priority** over rivals on the player's hardware. |
| C23 | Computers | **Rent first, build data centers later.** |
| C24 | Map | **About 9 US regions.** |
| C25 | Goals | **Milestones plus an optional goal.** |
| C26 | Pace | About **30 minutes of play to the first paying customers.** |
| C27 | Extras | **Battles vs rivals, interviews, discoveries, the player's own writing** (notebook, C2). |
| C28 | Later | Onboarding and tutorial (near launch; the owner asks questions meanwhile), sound and music, team hiring. |
| C29 | Demo worlds | Worlds from the M2 demo **do not carry over**; new games start fresh. |

---

## 2. Rules this layer must keep

- **R1 Real model.** Every word attributed to an AI is real model output, unedited. Social posts and news quote replies verbatim inside game-written frames ("@maya_codes asked it… and it said:"). The game's own text (headlines, rival trash-talk, report cards) is always presented as game commentary, never as an AI's words.
- **R2 Opacity.** Rivals see only public facts (§13.3). The player sees rivals the same way.
- **R3 Fair compute.** Game outcomes depend on in-game compute and money, never on the player's hardware speed (game-paced reveal stays, M2 Task 14).
- **R5 Emergence.** The orb, milestones, discoveries and fame come from measurements. There are no unlock lists or level tables.
- **R6 Lineage.** The model lineup is a set of labels over versions. Growth stays one-way.
- **R7 Plain language.** No ML jargon on any control.

---

## 3. Screens

The app gets a **side menu** with seven screens. The top bar (logo, company name, cash, profit this month, date, speed and pause) is on every screen.

| Screen | Contents |
|--------|----------|
| **Company** (home) | The orb with version name and personality words; buttons Train, Chat, Launch. Left: customers (total, weekly growth, mini region map), money (cash and profit sparkline). Right: a live feed of news and social posts. Optional goal progress and the next milestone along the bottom. |
| **Market** | The 9-region US map colored by market share (yours in your colors, rivals in theirs); customer groups with what each wants and how many you have; ad campaigns; spotlight customers. |
| **Money** | Coming in (by plan, group, region) and going out (training, serving, marketing, data checks, data centers, interest); plans and prices; the model lineup; computers and data centers; report-card archive; rescue status. |
| **AI Lab** | Today's Data → Prepare → Design → Train → Versions flow, restyled. The Train step shows the orb taking in data while the existing charts run. The notebook joins Data as a ninth dataset. |
| **Playground** | Chat (today's), plus an **Interview** mode. |
| **Arena** | Battles and league tables (users, revenue, overall score, each skill). |
| **Rivals** | Each rival's public profile: logo, products, prices, public scores, personality words, market share, reputation, followers, news. "Try it" chat with their live AIs. |

**Look (C7, C5):**
- Clean, smart, futuristic: glass-like panels over the neutral dark base, thin luminous accent lines and soft glows in the company palette.
- Generous spacing and large numbers for the things that grow, with animated count-ups and chart transitions.
- Space Grotesk for display text, a clean sans for body text.
- `prefers-reduced-motion` turns animation off.
- No sound (C28).

---

## 4. The orb (C8, C10)

- **Pure function:** `orbParams(m) → visual parameters`, computed in the client from the version's public measurements. It is deterministic, has no tiers, and every parameter is a continuous function.

| Visual | Driven by |
|--------|-----------|
| Core size and brightness | overall score (0–100) |
| Number and intricacy of rings | count of categories scoring ≥ 20, plus log(words read) |
| Ring colors | company primary → secondary gradient |
| Swirl speed and turbulence | traits: boldness, inventiveness |
| Smoothness of motion | trait: steadiness |
| Particle rate (sparks) | trait: verbosity |
| Glow softness | trait: warmth |
| Occasional flicker | trait: slip rate |

- **Before the first version:** a faint spark.
- **During training (Train step):** streams of particles flow into the orb. Their colors come from the dataset mix shares, and its brightness follows the revealed loss falling (the same events as the charts, game-paced).
- **Rendering:** a 2D canvas with requestAnimationFrame. A static frame for reduced motion and for tests, which assert on `orbParams` output.

---

## 5. Market and customers (C14, C24, C26)

**Regions:** the 9 US census divisions (New England, Mid-Atlantic, East North Central, West North Central, South Atlantic, East South Central, West South Central, Mountain, Pacific).
- Population is the real division population.
- The share of people willing to use an AI grows from about 10% in 2027 toward 60% over about ten game years (an S-curve; balance config).

**Groups and what they weigh** (balance config; the weights are over the eight benchmark categories plus product facts):

| Group | Cares most about | Price sensitivity | Speed vs smarts |
|-------|------------------|-------------------|-----------------|
| Students | knowledge, reasoning, price | high | balanced |
| Creators | creativity, language | medium | balanced |
| Developers | coding, instruction | medium | speed |
| Businesses | consistency, instruction, reliability, low slip rate | low | smarts |
| Everyday users | language, conversation, price | high | speed |
| Researchers | reasoning, knowledge | low | smarts |

Each region has its own mix of groups (for example more developers in the Pacific). It's balance config.

**Choice:**
- For every (region, group), each option is one company's plan for that group (Free, Personal or Business) or "no AI".
- An option's appeal combines:
  - the group-weighted **real scores** of the product behind the plan
  - price against the group's price sensitivity
  - speed (product size and distance to the nearest data center)
  - reliability (serving load)
  - satisfaction (from feedback, §7)
  - reputation
  - awareness of the company in that region and group
- A logit over the options gives target shares.

**Growth:**
- Actual customers move toward the target shares each game day at a rate set by awareness and word of mouth (∝ satisfied customers), minus churn.
- Unhappy groups churn faster.

**Awareness** comes from launch days, news, followers, word of mouth and **ad campaigns**. A campaign is a budget per day, a region (or nationwide), a group (or everyone) and a duration.

**Spotlight customers:**
- About 6 named people (one per group) are generated per world with a region, a group and a short bio.
- Their posts and reactions put a face on the numbers. Their questions come from their group's pool (§7), and their replies are real.

**Pace target:** the first paying customers arrive after about 30 minutes of play (C26). The balance harness (§14) holds this.

---

## 6. Products, plans and launches (C17, C18, C20)

- **Lineup:**
  - A company has up to **3 products**, each `{name, versionId, status: countdown | live | retired}`.
  - Any version can back a product (an older small one, the newest big one, or a version from a second, fresh lineage).
  - Product names are player text (2–24 characters, cleaned).
- **Plans:**
  - **Free** has no price.
  - **Personal** is a monthly price.
  - **Business** is a monthly price per seat. Businesses, developers and researchers buy several seats each (balance config).
  - Each plan points at one live product, or is closed.
- **Size trade-off:** bigger products score higher but are slower and cost more to serve per customer (§8). Groups weigh speed and smarts differently (§5).
- **Launch day:**
  - "Launch" picks a version, a product name (new or replacing), which plans it powers, and a countdown of 3–14 game days.
  - The countdown builds **hype** (from reputation, followers and any campaign).
  - On the day: an awareness boost scaled by hype, a burst of feedback sampling (§7) whose best and worst replies become the first posts, and a launch news story built from its real scores ("Acme Titan launches: #2 in coding").
  - A product can be **retired**. Its customers move to whatever product now powers their plan.

---

## 7. Feedback, news and fame (C13, C15, C21)

**Customer questions:**
- For each group, a pool of about 50 customer-style questions, written like benchmark items, **each with an automatic check**: a fact with accepted answers, a coding task checked by running it in MiniPy, an instruction with a format check, or an open prompt checked for legibility.
- These are new content, built by `airace_content` alongside the benchmarks.

**Weekly sampling:**
- Every game week, each live product answers a small batch (balance: 6 questions, one per group) as a low-priority `feedback` job.
- The worker generates the replies and scores them with the checks.
- The result enters the world as a logged input: `feedbackScored{productId, results: [{group, questionId, reply, score}]}`.
- Satisfaction per (product, group) is a moving average of the scores.

**Social posts:**
- The highest- and lowest-scoring replies of each batch become posts.
- Each post is a game-written frame, attributed to a spotlight customer or a generated handle, that quotes the **verbatim** reply and the question.
- Tone (praise, joke, roast) is picked from the score.

**News (rare and meaningful, C13):**
- The sim turns real events into headlines: launches, rank changes (#1 in a skill), user milestones, viral posts (an extreme score with a high follower count), outages (serving overload), battles, interviews, discoveries, rescues and rival moves.
- Each item has a sentiment.
- Random events exist only as consequences of state, for example a serving overload after viral growth, or a data-quality scandal when a product with planted false facts in its training is caught answering wrong at scale. Pure chance is limited to timing jitter.

**Fame:**
- **Reputation** (0–100): a moving average of news sentiment and customer satisfaction, weighted by reach.
- **Followers:** grow with launches, viral posts, battles and interviews, and decay slowly.
- **League tables:** users, monthly revenue, overall score and each category, against rivals.

---

## 8. Money and computers (C16, C19, C23)

**Ledger:** every amount is booked daily into categories.
- In: personal plans, business plans.
- Out: training, serving, marketing, data checks, data-center building, data-center upkeep, loan interest.
- The Money screen shows them by plan, group and region.

**Serving:**
- Each customer (free ones too) uses compute every day, scaled by the size of the product behind their plan.
- Owned data-center capacity is used first.
- The rest is rented from the cloud at a higher unit price.
- Cloud capacity can only grow by a balance-set percentage per day. A viral spike beyond that overloads serving: slower answers and lower reliability, which lower satisfaction and can make news.

**Data centers:**
- Built in a region: a capital cost, a build time in game days, a capacity, and monthly upkeep.
- Customers in that region and its neighbours get faster answers.
- **Training/serving slider:** the share of owned capacity lent to training. Training uses that share first, then rents the rest. Owned compute shortens and cheapens runs, and costs serving headroom.

**Monthly report card** (game month end):
- revenue, costs and profit against last month
- customer growth by group
- the best and worst moment (top post, biggest headline)
- rank changes against rivals
- a grade (A–F) from profit trend, growth and satisfaction

**Rescue (C19):**
- At a month end with cash below zero, the player chooses:
  1. **Emergency loan:** cash restored to a cushion, repaid monthly with heavy interest.
  2. **Investor rescue:** cash restored. The investor takes a share of the company, shown on the profile; company-value goals count only the player's share.
- If no choice is made within 14 game days, the game takes the loan.
- A company cannot otherwise die.

---

## 9. Rivals (C22)

**Cast:**
- Each new world draws an **arch-rival** archetype at random: arrogant tech giant, cheeky upstart or mysterious lab.
- **Three rivals** come from styles: premium, budget, specialist and consumer.
- Names, logos and palettes are generated per world from content lists.

**Real AIs, prepared once:**
- `airace-ml build-rivals` (new, one time, about an hour on the RTX 3050, like `build-judge`) trains a **rival pack**: for each of about 8 rival lineages, 3–5 versions of growing size and different data strategies.
- For each version it measures bench scores and fingerprint traits, and pre-generates replies to every customer-question pool (§7).
- It writes checkpoints and `rivals.json` under `data/rivals/v1/` (gitignored), with a data identity like the catalog's.
- A world uses 4 lineages from the pack.

**During play:**
- Rivals make business decisions by heuristic in the sim: prices, campaigns, data centers, launch timing, by style.
- They release pack versions over game time, with schedules jittered by the world's seed.
- Their feedback uses the pre-generated replies, so it costs nothing.
- Live generation happens only when the player chats with or battles a rival, as a low-priority job on small models.
- If the pack is missing, worlds start without rivals and the Rivals screen says how to build it.

**Trash-talk:** the arch-rival comments on real events in the news, as game text that is labelled as such.

---

## 10. Battles, interviews, discoveries, notebook (C27)

**Battles:**
- The player picks a rival product and a skill.
- A `battle` job asks both AIs 3–5 **fresh** challenges, drawn from seeded content generators like the benchmarks', never from the scored suite.
- Replies stream side by side.
- Scoring uses the benchmark checks. Creativity and writing use the reference judge and novelty index.
- The result (`battleResolved`) moves reputation, followers, a per-skill Elo ranking and the news.
- A cooldown per rival prevents farming.

**Interviews:**
- In the Playground's Interview mode, the player asks free questions or picks prompt cards (some with automatic checks).
- They choose answers to publish.
- An `interview` job scores the published answers: checks where they exist, otherwise judge fluency, plus traits.
- `interviewPublished` produces audience reaction: followers and reputation up or down, and a news item. An embarrassing answer can go viral.

**Discoveries:**
- After each eval, the worker compares the tag-level breakdown and traits with the previous version and with the population (rivals' and the player's other versions), as architecture spec §4.9 describes.
- Large differences become `discoveryFound` items with real evidence (the prompt and reply).
- The player may publish one as news.
- There is no fixed list.

**Notebook (C2):**
- Up to about 5,000 words of the company's own text, saved per company (private, R2) and filtered on save by a wordlist and pattern filter.
- It appears in the Data step as a ninth dataset with its own share. The trainer already accepts `notebook` text.
- It is too small to read much of, so a large share causes memorization and parroting, a visible trade-off.

---

## 11. Milestones and goals (C25)

**Milestones** are checked by the sim on each relevant input. All come from measurements:
- *First full sentence:* the eval reports at least one legible fingerprint reply, using the G2 legibility check.
- *First correct fact:* at least one knowledge item correct.
- *First working program:* at least one coding item passes.
- *First 1,000 / 100,000 / 1,000,000 customers.*
- *First paying customer.*
- *First profitable month.*
- *#1 in a skill.*
- *First battle won against the arch-rival.*
- *A data center opened.*
- *An A on the report card.*

They appear on a trophy shelf with the day they happened.

**Optional goal**, chosen at world creation:
- none
- reach N customers
- reach $X monthly profit
- best overall AI
- beat the arch-rival on customers
- company value

An optional deadline can be added. The Company screen shows progress. Reaching the goal shows a victory screen with the company's story (charts and highlights), and play can continue.

---

## 12. The player's hardware comes first (C22)

**Job priority, highest first** (broker, M2 Task 13):
1. The player's chats.
2. Battles and interviews (interactive).
3. The player's training.
4. The player's evals.
5. Feedback sampling.
6. Rival chats.

**Budgets:**
- Feedback is about 6 short generations per product per game week.
- Rival feedback is pre-generated.
- Rival models are small (tiny to medium) and are loaded only for chats and battles. The model cache evicts them first.

**Isolation:** the rival build is a one-time offline step. Nothing about rivals trains during play.

---

## 13. Architecture changes

### 13.1 Sim (`packages/sim`)
- **Version:** a new `SIM_VERSION` (`sim-2`). Worlds from `sim-1` are refused with a clear message (C29). The home screen lists them as "made with an older version" and cannot open them.
- **New state, by module:**
  - **`market`:** regions, groups, adoption, per-(company, region, group, plan) customers, awareness, satisfaction.
  - **`products`:** lineup, plans and prices, launches.
  - **`ledger`:** daily bookings, month summaries, report cards.
  - **`compute`:** cloud capacity, data centers, the training/serving slider.
  - **`fame`:** reputation, followers, rankings, Elo.
  - **`feed`:** the last 200 news items and posts in state; older items are archived to the server database.
  - **`rivals`:** NPC companies with their strategy and pack lineage.
  - **`milestones`** and **`goal`.**
  - **`spotlights`.**
- **New commands:** launch, setPlan, setPrice, retireProduct, startCampaign, stopCampaign, buildDataCenter, setTrainingShare, chooseRescue, publishInterview, publishDiscovery, setGoal (at creation), saveNotebook (a server-side write that logs a hash only, R2).
- **New system inputs:** feedbackScored, battleResolved, interviewScored, discoveryFound, rivalReleased (scheduled from the pack, logged for replay).
- **Tick:** the daily market step (choice and diffusion), serving and compute, ledger bookings, rival heuristics, launches. On weekly and monthly boundaries: sampling requests (as outputs the server acts on), report cards, milestone and goal checks.
- **Determinism:** all of it stays deterministic and replayable. Float math stays inside the sim on one engine (the server); clients only display.
- **Balance:** `balance-v2` (and `balance-test-v2` for fast tests) holds every constant.

### 13.2 Server
- **Feedback service:** on each weekly boundary it enqueues `feedback` jobs for live player products, and feeds rival feedback from the pack. Results become `feedbackScored`.
- **Battle and interview services:** handle commands and enqueue jobs. Results become logged inputs. Battle replies stream like chat.
- **Rival service:** loads `rivals.json` like the catalog, places rival checkpoints in the blob store, schedules `rivalReleased`, serves rival chats.
- **Notebook store:** a private table. Its text reaches only train jobs.
- **Archive:** news and post history beyond the sim's last 200.
- **Projection (R2):** the public fields of another company are name, logo, palette, products (name, launch day, public scores, personality words), plan prices, rounded customers by region, reputation, followers, rankings and news. Never another company's cash, ledger, campaigns, data-center plans, notebook, training or unreleased versions.

### 13.3 ML (`ml/`)
- **Content:** customer-question pools per group, each with its check; battle challenge generators (reusing the bench generators with fresh seeds); interview prompt cards.
- **Worker jobs:** `feedback` (generate and check), `battle` (both models, checks or judge), `interview` (scoring), plus eval extensions: milestone signals (legible replies, first correct fact, first passing program) and discovery detection.
- **`build-rivals`:** trains and measures the rival pack, and pre-generates rival replies to the question pools.

### 13.4 Client
- **Shell:** the new app shell (side menu, top bar) and a small design-system pass (panels, glows, number and chart animation).
- **Screens:** Company, Market (SVG map of the 9 regions), Money, Arena, Rivals, and Interview mode. The orb component uses `orbParams`. Lab and Playground are restyled.
- **Data:** all of it arrives through the existing snapshot and patch view, queries and streams.

### 13.5 Failure handling
- A job that fails (feedback, battle, interview) is retried by the broker. A final failure leaves state unchanged: no satisfaction change, and a battle is called off with the money back.
- A missing rival pack or catalog degrades with a plain message (§9).
- Serving overload is a game state, not an error.

---

## 14. Testing and balancing

- **Sim:**
  - unit tests per module
  - fast-check properties: privacy for every new public field, determinism and replay, and **money conservation** (cash = founding + revenue − costs + rescues, with every amount booked once)
  - customers are never negative and never exceed the adoptable population
- **Balance harness:** headless bot companies (and rival heuristics) run years of game time on `balance-v2`. Targets:
  - first paying customers at about 30 minutes of play at 1× with typical choices
  - at least three distinct viable strategies (premium, budget, specialist)
  - rivals neither runaway nor dead
  - report-card grades spread across A–F
- **Server:** integration tests with the fake worker for each new job and service. Restart and replay mid-launch and mid-battle.
- **Python:** handler tests (feedback scoring, battle scoring, milestone signals, discoveries); a quick `build-rivals` on the tiny corpus.
- **Client:**
  - component tests for each screen
  - the orb's `orbParams` monotonic in scores and continuous
  - hostile text rendered as text everywhere (posts, product names, notebook)
- **End-to-end:** the solo E2E grows each phase: found → train → launch → customers and money move → feedback posts quote real replies → (phase 3) a rival launches, a battle → (phase 4) an interview, a discovery, a milestone.

---

## 15. Phases

Each phase ends playable, with all tests green and the E2E extended.

| Phase | Contents | Done when |
|-------|----------|-----------|
| **1. A real company** | New look and app shell; Company screen and orb; products, lineup, plans and prices; launch day; national market (one region) with six groups and spotlights; feedback sampling with real-reply social posts; news; reputation and followers; Money screen (in/out); ledger; sim-2 and balance-v2 | A new player trains, launches, gets paying customers in about 30 minutes, reads real customer posts, and watches money and customers move |
| **2. A growing business** | 9-region map; ad campaigns; cloud ramp and overload; data centers and the training/serving slider; monthly report card; rescue | Regions diverge by choices; a data center changes costs and speed; the report card and rescue work end to end |
| **3. Rivals** | Content (rival names and archetypes); `build-rivals`; rival heuristics and releases; Rivals screen and rival chat; league tables; battles with Elo | A world plays a game year with an arch-rival and three rivals; rivals launch real AIs; battles stream and score |
| **4. Surprises and goals** | Interviews; discoveries; notebook; milestones and trophy shelf; optional goal and victory screen | Each feature works in the E2E; milestones fire only from measurements |

---

## 16. Out of scope for this layer

Multiplayer (Stage 2, after this layer), tutorial and onboarding, sound and music, team hiring, licensing between companies, coaching (thumbs up and edits), the AI Hub and global leaderboards.

---

## 17. Temporary assumptions (can change without the owner)

- **Numbers:** prices, population scale, adoption curve, costs and all constants live in `balance-v2` and are tuned by the harness against the targets in §14.
- **Spotlight customers:** six per world, one per group.
- **Rival lineages:** eight in the pack, four per world.
- **Data centers:** one size; several may be built per region.
