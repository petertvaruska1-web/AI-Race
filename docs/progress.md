# Progress log

Newest entries first. Each entry: date, what changed, what's next, open issues.

## 2026-10-29 — Game layer Phase 2: "A growing business"
- **What a player can now do:** see customers spread over the 9 US regions on a glowing tile map (brighter where more of the willing people use your AI), open any region; run ad campaigns by region and group; watch serving capacity (the cloud grows a fifth a day, so a viral spike overloads it: fewer answers, unhappier customers, a warning and the news); build data centers (60 game days, $1.5M) that answer faster nearby, serve at 30% of cloud prices and can lend compute to training; read a monthly report card with a grade from A to F and the month's best and worst posts; and, when cash runs out at a month end, choose an emergency loan or an investor (or the game takes the loan after 14 days).
- **Built** (`docs/superpowers/plans/2026-10-28-game-layer-phase2.md`, 14 tasks): sim-3 with regions, campaigns, compute, data centers, report cards, rescue and financing in the ledger; balance targets for each; the map, campaign, data-center, computers, report-card and rescue windows; the overload warning and rescue banner. The full record and every ruling: `docs/handoff/sdd/ledger-game-layer-phase2.md`.
- **Balance:** ads first cost $0.05 per person reached and were free money (one blitz left a company $633k richer in 110 days); at $0.15 they pay back in about three months.
- **Verified:** TypeScript 421 tests; checks clean; the end-to-end game (now with regions, a campaign, a report card and the computers) passed twice; played in the browser on the GPU. The game test found one real bug (campaigns and data centers had no ids over the live connection), now fixed.
- **Note for the owner:** worlds from Phase 1 now show as "made with an older version"; start a new world (restart `npm run dev` first).
- **Next:** Phase 3 "Rivals", after the owner's go.

## 2026-10-28 — Immersion pass (owner decision C30)
- **What changed for the player:** the screens now move and open up. Anything with a number opens a window that grows out of what was clicked and explains it with its history: customers (stacked day by day by group), each customer group (its customers, share, mood, plans, spotlight person and posts), each product, the AI itself (orb, score bars, style), money (cash day by day, profit month by month), each month's report, each income and cost line, and each post. Windows stack (Back returns), Escape and the backdrop close them, and they update live.
- **Motion:** screens slide in, a line under the top bar fills over each game day, cash and customers float their changes, new posts slide into the feed, news about your company pops up as a toast with a small burst for milestones, new customers stream into the orb in their groups' colors, the Market draws each group's customers as a crowd of dots, money bars grow in, and the training charts expand to full width. Everything is instant for players who ask their system for less motion.
- **Built:** design `docs/superpowers/specs/2026-10-28-immersion-design.md`. The sim keeps 120 days of history per company (in its owner's view only; worlds saved before keep working). Client: a window stack, day and month charts, the eight windows, and the screen changes. `npm run dev` can now run a second game beside the first (`AIRACE_WEB_PORT`).
- **Verified:** TypeScript 355 tests; lint and type checks clean; the end-to-end game passes (its news check now looks in the feed, since a toast shows the same headline). Played in the browser on the GPU: trained an AI (overall 39), launched it, and opened every kind of window with live numbers.
- **Next:** Phase 2 "A growing business", straight away (the owner's go), with the same feel in every new screen.

## 2026-10-28 — Game layer Phase 1: "A real company"
- **What a player can now do** (single player, `npm run dev`): train an AI, launch it as a named product on the free, personal and business plans after a hype-building countdown, watch customers arrive in six groups, read customers' posts that quote the AI's real replies, read news of real events, and follow money in and out month by month. The whole app has a new look: glass panels lit in the company's colors, and the AI as a glowing orb that grows and changes with its real scores and personality.
- **Built** (`docs/superpowers/plans/2026-10-27-game-layer-phase1.md`, 15 tasks):
  - Game rules (sim-2, balance-v2): products, plans and prices; a national market of six customer groups with spotlight customers; the money ledger; feedback satisfaction, posts, news, reputation and followers; what each player may see, with privacy checks over random games; a balance harness that retuned the economy (a typical first AI now stays a small startup).
  - Server: weekly customer feedback jobs (plus one on launch), jobs ordered chat, training, scoring, feedback; worlds from an older version listed as outdated.
  - Python: customer question pools drawn from the benchmark's held-out generators, and the worker's feedback job.
  - Web app: the design system and app shell, the orb, and the Company, Money and Market screens with the launch dialog.
- **Verified:** TypeScript 319 tests; Python 1,616 tests (2 min 45 s); lint and type checks clean; the end-to-end game (now with launch, customers, posts and money) passed three times. On the RTX 3050 a first AI trained in about 80 s and was scored in 15 s (overall 43); launched as "Wave", it had 5,555 customers two days later, posts quoted its real replies ("Phiated.", "BYetch"), and chat answered within a second while feedback batches ran (1–2 s each).
- **Final review** (a self-review, by the owner's wish for no helper agents this milestone): fixed one serious hole (an id named like a built-in object part, such as `__proto__`, could reach every object on the server) and the Money screen showing rising costs in green. Seven small items are deferred in `docs/handoff/sdd/ledger-game-layer-phase1.md`.
- **Owner feedback after playing it (C30):** make it engaging and immersive (motion, transitions, windows that open, things to click, expanding charts, visuals), not just graphs.
- **Next:** an immersion pass on these screens, then straight into Phase 2 "A growing business" (regions map, ad campaigns, data centers, report card, rescue).

## 2026-10-27 — Game-layer design interview ("Open for Business", single player)
- The owner judged the Stage 1 build a good demo but far from the game they want, and asked for a full design of the playable game before multiplayer.
- A design interview set decisions C7–C29:
  - a playful tycoon with a clean futuristic look in the company's colors
  - the AI as a glowing orb driven by real measurements
  - customers in six groups across 9 US regions
  - real-reply social posts and headlines
  - free, personal and business plans with a lineup of up to three AIs
  - launch days; reputation, followers and rankings
  - a surprise arch-rival plus three rivals with real, pre-built AIs (the player's AI comes first on the hardware)
  - rent-then-build data centers
  - a monthly report card and a costly rescue
  - milestones plus an optional goal
  - battles, interviews, discoveries and the notebook
- **Design:** `docs/superpowers/specs/2026-10-27-game-layer-design.md`, delivered in four playable phases.
- M2 Stage 2 (friends) moves after this layer. Demo worlds will not carry over (sim-2).
- **Next:** the owner reviews the design document; then the Phase 1 implementation plan.

## 2026-10-27 — Milestone 2 Stage 1 playable (Tasks 10–24)
- **The first playable version works on the owner's PC** (`npm run dev`, then http://localhost:5173). In one session it went:
  - sign-up, a world, a company in its own colors (the whole dashboard takes them)
  - a real first AI trained on the RTX 3050 through the game: 13 game days, $41,596, with live charts and its own words revealed at game pace
  - v1 scored on all eight skills (overall 42; "inventive, often slips, reserved")
  - a streamed chat ("Once upon a time there was a dog named John…")
- **Built:**
  - Server: the realtime gateway, signed blob store, dataset catalog, job broker with leases, training service with game-paced reveal, playground chat.
  - Python: `export-catalog`, streaming replies, the dial-out worker (`airace-ml worker`) with train, eval and chat jobs.
  - Web app: home, founding (logo builder, 12 palettes), dashboard, AI Lab (Data, Prepare, Design, live Train, Versions), Playground, Benchmarks.
  - `npm run dev` and an end-to-end test (`npm run e2e`) that plays a whole game, including a page reload and a server restart mid-run.
- **Tests:** TypeScript 217 passed; Python 1,610 passed (158 s); end-to-end passed twice (46 s); lint and type checks clean.
- **Found and fixed along the way:**
  - an interrupted model upload left a half-written file
  - a worker that missed heartbeats could be handed its own expired job again
  - a "review" setting without an earlier AI would have charged for a run that could only fail; it is now refused up front
  - the dev runner took the web app's port
- **Next:** Stage 2 (Tasks 25–28: invites, sharing AIs with friends, hosting from this PC), on the owner's go.

## 2026-10-23 — Milestone 2, Tasks 1–9 (simulation and server core)
- Branch `m2-two-friends`, executed inline with a commit and push after every task. `npm test`: 122 passed; `npm run check` clean.
- **Built:**
  - TypeScript monorepo (Tasks 1–2): shared message formats, with untrusted text cleaned and length-checked.
  - The deterministic game simulation (Tasks 3–6):
    - worlds, companies in their own colors (12 presets, readable on the dark dashboard), the clock and host handover
    - training cost and game-day duration from real compute
    - results revealed at game pace, refunds, versions and scores
    - a per-player view that hides everything private
  - The server core (Tasks 7–9):
    - an embedded database (no install needed) and guest accounts with hashed sessions
    - worlds run by a single-writer actor with a permanent input log, save points and crash recovery
- **Decisions** (ADR 0008 items 11–13):
  - Two balance numbers in the plan were off by a factor: the first AI costs $41,596 of the $250,000 founding cash, over 13 game days.
  - A shared AI's public entry does not show its brain size (R2).
  - Ids are unique across a world.
- **Found by the property tests and fixed:** a version id reused while another result with that id waited for its game day.
- **Next:** Task 10 (realtime gateway). Paused here because the week's usage is at 95%.

## 2026-10-22 — Milestone 1 closed; Milestone 2 planned
- **Milestone 1 closed** at 5 of 7 (owner decision C4): merged into `main` and tagged `m1-complete`.
- **Owner decisions:**
  - C5: each company's palette colors its logo and the player's whole dashboard, over a neutral dark base. This resolves D1; the base style is a temporary assumption.
  - C6: the first playable version may be single-player, with multiplayer designed in.
- **Milestone 2 plan:** `docs/superpowers/plans/2026-10-22-m2-two-friends.md`, 28 tasks.
  - Stage 1 (single player, Tasks 1–24):
    - TS monorepo, protocol and deterministic sim
    - server with guest accounts, world actors with input log and snapshots, realtime gateway, job broker and blob store
    - Python catalog export and dial-out worker (train, eval, chat)
    - React client (company founding, AI Lab, live training, playground, benchmarks)
  - Stage 2 (Tasks 25–28): invites, sharing AIs with friends, hosting from the PC, the two-friends end-to-end test.
- **Next:** execute Task 1 on branch `m2-two-friends`.

## 2026-10-22 — Local GPU session: real corpus, judge and feasibility gate (5 of 7 pass)
- **Setup (2026-10-21):** default suite 1559 passed in 153 s on the owner's PC. pytest -m gpu passed 4 tests on the RTX 3050. One Windows-only test was fixed.
- **Task 15, corpus v1** (ADR 0004; `docs/data-sources.md`):
  - 139.4M tokens, 936k documents, tokenizer tok-v1 frozen.
  - All licences re-verified; Simple Wikipedia dropped (CC-BY-SA, share-alike).
  - Simplicity thresholds calibrated on real data; sources pinned at the commits read; the TinyStories mojibake repair extended.
- **Evaluators:** novelty index (16 s) and judge (about 36 min on GPU), rebuilt after each corpus change.
- **Feasibility gate, four runs and three iterations** (ADR 0005–0007; report: `docs/reports/m1-feasibility.md`):
  - Run 1: 4 of 7. Run 2: 3 of 7 (the PC slept mid-run). Run 3: 4 of 7. **Run 4: 5 of 7.**
  - Content fixes:
    - Stories told on request (`tinystories_requests`): creativity 0 → 20.
    - Quiz sheets in Reference Facts (`fact_qa`): knowledge 5 → 77.
    - Denser planted false facts and a stricter quality score.
  - Training fixes: the first model trains on 12.6M tokens, and the default batch is 8,192 tokens.
  - Measurement fixes:
    - G2 pools 180 replies.
    - G4 fingerprints sample 8 replies per probe.
    - G7(a) prompts with clean pages.
    - G7(b) fact arms train on web alone.
  - **Passing:** G1 speed (68 s GPU, 17 min CPU), G3 differentiation, G4 seed variation, G5 growth, G6 forgetting.
- **Owner decision C3 (a babbling start):** the first AI's legibility bar is 60% (was 70%, which no first model reached).
- **Open, for the owner:**
  - **G2:** the first model is 59% legible (107 of 180), one reply short of 60%. Model size, budget, learning rate, batch and corpus mix were all swept; none moves it reliably.
  - **G7(a):** thorough cleaning lowers garbled output only slightly (2.9% → 2.4%), and not on every seed. Small models drop random typos by themselves, and neither more samples nor more training changes that. Fact-checking works (67% → 48%).
- **Spec updated:**
  - §0: C3.
  - §4.11: the first model trains on about 12.6M tokens, with a CPU fallback of at most 20 minutes.
  - §4.12: G2 is judged at 60% on pooled replies.
- **Next:** the owner's decisions on G2 and G7(a), then the milestone wrap-up and the merge to `main`. M2 starts with the D1 visual-direction question.

## 2026-10-21 — Milestone 1 code finished: final review and fix wave complete
- **Final whole-branch review:**
  - 18 finders, a completeness critic, and 6 extra finders that checked real Hugging Face rows.
  - De-duplication into 56 findings, verified by 13 Sonnet skeptics.
  - 43 confirmed findings were fixed; F6's refutation was overruled, and it is among the 43. 10 refutations were accepted, 2 items were deferred and 1 was already done.
  - Report: `docs/handoff/final-review/final-review.md`.
- **Fix wave:** six Sonnet implementer groups (benchmark items, data quality, build operations, trainer, evals, gate and CLI), each with an adversarial Sonnet review. All 43 fixes landed (`40381f0..9d803cd`), plus two controller follow-ups (`ceee83f`, `a2837b4`).
- **Highlights:**
  - The coding benchmark's 50 function items are answerable again.
  - Knowledge answers are even, and creativity multiplies by variety, so models that know nothing score near 0.
  - The judge trains in micro-batches, so it should fit an 8 GB card.
  - The hour-long build retries and resumes, pins its sources, and refuses a tiny build's tokenizer.
  - Real Gutenberg and TinyStories text arrive clean.
  - The gate runs 3 seeds, measures its own noise floor, and counts only garble-like words.
- **Decision records:** ADR 0001 addendum, ADR 0002 (gate measurement rules), ADR 0003 (corpus build rules).
- **Tests:** default suite **1561 passed, 1 skipped** (166–232 s on cloud CPUs, depending on the machine); slow suite 5 passed (280 s); ruff clean.
- **Model tiers** (the owner's suggestion): the top model for rulings only; Sonnet for verification, implementation and review.
- **Next: the local GPU session** (`docs/handoff/README.md` §5):
  - setup and GPU tests
  - Task 15 (license decision, simplicity calibration, full build)
  - build-novelty-index and build-judge
  - the real feasibility gate

## 2026-10-14 — Milestone 1 code complete (Tasks 12–20); final review in progress
- Cloud session, branch `claude/upbeat-franklin-k1192h`. `m1-it-learns` fast-forwards to it.
- **Tasks 12–14 and 16–20 (code and tests) are complete and reviewed.** Each task had its own review and fix rounds (1–3 per task), and every ruling is in `docs/handoff/sdd/ledger.md`.
  - What was built: code, instruction and grammar generators; content text processing; the offline build pipeline; the bench-v1 suite; the novelty index, judge and creativity scoring; the personality fingerprint; the CLI; the feasibility gate code.
  - Latest code commit: `93f2737`. Default suite: **1247 passed, 1 skipped**. Ruff is clean, and the quick feasibility gate passes on CPU.
- Measurement rules beyond the plan are recorded in `docs/decisions/0001-bench-v1-measurement-rules.md`. They keep scores of untrained, constant, looping and parroting models near 0.
- **In progress: the final whole-branch review.** Usage limits interrupted it repeatedly.
  - Finding and de-duplication are complete: 56 findings (1 Critical, 14 Important, 41 Minor).
  - Verification runs on Sonnet agents, after the owner suggested using lighter models where possible.
  - A fix plan with tentative rulings is drafted: `docs/handoff/final-review/fix-wave-plan.md`.
  - Next: decide the findings, then one fix wave of six Sonnet implementers, one re-review, and the handoff.
- **Pending on the local GPU machine:**
  - Task 15 (license check and the real dataset build)
  - `build-novelty-index` and `build-judge`
  - the real feasibility gate runs (Task 20 steps 6–9)
  - The checklist is in `docs/handoff/README.md` §5.

## 2026-10-07 — Milestone 1 in progress; handoff to cloud session
- Branch `m1-it-learns`. **Tasks 1–11 complete and reviewed** (scaffold, tokenizer, transformer, growth, inference, corpus/sampler, training config, trainer, MiniPy sandbox, knowledge base, reasoning/pattern benchmarks). 699 tests pass.
- Task 12 (code/instruction/grammar generators) is implemented at `f4eaa83` but not yet reviewed. Next step: `docs/handoff/task-12-dispatch-notes.md`.
- **Resume guide: `docs/handoff/README.md`.** The full execution ledger with every ruling: `docs/handoff/sdd/ledger.md`.
- Cloud session does the code tasks; the local GPU machine does the dataset build, judge training and feasibility gate runs.

## 2026-10-06 — Project start
- Repository initialized (greenfield; only the two design PDFs existed).
- Environment: Windows 11, Node 24.20, npm 11.19, Python 3.14 (system), git 2.55, RTX 3050 8 GB (CUDA cc 8.6), Ryzen 5 5500, 32 GB RAM. No Docker or Postgres; dev setup must not need them.
- Creative decisions: C1 shared session clock; C2 small Notebook dataset (spec §0).
- Architecture spec written: `docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md`.
