# Progress log

Newest entries first. Each entry: date, what changed, what's next, open issues.

## 2026-11-13 — The next-level plan for the training system (written; waiting for the owner's go)
- **Asked by the owner:** a full plan to take the training system "to the highest level", saved for later sessions.
- **Design:** `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`. Seven phases, each passed on one fixed scorecard: the measuring stick, small engine 3 items, training text 2 (conversations in which the AI is an AI, typos only on the user's side, pools twice as large), a voice of its own (an AI is told its name and company and has learned to read that), brain design 2 (only if a screening shows it earns its place), one release, and training that behaves on other people's computers.
- **Found while planning (measured):** the conversation text is 63% two people chatting, which is why AIs invent names and talk about school; typos were being put into the AI's own answers; the optimizer and the step size have no head-room left (bigger steps: 10% faster, worse; smaller: slower, no better); a larger vocabulary would hold at most 14% more text for the largest disruption possible, so tok-v1 stays.
- **Plan A (phases A–D, 19 tasks):** `docs/superpowers/plans/2026-11-13-next-level-a-text-and-voice.md`. Plans for brain design 2, the release and other computers are written when their gates are reached.
- **Next:** on the owner's go, plan A from Task 1. Two things the owner may still veto are in the spec's section 0 (an AI knows its name; the first AI's default reading leans toward conversations).

## 2026-11-12 — Training engine 2 (built; gate run again: 5 of 7)
- **What changed for players:** AIs learn about one and a half times as much from the same reading and train faster; they cope with messages typed in lowercase or without punctuation; they ramble less; new worlds can train two bigger brains (Large, Huge) and a longer Marathon length; friends' computers can train big brains through the free link; a helper keeps its PC awake while it trains.
- **Measured on the RTX 3050** (ADR `docs/decisions/0009-training-engine-2.md`): first AI 70 s → 55 s with held-out loss 2.57 → 2.45; fact questions typed in lowercase without "?" answered right 17% → 65% (first AI), 93% for a three-run Medium lineage; well-formed replies 50% → 80%. Large trains at 46,700 tokens a second, Huge at 26,700; the longest run (Huge, Marathon) takes about 32 minutes.
- **Design and rules:** spec `docs/superpowers/specs/2026-11-10-training-engine-2-design.md` (the owner asked for a fresh design, not the 2026-11-07 plan; only the owner's decisions C49 and C54–C56 were kept). New worlds use `balance-v3`; worlds made before keep `balance-v2`. Old AIs keep their scores.
- **Glitches fixed on the way:** a run with a parent that moved to another card failed for good; a finished run whose upload failed was trained again; a failing clean-up could leave a run "training" for ever; charts kept old points when a run started over; cards read "15× speed"; the Lab's minutes were about four times too long for a Medium brain; a 4 GB card was allowed a brain it could not hold; uploads over 100 MB could never get through the friends' tunnel; helpers from an older version of the game now get no work and their card says to run the setup file again.
- **Verified:** ML 1,800+ tests and ruff clean; TypeScript 832 tests and checks clean; the end-to-end browser tests pass (the solo game and both two-player worlds, Fair mode off and on). Not yet done: a played session with the new brains.
- **The gate with engine 2** (`docs/reports/2026-11-12-gate-engine-2.md`, ADR 0009 section 8): 5 of 7, as before, but not the same five. Legibility now passes (80% well-formed as the game talks; 59% before). Differentiation fails on one of its four leads: every AI knows far more facts (a conversation-heavy AI 46 → 64), the facts specialist a little less (77 → 71), so its lead is under the bar; nothing tunable recovers it and it is accepted (every specialist's overall score rose by 7 to 14 points). Cleaning's known gap (G7) is unchanged.
- **Rivals:** a second pack is built (`data/rivals/v2`, engine 2; the premium rivals end on the Large brain with twice the reading). Rivals' first AIs are 7 to 13 points stronger than before, their last ones about level, the top premium rival 60 → 64. Worlds made before keep the earlier pack (`data/rivals/v1`).
- **Learned on the way** (ADR 0009 section 9): a grown brain ends about where a brand-new brain of its size does on the same reading, so a bigger brain pays off on the longer lengths; two runs in a row cost a little against one long run.
- **Next:** (1) play a new world with a Large or Huge brain; (2) friends must run the helper's setup file again (their helper is the old engine); (3) the open points of ADR 0009 section 11.

## 2026-11-07 — Licensing (built)
- **What players can now do:** rent their AIs to other companies, or license others' AIs for their own customers, for royalties per customer a month. An AI on sale gets "Offer for license" in its window (price $0.10–$20 a customer a month, 3, 6 or 12 months, the customer groups, and "only one company may license it"). Offers are public: the AI Directory lists them and the Rivals cards show each rival's, with "License it" (pick your AI on sale and which of the offer's groups). Players can also "Ask to license" another player's AI; a request pops up as a toast with Accept and Decline. In the licensed groups the licensed AI competes with its own measured skills and answers those customers on its maker's card; posts say "powered by {AI}" and quote its real replies. Royalties build up daily and are paid at each month's close (Licenses / License income on the Money screen and the report card). A license ends at its term, when its AI leaves sale, or early by the licensee for one more month's royalties. A Licenses panel on the Company screen lists them. Rivals offer their newest AI at their style's price and ask to license a player's AI that beats theirs by 10 points in their specialty.
- **Balance:** keeping a rival's AI licensed for a typical company's weakest group for a year costs about 11% of the year's profit (royalties count free customers too), so it's a choice, not a must-have. A rival's request pays at most a tenth of its own revenue (otherwise the consumer rival would have paid a player about $1 million a month).
- **Played on the GPU** (`data-check`, a game year at 4x): Pengu licensed Sable Logic's Scholar for its creators; the news ran, creators' posts quoted Scholar's real replies "powered by Scholar", royalties took 11–17% of income, and the license ended when Sable Logic replaced Scholar. Without it, profit rose. No rival asked for Pengu (it led none of their specialties).
- **Verified:** TypeScript 819 tests, checks clean, `npm run e2e` 3/3 (the solo game takes a rival's offer and sees Licenses on its report card; two friends license an AI and the maker sees License income). Record: `docs/handoff/sdd/ledger-licensing.md`.
- **Next:** the AI Hub plan (`docs/superpowers/plans/2026-11-04-ai-hub.md`), on the owner's go; Stage 2's real two-PC session is still to do.

## 2026-11-06 — Departments (built)
- **What players can now do:** a **Team** panel on the Company screen funds four departments, each $0–$500,000 a month in $10,000 steps, booked daily as the money's new **Team** line (and in report cards). A team grows toward its budget over about a month and each extra dollar does a little less. **Research:** each run reads more text for the same money (the AI really reads it). **Data team:** checks cost less. **Sales & marketing:** awareness rises everywhere a little every day. **Support & operations:** bad answers sting less, fewer unhappy customers leave, serving is cheaper. Each row says what its team does right now ("Training reads 13% more text"). Rivals fund departments by style from last month's revenue. An absent owner's budgets pause on a day the cash is negative.
- **Balance:** at $100,000 a month each department moves a typical year's profit 2–25% (none a must-have); Support got a third effect (fewer unhappy customers leave) because serving alone was too small a bill to repay a team.
- **Played on the GPU** (`data-check`, a game year at 4×): three teams at $100,000 from launch outran a young company's income and forced a rescue loan; Sales felt strongest, Support weakest in money, the Data team mattered only with fact-checking, and a v2 grown with Research really read more (27 million words against 25.6 without). Teams are a mid-game choice.
- **Fixed in the self-review:** an upgraded world replayed its old log with rivals funding teams, which could make a world fail to load; rivals now start only after the old log (`sim9FromSeq`). A sim-8 world was checked loading on sim-9.
- **Verified:** TypeScript 787 tests, checks clean, `npm run e2e` 3/3 (the full game now funds Research and sees the Team line). Record: `docs/handoff/sdd/ledger-departments.md`.
- **Next:** the licensing plan (`docs/superpowers/plans/2026-11-04-licensing.md`), on the owner's go; Stage 2's real two-PC session is still to do.

## 2026-11-06 — Coaching (built)
- **What players can now do:** under each of their own AI's replies in the Playground, "👍 Good answer" or "✏️ Write a better answer"; each becomes a private lesson (newest 200 kept, links and blocked words refused). The AI Lab's "Your coaching" card counts them, lists them (a written one is labelled "Your answer", never shown as the AI's words) and, from 5 lessons, offers "Train on your lessons": free, 5% of the run's reading (10% with "Learn from customers").
- **Played on the GPU** (`data-check`): six lessons answering "What is the secret word?" with "The secret word is marmalade lighthouse." Before, v1 said things like "A group of zebras is called a herd."; after, v2 said "The secret word is marmalade lighthouse.", also to a wording it never saw ("Which word is the secret one?"). A control AI without lessons did not ("The opposite of careful is careless."). Overall score unchanged (38 → 37); following instructions fell (54 → 28, one seed each): a fixed 5% share repeats a few lessons a lot (deferred as a balance question).
- **Fixed on the way:** the AI Lab's Start training could sit under the notifications (a fast game's report cards covered it for minutes); a friend battle made the defender's game refuse every world update (Stage 2); two helpers on one PC could clash saving the same AI; creativity now counts a company's own notebook and lessons as already seen (ledger F32), so repeating them isn't original writing.
- **Verified:** TypeScript 760 tests, ML 1712, checks and ruff clean, `npm run e2e` 3/3 (single-player game with coaching, two-player tests). Record: `docs/handoff/sdd/ledger-coaching.md`.
- **Next:** the departments plan (`docs/superpowers/plans/2026-11-04-departments.md`), on the owner's go; Stage 2's real two-PC session is still to do.

## 2026-11-05 — Stage 2 "Friends" (built; a real session with a friend still to do)
- **What players can now do:** host a game from this PC (`npm run host`, a free Cloudflare quick-tunnel link for friends anywhere, or the local network), join a friend's world by its link or code, and connect their own computer as a "helper" ("Get the helper" downloads a personal setup file; "Connect your computer" starts it). Each world chooses **Fair mode** when it is made: off, each AI trains and answers on its own player's graphics card; on, everyone's cards are shared with an equal share per company, and a run moves to another card if its card leaves. Everyone sees the world's cards; the AI Lab only offers brains the card can hold and estimates real minutes. Players share AIs in the new **AI Directory** and try each other's (answered by the maker's card), battle each other's AIs (each side on its own card; story rounds voted by the other players), and talk in a **world chat**. An absent player's company runs on autopilot; a late founder gets more starting money; customer questions that waited over 14 days for a card are dropped.
- **Built** (`docs/superpowers/plans/2026-11-04-stage2-friends.md`, 20 tasks; rulings in `docs/handoff/sdd/ledger-stage2-friends.md`): sim-7, helper keys, cards, job routing by owner with Fair-mode shares, resume uploads, migrations 0009–0012, the join/invite/download routes, `airace-ml fetch-data`, and the screens. CLAUDE.md's R3 now reads: real work runs on the players' own helper cards (Fair mode pools them equally).
- **Verified:** TypeScript 743 tests, ML 1707, checks and ruff clean; `npm run e2e` (solo) and `npm run e2e:friends` (two players, two helpers, Fair off and on: join by link, training on each player's card, a shared AI's reply, a friend battle, chat both ways, an opacity check, a stopped helper, a server restart) pass; `npm run host` checked on `data-check` without the tunnel.
- **Self-review fix:** a player who joined by code but never opened the world counted as online forever (the clock kept running, autopilot never started); joining now marks them away until their page opens the world.
- **Not yet done:** a real session with the owner and one friend on two PCs, through `npm run host`'s public link (the tunnel and a friend's helper setup are first exercised there).
- **Next:** that session; then the coaching plan, on the owner's go.

## 2026-11-04 — The roadmap after Phase 4 planned (nothing built yet)
- **Owner interview C36–C48** (design `docs/superpowers/specs/2026-11-04-remaining-features-design.md`). Each player's AI is made on their own computer; **Fair mode** (chosen when a world is created) pools everyone's graphics cards so hardware doesn't decide; with it off, nobody shares. Friends can try each other's AIs, battle them (story battles voted by the other players) and chat. Absent players go on autopilot. The game stays on the owner's PC; guest names only until publishing (personal data has legal risks). Then coaching (thumbs and better answers), departments (Research, Data, Sales & marketing, Support & operations), licensing (with players and rivals), an AI Hub with leaderboards, and the orb as a skippable guide. Music is the owner's.
- **R3 amended:** in-game compute still sets game time and cost, but real work runs on the players' own cards (pooled equally in Fair mode).
- **Plans** (`docs/superpowers/plans/2026-11-04-*.md`): stage2-friends (20 tasks, replacing the M2 plan's Tasks 25–28), coaching, departments, licensing, ai-hub, tutorial.
- **Next:** Stage 2 on the owner's go.

## 2026-11-04 — Game layer Phase 4: "Surprises and goals" (complete)
- **What a player can now do:** interview their AI in front of the press (24 prompt cards, half with a right answer, or their own questions), tick up to five real answers and publish them; the outlet's story and posts quote the AI exactly, and the reaction moves followers (a bad answer with a big following goes viral). See discoveries the scoring finds in each AI ("Surprisingly good at facts about animals", "Much better at counting than v1", "More talkative than v1"), each with the real question and reply as proof, and publish them as news. Write a private notebook (up to 5,000 words) that joins training as a ninth share. Collect milestones on a trophy shelf, each earned by a measurement (first full sentence, first correct fact, first working program, 1,000 / 100,000 / 1,000,000 customers, first paying customer, first profitable month, #1 in a skill, beating the arch-rival, a data center, an A). Pick an optional goal when starting a world (customers, monthly profit, best AI, beat the arch-rival, company value; with a deadline of 1, 2 or 5 years or none), follow it on the Company screen, and celebrate it with a victory screen that tells the company's story. And a data-quality scandal: an AI trained on unchecked web text that gets caught stating false facts at scale makes the news.
- **Built** (`docs/superpowers/plans/2026-10-29-game-layer-phase4.md`, 15 tasks; rulings in `docs/handoff/sdd/ledger-game-layer-phase4.md`): evals report milestone signals, the tag breakdown and discoveries (compared with the parent version and the world's AIs, rivals included); an `interview` worker job; catalog v3 with the interview cards (the dev launcher rewrites an older catalog); sim-6 (sim-5 worlds upgrade; inputs logged before the upgrade replay without the scandal); the interview and notebook services; goals at world creation; migration 0008; the Interview mode, discovery, notebook, trophy-shelf, goal and victory screens.
- **Verified:** TypeScript 662 tests, ML 1690, checks clean; the end-to-end game (now with a goal, the notebook, a discovery, an interview, the trophy shelf and the victory) passed twice; played on the GPU (a real AI's discoveries, milestones from its scoring, an interview, the victory screen).
- **Also fixed** (found in the GPU playtest): the launch dialog couldn't scroll, so on a shorter screen its Launch button was out of reach; the victory screen now draws customers and cash on separate charts.
- **Also fixed** (found by the end-to-end test): a tiny model's reply holding the byte 0 lost its whole customer-feedback batch (the database can't store it; it now reads as the unknown-character mark), and the test's click on the Games pill.

- **Next:** M2 Stage 2 (friends: Tasks 25–28 of `docs/superpowers/plans/2026-10-22-m2-two-friends.md`) on the owner's go.

## 2026-11-03 — Playtest pack: fair battles, Playground games, things to buy (complete)
- **Why:** the owner's requests after playing Phase 3 (interview C31–C35, `docs/superpowers/specs/2026-11-03-playtest-pack-design.md`).
- **Fair battles:** battles asked every question as a chat, at a random-ish setting, with a long reply, and scored pass/fail, so two wrong answers drew 0-0. Now each AI is asked the way it learned that kind of question, a near miss earns part of the point ("Close"), the right answer shows under the replies, and draws need almost equal totals.
- **Why the Playground felt weaker than the posts:** the GPU sharing only made answers slower. Posts show each batch's best and worst replies, customers ask the kinds of question the AI learned from, and long Playground conversations confuse a small AI. Customer posts (and app reviews) now wait while the player is in the Playground, and a long chat suggests starting fresh.
- **Playground games** (every move the real AI's own: its most likely option, or its own typed words): Puzzle Maze (reasoning), Pattern Bridge (patterns, drawn as discs, animals and tiles), Quiz Show (knowledge, a prize ladder), Story Dice (stories judged for applause plus a word rule), Simon Says (instructions); Easy/Medium/Hard plus a ramp within each run; duels against a rival's AI; showcases for the press ($20,000: a great run wins fans, a bad one gets roasted); personal bests.
- **Things to buy:** launch keynotes with a live demo question (the news quotes the real answer), sponsorships (stadium, tech conference, arts festival: awareness in a region, map markers, news), looks (orb styles and logo motions), partner apps (grow with the AI's measured skill; reviews quote real replies), Learn from customers (train on conversations customers loved), tutoring packs (extra practice in one dataset), Fact-check plus (the strictest data cleaning), the investment fund; and office-and-support bills that grow with revenue (12% of a small company's, 21% of a giant's).
- **Also fixed:** the save-folder lock now carries a heartbeat, so a dead server's process number reused by another program can't block the game.
- **Built** (`docs/superpowers/plans/2026-11-03-playtest-pack.md`, 16 tasks; rulings in `docs/handoff/sdd/ledger-playtest-pack.md`): sim-5 (sim-4 worlds upgrade, with inputs logged before the upgrade replayed on the old money rules), catalog v2 (the dev launcher rewrites an older one), worker jobs minigame/keynote/appReview, migrations 0006–0007.
- **Next:** Phase 4 "Surprises and goals" (now sim-6) on the owner's go.

## 2026-10-30 — Game layer Phase 3: "Rivals" (complete)
- **What a player can now do:** meet an arch-rival (a tech giant, a cheeky upstart or a mysterious lab) and three rivals in every world; watch them launch real AIs over the game year, price, advertise and build data centers by their style, and trash-talk in the news (labelled game commentary); try any rival AI on sale in the Playground; battle a rival AI in knowledge, coding, instructions, patterns or creativity, with both answers streaming side by side, for ranking points; follow league tables (customers, revenue as places only, overall, each skill, ranking points); see the regions' leaders on the map and rank changes on report cards.
- **Built** (`docs/superpowers/plans/2026-10-29-game-layer-phase3.md`, 16 tasks): `airace-ml build-rivals` (8 lineages, 3 versions each, resumable), the worker's battle job, sim-4 (rivals, league, battles, trash-talk; sim-3 worlds upgrade, so the Phase 2 world carries over), the rival and battle services, rival chats, the Rivals and Arena screens. Every ruling: `docs/handoff/sdd/ledger-game-layer-phase3.md`.
- **Verified:** TypeScript 507 tests, ML 1632, checks clean; the end-to-end game (now with rivals, Try it and a battle, on tiny rivals) passed twice.
- **Real rival AIs:** built on the RTX 3050 in about an hour (24 versions; the story specialist leads creativity, the premium rival grows from 50 to 60 overall). Played on the GPU: rivals joined a new world, launched within weeks, a knowledge battle streamed (a 0-0 draw: early AIs miss trivia), and a rival's AI wrote a real short story in the Playground.
- **Next:** Phase 4 "Surprises and goals" on the owner's go.

## 2026-10-29 — Phase 3 and Phase 4 planned (nothing built yet)
- **Owner request:** plan Phases 3 and 4 without starting them, with instructions for a later session.
- **Plans:** `docs/superpowers/plans/2026-10-29-game-layer-phase3.md` ("Rivals", 16 tasks: the rival pack built once on the GPU, rivals as companies without an owner, releases and business habits, league tables, battles with ranking points, trash-talk, the rival and battle services, the Rivals and Arena screens) and `docs/superpowers/plans/2026-10-29-game-layer-phase4.md` ("Surprises and goals", 15 tasks: milestone signals and discoveries from evals, interview prompt cards and scoring, the notebook as a ninth share, milestones and the trophy shelf, interviews, the goal and victory screen, the data-quality scandal deferred from Phase 2).
- **Decisions recorded in the plans** (each plan ends with its list for the owner report). The main ones: rivals are ordinary companies without an owner; one logged `rivalsJoined` input carries the pack's measurements and releases follow on game days; rivals also join existing worlds once the pack exists, and sim-3 and sim-4 worlds upgrade, so the owner's Phase 2 world carries into Phases 3 and 4; battles cost $5,000 and use the five skills with written replies; the goal is chosen when a world is created; interviews are published through a server service so only real answers can be published.
- **Next:** on the owner's go, execute Phase 3 (resume guide `docs/handoff/README.md` §4).

## 2026-10-29 — Save-folder lock after a damaged database

- The game database in `data/server/db` was damaged: a second server (started for a check while the owner's game ran) opened it before failing on the busy port, so two processes wrote to it. PGlite then aborted on every start.
- The damaged copy is kept at `data/server/db-damaged-2026-10-29`, and a full backup of `data/server` at `data/server-backup-2026-10-29`. The game starts with a fresh database (the old worlds were Phase 1 worlds, which sim-3 can't open anyway).
- The server now claims `data/server/server.lock` before opening the database and refuses while another running server holds it (`apps/server/src/dataLock.ts`, 4b056ec). A lock left by a stopped server is taken over.

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
