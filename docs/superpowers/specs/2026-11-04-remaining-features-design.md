# AI Race — the remaining features: Stage 2 (friends, own hardware, Fair mode) and the roadmap after it

**Date:** 2026-11-04. **Status:** design approved by the owner in conversation (interview C36–C48, three design parts each approved); this document is the written record.
**Builds on:** the architecture spec `2026-10-06-ai-race-architecture-design.md`, the game-layer design `2026-10-27-game-layer-design.md` (single player, Phases 1–4, all complete), the immersion design and the playtest pack. Everything in those stays true unless this document says otherwise.
**Sub-projects, in build order (C48):** 1. Stage 2 "Friends" · 2. Coaching · 3. Departments · 4. Licensing · 5. AI Hub and leaderboards · 6. Tutorial. Each has its own plan in `docs/superpowers/plans/2026-11-04-*.md`; each is designed, built, tested and played on its own.

---

## 0. Owner decisions (interview 2026-11-04)

| # | Topic | Decision |
|---|-------|----------|
| C36 | Hardware | **Each player's AI is made on that player's own computer.** A world has **Fair mode** on or off, **chosen when the world is created** and fixed for its life. |
| C37 | Fair-mode privacy | In Fair mode **everything can run on any player's card** (training, the notebook, coaching and customer conversations included), **with a clear notice** when creating or joining such a world. |
| C38 | Friends' computers | Unknown: plan **Windows + NVIDIA first**, a **processor-only fallback** for any other PC, **Mac support later**. |
| C39 | Together (Stage 2) | Friends can **try each other's AIs**, **battle each other's AIs** (other players **vote** on story battles), and use a **world chat**. Interviewing a friend's AI: not now. |
| C40 | Absent players | **Autopilot**: prices stay, building finishes, nothing new starts, no bankruptcy; with Fair mode off its AI can't answer while its PC is off (its posts pause). |
| C41 | Where the game lives | **On the owner's PC** for now (free; friends join through the link while it runs). A rented server and **publishing** come later. **Guest names only** for now; sign-in waits for publishing because collecting personal data has legal risks. |
| C42 | Beyond a world | **AI Hub + global leaderboards.** |
| C43 | Staff | **Departments with budgets**: Research, Data team, Sales & marketing, Support & operations. No individual people. |
| C44 | Licensing | Between **players and rivals**. |
| C45 | Coaching | **Thumbs up and "write a better answer"** in the Playground. |
| C46 | Tutorial | **The orb as a guide**, and the tutorial can always be **skipped or turned off**. |
| C47 | Sound and music | The owner will lead it; **out of these plans**. |
| C48 | Order | Stage 2 → coaching → departments → licensing → AI Hub and leaderboards → tutorial. |

Owner clarification on C36: with Fair mode off, **every player uses their own graphics card** (not the host's): the host's card works only for the host's own company; the host's PC only stores the world and the AIs' files.

## 1. Changes to the non-negotiables

- **R3 (fair compute) is amended (C36).** In-game compute still sets every run's game-time duration and cost (unchanged). But real work now runs on the players' own computers:
  - **Fair mode on:** all members' cards form one pool and every company gets an equal share of it, so a player's hardware does not decide anything. R3 holds in spirit.
  - **Fair mode off:** a player's own card limits how big a brain they can train (its memory) and how soon a run's real result arrives (its speed). The world clock does not wait for a slow card; a late result shows "finishing on your computer", exactly as a late worker does today. Hardware is part of the game in this mode, by the owner's choice.
  - "Training and inference run server-side" (CLAUDE.md R3) becomes: **they run on the players' helper programs, which the server directs**. Clients still never run models in the browser.
- **R2 (opacity) is unchanged between players' screens** (`projectView` stays the only path from state to a client). Two physical facts are disclosed instead of hidden:
  - Every AI's files are **stored on the host's PC** (the blob store), as today: that is where the world lives.
  - In **Fair mode**, another player's card may hold an AI, its plan, its notebook and its lessons while it works on them (C37: the notice).
  - With Fair mode off, when a friend tries your AI or battles it, **your** card answers, so your PC sees their questions (never their other chats). The Try-it panel says so.
- R1, R4–R8 unchanged.

## 2. Sub-project 1 — Stage 2 "Friends"

Replaces Tasks 25–28 of `2026-10-22-m2-two-friends.md` (none were built). Plan: `2026-11-04-stage2-friends.md`.

### 2.1 Cards and the helper program
- **The helper** is the existing ML worker (`airace-ml worker`), delivered to friends as a one-click setup:
  - "Connect your computer" in the game (Home and the world's top bar) offers **"Get the helper"**: a small personalized Windows setup file (`AI Race Helper setup.cmd`) served by the host. It installs `uv` if missing, downloads the helper's code (a zip of `ml/` served by the host), lets `uv` install Python and the AI libraries (about 3 GB the first time), downloads the game data from the host (corpus, tokenizer, judge, novelty index and rival pack, about 1.1 GB, checked file by file), registers an `airace-helper://` link handler, and creates a Start-menu shortcut. About 15 minutes the first time.
  - Afterwards **"Connect your computer"** opens `airace-helper://connect?url=<today's link>&key=<key>`; the browser asks to open the helper, which connects to that link. (The free Cloudflare quick-tunnel link changes every session, so the helper never stores it.)
  - The helper refuses to work with data whose identity differs from the server's catalog (as today's workers do) and offers to re-download.
- **Personal keys.** Each account has one **helper key** (server table `helper_keys`: user, key hash, created). The worker's `whello.token` is that key; the gateway binds the connection to the user. The server's own token (`data/server/worker-token`, used by `npm run dev` / `npm run host` on the host's PC) stays and marks the **house card**: it counts as the card of **each world's host** (the world's creator), so single-player worlds behave exactly as today.
- **What a card tells the game** (`whello.capabilities` gains): `gpuName` (string, "CPU" for none), `vramMb` (exists), `maxParams` (exists; VRAM-derived), and `speed`: training tokens per second on a fixed reference model, measured by a 5-second test at connect (cached per device). The game shows a card as "RTX 3050 · 8 GB · 1.0× speed" (speed relative to 30,000 tokens/s, the RTX 3050 reference).
- **Card view.** Cards are not world state (they come and go in real time): a `cards` query and a `cards:<worldId>` stream give every member each player's card(s): name, memory, speed, online. Cards of other players are public within the world (they are part of the game); keys never leave their owner.
- **No card.** A player can join, found a company and look around without one; training, chats with their own AI and launches that need answers say "Connect your computer to do this." With only a processor, the helper works (slowly) and the Lab says so.

### 2.2 Who does each job
- Every job gets an **owner** (the user whose card must do it, with Fair mode off; whose share it counts against, with it on) and a **need** (`minParams`: the biggest model it loads).

| Job | Owner |
|-----|-------|
| train, eval, feedback, keynote, appReview, minigame, showcase, interview | the company's owner |
| generate (chat) with your own AI | you |
| generate with a friend's shared AI (Try it) | **that AI's maker** |
| generate with a rival's AI | **the asker** |
| battle vs a rival | the challenger (both sides) |
| battle vs a friend | split in two **side jobs**: each side's owner |

- **Fair mode off:** a job may only be leased to a card of its owner (the house card counts as the world host's). A friend's AI never runs on your card and yours never on theirs.
- **Fair mode on:** a job may be leased to any card of any member of its world. The broker serves **the company that has used the least card time** in that world first (decayed over the last game month), then priority, then age, and gives each job **the fastest free card that can fit it**. Jobs from rivals' work count against the player who asked.
- **Brain-size limit.** The AI Lab shows the biggest brain the run can use: the owner's card (Fair off) or the pool's biggest card (Fair on), and an estimate in real minutes ("about 12 minutes on your card"). The server refuses a `startTraining` over that limit ("Your graphics card can train up to the Small brain.") before the input is logged; a card that disconnects later just makes the run wait.
- **Runs that move.** In Fair mode a training run saves its progress (the trainer's resume state) to the host's blob store every 2 minutes; if its card leaves, the next card continues from there. With Fair mode off the run waits for its owner's card and resumes locally, as today.
- **Stale customer questions.** Feedback and app-review jobs that wait longer than 14 game days (their owner's card was off) are dropped: those customers' questions went unanswered, so there are no posts and satisfaction stays as it was. No backlog floods in when the player returns.
- **Interactive jobs** (chats, battles, games) keep their priorities. In Fair mode the asker's share pays for them.

### 2.3 Fair mode in the world
- `POST /api/worlds {name, goal?, fairMode}`; the "Start a new world" form gets a **Fair mode** choice with the notice: *"Everyone's graphics cards are shared. Your AI, its reading list and your notebook may be trained on a friend's computer for a while. Play with people you trust."* Default: off. Single-player worlds: either (alone it makes no difference).
- `WorldState.fairMode: boolean` (sim-7; upgraded worlds: false), shown in `world.fairMode`; the join screen repeats the notice for Fair worlds.

### 2.4 Joining and running a shared world
- **Joining:** `POST /api/worlds/join {code}` (idempotent; the 9th human gets "This world is full (8 players)."), the invite link `/join/<code>` (a guest without a name is asked for one first), a lobby with members, their online dots and their cards, and "Invite friends" (code and copy-link).
- **Late joiners** found a company in a running world with a **founder grant**: founding cash + $2,000 × world day, at most twice the founding cash.
- **Rivals:** the four rival companies stay in every world (they make the market); friends are extra companies.
- **Speed and pause:** the host's; when the host's browser is closed, the earliest-joined online member's (`effectiveHost`, exists). When nobody is online the world rests (exists).
- **Autopilot** (C40), for a company whose owner is offline: everything keeps running by the rules as today (prices stay, building finishes, no commands are made). Added: when its cash would go negative at a day's end, its **ad campaigns pause** (discretionary spend); its rescue offer waits until its owner is back (the 14-day automatic loan does not run while they're away). It can't go bankrupt (there is no bankruptcy in the game).
- **Hosting:** `npm run host` builds the client, serves it from the game server, starts the house card and, when `cloudflared` is installed, opens a free quick tunnel and prints the link; `publicUrl` becomes that link (secure cookies, signed blob and helper URLs). Without `cloudflared` it prints the LAN link and how to install it (`winget install Cloudflare.cloudflared`). The script never downloads anything itself. A plain-language `docs/hosting.md` for the owner.

### 2.5 Trying each other's AIs (the AI Directory)
- **"Share with friends"** on a version (the sim's existing `publishVersion` command); a new **AI Directory** screen lists every company's shared AIs and the rivals' AIs on sale, with logo, name, brain size, scores, personality words and "Try it".
- Try-it chats are **private to the asker**; the maker sees only counts ("tried 23 times this week", query `versionTries`). Limits: 10 messages per 10 minutes per asker per AI. Answers run on the maker's card (Fair off) or the pool (Fair on); offline: "Its maker's computer is offline, so it can't answer right now."
- Stopping sharing ends Try-it at the next message.

### 2.6 Battles between friends
- `startBattle` takes **any other company's live product** (rival or player). Same skills, fee, cooldown (per opponent) and ranking points (Elo per skill, players and rivals together).
- Each side answers on its own owner's card (Fair off) or the pool (Fair on), as two **side jobs**; the server joins them into the battle. The defender's PC must be on (Fair off): otherwise "Its maker's computer is offline, so it can't battle right now."
- **Votes on story battles:** players other than the two in the battle see the two stories side by side, unlabelled, and vote; voting closes after **5 real minutes** or when every other online player has voted. Each round goes to the story with more votes; a tie or no voters falls back to the judge's score (as today). Votes are private; only totals show.
- Both companies get the news; the defender sees the battle in their Arena.

### 2.7 World chat
- A collapsible **chat panel** on every world screen for the players themselves (not AIs): name, colour of their company, time.
- Messages: 1–300 characters; the notebook's filter (no links, email addresses, phone numbers, AI control text, blocked words) refuses in plain words; at most 5 messages per 10 seconds per player; the last 200 messages per world are kept.
- Rivals don't post here (their trash talk stays in the news).

### 2.8 Done means
- An automated end-to-end test with two players, two browser profiles and two helper programs on the CPU, in **both modes**: joining by link, both training, try-it across companies, a friend battle with a vote, world chat, a disconnect and a server restart, and the opacity checks (no other player's cash, plan, notebook or blob URL ever reaches a client).
- A real session: the owner and one friend, each on their own PC, through the link.

## 3. Sub-project 2 — Coaching

Plan: `2026-11-04-coaching.md`.
- In the Playground (Chat mode, your own AIs), under each AI reply: 👍 **"Good answer"** and ✏️ **"Write a better answer"** (1–500 characters).
- Each becomes a **lesson**: `{question, answer, kind: "liked" | "written"}`, where `question` is the user message just before the reply, and `answer` is the AI's reply (liked) or the player's text (written). Lessons pass the notebook filter; at most 200 per company (the newest replace the oldest); stored on the server (R2), counted in the world (`company.lessons`).
- The AI Lab's Data step gets a **"Your coaching"** card: the lesson count and a switch "Train on your lessons", enabled from 5 lessons. Free. When on, the run's config carries the lessons as coaching pairs at **5%** of the reading (the trainer's existing `coaching` share), added to Learn from customers' 5% when both are on.
- A written answer is the player's own training text: it is never shown as the AI's words anywhere (R1).
- Fair mode: lessons travel with the run (covered by the notice).

## 4. Sub-project 3 — Departments

Plan: `2026-11-04-departments.md`.
- A **Team** panel on the Company screen with four monthly budgets, $0 to $500,000 a month each in $10,000 steps (`setDepartment {department, budget}`). Spending is booked daily as a new cost line **Team**.
- **Strength** of a department: `s = 1 − exp(−staffed / 200,000)`, where `staffed` moves toward the budget by 1/30 of the gap each game day (teams take about a month to grow or shrink). Each extra dollar does less than the one before.
- **Effects** (first guesses, tuned with the balance harness so none is a must-have or a waste):

| Department | Effect at strength s |
|---|---|
| Research | Each run reads `1 + 0.3·s` times the tokens for the same money (the preview and the trainer's token budget both use it). Real: the AI really reads more. |
| Data team | Fact-checking, Fact-check plus and the data checks cost `1 − 0.5·s` of their price. |
| Sales & marketing | Every day, in every region and group, awareness rises by `0.002·s × (1 − awareness)`. |
| Support & operations | A batch's satisfaction drop is `1 − 0.3·s` as large (rises are unchanged); serving compute costs `1 − 0.2·s`. |

- Rivals fund departments by style (premium: research; budget: support; specialist: research and data; consumer: sales), scaled to their revenue.
- Autopilot: when cash would go negative, department budgets pause with the ad campaigns.
- Upgraded worlds: every department at $0.

## 5. Sub-project 4 — Licensing

Plan: `2026-11-04-licensing.md`.
- **Offer:** from an AI's window, "Offer for license": `price` per customer per month ($0.10–$20), `months` (3, 6 or 12), `groups` (one or more of the six), `exclusive` (only one licensee). Offers are public in the world (the AI Directory and the Rivals screen show them).
- **Take:** a company takes an offer for **one of its products** (`takeLicense {offerId, productId}`). Exclusive offers close when taken. Between players, a player can also **propose** terms to another player's AI; the maker accepts or declines.
- **Effect:** for the licensed groups, the licensee's product uses the licensed AI's **measured scores** in the market, and its customer questions in those groups are answered by **the licensed AI** (feedback jobs owned by the licensor: the licensor's card, or the pool). Posts show "powered by {AI}" and quote the licensed AI's real replies. Its traits apply to those groups.
- **Money:** each month the licensee pays `price × its customers in those groups` (cost line **Licenses**) to the licensor (income line **License income**).
- **Ends:** at the end of the term; when the licensor retires the AI (customers in those groups fall back to the licensee's own AI); or early by the licensee (paying the remaining term's minimum, one month's royalties).
- **Rivals:** offer their AIs to the player at style prices (premium dear, budget cheap); and ask to license the player's AI when it beats theirs by 10 points in their focus skill, offering a price the player accepts or declines (a toast and a window).
- **Opacity:** the licensee never sees how the AI was made; only its public facts and its answers to their customers.
- News: "Basalt Analytics licenses Nova Labs' Pengu for its students."

## 6. Sub-project 5 — AI Hub and global leaderboards

Plan: `2026-11-04-ai-hub.md`.
- **Hub** (server-wide, across every world on the host's PC): "Publish to the Hub" on an AI version (owner only; it must be scored). A hub entry keeps a snapshot of public facts only: AI label and product name, company name, world name, scores, overall, personality words, brain size, published date. Unpublish removes it.
- **AI Hub screen** (from Home): entries with filters by skill, and "Try it" (a private chat with that AI, answered by its maker's card; offline: the same message as the Directory).
- **Leaderboards:** published AIs and **each world's best AI** (its highest overall, recorded at each eval), ranked overall and per skill on bench-v1. Names are guest names and company names only (C41). Rival AIs appear once, marked "Rival", as reference points.
- Later, when the game moves online, the same Hub and boards serve every player.

## 7. Sub-project 6 — Tutorial

Plan: `2026-11-04-tutorial.md`.
- **The guide:** the orb with a speech bubble **labelled "Guide"**, in a style distinct from chat bubbles, with game-written lines: the AI's real words only ever appear in chat and posts (R1).
- **Steps, advanced by what the player actually does:** found the company → pick the first AI's reading (the data mix explained) → start training and watch it learn → meet your AI (a first chat) → see its scores → launch it → first customers and money → done (a short closing line). Each step highlights the control to use.
- **Control:** "Skip this step", "Skip the tutorial", and a **"Show the guide"** setting (Home → settings) to turn it off or start it again. Progress is remembered in the browser (local storage, wrapped in try/catch), per player name.
- New players see it in their first world; anyone can restart it.

## 8. Out of scope for these plans
Music and sound (C47, the owner leads), sign-in and publishing (C41), a rented server (C41), the helper on Mac (C38, after Windows), interviewing a friend's AI (C39), team hiring as individual people (C43).

## 9. Temporary assumptions (can change without the owner)
- Fair-mode share: least card time over the last 30 game days.
- Stale customer questions: 14 game days.
- Story-battle votes: 5 real minutes.
- World chat: 300 characters, 5 per 10 s, last 200 kept.
- Founder grant: $2,000 a game day, at most 2× founding cash.
- Department curve and effect sizes (§4), licensing price range (§5).
- Card speed reference: 30,000 tokens/s = 1.0×.
