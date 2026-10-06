# AI Race — System Architecture & Design Spec

- **Date:** 2026-10-06
- **Status:** Approved direction; Milestone 1 plan follows in `docs/superpowers/plans/`
- **Design authority:** `AI_Race_Game_Design_Document_Updated.pdf` (GDD) and `Fun_Features_to_Add.pdf` (FFD). This spec does not restate them; it records the engineering design that realizes them. Section references like *GDD §7* point into those documents.

---

## 0. Decisions log

### Creative decisions (from the product owner)
| ID | Decision | Date |
|----|----------|------|
| C1 | **Shared session clock.** World time runs while the group plays. The world's creator (host) sets speed and can pause. The world freezes when every member has left and resumes next session. Absent or disconnected members run on **autopilot**. | 2026-10-06 |
| C2 | **Notebook dataset, small.** Each company may write a size-capped, abuse-filtered dataset of its own (facts, example Q&As, style samples) that joins the training mix like any other dataset. | 2026-10-06 |

### Pending creative decision points (do not block the architecture)
| ID | Question | When to ask | Temporary assumption |
|----|----------|-------------|----------------------|
| D1 | Visual direction (art style, palette, typography, motion) | Start of Milestone 2, with 2–3 rendered mockups | Dark "neon research lab" UI: deep navy/graphite, one vivid accent taken from the company's own palette, rounded geometric type, small playful motion |
| D2 | Name and flavor of the in-game dataset sources (fictional brand names vs. generic descriptions) | Milestone 2 content pass | Generic descriptive names ("Storybooks", "Web Crawl", …) |

### Technical decisions
These are owned by engineering and recorded with rationale in §3. Later changes go in `docs/decisions/NNNN-*.md` ADRs.

---

## 1. Requirements derived from the design documents

These are the non-negotiable engineering properties. Each one maps to tests (§13).

- **R1 — Real model.** Every company AI is a real decoder-only transformer trained by gradient descent on the actual text mixture the player assembled. Nothing in the path from model to player output is scripted, templated or rewritten. The only exception is the abuse filter on public surfaces (§11), which can only *block* output, never author it. *(GDD §2, §3, §34.2)*
- **R2 — Opacity.** Weights, training data, data mix, preparation and design choices, training logs, notebook and coaching data never leave the server and never appear in another player's view. Other players learn about an AI only by interacting with it and through its public product facts (§6.4). *(FFD core rule)*
- **R3 — Fair compute.** How fast and how much a model trains depends on the company's **in-game** compute and money, never on the player's own hardware. *(GDD §19)*
- **R4 — Server authority.** All game state is owned by the server. Clients send intents; the server validates, applies and broadcasts.
- **R5 — Emergence over enumeration.** No unlock lists, capability tiers, personality menus or level caps. Capability scores, personality traits and "discoveries" are all **measurements** of the model's actual outputs (§4.7–4.9). Limits come from model capacity, data, compute, money and time. *(GDD §2, §10, §11, §33; FFD 4, 5)*
- **R6 — Continuous lineage.** Each training run starts from the previous checkpoint. Increasing model size uses function-preserving growth, so the *same* AI keeps developing (§4.3). *(GDD §29, §33)*
- **R7 — Plain language.** No player-facing control requires ML vocabulary. Every control maps to a real training or serving parameter (§4.5). *(GDD §5, §9, §31)*
- **R8 — Friends-first multiplayer.** Invite-link joining, persistent worlds, reconnect without loss, and visible social feedback. All of it is designed in from the start, not added later.

---

## 2. System overview

```
 Browser client (React)              Game server (Node, authoritative)                ML workers (Python/PyTorch)
 ┌───────────────────────┐  WSS   ┌──────────────────────────────────────┐  WSS out  ┌───────────────────────────┐
 │ UI · net layer        │◄──────►│ Gateway: auth, sessions, rate limits │◄──────────│ Worker agent              │
 │ synced world view     │  HTTPS │ World actors (1 per active world):   │  (worker  │  train · eval · generate  │
 │ streams (chat, train) │◄──────►│   deterministic sim reducer          │  dials    │  grow · quantize          │
 └───────────────────────┘        │   input log + snapshots              │  server)  │ LRU model cache (GPU/CPU) │
                                  │   per-viewer view projection + diff  │           └─────────────┬─────────────┘
                                  │ Job broker (queue, leases, fairness) │                         │ HTTPS (signed URLs)
                                  │ Model registry · social services     │                         ▼
                                  └───────────────┬──────────────────────┘           ┌───────────────────────────┐
                                                  │                                  │ Blob store                │
                                                  ▼                                  │ checkpoints, corpora      │
                                  ┌──────────────────────────────────────┐           │ shards, tokenizer, logos  │
                                  │ PostgreSQL (source of truth)         │           │ local FS (dev) / S3 (prod)│
                                  └──────────────────────────────────────┘           └───────────────────────────┘
```

**Responsibilities**
- **Client:** presentation, input, local form state, and rendering of the synced view and streams. It holds no authority and never sees model internals.
- **Game server:** identity, worlds, simulation, commands, privacy projection, job scheduling, the model registry, social systems and persistence.
- **ML workers:** stateless compute. Workers **dial out** to the server, which lets a GPU behind a home router serve a cloud-hosted game. They pull jobs, read and write blobs through signed URLs, and stream telemetry back. Workers have no database credentials.
- **Postgres:** accounts, worlds, input logs, snapshots, model registry, jobs, social records and leaderboards.
- **Blob store:** large immutable artifacts.

---

## 3. Technology decisions

| Area | Choice | Why | Rejected |
|------|--------|-----|----------|
| Client | **React 19 + TypeScript + Vite**, Zustand (state), Tailwind CSS + Radix primitives (accessible UI), Motion (animation), d3-geo + us-atlas (map), uPlot (fast live charts) | The game is UI-dense (lab, dashboards, chat, map, charts). DOM UI is the right tool, and the browser gives the fastest possible friend joining (open a link). | Unity/Godot: install friction, weak for dense UI, no link-join. Phaser/Pixi: canvas engines solve problems this game doesn't have. |
| Server | **Node 24 + TypeScript**, Fastify 5 (HTTP), `ws` via @fastify/websocket (realtime), zod 4 (validation), pino (logs) | Shares types and the sim package with the client. Its event-loop concurrency suits the world-actor model. | Colyseus: schema-class state fights per-viewer private projection and we'd rebuild persistence anyway. Nakama: heavy ops, Go/Lua split. Python server: loses shared types. |
| Simulation | **Pure deterministic TS package** (`@airace/sim`), seeded PRNG, no wall-clock or Math.random | Enables replay, crash recovery by input-log replay (§9), property tests, and client-side cost previews. | Ad-hoc mutable server logic. |
| ML | **Python 3.13 (uv-managed) + PyTorch 2.14** (CUDA when available, CPU fallback), Hugging Face `tokenizers`, `safetensors`, numpy | PyTorch is the robust standard for training. Tiny models are fast on a consumer GPU. | In-browser WebGPU training breaks R2 (weights on the client) and R3 (hardware unfairness), and needs the tab open. TF.js: weaker training stack. |
| Database | **PostgreSQL 18**. Dev uses `embedded-postgres` (real Postgres binaries via npm, zero install); tests use PGlite (in-process Postgres); prod uses managed Postgres. **Drizzle ORM** + drizzle-kit migrations | One SQL dialect everywhere. JSONB for snapshots. `SKIP LOCKED`-friendly job tables. | SQLite: single-writer and awkward for multi-instance. Mongo: weaker transactional guarantees for the economy. |
| Blobs | `BlobStore` interface: `LocalFsBlobStore` (dev/self-host), `S3BlobStore` (any S3-compatible, e.g. Cloudflare R2) | Checkpoints are 5–300 MB immutable files. | Storing weights in Postgres. |
| Repo | npm workspaces (TS), uv project (Python), Biome (lint+format TS), ruff (Python), Vitest, pytest, Playwright (E2E) | Zero extra installs on this machine (npm present), fast tooling. | pnpm/turbo: not needed yet. |
| Protocol | JSON over WebSocket, zod-validated, versioned envelope; per-viewer JSON diffs | View sizes are small (KBs) and ticks are slow. JSON is debuggable. msgpack is a later drop-in if needed. | Binary schemas now (premature). |
| Hosting | **Phase A "Host from your PC":** everything on the owner's PC, exposed with a Cloudflare quick tunnel; friends open the URL. **Phase B cloud:** server + Postgres on a small VM or Fly.io, blobs on R2, GPU workers anywhere (home PC and/or rented GPU) dialing in. | Friends can play online in Milestone 2 at near-zero cost, and the same architecture scales by adding workers. | GPU cloud for everything from day one (expensive while unproven). |

---

## 4. The miniature language model

This is the heart of the game. Everything here lives in `ml/` (Python).

### 4.1 Tokenizer
- Byte-level BPE with a **4096-token vocabulary**, trained once on a balanced sample of all corpora. It is versioned (`tok-v1`) and **frozen**: every model, benchmark and NPC shares it, so models can grow and be compared.
- Special tokens: `<|pad|> <|bos|> <|end|> <|user|> <|ai|> <|sep|>`. The chat format is `<|bos|><|user|>…<|end|><|ai|>…<|end|>`.
- A model never trained on conversation data has never seen `<|ai|>`. In the playground it simply continues the text instead of chatting. This is a genuine, intended discovery.

### 4.2 Architecture
- Decoder-only transformer: pre-norm RMSNorm, rotary position embeddings (RoPE), SwiGLU MLP (hidden ≈ 8/3·d rounded to 64), tied input/output embeddings, no biases, head_dim = 32.
- Parameterized by `(n_layer, d_model, ctx_len)`. Reference sizes (vocab 4096):

| Example shape | ≈ Params | Role |
|---|---|---|
| 4 L × 128 d | 1.3 M | Typical first model |
| 6 L × 192 d | 3.4 M | Early growth |
| 8 L × 256 d | 7.3 M | Mid game |
| 10 L × 384 d | 19 M | Large |
| 12 L × 512 d | 40 M | Very large |
| 16 L × 640 d | 81 M | Current **system ceiling** (real hardware limit, config value `ml.maxParams`; raised when hardware allows) |

The ceiling is a real system limit (allowed by GDD §2), not a game level. In practice the cost of compute, data and serving stops players long before it.

### 4.3 Growth operators (continuous lineage, R6)
Growth is **exactly function-preserving**: logits before and after match within 1e-4 (tested).
- **Depth:** insert new blocks whose attention output projection and MLP down-projection are zero-initialized, so each new block is an identity on the residual stream.
- **Width d→d′:** zero-pad embeddings and the residual stream. Rescale old RMSNorm gains by √(d/d′) to cancel the larger RMS denominator. New input columns are randomly initialized (they read zeros at first), and all weights *writing* new residual dimensions are zero-initialized. New attention heads get zero output projections; new MLP hidden units get zero down-projection columns.
- **Context length:** RoPE needs no new parameters. Longer contexts take effect when trained on longer sequences.
- After growth the optimizer state resets and a short warmup runs. Growth is applied by the worker at the start of a training run whose requested shape exceeds the parent's.
- **Size only grows within a lineage.** A shape smaller than the parent in any dimension is rejected; going smaller means starting a new lineage from scratch, which is always allowed, so the sandbox stays open. Context length may go up or down freely, since it is not a parameter shape.

### 4.4 Training engine
- AdamW, warmup-stable-decay (WSD) learning-rate schedule per run, bf16 autocast on CUDA, fp32 on CPU, gradient clipping. The data loader reads pre-tokenized `uint16` shards through memory maps.
- **Mixture sampler:** each sequence is drawn from a dataset by mixture weight, then from that dataset's *eligible pool*. The pool is the document set after the player's preparation filters, which are cheap boolean masks over precomputed document tags (§4.10).
- **Instability is real.** Aggressive learning settings raise the peak LR and loosen clipping. Loss spikes are detected (EMA z-score); the run rolls back to the last good checkpoint, records an "instability" event, and continues at a reduced LR or stops if it repeats.
- **Checkpoints:** saved periodically during the run for resumability (a worker crash resumes elsewhere) and at the end as the new **model version**.
- **Live telemetry every k steps:** training loss; held-out loss per dataset (drives the "what it's learning" chart); and samples from 3 fixed probe prompts plus 1 player-chosen prompt. The player watches the AI go from gibberish to words to sentences.

### 4.5 Player decisions → real parameters (R7)
| Player-facing control (plain language) | Real effect |
|---|---|
| **Data mix**: shares of each dataset (§4.10) + Notebook + Coaching | Mixture sampling weights |
| **Buy more of a dataset** | Larger unique pool, so less repetition per token budget (real memorization/overfitting dynamics) |
| **Cleaning**: Light / Standard / Thorough | Quality-score threshold on documents. Removes noise, spam and garbled text, but shrinks the pool |
| **Remove duplicates** | Collapses near-duplicate clusters |
| **Fact-checking** (costs money) | Excludes documents tagged as carrying injected false facts |
| **Variety**: Natural / Balanced topics | Topic-stratified sampling within a dataset |
| **Brain size** | Target `(n_layer, d_model)`; growth from the current shape |
| **Attention span** | `ctx_len` (memory in conversations; higher cost per token) |
| **Training length** | Token budget for the run |
| **Learning style**: Careful ↔ Bold | Peak LR, clip norm, warmup (stability vs. speed/risk) |
| **Review what it knows** | Replay share of the previous run's mixture (counteracts real catastrophic forgetting) |
| **Finishing focus** | Mixture used during the decay phase (annealing emphasis) |
| **Answer style**: Precise ↔ Inventive (release setting) | Sampling temperature / top-p at serving |
| **Compact serving** (release setting) | int8 weight quantization: lower serving cost and latency, measured quality change |

No control maps to a capability number. Each changes what or how the real model learns or runs.

### 4.6 Inference
- KV-cached autoregressive generation, stopping at `<|end|>` or a token cap. The chat context is truncated to the model's `ctx_len`, so limited memory is a real, felt property.
- The worker batches concurrent requests per model and keeps an LRU cache of loaded models (GPU, spilling to CPU).
- Streams tokens back through the server to the client.

### 4.7 Benchmarks (suite `bench-v1`, versioned)
- Items are **procedurally generated from held-out seeds and templates**. Training corpora are produced from disjoint seeds, so no item text is ever trained on.
- Scores are normalized to 0–100 where 0 = chance. Every category stores per-item results tagged by **topic × skill × format**; those tags feed discoveries.

| Category | Method |
|---|---|
| Language | Grammatical vs. ungrammatical minimal pairs; log-prob comparison |
| Reasoning | Multiple choice by log-prob over generated comparisons, transitive orderings, syllogisms, simple word problems |
| Pattern recognition | Sequence continuation (numbers, letters, words); exact match on greedy generation |
| Knowledge | Fact questions from the knowledge base (§4.10), multiple choice by log-prob + short-answer exact match |
| Coding | **MiniPy** (a safe Python subset): predict program output, and complete functions verified by running hidden tests in our own sandboxed AST interpreter (never `exec`) |
| Creativity | Stories from prompts, scored as coherence (perplexity under the fixed **reference judge model**) × novelty (1 − overlap with a sampled 8-gram index of all corpora) × diversity (distinct-n across samples) |
| Consistency | Agreement across paraphrases and resamples of knowledge/reasoning items |
| Instruction following | Templated instructions ("one word", "list three", "start with…", "yes or no") checked programmatically |

- Full suite runs automatically on every new version: under 30 s on GPU for a first model. A tiny *mini-eval* runs during training for live charts.
- Leaderboards key on `(bench version, category)`.

### 4.8 Personality fingerprint (FFD 5, GDD §11)
- A fixed probe set of about 60 open prompts, each sampled several times. Measured traits:
  - verbosity (length)
  - confidence (mean top-1 probability)
  - inventiveness (diversity)
  - steadiness (self-agreement)
  - precision (factual exactness)
  - boldness (attempt rate vs. empty or degenerate replies)
  - slip rate
  - register (formal ↔ casual lexicon)
  - warmth (affect lexicon)
  - repetitiveness
- Traits display as words relative to the world/global population percentile ("unusually talkative", "careful with facts"). A trait shows only when it is notable, so some AIs are plainly neutral, as FFD 5 intends.
- Personality emerges because the corpora genuinely differ in voice (web casual, books narrative, educational formal, varied conversational personas) and because seeds, mixtures, preparation, sampling style and coaching shape what the model absorbs.

### 4.9 Discoveries (FFD 4)
- After each evaluation, the discovery engine compares fine-grained results (the topic × skill × format tags, a combinatorial space with no fixed list) against:
  1. the model's previous version
  2. the world/global population
  3. what its data mix predicts (a regression fitted over population results)
- Large residuals become **discovery cards**, each with real evidence: the actual prompts and outputs. Example: "Surprisingly good at counting", "Forgot how to rhyme since v4", "Mixes up animals and colors".
- Players' own playground experiments remain the primary discovery channel; the engine just highlights.

### 4.10 Content pipeline (`ml/src/airace_content`, build-time)
- **Datasets:** 8 categories (GDD §7): Web Crawl, Books, Educational, Conversations, Code, Reasoning, Reference Facts, Creative Writing. Plus the per-company **Notebook** (C2) and **Coaching** sets.
- **Sources:** only permissive or open licenses, with provenance recorded in a manifest. Candidates: TinyStories (CDLA-Sharing), FineWeb-Edu (ODC-By), Cosmopedia (Apache-2.0), SODA (CC-BY-4.0), SmolTalk everyday conversations (Apache-2.0), Gutenberg English (MIT), GSM8K (MIT), tiny-codes (MIT), Wikidata (CC0) for the knowledge base. Each license is re-verified when its dataset is added.
- **Procedural generators** (deterministic, seeded) for reasoning, patterns, arithmetic, MiniPy code with execution traces, and knowledge-base facts rendered as varied prose, Q&A and conversation. These guarantee learnable skill signals at tiny scale and share templates with the benchmarks while using disjoint seeds.
- **Simplicity filtering:** text is selected for short sentences and common vocabulary, so 1–40 M-parameter models can learn it.
- **Deliberate dirt, tagged.** Real noise is injected into lower-quality pools: typos, spam, boilerplate, garbled encoding, duplicates, and false facts made by swapping knowledge-base entities. Every document carries tags (`quality`, `dup_cluster`, `false_fact`, `topic`, `voice`). Preparation choices are therefore real filters with real consequences.
- **Outputs:** tokenizer, per-dataset token shards with document tag tables, held-out eval slices, the benchmark item generator, the 8-gram novelty index, the **reference judge model** (trained once on everything, frozen), and later the NPC model lineages (M3).

### 4.11 Performance targets (RTX 3050 8 GB reference; CPU = Ryzen 5 5500)
| Operation | Target |
|---|---|
| First-model training run (~1.3 M params, ~4 M tokens) | ≤ 90 s GPU; ≤ 8 min CPU fallback |
| Full benchmark suite, first model | ≤ 30 s GPU |
| Chat first token / throughput (≤ 20 M params) | ≤ 300 ms / ≥ 50 tok/s |
| Growth op | ≤ 2 s |

### 4.12 Feasibility gate (Milestone 1 exit, the project's biggest risk)
Measured by an automated experiment suite (3 seeds each) and recorded in a report:
1. **Speed:** the first-model targets above are met.
2. **Legibility:** after a balanced first run, ≥ 70 % of 20 probe replies are *well-formed*. Well-formed means:
   - ≥ 4 words
   - no word 3-gram repeated 3+ times
   - per-token perplexity under the reference judge at or below the 90th percentile the judge assigns to real held-out conversation replies

   A sample transcript is included in the report for human review.
3. **Differentiation:** four contrasting mixes (story-, code-, fact-, conversation-heavy) at equal compute each rank first in their matching benchmark category. The margin must exceed 2× the seed standard deviation.
4. **Seed variation:** the same config with different seeds yields measurably different personality fingerprints. The difference is smaller than the difference between mixes.
5. **Growth:** function preservation holds. After growth, continued training reaches lower loss than the un-grown model given the same extra tokens.
6. **Forgetting is real and manageable:** a code-only continuation of a story model drops creativity or language scores; 30 % replay recovers at least half of that drop.
7. **Preparation matters:** Thorough cleaning measurably lowers the garbled-output rate, and fact-checking measurably lowers the false-fact rate on knowledge probes.

If a criterion fails, the corpora, sizes or mapping are adjusted before any game UI depends on them.

---

## 5. Game simulation (`packages/sim`, deterministic, shared)

### 5.1 World and clock (C1)
- `1 tick = 1 game day`. At 1×, a tick is 8 real seconds (≈ 49 min per game year), with 2× and 4× options and pause. All of these are balance config.
- The **host** controls speed and pause. If the host is offline, the earliest-joined online member acts as host. When no member is online the world **freezes**: the actor is unloaded and its snapshot persisted. It resumes when someone returns.
- Absent and disconnected members are on **autopilot**, defined by a fixed rule set:
  - keep current price, release and infrastructure
  - finish queued construction
  - start no new training runs or purchases
  - if cash would go negative, cut discretionary spend first
  - an absent company cannot go bankrupt; it enters maintenance mode instead
- The game calendar starts 2027-01-01.

### 5.2 Reducer model
- `step(state, input) → state` for inputs of these kinds:
  - player commands
  - `tick`
  - system results: training completed or failed, eval results, satisfaction batch results, battle results
- All randomness comes from a seeded PRNG stored in state. The sim never reads wall-clock time.
- The same package powers server authority, crash recovery by replay (§9), headless balancing runs, and client-side "what will this cost" previews.

### 5.3 Company economy
- Cash, revenue (subscriptions + license royalties), and expenses: serving compute, data-center upkeep, cloud rental, training compute, data purchases, fact-checking, marketing.
- Starting capital is enough for a first model trained on rented cloud compute. Debt is allowed with interest and a restructuring option, which keeps the sandbox forgiving.
- Late joiners get a founder grant scaled to world age.

### 5.4 Compute and infrastructure (GDD §19)
- **Cloud rental:** instant and elastic, but expensive per unit.
- **Owned data centers:** chosen by region and size, with build time, capital cost and running cost; cheaper per unit.
- A company's total compute is split by a **training ↔ serving allocation**: compute spent improving the next model is not serving today's users. This is a core strategic tension.
- Serving demand ∝ users × model cost per token (it grows with parameters and shrinks with compact serving). Demand above serving capacity reduces reliability.
- Latency for a region depends on the distance to the company's nearest data center and on model size.

### 5.5 Market (GDD §15–18, §21)
- **Regions:** the 9 US census divisions, each with population, internet adoption growth, and a mix of segments.
- **Segments:** Students, Creators, Developers, Businesses, Everyday users, Researchers. Each has preference weights over benchmark categories, price sensitivity, latency and reliability sensitivity, and trait preferences (e.g. Businesses penalize slip rate; Creators reward inventiveness).
- **Choice:** a multinomial logit per (region, segment) over every company's released product plus a "no AI" outside option. Attributes are measured scores, price, latency, reliability, satisfaction, reputation and awareness.
- **Adoption dynamics:** actual users move toward the logit shares through Bass-style diffusion, driven by awareness (marketing, news) and word of mouth (∝ satisfied users), plus churn.
- **Price:** a monthly price per product plus an optional free tier (users and reputation without revenue).

### 5.6 Satisfaction sampling (ties the business to the real model)
- Every game week, for each company's released model, the server enqueues a small batch of real "customer queries" drawn from each segment's query pool (about 8 per segment). Workers run them on the model; checkers and the reference judge score the replies.
- The scores drive satisfaction. Real replies surface as **customer feedback quotes** on the dashboard ("Asked 'what is the capital of France?' — got 'the cat sat'").
- Results enter the sim as logged inputs (§9).

### 5.7 Reputation and news (GDD §17)
- Events are generated by the sim and the social systems: releases, benchmark records, battle results, published interviews, outages, discoveries, price moves.
- Each event becomes a world news item with a sentiment value. Reputation is an exponential moving average of event sentiment and satisfaction, weighted by audience reach.

### 5.8 NPC rival companies
- 0–5 per world, configurable at creation, so solo play and small groups have competition.
- Each NPC has a strategy profile (powerhouse, budget, specialist, consumer, experimental) and makes business decisions by heuristic.
- **Their models are real.** They are pre-trained lineages built by the content pipeline with scripted data and design strategies, and are released on schedules relative to world time.
- Players can chat with, battle and license NPC models like any other, under the same opacity rules.

### 5.9 Goals and victory (GDD §25)
- At world creation the host picks a mode:
  - **Sandbox:** endless, with achievements and milestones.
  - **Race:** first to a goal, or best by an end date. Goals include users, monthly revenue, overall benchmark, dominance of one category, reputation, or valuation.
- Achievements measure real outcomes such as "First AI to say a full sentence", "#1 in Coding", "1M users" and "Survived an outage".

### 5.10 Balancing
- All constants live in a versioned `balance` config.
- A headless harness runs bot companies through years of game time to tune pacing. Target: the first model takes 30–60 min of real play, and the first release happens around game months 4–8.

---

## 6. Multiplayer and networking

### 6.1 Topology and concurrency
- Each active world is a **single-writer actor** inside the server process. Inputs are applied sequentially, so there are no data races on game state.
- Cross-world records (accounts, friends, hub, global leaderboards) use ordinary database transactions.
- One process can host hundreds of worlds; the ML job system is the scaling bottleneck.
- A world-ownership **lease** (`worlds.owner_instance`, `lease_until`) is designed in now, so additional server instances behind a router need no data migration.

### 6.2 Session structure
- **Account → World → Membership → Company.**
- A world is private, holds up to 8 humans plus NPCs, and is reachable by invite link or code.
- New worlds start in **Setup**: players join and create companies (name, logo builder, colors), then the host starts the clock.
- Members may join a running world. "Quick solo" creates a private world with NPC rivals.
- A player can belong to many worlds, with a separate company and AI in each.

### 6.3 Protocol
- WebSocket at `/ws` after HTTP authentication. Envelope: `{ v, type, id?, seq? }`, with zod schemas in `@airace/protocol`.
- **Client → server:** `hello/resume`, `command {cid, ...}` (idempotent by client command id), `query` (request/response), `subscribe/unsubscribe` (streams).
- **Server → client:**
  - `snapshot {version, view}`
  - `patch {from, to, ops}`, a per-viewer structural diff broadcast at most 4 Hz
  - `ack/reject {cid}`
  - `event` (toasts, news)
  - `stream` (chat tokens, training telemetry, battle outputs)
- If the client sees a version gap, it requests a fresh snapshot.

### 6.4 View projection (privacy, R2)
- `projectView(world, viewerId)` is the only path from world state to the wire.
  - **Public:** company identity, released products (name, version number, release date, public benchmark scores, personality words), price, users and market share (rounded), reputation, data-center locations, news.
  - **Private (owner only):** cash, unreleased versions, every training run and its configuration, data mix and preparation, notebook, coaching data, internal evals.
- Tests assert that private fields never appear in any other viewer's projection (§13).

### 6.5 Disconnects and reconnects
- Clients hold a resume token. Reconnecting restores the session, sends a snapshot, and resubscribes streams.
- Command retries are safe because commands are idempotent.
- Chat replies and battle outputs are persisted as they generate, so a reconnecting client receives the completed text.
- Training is server-side and unaffected by disconnects. A disconnected member switches to autopilot after a 60 s grace period.

### 6.6 Latency
- A management sim tolerates 100–250 ms command round-trips.
- The client applies local form state immediately (slider positions, mix editing) and commits on confirm. Server-derived numbers update when the server acks.
- Chat and telemetry stream over the same socket.

---

## 7. ML job system

### 7.1 Worker protocol
- A worker connects to `/worker` with a worker token and advertises its capabilities: device, VRAM, maximum parameters, job types.
- The server **leases** jobs to it. The worker streams progress and telemetry, uploads artifacts to signed blob URLs, and reports results.
- A missed heartbeat expires the lease, and the job resumes on another worker from its last checkpoint.

### 7.2 Job types and priorities
Priority order:
1. `generate` (chat, battle, interview: interactive)
2. `eval` (benchmarks, personality, discovery features)
3. `satisfaction` (customer-query batches)
4. `train` (with growth and quantize as steps)

Training yields between step chunks, so a single GPU can interleave interactive generation. Fair sharing applies across worlds and companies.

### 7.3 Game-time coupling
- A run's **in-game duration** = token budget ÷ the company's training compute (in-game units). Its **cost** = FLOPs × the company's compute price. Both come from the sim.
- The real worker normally finishes earlier, by design of the capacity budget. Telemetry is then revealed to clients at game pace, so the live view stays in sync with the world clock.
- If real execution lags, completion waits for the real result and the UI shows "finalizing". Results are held until their in-game completion time, which keeps fairness inside the world.
- If the world freezes, running jobs complete in the background and their results apply at the correct in-game time after resume.

### 7.4 Model registry
- `lineages` → `model_versions`. Each version records its parent, training run, shape, checkpoint blob key, eval results and fingerprint.
- Players can branch (keep an experiment) and roll back.
- Retention keeps released, starred and latest-N versions. Others may be pruned, keeping their metadata.

---

## 8. Social and ecosystem features (FFD)

| Feature | Design |
|---|---|
| **Try other players' AI** (FFD 1) | Every released product is public within its world. An **AI Directory** lists them with a "Try it" chat. Rate-limited, transcripts private to the asker, owner sees only counts ("tried 23 times this week"). Unreleased models: owner only. |
| **Battles** (FFD 2) | Head-to-head on fresh items in a chosen category, with outputs streamed side-by-side live. Objective categories are auto-scored like benchmarks. Subjective ones (writing, creativity, conversation) are **blind-voted** by non-participant members, with the reference judge as fallback when no voters are present. Per-category Elo per world. Results feed news and reputation. |
| **Trading and licensing** (FFD 3) | License contracts specify model version (pinned or latest), allowed segments, royalty per user or revenue share, duration, and an optional exclusivity fee. A licensee can route segments to the licensed model inside its own product; inference still runs on the licensor's model, so opacity holds. Outright sale of weights is excluded because it would break opacity and ownership. |
| **Discoveries** (FFD 4) | §4.9; discoveries can also become news items if the owner publishes them. |
| **Emergent personality** (FFD 5) | §4.8; shown on the AI's public profile. |
| **Interviews** (FFD 6) | Interview your own or any public AI with free questions plus optional prompt cards, then optionally **publish** to the world feed. Members react; publicity and reputation follow from reactions and from measured answer quality. |
| **News feed** | The shared social surface: releases, records, battles, interviews, discoveries, outages. |
| **Notebook** (C2) | Up to ~5,000 words, private training data, abuse-filtered on save. Because it is tiny, a large mix share causes real memorization (parroting), which is a genuine trade-off. |
| **Coaching** | 👍/✏️ on playground replies builds a private fine-tuning set (capped). It is how "continued interaction" shapes personality (FFD 5). |
| **AI Hub and global leaderboards** (M6) | Opt-in publishing of a released AI beyond its world, global per-category leaderboards on standard benchmarks, and player profiles ("known for the AI they built"). |

**Moderation:** wordlist and pattern filters on Notebook, Coaching edits and names; an output filter on public surfaces (directory chat, battles, interviews, feed); and a report button.

---

## 9. Persistence

- **Event-sourced world state.** Every accepted world input (command, tick, system result) is appended to `world_inputs(world_id, seq, kind, payload, at)` **before** the command is acked.
- A `world_snapshots(world_id, seq, sim_version, state jsonb)` row is written every 15 s while running, on pause or freeze, and on graceful shutdown.
- **Recovery** = latest snapshot + replay of later inputs through the deterministic reducer. External results (job results, satisfaction scores) are inputs, so replay never re-runs ML.
- **Job durability:** `jobs` and `training_runs` rows are the source of truth for compute work. On recovery, the broker reconciles runs finished in the database but not yet applied to the world by re-emitting the idempotent `training_completed{runId}`.
- **Other tables (overview):** `users`, `auth_identities` (guest token now; Discord/Google OAuth later, so the schema supports multiple identities per user), `sessions`, `worlds`, `memberships`, `companies` (identity mirror for cross-world queries), `lineages`, `model_versions`, `eval_results`, `jobs`, `chat_threads/messages`, `battles`, `interviews`, `licenses`, `news_items`, `leaderboard_entries`, `reports`.
- Migrations are managed by drizzle-kit. A backup policy is defined in M6.

---

## 10. Client architecture

- **Screens:**
  - Home (my worlds, create/join)
  - World lobby and company creation (name, logo builder, palette)
  - **Dashboard** (GDD §20)
  - **AI Lab:** Data → Prepare → Design → Train (live) → Versions
  - **Playground**
  - **Benchmarks**
  - **Market map** (US, GDD §21)
  - Infrastructure
  - Pricing and release
  - **AI Directory**, Battles, Interviews, News
- **State:** one Zustand store holds the synced world view (snapshot + patches) plus per-feature UI state. A net layer handles the socket, resume, the command queue and acks, and stream subscriptions.
- **UX principles:**
  - Every control shows a plain-language "what this does" and its real cost.
  - Dataset cards show real sample documents from the pool.
  - Preparation shows real examples of what would be removed.
  - The UI never fabricates predictions of model quality. Previews show costs and data facts only; quality is discovered by training and testing.
- **Accessibility:** keyboard navigable, sufficient contrast, reduced-motion support.
- Visual direction is pending as D1.

---

## 11. Security model

- **Server-authoritative:** commands are validated with zod plus domain rules, and authorized (you may act only on your own company).
- **Opacity:** by projection (§6.4); no endpoint ever serves weights or private configuration to anyone but the owner, and never serves weights even to the owner.
- **Workers:** authenticated by token. Blob access uses short-lived signed URLs scoped per object. Workers have no database access.
- **Code execution:** model-written code runs only in the MiniPy AST interpreter, with step, memory and time limits and no I/O or imports.
- **Rate limits:** per user and per model on generation, per world on commands, with connection limits at the gateway.
- **Moderation:** as in §8. Inputs from model output and from players are treated as untrusted text everywhere they are rendered (no HTML injection).
- **Auth:** opaque random session tokens, stored hashed; httpOnly cookies plus a WebSocket handshake token.
- **Economy exploits:** prevented by the reducer's validation and invariants (no negative quantities, money conservation in transfers) checked by property tests.

---

## 12. Observability

- **Structured JSON logs** (pino; Python `logging` with JSON formatting) carrying `worldId`, `userId`, `cid`, `jobId` and `runId`, correlated end-to-end.
- **Metrics** (Prometheus text endpoint): tick duration, inputs/s, connected clients, patch bytes, job queue depth and wait time by type, generation latency, tokens/s, worker utilization.
- **Developer admin page:** worlds, workers, jobs, and replaying any world to any input sequence number (event sourcing makes this free).
- **Training runs** keep their full telemetry for debugging model behavior.

---

## 13. Testing strategy

- **Sim (Vitest + fast-check):**
  - reducer unit tests
  - invariant property tests (money conservation, no negative users or cash without debt, capacity bounds)
  - determinism: same inputs give the same state hash
  - replay equivalence: snapshot + replay = live state
- **Privacy:** property tests generating random worlds and asserting that no private field reaches non-owner projections.
- **Protocol and server integration:** spin up the server with PGlite and headless clients; test join, commands, acks, patches, version-gap resync, reconnect with resume, idempotent retries, and the autopilot transition.
- **ML (pytest):**
  - tokenizer round-trip and special tokens
  - forward shapes
  - growth function-preservation
  - a short run lowers loss
  - mixture sampler proportions
  - preparation filters
  - checkpoint resume equivalence
  - MiniPy interpreter correctness and sandbox limits
  - benchmark scorers verified on synthetic "oracle" and "random" models
- **Design-validation experiments:** the §4.12 suite, rerun whenever corpora or the ML core change.
- **Bot playtests:** headless bots play complete games through the protocol (balancing and soak tests).
- **E2E (Playwright):** core flows; two browser contexts verify friend interactions.

---

## 14. Repository layout

```
/
├─ apps/
│  ├─ client/                 React web client
│  └─ server/                 authoritative game server, job broker, registry, social
├─ packages/
│  ├─ protocol/               zod schemas, messages, view types
│  └─ sim/                    deterministic reducer, economy, market, balance config
├─ ml/                        uv project
│  ├─ src/airace_ml/          tokenizer, model, growth, train, infer, evals, personality,
│  │                          discovery, minipy, worker agent, CLI
│  ├─ src/airace_content/     dataset build pipeline, generators, knowledge base, noise injection
│  └─ tests/
├─ data/                      build outputs (gitignored): shards, tokenizer, indexes, ref models
├─ tools/                     dev orchestration, bots, tunnel helper
├─ docs/
│  ├─ superpowers/specs/ plans/
│  ├─ decisions/              ADRs
│  └─ progress.md             running progress log for cross-session continuity
├─ CLAUDE.md                  concise project instructions
└─ package.json               npm workspaces
```

---

## 15. Milestones (dependency order)

Each milestone gets its own implementation plan when it starts.

| # | Milestone | Contents | Exit criteria |
|---|-----------|----------|---------------|
| **M1** | **"It learns"** (ML core + content v0) | Python project; tokenizer; model; growth; trainer; inference; content pipeline v0 (8 datasets + knowledge base + tags + noise); benchmarks v1; personality probe; MiniPy; reference judge; experiment CLI and report | §4.12 feasibility gate passes; report committed |
| **M2** | **"Two friends, two AIs"** (vertical slice) | D1 visual direction; TS monorepo; protocol; sim core (clock, companies, training-run economics); server (auth guest, worlds, lobby, actors, input log, snapshots, projection, patches, reconnect); job broker + worker agent; client (lobby, company creation, AI Lab, live training, playground, benchmarks, try a friend's AI); host-from-PC tunnel | Two people on different networks create companies, each train a first model with live view, chat with their own and each other's AI (opacity verified), survive disconnect/reconnect and a server restart |
| **M3** | **"Open for business"** | Release flow, pricing, market sim, satisfaction sampling, reputation and news, infrastructure and compute allocation, US map, dashboard, NPC rivals with real lineages, autopilot, goals/victory, balancing harness | A 3-player + 2-NPC world plays 3 game years with meaningful divergence; bots show viable distinct strategies |
| **M4** | **"Living ecosystem"** | Battles (live, voting, Elo), interviews and publishing, discoveries, licensing contracts, Notebook, Coaching, moderation | Each FFD feature is playable among friends, with opacity tests passing |
| **M5** | **"Feels great"** | Onboarding for the first 30–60 min, polish, motion and sound pass, achievements, balance tuning from playtests, performance pass | First-time player reaches a first model without help in 30–60 min in playtests |
| **M6** | **"Online for real"** | Cloud deployment, Discord/Google sign-in with guest upgrade, AI Hub + global leaderboards, backups, monitoring, CI/CD | Public deployment with monitoring; a returning player keeps identity across devices |

---

## 16. Major risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tiny models are too incoherent to be charming, or choices don't differentiate | Breaks the core fantasy | M1 first, with a quantitative gate (§4.12); simplified-language corpora; procedural skill data; adjust before UI work |
| GPU capacity with many concurrent players | Slow training, laggy chat | Priority scheduling and preemption, batching, a CPU-fallback pool, in-game compute pricing that bounds real usage, dial-in workers to add capacity cheaply |
| Real compute lags game time | Desync feelings | Capacity budget per world speed, "finalizing" state, results held until their game time |
| Opacity leaks via a careless view or endpoint | Violates the core rule | Single projection path + property tests + code review checklist |
| Event-log replay divergence after code changes | Bad recovery | Snapshots record `sim_version`; replay only on the same version; frequent snapshots |
| Toxic content via Notebook, Coaching or chat | Harm in public spaces | Input and output filters, size caps, reporting, private-by-default training data |
| Dataset license problems | Legal | Permissive-only policy, provenance manifest, re-verification at ingestion |
| Scope size | Never finishing | Strict milestone order; each milestone ships something playable; YAGNI on hub and scale-out until M6 |
