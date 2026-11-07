# AI Race — the AI quality upgrade: honest scores, a measured diagnosis, a better training backbone, and Fair mode

**Date:** 2026-11-07. **Status:** design agreed with the owner in conversation (interview C49–C56 below); this document is the written record, awaiting the owner's review.
**Builds on:** the architecture spec (`2026-10-06-ai-race-architecture-design.md`, §4 the ML system), the feasibility-gate decisions (`docs/decisions/0001`–`0008`), and the remaining-features design (`2026-11-04-remaining-features-design.md`, Stage 2 and Fair mode). Everything there stays true unless this document says otherwise.
**Kind of work:** an upgrade of the AI development system, not a rewrite. The model design, the tokenizer, the chat format, lineage and growth, the job system and the game's rules stay; what the AIs read, how they learn, how their answers are drawn and how they are measured improve, each change kept only if it measurably helps.

---

## 0. Owner decisions (interview 2026-11-07)

| # | Topic | Decision |
|---|-------|----------|
| C49 | Training time | The longest training option may take **up to about 30 real minutes** on a card like the owner's (RTX 3050, 8 GB). |
| C50 | What a good AI does | All four, in this order of the test's weight: **chat that makes sense**, **correct simple facts**, **following simple instructions**, **stories and simple code**. |
| C51 | Scores | **Scores measure the AI's real answers**, so an AI that talks nonsense scores low. |
| C52 | The friend's PC | An **NVIDIA graphics card** (it trained about 3–4× slower than the owner's per unit of work). |
| C53 | The owner's GPU | Experiments may use it **any time** (the PC is left on; keep-awake held, Monitor heartbeats on long runs). |
| C54 | Typos and junk in web text | **Keep the mechanic, gentler:** an AI trained with normal cleaning spells properly; skipping cleaning still visibly hurts. |
| C55 | Learning style slider | Claude decides. The owner read "Careful" as *quality* and "Bold" as *fast*; the slider is retuned so that is true: Careful = the steady, reliable choice that is never a handicap; Bold = learns faster on short runs, with a real risk of going wrong. |
| C56 | AIs made before the upgrade | **Keep their old scores.** So that old and new numbers never meet in one world, **each world keeps the test it started with**: existing worlds stay on today's test (bench-v1) and economy (balance-v2); new worlds use the new test (bench-v2) and economy (balance-v3). |

## 1. What we know (survey, 2026-11-07)

### 1.1 The system today
- **Model:** a modern decoder (RMSNorm, rotary positions, SwiGLU, tied embeddings, PyTorch fused attention, KV cache), heads of 32. Already state of the art for its size: not the bottleneck.
- **Tokenizer tok-v1:** byte-level BPE, 4,096 tokens (the right size for models this small), 16 special tokens. Chat = `<|bos|> <|user|> … <|end|> <|ai|> … <|end|>`, the same in training data and in the game.
- **Brains (balance-v2):** tiny 4×128 (≈1.3M parameters), small 6×192 (≈3.5M), medium 8×256 (≈7.5M). Attention spans 128/256/512. Run lengths 6.3M / 12.6M / 25.2M tokens (plus Research and data-center extras).
- **Trainer:** AdamW, warmup-stable-decay schedule with a finishing mix in the decay, **bf16 autocast on every CUDA card**, 8,192-token batches (ADR 0005). Learning style: peak rate 6e-4 (Careful) to 1.2e-2 (Bold).
- **Reading (corpus v1, ≈139M tokens):** web 32.7M (FineWeb with injected typos, spam, garble, duplicates and false facts), books 14.8M, educational 20.4M, conversations 19.8M, code 10.2M, reasoning 9.7M, facts 12.0M, creative 19.9M.
- **Answering:** chat draws at temperature 0.8, top-p 0.95, up to 96 new tokens; no repetition control.
- **Measuring (bench-v1):** eight categories (language, reasoning, pattern, knowledge, coding, creativity, consistency, instruction), mostly by **which option the AI finds likeliest** (pairs, multiple choice, paraphrase groups); a few greedy replies are checked (knowledge, instruction, coding); creativity is written stories judged by the reference AI ("the judge", 8×384, 120M tokens) and a novelty index.

### 1.2 What happened in the owner's Fair-mode session (a copy of the saved game, world "Prva online stretavka", day 1005)
- **Owner, Zula AI:** medium brain, **Careful** on every run, the default equal mix. v1 (12.6M tokens) overall 49 → v2 (51M) 62 → **v3 (99M) 64**: language 78, pattern 95, consistency 83, instruction 77, knowledge 57, reasoning 54, coding 46, creativity 24. Its chats:
  - "hello" → "whae yd, Tanyton. You're a sore thhe amazing pyeache."
  - "what is your name?" → "It's about the tsough i don't have a chance to siet a trip or a hunt."
  - "is 3 bigger than 2?" → "Jimiya, can I give you something?"
  - "Hello, how are you?" → "I'm looking for some coins to use."
- Real minutes on the owner's card: 12.6M tokens 3.9 min, 38.8M 11.2 min, 47.9M 15.8 min (≈50–54k tokens a second for the medium brain).
- **The owner's v4 run failed twice** ("the resume state … belongs to a different run config" on the retry); the world still shows it training: it will hang "finishing on your computer".
- **Friend, "my dih":** v1 small brain, code-heavy mix (code 49%), 12.6M tokens → overall 25 in **6.8 min** (the owner's medium brain, twice the work, took 3.9). v2 medium, span 512, 18M tokens (31M in all) → overall 35 in 23 min. "whtas capital of france" → "wordstankara, language spoken in humming live in Nicaranian."
- Elsewhere in that world: 260 customer batches answered, 42 dropped as stale, 10 battles, 11 minigames.

### 1.3 Hypotheses (Phase 2 confirms or refutes each with numbers)
1. **The score measures the wrong thing:** likelihood tests reward a model that talks nonsense when it speaks.
2. **Too little focus for the size:** a 7.5M-parameter AI spreads itself over eight kinds of reading; conversation is 12% of the default mix.
3. **The injected typos teach misspellings** ("thhe", "tsough", "siet") at standard cleaning.
4. **Answers are drawn too hot** for models this small (0.8 / 0.95, no repetition control).
5. **Careful under-trains:** 6e-4 is far below the best rate at these sizes.
6. **Tokens per real minute can rise** (fused optimizer, larger batches on long runs, data loading).
7. **Older NVIDIA cards are slow because of bf16:** cards before the RTX 30 series emulate it; fp16 with loss scaling (or TF32/fp32) is their fast path.
8. **Fair mode shares card *time*, not work:** a slow card's minute counts like a fast card's, and any idle card may take any job.
9. **A retried or moved run can fail on a stale resume state**, and a failed training job doesn't always reach the world.

## 2. Success criteria

Targets for the reference card (RTX 3050). Phase 1 measures today's baseline; the targets below are confirmed or adjusted against it, with the owner told the numbers, before Phase 3.
- **Sensible answers:** share of answers to the everyday test prompts that the calibrated grader (§3.1) accepts:
  - a first AI in about 4 minutes: at least **35%**;
  - the best AI a player can make in one 30-minute run: at least **60%**;
  - each at least **three times** today's equivalent AI.
- **Spelling:** misspelled or invented words below **2%** of the words in answers from an AI trained with normal cleaning.
- **The four areas (C50)** each improve over the baseline, none gets worse.
- **Honest scores:** across the test AIs, the new overall score ranks AIs the same way the blind ratings do (rank correlation ≥ 0.8).
- **Fair mode:** with one card three times slower than the other, two companies with the same plan (a) get equal work over a game month (±10%), (b) wait about as long in real time (medians within ±25%), (c) get AIs of the same quality whichever card trained them; and no run is ever lost when retried or moved.
- **Nothing breaks:** every test suite, both browser tests, every existing world loads and plays on.
- **Honesty clause:** if a target proves out of reach at these sizes, the report says so with the best measured numbers and why. It will still be a tiny AI: the aim is sensible simple conversation, not a large chatbot.

## 3. The design

### 3.1 Phase 1 — an honest answer test (bench-v2)
- **Same eight skills, honest measurement.** bench-v2 keeps bench-v1's category names (so the market, battles, apps, league tables, goals and discoveries keep working) but measures each through **the AI's real answers**, drawn exactly as the game draws them (fixed seeds, several samples a prompt):
  - **language** — chat that makes sense: readable, spelled right, on topic, to greetings and small talk.
  - **knowledge** — the answer contains the right fact.
  - **instruction** — the requested form *and* content (yes/no that is right, a list of the asked items, one word that answers).
  - **consistency** — the same right answer to paraphrases of one question, in words.
  - **reasoning, pattern** — the answer to a simple puzzle, written out.
  - **coding** — tiny code that runs and passes its check (the existing mini-Python checker).
  - **creativity** — short stories on request (today's judge-and-novelty measure, on the new judge).
- **Graders are measurements (R5)** and run on the players' helpers like today's evals. Readability comes from the reference AI's surprise at the answer; spelling from a known-word check; *on topic* from how much more likely the reference AI finds the answer after its own question than after an unrelated one.
- **Calibrated against people before it is trusted.** Helper agents (development only, never in the game) rate a few hundred anonymised answers from AIs of very different quality as "makes sense" or not, blind to which AI wrote them. The automatic grader must agree with them on at least 85% of answers and rank AIs the same way. If today's reference AI can't grade well enough, a stronger reference AI is trained on the new reading (overnight); if even that fails, an openly licensed small model (Apache-2.0, e.g. SmolLM2-135M) is used as the referee, recorded in the content manifest.
- **The baseline** is measured on: the gate's models, today's starter recipe, and copies of the owner's session AIs (Zula v1–v3 and the friend's v1–v2, from copies of their checkpoint files; the owner's saved game is never opened in place).
- **Per world (C56):** a world records its test version. Existing worlds keep bench-v1 for every AI they make; new worlds use bench-v2. The UI names the test where it matters ("Scores from the new answer test"). There is one reference AI for everyone, so an existing world's *creativity* scores may shift slightly once it is rebuilt; its other bench-v1 numbers are unchanged.

### 3.2 Phase 2 — the diagnosis
Controlled experiments on the owner's card, each changing one thing against the baseline at the same real time, two seeds or more, every AI measured with bench-v2 and a blind-rated sample:
- **D1 Drawing answers** (no training): temperature, top-k, top-p, repetition control, answer length, on existing AIs.
- **D2 Learning rate:** the slider's range swept for tiny and medium brains at short, long and 30-minute budgets; where quality peaks and where runs go wrong.
- **D3 Reading mix:** today's equal default against conversation-heavy and per-area mixes; what each dataset adds to each area.
- **D4 Typos:** noise share × cleaning level → misspellings in answers.
- **D5 Size against time:** tiny, small, medium and one or two larger sizes at 5, 15 and 30 minutes, single runs and lineages (a v1 grown into a v3).
- **D6 Speed:** tokens a second per brain on the RTX 3050 with today's code and with each speed change; the fp16 path checked for correctness here and on an older card when one is available.
- **D7 Batch and schedule** at the longer budgets (ADR 0005 was decided at short ones).
- **D8 New reading candidates** (from §3.3) against today's conversations dataset.
- **Output:** a diagnosis report ranking every lever by measured gain per minute of training, with before/after answers in plain language for the owner.

### 3.3 Phase 3 — the upgrades (only the winners of Phase 2)
- **U1 Answering:** new drawing defaults for chat, customer questions, battles, interviews and the rest (R1: the AI's real output, drawn differently; nothing is rewritten).
- **U2 Learning:** the slider retuned around the measured best rate (C55); the plain-language hints rewritten to match ("Careful: steady and reliable", "Bold: learns faster, can go wrong"); batch and schedule from D7.
- **U3 Speed:** the speed changes that win D6; **the precision path chosen per card** (bf16 on RTX 30-series and newer, fp16 with loss scaling or TF32 on older NVIDIA cards, fp32 on processors), applied to training and answering. Faster cards mean more reading per real minute for everyone.
- **U4 Reading (corpus v2):**
  - More and simpler conversations, short question-and-answer about simple facts in chat form, simple instructions with their answers, stories on request and tiny code tasks in chat form.
  - Sources: openly licensed datasets first (candidates include `HuggingFaceTB/smol-smoltalk` Apache-2.0, OpenAssistant Apache-2.0, TinyStoriesInstruct CDLA-Sharing), and, if D8 shows it pays, **practice conversations written offline by a bigger openly licensed AI** (e.g. SmolLM2-Instruct, Apache-2.0) in simple language, as Cosmopedia already is. Every source licence goes in the content manifest (CLAUDE.md).
  - Gentler typos and junk (C54), tuned by D4.
  - tok-v1 stays, so every existing AI can keep training on the new reading (R6).
- **U5 Training length and sizes (balance-v3, new worlds):** run lengths and prices so that the longest option takes at most about 30 minutes on an RTX 3050 (C49); a larger brain only if D5 shows it pays within that time or over a lineage, and offered only on cards that can hold it (as today). The economy is rebalanced to bench-v2's score distribution and checked with the balance harness.
- **U6 Defaults and hints:** the default plan leans towards what D3 shows makes a sensible AI; each dataset's hint says, in plain words, what it measurably teaches. These are defaults and advice, not unlocks (R5).
- **What applies where:** U1–U4 reach every world (existing AIs improve from their next run); U5 and bench-v2 apply to new worlds (C56).

### 3.4 Phase 4 — rebuild and prove
- Rebuild in order: corpus v2 → judge (and the novelty index) → catalog → rival pack v2 (new worlds' rivals, trained with the new recipe and measured with bench-v2). **Rival pack v1 stays on disk** for existing worlds' rivals. The helper download updates (friends' helpers fetch the new data once).
- All suites (TypeScript, ML, ruff, check), `npm run e2e`, the balance harness, and the gate's still-relevant criteria.
- A real play on the GPU in a new world: a first AI in about 4 minutes, a best AI in about 30, blind-rated against the baseline; a before/after page of answers for the owner.

### 3.5 Phase 5 — Fair mode
- **F1 Work, not time:** a company's Fair-mode share counts work done (tokens × model size), not card-seconds.
- **F2 The right card:** a waiting training job goes to the free card that would finish it soonest; a slow card takes it only when that is still the quickest way.
- **F3 Older cards:** the precision fix from U3, and the helper reports the card's real speed so the Lab's minute estimates stay true.
- **F4 Nothing lost:** a retried or moved run resumes or restarts cleanly (the "different run config" failure fixed and tested); a training job that finally fails always reaches the world as a failed run with its refund; the owner's stuck v4 run is recovered (with the owner's go, while their game isn't running).
- **F5 Check it:** a two-card test (one helper deliberately slowed) for F1–F4, and the next real session with a friend.

### 3.6 How the operation runs
- **Plans:** the operation is planned in two parts. **Part A** (Phases 1–2: the answer test and the diagnosis) is written in full now. **Part B** (Phases 3–5) is written now with its tasks, tests and acceptance bars; the values the diagnosis decides (rates, mixes, drawing settings, lengths, sizes) are filled in from the diagnosis report before Part B starts, and any upgrade that lost in Phase 2 is dropped with a recorded ruling.
- **Order:** Phase 1 → 2 → 3 → 4 → 5; the owner gets a short plain-language report at the end of each phase (with real answers before and after), and a go or a change is asked only when a result changes the plan.
- **Helpers:** subagents are allowed for this operation (the owner, 2026-11-07): source and licence research, blind raters, experiment analysis, and a fresh reviewer per phase. GPU work runs one experiment at a time on the owner's card, queued and logged, with a Monitor heartbeat.
- **Records:** a ledger per phase in `docs/handoff/sdd/`; experiment results under `ml/runs/upgrade/` (not in git) with their summaries and decisions in `docs/decisions/` (ADRs) and the diagnosis report in `docs/handoff/`.
- **Estimated GPU time:** roughly 2–4 days of the owner's card in all, mostly unattended (Phase 2 ≈ 20 h; reading build and any practice-conversation writing ≈ 10–20 h; reference AI ≈ 5–8 h; rivals ≈ 3–5 h; verification ≈ 5 h).

## 4. Non-negotiables
- R1–R8 unchanged. R1: answers are still the AI's own output; new drawing settings and graders don't rewrite them. R5: scores, traits and discoveries remain measurements; defaults and hints are not unlocks. R6: lineage continues; tok-v1 and the model design stay, so growth and continuation keep working.
- The game's deterministic simulation is untouched except for new balance and test versions recorded per world.

## 5. Temporary assumptions (can change without the owner)
- Practice conversations written by an openly licensed AI are acceptable reading (precedent: Cosmopedia), if D8 shows they help.
- The friends' one-time helper download may grow by up to about 1.5 GB.
- A larger brain size, if added, is offered only where a card (or, in Fair mode, a pooled card) can hold it.
- Answer drawing settings are the same for every world (they don't change scores already recorded).

## 6. Out of scope
A new tokenizer or model family, cloud training, the Mac helper, the AI Hub, the tutorial, music, and anything about sign-in or publishing.
