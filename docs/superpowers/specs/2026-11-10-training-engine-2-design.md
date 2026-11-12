# AI Race — Training engine 2: faster, more efficient, smarter AIs

**Date:** 2026-11-10. **Status:** the owner asked for this operation and its own plan on 2026-11-10 ("analyze, then upgrade the AI training system … faster, more efficient training … better, smarter and larger models"; "plan out your own … do not use the previous plan as your guide"; "remove all bugs and glitches you can identify"; "most importantly smarter models") and said "continue" to the plan in §3. Technical decisions are Claude's (CLAUDE.md); measured values are recorded in `docs/decisions/0009-training-engine-2.md`.
**Relation to earlier documents:** the architecture spec (`2026-10-06-…`) stays true unless this says otherwise. This operation does not follow the 2026-11-07 "AI quality upgrade" design; it keeps only the owner's own decisions from that interview (C49 the longest training may take about 30 real minutes on an RTX 3050; C54 the typo mechanic stays but gentler; C55 Careful must not be a handicap; C56 old AIs keep their scores and old worlds their rules).
**Evidence:** `docs/handoff/2026-11-10-training-upgrade-findings.md` and `docs/handoff/upgrade-2026-11-10/` (scripts and every run's result).

---

## 1. What the owner gets (plain language)
- **Smarter AIs from the same reading.** The same first AI, with the same brain and the same reading, answers 86–88% of simple fact questions right (today 67%) and answers "hello" with "Hello! How can I help you today?" (today: gibberish).
- **AIs that understand how people really type:** lowercase, no question mark.
- **Faster training:** a first AI in about 40 seconds instead of 70; the same share for bigger brains.
- **Bigger brains and longer training** for new worlds, sized so the longest run stays near 30 minutes on a card like the owner's.
- **Fewer things going wrong:** runs that move between friends' cards, failed runs that hang, out-of-date helpers.

## 2. What was wrong (measured, RTX 3050)
1. Every AI learned only conversations that start with a capital letter and end with punctuation; a lowercase "hello" is three rare tokens and gets gibberish. The owner's test messages were all lowercase.
2. Answers are drawn at temperature 0.8, too random for brains this small (50–60% well-formed replies; 77–87% at 0.6).
3. The learning-style slider's ends are traps: Careful 2.86 and Bold 2.83 held-out loss against 2.57 in the middle. The owner trained every AI on Careful.
4. The optimizer (AdamW) is about 1.5x less token-efficient than an orthogonalised-momentum optimizer (Muon) on this task.
5. A fifth of a short run is the trainer writing sample sentences one token and one prompt at a time; a fifth of every step is a hand-written normalisation that PyTorch has as one fused operation.
6. Bugs: §3.5.

## 3. The design

### 3.1 Engine speed (the same mathematics)
- `RMSNorm` uses the fused `F.rms_norm` (bit-identical on the CPU, slightly more accurate under bf16); each block casts its normalised input to the autocast type once instead of once per matrix; the loss is computed inside autocast (no explicit float copy of the logits).
- RoPE tables for the usual positions are built once per length and reused.
- The trainer writes all probe samples in one batch; generation skips the padding masks when nothing is padded; held-out losses are computed in one pass.
- Token data is read through a plain array view of the memory map; rollback snapshots stay on the training device; resume states are saved by time (about every 30 s) instead of every 200 steps.
- Rejected: `torch.compile` (needs a community compiler package on every Windows helper), merged q/k/v and gate/up matrices (measured slower), larger batches (the card is already full at 8,192 tokens; ADR 0005's per-token finding stands).
- Acceptance: every existing model, growth, inference and trainer test passes unchanged in meaning; first AI at most 45 s on the RTX 3050 (70 s today); a timing table in ADR 0009. **As built (2026-11-12): 54.9 s; the 45 s target was not reached (ADR 0009 section 5 says why), while the time to a given quality fell by about half.**

### 3.2 Smarter: how conversations are read and answered
- **Casual typing (training).** When the sampler hands out a document with user turns, with probability `CASUAL_SHARE` it retypes every user turn of that document the way people type: all lowercase; lowercase with no final punctuation; no final punctuation; or only the first letter lowercased. The AI's replies are never changed. It is the same conversation with real-world typing, applied to every base dataset alike (not the player's notebook or lessons), deterministic from the run's seed, and part of the engine version. No corpus rebuild.
- **Chat format (serving).** A message a player types in the Playground reaches the AI with its first letter capitalised, as in every conversation the AIs learned from. Nothing the AI says is touched (R1). This helps AIs trained before the upgrade at once.
- **Answer drawing.** Chat, customer answers, keynotes and the trainer's live samples draw at `SERVING_TEMPERATURE`/`SERVING_TOP_P` (0.6 / 0.9; values confirmed in ADR 0009) instead of 0.8 / 0.95. A reply that falls into a loop (the same short token sequence four times in a row) ends there: generation stops, nothing is rewritten. bench-v1, battles, minigames and the personality fingerprint keep their own settings, so recorded scores stay comparable (C56).
- Acceptance: on the casual probe (`probe_casual.py`) lowercase fact questions are answered at least 1.5x as often as by today's engine at equal reading, with clean questions no worse than 3 points lower; the gate's legibility measure at serving settings is at least 75% for a first AI (bar today 60%, measured 59%).

### 3.3 More efficient learning
- **Optimizer.** One optimizer object, `MuonAdamW`: weight matrices of the blocks get momentum whose update is orthogonalised by Newton–Schulz iterations (batched over same-shaped matrices, bf16 on CUDA, fp32 on the CPU); the tied embedding and the norm gains get AdamW. Its state is ordinary per-parameter state, so snapshots, rollbacks and exact resume work as before.
- **Schedule.** Warmup–stable–decay stays; the decay phase is the last 40% and ends at 2% of the peak (was 20% / 10%). The finishing mix stays tied to the decay phase.
- **Learning-style slider (C55).** Boldness maps log-linearly onto the range in which the optimizer is near its best: Careful is the low end (never more than about 0.02 nats worse than the best on a from-scratch run), Bold the high end and beyond it (measurably worse on long runs; whether it adapts faster on short continuation runs is measured before the hint text is written). Clip norm and warmup keep their roles.
- **Engine version.** `ENGINE_VERSION = 2` is written into every resume state and run summary. A resume state from another engine version is never continued: the run starts again from its parent (§3.5).
- **Lineage (R6).** Checkpoints, growth and continuation are unchanged; an AI trained by engine 1 continues under engine 2.
- Acceptance: at equal tokens the held-out loss of tiny, small and medium brains is lower than engine 1's default by at least the gain AdamW gets from 1.3x tokens; continuation after growth is no worse; then the feasibility gate is rerun and its criteria that passed still pass.

### 3.4 Larger brains and longer runs (new worlds)
- A new balance version for new worlds only (existing worlds keep balance-v2 and replay exactly, C56): brain sizes Large (10x384, 19M parameters) and Huge (12x512, 41M) beside today's three, and longer run lengths chosen from the measured speed so that the longest run on the largest brain takes about 30 minutes on an RTX 3050 (C49). The economy is re-checked with the balance harness.
- What a card may train follows measured memory, not `vram_mb * 10,000`; resume states of big brains shrink (no duplicate snapshot in the uploaded copy) to fit the blob limit.
- Rivals for new worlds are rebuilt with engine 2 in a second pack; pack v1 stays for existing worlds.
- Built after §3.1–§3.3 and §3.5 are verified.
- **As built (2026-11-12, ADR 0009 section 6):** `balance-v3` with Large, Huge and a Marathon length (the longest run about 32 minutes); card limits from measured memory; uploads in parts with a 1 GiB blob limit instead of smaller resume states (the free tunnel's 100 MB limit per upload made parts necessary anyway); the economy checked by priced invariants in the sim's tests, not by the balance bot (it has no brain choice). Rivals are not rebuilt yet.

### 3.5 Reliability
1. A run with a parent that moves to another card failed for good ("belongs to a different run config"): the resume check compared the parent's folder path. It now compares the run's content and the parent's version id; a state that cannot be continued is discarded and the run starts over.
2. A failed or finished job tells the world first and cleans up afterwards, so a cleanup error can no longer leave a run "finishing" forever.
3. A finished run whose upload failed is uploaded again on the retry, not retrained.
4. Helpers report their engine version; the game hands training only to helpers of its own version and tells the player to run the setup file again otherwise.
5. Telemetry is reset when a run really starts over.

### 3.6 Order and proof
1. §3.1 speed → 2. §3.2 casual typing and answers → 3. §3.3 optimizer, schedule, slider → 4. §3.5 reliability → 5. gate rerun, timing and before/after answers → 6. §3.4 larger brains → 7. rivals rebuilt.
Each step: tests first, the whole ML suite, then a commit. Experiments run one at a time on the owner's card (C53). The record: ADR 0009, `docs/progress.md`, the handoff README.

## 4. Non-negotiables
R1–R8 as written. R1: the AI's words are never rewritten; drawing settings, the loop stop and the capitalised first letter of the *player's* message are not edits of the AI's output. R5: nothing here is an unlock; sizes are limited by the card and the money. R6: old AIs keep growing.

## 5. Temporary assumptions
- Capitalising the first letter of a typed message is acceptable to the owner as part of the chat format (it can be turned off in one place).
- The customer answers of rival pack v1 were drawn at 0.8 and stay so until the pack is rebuilt: in existing worlds the player's AI gets a small edge.
- The judge and the novelty index are not rebuilt (their data is unchanged).

## 6. Out of scope
A corpus rebuild, a new tokenizer or model family, the honest-score test (bench-v2), Fair-mode scheduling by work, the AI Hub, the tutorial.
