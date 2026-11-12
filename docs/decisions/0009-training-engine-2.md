# ADR 0009: training engine 2 (measurements and the values they set)

- **Date:** 2026-11-10/11
- **Status:** Accepted (engineering decisions; the owner's part is C49, C54–C56 and the 2026-11-10 request). Design: `docs/superpowers/specs/2026-11-10-training-engine-2-design.md`.
- **Evidence:** `docs/handoff/upgrade-2026-11-10/` (every script; `results.jsonl` has one line per training run). All numbers are from the owner's RTX 3050 with nothing else on the card. "Loss" is the mean held-out loss over the eight datasets (128 sequences each, the same for every run); seed-to-seed it varies by about 0.005 (AdamW) to 0.015 (Muon).

## 1. Where the time went (engine 1)
| Run | Wall | Model step | Sample writing + held-out | Sampler | Snapshots, saves |
|---|---|---|---|---|---|
| First AI, 4×128, 12.6M tokens | 70.4 s | 65% | 20% | 7% | 2% |
| Medium 8×256, 6.3M tokens | 102 s | 70% | 24% | 1% | 2% |

Inside the model step only 20% (tiny) to 27% (medium) of the GPU time is matrix products; the rest is memory-bound elementwise work, 20% of it the hand-written RMSNorm.

Raw step throughput (tokens a second, 8,192-token batches): tiny 272k → **344k** with the fused norm and the loss inside autocast; medium 87k → **108k**. Batches of 16k–65k tokens add 5–15% (the card is full at 8k), merged q/k/v or gate/up matrices are 4–8% slower, a `roll`-based RoPE is no faster. `torch.compile` needs Triton, which PyTorch does not ship for Windows.

## 2. The optimizer (first-AI recipe: 4×128, 12.6M tokens, even mix, standard cleaning)
| Setting | Loss |
|---|---|
| AdamW 6e-4 (the old **Careful**) | 2.855 |
| AdamW 1.5e-3 | 2.617 |
| AdamW 2.68e-3 (the old default) | 2.568, 2.572 |
| AdamW 4e-3 | 2.546 |
| AdamW 6e-3 | 2.591 |
| AdamW 1.2e-2 (the old **Bold**) | 2.835 |
| AdamW 4e-3 on 16.4M / 18.9M tokens | 2.484 / 2.459 |
| AdamW, no weight decay / linear-to-zero schedule / 4,096-token batches | 2.567 / 2.636 / 2.574 |
| Muon 0.005 / 0.01 / 0.015 / 0.02 (embedding at 1.2e-2) | 2.482 / 2.470 / 2.473 / 2.480 |
| Muon 0.04 / 0.08 | 2.757 / 3.238 |
| Muon 0.015, momentum 0.9, 3 iterations, decay 40% to 2% (**chosen**), 3 seeds | **2.447, 2.451, 2.452** |

- Muon with the chosen settings is 0.12 better than the old default and reaches what AdamW needs about 1.5× the tokens for. It costs about 6% more time per step (batched by matrix shape).
- Bigger brains: small 6×192 at 12.6M tokens 2.439 → **2.315**; medium 8×256 at 25.2M tokens 2.189 → **2.114**.
- Its good range is wide (0.005–0.02 within 0.02 of the best); beyond it quality falls off without the run diverging.
- 3 Newton–Schulz iterations learn as well as 5; momentum 0.9 a little better than 0.95; weight decay on the matrices changes nothing (left at 0); gradient clipping helps slightly (kept).
- The embedding and gains' AdamW rate: 2.7e-3 → 2.492, 6e-3 → 2.477, 1.2e-2 → 2.468, 2.4e-2 → 2.480. Chosen: 0.8 × the matrices' rate.
- Schedule: decay over the last 40% to 2% of the peak is 0.01–0.02 better than the last 20% to 10%, for both optimizers.

**Decisions:** `MuonAdamW` (`ml/src/airace_ml/train/optim.py`); `DECAY_FRAC = 0.4`, `MIN_LR_RATIO = 0.02`; the slider maps boldness 0…1 onto a matrix rate of 0.006…0.03 (log scale; 0.0134 in the middle), so Careful and the default are inside the good range and only Bold leaves it (C55).

## 3. Talking
- **Casing.** 99.5% of the corpora's user turns start with a capital and all end in `.`, `?` or `!`. `hello` is the tokens `he|ll|o`; `Hello` is one token.
- **Fact questions in the training's own chat phrasing, share answered right (greedy, 110 questions over all 22 relations):**

| First AI (4×128, 12.6M tokens) | as written | all lowercase | lowercase, no "?" | no "?" | first letter capital, rest lowercase, no "?" |
|---|---|---|---|---|---|
| engine 1 (the gate's `starter`) | 67% | 34% | 17% | 60% | 48% |
| Muon, no casual typing | 88% | 62% | 59% | 87% | 71% |
| Muon + 30% casual typing | 86% | 74% | 75% | 86% | 74% |

  Casual typing costs nothing on clean text (held-out loss 2.441 against 2.447).
- **Temperature** (well-formed replies by the gate's measure, 60 replies): engine-1 first AI 50% at 0.8/0.95, 77% at 0.6/0.95, 85% at 0.5/0.95, 80% greedy; engine-1 small brain 60% / 87% / – / 95%; the judge 85% at every setting. Garble-like words 1.2% → 0.7%.
- **Decisions:** `CASUAL_SHARE = 0.3` with the five typing styles of `airace_ml/data/casual.py`; serving at temperature 0.6, top-p 0.9; a looping reply ends (`airace_ml/infer/serving.py`); the Playground capitalises the first letter of a typed message.

## 4. Considered and not done
- A corpus rebuild with casual conversations: the sampler does it without one.
- Lower temperature for bench-v1, battles, games and the fingerprint: recorded scores must stay comparable (C56).
- Tightening "standard" cleaning: typo pages are 10.7% of the web pool and 2.5% of conversations at standard; garble in replies is already under 1% at the new serving settings. To be measured again after the gate.
- Architecture changes (QK-norm, untied head): they would need a second model family for growth and old checkpoints; the optimizer gave more for less.
- Fewer launches in the optimizer's Newton–Schulz step (`baddbmm`) and a sampler prefetch thread: at most 2% and 5% of the smallest brain's run, nothing on the larger ones; the prefetch would also complicate exact resume.

## 5. Engine 2 as built (RTX 3050, the product trainer, nothing else running)

| Run | engine 1 | engine 2 |
|---|---|---|
| First AI (Tiny 4×128, 12.6M tokens): time | 70.4 s | 54.9 s |
| First AI: held-out loss (mean of the 8 datasets) | 2.568 | 2.447–2.452 |
| Small 6×192, 12.6M tokens: held-out loss | 2.439 | 2.315 |
| Medium 8×256, 6.3M tokens: time | 102 s | 78.2 s |
| Medium 8×256, 25.2M tokens: held-out loss | 2.189 | 2.114–2.117 |

- The spec's target of 45 s for the first AI was not reached. Of the 54.9 s, forward and backward are 35.2 s, which is the floor without compiling the model (not dependable on Windows). What did fall by about half is the time to a given quality: engine 1 needs 18.9M tokens (about 106 s) for the loss engine 2 reaches in 12.6M (55 s).
- First AI by the product path, casual questions answered right (as written / all lowercase / lowercase without "?"): engine 1 67 / 34 / 17%, engine 2 85 / 66 / 65%; well-formed replies at the old and new talking settings: 50% → 80–82%.
- A lineage as a player builds one (Medium brain, three runs, each continuing the last; 144 s, 424 s and 476 s on the card): held-out conversations 2.179 → 1.900 → 1.818; the third AI answers the fact questions 100% as written and 93–94% typed in lowercase without "?".

## 6. Larger brains (new worlds: `balance-v3`)
Measured with the product trainer at the longest attention span (512), 3.1M tokens each:

| Brain | parameters | training tokens a second | µs a token | peak card memory | saved progress |
|---|---|---|---|---|---|
| Tiny 4×128 (span 256) | 1.38M | 229,000 | 4.4 | – | – |
| Medium 8×256 | 7.48M | 95,000 | 10.5 | 1,380 MB | 126 MB |
| Large 10×384 | 19.28M | 46,700 | 21.4 | 2,318 MB | 312 MB |
| Huge 12×512 | 40.65M | 26,700 | 37.4 | 3,532 MB | 644 MB |

**Decisions:**
- `balance-v3` is `balance-v2` plus the Large and Huge brains and a Marathon length of 50,331,648 tokens; new worlds get it, a world made earlier keeps the version it recorded (C56). The longest run (Huge, Marathon) is about 32 minutes on the RTX 3050 (C49: about half an hour).
- Compute keeps its price (6 × parameters × tokens): on rented compute alone a Huge Marathon run is about 1,530 game days and $4.9M; with three data centers lent to training, 78 days and about $250,000; a first Huge brain on the Short length with one data center, 27 days and $85,000. Serving keeps its price per customer and million parameters (Huge: about $12 a customer a month rented, under $4 on the company's own centers). The big brains are what a company grows into. The balance bot does not play them yet (it has no brain choice); the invariants above are tests in `packages/sim/test/training.test.ts`.
- The Lab's minute estimate: a token costs `2.6 + 1.25 × (millions of parameters)^0.9` µs on the reference card (4.3, 10.2, 20.6, 37.7 against the measured column), scaled by the card's speed-test result against the RTX 3050's 450,000. The old estimate scaled the tiny test model by size and was about four times too slow for Medium.
- A card's largest brain comes from measured memory: a run peaks at about 1,000 MB + 68 MB per million parameters, and 500 MB is left for everything else on the card (`max_params_for_vram`). 8 GB and 6 GB cards train Huge, 4 GB Large, 2 GB Medium. The old rule (10,000 parameters per MB) let a 4 GB card start a Huge run it could not hold.
- Uploads over 32 MB go in parts (`&part=i&parts=n` on the same signed address; the blob appears when the last part is in): the free Cloudflare tunnel friends play through refuses a single upload over 100 MB, which a Huge brain's checkpoint (163 MB) and every saved progress from Medium up exceed. The blob limit is 1 GiB.

## 7. Helpers and reliability
- A helper names its training engine in its hello (`airace_ml/engine.py`, `TRAINING_ENGINE` in `packages/protocol/src/worker.ts`, tied by the shared `whello` example). One from an older version gets no work, trains for nobody, and its card says "needs an update: run the helper's setup file again".
- A run already trained on a helper whose upload failed is uploaded, not trained again (`finished_run`); saved progress that is not this run's (other settings, an older engine, a save cut short) is dropped and the run starts over; the game then clears the run's charts.
- The world is told that a run finished or failed before the saved progress is cleaned up, so a failing delete cannot leave a run "training".
- A helper keeps its PC awake while it works on a job, and so do the long commands (`train`, `bench`, `build-novelty-index`, `build-judge`, `build-rivals`, `gate`): `airace_ml/awake.py`. Nothing in the PC's power settings is changed.
- Cards: 1.0× speed is what an RTX 3050 measures on the speed test (it read "15×").

## 8. The feasibility gate with engine 2 (2026-11-12, `docs/reports/2026-11-12-gate-engine-2.md`)

| | engine 1 (`m1-feasibility.md`) | engine 2 |
|---|---|---|
| G1 Speed | PASS: first AI 68.5 s on the card, 1,008 s on the CPU; 124 tokens/s in chat | PASS: 50.4 s, 1,028 s; 126–168 tokens/s |
| G2 Legibility (needs 60% well-formed) | FAIL: 59% | PASS: 80% as the game talks (58% at the earlier drawing settings) |
| G3 Differentiation | PASS | FAIL on one of four leads: knowledge |
| G4 Seed variation | PASS | PASS |
| G5 Growth | PASS: grown model gains 0.072 | PASS: gains 0.045 (both models are better: 2.339 against 2.385; engine 1: 2.380 against 2.452) |
| G6 Forgetting | PASS | PASS |
| G7 Preparation | FAIL: garble 2.4% thorough against 2.9% light | FAIL: 2.2% against 2.4% (fact-checking works: false facts preferred 47.7% against 71.1%) |

Five of seven, as before, but not the same five.

- **G2 is measured as the game talks.** The gate drew its replies at temperature 0.8 / top-p 0.95, which the game no longer uses anywhere a person reads an AI. At those settings engine 2 changes nothing (58% against 59%): the gain a player sees comes from the calmer drawing and the loop stop, not from the optimizer.
- **G3, knowledge.** Every AI knows more: the conversation-heavy model's knowledge rose from 46 to 64, the code- and story-heavy ones from 7 to 27 and 23 (means over the three seeds). The facts specialist itself fell from 77.4 to 71.2, so its lead (7.8) is under the bar (10.4, twice the seed spread). The other three leads grew or held (creativity +30.6, coding +15.6, instruction +14.5). Every specialist's overall score rose: code 23.6 → 37.9, stories 26.1 → 38.4, conversations 38.4 → 49.9, facts 42.6 → 49.7.
- **What was tried on the facts specialist** (`docs/handoff/upgrade-2026-11-10/specialist.py`, the gate's recipe, 2–3 seeds each; one run varies by about ±5): no casual typing 70.8 against 68.8 (about 2 points); learning style Careful 63.5, a quarter 71.4, middle 68.8–71.2, three quarters 75.4; embedding rate at a quarter of the matrices' 72.1 (71.4–72.8, the steadiest); no embedding weight decay 71.0; the earlier schedule 72.1. Nothing reaches 77: it is how this optimizer learns a small, heavily repeated dataset, and it is accepted for what it gives everywhere else. Its held-out loss on the facts is lower than engine 1's (0.43 against 0.45); the lost answers are in a few relations (an animal's legs 97% → 81%, a chemical symbol 88% → 77%).
- G7's known gap (cleaning barely lowers garbled output) is unchanged.

## 9. Still open
- The rival pack for worlds of the newer rules (`data/rivals/v2`) is being built at the time of writing; until it is there, new worlds draw their rivals from the earlier pack.
- A Huge brain's saved progress (644 MB) is uploaded every two minutes in Fair mode; over a home line that is heavy. Leaving the roll-back copy out of the uploaded state would halve it.
- The balance bot has no brain choice, so the economy of the Large and Huge brains is checked by priced invariants only, not by a played year.
- A played session with the new brains (`npm run e2e` passes: solo and both two-player worlds).
