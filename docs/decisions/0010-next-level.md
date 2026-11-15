# ADR 0010: the next level of the training system (measurements and the values they set)

Date started: 2026-11-13. Design: `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`. Plan A: `docs/superpowers/plans/2026-11-13-next-level-a-text-and-voice.md`. Earlier measurements: ADR 0009.

Every task of the plan that measures something writes its numbers and its decision here, kept or removed.

## 1. Measured while planning (2026-11-13, RTX 3050, engine 2)
- **Vocabulary** (`docs/handoff/upgrade-2026-11-10/tok_probe.py`): tok-v1 holds 3.22 bytes a token; a retrained 8,192 vocabulary 3.68 (conversations 3.67 against 3.53), 16,384 holds 3.99. "hello" is 3 tokens in tok-v1 and 2–3 after retraining. **Decision:** tok-v1 stays; casual typing is taught, not tokenized away.
- **Tokens a step** (`train_product.py --batch-tokens`), held-out loss and time for 12.6M tokens: Tiny at 8,192 (the product) 2.447 in 54.9 s; at 16,384: 2.502 in 49.4 s (2.478 with a bolder rate); at 4,096: 2.482 in 75.9 s. Small at 8,192: 2.352 in 82.9 s; 16,384: 2.359 in 79.8 s; 4,096: 2.357 in 107 s. **Decision:** 8,192 stays.
- **Corpus v1's conversations:** 19.8M tokens in 235,749 documents (median 49 tokens): SODA 12.5M (two people; the first speaker is the user, every other the AI), fact questions 3.4M, instructions 3.4M, everyday conversations 0.5M. 3% of the documents carry typos over the whole document, the AI's turns included. `<|sys|>` is used nowhere. The sampler draws documents with replacement.

## 2. Sources and licences for training text 2
- **`HuggingFaceTB/smol-smoltalk`** (card licence apache-2.0; 484,570 rows; columns `messages`, `source`). Its parent card (`HuggingFaceTB/smoltalk`) says: "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0" and "For the existing public datasets, please refer to the original dataset for the license". **Decision:** only rows whose `source` is `smol-magpie-ultra-short` or `smol-contraints` (the card's spelling) are used. Rows from OpenHermes, Self-OSS-Instruct, SystemChat and the other existing public datasets are left out: their licences are their originals', not checked. The rewriting and summarizing rows are Apache-2.0 but are not conversation practice for these brains.
- Everyday conversations keep coming from `HuggingFaceTB/everyday-conversations-llama3.1-2k` (apache-2.0, already in the manifest).
- Checked on the dataset cards on 2026-11-13; the revision is pinned in `airace_content/sources.py` when the source is added (plan A, Task 11).
- **Dropped on 2026-11-15: `HuggingFaceTB/smol-smoltalk`.** A survey of all 460,341 rows of the pinned revision (`docs/handoff/next-level/smol_survey.py`): of the 270,838 rows of `smol-magpie-ultra-short` the adapter keeps 2,685 (1.4M tokens); a quarter of those open with a preamble that tells the AI who it is, and the rest is adult-level explanation, mathematics and role-play ("I'm Bob, the owner of Sunset Motel"). Of the 34,433 rows of `smol-contraints` it keeps 21,088, and 99% of them are rule text ("Your response should contain at least 3 sentences … at least 2 placeholders represented by square brackets … P.S."). A kinder rule (plain opening exchanges, `smol_first_exchange.py`) gives 0.35M to 1.4M tokens of the same kind; 15M were planned. The source is removed (Task 12b).
- Also looked at: `starhopp3r/TinyChat` (a million simple chats): its card states no licence, and its second speaker talks as a person ("I have been making wet soap at home"). Not used.
- **Decision:** no simple, permissively licensed assistant conversations exist at the scale needed, so the game writes its own (`assistant_chat`, Task 13b: short conversations with a small, friendly AI that never speaks as a person), and SODA is narrowed to dialogues whose AI side only listens and uses no speaker's name (Task 12b).

## 3. The scorecard baseline (engine 2, text 1; `docs/reports/2026-11-scorecard-engine2.md`)
`airace-ml scorecard --out runs/scorecard-engine2 --timing`, RTX 3050; the models were trained on 2026-11-13 (nothing else running during the timed run) and measured for the last time on 2026-11-15 with the measuring code of commit a717d2e. Probes are scored on whole words (a reply holding "seventeen" does not answer "seven"). Small talk is 20 lines, 5 replies each; follow-ups 120; identity 6 names, 3 replies each.

| Model | held-out loss | yardstick loss | bench-v1 | well-formed | fits the question | on topic | fits and on topic | casual facts (all five ways / lowercase, no "?") | follow-ups | says its name (told) | says it is an AI (not told) | cut off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| First AI, seed 1 (Tiny, 12.6M) | 2.489 | 2.117 | 51.1 | 70% | 75% | 65% | 55% | 74% / 65% | 8% | 0% | 0% | 6% |
| First AI, seed 2 | 2.465 | 2.109 | 49.0 | 80% | 76% | 67% | 53% | 69% / 64% | 8% | 2% | 0% | 4% |
| Small, 12.6M | 2.346 | 1.959 | 57.0 | 88% | 87% | 62% | 59% | 88% / 83% | 5% | 0% | 0% | 5% |
| Second generation (Tiny grown to Small) | 2.346 | 1.973 | 54.9 | 65% | 78% | 66% | 57% | 88% / 81% | 3% | 0% | 0% | 3% |
| Third generation (grown to Medium) | 2.268 | 1.888 | 59.1 | 79% | 90% | 82% | 75% | 92% / 87% | 4% | 1% | 0% | 5% |
| Conversation-heavy Small, 8.4M | 2.562 | 1.736 | 48.2 | 75% | 89% | 83% | 78% | 94% / 94% | 3% | 0% | 0% | 4% |
| Large, 25.2M | 2.083 | 1.654 | 65.3 | 83% | 95% | 85% | 83% | 95% / 91% | 7% | 12% | 0% | 5% |

The first AI trained in 53.0 s. Real held-out replies themselves: fits 90% (by the rule's definition), on topic 94%, both 86%.

**What the baseline says:**
- **Follow-up questions are barely answered at all** (3–8%), by any brain. This is the largest gap the card shows.
- **No AI says its name when told it, or says it is an AI when asked**, as expected: nothing in text 1 teaches either.
- Loops are gone (0–1%); cut-off replies are 3–6%, under the 10% target.
- By the word list, claims of a person's life are rare (0–1%); a reader sees more (below).

**How the talk measures came to be (2026-11-13 to 15).** Each was added after reading the replies showed what the one before missed; the stick is frozen at a717d2e.
1. *Well-formed* (the gate's rule) fails good short replies ("You're welcome." to "Thank you.") because it wants a minimum length and never sees the question. Kept for continuity with G2.
2. *Fits the question:* the judge's loss of the reply given the question, no higher than the 90th percentile of 1,000 real held-out replies (2.524). It passes fluent answers to another question.
3. *On topic:* how much likelier the judge finds the reply after its own question than after other questions. The first form (mean per token over the whole reply, 8 other questions) told real replies from swapped ones with balanced accuracy 87%, failed every story (a long reply dilutes the mean) and rejected only 57% of hard negatives (the neighbouring pair's reply). `docs/handoff/next-level/topic_probe.py` compared forms; the card uses the best: the total over the reply's first 24 tokens against 32 other questions, threshold 10.02 nats: balanced accuracy 94%, 94% of swapped and 81% of hard-swapped replies rejected.
4. *What no judge rule sees.* The judge was trained on text 1, so it finds "We call a group of monkeys a troop." a likely reply to "What is your name?" and "Hello! How can I help you today?" a likely reply to "Goodbye.". The rules are therefore **guards** (a model must not get worse on them), not the measure of talk.
5. **The blind reading** (`docs/handoff/next-level/blind_read.py`) is the measure of talk: the stored replies of the models to compare go on one sheet, shuffled and unlabelled; a separate reader grades each 2 (sensible), 1 (partly) or 0 (not), and marks a reply that speaks as a person with a life. Checked on this baseline (`docs/reports/2026-11-blind-reading-engine2/`, 258 items):

| | sensible | nonsense | speaks as a person |
|---|---|---|---|
| First AI, seed 1 | 34% | 43% | 9% |
| First AI, seed 2 | 27% | 43% | 10% |
| Large | 55% | 20% | 23% |

  Large minus first AI: +21 and +28 points (90% intervals, resampling the lines, exclude zero); the two first AIs: 7 points apart (interval includes zero). So the reading tells a much better brain from a first AI and puts two seeds of one recipe together. "Speaks as a person" is mostly the AI calling the person by a made-up name ("Of course, Caul."): SODA's dialogues are between two named people.
- Twenty more lines are held out (`airace_content/held.py`, fixed before any generator of small talk was written; no generator may write them): the reading of text 2 uses the 20 standard lines and these 20.

**Targets fixed from the baseline** (the spec's section 7 rules):

| | baseline | target |
|---|---|---|
| First AI: small talk a reader finds sensible (blind reading) | 27–34% | higher than text 1's first AI, with a 90% interval that excludes zero, on the standard lines and on the held-out lines; aim: at least 50% (today's Large brain reads 55%) |
| First AI: speaks as a person (blind reading) | 9–10% | lower |
| First AI: fits and on topic (guard; real replies score 86%) | 53–55% | not lower |
| First AI: well-formed (the gate's rule, guard) | 70–80% | not lower |
| First AI: facts typed in lowercase without "?" | 64–65% | at least 80% |
| Says its name when told / says it is an AI when not told | 0–2% | at least 60% for Tiny, 85% for Small and larger |
| Follow-ups, Small | 5% | at least 60% |
| Third generation: bench-v1 overall | 59.1 | at least 64 |
| Life claims by the word list | 0–1% | at most 5% (guard) |
| Cut-off replies | 3–6% | at most 10% (guard) |
| First AI: training time | 53.0 s | at most 55 s |
| Lessons answered after one run | measured in Task 8 | at least 7 of 10 |

**A target that was replaced, and why.** On 2026-11-13 the first target read "fits the question (and on topic, once measured): at least 90%". Real replies themselves score 86% on that pair of rules, so no model can reach 90%, and the blind reading showed the pair is lenient (53–55% where a reader finds 27–34% sensible). The target is now the blind reading; the pair stays as a guard.

**Rulings on the measuring stick itself** (plan A, Task 3): the yardstick loss is the trainer's held-out rule over 128 packed rows of text 1's held-out conversations; a baseline measured with another yardstick, judge, novelty index, scale or on-topic calibration is refused; cards of different training text are not compared on their own held-out losses; a lost measure or a run that stopped early counts as worse. The tolerances are allowances, not tests of significance: one 100-reply rate has a standard error of about 5 points, so a flagged small-talk or identity change is confirmed with a second seed or by reading the stored replies before anything is decided on it.

## 4. Engine 3 items
Engine 3 is one number for every item of this section that is kept: nothing is released between them.

### 4.1 Documents drawn without replacement (Task 5): kept
- **The change** (f011be5): a pool walks a shuffled order of its documents and shuffles again when the order is used up; a balanced pool keeps one order per topic. The saved state is a seed and two integers per order. Progress saved by engine 2 is not continued.
- **A measuring error found on the way** (fixed in 99ca8e5). The trainer's held-out batches (8 rows a dataset) and the scorecard's yardstick (128 rows) were drawn through the training sampler, so the change silently altered which rows a model is scored on: the first measurement showed held-out conversation loss falling from 2.41 to 1.99, which was a different set of rows, not a better model. Evaluation pools now draw with replacement and take nothing from the sampler's generator; the yardstick of the engine-2 baseline models is reproduced to six decimals. A held-out loss means the same before and after engine 3.
- **Measured** (`docs/handoff/next-level/replacement_lab.py`): the first AI trained with and without replacement, seeds 1 to 6, the whole scorecard on each; difference (without minus with) and its standard error:

| | with | without | difference |
|---|---|---|---|
| held-out loss | 2.470 | 2.464 | -0.006 ± 0.009 |
| yardstick loss | 2.107 | 2.109 | +0.002 ± 0.003 |
| bench-v1 overall | 49.0 | 50.0 | +1.0 ± 0.8 |
| well-formed | 76% | 79% | +2 ± 2 points |
| fits and on topic | 54% | 54% | 0 ± 3 points |
| casual facts | 72% | 76% | +3 ± 2 points |
| follow-ups | 6% | 3% | -3 ± 1 points |

- **Decision: kept.** On a 12.6M-token run few documents repeat either way, so nothing moves, as expected; the rule is right for long runs and small pools (every document once before any twice). Follow-ups sit near zero in text 1 either way; text 2 teaches them.
- **What the six seeds say about noise:** between seeds of one recipe, bench-v1 differs by about 2 points (48.0 to 51.6), talk rates by 5 to 10 points, the yardstick loss by 0.006. The single-seed flags of the first measurement (bench 2 to 3 points lower) were noise. Decisions on bench and talk need several seeds or a paired design; losses are the precise measures.

### 4.4 Lessons that stick (Task 8): already do; nothing changed
- **Measured** (`docs/handoff/next-level/lessons_lab.py`, engine 3): a first AI continued with one run built field for field as the game builds it for "Train on your lessons" (the same Tiny brain, the Short length of 6,291,456 tokens, the even mix, the lessons at the game's share of 5%), with ten made-up lessons ("What colour is a glimmerfin?" → "A glimmerfin is teal."):

| lessons' share | answered | answered when typed in lower case without "?" | yardstick loss (parent 2.110) | times each lesson was read |
|---|---|---|---|---|
| before the run | 1 of 10 | 1 of 10 | 2.110 | - |
| 5% (the game's) | 10 of 10 | 10 of 10 | 2.097 | 1,122 |
| 10% | 10 of 10 | 9 of 10 | 2.107 | 2,280 |
| 20% | 10 of 10 | 10 of 10 | 2.120 | 4,637 |

- **Decision:** the target (at least 7 of 10) is met at the game's share, so the share stays 5%. A larger share buys nothing and starts to cost the rest of what the AI knows (the yardstick loss rises with it).
- Ten short lessons are read more than a thousand times in a Short run. A player with many long lessons gets fewer readings of each; that case was not measured.

## 5. Training text 2 against text 1
(Tasks 14–15.)

## 6. A voice of its own
(Tasks 16–18.)
