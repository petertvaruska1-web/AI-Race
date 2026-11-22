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

### 4.2 Counting the person's turns less in the loss (Task 6): removed
- **The idea:** a chat document trains the model on every token, the person's turns included, although the model never writes a person's turn. A weight below 1 on those targets (the person's marker, text and closing token) would spend the brain on the AI's own turns.
- **Measured** (`docs/handoff/next-level/user_weight_lab.py`, engine 3, text 1): the yardstick loss on the AI's turns alone (14,179 targets of the 128 yardstick rows):

| weight of the person's turns | first AI, seed 1 | first AI, seed 2 | conversation-heavy Small |
|---|---|---|---|
| 1.0 (as before) | 2.318 | 2.321 | 1.836 |
| 0.3 | 2.360 | 2.400 | 1.859 |
| 0.0 | 2.383 | 2.428 | 1.924 |

- **Decision: removed** (the rule asked for the AI-turn loss to fall by 0.02; it rose by 0.02 to 0.11, steadily as the weight fell; seed noise on this loss is about 0.006). Learning to predict the person's turns teaches language the AI's turns use too. The mechanism (`_loss_weights`, commit ec8b390) is deleted again; the lab script stays as a record.
- Not measured on text 2. Its person's turns are short questions with typing noise; the result above is so steady in the weight that a different sign is not expected, and a brain that reads noisy typing has to learn that typing somewhere.

### 4.3 What a grown brain keeps of its parent (Task 7): no change
- **The question:** when a player grows an AI to a bigger brain, how much of what the parent learned does the child keep? `docs/handoff/next-level/growth_lab.py` trains the scorecard's first AI (Tiny) and grows it to Small, in several ways, two seeds each, with a check that growth keeps the parent's answers exactly (largest difference 2.5e-5) and that every variant changes only the elements it should.
- **Round 1 (child 12.6M tokens):** grown as today (A) 2.337 held-out loss; a new Small with no parent (E) the same (A minus E: +0.000, −0.004); a Small continuing a Small (D) 0.085 better. Speeding up the new parts in the first third (B) or slowing the old parts in the first third (C): 0.013–0.017 worse. The curves showed why: the child's first steps at the full learning rate wipe out most of the parent's knowledge (held-out 2.46 → 2.65), and the run wins it back.
- **Round 2:** holding the inherited parts at 0.3 or 0.1 of their update for the whole run (F, G) keeps the parent's knowledge from the first step (2.44 instead of 2.65), but the runs end level with A (2.336 and 2.335 against 2.332): a bigger brain learns by itself in 12.6M tokens what the small parent knew.
- **By length of the child run (G against A, two seeds each):**

| child run | A | G (inherited parts at a tenth, whole run) | H (a tenth rising to the full rate) | a new brain (E) |
|---|---|---|---|---|
| Short, 6.3M tokens | 2.419 | **2.391** (0.028 better) | 2.416 (0.003 better) | 2.548 |
| 12.6M tokens | 2.332 | 2.335 | – | 2.336 |
| Long, 25.2M tokens | 2.254 | 2.273 (0.020 worse) | – | – |

- **Decision: no change.** No single rule helps at every length: G helps a Short run and costs a Long one; letting the protection fade (H) loses the Short gain. The rule fixed before H was measured asked for both, so the trainer is unchanged. A rule that depends on the run's length (protect only Short runs) is noted for a later engine; it was measured on one growth (Tiny to Small) only.
- **What this means for players:** a grown AI ends about where a new brain of its size ends on the same reading, and on a Short run inheriting is worth a lot (0.13 better than a new brain). The value of a lineage is the bigger brain and the longer reading; the first AI is not made worse by any of this.

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
### 5.1 What text 2 is (built 2026-11-15 to 17 into `data2/`)
`AIRACE_DATA=../data2 airace-content build --scale full --text 2` (48 minutes), then the conversations again alone after the generator was made reproducible (`--datasets conversations`, 10 minutes). 258M tokens in all (text 1: 139M):

| dataset | tokens | target | made of |
|---|---|---|---|
| web | 64.9M | 60M | fineweb 64.9M |
| books | 29.9M | 30M | gutenberg 29.9M |
| educational | 41.2M | 40M | cosmo_khan 10.9M, cosmo_wikihow 9.9M, cosmo_openstax 8.1M, fineweb_edu 12.3M |
| conversations | 30.7M | 30M | assistant_chat 9.8M, fact_chat 7.2M, instructions 6.1M, followup_chat 4.2M, identity_chat 2.8M, everyday_conv 0.5M |
| code | 20.4M | 20M | code 20.3M, mbpp 0.0M |
| reasoning | 19.3M | 20M | reasoning 11.6M, patterns 6.5M, gsm8k 1.2M |
| facts | 12.0M | 12M | fact_prose 9.0M, fact_qa 3.0M |
| creative | 39.7M | 40M | tinystories 28.0M, tinystories_requests 5.5M, cosmo_stories 6.2M |

- **Conversations** are written by the game's own generators only: short chats with a small, friendly AI that never speaks as a person (`assistant_chat`), fact questions, instructions, follow-up questions that refer back, and chats in which the AI knows its name and company; 40% of the chats of most kinds open with an identity line. Typos fall on the person's turns only. No two-person dialogues from SODA (section 2).
- The build warned that `identity_chat` reached 86% of its plan (its bytes-per-token estimate was low) and that `everyday_conv`, `mbpp`, `gsm8k` and `tinystories_requests` ran out (their sources are small; the other components of their datasets made up the difference).
- 20 random conversation documents read by the lead: correct, short, child-safe; typos only in the person's lines ("Make a sentenfe with the word train in it.").


### 5.2 Round 1: text 2 against text 1 (engine 3, 2026-11-17)
The whole scorecard on each text (`docs/handoff/next-level/text_ab.sh`: `runs/scorecard-e3-text1`, `runs/scorecard-e3-text2`, yardstick text 1 for both), then the blind reading of the two first AIs of each text on the 20 standard lines and on the 20 held-out lines (100 replies each; the text-2 AIs told "You are Toby, an AI made by Sunrise Studio.", as the game will tell them their name; text-1 AIs get no line, as the helper gives them none; two separate readers, one per sheet, neither knowing which AI wrote what; `docs/reports/2026-11-blind-reading-text2-round1/`).

**Small talk, blind reading (share of replies a reader finds sensible / nonsense / speaking as a person):**

| | standard lines | held-out lines |
|---|---|---|
| text 1, first AI seed 1 | 29% / 36% / 8% | 13% / 53% / 9% |
| text 1, first AI seed 2 | 18% / 51% / 15% | 9% / 60% / 8% |
| text 2, first AI seed 1 | **56%** / 12% / 1% | **35%** / 23% / 0% |
| text 2, first AI seed 2 | **48%** / 13% / 1% | **31%** / 18% / 1% |

Each text-2 AI against each text-1 AI: +19 to +38 points on the standard lines and +18 to +26 on the held-out lines; every 90% interval excludes zero except one that touches it (+19, interval 0 to +37). The gain carries over to lines no generator wrote, at a lower level.

**The rest of the card (first AI seeds 1 and 2; Small; third generation; Large):**

| | text 1 | text 2 |
|---|---|---|
| follow-up questions answered | 2%, 0%; 4%; 8%; 2% | **71%, 66%; 71%; 80%; 75%** |
| says it is an AI when not told | 0% everywhere | **99%, 89%; 90%; 64%; 89%** |
| says its name when told (6 names, 3 questions each, plus "are you a person") | 2%, 3%; 1%; 0%; 0% | 41%, 63%; 61%; 60%; 84% |
| facts typed casually, all five ways | 73%, 76%; 82%; 92%; 97% | 80%, 76%; 87%; 94%; 96% |
| … lowercase without "?" | 63%, 72%; 75%; 86%; 95% | 71%, 68%; 79%; 90%; 94% |
| bench-v1 overall | 49.7, 47.5; 53.0; 57.8; 63.5 | 49.8, 52.1; 58.2; 59.1; 65.6 |
| well-formed (gate rule) | 79%, 87%; 89%; 74%; 70% | 79%, 81%; 85%; 80%; 84% |
| fits and on topic (text-1 judge) | 56%, 44%; 62%; 68%; 80% | 36%, 22%; 36%; 41%; 46% |
| life claims (word list) | 0–1% | 0–1% |
| cut off | 3–5% | 3–6% |

- **The judge-based talk rules do not travel across texts.** The judge was trained on text 1 and scores text-2 replies as unlikely ("fits" falls from 75–92% to 31–54%) while the readers find far fewer nonsense replies in text 2 (12–23% against 36–60%). Ruling: those rules stay guards between models of one text; between texts the blind reading decides (the caveat recorded in section 3 when the stick was frozen). The yardstick loss (text 1's held-out conversations, 63% SODA two-person dialogue) is likewise not a guard across texts: text 2 deliberately does not teach that kind of talk (2.80–2.84 against 2.11 for the first AI).
- **Names are not copied, they are recalled.** Told "You are Zula, …", the AIs answer "I'm Zee", "Zesterester", "Zimbabwe"; told "Pip", the Large brain says "Pipi"; told "Pengu", it says "Penguin, an AI made by Pellam Works", a company of the training text. With 2,100 names and 677 companies used again and again, a brain learns the names and says one it knows that starts like the one it was told. Round 2 draws a new name and company for every identity chat, so that copying is the only way to be right.
- **Targets after round 1:** small talk (blind reading) met; speaking as a person lower: met; follow-ups (Small ≥ 60%): met (71%); says it is an AI when not told: met (89–100%, the third generation 64%); says its name when told: not met (41–63% for a first AI against 60%, 60–84% for larger brains against 85%); lowercase facts for a first AI (≥ 80%): not met (68–71%); third-generation bench (≥ 64): not met (59.1); life claims and cut-off replies: within their guards; training time: measured at the gate.

### 5.3 Round 2 (fresh identity names) and the default reading of a first AI
- **Round 2:** every identity chat got a name and a company of its own (no name can be learned by heart). Says its name when told: first AI 44% and 58% (round 1: 41%, 63%), Small 62%, Large 91% (84%). Small brains still answer with a name that starts like the right one. Other measures level within seed noise (`docs/reports/2026-11-scorecard-e3-text2-round2.*`).
- **The default reading of a first AI** (`docs/handoff/next-level/mix_lab.py`; the spec's owner assumption, section 0): conversations at 30% and the other seven datasets at 10% each, against the even mix (12.5% each), on round 2's text, two seeds; rule fixed before measuring (lowercase facts at least 80%, blind reading not lower, bench not lower by more than 2 points):

| first AI, text 2 | even mix (today's default) | conversations 30% |
|---|---|---|
| small talk sensible, standard lines (blind) | 46%, 39% | **60%, 57%** |
| … held-out lines (blind) | 29%, 32% | **52%, 47%** |
| nonsense, standard / held-out | 13–16% / 31–38% | 5–9% / 17–19% |
| facts typed in lowercase without "?" | 73%, 72% | **91%, 88%** |
| follow-up questions | 67%, 55% | 81%, 75% |
| says its name when told | 44%, 58% | 46%, 49% |
| bench-v1 overall | 47.0, 48.7 | 48.8, 53.3 |

  Each conversation-leaning AI against each even-mix AI: +14 to +21 points on the standard lines, +15 to +23 on the held-out lines, every 90% interval above zero (`docs/reports/2026-11-blind-reading-mix-trial/`). **Decision: kept.** The release makes "conversations 30%, the rest 10% each" the default reading of a new AI in new worlds (a balance change, so new worlds only; the player can still move every slider). Bench did not fall: the lost share of the other datasets is small and a Tiny brain gains more from talk practice than from a little more of everything.

### 5.4 Round 3 (practice in copying a name) and the final comparison
- **Round 3:** in half the assistant chats the person's own name and their pets' names are made fresh, the AI says its own name in some greetings and when asked in chats that carry a line, and identity chats rise to 16% of the conversations (`docs/reports/2026-11-scorecard-e3-text2-round3.*`). Says its name when told: Small 70% (round 2: 62%), second generation 70% (67%), third generation 94% (62%), Large 87% (91%), first AI 39% and 68% (44%, 58%). Nothing else moved beyond seed noise. **Kept:** text 2 is round 3's.
- **The new first AI** (text 2 of round 3, the default reading "conversations 30%", `docs/reports/2026-11-scorecard-text2-final-first.*`) against today's first AI (text 1, even reading), blind reading by two new readers (`docs/reports/2026-11-blind-reading-final/`):

| | today's first AI (seeds 1, 2) | the new first AI (seeds 1, 2) |
|---|---|---|
| small talk sensible, standard lines | 27%, 18% | **59%, 55%** |
| small talk sensible, held-out lines | 14%, 4% | **56%, 50%** |
| nonsense, standard / held-out | 38–46% / 59–72% | 10% / 12–16% |
| speaks as a person | 7–12% | **0%** |
| follow-up questions answered | 2%, 0% | **78%, 73%** |
| says it is an AI when not told | 0% | **98%, 89%** |
| says its name when told | 2%, 3% | **54%, 74%** |
| facts typed in lowercase without "?" | 63%, 72% | **87%, 87%** |
| bench-v1 overall | 49.7, 47.5 | **52.6, 53.6** |

  Each new AI against each of today's: +28 to +41 points on the standard lines and +36 to +52 on the held-out lines; every 90% interval is far above zero.
- **Targets (section 3) at the end of plan A's text work:** small talk (blind) met, the aim of 50% reached on both kinds of line; speaking as a person met (0%); lowercase facts met (87%); follow-ups for a Small met (71%); says it is an AI when not told met (89–100%); says its name when told: first AI 54% and 74% against 60% (one seed under), larger brains 70–94% against 85% (Small, second generation and the conversation-heavy Small under) — partly met; third-generation bench 60.2 against 64: not met (text 1: 57.8); life claims and cut-off replies within their guards; well-formed (the gate's rule, a text-1 judge) falls for the first AI (69%, 78% against 79%, 87%) while readers find far fewer nonsense replies — a guard between models of one text only (section 5.2); training time and lessons on text 2: measured at the gate (Task 19).
- **Measured at the gate (Task 19):** a first AI trains on text 2 in 49.4 s and 47.6 s (target: at most 55 s; nothing else running). Lessons on text 2: the new first AI answers 10 of 10 after one Short run at the game's 5% share (10 of 10 typed casually; 1 of 10 before). The run, read with the game's even mix today, raises the AI's loss on text 2's conversations (0.91 to 1.02), because the parent read 30% conversations and the game's default reads 12.5%: the release's new default reading removes that mismatch.

## 6. A voice of its own
- **Told its name (Tasks 16, 17):** the game sends each AI's public name and its company with every job that makes it speak (chat, customer feedback, keynote, each side of a battle and of a minigame, app reviews); an AI not launched yet is told its version label, the only name its owner sees for it. The helper puts "You are {name}, an AI made by {company}." in front of the conversation only when the AI's last run read text 2 (`persona.learned_identity_line`); text-1 AIs are never given a line. Rivals trained on text 2 are told "You are {product}, an AI." for their stored answers (a rival pack does not know the company a world will give it).
- **Blocking repeats and unlikely words (Task 18): removed.** Built off by default (9ec58de) and tried on the round-1 text-2 models, the two first AIs and the Small, 100 small-talk replies each (`docs/handoff/next-level/decode_trial.py`):

| well-formed replies | first AI, seed 1 | first AI, seed 2 | Small |
|---|---|---|---|
| as served (top-p 0.9) | 79% | 81% | 85% |
| + a 4-token repeat blocked | 82% | 83% | 84% |
| min-p 0.05 instead of top-p | 84% | 78% | 81% |
| min-p 0.1 instead of top-p | 86% | 76% | 78% |

  The rule was a rise of at least 3 points; no setting reaches it on all three models (the changes are within the 5-to-10-point spread between seeds, and they point both ways), fits and on-topic rates barely move, and loops are already stopped by the served loop check. The code was removed again, as the plan requires when nothing is adopted.


## 7. Brain design 2 (plan B, spec §5.5): dropped
**What was tried (2026-11-22, RTX 3050, engine 3, training text 2 in `data2/`).** The lab model `LabTransformer` (`docs/handoff/next-level/arch_lab.py`, a8c9e0f and 899da37) switches on each of the five changes of spec §5.5 alone and then all together: output projections that start at zero (`zero`), normalised queries and keys with a learned scale (`qk`), a norm right after the embedding (`emb`), a two-matrix squared-ReLU feed-forward of the same size in place of SwiGLU (`relu2`), RoPE base 500 (`rope`), all five (`all`). Each design trained a brand-new Tiny, Small and Medium brain, two seeds each, for the same wall time as today's design (`base`) at 12.6M, 12.6M and 25.2M tokens: a design's tokens are today's tokens times its measured speed share. The first attempt (2026-11-17) stopped when the session ended during the Medium timing and was run again from its cached timings. Command: from `ml/`, `AIRACE_DATA=../data2 python -m uv run python ../docs/handoff/next-level/arch_lab.py --out runs/arch-lab --bench base all`; results in `ml/runs/arch-lab/results.json` (not committed).

**Speed (tokens per second of training, 80 timed steps after 20 warm-up steps; today's design timed before and after the others).** The owner's own GPU work did not disturb the measurements: today's design measured 297,043 then 296,897 (Tiny), 175,341 then 174,407 (Small), 105,853 then 105,791 (Medium), at most 0.5% apart.

| design | Tiny | share | Small | share | Medium | share |
|---|---|---|---|---|---|---|
| base | 296,970 | 100% | 174,874 | 100% | 105,822 | 100% |
| zero | 296,808 | 99.9% | 174,617 | 99.9% | 106,539 | 100.7% |
| qk | 240,970 | 81.1% | 135,029 | 77.2% | 80,983 | 76.5% |
| emb | 293,575 | 98.9% | 173,616 | 99.3% | 105,861 | 100.0% |
| relu2 | 257,497 | 86.7% | 148,632 | 85.0% | 88,158 | 83.3% |
| rope | 296,724 | 99.9% | 174,464 | 99.8% | 105,898 | 100.1% |
| all | 214,320 | 72.2% | 117,955 | 67.5% | 68,388 | 64.6% |

**Held-out loss at equal time (text 2; mean of the two seeds, in brackets the difference to base; lower is better).**

| design | Tiny | Small | Medium |
|---|---|---|---|
| base | 2.3442 | 2.2124 | 2.0169 |
| zero | 2.3540 (+0.0098) | 2.2138 (+0.0015) | 2.0142 (−0.0028) |
| qk | 2.3777 (+0.0335) | 2.2312 (+0.0188) | 2.0369 (+0.0200) |
| emb | 2.3565 (+0.0122) | 2.2213 (+0.0090) | 2.0225 (+0.0056) |
| relu2 | 2.3699 (+0.0257) | 2.2380 (+0.0256) | 2.0374 (+0.0204) |
| rope | 2.3444 (+0.0001) | 2.2308 (+0.0184) | 2.0214 (+0.0045) |
| all | 2.4015 (+0.0573) | 2.3010 (+0.0886) | 2.0826 (+0.0656) |

The two seeds of today's design differ by 0.003 (Tiny), 0.011 (Small) and 0.006 (Medium). `all` is worse than both seeds of base in each of its six runs (Tiny 2.380 and 2.423, Small 2.285 and 2.317, Medium 2.077 and 2.088). The only design ahead anywhere, `zero` on Medium, is ahead by 0.003, inside the spread between seeds. The loss on the yardstick's text-1 lines points the same way for `all` (+0.025, +0.018, +0.022); for single changes it moves by up to 0.08 between seeds on Medium and decides nothing.

**Bench v1 (overall, mean of two seeds).** base 48.9, 56.2, 64.2; all 46.5 (−2.4), 52.4 (−3.8), 62.5 (−1.8) on Tiny, Small and Medium.

**Gate lines as printed:**
- all together: better by ≥ 0.04 held-out loss on every size: no; by ≥ 3 bench points on every size: no
- zero, qk, emb, relu2, rope: better by ≥ 0.04 held-out loss on every size: no

**Decision: dropped; today's design stays.** On a card like the owner's, each change either slows every step more than it helps each step (normalised queries and keys by about a fifth, the squared-ReLU feed-forward by about a sixth, all five by about a third) or keeps the speed and moves held-out loss by no more than the difference between two seeds, so together they lose on every size and every seed. Nothing of design 2 enters the product: no `arch: 2`, no change to the helper or the engine; the lab model stays in `docs/handoff/next-level/arch_lab.py` as the record. Not measured, and not a reason to try again: whether the changes are better per token (in the game a run's length is real time on a player's card, so equal time is the fair comparison). Plan C, the release (spec §5.6), is next.
