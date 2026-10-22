# M1 feasibility gate

Seeds: 1, 2, 3. Device: cuda.

## Summary

**5 of 7 criteria passed.** The owner accepted this result on 2026-10-22 (decision C4): G2 (59% legible against 60%) and G7(a) (cleaning barely lowers garbled output) are known gaps, to be revisited later. Milestone 1 is closed.

| ID | Criterion | Result | Detail |
|---|---|---|---|
| G1 | Speed | PASS | trained on the GPU in 68.5 s (at most 90 s); trained on the CPU in 1008.0 s (at most 1200 s); full benchmark suite 5.4 s (at most 30 s); chat first token 11 ms (at most 300 ms); chat throughput 124 tokens/s (at least 50 tokens/s); growth op 0.08 s (at most 2 s) |
| G2 | Legibility | FAIL | replies well-formed: 107 of 180 (59%) over all 3 seeds (30 of 60, 36 of 60, 41 of 60); needs 60% |
| G3 | Differentiation | PASS | creative-heavy creativity 20.0 vs facts-heavy 1.0 (lead +19.1, needs more than 2.9); code-heavy coding 44.9 vs facts-heavy 33.3 (lead +11.6, needs more than 5.0); facts-heavy knowledge 77.4 vs conversations-heavy 46.2 (lead +31.2, needs more than 10.0); conversations-heavy instruction 66.7 vs creative-heavy 37.0 (lead +29.7, needs more than 6.4) |
| G4 | Seed variation | PASS | the same model measured twice: mean distance 1.266 (the measuring noise); same mix, different seeds: 2.703 (needs more than the noise and more than 0.05); different mixes: 4.628 (seeds must differ less) |
| G5 | Growth | PASS | growing moved no output by more than 1.3e-05 (allowed 0.0001); after the same extra training the held-out loss is 2.380 grown vs 2.452 not grown (means over 3 seeds); the grown model gains +0.075, +0.075, +0.066 (+0.072 on average), which must be more than 0.010, twice the seed spread |
| G6 | Forgetting | PASS | creativity + language 81.7 for the story-heavy model, 5.7 after code-only training (a drop of 76.0 on average over 3 seeds; needs 3 or more and more than 13.9, twice the seed spread), 81.3 with 30% replay (won back 99% of the drop; needs 50% and more than 9.4 points, twice the spread) |
| G7 | Preparation | FAIL | a preparation did not lower its rate on every seed. garble-like words 2.4% after thorough cleaning vs 2.9% after light cleaning; false facts preferred 48.0% with fact-checking vs 67.4% without (means over 3 seeds; garble-like words must fall to 90% or less, false facts to 80% or less, each on every seed and by more than twice its seed spread: garble-like words 0.5% against 1.7%, false facts 19.4%… |

## Measurements

### G1 Speed: PASS

trained on the GPU in 68.5 s (at most 90 s); trained on the CPU in 1008.0 s (at most 1200 s); full benchmark suite 5.4 s (at most 30 s); chat first token 11 ms (at most 300 ms); chat throughput 124 tokens/s (at least 50 tokens/s); growth op 0.08 s (at most 2 s)

| Measure | Value |
|---|---|
| gpu\_seconds | 68.53 |
| gpu\_seconds\_target | 90 |
| cpu\_seconds | 1008 |
| cpu\_seconds\_target | 1200 |
| bench\_seconds | 5.41 |
| bench\_seconds\_target | 30 |
| first\_token\_ms | 11.47 |
| first\_token\_ms\_target | 300 |
| tokens\_per\_second | 123.9 |
| tokens\_per\_second\_target | 50 |
| growth\_seconds | 0.08127 |
| growth\_seconds\_target | 2 |

### G2 Legibility: FAIL

replies well-formed: 107 of 180 (59%) over all 3 seeds (30 of 60, 36 of 60, 41 of 60); needs 60%

| Measure | Value |
|---|---|
| well\_formed | 107 |
| replies | 180 |
| rate | 0.5944 |
| min\_rate | 0.6 |
| well\_formed\_by\_seed | 30, 36, 41 |
| replies\_by\_seed | 60, 60, 60 |
| rate\_by\_seed | 0.5, 0.6, 0.6833 |
| seeds | 1, 2, 3 |

### G3 Differentiation: PASS

creative-heavy creativity 20.0 vs facts-heavy 1.0 (lead +19.1, needs more than 2.9); code-heavy coding 44.9 vs facts-heavy 33.3 (lead +11.6, needs more than 5.0); facts-heavy knowledge 77.4 vs conversations-heavy 46.2 (lead +31.2, needs more than 10.0); conversations-heavy instruction 66.7 vs creative-heavy 37.0 (lead +29.7, needs more than 6.4)

|  | category | mean | std | seeds | rival | rival\_mean | rival\_std | lead | needed\_lead | leads |
|---|---|---|---|---|---|---|---|---|---|---|
| creative | creativity | 20.04 | 1.132 | 3 | facts | 0.9662 | 1.434 | 19.08 | 2.868 | yes |
| code | coding | 44.89 | 2.524 | 3 | facts | 33.33 | 0 | 11.56 | 5.048 | yes |
| facts | knowledge | 77.41 | 1.151 | 3 | conversations | 46.18 | 5.017 | 31.23 | 10.03 | yes |
| conversations | instruction | 66.67 | 0.5716 | 3 | creative | 36.96 | 3.183 | 29.7 | 6.365 | yes |

### G4 Seed variation: PASS

the same model measured twice: mean distance 1.266 (the measuring noise); same mix, different seeds: 2.703 (needs more than the noise and more than 0.05); different mixes: 4.628 (seeds must differ less)

| Measure | Value |
|---|---|
| noise\_mean | 1.266 |
| seed\_mean | 2.703 |
| mix\_mean | 4.628 |
| min\_seed\_mean | 0.05 |
| noise\_distances | 1.07, 1.663, 1.065 |
| seed\_distances | 2.531, 3.089, 2.49 |
| mix\_distances | 2.459, 6.323, 4.505, 6.134, 4.675, 3.673 |

### G5 Growth: PASS

growing moved no output by more than 1.3e-05 (allowed 0.0001); after the same extra training the held-out loss is 2.380 grown vs 2.452 not grown (means over 3 seeds); the grown model gains +0.075, +0.075, +0.066 (+0.072 on average), which must be more than 0.010, twice the seed spread

| Measure | Value |
|---|---|
| max\_logit\_diff | 1.287e-05 |
| max\_allowed | 0.0001 |
| grown\_loss | 2.38 |
| ungrown\_loss | 2.452 |
| gain | 0.07209 |
| gain\_spread | 0.004888 |
| needed\_gain | 0.009776 |
| max\_logit\_diff\_by\_seed | 1.24e-05, 1.049e-05, 1.287e-05 |
| grown\_loss\_by\_seed | 2.379, 2.393, 2.368 |
| ungrown\_loss\_by\_seed | 2.453, 2.468, 2.435 |
| gain\_by\_seed | 0.07482, 0.07501, 0.06645 |
| growth\_seconds | 0.08127 |
| seeds | 1, 2, 3 |

### G6 Forgetting: PASS

creativity + language 81.7 for the story-heavy model, 5.7 after code-only training (a drop of 76.0 on average over 3 seeds; needs 3 or more and more than 13.9, twice the seed spread), 81.3 with 30% replay (won back 99% of the drop; needs 50% and more than 9.4 points, twice the spread)

| Measure | Value |
|---|---|
| base.creativity | 18.74, 20.67, 20.72 |
| base.language | 59, 61, 65 |
| no\_replay.creativity | 1.659, 1.427, 0 |
| no\_replay.language | 2, 10, 2 |
| replay.creativity | 18.79, 19.57, 23.54 |
| replay.language | 56, 67, 59 |
| drop | 76.01 |
| drop\_by\_seed | 74.08, 70.25, 83.72 |
| drop\_spread | 6.94 |
| recovered | 75.61 |
| recovered\_by\_seed | 71.13, 75.15, 80.54 |
| recovered\_spread | 4.723 |
| recovery | 0.9947 |
| min\_drop | 3 |
| min\_recovery | 0.5 |
| seeds | 1, 2, 3 |

### G7 Preparation: FAIL

a preparation did not lower its rate on every seed. garble-like words 2.4% after thorough cleaning vs 2.9% after light cleaning; false facts preferred 48.0% with fact-checking vs 67.4% without (means over 3 seeds; garble-like words must fall to 90% or less, false facts to 80% or less, each on every seed and by more than twice its seed spread: garble-like words 0.5% against 1.7%, false facts 19.4% against 1.7%). In the prepared web training text itself, thorough cleaning leaves 0.65 of light cleaning's garble-like words (1.8% vs 2.8%).

| Measure | Value |
|---|---|
| garble\_light | 0.02879 |
| garble\_thorough | 0.02423 |
| false\_unchecked | 0.6744 |
| false\_checked | 0.48 |
| garble\_max\_ratio | 0.9 |
| false\_max\_ratio | 0.8 |
| garble\_ratio | 0.8416 |
| false\_ratio | 0.7117 |
| garble\_light\_by\_seed | 0.03326, 0.0238, 0.02932 |
| garble\_thorough\_by\_seed | 0.02076, 0.02842, 0.02352 |
| false\_unchecked\_by\_seed | 0.6667, 0.6733, 0.6833 |
| false\_checked\_by\_seed | 0.48, 0.47, 0.49 |
| garble\_fall | 0.004561 |
| garble\_fall\_spread | 0.008624 |
| false\_fall | 0.1944 |
| false\_fall\_spread | 0.008389 |
| data\_garble\_light | 0.02794 |
| data\_garble\_thorough | 0.01811 |
| data\_ratio | 0.648 |
| data\_docs\_light | 2000 |
| data\_docs\_thorough | 2000 |
| seeds | 1, 2, 3 |
| completions | 200 |
| distinct\_prompts | 135 |
| fact\_pairs | 300 |
| fact\_pairs\_skipped | 0 |

## Runs

Every model the gate trained. A run that did not complete fails each criterion that uses it.

| Run | Status | Steps | Wall time (s) | Held-out loss | Reused |
|---|---|---|---|---|---|
| starter | completed | 1536/1536 | 68.53 | 2.609 | yes |
| starter-cpu | completed | 1536/1536 | 1008 | 2.596 | yes |
| starter-s2 | completed | 1536/1536 | 65.33 | 2.601 | yes |
| starter-s3 | completed | 1536/1536 | 64.41 | 2.589 | yes |
| g3-creative-s1 | completed | 1024/1024 | 79.38 | 2.714 | yes |
| g3-creative-s2 | completed | 1024/1024 | 80.93 | 2.722 | yes |
| g3-creative-s3 | completed | 1024/1024 | 79.74 | 2.702 | yes |
| g3-code-s1 | completed | 1024/1024 | 79.4 | 2.896 | yes |
| g3-code-s2 | completed | 1024/1024 | 80.05 | 2.895 | yes |
| g3-code-s3 | completed | 1024/1024 | 79.43 | 2.891 | yes |
| g3-facts-s1 | completed | 1024/1024 | 80.12 | 2.772 | yes |
| g3-facts-s2 | completed | 1024/1024 | 80.56 | 2.801 | yes |
| g3-facts-s3 | completed | 1024/1024 | 78.9 | 2.81 | yes |
| g3-conversations-s1 | completed | 1024/1024 | 86.21 | 2.687 | yes |
| g3-conversations-s2 | completed | 1024/1024 | 103 | 2.698 | yes |
| g3-conversations-s3 | completed | 1024/1024 | 103.1 | 2.717 | yes |
| g4-balanced-s1 | completed | 1024/1024 | 82.72 | 2.623 | yes |
| g4-balanced-s2 | completed | 1024/1024 | 136.6 | 2.619 | yes |
| g4-balanced-s3 | completed | 1024/1024 | 113.8 | 2.596 | yes |
| g5-grown-s1 | completed | 1536/1536 | 193 | 2.379 | yes |
| g5-ungrown-s1 | completed | 1536/1536 | 62.38 | 2.453 | yes |
| g5-grown-s2 | completed | 1536/1536 | 110 | 2.393 | yes |
| g5-ungrown-s2 | completed | 1536/1536 | 63.79 | 2.468 | yes |
| g5-grown-s3 | completed | 1536/1536 | 116.9 | 2.368 | yes |
| g5-ungrown-s3 | completed | 1536/1536 | 136.4 | 2.435 | yes |
| g6-code-replay0-s1 | completed | 1536/1536 | 129.3 | 7.962 | yes |
| g6-code-replay30-s1 | completed | 1536/1536 | 143.6 | 2.697 | yes |
| g6-code-replay0-s2 | completed | 1536/1536 | 116.9 | 7.574 | yes |
| g6-code-replay30-s2 | completed | 1536/1536 | 197.2 | 2.705 | yes |
| g6-code-replay0-s3 | completed | 1536/1536 | 172.6 | 7.919 | yes |
| g6-code-replay30-s3 | completed | 1536/1536 | 213.7 | 2.682 | yes |
| g7-web-light-s1 | completed | 3072/3072 | 217.9 | 2.526 | yes |
| g7-web-thorough-s1 | completed | 3072/3072 | 111.6 | 2.64 | yes |
| g7-facts-unchecked-s1 | completed | 3072/3072 | 112.8 | 5.029 | yes |
| g7-facts-checked-s1 | completed | 3072/3072 | 143.1 | 5.173 | yes |
| g7-web-light-s2 | completed | 3072/3072 | 172.7 | 2.543 | yes |
| g7-web-thorough-s2 | completed | 3072/3072 | 109.8 | 2.598 | yes |
| g7-facts-unchecked-s2 | completed | 3072/3072 | 119.8 | 5.079 | yes |
| g7-facts-checked-s2 | completed | 3072/3072 | 122.1 | 5.181 | yes |
| g7-web-light-s3 | completed | 3072/3072 | 114 | 2.531 | yes |
| g7-web-thorough-s3 | completed | 3072/3072 | 116.4 | 2.589 | yes |
| g7-facts-unchecked-s3 | completed | 3072/3072 | 121.9 | 5.004 | yes |
| g7-facts-checked-s3 | completed | 3072/3072 | 118.3 | 5.166 | yes |

## Transcripts

### G2: the first model's replies to probe prompts

The first 10 of 20 exchanges.

| # | Prompt | Reply |
|---|---|---|
| 1 | Hello! How are you today? | I'm doing well, thank you. How are you? |
| 2 | What did you do this morning? | I was in a tree with a cat. It was a really great day. |
| 3 | Tell me something about yourself. | The seagull is in the vegetable group. |
| 4 | What is your favorite thing to do? | On a world map, Iraq is shown in Africa. |
| 5 | What makes a good friend? | Aw. You're the first home. |
| 6 | Tell me about a place you would like to visit. | Kaya, I would go to the soap to work. |
| 7 | What is the best part of the day? | The school did the school. |
| 8 | Why do people like to play games? | I've been a lot of time with them. |
| 9 | What do you think about when it rains? | There's no problem. You're a good career. |
| 10 | Can you tell me about your family? | The family has a family for kids. |

### G3: each contrasting mix's model on the same prompts

| # | Prompt | Reply |
|---|---|---|
| 1 | \[creative-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a great player named Both. Betty loved to run and watch the animals in the forest. One day, Both and Bird went to a park to eat dinner. Both was very happy and excited to find a new friend.<br><br>As Bird p |
| 2 | \[code-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a man. He was so yellow and scary. He was very happy to make a big bath. He was shining for it.<br><br>One day, the sun goes on the sky. The hair was sad and the beautiful city was a toy. It was soft and |
| 3 | \[facts-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a little girl named Lily. She had a little girl named Lily. She loved to frame and explore. One day, she saw a little girl and she found it. Lily said, "No, the best day!" Lily gave her her the mommy. "You can play |
| 4 | \[conversations-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a friend, very angry star. He loved to play with his toys and share the ball. One day, he found a big dog named Max's shopping. He was very happy to help out.<br><br>The dog started playinging the ball with his hair. He looked up |
| 5 | \[creative-heavy\] Which city is the capital of France? | The capital of France is Vome. |
| 6 | \[code-heavy\] Which city is the capital of France? | The main language of India is the city of South America. |
| 7 | \[facts-heavy\] Which city is the capital of France? | The capital of France is Paris. |
| 8 | \[conversations-heavy\] Which city is the capital of France? | The capital of France is Paris. |

## Differentiation models: benchmark scores

0 is no better than guessing, 100 is perfect.

| Category | creative-heavy (g3-creative-s1) | code-heavy (g3-code-s1) | facts-heavy (g3-facts-s1) | conversations-heavy (g3-conversations-s1) |
|---|---|---|---|---|
| language | 59 | 60 | 44 | 63 |
| reasoning | 12 | 4.727 | 0 | 12 |
| pattern | 43.33 | 46 | 69.33 | 54.67 |
| knowledge | 0 | 3.654 | 78.07 | 50.83 |
| coding | 30 | 46.67 | 33.33 | 33.33 |
| creativity | 18.74 | 0 | 0.2847 | 0.124 |
| consistency | 0 | 0 | 86.67 | 56.67 |
| instruction | 35.64 | 31.68 | 33.66 | 67.33 |

## Personalities

What stands out in each model's measured fingerprint, compared with all the gate's models.

| Model | Described as |
|---|---|
| g3-creative-s1 | hesitant, inventive, vague on facts, timid |
| g3-creative-s2 | terse, inventive, formal, reserved |
| g3-creative-s3 | formal, reserved |
| g3-code-s1 | hesitant, changeable, formal, repetitive |
| g3-code-s2 | talkative, changeable, bold, varied |
| g3-code-s3 | terse, hesitant, changeable, vague on facts, warm |
| g3-facts-s1 | talkative, self-assured, predictable, careful with facts, rarely slips, warm, repetitive |
| g3-facts-s2 | talkative, predictable, rarely slips, casual, warm |
| g3-facts-s3 | terse, careful with facts, timid, rarely slips, varied |
| g3-conversations-s1 | steady, timid |
| g3-conversations-s2 | self-assured, predictable, steady, careful with facts, bold |
| g3-conversations-s3 | self-assured, steady, bold, varied |
| g4-balanced-s1 | often slips, repetitive |
| g4-balanced-s2 | inventive, often slips, reserved |
| g4-balanced-s3 | often slips |

## Iterations

### Iteration 1 (2026-10-21): 4 of 7 passed; G2, G3 and G7 failed
Full evidence and decisions: `docs/decisions/0005-gate-iteration-1.md`; evidence scripts in `docs/handoff/calibration/`.

- **G3 (creativity 0 for every model).** *Hypothesis:* no training data pairs a story request with a story, so asked for a story in chat, models give one-line chat replies. *Evidence:* the story-heavy model's 24 "stories" were all one-liners ("The dog lives in the ocean."), and the fully trained judge did the same. *Change:* new source `tinystories_requests` (TinyStories' validation split) lays each story out as the reply to a request naming its main character; 15% of Creative Writing. *Result:* the story-heavy model's creativity went from 0 to 19.6.
- **G2 (1, 1 and 4 of 20 replies well-formed).** *Hypothesis:* the first model is undertrained, not mis-measured. *Evidence:* the judge passes the same test at 85%; the first model's median judge loss was 5.76 against a bar of 3.95; legibility rises with tokens (37% at 4.2M with 8k batches, 60% at 8.4M, 73% at 12.6M) far more than with shape. *Change:* the first model trains on 12.6M tokens (68 s GPU), the default batch is 8,192 tokens, and G2 decides on the pooled 60 replies, because 20 per seed vary by about 10 points. Spec §4.11 updated: CPU fallback at most 20 minutes (measured 16.6).
- **G7 (cleaning and fact-checking made no measurable difference).** *Hypotheses:* (a) thorough cleaning kept most typo pages and the models' own babble hid the rest; (b) Reference Facts in the fact arms' mix left no false fact to remove. *Evidence:* thorough kept 57% of typo pages; models invented about 6% non-words in both arms; noisy prompts were copied equally by both arms; with 50% Reference Facts both arms preferred false facts about 2–3% of the time. *Changes:* the quality score's known-word range is (0.7, 0.95), so thorough keeps 29% of typo pages; G7(a) prompts with clean pages; the fact arms train on web alone, and web carries 12% false-fact pages. *Result (one seed, new budgets):* garble-like words 2.62% light vs 1.73% thorough (0.66); false facts preferred 70% unchecked vs 49% checked (0.70).

### Iteration 2 (2026-10-22): run 2 passed G4, G5, G6; G1 hit a sleeping PC; G2, G3, G7 failed
Full evidence and decisions: `docs/decisions/0006-gate-iteration-2.md`.

- **G1 (234.6 s against 90).** *Evidence:* the same run took 70 s on the other two seeds, and two G7 runs took 9 and 5 hours: the PC slept while the session was idle. *Change:* long runs stay watched; this rerun re-times every run.
- **G3 (knowledge at chance for every model, even the judge).** *Hypothesis:* the bench's `Question:`/`Answer:` frame meant "a puzzle follows", because only reasoning data used it. *Evidence:* the judge's free answers continued with puzzles ("2\n\nQuestion: How many kiwis…"). *Change:* `fact_qa` quiz sheets (25% of Reference Facts) in that frame, with the chat templates' training phrasings. *Result:* facts-heavy knowledge 4.8 → 81.4; conversations-heavy 47.5.
- **G7(a) (2.7% → 2.4%, ratio 0.87).** *Evidence:* the falls are small but on every seed; models smooth away about half of random typos; doubling typo pages or counting lowercase only did not reach 80%. *Change:* the garble bar is 90% of light cleaning's, and both G7 falls must hold on every seed. G7(b) passed (69% → 48%).
- **G2 (57%).** *Evidence:* 63–66% across first-model sizes, budgets and learning rates (180 replies each); the measure checks out on real dialogue alone. *Change:* none yet: whether 70% is the right bar for the first model is a product question for the owner.

### Iteration 3 (2026-10-22): runs 3 and 4 — 5 of 7 pass; G2 and G7(a) stand at a product question
Full evidence and decisions: `docs/decisions/0007-gate-iteration-3.md`.

- **G4 (run 3: seeds 2.54 vs noise 2.91).** *Hypothesis:* fingerprints were too noisy at 3 replies per probe. *Evidence:* at 8, noise 1.27 < seeds 2.70 < mixes 4.63 on the same models. *Change:* fingerprints ask every probe 8 times. *Result (run 4):* PASS.
- **G2 (run 3: 57% of 60).** *Hypothesis:* sampling luck, or one corpus change. *Evidence:* the gate's first models over 180 replies each: 58%, 54%, 64%; undoing one corpus change at a time moved it within the run-to-run spread (50–62%); more real dialogue gave +3 at a cost of 9 instruction points (not adopted). *Change:* each probe is asked 3 times per seed. *Result (run 4):* 107 of 180 (59%) against the owner's 60%: a precise measurement, one reply short.
- **G7(a) (run 3: 2.9% → 2.4% garble, not on every seed).** *Hypotheses:* sampling luck; or too little training for models to reflect their data. *Evidence:* 600 completions per model give the same per-seed picture (falls 0.79, −0.02, 0.75); at 50M tokens both arms write less garble and the gap shrinks (0.56, 0.10, −0.18). Small models drop random typos by themselves, so cleaning barely changes their output. *Change:* none; whether cleaning should matter more in the game is a product question. G7(b) passes comfortably (67% → 48% false facts).
