# The feasibility gate with training engine 2 (2026-11-12)

The gate of Milestone 1 (`airace-ml gate --out runs/gate-2 --cpu-speed`) run again after training engine 2
(ADR 0009). Read with `docs/reports/m1-feasibility.md`, the same gate with engine 1. G2 is measured as the
game draws an AI's replies; at the earlier drawing settings it was 105 of 180 (58%).

Seeds: 1, 2, 3. Device: cuda.

## Summary

**5 of 7 criteria passed.**

| ID | Criterion | Result | Detail |
|---|---|---|---|
| G1 | Speed | PASS | trained on the GPU in 50.4 s (at most 90 s); trained on the CPU in 1028.3 s (at most 1200 s); full benchmark suite 4.4 s (at most 30 s); chat first token 8 ms (at most 300 ms); chat throughput 126 tokens/s (at least 50 tokens/s); growth op 0.08 s (at most 2 s) |
| G2 | Legibility | PASS | replies well-formed: 144 of 180 (80%) over all 3 seeds (46 of 60, 49 of 60, 49 of 60); needs 60% |
| G3 | Differentiation | FAIL | creative-heavy creativity 35.2 vs conversations-heavy 4.7 (lead +30.6, needs more than 9.4); code-heavy coding 53.3 vs facts-heavy 37.8 (lead +15.6, needs more than 4.8); facts-heavy knowledge 71.2 vs conversations-heavy 63.5 (lead +7.8, needs more than 10.4); conversations-heavy instruction 69.3 vs facts-heavy 54.8 (lead +14.5, needs more than 8.2) |
| G4 | Seed variation | PASS | the same model measured twice: mean distance 2.000 (the measuring noise); same mix, different seeds: 3.019 (needs more than the noise and more than 0.05); different mixes: 5.563 (seeds must differ less) |
| G5 | Growth | PASS | growing moved no output by more than 1.2e-05 (allowed 0.0001); after the same extra training the held-out loss is 2.339 grown vs 2.385 not grown (means over 3 seeds); the grown model gains +0.035, +0.047, +0.055 (+0.045 on average), which must be more than 0.019, twice the seed spread |
| G6 | Forgetting | PASS | creativity + language 102.6 for the story-heavy model, 4.7 after code-only training (a drop of 97.9 on average over 3 seeds; needs 3 or more and more than 18.4, twice the seed spread), 96.2 with 30% replay (won back 94% of the drop; needs 50% and more than 12.5 points, twice the spread) |
| G7 | Preparation | FAIL | garble-like words 2.2% after thorough cleaning vs 2.4% after light cleaning; false facts preferred 47.7% with fact-checking vs 71.1% without (means over 3 seeds; garble-like words must fall to 90% or less, false facts to 80% or less, each on every seed and by more than twice its seed spread: garble-like words 0.2% against 0.2%, false facts 23.4% against 6.2%). In the prepared web training text it… |

## Measurements

### G1 Speed: PASS

trained on the GPU in 50.4 s (at most 90 s); trained on the CPU in 1028.3 s (at most 1200 s); full benchmark suite 4.4 s (at most 30 s); chat first token 8 ms (at most 300 ms); chat throughput 126 tokens/s (at least 50 tokens/s); growth op 0.08 s (at most 2 s)

| Measure | Value |
|---|---|
| gpu\_seconds | 50.4 |
| gpu\_seconds\_target | 90 |
| cpu\_seconds | 1028 |
| cpu\_seconds\_target | 1200 |
| bench\_seconds | 4.38 |
| bench\_seconds\_target | 30 |
| first\_token\_ms | 8.177 |
| first\_token\_ms\_target | 300 |
| tokens\_per\_second | 126.5 |
| tokens\_per\_second\_target | 50 |
| growth\_seconds | 0.08274 |
| growth\_seconds\_target | 2 |

### G2 Legibility: PASS

replies well-formed: 144 of 180 (80%) over all 3 seeds (46 of 60, 49 of 60, 49 of 60); needs 60%

| Measure | Value |
|---|---|
| well\_formed | 144 |
| replies | 180 |
| rate | 0.8 |
| min\_rate | 0.6 |
| well\_formed\_by\_seed | 46, 49, 49 |
| replies\_by\_seed | 60, 60, 60 |
| rate\_by\_seed | 0.7667, 0.8167, 0.8167 |
| seeds | 1, 2, 3 |

### G3 Differentiation: FAIL

creative-heavy creativity 35.2 vs conversations-heavy 4.7 (lead +30.6, needs more than 9.4); code-heavy coding 53.3 vs facts-heavy 37.8 (lead +15.6, needs more than 4.8); facts-heavy knowledge 71.2 vs conversations-heavy 63.5 (lead +7.8, needs more than 10.4); conversations-heavy instruction 69.3 vs facts-heavy 54.8 (lead +14.5, needs more than 8.2)

|  | category | mean | std | seeds | rival | rival\_mean | rival\_std | lead | needed\_lead | leads |
|---|---|---|---|---|---|---|---|---|---|---|
| creative | creativity | 35.25 | 4.722 | 3 | conversations | 4.686 | 2.063 | 30.56 | 9.444 | yes |
| code | coding | 53.33 | 2.404 | 3 | facts | 37.78 | 1.388 | 15.56 | 4.807 | yes |
| facts | knowledge | 71.21 | 4.714 | 3 | conversations | 63.46 | 5.19 | 7.752 | 10.38 | no |
| conversations | instruction | 69.31 | 3.43 | 3 | facts | 54.79 | 4.122 | 14.52 | 8.244 | yes |

### G4 Seed variation: PASS

the same model measured twice: mean distance 2.000 (the measuring noise); same mix, different seeds: 3.019 (needs more than the noise and more than 0.05); different mixes: 5.563 (seeds must differ less)

| Measure | Value |
|---|---|
| noise\_mean | 2 |
| seed\_mean | 3.019 |
| mix\_mean | 5.563 |
| min\_seed\_mean | 0.05 |
| noise\_distances | 1.977, 2.12, 1.904 |
| seed\_distances | 2.721, 2.973, 3.362 |
| mix\_distances | 5.808, 5.738, 6.9, 3.873, 5.889, 5.171 |

### G5 Growth: PASS

growing moved no output by more than 1.2e-05 (allowed 0.0001); after the same extra training the held-out loss is 2.339 grown vs 2.385 not grown (means over 3 seeds); the grown model gains +0.035, +0.047, +0.055 (+0.045 on average), which must be more than 0.019, twice the seed spread

| Measure | Value |
|---|---|
| max\_logit\_diff | 1.192e-05 |
| max\_allowed | 0.0001 |
| grown\_loss | 2.339 |
| ungrown\_loss | 2.385 |
| gain | 0.04548 |
| gain\_spread | 0.009749 |
| needed\_gain | 0.0195 |
| max\_logit\_diff\_by\_seed | 1.192e-05, 1.097e-05, 1.091e-05 |
| grown\_loss\_by\_seed | 2.35, 2.331, 2.338 |
| ungrown\_loss\_by\_seed | 2.385, 2.378, 2.392 |
| gain\_by\_seed | 0.0352, 0.04666, 0.05459 |
| growth\_seconds | 0.08274 |
| seeds | 1, 2, 3 |

### G6 Forgetting: PASS

creativity + language 102.6 for the story-heavy model, 4.7 after code-only training (a drop of 97.9 on average over 3 seeds; needs 3 or more and more than 18.4, twice the seed spread), 96.2 with 30% replay (won back 94% of the drop; needs 50% and more than 12.5 points, twice the spread)

| Measure | Value |
|---|---|
| base.creativity | 32.46, 40.7, 32.58 |
| base.language | 64, 70, 68 |
| no\_replay.creativity | 0, 0, 0 |
| no\_replay.language | 7, 3, 4 |
| replay.creativity | 34.96, 30.23, 25.48 |
| replay.language | 66, 69, 63 |
| drop | 97.91 |
| drop\_by\_seed | 89.46, 107.7, 96.58 |
| drop\_spread | 9.192 |
| recovered | 91.56 |
| recovered\_by\_seed | 93.96, 96.23, 84.48 |
| recovered\_spread | 6.233 |
| recovery | 0.9351 |
| min\_drop | 3 |
| min\_recovery | 0.5 |
| seeds | 1, 2, 3 |

### G7 Preparation: FAIL

garble-like words 2.2% after thorough cleaning vs 2.4% after light cleaning; false facts preferred 47.7% with fact-checking vs 71.1% without (means over 3 seeds; garble-like words must fall to 90% or less, false facts to 80% or less, each on every seed and by more than twice its seed spread: garble-like words 0.2% against 0.2%, false facts 23.4% against 6.2%). In the prepared web training text itself, thorough cleaning leaves 0.65 of light cleaning's garble-like words (1.8% vs 2.8%).

| Measure | Value |
|---|---|
| garble\_light | 0.02418 |
| garble\_thorough | 0.02239 |
| false\_unchecked | 0.7111 |
| false\_checked | 0.4767 |
| garble\_max\_ratio | 0.9 |
| false\_max\_ratio | 0.8 |
| garble\_ratio | 0.926 |
| false\_ratio | 0.6703 |
| garble\_light\_by\_seed | 0.02183, 0.02478, 0.02591 |
| garble\_thorough\_by\_seed | 0.01974, 0.02214, 0.02529 |
| false\_unchecked\_by\_seed | 0.73, 0.7167, 0.6867 |
| false\_checked\_by\_seed | 0.4867, 0.4567, 0.4867 |
| garble\_fall | 0.00179 |
| garble\_fall\_spread | 0.001044 |
| false\_fall | 0.2344 |
| false\_fall\_spread | 0.03097 |
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
| starter | completed | 1536/1536 | 50.4 | 2.489 | yes |
| starter-cpu | completed | 1536/1536 | 1028 | 2.48 | yes |
| starter-s2 | completed | 1536/1536 | 56.33 | 2.471 | yes |
| starter-s3 | completed | 1536/1536 | 86.08 | 2.495 | yes |
| g3-creative-s1 | completed | 1024/1024 | 71.38 | 2.544 | yes |
| g3-creative-s2 | completed | 1024/1024 | 58.87 | 2.539 | yes |
| g3-creative-s3 | completed | 1024/1024 | 59.69 | 2.53 | yes |
| g3-code-s1 | completed | 1024/1024 | 59.72 | 2.685 | yes |
| g3-code-s2 | completed | 1024/1024 | 59.74 | 2.637 | yes |
| g3-code-s3 | completed | 1024/1024 | 59.76 | 2.64 | yes |
| g3-facts-s1 | completed | 1024/1024 | 58.4 | 2.639 | yes |
| g3-facts-s2 | completed | 1024/1024 | 56.27 | 2.641 | yes |
| g3-facts-s3 | completed | 1024/1024 | 56.25 | 2.637 | yes |
| g3-conversations-s1 | completed | 1024/1024 | 58.15 | 2.563 | yes |
| g3-conversations-s2 | completed | 1024/1024 | 58.38 | 2.554 | yes |
| g3-conversations-s3 | completed | 1024/1024 | 61.74 | 2.535 | yes |
| g4-balanced-s1 | completed | 1024/1024 | 56.31 | 2.441 | yes |
| g4-balanced-s2 | completed | 1024/1024 | 56.74 | 2.468 | yes |
| g4-balanced-s3 | completed | 1024/1024 | 57 | 2.452 | yes |
| g5-grown-s1 | completed | 1536/1536 | 81.08 | 2.35 | yes |
| g5-ungrown-s1 | completed | 1536/1536 | 48.93 | 2.385 | yes |
| g5-grown-s2 | completed | 1536/1536 | 80.78 | 2.331 | yes |
| g5-ungrown-s2 | completed | 1536/1536 | 49.33 | 2.378 | yes |
| g5-grown-s3 | completed | 1536/1536 | 82.16 | 2.338 | yes |
| g5-ungrown-s3 | completed | 1536/1536 | 48.71 | 2.392 | yes |
| g6-code-replay0-s1 | completed | 1536/1536 | 80.43 | 7.396 | yes |
| g6-code-replay30-s1 | completed | 1536/1536 | 80.37 | 2.594 | yes |
| g6-code-replay0-s2 | completed | 1536/1536 | 80.06 | 7.423 | yes |
| g6-code-replay30-s2 | completed | 1536/1536 | 81.66 | 2.573 | yes |
| g6-code-replay0-s3 | completed | 1536/1536 | 80.31 | 7.778 | yes |
| g6-code-replay30-s3 | completed | 1536/1536 | 80.36 | 2.577 | yes |
| g7-web-light-s1 | completed | 3072/3072 | 92.36 | 2.439 | yes |
| g7-web-thorough-s1 | completed | 3072/3072 | 89.9 | 2.499 | yes |
| g7-facts-unchecked-s1 | completed | 3072/3072 | 88.98 | 4.974 | yes |
| g7-facts-checked-s1 | completed | 3072/3072 | 88.9 | 5.135 | yes |
| g7-web-light-s2 | completed | 3072/3072 | 90.31 | 2.465 | yes |
| g7-web-thorough-s2 | completed | 3072/3072 | 89.86 | 2.513 | yes |
| g7-facts-unchecked-s2 | completed | 3072/3072 | 89.15 | 4.968 | yes |
| g7-facts-checked-s2 | completed | 3072/3072 | 89.05 | 5.101 | yes |
| g7-web-light-s3 | completed | 3072/3072 | 90.19 | 2.448 | yes |
| g7-web-thorough-s3 | completed | 3072/3072 | 90.68 | 2.519 | yes |
| g7-facts-unchecked-s3 | completed | 3072/3072 | 89.13 | 4.904 | yes |
| g7-facts-checked-s3 | completed | 3072/3072 | 89.09 | 5.061 | yes |

## Transcripts

### G2: the first model's replies to probe prompts

The first 10 of 20 exchanges.

| # | Prompt | Reply |
|---|---|---|
| 1 | Hello! How are you today? | I'm doing well, thank you. How are you? |
| 2 | What did you do this morning? | I'm doing well, thank you. |
| 3 | Tell me something about yourself. | It's okay. |
| 4 | What is your favorite thing to do? | The symbol for antimony is Si. |
| 5 | What makes a good friend? | One word only: What is the capital of the Bahamas? |
| 6 | Tell me about a place you would like to visit. | It is road. |
| 7 | What is the best part of the day? | The month after July is August. |
| 8 | Why do people like to play games? | I'm really excited to hear how they get around. |
| 9 | What do you think about when it rains? | I'm feeling really uncreasonable. I'm so glad we could get started. |
| 10 | Can you tell me about your family? | You can see the police officer in China. |

### G3: each contrasting mix's model on the same prompts

| # | Prompt | Reply |
|---|---|---|
| 1 | \[creative-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time there was a very curious balloon. It was baking a spoon. It was not a big balloon. It could not move its shoulders.<br><br>One day, a little girl came along. She had a dress on her mouth. She took a hand of it and gently |
| 2 | \[code-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a big cat named Fluffy. Py loved to play with the balloon. One day, they decided to go to the park.<br><br>Sub soon, Lily saw a big sparrow in the land. "Let's look at the floor," said Max. |
| 3 | \[facts-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a little girl named Lily. She loved to play outside in the garden with her mom. One day, she would go to the park and see a big bird. Lily felt very happy and she picked it up and jumped on it. Lily explained, "Mommy, I wanted |
| 4 | \[conversations-heavy\] Tell me a tiny story about a lost balloon. | Once upon a time, there was a little boy named Timmy. Timmy loved to play outside and play with his toys and friends. One day, Timmy's mom found his toys and noticed a big dog. Timmy's mom said, "That's a big dog! Let me help you, Timmy! What do you want |
| 5 | \[creative-heavy\] Which city is the capital of France? | The main language spoken in France is Spanish. |
| 6 | \[code-heavy\] Which city is the capital of France? | The capital of France is Vientiane. |
| 7 | \[facts-heavy\] Which city is the capital of France? | I do. Asinau is the capital of France. |
| 8 | \[conversations-heavy\] Which city is the capital of France? | The capital of France is Paris. |

## Differentiation models: benchmark scores

0 is no better than guessing, 100 is perfect.

| Category | creative-heavy (g3-creative-s1) | code-heavy (g3-code-s1) | facts-heavy (g3-facts-s1) | conversations-heavy (g3-conversations-s1) |
|---|---|---|---|---|
| language | 64 | 49 | 54 | 61 |
| reasoning | 6.182 | 12.73 | 8.364 | 7.636 |
| pattern | 79.33 | 84 | 75.33 | 61.33 |
| knowledge | 24.25 | 27.57 | 75.42 | 57.48 |
| coding | 35.33 | 56 | 36.67 | 33.33 |
| creativity | 32.46 | 1.35 | 2.82 | 4.3 |
| consistency | 26.67 | 26.67 | 86.67 | 86.67 |
| instruction | 36.63 | 41.58 | 51.49 | 71.29 |

## Personalities

What stands out in each model's measured fingerprint, compared with all the gate's models.

| Model | Described as |
|---|---|
| g3-creative-s1 | talkative, inventive, changeable, timid, formal, repetitive |
| g3-creative-s2 | hesitant, predictable, vague on facts, timid, repetitive |
| g3-creative-s3 | vague on facts, reserved, repetitive |
| g3-code-s1 | often slips, warm |
| g3-code-s2 | talkative, hesitant, formal |
| g3-code-s3 | hesitant, changeable, timid, often slips, warm, varied |
| g3-facts-s1 | inventive, changeable, casual, warm, varied |
| g3-facts-s2 | predictable, varied |
| g3-facts-s3 | terse, casual |
| g3-conversations-s1 | talkative, self-assured, steady, rarely slips |
| g3-conversations-s2 | self-assured, predictable, steady, rarely slips |
| g3-conversations-s3 | self-assured, bold |
| g4-balanced-s1 | terse, inventive, bold, reserved |
| g4-balanced-s2 | steady, bold, rarely slips, reserved |
| g4-balanced-s3 | terse, formal |

## Iterations

_None yet. Each attempt to fix a failing criterion is recorded here: the hypothesis, the change and the result._
