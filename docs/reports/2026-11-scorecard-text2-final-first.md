# Scorecard

Data root: ..\data2; yardstick root: ..\data; engine 3; 2026-11-17

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.363 | 1.018 | 2.958 | 52.6 | 69% | 42% | 62% | 31% | 93% | 78% | 54% | 98% | 0% | 5% | 0% |
| first-s2 | first | 2.362 | 1.026 | 2.985 | 53.6 | 78% | 49% | 62% | 39% | 91% | 73% | 74% | 89% | 0% | 3% | 1% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
