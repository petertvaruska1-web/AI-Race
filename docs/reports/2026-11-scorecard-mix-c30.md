# Scorecard

Data root: ..\data2; yardstick root: ..\data; engine 3; 2026-11-17

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.376 | 1.140 | 3.031 | 48.8 | 78% | 43% | 68% | 35% | 92% | 81% | 46% | 97% | 0% | 3% | 0% |
| first-s2 | first | 2.387 | 1.169 | 2.962 | 53.3 | 81% | 41% | 64% | 36% | 92% | 75% | 49% | 96% | 0% | 4% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
