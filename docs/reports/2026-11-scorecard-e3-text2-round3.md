# Scorecard

Data root: ..\data2; yardstick root: ..\data; engine 3; 2026-11-17

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.342 | 1.224 | 2.827 | 49.0 | 79% | 38% | 51% | 27% | 82% | 67% | 39% | 93% | 0% | 4% | 0% |
| first-s2 | first | 2.346 | 1.238 | 2.822 | 48.6 | 75% | 48% | 56% | 35% | 81% | 65% | 68% | 100% | 0% | 4% | 0% |
| small | small | 2.211 | 1.095 | 2.739 | 58.0 | 79% | 41% | 58% | 29% | 90% | 71% | 70% | 91% | 0% | 4% | 0% |
| lineage-small | lineage | 2.212 | 1.138 | 2.805 | 55.0 | 79% | 50% | 64% | 40% | 88% | 72% | 70% | 97% | 0% | 2% | 0% |
| lineage-medium | lineage | 2.126 | 1.046 | 2.765 | 60.2 | 82% | 47% | 65% | 40% | 93% | 73% | 94% | 96% | 0% | 5% | 0% |
| chatty | chatty | 2.457 | 0.930 | 3.168 | 48.8 | 78% | 47% | 75% | 43% | 97% | 86% | 52% | 100% | 0% | 4% | 0% |
| large | large | 1.948 | 0.903 | 2.661 | 68.1 | 73% | 43% | 76% | 34% | 97% | 89% | 87% | 96% | 0% | 4% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
