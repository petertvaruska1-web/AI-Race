# Scorecard

Data root: ..\data2; yardstick root: ..\data; engine 3; 2026-11-17

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.365 | 1.384 | 2.854 | 47.0 | 82% | 43% | 56% | 33% | 78% | 67% | 44% | 88% | 0% | 2% | 1% |
| first-s2 | first | 2.348 | 1.364 | 2.826 | 48.7 | 75% | 45% | 51% | 36% | 79% | 55% | 58% | 99% | 0% | 4% | 0% |
| small | small | 2.222 | 1.241 | 2.748 | 55.7 | 84% | 49% | 58% | 35% | 87% | 69% | 62% | 92% | 0% | 5% | 1% |
| lineage-small | lineage | 2.216 | 1.224 | 2.796 | 55.5 | 82% | 56% | 67% | 42% | 90% | 72% | 67% | 76% | 0% | 4% | 0% |
| lineage-medium | lineage | 2.134 | 1.136 | 2.794 | 59.9 | 74% | 48% | 62% | 39% | 93% | 82% | 62% | 99% | 0% | 5% | 0% |
| chatty | chatty | 2.485 | 1.023 | 3.117 | 46.7 | 81% | 48% | 72% | 46% | 98% | 88% | 43% | 92% | 0% | 4% | 0% |
| large | large | 1.956 | 1.000 | 2.660 | 69.2 | 80% | 43% | 72% | 37% | 98% | 88% | 91% | 93% | 0% | 4% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
