# Scorecard

Data root: ..\data2; yardstick root: ..\data; engine 3; 2026-11-17

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.349 | 1.277 | 2.839 | 49.8 | 79% | 42% | 63% | 36% | 80% | 71% | 41% | 99% | 0% | 6% | 0% |
| first-s2 | first | 2.334 | 1.291 | 2.799 | 52.1 | 81% | 31% | 50% | 22% | 76% | 66% | 63% | 89% | 1% | 3% | 1% |
| small | small | 2.221 | 1.126 | 2.755 | 58.2 | 85% | 52% | 61% | 36% | 87% | 71% | 61% | 90% | 0% | 3% | 1% |
| lineage-small | lineage | 2.216 | 1.135 | 2.867 | 57.1 | 78% | 41% | 53% | 27% | 91% | 77% | 71% | 100% | 0% | 4% | 0% |
| lineage-medium | lineage | 2.127 | 1.044 | 2.785 | 59.1 | 80% | 54% | 60% | 41% | 94% | 80% | 60% | 64% | 0% | 4% | 0% |
| chatty | chatty | 2.474 | 0.904 | 3.187 | 49.3 | 82% | 43% | 74% | 40% | 98% | 83% | 46% | 99% | 0% | 3% | 0% |
| large | large | 1.943 | 0.848 | 2.685 | 65.6 | 84% | 52% | 72% | 46% | 96% | 75% | 84% | 89% | 0% | 4% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
