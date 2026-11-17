# Scorecard

Data root: C:\Users\petko\Desktop\AI Race\data; yardstick root: C:\Users\petko\Desktop\AI Race\data; engine 3; 2026-11-17

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.458 | 2.393 | 2.110 | 49.7 | 79% | 84% | 62% | 56% | 73% | 2% | 2% | 0% | 0% | 4% | 0% |
| first-s2 | first | 2.466 | 2.391 | 2.113 | 47.5 | 87% | 75% | 49% | 44% | 76% | 0% | 3% | 0% | 1% | 5% | 0% |
| small | small | 2.335 | 2.254 | 1.962 | 53.0 | 89% | 89% | 70% | 62% | 82% | 4% | 1% | 0% | 0% | 5% | 0% |
| lineage-small | lineage | 2.326 | 2.237 | 1.957 | 52.4 | 68% | 82% | 69% | 62% | 85% | 6% | 2% | 0% | 0% | 4% | 0% |
| lineage-medium | lineage | 2.255 | 2.146 | 1.883 | 57.8 | 74% | 88% | 74% | 68% | 92% | 8% | 0% | 0% | 0% | 3% | 0% |
| chatty | chatty | 2.539 | 1.990 | 1.728 | 49.3 | 74% | 85% | 70% | 63% | 91% | 6% | 0% | 0% | 0% | 5% | 0% |
| large | large | 2.087 | 1.926 | 1.645 | 63.5 | 70% | 92% | 85% | 80% | 97% | 2% | 0% | 0% | 0% | 4% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
