# Scorecard

Data root: C:\Users\petko\Desktop\AI Race\data; yardstick root: C:\Users\petko\Desktop\AI Race\data; engine 2; 2026-11-15

First AI (first-s1) trained in 53.0 s on cuda (reused: the time of its own earlier training).

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | On topic | Fits and on topic | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.489 | 2.413 | 2.117 | 51.1 | 70% | 75% | 65% | 55% | 74% | 8% | 0% | 0% | 0% | 6% | 0% |
| first-s2 | first | 2.465 | 2.407 | 2.109 | 49.0 | 80% | 76% | 67% | 53% | 69% | 8% | 2% | 0% | 1% | 4% | 1% |
| small | small | 2.346 | 2.253 | 1.959 | 57.0 | 88% | 87% | 62% | 59% | 88% | 5% | 0% | 0% | 0% | 5% | 0% |
| lineage-small | lineage | 2.346 | 2.257 | 1.973 | 54.9 | 65% | 78% | 66% | 57% | 88% | 3% | 0% | 0% | 1% | 3% | 0% |
| lineage-medium | lineage | 2.268 | 2.183 | 1.888 | 59.1 | 79% | 90% | 82% | 75% | 92% | 4% | 1% | 0% | 0% | 5% | 0% |
| chatty | chatty | 2.562 | 2.021 | 1.736 | 48.2 | 75% | 89% | 83% | 78% | 94% | 3% | 0% | 0% | 0% | 4% | 0% |
| large | large | 2.083 | 1.930 | 1.654 | 65.3 | 83% | 95% | 85% | 83% | 95% | 7% | 12% | 0% | 1% | 5% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.

Real replies themselves: fits 90%, on topic 94%, both 86% (the on-topic rule rejects 94% of swapped replies, 81% of the hard ones).
