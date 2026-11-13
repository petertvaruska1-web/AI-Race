# Scorecard

Data root: C:\Users\petko\Desktop\AI Race\data; yardstick root: C:\Users\petko\Desktop\AI Race\data; engine 2; 2026-11-13

First AI (first-s1) trained in 53.0 s on cuda (reused: the time of its own earlier training).

| Model | Recipe | Held-out loss | Held-out conv. loss | Yardstick loss | Bench | Well-formed | Fits the question | Casual facts | Follow-ups | Identity (named) | Says it is an AI (unnamed) | Life claims | Cut off | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-s1 | first | 2.489 | 2.413 | 2.117 | 51.1 | 77% | 77% | 74% | 12% | 0% | 0% | 0% | 5% | 0% |
| first-s2 | first | 2.465 | 2.407 | 2.109 | 49.0 | 80% | 78% | 69% | 10% | 0% | 0% | 0% | 5% | 2% |
| small | small | 2.346 | 2.253 | 1.959 | 57.0 | 90% | 90% | 88% | 8% | 0% | 0% | 0% | 5% | 0% |
| lineage-small | lineage | 2.346 | 2.257 | 1.973 | 54.9 | 60% | 77% | 88% | 2% | 0% | 0% | 2% | 3% | 0% |
| lineage-medium | lineage | 2.268 | 2.183 | 1.888 | 59.1 | 77% | 90% | 92% | 7% | 0% | 0% | 0% | 7% | 0% |
| chatty | chatty | 2.562 | 2.021 | 1.736 | 48.2 | 75% | 90% | 94% | 5% | 0% | 0% | 0% | 5% | 0% |
| large | large | 2.083 | 1.930 | 1.654 | 65.3 | 83% | 97% | 95% | 5% | 10% | 0% | 2% | 5% | 0% |

Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of replies: well-formed, casual facts, follow-ups and identity are better higher; life claims, cut-off replies and loops better lower.
