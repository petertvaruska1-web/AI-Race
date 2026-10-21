# ADR 0005: feasibility gate, iteration 1

- **Date:** 2026-10-21
- **Status:** Accepted (engineering decision; no creative impact)
- **Context:** The first real gate on the RTX 3050 (Task 20 step 7) passed 4 of 7 criteria: G1, G4, G5 and G6 passed; G2, G3 and G7 failed. Plan step 8 asks for root causes and the smallest principled change. The evidence scripts are under `docs/handoff/calibration/` (`g2_diagnosis.py`, `creativity_probe.py`, `g7_*.py`, `quality_range_sim.py`).

## Root causes found
1. **G3: creativity was 0 for every model, the story-heavy one included.**
   - Asked "Write a short story about …" in chat, models answered with one-line chat replies ("The dog lives in the ocean.").
   - No training data paired a story request with a story. Stories existed only as plain documents, and the chat form was learned from short dialogues alone. The fully trained judge did the same.
2. **G2: the first model was too undertrained to be legible.**
   - The first model (4 layers × 128, about 1.3M parameters) trained on about 4.2M tokens in 256 steps of 16,384 tokens. Its replies had a median judge loss of 5.76 against the bar of 3.95, and 1 of 20 replies was well-formed.
   - The measure itself is sound: the fully trained judge passes it at 85%.
   - Fluency comes from tokens, not shape, at these sizes. Legible share of 60 replies:

     | First model | Tokens | Batch | GPU time | Legible |
     |---|---|---|---|---|
     | 4×128 | 4.2M | 16k | 27 s | 5% |
     | 4×128 | 4.2M | 8k | 33 s | 37% |
     | 4×128 | 8.4M | 8k | 49 s | 60% |
     | **4×128** | **12.6M** | **8k** | **68 s** | **73%** |
     | 6×192 | 6.3M | 8k | 67 s | 57% |

3. **G2's every-seed rule was miscalibrated.** It required 70% on each seed's 20 sampled replies. Twenty replies vary by about 10 points, so a model that is exactly 70% legible passes all three seeds about 23% of the time, and the reference judge (about 85%) fails about one run in five.
4. **G7(a): cleaning hardly changed the web text, and the model's own babble hid what it did change.**
   - Typo pages supply most of the web's garble-like words (12% of their words). Their quality averaged 0.66, at thorough cleaning's 0.65 bar, so thorough kept 57% of them. The known-word factor saturated at 90% known words, below the typo pages' upper range.
   - The models' own invented words (about 6% at 8.4M tokens) outweighed the data's (about 3%). At 25M tokens with 8k batches they fall to about 2.5%.
   - Half the prompts were noisy held-out pages. On those, both arms copy the prompt's noise equally (2.51% vs 2.49%), which dilutes the effect.
5. **G7(b): fact-checking had nothing to remove.**
   - The fact arms trained on 50% Reference Facts, which states every true fact thousands of times, so both arms preferred the planted false object about 2–3% of the time.
   - Web alone shows the effect, but weakly: 60% vs 49% at 6% false-fact pages, about 3 sightings per false fact.

## Decisions
1. **Stories on request (content).**
   - New source `tinystories_requests`: TinyStories' validation split, so the stories are not in the train split the creative dataset already reads. Licence CDLA-Sharing-1.0, pinned at the same commit.
   - Each story becomes the AI's reply to a fixed request frame. The frame names the story's main character when its first sentence introduces one ("there was a little bird named Pip" gives "Write a short story about a little bird."). This covers 82% of a 400-story sample, with no false subject found. Otherwise the frame asks for any story.
   - It is 15% of Creative Writing (tinystories 0.70, tinystories_requests 0.15, cosmo_stories 0.15).
   - Creative Writing's injected duplicate noise (3%) is dropped, because a chat takes typo noise only. Its 2% typos stay.
   - Measured: the story-heavy model's creativity went from 0 to 19.6. Its stories' judge losses (1.4–3.1) are on the real-story scale.
2. **The first model trains on 12.6M tokens** (`STARTER_BUDGET` 12,582,912, three times 4.2M), with the same shape (about 1.3M parameters). It is 73% legible and trains in about 68 s on the GPU, within the 90 s target. Spec §4.11 is updated.
3. **The default batch is 8,192 tokens** (was 16,384), for the game's training too.
   - At these budgets twice the steps learn far more per token: the first model's held-out conversation loss on the same 4.2M tokens drops from 3.40 to 2.98.
   - Players pay for tokens, not wall time, so learning per token is what matters. The GPU does about 20% fewer tokens per second.
   - The judge keeps its own pinned 32,768-token batch.
4. **G2 decides on the pooled replies of all seeds:** 70% of 60. It still reports each seed's count. This is the seed-mean approach ADR 0002 already uses for G3 and G5–G7, and it supersedes the every-seed rule of ADR 0002 rule 1 (ruling F10) for G2 only.
5. **The quality score's known-word range is (0.7, 0.95)** (was (0.5, 0.9)). It was calibrated on the real web, where 90% of clean pages have 90% or more known words and typo pages have a median of 82%.

   | Web pages | Thorough keeps (old) | Thorough keeps (new) |
   |---|---|---|
   | clean | 90% | 85% |
   | typo | 57% | 29% |
   | garbled | 22% | 15% |

   Books, Educational, Creative Writing, Conversations and Reference Facts keep the same share at every cleaning level.
6. **G7(a) prompts with clean held-out web pages only.** A noisy prompt measures copying from the prompt, not what cleaning taught. Measured at the new budgets: light 2.62% vs thorough 1.73% garble-like words on clean prompts (ratio 0.66).
7. **G7(b)'s fact arms train on web alone, and web carries twice the planted false facts (12% of pages, was 6%).** Measured at the new budgets: 70% vs 49% false facts preferred (ratio 0.70). In the game, fact-checking now matters most to a company that leans on the web rather than on Reference Facts.

## Costs
- The CPU fallback for the first model is no longer 8 minutes (spec §4.11 records the measured figure). The GPU target, the one the reference machine uses, still holds.
- The full gate runs about an hour longer: the first model and its continuations train on three times the tokens, and G7's arms on 25M tokens.
- `tinystories_requests` and the stricter quality score change the corpus, so the novelty index and the judge are rebuilt.
