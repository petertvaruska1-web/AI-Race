# ADR 0009: training engine 2 (measurements and the values they set)

- **Date:** 2026-11-10/11
- **Status:** Accepted (engineering decisions; the owner's part is C49, C54–C56 and the 2026-11-10 request). Design: `docs/superpowers/specs/2026-11-10-training-engine-2-design.md`.
- **Evidence:** `docs/handoff/upgrade-2026-11-10/` (every script; `results.jsonl` has one line per training run). All numbers are from the owner's RTX 3050 with nothing else on the card. "Loss" is the mean held-out loss over the eight datasets (128 sequences each, the same for every run); seed-to-seed it varies by about 0.005 (AdamW) to 0.015 (Muon).

## 1. Where the time went (engine 1)
| Run | Wall | Model step | Sample writing + held-out | Sampler | Snapshots, saves |
|---|---|---|---|---|---|
| First AI, 4×128, 12.6M tokens | 70.4 s | 65% | 20% | 7% | 2% |
| Medium 8×256, 6.3M tokens | 102 s | 70% | 24% | 1% | 2% |

Inside the model step only 20% (tiny) to 27% (medium) of the GPU time is matrix products; the rest is memory-bound elementwise work, 20% of it the hand-written RMSNorm.

Raw step throughput (tokens a second, 8,192-token batches): tiny 272k → **344k** with the fused norm and the loss inside autocast; medium 87k → **108k**. Batches of 16k–65k tokens add 5–15% (the card is full at 8k), merged q/k/v or gate/up matrices are 4–8% slower, a `roll`-based RoPE is no faster. `torch.compile` needs Triton, which PyTorch does not ship for Windows.

## 2. The optimizer (first-AI recipe: 4×128, 12.6M tokens, even mix, standard cleaning)
| Setting | Loss |
|---|---|
| AdamW 6e-4 (the old **Careful**) | 2.855 |
| AdamW 1.5e-3 | 2.617 |
| AdamW 2.68e-3 (the old default) | 2.568, 2.572 |
| AdamW 4e-3 | 2.546 |
| AdamW 6e-3 | 2.591 |
| AdamW 1.2e-2 (the old **Bold**) | 2.835 |
| AdamW 4e-3 on 16.4M / 18.9M tokens | 2.484 / 2.459 |
| AdamW, no weight decay / linear-to-zero schedule / 4,096-token batches | 2.567 / 2.636 / 2.574 |
| Muon 0.005 / 0.01 / 0.015 / 0.02 (embedding at 1.2e-2) | 2.482 / 2.470 / 2.473 / 2.480 |
| Muon 0.04 / 0.08 | 2.757 / 3.238 |
| Muon 0.015, momentum 0.9, 3 iterations, decay 40% to 2% (**chosen**), 3 seeds | **2.447, 2.451, 2.452** |

- Muon with the chosen settings is 0.12 better than the old default and reaches what AdamW needs about 1.5× the tokens for. It costs about 6% more time per step (batched by matrix shape).
- Bigger brains: small 6×192 at 12.6M tokens 2.439 → **2.315**; medium 8×256 at 25.2M tokens 2.189 → **2.114**.
- Its good range is wide (0.005–0.02 within 0.02 of the best); beyond it quality falls off without the run diverging.
- 3 Newton–Schulz iterations learn as well as 5; momentum 0.9 a little better than 0.95; weight decay on the matrices changes nothing (left at 0); gradient clipping helps slightly (kept).
- The embedding and gains' AdamW rate: 2.7e-3 → 2.492, 6e-3 → 2.477, 1.2e-2 → 2.468, 2.4e-2 → 2.480. Chosen: 0.8 × the matrices' rate.
- Schedule: decay over the last 40% to 2% of the peak is 0.01–0.02 better than the last 20% to 10%, for both optimizers.

**Decisions:** `MuonAdamW` (`ml/src/airace_ml/train/optim.py`); `DECAY_FRAC = 0.4`, `MIN_LR_RATIO = 0.02`; the slider maps boldness 0…1 onto a matrix rate of 0.006…0.03 (log scale; 0.0134 in the middle), so Careful and the default are inside the good range and only Bold leaves it (C55).

## 3. Talking
- **Casing.** 99.5% of the corpora's user turns start with a capital and all end in `.`, `?` or `!`. `hello` is the tokens `he|ll|o`; `Hello` is one token.
- **Fact questions in the training's own chat phrasing, share answered right (greedy, 110 questions over all 22 relations):**

| First AI (4×128, 12.6M tokens) | as written | all lowercase | lowercase, no "?" | no "?" | first letter capital, rest lowercase, no "?" |
|---|---|---|---|---|---|
| engine 1 (the gate's `starter`) | 67% | 34% | 17% | 60% | 48% |
| Muon, no casual typing | 88% | 62% | 59% | 87% | 71% |
| Muon + 30% casual typing | 86% | 74% | 75% | 86% | 74% |

  Casual typing costs nothing on clean text (held-out loss 2.441 against 2.447).
- **Temperature** (well-formed replies by the gate's measure, 60 replies): engine-1 first AI 50% at 0.8/0.95, 77% at 0.6/0.95, 85% at 0.5/0.95, 80% greedy; engine-1 small brain 60% / 87% / – / 95%; the judge 85% at every setting. Garble-like words 1.2% → 0.7%.
- **Decisions:** `CASUAL_SHARE = 0.3` with the five typing styles of `airace_ml/data/casual.py`; serving at temperature 0.6, top-p 0.9; a looping reply ends (`airace_ml/infer/serving.py`); the Playground capitalises the first letter of a typed message.

## 4. Considered and not done
- A corpus rebuild with casual conversations: the sampler does it without one.
- Lower temperature for bench-v1, battles, games and the fingerprint: recorded scores must stay comparable (C56).
- Tightening "standard" cleaning: typo pages are 10.7% of the web pool and 2.5% of conversations at standard; garble in replies is already under 1% at the new serving settings. To be measured again after the gate.
- Architecture changes (QK-norm, untied head): they would need a second model family for growth and old checkpoints; the optimizer gave more for less.
