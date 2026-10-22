# ADR 0007: feasibility gate, iteration 3 (steadier measurements)

- **Date:** 2026-10-22
- **Status:** Accepted (engineering decision; no creative impact)
- **Context:**
  - Gate run 3 (after ADR 0006, with the owner's 60% bar, C3) passed G1, G3, G5 and G6.
  - G2 measured 57% (34 of 60). G4 failed: seeds 2.54 against noise 2.91.
  - Evidence: `docs/handoff/calibration/g2_diagnosis.py`, `g4_k_probe.py`, `ml/runs/bisect/` (not in git).

## Findings
1. **G4 was measurement noise.** With 3 replies per probe, one model fingerprinted twice gave distances of 1.52 to 3.98. The seed comparison rode on that noise: it passed in runs 1 and 2 and failed in run 3. On the same models:

   | | 3 replies per probe | 8 replies per probe |
   |---|---|---|
   | one model measured twice (noise) | 2.91 (1.52–3.98) | 1.27 (1.07–1.66) |
   | same mix, different seeds | 2.54 | 2.70 (2.49–3.09) |
   | different mixes | 4.77 | 4.63 |

   At 8, every seed distance is above every noise distance, and mixes stay well apart.
2. **G2: the first model sits at the owner's bar; no corpus change moves it reliably.**
   - The gate's three first models, over 180 replies each: 58%, 54% and 64% (mean 59%).
   - Undoing one corpus change at a time (seed 1): no story requests 53%, no quiz sheets 50%, old web settings 62%. All are within the run-to-run variation.
   - More real dialogue in Conversations (Soda 0.70, everyday 0.15, fact_chat and instructions 0.075 each): 68%, 56% and 61% (mean 62%). That is +3, within the variation, and it costs the chat-heavy model 9 instruction points (67.3 to 58.4) and 37 knowledge points. **Not adopted.**
   - With 20 replies per seed, which replies happen to be sampled moves the pooled rate by about 6 points, so at a true rate near 60% the verdict is close to a coin flip.

## Decisions
1. **Fingerprints ask every probe 8 times** (was 3): `measure_fingerprint`'s default and the gate's full scale. The game's personality words come from the same steadier measurement. It costs 2.7 times the generation per fingerprint, a few seconds a model.
2. **G2 asks each of its 20 probes 3 times per seed** (`LEGIBILITY_SAMPLES`): 60 replies a seed, 180 in all.
   - The bar (60%, the owner's) and the probes are unchanged; the sampling luck shrinks, in both directions.
   - Spec §4.12 is updated.
3. **No change to training or the corpus in this iteration.** Neither touches what is trained, so a gate rerun reuses every run and measures again.
