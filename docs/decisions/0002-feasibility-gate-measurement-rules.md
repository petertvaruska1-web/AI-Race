# ADR 0002: feasibility gate measurement rules beyond the M1 plan

- **Date:** 2026-10-21
- **Status:** Accepted (engineering decision; no creative impact)
- **Context:** Milestone 1, Task 20 and the final whole-branch review (`ml/src/airace_ml/experiments/gate.py`). The gate (spec §4.12) decides whether M1's real model is good enough to build the game on. The owner runs it on the RTX 3050.

## Why
A gate verdict is only worth something if a PASS cannot come from measurement noise, and a FAIL cannot come from a measure that counts the wrong thing. The final review found both. Each rule below is a ledger ruling ("FINAL REVIEW: Ruling Fxx").

## Decisions
1. **Three seeds, as spec §4.12 says (F10).**
   - G2, G5, G6 and G7 now run for every seed, like G3 and G4. G2 must pass on every seed.
   - G5, G6 and G7 keep their spec thresholds on the seed means, **and** their mean effect must exceed `SPREAD_FACTOR` (2.0) × the seed spread. This is stricter than the spread rule alone, and a criterion fails if either part fails.
   - **G1 stays single-seed.** It is a timing, and timing does not depend on the seed.
   - Quick mode stays single-seed and decides on that seed alone.
   - Cost: about 18 more runs, roughly 30–40 GPU minutes.
2. **G4 measures its own noise floor (F9).**
   - Each balanced seed model is fingerprinted a second time at `MEASURE_SEED + 1`. Same-model distances (first measurement against second) give the noise level, using one population that includes the re-measures.
   - G4 passes only if the seed distance exceeds that noise **and** stays below the mix distance.
   - `SEED_MIN_DISTANCE` (0.05) remains only as an absolute sanity bound. Fingerprint distances are in population-std units, so 0.05 sat about 40× below sampling noise.
   - Traits whose population std is below 1e-6 are left out of every distance.
3. **G5's continuations train on fresh data (F8).** Both arms (grown and ungrown) use the seed + `CONTINUATION_SEED_OFFSET` (10 000). They stay paired with each other but no longer replay the starter's exact token stream.
4. **G7(a) counts garble-like words, not every unknown word (F13).**
   - A word counts if it holds a character of the noise module's mojibake set, or if it is unknown, has at least `GARBLE_MIN_TYPO_LENGTH` (4) letters, and is one edit away from a known word. An edit is an insertion, deletion, substitution or swap of two neighbouring letters: the four typo kinds the noise module writes.
   - Real web text keeps 5–7% unknown words (names, brands, rare words) at every cleaning level. Under the old measure, even a perfect copier sat on the 0.8 bar.
   - Full mode samples 200 completions per arm, up from 60.
   - Beside the verdict, the gate reports the same measure on each arm's prepared web training text (`DATA_SAMPLE_DOCS` = 2000 docs). If that data-level ratio is above `DATA_RATIO_WARNING` (0.7), the warning leads the G7 detail: the data itself barely changes with cleaning, so strengthen the web noise or the cleaning thresholds before trusting G7(a).
   - The 0.8 bar stays.
5. **A gate folder remembers its setup (F20).** `gate_setup.json` records quick or full, the device and the seeds; another setup into the same `--out` is refused.
6. **No silent CPU fallback (F21).** The full gate and `build-judge` refuse it unless `--device cpu` is given. `--quick` may run on the CPU.

## Kept as they were (refuted in review)
- G2 stays a legibility check (F35); responsiveness is measured elsewhere.
- G7 has no word-volume guard (F36).
- A run that stopped unstable is retrained on rerun (F37), as Task 20's dispatch ruling 5 says.

## Consequences
- A full-scale G4 or G7 may now FAIL where it would have passed by noise or by counting names. That is the honest answer, and Task 20 step 8's iteration loop exists for it.
- The full gate takes about 30–40 minutes longer.
