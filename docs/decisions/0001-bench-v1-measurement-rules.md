# ADR 0001: bench-v1 measurement rules beyond the M1 plan

- **Date:** 2026-10-08
- **Status:** Accepted (engineering decision; no creative impact)
- **Context:** Milestone 1, Tasks 12, 16 and 17 (`ml/src/airace_ml/skills`, `ml/src/airace_ml/evals`)

## Why
Benchmark scores are public measurements (rule R5) and **0 means chance**. Reviews of the first implementation found several places where a model that does not understand anything still scored well. Typical cases are an untrained network, a model that repeats itself, one that parrots the question, or one that gives the same answer to everything. Each rule below closes one such gap. bench-v1 freezes once real players use it, so these had to be settled before then.

## Decisions
1. **Consistency measures robust knowledge.**
   - A group of three paraphrased questions scores 1 only if every paraphrase is answered correctly.
   - Chance is 1 / number of options, which is exactly what a guesser that always picks the same option text earns.
   - The plan scored agreement between the answers instead. Untrained models scored 62–91 on it, because the paraphrases share most of their words and a random network agrees with itself. Calibrating each option against a neutral context did not fix this (untrained models still scored 62–83).
2. **Instruction following has a measured chance level.**
   - Chance is the best pass rate any single constant reply gets on the items scored. Candidate replies include empty text, yes/no, a few degenerate replies, and every item's own reference reply.
   - The checks themselves stay format-only (one word, yes/no, list of N, …), as the design documents define instruction following.
   - Without this, a model that always says "Yes" scored 14.2.
3. **Creativity penalises loops and parroting.**
   - Each story's score is multiplied by (1 − its share of repeated word 3-grams), so a looping story scores ≈ 0. Before this, "the the the …" scored about 44.
   - Text copied from the story's own prompt counts as not new.
4. **"Well-formed" (gate criterion G2) also needs at least 3 different words.** The plan's "no 3-gram repeated 3+ times" rule only catches loops of five or more words, and "go go go go" passed it.
5. **Function-completion coding items are partitioned by their `def` line**, not by their docstring wording. This keeps the same function header and body out of training. Before this, 42 of 50 bench items appeared verbatim in training text.
6. **Quality scores for code and reasoning text** use only the garble, spam and repeated-line factors. Prose-only factors (known words, share of letters) would otherwise make the player's "cleaning" choice delete most clean code.

## Consequences
- Scores for untrained and degenerate models sit near 0, as R5 intends.
- Consistency is now correlated with knowledge, but it measures something knowledge doesn't: robustness across phrasings, on separate facts. A consistently wrong model earns no consistency credit.
- The full list of rulings, each with what it costs if wrong, is in `docs/handoff/sdd/ledger.md`.

## Addendum (2026-10-21): final-review changes before bench-v1 freezes
The whole-branch review of Milestone 1 found more places where a model that knows nothing still scored, or where a bench item could never be answered. bench-v1 is not frozen yet (no real player has used it), so these change the suite under the same version label. Each one is a ledger ruling ("FINAL REVIEW: Ruling Fxx").

7. **Function-completion prompts end at the docstring's closing quotes, without the trailing newline (F1).**
   - This deviates from plan line 1176.
   - A byte-level BPE merges that newline with the body's indentation, so in training a bare newline token is only ever followed by unindented lines. With the newline in the prompt, all 50 function items scored 0 for every model.
   - The checker still receives the full header.
8. **Knowledge and consistency answers are even within each relation (F5).**
   - The bench draws each relation's answer objects round-robin, then a fact for each.
   - With the 4 options exchangeable, a model that ignores the subject gets close to 1/4 on multiple choice. A residual skew of about 4–9 points remains: some relations have only one fact for a rare object, so the common objects fill the gap.
   - The chance level for free answers is a measured floor: for each relation, the share of its free-answer items that its most common accepted answer form would get right. It is an oracle constant, so it sits above what a real "always the common answer" model reaches.
9. **Knowledge free answers get a 12-token reply budget (F24),** so that "Bandar Seri Begawan" fits. The local session confirms this with the tok-v1 reply-budget test.
10. **Accepted forms:**
    - Animal classes accept plurals: mammals, birds, … (F53).
    - Planet order accepts the bare digit (F54).
11. **Creativity multiplies by variety (F6).**
    - The category score is `100 × mean(story) × min(1, d / d_ref)`.
      - `d` is the distinct-2 of the run's stories.
      - `d_ref` is how varied real stories are: the mean distinct-2 over sets of held-out creative openings as large as a run. The judge's calibration records it as `creative_distinct2`.
    - The plan's `0.5 + 0.5·d` let a model that tells the same story to every prompt keep half its score (21–52), against spec §4.7 (coherence × novelty × diversity) and rule 3 above.
    - The 0.4 novelty floor, the loop penalty and the prompt-echo rule are unchanged.
    - No length term was added; that part rested on an unverified guess about how the real judge scores short replies.
    - The local session checks the real numbers after the first gate run.
12. **The item set is pinned (F25).** A test hashes the canonical content of `build_suite()` (prompts, options, answers, checker arguments, tags, budgets). Any intended change to an item needs a `SUITE_VERSION` bump and a new recorded digest.

Refuted, and kept as they are: bird → reptile distractors (F55: the KB uses school-level classes); "a baby hen" (F56); the tag scheme (F49: tags join by item id later, for the M4 discovery engine).
