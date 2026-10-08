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
