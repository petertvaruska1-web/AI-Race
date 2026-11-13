# ADR 0010: the next level of the training system (measurements and the values they set)

Date started: 2026-11-13. Design: `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`. Plan A: `docs/superpowers/plans/2026-11-13-next-level-a-text-and-voice.md`. Earlier measurements: ADR 0009.

Every task of the plan that measures something writes its numbers and its decision here, kept or removed.

## 1. Measured while planning (2026-11-13, RTX 3050, engine 2)
- **Vocabulary** (`docs/handoff/upgrade-2026-11-10/tok_probe.py`): tok-v1 holds 3.22 bytes a token; a retrained 8,192 vocabulary 3.68 (conversations 3.67 against 3.53), 16,384 holds 3.99. "hello" is 3 tokens in tok-v1 and 2–3 after retraining. **Decision:** tok-v1 stays; casual typing is taught, not tokenized away.
- **Tokens a step** (`train_product.py --batch-tokens`), held-out loss and time for 12.6M tokens: Tiny at 8,192 (the product) 2.447 in 54.9 s; at 16,384: 2.502 in 49.4 s (2.478 with a bolder rate); at 4,096: 2.482 in 75.9 s. Small at 8,192: 2.352 in 82.9 s; 16,384: 2.359 in 79.8 s; 4,096: 2.357 in 107 s. **Decision:** 8,192 stays.
- **Corpus v1's conversations:** 19.8M tokens in 235,749 documents (median 49 tokens): SODA 12.5M (two people; the first speaker is the user, every other the AI), fact questions 3.4M, instructions 3.4M, everyday conversations 0.5M. 3% of the documents carry typos over the whole document, the AI's turns included. `<|sys|>` is used nowhere. The sampler draws documents with replacement.

## 2. Sources and licences for training text 2
- **`HuggingFaceTB/smol-smoltalk`** (card licence apache-2.0; 484,570 rows; columns `messages`, `source`). Its parent card (`HuggingFaceTB/smoltalk`) says: "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0" and "For the existing public datasets, please refer to the original dataset for the license". **Decision:** only rows whose `source` is `smol-magpie-ultra-short` or `smol-contraints` (the card's spelling) are used. Rows from OpenHermes, Self-OSS-Instruct, SystemChat and the other existing public datasets are left out: their licences are their originals', not checked. The rewriting and summarizing rows are Apache-2.0 but are not conversation practice for these brains.
- Everyday conversations keep coming from `HuggingFaceTB/everyday-conversations-llama3.1-2k` (apache-2.0, already in the manifest).
- Checked on the dataset cards on 2026-11-13; the revision is pinned in `airace_content/sources.py` when the source is added (plan A, Task 11).

## 3. The scorecard baseline (engine 2, text 1; `docs/reports/2026-11-scorecard-engine2.md`)
`airace-ml scorecard --out runs/scorecard-engine2 --timing` on 2026-11-13, RTX 3050, nothing else running during the timed run. Probes are scored on whole words (a reply holding "seventeen" does not answer "seven").

| Model | held-out loss | yardstick loss | bench-v1 | well-formed | casual facts (all five ways / lowercase, no "?") | follow-ups | says its name (told) | says it is an AI (not told) | cut off |
|---|---|---|---|---|---|---|---|---|---|
| First AI, seed 1 (Tiny, 12.6M) | 2.489 | 2.117 | 51.1 | 77% | 74% / 65% | 12% | 0% | 0% | 5% |
| First AI, seed 2 | 2.465 | 2.109 | 49.0 | 80% | 69% / 64% | 10% | 0% | 0% | 5% |
| Small, 12.6M | 2.346 | 1.959 | 57.0 | 90% | 88% / 83% | 8% | 0% | 0% | 5% |
| Second generation (Tiny grown to Small) | 2.346 | 1.973 | 54.9 | 60% | 88% / 81% | 2% | 0% | 0% | 3% |
| Third generation (grown to Medium) | 2.268 | 1.888 | 59.1 | 77% | 92% / 87% | 7% | 0% | 0% | 7% |
| Conversation-heavy Small, 8.4M | 2.562 | 1.736 | 48.2 | 75% | 94% / 94% | 5% | 0% | 0% | 5% |
| Large, 25.2M | 2.083 | 1.654 | 65.3 | 83% | 95% / 91% | 5% | 10% | 0% | 5% |

The first AI trained in 53.0 s.

**What the baseline says:**
- **Follow-up questions are barely answered at all** (2–12%), by any brain. This is the largest gap the card shows.
- **No AI says its name when told it, or says it is an AI when asked**, as expected: nothing in text 1 teaches either.
- **Claims of a person's life in small talk are rare** (0–2% by the word list): the visible problem is the invented name and the missing "I am an AI", not family talk. The spec's target for life claims (at most 5%) is already met and stays as a guard; the measure that carries the target is "says it is an AI (not told)".
- Well-formed replies: 77–80% for a first AI, 90% for a new Small brain, and only 60% for the grown Small brain (60 replies each; to be looked at in the growth task).
- Loops are gone (0–2%); cut-off replies are 3–7%, already under the 10% target.

**A second talk measure, and what it showed (same day).** Reading the replies (`docs/handoff/next-level/talk_show.py`) showed that the gate's well-formed rule fails good short replies ("You're welcome." to "Thank you.", "I'm good." to "How are you?") because it wants a minimum length and never sees the question. The card therefore also carries **fits the question**: the judge's loss of the reply given the question, no higher than the 90th percentile of 1,000 real held-out replies (2.524), with no minimum length; and it stores every small-talk reply with both verdicts.

| Model | well-formed | fits the question |
|---|---|---|
| First AI, seeds 1 and 2 | 77%, 80% | 77%, 78% |
| Small | 90% | 90% |
| Second generation (grown to Small) | 60% | 77% |
| Third generation (grown to Medium) | 77% | 90% |
| Conversation-heavy Small | 75% | 90% |
| Large | 83% | 97% |

- The grown Small brain's 60% was mostly the length rule: it answers briefly. By "fits" it is level with a first AI and behind a new Small brain (77% against 90%), so something real remains for the growth task.
- "Fits" still passes fluent answers to another question ("What is your name?" → "We call a group of monkeys a troop.") and fails a bare "Yes." to "Do you like dogs?": the loss is mostly about how fluent the reply is. An **on-topic** check is added next (is the reply more likely after its own question than after other questions), calibrated on real against shuffled question-reply pairs so that its own error rate is known. Until then no decision rests on small-talk rates alone; follow-ups, identity and casual facts are exact.

**Targets fixed from the baseline** (the spec's section 7 rules):

| | baseline | target |
|---|---|---|
| First AI: fits the question (and on topic, once measured) | 77–78% | at least 90% |
| First AI: well-formed (the gate's rule, for continuity) | 77–80% | not lower |
| First AI: facts typed in lowercase without "?" | 64–65% | at least 80% |
| Says its name when told / says it is an AI when not told | 0% | at least 60% for Tiny, 85% for Small and larger |
| Follow-ups, Small | 8% | at least 60% |
| Third generation: bench-v1 overall | 59.1 | at least 64 |
| Life claims in small talk | 0–2% | at most 5% (guard) |
| Cut-off replies | 3–7% | at most 10% (guard) |
| First AI: training time | 53.0 s | at most 55 s |
| Lessons answered after one run | measured in Task 8 | at least 7 of 10 |

**Rulings on the measuring stick itself** (plan A, Task 3): the yardstick loss is the trainer's held-out rule over 128 packed rows of text 1's held-out conversations; a baseline measured with another yardstick, judge, novelty index or scale is refused; cards of different training text are not compared on their own held-out losses; a lost measure or a run that stopped early counts as worse.

## 4. Engine 3 items
(Tasks 5–8: numbers and keep/remove for each.)

## 5. Training text 2 against text 1
(Tasks 14–15.)

## 6. A voice of its own
(Tasks 16–18.)
