# Task 11 report: reasoning and pattern skill generators

Status: DONE_WITH_CONCERNS (design notes below; nothing blocks Tasks 12-20).
Commit: `a563e27 feat(ml): reasoning and pattern skill generators with bench partition`

## What I implemented

### `ml/src/airace_ml/skills/reasoning.py`
Public: `reasoning_train_docs(rng, n)`, `reasoning_bench_items(rng, n=200)`, plus constants `FAMILIES`, `NAMES`, `BOYS`, `GIRLS`, `NONCE_WORDS`, `TOPIC` (used by the tests).

Four families of puzzles, all in the frame `Question: <text>\nAnswer: <answer>`:

| Family | Puzzle | Options (bench) |
|---|---|---|
| `compare` | 3 names in a taller/older/faster/heavier chain, two premises (premise order, and positive versus opposite wording "shorter"/"slower"/..., are randomized), asks for the top or the bottom ("Who is the tallest/shortest?") | the 3 names + 1 extra name that is never in the prompt, shuffled |
| `syllogism` | nonce words and a person name: one-step ("All blicks are fenks. Tom is a blick. Is Tom a fenk?" / "No ...") or two-step ("All A are B. All B are C. Is a A a C?" / "No B are C"), premise order randomized | `["yes", "no"]` fixed order |
| `word_problem` | one addition or subtraction story (3 phrasings each), start 2-20, result 1-20 (adds: total <= 20; subtracts: result >= 1) | the answer + 3 different numbers within 3 of it, never negative, shuffled |
| `count` | "How many cats are in this list: cat, dog, cat, bird?", 4-8 items from an animal or a fruit group, answer 0-5 | same number options as above |

- **Canonical key** = the bench prompt `Question: <text>\nAnswer:`. Train blocks are `<prompt> <answer>`, optionally followed on the next line by one explanation sentence (~40% of blocks), blocks separated by a blank line; 3-8 blocks per doc; families mixed at random inside a doc.
- **Retry loop** (as the brief says): a `_Drawer` draws a puzzle of the wanted family, accepts it when `reserved_for_bench(prompt) == wanted` (bench: reserved; train: not reserved), the prompt is not already used (bench: never repeats; train: within one doc), and (syllogism only) the answer is the wanted one. The budget is 50 draws per requested item (bench: `50*n`; train: `50 * total blocks`); running out raises `RuntimeError("reasoning: could not draw enough <family> puzzles within the budget")`.
- **Bench**: ids `reasoning-0000...`, category `"reasoning"`, tags `("fam:<name>",)`. Families take turns (compare, syllogism, word_problem, count), so 200 items are exactly 50 per family and any prefix is balanced. Syllogisms alternate yes/no, so exactly 25 yes and 25 no (the answer is enforced inside the retry loop). Option order is shuffled for everything except yes/no.
- `TextDoc.topic = "school"` (a member of `TOPICS`). No `topic:` tag on bench items (not natural: one topic for the whole category).
- Words: 24 names (12 boys, 12 girls; no name is a substring of another), 30 nonce words (consonant start, plain `+s` plural, none equal to a name), pronouns match the name's gender.

### `ml/src/airace_ml/skills/patterns.py`
Public: `pattern_train_docs(rng, n)`, `pattern_bench_items(rng, n=150)`, plus constants `FAMILIES`, `SHOWN_TERMS = (4, 5, 6)`, `CYCLE_WORDS`, `TOPIC`.

Families (all as specified in the brief): `step` (start 1-20, step 1-9), `letters` (step 1-3, whole sequence including the answer within A-Z), `cycle` (2-3 distinct words from 12 colors or 12 animals, never mixed), `double` (start 1-5), `countdown` (start up to 40, step 1-3, answer >= 0).

- A pattern is a prompt of 4, 5 or 6 shown terms plus the next term. Bench prompt/key: `Next: 2, 4, 6, 8,`; answers `["10"]`; `extract="first_item"`; ids `pattern-0000...`; tags `("fam:<name>",)`; `max_new_tokens` and `chat` stay at the defaults.
- Train line: `Next: 2, 4, 6, 8, 10`; 3-8 different lines per doc (joined with `\n`), family chosen uniformly per line, then a uniformly random pattern of it.
- **Partition at the prefix level (important).** A train line `Next: a1, ..., a6, a7` contains the strings `Next: a1, ..., a4,` and `Next: a1, ..., a5,`, which are valid bench prompts of other (shorter) items. So a train pattern is allowed only if *none* of its prefixes that show 4..k terms is reserved (`_blocked`). This is what makes `it.prompt not in train` true for every possible bench prompt, not only for the lengths the line happens to have. I did not check prefixes shorter than 4 terms (a bench never asks them, and checking would wrongly remove 10% of all start values).
- **Enumerated, not retried (deviation from the brief's wording, see Concerns).** All patterns of every family are listed once (`_space()`, cached: step 540, letters 144, cycle 8712, double 15, countdown 279 = 9690 patterns). The bench pool is the reserved ones, the train pool is the unblocked ones. Bench takes quotas per family with the same water-filling as `facts._fair_quota` (small families first, then an even split), picks without replacement, then shuffles with the rng. Asking for more items than there are reserved patterns raises `ValueError("pattern: cannot take N items, only M exist")`.
- Resulting default bench (150): cycle 46, step 45, countdown 40, letters 17 (all it has), double 2 (all it has). Train pools: step 419, letters 117, cycle 7141, double 11, countdown 209.

## Tests and results

`ml/tests/test_reasoning_patterns.py`: the 3 brief tests (unchanged assertions) plus 21 more = **24 tests**. The extra tests use **independent solvers** (regex parsers that know each puzzle's rule, not the generators' internals) to check every bench item and every train block:

- Reasoning bench: ids/category/frame, no repeated prompt, exactly 50 per family; solver answer equals `options[answer_index]` and is the only option equal to the solved answer; compare options = the 3 names + a stranger not in the prompt, attribute of premise matches the asked superlative, all 4 attributes and all 8 superlatives occur; syllogisms are determinate (the solver raises if premises do not settle the question), options `["yes","no"]`, 25/25, both one-step and two-step with both answers; word problems have start <= 20, answer in 0..20, additions and subtractions occur, 3 distractors within 1..3 of the answer; count lists have 4-8 items and answer 0 occurs; answer positions spread over all 4 slots (>= 20 each).
- Reasoning train: 2000 docs, 3-8 blocks, 2-3 lines per block, every `Answer:` equals the solver's answer, all families > 20% of blocks, 20-60% explained, explanation is one line ending in ".", never contains a bench prompt (checked on two different seed streams and as a substring check over the whole text), determinism, stream dependence, `n=0`.
- Attempt cap: with `reserved_for_bench` patched to never accept, the bench raises `RuntimeError` after exactly `50*3` draws; the same for train docs.
- Pools: names unique, no name inside another, nonce words consonant-initial, plural-safe, not names.
- Patterns: bench shape (ids, `first_item`, 150 unique prompts, 4-6 shown), `families_that_continue` (independent rule check) says exactly one family explains each prompt and it predicts the item's answer and equals its tag; every family appears (double >= 1, others >= 10, step and cycle within 20 of each other); family limits (letters within A-Z, countdown >= 0 and start <= 40, cycle words from one group with 2-3 distinct, double start 1-5, step start 1-20 and step 1-9); train lines (two seed streams, 3000 docs): 3-8 distinct lines, correct continuation, each explained by exactly one family, and *every prefix of 4..k terms is not reserved and not a bench prompt*; the whole pattern space has one answer per key and exactly one family per key (no ambiguous prompt); determinism, `n=0`, `ValueError` when asking for more than exists.

Results:
- Focused: `python -m uv run pytest tests/test_reasoning_patterns.py -v` -> `24 passed in 3.47s`
- Full suite: `python -m uv run pytest -q` -> `621 passed, 3 deselected in 55.28s` (no warnings)
- `python -m uv run ruff check .` -> `All checks passed!`

Mutation checks (temporary edits, restored afterwards; the suite fails on each): train ignoring the partition; bench using the train filter; patterns blocking only the line's own key (not its shorter prefixes: caught by `test_partition_disjoint`, the brief's own test); pattern bench ignoring the reserved rule; syllogism balance removed; `double` tripling; `countdown` allowed below 0; word problem totals above 20; compare stranger option inside the premise; wrong compare superlative answer; letters with step 4; wrong cycle answer. A no-op control mutation survived, as it should.

## TDD evidence

RED (tests first, no implementation):
```
$ cd ml && python -m uv run pytest tests/test_reasoning_patterns.py -q -x
tests\test_reasoning_patterns.py:6: in <module>
    from airace_ml.skills import reasoning
E   ImportError: cannot import name 'reasoning' from 'airace_ml.skills' (...\skills\__init__.py)
ERROR tests/test_reasoning_patterns.py
1 error in 0.24s
```
Expected: `reasoning.py` and `patterns.py` did not exist yet.

GREEN (after both modules):
```
$ python -m uv run pytest tests/test_reasoning_patterns.py -q
........................                                                 [100%]
24 passed in 3.19s
```
(24 passed in 3.47s after the final ruff reformat and docstring edit; the first run was green with no iteration on the implementation, so I added the mutation checks above to show the tests can fail.)

## Files changed (all new)

- `ml/src/airace_ml/skills/reasoning.py`
- `ml/src/airace_ml/skills/patterns.py`
- `ml/tests/test_reasoning_patterns.py`

Brief test code: assertions unchanged; `ruff format` re-wrapped the long `for` line in `test_partition_disjoint`. `ruff check` here also enforces isort order and RUF007 (`itertools.pairwise`), which I fixed in my extra tests. 9 older files in the repo are not `ruff format`-clean; I did not touch them.

## Self-review

- Completeness: both interfaces, ids, categories, tags, `extract`, sizes (200/150), determinism, partition both ways, all brief clarifications (extra name not in premise, nonce words, yes/no balance, results 0..20, A-Z, countdown >= 0, cycle next word).
- Quality: one responsibility per file; the two shared patterns (round-robin families, retry-with-budget) live in `reasoning.py`; the water-filling helper is a local copy of `facts._fair_quota` (8 lines) rather than a cross-module private import or an edit to Task 10 files.
- Tests are behavior tests: the solvers re-derive each answer from the prompt text.

## Concerns

1. **Patterns enumerate instead of retrying.** The brief says to use a retry loop capped at 50x for both modules. For `reasoning` I did exactly that. For `patterns` the spaces are tiny (double: 15 prompts), so a retry loop cannot fill family quotas, would repeat items, and cannot tell "exhausted" from "unlucky". Listing the space gives the same partition rule exactly, no repeats, an even family spread, and the same "raise if it cannot fill" behavior (`ValueError` for the bench, `RuntimeError` if a train doc cannot get distinct lines). Easy to swap if you want the literal loop.
2. **`double` is thin by construction** (start 1-5, 4-6 shown = 15 prompts): the bench has only 2 `fam:double` items (`Next: 4, 8, 16, 32,` and `Next: 5, 10, 20, 40, 80, 160,`), and only 11 double lines exist for training, so training text repeats them many times (the model can memorize them). `letters` is also small (17 bench items). Per-family scores in `tag_breakdown` for `double` and `letters` will be noisy. Widening the ranges (more shown terms, or a larger start range for training only) would fix it, but it changes the brief's stated ranges, so I left it.
3. **Premise-level overlap**: the key is the question, as the brief says. The same premises asked with a different question (e.g. "tallest" in the bench, "shortest" in training) are different keys, so a training block may share its premises with a bench item. Making the key the premise would be stricter if wanted.
4. **Surface cues in syllogisms**: the answer is largely readable from "All" versus "No" (in two-step puzzles from the second rule). The accuracy normalization (chance 50%) handles chance level, but a tiny model can score above chance on this family without reasoning. I kept the language simple as asked rather than adding contrapositive cases.
5. Options are not shown in the prompt (frame is `Question: ...\nAnswer:` scored with `" " + option`), so compare chance is 25%, yes/no 50%, numbers 25%.

---

# Fix round 1 (reasoning bench as a measurement, spec R5)

Status: DONE_WITH_CONCERNS (one concern the coordinator should rule on: the syllogism lookup shortcut, Concerns 1).

## What changed

**I1 syllogism.** Every puzzle now has the same quantifiers whatever its answer, with the four templates as ruled (decoy premise on fresh nonce words, premise order shuffled over all 3! orders): yes/1 step `All a are b. No c are d. N is an a. Is N a b?`, no/1 step `No a are b. All c are d. ...`, yes/2 steps `All a are b. All b are c. No d are e. Is a a c?`, no/2 steps `All a are b. No b are c. All d are e. ...`. The benchmark steps through the 4 templates (13/13/12/12, so 25 yes and 25 no). Explanations ignore the decoy.

**I2 compare.** About 1/3 of questions each ask the top, the bottom and the middle ("Who is in the middle?"; bench 17/17/16, training random). The fourth (odd) name is now in the prompt, in an irrelevant third sentence ("Zoe is a friend of Tom.", 6 wordings), so it is no free elimination. The old rule "extra name must not appear in the premise" is superseded. I made the aside also name one of the three chain names (chosen at random): without that, "Who is in the middle?" is answered by "the name mentioned twice" (a question-aware mention counter scored 0.66 on that variant; with the aside it scores 0.38). The 2 chain sentences and the aside are shuffled over all orders; polarity (taller/shorter) is random. 4-name chains: not done (optional).

**I3 train/bench leak.** The partition is now on the *world* of a puzzle (the facts without their wording), and the prompt must be reserved too:
- compare world = (attribute, top, middle, bottom), which covers every polarity, premise order, question, odd name and aside; syllogism world = (set of premises, question), which covers every premise order; word_problem world = (name, thing, operation, a, b), which covers every story wording (5 per operation now); count world = (word, multiset of list items), which covers every list order.
- Bench: world key reserved AND wording (prompt) reserved, and no two bench items share a world. Training: world key not reserved AND prompt not reserved (so "train skips reserved keys" still holds literally).
- **Deliberate deviation from the ruled mechanism.** The ruling was to mirror `patterns._blocked` (accept a puzzle only if none of its variant prompts is reserved). With the odd-name aside a compare world has 2 polarities x 6 premise orders x 3 questions = 36 prompts, so a training draw would be accepted with probability 0.9^36 = 2%: about 45 draws and 36 hashes per block (minutes at 100k blocks, and 45 on average is right at the 50x budget). Hashing the world key gives the same guarantee (no benchmark world in any wording, also across odd names and asides, which a variant list cannot cover) for one extra hash per draw. The brief's assertion `reserved_for_bench(it.prompt)` still holds for every bench item. The budget is now 50 *world* draws per requested item plus up to 40 wordings per accepted world (so the bench prompt can be reserved too); running out raises `RuntimeError`.
- Word problems with the same (operation, a, b) about other people or things do recur; documented in the module docstring and accepted as ruled.
- Leak measured with an independent world parser on 18,000 training docs (99,125 blocks) against the 200 bench items:

| family | before (HEAD) | after |
|---|---|---|
| compare | 11/50 bench worlds in train | 0/50 |
| syllogism | 7/50 | 0/50 |
| word_problem | 14/50 | 0/50 |
| count | 14/50 | 0/50 |

**I4 double.** Start 1-9 x factor 2 or 3 x 4-7 terms shown, answer up to 10,000 (the name stays `double`; the docstring says times 2 or times 3). Space 15 -> 67 prompts, bench 2 -> 7 items. All families now show 4-7 terms (`SHOWN_TERMS = (4, 5, 6, 7)`; one global set keeps the prefix check simple), so the other spaces grew too (step 540 -> 720, letters 144 -> 180, cycle 8712 -> 11616, countdown 279 -> 360). Default pattern bench: step 41, cycle 41, countdown 41, letters 20, double 7 (was 45/46/40/17/2). The test asserts double >= 5.

**I5 plurals.** Only words that take a plain "s". Dropped peach, mango, cherry (new: fig, lime, kiwi). The pool test now covers `COUNT_GROUPS` and `THINGS` (a word is rejected if it ends in s, x, z, ch, sh, o or consonant+y; for `THINGS` the singular is checked), plus a scan of all generated text for known bad plurals, and a test that pronouns are capitalized only at the start of a sentence and "1 apples" never occurs. (The new "Then she got ..." stories first read "Then She got"; that test caught it and it is fixed.)

**I6 fair_quota.** One public `fair_quota(capacity, total)` in `skills/types.py` (docstring added, same algorithm); `facts.py` and `patterns.py` import it and both private copies are gone. The error text is the one from facts (`cannot take N from M available`). New unit test of `fair_quota`, including the 150-over-5 example.

**M1 number options.** The answer's rank among the 4 sorted numbers is spread evenly (least-used feasible rank, ties random): word_problem ranks 12/12/13/13, count 13/13/12/12. Distractors come from the valid answer range (0-20 for word problems, 0-8 for counts) within 5 of the answer. Word-problem results are also uniform now (the result is drawn first, then a start), so answer values spread over 2..20 instead of a U shape.

**M2 pattern train mix.** Family chosen with probability proportional to sqrt(number of patterns). At about 99k training lines (18,000 docs): shares cycle 62.5%, step 14.9%, countdown 10.3%, letters 7.6%, double 4.7%; the most repeated line occurs 109 times (before, with equal weights: 1,790 times at the same scale, a double line).

**M3 shortcut audit.** `test_no_single_cheap_feature_predicts_the_answers`: limit 0.6 (0.65 for yes/no) for every predictor in the table below, plus tighter checks (syllogism "contains No" exactly 0.5, compare question-aware mention counter <= 0.5, word_problem predictors <= 0.4, count <= 0.45). "Answer prior" picks the option most often the answer in the *training text* (using the bench itself would overfit). A predictor that picks among several candidates counts as the chance of picking the right one.

**M4 count answers.** Answers go through 0..5 in turn: 9/9/8/8/8/8 (before, 3 was over-represented at 17/50). Training draws them uniformly.

## Shortcut audit, before (HEAD `a563e27`) and after, same seed and same predictors

| family | predictor | before | after |
|---|---|---|---|
| syllogism | "contains No" | **1.00** | 0.50 |
| syllogism | first premise says All | 0.80 | 0.52 |
| syllogism | answer prior | 0.50 | 0.50 |
| compare | once-mentioned name | 0.50 | 0.18 |
| compare | mentions by question (most-mentioned if "middle", else once-mentioned) | 0.50 | 0.38 |
| compare | most-mentioned / least-mentioned name | 0.00 / 0.00 (the absent odd name was a free elimination) | 0.31 / 0.18 |
| compare | first / last-mentioned name | 0.20 / 0.26 | 0.26 / 0.36 |
| compare | answer prior / most common answer slot | 0.34 / 0.26 | 0.23 / 0.28 |
| word_problem | smallest / 2nd smallest / 2nd largest / largest option | 0.02 / **0.46 / 0.48** / 0.04 | 0.24 / 0.24 / 0.26 / 0.26 |
| word_problem | option nearest the first number | 0.10 | 0.27 |
| word_problem | answer prior / most common answer slot | 0.32 / 0.30 | 0.22 / 0.30 |
| count | smallest / 2nd smallest / 2nd largest / largest option | 0.18 / **0.48** / 0.32 / 0.02 | 0.26 / 0.26 / 0.24 / 0.24 |
| count | answer prior / most common answer slot | 0.38 / 0.32 | 0.34 / 0.32 |

## Tests and results

- `tests/test_reasoning_patterns.py`: 33 tests in the default suite plus 1 `@pytest.mark.slow` (the 18,000-doc world-leak check, 7.4 s; a 3,000-doc version of the same check runs in the default suite). The brief's 3 tests are unchanged.
- `python -m uv run pytest tests/test_reasoning_patterns.py tests/test_kb_facts.py -q` -> `105 passed, 1 deselected in 14.68s`
- `python -m uv run pytest tests/test_reasoning_patterns.py -m slow -q` -> `1 passed, 33 deselected in 7.35s`
- Full suite `python -m uv run pytest -q` -> `630 passed, 4 deselected in 50.24s` (no warnings)
- `python -m uv run ruff check .` -> `All checks passed!`; the 5 files I touched are `ruff format` clean.

Mutation checks (temporary edits, restored; each fails the suite): train accepts reserved worlds; compare world key includes the question (variants not covered); syllogism decoy follows the answer (caught by the quantifier test and by the audit, "contains No" 0.76); compare never asks the middle; odd name not in the prompt (caught by the audit, 0.66); number ranks skewed to the middle; count answers skewed; "peach" back in the pool; pattern families equally likely; sloppy `fair_quota`; double limited to start 1-5; word-problem change off by one; capital "Then She".

## Files changed

- `ml/src/airace_ml/skills/reasoning.py` (rewritten around worlds, wordings and controls)
- `ml/src/airace_ml/skills/patterns.py` (4-7 terms, wider double, sqrt family weights, shared `fair_quota`)
- `ml/src/airace_ml/skills/types.py` (new public `fair_quota`)
- `ml/src/airace_ml/skills/facts.py` (uses `fair_quota`; private copy removed)
- `ml/tests/test_reasoning_patterns.py`

## Concerns

1. **Syllogisms are still solvable by lookup, as ruled.** The decoy is on fresh words, so exactly one premise mentions the queried class word, and its quantifier ("All" or "No") gives the answer without any chaining. That predictor scores **50/50 = 1.00** on the bench (it is not in the audit list, so the audit passes). A tiny model can learn "find the premise with the target word, read its quantifier". If the family should need the chain, a one-line change makes the decoy share the queried word with the opposite quantifier, e.g. yes/1 step `All a are b. No c are b. N is an a. Is N a b?` and yes/2 steps `All a are b. All b are c. No d are c. Is a a c?`. I did not change it because the ruling named fresh words; say the word and I will switch it and add the lookup predictor to the audit.
2. The partition mechanism differs from the ruled wording (world-key hash instead of listing 36 variant prompts); reasons and numbers are above. The guarantee is equal or stronger and the leak test is the one asked for.
3. Word problems with the same (operation, a, b) recur between train and bench (accepted as ruled).
4. The `double` name now also covers tripling.
5. Audit values are deterministic for the frozen seed; the closest to its limit are noise-level ones: syllogism "first premise says All" 0.52 (limit 0.65) and compare "last-mentioned name" 0.36 (limit 0.6).

---

# Fix round 1, rulings applied (syllogism decoy shares the queried word)

Status: DONE. Rulings 2-4 (world-key partition, recurring word-problem numbers, `double` covering times 3) were accepted; ruling 1 is implemented below. `pattern_bench_items` still gives exactly 150 items with all 5 families (step 41, cycle 41, countdown 41, letters 20, double 7).

## What changed

The decoy premise now mentions the queried class word with the opposite quantifier, so the single premise that mentions it no longer settles the answer and the chain has to be followed:

| answer, steps | puzzle (premises shuffled over all orders) |
|---|---|
| yes, 1 step | `All a are b. No c are b. N is an a. Is N a b?` |
| no, 1 step | `No a are b. All c are b. N is an a. Is N a b?` |
| yes, 2 steps | `All a are b. All b are c. No d are c. Is a a c?` |
| no, 2 steps | `All a are b. No b are c. All d are c. Is a a c?` |

- Same quantifier multiset whatever the answer is kept (1 step: All + No; 2 steps: All + All + No), and in both answers the queried word is the second word of exactly one All premise and one No premise. A one-step puzzle uses 3 made-up words, a two-step puzzle 4. Explanations still ignore the decoy. World key, partition and leak behaviour are unchanged (leak at 99,125 blocks: 0/50 for every family, `-m slow` test passes).
- **Audit:** new predictor "premise mentioning the queried word" (each premise that mentions the queried word votes yes if it says All, otherwise no; the vote share that is right counts). Before (`a7e762f`, fresh-word decoy): **1.00** (50/50 items). After: **0.50** (exactly, since two premises mention it, one All and one No). The test asserts it equals 0.5 and is below the chance + 0.15 limit.
- **Exactly one logically correct answer.** Besides the lookup solver (now reading "No x are y" symmetrically), the tests contain `logical_answers`, an independent model checker: it tries every way to fill a world of 3 things with the made-up classes ("All x are y" = x inside y, "No x are y" = nothing shared, the named thing is in its class, a generic question "Is a x a y?" for a non-empty class) and reports which answers hold in all models. For all 50 bench syllogisms and for all 2,000+ syllogism blocks of the 2,000-doc training fixture it returns exactly the answer in the item (no contradictory premises, no unsettled question). A control checks that an unsettled question ("All as are bs. Tom is a c. Is Tom a b?") yields no answer.

## Shortcut audit, final numbers (after this change; the bench items shifted because the syllogism draws consume the rng differently)

| family | predictor | before (HEAD `a563e27`) | after |
|---|---|---|---|
| syllogism | "contains No" | **1.00** | 0.50 |
| syllogism | premise mentioning the queried word | **1.00** | 0.50 |
| syllogism | first premise says All / answer prior | 0.80 / 0.50 | 0.52 / 0.50 |
| compare | once-mentioned name | 0.50 | 0.17 |
| compare | mentions by question (most-mentioned if "middle", else once-mentioned) | 0.50 | 0.37 |
| compare | most-mentioned / least-mentioned name | 0.00 / 0.00 | 0.33 / 0.17 |
| compare | first / last-mentioned name | 0.20 / 0.26 | 0.18 / 0.26 |
| compare | answer prior / most common answer slot | 0.34 / 0.26 | 0.15 / 0.38 |
| word_problem | smallest / 2nd smallest / 2nd largest / largest option | 0.02 / **0.46 / 0.48** / 0.04 | 0.24 / 0.24 / 0.26 / 0.26 |
| word_problem | option nearest the first number | 0.10 | 0.17 |
| word_problem | answer prior / most common answer slot | 0.32 / 0.30 | 0.35 / 0.36 |
| count | smallest / 2nd smallest / 2nd largest / largest option | 0.18 / **0.48** / 0.32 / 0.02 | 0.26 / 0.26 / 0.24 / 0.24 |
| count | answer prior / most common answer slot | 0.38 / 0.32 | 0.26 / 0.28 |

(All "after" values are within the limits: 0.6 for 4-option families, 0.65 for yes/no.)

## Tests and results

- `tests/test_reasoning_patterns.py`: 34 tests in the default suite plus 1 `@pytest.mark.slow`.
- `python -m uv run pytest tests/test_reasoning_patterns.py tests/test_kb_facts.py -q` -> `106 passed, 1 deselected in 14.63s`
- `python -m uv run pytest tests/test_reasoning_patterns.py -m slow -q` -> `1 passed, 34 deselected in 8.28s`
- Full suite `python -m uv run pytest -q` -> `631 passed, 4 deselected in 56.62s` (no warnings)
- `python -m uv run ruff check .` -> `All checks passed!`; the files I touched are `ruff format` clean.

Mutation checks for this change (temporary edits, restored; each fails the suite): decoy back on fresh words for one-step puzzles (quantifier-share test and audit fail); decoy back on fresh words for two-step puzzles (the audit fails: lookup 1.00); decoy with the same quantifier as the rule; the named thing put in the wrong class (logical-answer test, bench-answer test and train-block test fail); a two-step "no" whose premises no longer settle the answer (logical-answer test fails).

## Files changed in this round

- `ml/src/airace_ml/skills/reasoning.py` (syllogism world and module docstring)
- `ml/tests/test_reasoning_patterns.py` (model checker, new audit predictor, word-count check)

---

# Fix round 2 (count shortcut R1, syllogism one-off word R2)

Status: DONE_WITH_CONCERNS. Both findings are fixed. Two leftover syllogism cues need a ruling from the coordinator (Concerns 1-2).
Commit: `4db5e74 fix(ml): count and syllogism benches give nothing away: ...`

## What changed

**R1 count.** The old list held the asked word `t` times plus 1-8 random other words. So the asked word was the most frequent word in 74% of bench lists (71-72% at scale), and "answer = frequency of the most common word" scored 0.82. The new world (`_count_world`):
- The list has exactly 3 different words, each there a different number of times. The 3 numbers are a random 3-set of 1-5.
  - For answer `t >= 1`, the asked word is one of the 3; the other two numbers are a random pair from 1-5 without `t`.
  - For `t = 0`, all 3 are other words.
- So the pair (set of numbers, which word is asked) is uniform. The asked word is the most frequent, the middle or the least frequent word equally often: bench 13/41 most frequent, training 742/2304 = 0.32, was about 0.72. In 2/3 of the lists that contain it, another word is more frequent.
- Answers still go 0-5 in turn (9/9/8/8/8/8).
- No ties, on purpose. Scratch simulations (not committed) showed that when two words can share a number, two predictors gain: "the number two words share" (mode) scores 0.40-0.48, and "the number only one word has" also does better than chance. With 3 different numbers, every frequency-only predictor is the same as guessing one of the 3 words.
- **Options are now `0` and the 3 numbers in the list** (shuffled). Before, they were "the answer + 3 numbers within 5 of it, in 0-8".
  - The old options had a leak of their own: options 6-8 can never be the answer, because answers are 0-5. With a nearest-option predictor even a constant did well: "option nearest 4" scored 0.40 on the bench and 0.45 on 1,000 items at HEAD.
  - Now every option is possible: each is either "not there" or how often a listed word is there. Given the options, the answer is 0 with probability exactly 1/6 and each listed number with exactly 5/18 = 0.28. So no predictor that ignores where the asked word is beats 0.28 in expectation.
  - Options stay in 0-5, within 5 of the answer, and 4 distinct.
  - The answer's rank among the sorted options is 9/16/12/13 on the bench. Rank 0 is exactly the "answer 0" items, because 0 is always the smallest option. The existing 8-17 bounds hold.
- Lists are now 6-12 items (was 4-8). They have to be longer so that another word often beats the asked word even when it is there 4 or 5 times.
- `NUMBER_RANGE` now holds only `word_problem`. Word problems are unchanged.

**R2 syllogism.** The decoy's class now has a member of its own, a second person who is not the one asked about:
- One step: `All a are b. No c are b. N is an a. K is a c. Is N a b?`, and the "no" mirror.
- Two steps: `All a are b. All b are c. No d are c. K is a d. Is a a c?`, and the mirror.

What stays the same or is guaranteed:
- Every made-up word is in at least two sentences.
- The 4 sentences are shuffled over all 24 orders.
- The quantifier multiset is unchanged for yes and no: one step All + No, two steps All + All + No.
- The queried word is still in exactly one All premise and one No premise.
- Explanations ignore the decoy.
- `K` is a fresh name, different from `N`.
- The world key (sorted premises + question) now includes the member sentence. The partition is unchanged.

**Model checker re-run.** `logical_answers` and `entailed` now handle several named people: the person asked about is thing 0, and others are things 1-2. Two new controls: a fact about Kim settles nothing about Tom, and an unrelated second person does not change a settled answer.
- All 50 bench syllogisms have exactly one entailed answer, the item's. So does every syllogism block in the 2,000-doc training fixture (more than 2,000 blocks).
- The new structural test also checks, for every bench and training syllogism, that dropping the second person's sentence leaves the entailed answer the same (the mention is irrelevant).

## Shortcut audit, before and after

Before is HEAD `1db0d96`. The bench seed is frozen. A number predictor picks the option nearest its number (ties split), which reproduces the reviewer's 0.82.

| family | predictor | before | after | after, 1,000 items |
|---|---|---|---|---|
| count | frequency of the most common word (new) | **0.82** | 0.26 | 0.26 |
| count | frequency of any non-target word (new) | 0.31 | 0.00 | 0.00 |
| count | frequency of the least common word (new) | 0.29 | 0.32 | 0.28 |
| count | frequency of the first word in the list (new) | **0.60** | 0.18 | 0.29 |
| count | answer prior for the list length, fit on training (new) | **0.47** | 0.30 | 0.29 |
| count | answer prior / most common answer position | 0.26 / 0.28 | 0.18 / 0.32 | 0.17 / 0.26 |
| count | smallest / 2nd smallest / 2nd largest / largest option | 0.26 / 0.26 / 0.24 / 0.24 | 0.18 / 0.32 / 0.24 / 0.26 | 0.17 / 0.28 / 0.29 / 0.26 |
| syllogism | premise with a word occurring once is the decoy (new) | **1.00** | 0.50 | 0.50 |
| syllogism | contains No / premise mentioning the queried word | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| syllogism | first premise says All / answer prior | 0.52 / 0.50 | 0.52 / 0.50 | 0.49 / 0.50 |

- "Frequency of any non-target word" is 0.00 because no other word is ever there as often as the asked word. Using it requires finding the asked word first.
- The compare and word_problem designs are unchanged, and all their predictors are within limits. Their bench items shifted because the syllogism and count draws use the rng differently. Compare max: 0.38 ("most-mentioned name"). Word_problem max: 0.34.
- The test now asserts:
  - the two named count predictors <= 0.40;
  - every count predictor <= 0.40 (chance + 0.15; was 0.45);
  - the once-word syllogism predictor == 0.50 (limit 0.65).

## Tests and results

`tests/test_reasoning_patterns.py` has 36 tests in the default suite plus 1 `@pytest.mark.slow`. The brief's 3 tests are unchanged.

New tests:
- `test_every_made_up_word_of_a_syllogism_occurs_at_least_twice`, on bench and training. Every made-up word appears at least twice. In one-step puzzles the asked person appears twice. The second person appears once, only as a member of the decoy's class, and that class is in one more premise. Without that person the entailed answer is the same.
- `test_the_asked_word_is_just_one_of_the_words_in_the_list`, on bench and training. Lists have 3 different words with 3 different numbers in 1-5. Among the lists that contain the asked word, it is the most frequent, the middle and the least frequent each in 28-39%. Bench options are exactly 0 and the 3 numbers.

Changed tests:
- syllogisms have 4 sentences;
- count lists are 6-12 items;
- count options are in 0-5;
- the audit has the new predictors above.

Commands:
- `python -m uv run pytest tests/test_reasoning_patterns.py tests/test_kb_facts.py -v` -> `108 passed, 1 deselected in 10.79s`
- `python -m uv run pytest -m slow tests/test_reasoning_patterns.py` -> `1 passed, 36 deselected in 5.78s` (no benchmark world in 18,000 training docs)
- Full suite `python -m uv run pytest -q` -> `633 passed, 4 deselected in 40.44s` (no warnings)
- `python -m uv run ruff check .` -> `All checks passed!`. Both touched files are `ruff format` clean.

Invariants:
- 200 reasoning and 150 pattern bench items.
- Every bench prompt is reserved (brief test).
- No world-level leak (slow test).
- Sentences are natural. Examples: `Liz is a jorn. Max is a hup. All jorns are fenks. No hups are fenks. Is Max a fenk?` and `How many cows are in this list: frog, bird, bird, frog, cow, frog, frog, frog, bird, bird?`

## TDD evidence

RED: this round's test file against HEAD's `reasoning.py`, restored with `git show HEAD:...`:
```
$ cd ml && python -m uv run pytest tests/test_reasoning_patterns.py -q
E   AssertionError: assert (3 == 4)
E   AssertionError: Ana is a zorp. No sarns are tuks. All zorps are tuks. Is Ana a tuk?
E   AssertionError: assert 6 <= 4
E   AssertionError: How many figs are in this list: grape, apple, pear, apple?
E   AssertionError: ('syllogism', 'premise with a word occurring once is the decoy', 1.0)
FAILED ...::test_syllogisms_have_the_same_quantifiers_whatever_the_answer
FAILED ...::test_every_made_up_word_of_a_syllogism_occurs_at_least_twice
FAILED ...::test_count_answers_are_equalised
FAILED ...::test_the_asked_word_is_just_one_of_the_words_in_the_list
FAILED ...::test_number_options_are_possible_answers_near_the_answer
FAILED ...::test_no_single_cheap_feature_predicts_the_answers
6 failed, 30 passed, 1 deselected in 3.73s
```
These failures were expected:
- the decoy word occurs once ("sarn");
- lists have ties and a dominant asked word;
- options go up to 8;
- the audit hits 1.00 on syllogisms, and the count predictor is at 0.82 behind it.

The generalized model checker also passes on the old puzzles.

GREEN: `36 passed, 1 deselected in 4.30s` after the change, green on the first run, then the runs listed above.

Mutation checks: temporary edits, each restored afterwards; each one fails the suite.

| mutation | tests that fail |
|---|---|
| two-step decoy without its member | 3, including the audit |
| one-step decoy without its member | 3 |
| the decoy's member is the asked person (a contradiction) | 4, including the model checker |
| count numbers may repeat | 4 |
| asked word made the most frequent most of the time | 3, including the audit |
| count options back to "answer + 3 nearby in 0-8" | 3, including the audit |

## Files changed in this round

- `ml/src/airace_ml/skills/reasoning.py`: syllogism world, count world and options, docstrings.
- `ml/tests/test_reasoning_patterns.py`: model checker with several people, new audit predictors, 2 new tests, updated bounds.

## Concerns

1. **Two-step syllogisms: "an All-All chain means yes" scores 0.75** on the bench and on 1,000 items. This has been there since round 1 and is not one of the findings.
   - A two-step "yes" always has `All a are b. All b are c.`, and a two-step "no" never has two All premises that link.
   - It cannot be removed within the 3-quantifier templates. I checked every option: a linking All premise in the "no" case either empties a class, or leaves the queried word in only a No premise, which brings back the lookup shortcut.
   - Fixing it needs a fourth quantifier premise in two-step puzzles, for example an off-path All chain in "no" puzzles and a matching premise in "yes" ones.
   - The rule has to link two premises through a shared word, so it is not a one-feature lookup. Say the word and I will extend the templates.
2. **"The class of the once-mentioned person is the decoy" scores 1.00.**
   - The second person is mentioned once: names are not made-up words, and this is the construction the brief suggested.
   - A 3-step rule gets every item: find the once-named person, take their class, drop the premise with that class, and read the other premise with the queried word.
   - It mirrors the real chain and is no shorter, so I left it. A second mention of that person would need another sentence.
3. Count lists are longer: 6-12 items, was 4-8. Count options also changed meaning: they are now 0 and the listed numbers, the way compare's options are the listed names. R1 needs both changes, and the rank-test bounds still hold (9/16/12/13).
