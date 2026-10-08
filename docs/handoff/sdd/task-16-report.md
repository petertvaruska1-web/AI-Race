# Task 16 report: benchmark suite bench-v1

**Status:** DONE_WITH_CONCERNS
**Commit:** `20a6e53 feat(ml): bench-v1 suite with normalized category scores and tag breakdowns`
**Branch:** `claude/upbeat-franklin-k1192h` (not pushed)

## What I implemented

- **`ml/src/airace_ml/evals/scoring.py`**: pure arithmetic with no model calls.
  - `normalize(acc, chance)`. Returns 0.0 for chance ≥ 1 or NaN, so it never divides by zero and never returns NaN.
  - Exact matching (ruling 1): `extract_answer` (`first_item` / `first_line`), `normalize_answer` and `exact_match`.
    - `normalize_answer` strips surrounding spaces and quotes (`"'‘’“”`) and trailing `.!?;:,`, repeating until the text is stable so that both `"Paris".` and `"Paris."` work. It then collapses whitespace and casefolds.
    - A minus sign, brackets and inner commas are kept.
    - An empty normalized reply never matches.
  - `mean_logprob`, `mc_choice` and `pair_correct`.
    - MC takes the argmax of the mean log-prob per token, and the first option wins ties.
    - A pair is right only when the good sentence's total log-prob is strictly higher.
    - A NaN log-prob counts as −inf, and an empty continuation never wins.
  - `agreement(choices)`: the fraction of equal pairs.
  - `ItemResult`, `CategoryScore`, `category_score`, `BenchReport`, `to_dict()`, `tag_breakdown()` and `MIN_TAG_ITEMS = 10`.
- **`ml/src/airace_ml/evals/suite.py`**:
  - `CATEGORIES`, `SUITE_VERSION = "bench-v1"`, `Suite`, `build_suite()`, `run_benchmarks(...)`, with the exact brief signature.
  - Every category is built with `skill_rng(<category>, "bench")` and the brief's counts.
  - `build_suite` caches the built items with `functools.cache` and returns a `copy.deepcopy`, which takes 13.6 ms.
- **`ml/src/airace_ml/evals/__init__.py`**: a docstring only.
- **`ml/tests/fakes.py`**: `ScriptedLM` and `answer_key_lm`, following the brief and ruling 10.
  - The answer key is keyed by exact token tuples.
  - It encodes prompts, contexts and options on its own, so encoding bugs in `suite.py` cannot hide from it.
  - It raises `ValueError` when two items disagree on the same key. None do in bench-v1.
- **`ml/tests/test_bench.py`**: the 7 brief tests (9 cases) verbatim, plus 30 more tests (56 cases), for 65 in total.

## Scores per category

These are full-suite runs, with `run_benchmarks(lm, tiny_tok, suite=build_suite())`.

| Category | Answer-key model | Constant model (`ScriptedLM()`: every score 0.0, empty replies) |
|---|---|---|
| language | 100.00 (raw 1.000, n 200) | 0.00 (raw 0.000, n 200) |
| reasoning | 100.00 (raw 1.000, n 200) | 0.00 (raw 0.295, chance 0.3125, n 200) |
| pattern | 100.00 (raw 1.000, n 150) | 0.00 (raw 0.000, n 150) |
| knowledge | 100.00 (raw 1.000, n 200) | 0.00 (raw 0.170, chance 0.1875, n 200) |
| coding | 100.00 (raw 1.000, n 150) | 0.00 (raw 0.000, n 150) |
| consistency | 100.00 (raw 1.000, n 40 groups) | 0.00 (raw 0.225, chance 0.25, n 40) |
| instruction | 100.00 (raw 1.000, n 120) | 0.00 (raw 0.000, n 120) |
| **overall** | **100.00**, missing `["creativity"]` | **0.00**, missing `["creativity"]` |

- With `max_items_per_category=40`, the constant model scores reasoning 9.09 (raw 0.375), consistency 7.69 (raw 0.308, 13 groups) and 0 everywhere else. That is under the brief's limit of 25.
- For reference, an untrained real `TorchLM` (2 layers, d64, ctx 64, CPU) over the full suite takes 1.1 s and scores: language 5.0, reasoning 0.0, pattern 0.0, knowledge 0.3, coding 0.0, **consistency 86.7**, instruction 0.0. See concern A.

## Timings (cloud Linux CPU, `ScriptedLM`)

| What | Time |
|---|---|
| `build_suite()` first call (uncached build) | 0.661 s |
| `build_suite()` cached call (deepcopy), median of 10 | 13.6 ms (13.2–14.6) |
| `answer_key_lm(tok, suite)` construction | 0.051 s |
| `run_benchmarks` full suite, answer-key model, suite passed in, median of 5 | 87 ms |
| `run_benchmarks` full suite, constant model, suite passed in | 56 ms |
| `run_benchmarks` full suite, repetition replies (`"the the the "*30`) | 125 ms |
| `run_benchmarks` full suite, constant model, suite built inside (cached) | 80 ms |
| `tests/test_bench.py` (65 tests) | 2.9 s |
| Full default suite | 925 passed, 4 deselected, **68.7 s** (it was about 65 s before) |

## Interpretations and decisions (please confirm)

1. **`tag_breakdown()` keys are `"<category>/<tag>"`. This is the main interpretation in this task.**
   - The brief says "per tag". Counting a bare tag across categories breaks ruling 4's own example: `fam:double` is 7 pattern items (a number sequence) plus 5 coding items (a function). Merged, that is 12 ≥ 10, so it would be reported.
   - `topic:*` and `rel:*` tags are also shared between knowledge (accuracy), consistency (agreement) and instruction (checks). Merging them would average different measurements.
   - With per-category keys, the full suite reports 48 keys and no `fam:double`. `test_small_tags_never_reach_the_breakdown` pins this.
   - If you want cross-category topic discoveries, they can be computed from `items` later.
2. **`ScriptedLM.reply` receives the prompt token ids** (`Callable[[list[int]], str]`, as in the brief).
   - Ruling 10's "decodes prompt ids with `tok.decode`" is ambiguous. I took it to mean that a reply or score function decodes when it needs text, as my tests do.
   - The answer key keys by exact token tuples, which needs ids. Task 17's `lambda p: f"... {len(p)} ..."` works either way.
3. **Batching (ruling 9).** Each category makes one `score_continuations` call, plus one `generate` call per distinct `max_new_tokens`.
   - Coding makes 2 generate calls (outputs at 16 tokens, functions at 64). Every other category makes at most 1.
   - I read "per prompt kind" this way. One call at the largest budget would make the 100 output items decode 64 steps instead of 16.
   - `test_one_batched_call_per_category` pins this, computed from the suite.
4. **Empty categories** go into `missing`. They are not in `scores` and do not count towards `overall`.
   - This covers 0 items after limits, and a category absent from a custom suite.
   - `max_items_per_category=0` makes every category missing, with `overall == 0.0` and nothing NaN.
   - "At least one group" applies to limits ≥ 1.
   - A negative limit, or an unknown category, raises `ValueError`. A bare string such as `categories="language"` is also rejected.
5. **Creativity hook (ruling 11).**
   - When both `judge` and `novelty` are given, `_creativity` lazily imports `airace_ml.evals.creativity.score_creativity` and calls `score_creativity(lm, tok, judge, novelty, seed=seed)`. With either one missing, `"creativity"` goes into `missing`.
   - I did not create `creativity.py`. `test_creativity_hook` puts a fake module in `sys.modules` to pin the contract: the score and items are merged, it counts in `overall`, and it is called once.
   - If Task 17 moves the import to the top level, that test has to monkeypatch `suite.score_creativity` instead.
6. **Robustness beyond the brief (Review Focus 4):**
   - NaN or ±inf log-probs (a diverged model) give finite scores.
   - A reply made only of control-token ids (user, ai, sep, pad, r0) decodes to "" and scores 0.
   - An answer that normalizes to empty is never matched by an empty reply.
7. **`build_suite` deep-copies** instead of only copying the lists. The items are mutable dataclasses, and a test mutating `item.answers` would otherwise poison later runs. The mutation check showed that a shallow list copy fails 2 tests.
8. **`ItemResult.output`:**
   - Exact and Check items: the full decoded reply (R1: never rewritten).
   - MC: the chosen option's text.
   - Pairs and consistency groups: `None`.
9. **Consistency validation.**
   - Every consistency item must be a grouped `MCItem`, and a group needs at least 2 members. Otherwise `ValueError`.
   - A group's chance is the mean of its members' `1/len(options)`.
   - Tags are the union of member tags, in first-seen order.
10. **`ScriptedLM.generate`** does not cut replies to `max_new_tokens`, as the brief specifies. It returns `stopped=True`.

## Tests and results

- `uv run --no-sync pytest tests/test_bench.py -q` → **65 passed in 2.86 s**
- `uv run --no-sync pytest` → **925 passed, 4 deselected in 68.72 s**, with no warnings
- `uv run --no-sync ruff check .` → **All checks passed!**
- `ruff format --check` on the 5 new files → already formatted

The added tests pin rulings 1–12:
- exact match: 11 positive and 10 negative cases, including `"Paris."`/`"paris"`, curly quotes, `"-3"`≠`"3"` and `"[1, 2]"`≠`"1, 2"`
- `first_item` vs `first_line`
- MC argmax by mean, with ties going to the first option, both directly and through the runner with options of different token lengths
- pairs compared by total log-prob with a strict `>` after `[bos]`
- chat options without a leading space, and plain options with one
- generation: greedy, seeded, default stops, encode_chat for chat items vs encode_doc for plain ones
- one batched call per category, using a recording fake
- checkers receive the full reply and the identical, unmutated `check_args` object (170 items; the extra keys `instruction`, `not` and `min_words` are present)
- consistency agreement by text and the chance arithmetic (raw 2/3, chance 3/8, score 46.67), plus index-biased models getting text-based agreement
- limits take the first N in suite order, and consistency takes whole groups ((1, 1), (2, 1), (3, 1), (5, 1), (6, 2), (7, 2), (40, 13), (119, 39), (None, 40))
- empty categories; an `overall` mean of 50 from 100 and 0
- 5 degenerate repliers score exactly 0 on the full pattern, coding and instruction sets
- NaN/±inf log-probs; `to_dict` round-trips through JSON with `allow_nan=False` and non-ASCII outputs
- `tag_breakdown`'s minimum count and per-category keys
- `build_suite` copies; the suite equals the builders called on the bench streams
- suite validity: unique ids, distinct MC options, every Exact answer survives its own extraction, consistency groups share option sets
- the creativity hook, real-model reproducibility, and the defensive errors

**Mutation check.** I applied 30 deliberate bugs one at a time to `suite.py`/`scoring.py`, and every one made at least one test fail. Examples:
- options with or without the leading space
- MC by sum, or ties going last
- flat tag keys, or `MIN_TAG_ITEMS=1`
- stripping `-` or `[]`, no quote stripping, no casefold, no whitespace collapse, empty-matches-empty
- `first_item` treated as `first_line`
- consistency by index, or chance 0
- limits that cut groups, or no minimum of one group
- checkers getting a copy of `check_args`
- non-greedy generation, or an ignored seed
- a shared cache, or a shallow copy
- pair ties winning, or pairs by mean
- unguarded NaN or `normalize`
- the wrong pair context, or chat items encoded as documents
- `overall` over the requested categories
- one generate call per item

## TDD evidence

**RED.** I wrote the brief's tests verbatim, then ran `cd ml && uv run --no-sync pytest tests/test_bench.py -v`:
```
tests/test_bench.py:3: in <module>
    from airace_ml.evals.scoring import normalize
E   ModuleNotFoundError: No module named 'airace_ml.evals'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.09s ===============================
```
This was expected: the `evals` package and `tests/fakes.py` did not exist yet.

**GREEN.** After implementing `scoring.py`, `suite.py` and `fakes.py`, the same command gave:
```
tests/test_bench.py::test_normalize PASSED
tests/test_bench.py::test_suite_deterministic_and_sized PASSED
tests/test_bench.py::test_answer_key_scores_100 PASSED
tests/test_bench.py::test_constant_model_near_zero PASSED
tests/test_bench.py::test_degenerate_outputs[<lambda>0] PASSED
tests/test_bench.py::test_degenerate_outputs[<lambda>1] PASSED
tests/test_bench.py::test_degenerate_outputs[<lambda>2] PASSED
tests/test_bench.py::test_report_serializable_and_tags PASSED
tests/test_bench.py::test_real_tiny_model_end_to_end PASSED
============================== 9 passed in 1.17s ===============================
```
I wrote the additional tests after the implementation (they passed at once), so I showed they have teeth with the 30-mutation check above.

**Brief test formatting.** `ruff check` flagged only I001 (import order), which `ruff check --fix` fixed. The import block was then extended with the additional tests' imports and re-sorted.
- The brief's test bodies are byte-for-byte unchanged.
- They are fenced in `# fmt: off` / `# fmt: on` so that `ruff format` formats only the added tests and leaves the brief's compact style (semicolons, long lines) alone.
- No assertion was changed.

## Files changed

- `ml/src/airace_ml/evals/__init__.py` (new)
- `ml/src/airace_ml/evals/scoring.py` (new)
- `ml/src/airace_ml/evals/suite.py` (new)
- `ml/tests/fakes.py` (new)
- `ml/tests/test_bench.py` (new)

## Self-review

- **Completeness.** Every interface in the brief and every ruling (1–12) is implemented and tested. No `pyproject.toml`/`uv.lock` changes; no new dependencies.
- **Separation.** `scoring.py` is pure arithmetic (testable without a model); `suite.py` builds items and talks to the model through `LanguageModel` only.
- **Determinism.** No `random` or time-based logic in scoring. `time.perf_counter` is used only for `seconds`. Greedy replies are seeded, and a real model is reproducible across runs (tested).
- **R1.** Model output is stored as decoded and never rewritten. Normalization is used only for comparison.
- **YAGNI.** No sorting or bucketing of prompts by length, no exception swallowing around checkers (`BLE001` is on, and MiniPy already guards its errors), and no ADR or docs edits (left to the controller).
- **Memory (ruling, Task 5).** `TorchLM.batch_size` is left at its default. A category's full request list goes in one call and TorchLM chunks it by 32.

## Concerns

**A. The consistency metric rewards models that ignore the question (product validity, needs a ruling).**
- Untrained random-init models score **86.7 / 91.1 / 90.0** on consistency (init seeds 0/1/2, full suite), while their knowledge score is about 0.
- All paraphrases end in the same `"\nAnswer:"`, so a weak model's option preference depends almost only on the option text. It picks the same text in every paraphrase and agrees with itself, even though it gets all three paraphrases right in only 8–11 of 40 groups.
- I implemented the metric exactly as specified (brief table and ruling 5). The constant `ScriptedLM` stays at chance only because its ties pick index 0 and the options are shuffled.
- Cost if left: on the leaderboard, untrained or badly trained models would top consistency, and the "steadiness" discovery would praise question-blind models.
- Possible fixes, for you to choose:
  - (i) Pick consistency answers by calibrated scoring: each option's log-prob after the question minus its log-prob after a neutral context such as `"Answer:"`.
  - (ii) Normalize against the model's own question-blind agreement instead of `1/len(options)`.
  - (iii) Count a group only when the agreed answer is right, or weight agreement by correctness.
  - (iv) Add the "resamples" half of the spec (§4.7) with sampled answers.

**B. Tag key format (decision 1).** Please confirm `"<category>/<tag>"`. Downstream consumers (discoveries in Task 18 or later) should expect these keys.

**C. Minor, to verify locally once `tok-v1` exists.** With the 512-vocab test tokenizer, 2 knowledge answers and 2 instruction references are longer than their item's `max_new_tokens`:
- knowledge (budget 8): "Bandar Seri Begawan" is 12 tokens, and "Santo Domingo" is exactly 8.
- instruction (budget 64): two ALL-CAPS sentences are 65 and 67 tokens.
- The 4096-vocab `tok-v1` is far more compact, and a truncated all-caps reply still passes `all_caps`, so a perfect model should be unaffected.
- I couldn't check with the real tokenizer here. `ScriptedLM` doesn't truncate, so the answer-key test would not catch this.
- Suggested local check: for each Exact/Check item, `len(tok.encode(answer or reference)) <= max_new_tokens`.

---

## Pre-review changes (rulings A, B, C)

**Commit:** `6506b12 fix(ml): calibrate consistency choices against a neutral context; check reply budgets with tok-v1`. It sits on top of the controller's `4dcf1a4`. Not pushed.
**Status:** DONE_WITH_CONCERNS. Calibration is implemented as ruled, but it barely moves untrained real models (concern A2 below).

### What changed

**A: calibrated choice for consistency only.**
- `scoring.calibrated_choice(option_scores, neutral_scores)`:
  - Each option's gain is its mean log-prob per token after the paraphrase minus its mean log-prob per token after the neutral context. The answer is the option with the highest gain, and the first option wins ties.
  - A gain that is not finite on either side (NaN, ±inf, an empty continuation) counts as −inf, so it never wins.
  - Mismatched lengths raise `ValueError`.
- `suite.NEUTRAL_CONTEXT = "Answer:"`. The neutral context is `encode_doc(tok, NEUTRAL_CONTEXT)`, with the same continuation tokens as the member's option (a leading space for plain items).
- `_ask(..., calibrate=...)`:
  - `_run_category` passes `calibrate=category == "consistency"`. Knowledge and reasoning keep `mc_choice`.
  - Each category still makes one `score_continuations` call. Requests are now deduplicated by `(context, continuation)`, so each neutral pair is asked once per category.
  - For the full consistency set, the call holds 480 paraphrase pairs plus one neutral pair per distinct option text. That is fewer than 480, and the test asserts it.
- Agreement, chance (the mean of `1/len(options)`) and normalization are unchanged from ruling 5.
- `answer_key_lm` already scores unknown pairs at −10, including every neutral pair, so its behaviour is unchanged.
  - Correct option: gain is 0 − (−10/n) > 0.
  - Wrong option: gain is −10/n − (−10/n) = 0.
  - So the correct option wins and the answer key stays at 100. I updated its docstring to say this.
- The `suite.py` module docstring documents the rule, including what it does not cancel.

**B: no code change.** The `tag_breakdown()` docstring already documents `"<category>/<tag>"` keys and the per-category `MIN_TAG_ITEMS`.

**C: reply budgets under the real tokenizer.**
- `_reply_budget_overruns(tok)` lists every Exact answer (each accepted form) and every Check reference whose token count exceeds the item's `max_new_tokens`.
  - Plain Exact answers are counted both bare and with a leading space, since a reply after `Answer:` usually starts with one.
  - Check references are counted as they stand.
- `test_every_answer_fits_its_reply_budget_with_the_real_tokenizer` runs this with `Tok.load(tokenizer_path())`. It is marked `@pytest.mark.skipif(not tokenizer_path().exists(), reason="needs the real tok-v1 tokenizer")`, so it is skipped here and runs on the owner's machine once `data/tokenizer/tok-v1/tokenizer.json` exists.
- `test_reply_budget_check_finds_overruns` runs the same check on the 512-vocab test tokenizer and must find the known overruns (`knowledge-0168`, `instruction-0004`, `instruction-0053`). This proves the check has teeth.
- I also ran the skipped test against a tokenizer: `AIRACE_DATA=<scratch root with the 512-vocab tokenizer> pytest -k real_tokenizer` fails and lists 4 overruns, starting with `('knowledge-0168', 'Bandar Seri Begawan', 12, 8)`.

### Tests

Added:
- `test_calibrated_choice_subtracts_the_neutral_score`: the gain beats the plain argmax, ties go to the first option, and NaN/±inf/empty on either side never wins.
- `test_consistency_ignores_a_question_blind_option_prior`: the fake from ruling A1 gives each option text a fixed, distinct log-prob whatever the context.
  - The test first proves that the plain rule would make this model agree with itself (mean agreement > 0.9).
  - It then asserts the consistency score is ≤ 25. It is actually 0.0: every gain is 0, so the model picks the first option, which matches the constant model.
- `test_consistency_rewards_answers_that_come_from_the_question[the right answer | the same wrong answer]`: the same strong text prior, plus a +2 lift that the question gives one option.
  - Variant one lifts the correct option; variant two lifts the same wrong option in every paraphrase.
  - Both score `CategoryScore(100.0, 1.0, 40)`.
- `test_only_consistency_is_calibrated`: reasoning and knowledge make 0 neutral requests, and consistency makes exactly one per distinct option text.
- `test_reply_budget_check_finds_overruns` and the skipped real-tokenizer test described under C.

Updated:
- `test_one_batched_call_per_category` now expects the distinct `(context, continuation)` pairs, with consistency's neutral pairs.
- `test_consistency_agreement_by_text_and_chance`: its scripted score now handles the neutral context with `.get`. The expected numbers are unchanged.

**Mutation check (7 more).** Every one is caught:
- calibrating every category, or never calibrating
- no deduplication
- a different neutral context (`"Answer: "`)
- the gain sign reversed
- non-finite gains allowed
- calibrated ties going last

**Dropped variant (reported, not committed).** I also tried a stronger fake: the text prior plus a tiny deterministic wobble that depends on the context.
- It scored 35.6 (raw 0.517), above chance.
- The neutral context's own wobble is shared by all three paraphrases, so subtracting it acts as a new text bias.
- It goes beyond the ruled fake and fails under the ruled design, so I removed it and report it as a finding.

### Commands and output

```
$ cd ml && uv run --no-sync pytest tests/test_bench.py -q        # RED 1: before calibrated_choice existed
E   ImportError: cannot import name 'calibrated_choice' from 'airace_ml.evals.scoring'
1 error in 0.18s
$ uv run --no-sync pytest tests/test_bench.py -q                 # RED 2: calibrated_choice added, runner unchanged
E       assert 100.0 <= 25
E        +  where 100.0 = CategoryScore(score=100.0, raw=1.0, n=40).score
FAILED tests/test_bench.py::test_one_batched_call_per_category
FAILED tests/test_bench.py::test_consistency_ignores_a_question_blind_option_prior[0.0]
FAILED tests/test_bench.py::test_consistency_ignores_a_question_blind_option_prior[1.0]
FAILED tests/test_bench.py::test_only_consistency_is_calibrated
4 failed, 68 passed, 1 skipped in 3.19s
$ uv run --no-sync pytest tests/test_bench.py -q                 # after the runner change (before dropping [1.0])
E       assert 35.555555555555564 <= 25
FAILED tests/test_bench.py::test_consistency_ignores_a_question_blind_option_prior[1.0]
1 failed, 71 passed, 1 skipped in 3.09s
$ uv run --no-sync pytest tests/test_bench.py -q -rs             # GREEN (ruled fake only)
SKIPPED [1] tests/test_bench.py:749: needs the real tok-v1 tokenizer
71 passed, 1 skipped in 3.08s
$ uv run --no-sync pytest
=========== 931 passed, 1 skipped, 4 deselected in 67.89s (0:01:07) ============
$ uv run --no-sync ruff check .
All checks passed!
$ uv run --no-sync ruff format --check src/airace_ml/evals tests/fakes.py tests/test_bench.py
5 files already formatted
```

### Scores after the change

- Answer-key model: still 100.00 in all 7 categories, consistency included.
- Constant model: still 0.00 in all categories (consistency raw 0.225). At 40 items per category: reasoning 9.09, consistency 7.69.
- `run_benchmarks` with `ScriptedLM` over the full suite: 87 ms (answer key) and 58 ms (constant), essentially unchanged.

**Consistency of untrained real models, before and after.** Each is a `TorchLM` with 2 layers, d64, ctx 64, the 512-vocab test tokenizer, CPU, full suite.

| Init seed | Before (plain choice) | After (calibrated, `"Answer:"`) | Knowledge | Full-suite time |
|---|---|---|---|---|
| 0 | 86.7 (raw 0.900) | **83.3** (raw 0.875) | 0.3 | 1.02 s |
| 1 | 91.1 (raw 0.933) | **62.2** (raw 0.717) | 0.0 | 0.96 s |
| 2 | 90.0 (raw 0.925) | **77.8** (raw 0.833) | 0.0 | 1.06 s |

No test depends on these random-init numbers.

### Concern A2: calibration with `"Answer:"` does not make question-blind real models score near chance

- Untrained models still score 62–83 on consistency while knowledge is about 0.
- Calibration cancels only the part of a preference that is the same in every context. A weak or random network's choice depends on the whole context, and the 3 paraphrases share most of their tokens: the subject, `Question:`, `?\nAnswer:`. Each paraphrase therefore differs from the others far less than from `<|bos|>Answer:`, and the leftover bias is common to all three, so they still agree.
- Measured, not committed: neutral contexts that match the paraphrase frame help more but are still far from chance.

  | Init seed | `"Question: ?\nAnswer:"` | `"Question:\nAnswer:"` |
  |---|---|---|
  | 0 | 65.6 | 63.3 |
  | 1 | 62.2 | 63.3 |
  | 2 | 40.0 | 48.9 |

- Agreement among paraphrases of the same subject seems to reward a model that is smooth in its input, whether or not it knows anything. Options for a follow-up ruling:
  - (i) Gate on correctness: a group counts only if its members agree on the correct answer, or use the conditional consistency P(all three right | at least one right).
  - (ii) Weight agreement by how far knowledge accuracy is above chance.
  - (iii) Normalize against a control: the model's agreement on paraphrase triples whose subject is swapped, keeping the frame and changing the fact.
  - (iv) Accept the metric and caption it in the UI as "steadiness of answers", not knowledge.
- Until then, an untrained model will still post a high consistency score.

---

## Pre-review change 2 (revised consistency ruling)

**Commit:** `3dfc3ed fix(ml): score consistency as robust knowledge (every paraphrase right), drop calibrated choice`. It sits on top of the controller's `d2a8070`. Not pushed.

### What changed

**Calibration removed.**
- Gone: `scoring.calibrated_choice`, `suite.NEUTRAL_CONTEXT`, the neutral-context requests and the `calibrate` flag.
- The request dedup is gone too. Without neutral pairs, bench-v1 has no repeated (context, continuation) pairs, so it no longer earned its place.
- `_ask` and `_judge_scores` are back to exactly their `20a6e53` form: one `score_continuations` call per category, built from spans.
- Each consistency member uses the plain MC rule: the argmax of mean log-prob per token, with the first option winning ties.

**New group rule.** In `_consistency`:
- A group's score is `float(all(member outcome == 1.0))`, i.e. 1 only if every paraphrase chose its own `answer_index`.
- `raw` is the mean over groups.
- Chance is the mean over groups of the members' mean `1/len(options)`.
- `score = normalize(raw, chance)`.
- `n` is the number of groups.
- ItemResults are still one per group: the group id, unioned tags, the 0/1 score and `output=None`.
- A group with fewer than 2 members, or an ungrouped item, still raises `ValueError`.

**`scoring.agreement` removed.** Nothing uses it any more.

**Docstrings.**
- `suite.py` module docstring: the table row, plus a paragraph on why chance is `1/options`. That is the expected score of a guesser that keeps one option text across all paraphrases; one that guesses each paraphrase afresh scores below it. So a question-blind model normalizes to about 0, and agreement-only rules can be gamed.
- `ItemResult`: 1 or 0, and a group counts as right only when every paraphrase is.
- `tag_breakdown`: in consistency, a `topic:` tag counts whole groups.
- `answer_key_lm`: it answers every paraphrase correctly, so consistency scores 100. Its behaviour is unchanged.

### Tests

Removed, since the rule they pinned is gone: `test_agreement_arithmetic`, `test_calibrated_choice_subtracts_the_neutral_score` and `test_only_consistency_is_calibrated`. `test_one_batched_call_per_category` is back to plain counts (480 consistency pairs, no neutral ones).

New or rewritten:
- `test_consistency_group_is_right_only_when_every_paraphrase_is` (the arithmetic). It uses 4 groups:

  | Group | Options | Answers given | Group score |
  |---|---|---|---|
  | a | 4 | all right | 1 |
  | b | 2 | right on 2 of 3 | 0 |
  | c | 4 | the same wrong answer every time | 0 |
  | d | 2 | all right | 1 |

  - It expects raw = 0.5, chance = (¼ + ½ + ¼ + ½)/4 = 0.375 and score = `normalize(0.5, 0.375)` = 20.
  - It checks per-group ItemResults: ids, 0/1 scores, unioned tags `("x", "y", "z")`, and `output=None`.
  - A constant model is right only in group d, where every paraphrase lists the answer first.
- `test_consistency_gives_a_question_blind_option_prior_chance_level`: the option-text-prior fake.
  - The test proves the fake picks the same text in every paraphrase of all 40 groups.
  - It asserts a score ≤ 25. The measured score is **0.00** (raw 0.250 = 10/40 groups, which is chance).
- `test_consistency_needs_every_paraphrase_right`: a strong text prior, outweighed by a +20 lift that the question gives one option.

  | Variant | Expected |
  |---|---|
  | the right answer | `CategoryScore(100.0, 1.0, 40)` |
  | right but once wrong (wrong on each group's third paraphrase) | `CategoryScore(0.0, 0.0, 40)` |
  | the same wrong answer | `CategoryScore(0.0, 0.0, 40)` |
- `test_an_untrained_model_is_not_consistent`: the untrained `tiny_lm` fixture scores ≤ 50 on all 40 groups.
  - This is a robust bound. The measured score is 0.0; theory says about 0, since a group is right by luck about 1 time in 4. Agreement-only rules gave 62–91.
  - It guards against regressing to an agreement-only rule.

**Mutation check.** I tried 5 mutations of the new rule, and every one was caught:
- any member right
- the fraction of members right
- the old text-agreement rule
- chance 0
- independent-guesser chance, `(1/k)^3`

### Commands and output

```
$ cd ml && uv run --no-sync pytest tests/test_bench.py -q        # RED: new tests, calibrated code still in place
E           assert [('score', 575)] == [('score', 480)]          # (consistency still sent neutral pairs)
E       KeyError: (1, 48, 93, 98, 102, 278, 41)                    # (neutral context sent to fakes that know only paraphrases)
FAILED tests/test_bench.py::test_one_batched_call_per_category
FAILED tests/test_bench.py::test_consistency_group_is_right_only_when_every_paraphrase_is
FAILED tests/test_bench.py::test_consistency_needs_every_paraphrase_right[the right answer-100.0]
FAILED tests/test_bench.py::test_consistency_needs_every_paraphrase_right[right but once wrong-0.0]
FAILED tests/test_bench.py::test_consistency_needs_every_paraphrase_right[the same wrong answer-0.0]
5 failed, 64 passed, 1 skipped in 3.06s
$ uv run --no-sync pytest tests/test_bench.py -q -rs             # GREEN
SKIPPED [1] tests/test_bench.py:710: needs the real tok-v1 tokenizer
70 passed, 1 skipped in 2.90s
$ uv run --no-sync pytest
=========== 930 passed, 1 skipped, 4 deselected in 69.31s (0:01:09) ============
$ uv run --no-sync ruff check .
All checks passed!
$ uv run --no-sync ruff format --check src/airace_ml/evals tests/fakes.py tests/test_bench.py
5 files already formatted
```

### Scores after the change

- Answer-key model: 100.00 in all 7 categories. `run_benchmarks` takes 82 ms over the full suite.
- Constant model: 0.00 in all categories. Its consistency raw is now 0.000 (it was 0.225 under agreement), and `run_benchmarks` takes 56 ms.
  - At 40 items per category: reasoning 9.09; consistency is now 0.00 (it was 7.69).

**Untrained real models.** Each is a `TorchLM` with 2 layers, d64, ctx 64, the 512-vocab test tokenizer, CPU, full suite.

| Init seed | Plain agreement (ruling 5) | Calibrated agreement (ruling A) | **Every paraphrase right (revised)** | Groups right | Knowledge |
|---|---|---|---|---|---|
| 0 | 86.7 | 83.3 | **0.0** (raw 0.250) | 10/40 | 0.3 |
| 1 | 91.1 | 62.2 | **0.0** (raw 0.200) | 8/40 | 0.0 |
| 2 | 90.0 | 77.8 | **3.3** (raw 0.275) | 11/40 | 0.0 |

Their overall scores are now 0.76 / 0.57 / 1.88; under ruling 5 they were about 13. Concerns A and A2 are resolved by this ruling. Concern C (the real-tokenizer budget test) is unchanged and still waits on `tok-v1`.

---

## Fix round 1

**Commit:** `10b1edf fix(ml): instruction chance is the best constant-reply pass rate; consistency groups show their choices`. It sits on top of the controller's `276013d`. Not pushed.

### I1: the instruction chance is a measured floor

What changed:
- **`scoring.constant_reply_floor(items, replies)`**: the best pass rate over `items` of any single reply in `replies` given to every item. It looks up `CHECKERS[item.check](reply, item.check_args)` at call time, dedupes replies, and returns 0.0 when there are no items or no replies.
- **`suite.CONSTANT_REPLIES`**: `""`, `"yes"`, `"no"`, `"Yes."`, `"No."`, `"ok"`, `"I don't know."`, `"1, 2, 3"`, `"a, b and c"`, `"HELLO"` (the all-caps word) and `"the"` repeated 40 times (the long repeated word).
- **`_run_category`, instruction category only**:
  - The floor is computed over the instruction `CheckItem`s actually scored (after `max_items_per_category`).
  - Its candidates are `CONSTANT_REPLIES` plus every scored item's own `reference`.
  - Each checked item's chance becomes the floor, so `score = normalize(raw, floor)`.
  - Item scores stay 0/1, and the checkers are unchanged.
  - Coding's checked items and all Exact items keep chance 0.
- **Docs**: the chance table now says "0; instruction: the floor below", and a module-docstring paragraph explains the rule and why it exists.

Measured on the full suite:
- **Floor = 0.1583 (19/120)**. It is set by a reference used as a constant reply: "ZOOLOGISTS PUT THE CHIMPANZEE IN THE MAMMAL GROUP OF ANIMALS." passes the 18 all-caps items and 1 more. The fixed replies alone reach 0.1417 (17/120).
- The floor follows the item limits. Over the first 5, 10, 20 and 40 items it is 0.200, 0.300, 0.200 and 0.175.
- A constant `"Yes"` now scores 0.00 on instruction (raw 0.1417), where the review measured 14.2.
- The answer key still scores 100.00.
- A format-cue reader at the review's raw 0.258 would now score about 11.8, not 25.8. Reading the format cue is a legitimate part of instruction following.
- `run_benchmarks` over the full suite with `ScriptedLM` now takes 235–278 ms, up from 56–87 ms. Computing the floor costs about 0.19 s per full instruction run (115 candidates × 120 items of checker calls).

Covering tests:
- `test_constant_reply_floor_arithmetic` (unit, 4 hand-made items). "yes" gives 0.5; "Paris" or "1, 2, 3" gives 0.25; the best of several replies is what counts; "" and "maybe so" give 0; no items or no replies give 0.
- `test_instruction_chance_is_the_constant_reply_floor`: 3 of 4 right against a floor of 0.5 gives `CategoryScore(50.0, 0.75, 4)`. The same items under `coding` give `CategoryScore(75.0, 0.75, 4)`, so coding keeps chance 0.
- `test_no_constant_reply_beats_the_instruction_floor`, 17 cases over the full instruction set (120 items). Each of the 11 `CONSTANT_REPLIES`, the reviewer's "Paris", "True", "0", "None" and "A", and the best reference used as a constant must score exactly 0.
  - The best reference is the highest-scoring of all references, so every other reference scores 0 too.
  - Before the fix these cases failed with scores of 14.17, 3.33 and 11.67.
- `test_instruction_floor_follows_the_items_run`: with `max_items_per_category=10`, the best reference among those 10 scores `CategoryScore(0.0, 0.3, 10)`. A floor computed over all 120 items would give it a positive score.
- `test_instruction_floor_includes_the_fixed_replies`: in a 3-item suite where every reference fails the others, a constant "ok" (raw 2/3) must score 0. This needs the fixed replies; the references alone give 1/3.
- `test_answer_key_scores_100`: instruction is still 100.
- `test_checkers_get_the_full_reply_and_untouched_check_args` was restructured, because the floor now also calls the checkers. It checks that every item's own reply reaches its checker with the identical `check_args` object, that all args stay unmodified, and that no call ever gets a copy.

### M1: `tag_breakdown()` values are raw accuracy

The docstring now says the values are raw accuracy (the share of a tag's items that are right, × 100), not normalized against chance like category scores. For example, a tag of 4-option items sits at 25 by luck.

### M2: consistency groups show their choices

- A consistency group's `ItemResult.output` is now the members' chosen option texts, in member order, joined by `" | "`, e.g. `"yes | yes | no"`.
- The `ItemResult` docstring is updated.
- Covered by `test_consistency_group_is_right_only_when_every_paraphrase_is`, which expects `"Paris | Paris | Paris"`, `"yes | yes | no"`, `"6 | 6 | 6"` and `"no | no | no"`.

### M4: two fragile tests fixed

- `test_one_batched_call_per_category` no longer asserts on the leftover loop variable. It records the generate calls per category and asserts the whole map: `{language 0, reasoning 0, pattern 1, knowledge 1, coding 2, consistency 0, instruction 1}`.
- `test_reply_budget_check_finds_overruns` now asserts a property instead of tokenizer-dependent ids. It must find at least one overrun, and every reported overrun must name a real item, carry its `max_new_tokens`, really exceed it, and quote one of that item's accepted answers or its reference.

### Mutation check (11, all caught)

| Mutation | Caught by |
|---|---|
| No floor (chance 0) | 16 failures |
| Floor applied to coding too | the coding assertion |
| Fixed replies only | the best-reference and limits tests |
| References only | `test_instruction_floor_includes_the_fixed_replies`; the fix round's first version missed this, so I added that test |
| Floor computed over the whole suite instead of the items run | the limits test |
| Mean instead of best | the floor tests |
| First reply only | the floor tests |
| Consistency output dropped | the consistency arithmetic test |
| Consistency output reordered | the consistency arithmetic test |

Two of the 11 were rewrites of a malformed and a surviving first attempt, so the table has 9 rows.

### Commands and output

```
$ cd ml && uv run --no-sync pytest tests/test_bench.py -q     # RED 1: tests written first
E   ImportError: cannot import name 'constant_reply_floor' from 'airace_ml.evals.scoring'
1 error in 0.19s
$ uv run --no-sync pytest tests/test_bench.py -q              # RED 2: floor function added, runner not wired
E       assert (14.166666666666666 == 0)        # "yes", "no", "Yes.", "No.", "ok", "HELLO", "Paris", "True", "0", "None", "A"
E       assert (3.3333333333333335 == 0)        # "1, 2, 3", "a, b and c"
E       assert (11.666666666666666 == 0)        # best reference
FAILED tests/test_bench.py::test_consistency_group_is_right_only_when_every_paraphrase_is   # M2
FAILED tests/test_bench.py::test_instruction_chance_is_the_constant_reply_floor
FAILED tests/test_bench.py::test_no_constant_reply_beats_the_instruction_floor[...]   (14 cases)
16 failed, 73 passed, 1 skipped in 3.92s
$ uv run --no-sync pytest tests/test_bench.py -q -rs          # GREEN
SKIPPED [1] tests/test_bench.py:737: needs the real tok-v1 tokenizer
91 passed, 1 skipped in 8.49s
$ uv run --no-sync pytest
=========== 951 passed, 1 skipped, 4 deselected in 73.58s (0:01:13) ============
$ uv run --no-sync ruff check .
All checks passed!
$ uv run --no-sync ruff format --check src/airace_ml/evals tests/fakes.py tests/test_bench.py
5 files already formatted
```

The default suite takes 73.6 s, well under 3 minutes. `test_bench.py` grew from 2.9 s to 8.5 s, mostly from the 17 full-set constant-reply runs at about 0.2 s each.
