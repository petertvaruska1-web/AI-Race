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
