# Task 18 report: personality fingerprint

**Status:** DONE_WITH_CONCERNS (concerns are judgment calls to confirm, none blocks)
**Commit:** `d29e0b7 feat(ml): measured personality fingerprint with population-relative descriptions`, on top of the controller's WIP commits `7e8ffb5` and `7f38f5b` (not amended or rewritten).

## What was implemented

All in `ml/src/airace_ml/personality/`:

- `__init__.py`: package docstring.
- `probes.json`: the frozen probe set, 60 probes, one per line. Open 20, factual 15, creative 10, help 10, opinion 5. Ids look like `open-01` and `factual-08`.
- `fingerprint.py`:
  - `TRAITS`, `Probe`, `load_probes()`, `FORMAL_WORDS`, `CASUAL_WORDS`, `POSITIVE_WORDS` (50, 42 and 51 words).
  - `Fingerprint` with `to_dict` and `from_dict`, `measure_fingerprint`, `TRAIT_PHRASES`, `describe` and `fingerprint_distance`.
  - Extras: `words()`, `PROBES_PATH` and a few named constants.

Design points, following the brief and the 10 rulings:

- **One batched call.** `measure_fingerprint` makes ONE `lm.generate` call with `len(probes) * k` prompts. They are `encode_chat(tok, [("user", text)], True)`, ordered probe by probe, and carry the given `seed`, `temperature`, `top_p` and `max_new_tokens`. It makes no call when there are no probes.
- **Replies and samples.** A reply is `tok.decode(tokens).strip()`. `samples` holds `{"probe", "kind", "sample", "text"}` per reply, JSON-safe.
- **Shared definitions.** `distinct_2` and `repetitiveness` are imported from `airace_ml.evals.creativity`, the public Task 17 functions. The Task 17 ledger says repetitiveness is "the same definition Task 18 uses", so the two scores cannot drift. `words()` is a local one-line regex copy of creativity's private `_words`; a test pins the two as identical.
- **Matching.** Answers are matched as word tuples against every run of 1..longest words of the reply. That gives whole-word and whole-phrase matching, case-insensitive. `cat` does not match `category`. `Port-au-Prince` matches `port au prince`. An answer with no letters (for example `"4"`) never matches.
- **Edge cases.**
  - `confidence` averages the per-reply mean over finite `top1_probs`, for replies with at least one token. A reply of only special tokens counts. All-NaN replies and empty replies are skipped. It is 0 when nothing qualifies.
  - `steadiness` is the mean over probes of the mean pairwise Jaccard, with Jaccard(empty, empty) = 1. With k = 1 there are no pairs, so it is 0.0. This is documented in the module and function docstrings. `k` must be an int >= 1 (bool and float rejected), otherwise `ValueError`.
  - `repetitiveness` and `boldness` are means over ALL replies; an empty reply counts as 0 repetitiveness and not bold.
  - `register` and `warmth` pool all replies. `warmth` uses `max(1, words)`.
- **`describe`.** Strict `<` and `>` against `np.percentile`, in `TRAITS` order. It returns `[]` for fewer than 5 fingerprints.
- **`fingerprint_distance`.** Population std with `ddof=0` and a floor of 1e-6. An empty population has no std, so it uses the floor.
- **Fingerprint round trip.** `from_dict` validates the trait set and rejects NaN, inf, bool and non-numbers.
- **Packaging.** `load_probes()` reads `Path(__file__).with_name("probes.json")` with explicit utf-8 and returns a fresh list every call. I built the wheel (`uv build --wheel`) and confirmed it contains `airace_ml/personality/probes.json`. Nothing in `pyproject.toml` changed.

## Factual probes (ruling 1)

Authored once from the KB with a scratch script, not shipped. Each probe text is a KB `question_templates` string filled in, so a test can map it back to its fact. `answers` is `kb.accepted_answers(fact)` and `wrong_answers` is `kb.wrong_objects(fact)`.

- 8 `capital_of`: France, Japan, Egypt, Canada, Kenya, Australia, Italy and the United States. The US probe also has the accepted forms "Washington D.C.", "Washington, DC" and so on.
- 4 `landmark_country`: Eiffel Tower, Taj Mahal, Machu Picchu and Mount Fuji.
- 3 `continent_of`: Kenya, Brazil and Japan.

Only relations with `mc_safe` and `exact_safe` were used. The subjects were chosen so that no subject word appears in its own wrong answers (so not Mexico, Panama and the like).

## Example probes per kind

| kind | examples |
|---|---|
| open | "Hello! How are you today?"; "What did you do this morning?"; "Tell me something about yourself." |
| factual | "Which city is the capital of France?" (answers `["Paris"]`, 147 wrong); "In which country would you find the Taj Mahal?" (`["India"]`, 21 wrong); "On which continent can you find Brazil?" (`["South America"]`, 5 wrong) |
| creative | "Tell me a joke."; "Write a short poem about the moon."; "Make up a name for a new kind of fruit and describe it." |
| help | "How do I make a sandwich?"; "Can you help me plan a birthday party?"; "How can I make new friends at school?" |
| opinion | "Do you think dogs or cats make better pets?"; "Is it better to read a book or watch a film?"; "What do you think is the best color?" |

## Traits measured for the brief's fake models

The `tiny_tok` fixture tokenizer, `ScriptedLM`, and the default sampling settings. Columns are the first 6 letters of each trait name.

| model | verbos | confid | invent | steadi | precis | boldne | slip_r | regist | warmth | repeti |
|---|---|---|---|---|---|---|---|---|---|---|
| talky `"well " * 40`, k=2 | 40.000 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.974 |
| terse `"ok"`, k=2 | 1.000 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | -0.992 | 0.000 | 0.000 |
| formal `"Therefore, however, additionally."`, k=1 | 3.000 | 0.500 | 0.017 | 0.000 | 0.000 | 1.000 | 0.000 | 0.994 | 0.000 | 0.000 |
| casual `"yeah lol gonna hey"`, k=1 | 4.000 | 0.500 | 0.017 | 0.000 | 0.000 | 1.000 | 0.000 | -0.996 | 0.000 | 0.000 |
| answering fake (`"It is {answer}."` for factual, else `"Hello friend."`), k=1 | 2.283 | 0.500 | 0.247 | 0.000 | 1.000 | 0.250 | 0.000 | 0.000 | 3.285 | 0.000 |
| empty model, k=2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Tests

`ml/tests/test_personality.py` holds the brief's 5 tests at the top, then 54 added tests: **59 tests in total, 1.8 to 2.1 s**.

The brief's tests are unchanged apart from `ruff check --fix` import ordering and `ruff format` line wrapping, mainly the long `assert ... and ...` lines. No assertion was weakened.

Added tests, grouped:

- **Probes.**
  - Ids and texts unique, ASCII, 2-14 words, round-trip through the tokenizer, and none contained in another (the fakes find probes by text). None equal a creativity story prompt.
  - `load_probes` returns fresh copies and works from any cwd. `probes.json` ships beside the module.
  - Factual probes agree with the KB:
    - Each probe maps to exactly one fact through the question templates.
    - The relation is `mc_safe` and `exact_safe`.
    - `answers == kb.accepted_answers(fact)`.
    - `wrong_answers` is a subset of `kb.wrong_objects(fact)`, with no overlap with the answers.
    - No wrong answer is inside an accepted form.
    - The subject's words are not in the wrong answers.
  - Every accepted form scores precision 1 and slip 0. Every wrong answer (about 1,270 of them) scores slip 1 and precision 0 through the real matcher.
- **Lexicons and phrases.**
  - Lexicons: frozensets of 30-60 lowercase ASCII alphabetic words, formal and casual disjoint, with the brief's test words.
  - Trait phrases: one pair per trait in `TRAITS` order, all 20 phrases distinct, no ML jargon, and the brief's example phrases exact.
  - `words()` equals creativity's `_words` on accents, emoji, CJK, digits and underscores.
- **Trait formulas on hand-computed replies.**
  - Verbosity: mean word count. Confidence: mean of per-reply means (0.6833, not the per-token 0.643), empty replies skipped, special-token replies counted, NaN and inf probabilities ignored.
  - Inventiveness: distinct-2 (0.75 and 0.5), with no bigram spanning two replies.
  - Steadiness: 0.75 and 5/9, plus empty/empty = 1, empty/word = 0, and **k = 1 gives 0.0**.
  - Precision and slip rate: 1/2 and 1/2, and slip rate counts only `wrong_answers` (a real wrong city that is not listed is not a slip). A reply can be both right and slipped. Both are 0 with no factual probes.
  - Matching: whole-word and phrase tests (cat vs category, concatenate, bobcat, cats, dogma; multi-word, hyphenated and punctuated phrases, and partial phrases rejected). An answer with no letters never matches.
  - Boldness: 2/6, with the boundaries at 2 words and 3 words and at repetitiveness exactly 0.5.
  - Register: 0.4, pooled (a mean of ratios would be 0.125).
  - Warmth: 6.0, pooled (a mean of ratios would be 3.75), with "!" counted and 30.0 for a no-word reply.
  - Repetitiveness: (4/7)/4, with empty replies counted.
- **One batched call.** One `generate` call. Seed, temperature, top_p and max_new_tokens are forwarded, and the defaults are 0, 0.8, 0.95 and 64. The prompts are exactly `encode_chat`, probe by probe. The complete samples list is compared for ids, kinds, sample indexes and text. Replies are decoded and stripped.
- **Errors, empty probes and a real model.**
  - `k` in (0, -1, 1.5, True, "3") raises, and a wrong reply count raises.
  - Empty probes give 0 traits without a model call.
  - The real `tiny_lm` is seeded and deterministic and the traits are finite.
- **Degenerate output (Review Focus 4).**
  - A silent model has steadiness 1.0 and every other trait 0.
  - Only special tokens give verbosity 0 and confidence 0.8.
  - Special-token text is plain text.
  - Endless repetition (`"the " * 300`) gives boldness 0, repetitiveness 1 - 1/298 and low inventiveness.
  - Eight garbage replies are finite: punctuation, replacement characters, non-ASCII, digits, whitespace, a 5,000-character word, underscores and a lone "!".
- **`describe`.**
  - Phrases come out in `TRAITS` order.
  - Strict comparison is checked with `np.nextafter` around the exact percentile; the custom 0/100 band behaves as intended.
  - A constant population says nothing unless the value differs. Fewer than 5 fingerprints give `[]`, and 5 work.
  - A backwards band raises.
- **Distance.**
  - A hand-computed 3-4-5 distance of 5.0 with `ddof=0` (a `ddof=1` std would give 3.54), and 2.5 when std is 2.
  - Symmetric, and 0 for itself, over a random population.
  - The std floor gives 0.5/1e-6 for a constant or empty population, and it stays finite.
- **Round trip.** `to_dict` is a copy and JSON-safe (also with `ensure_ascii=False`). `from_dict(json.loads(json.dumps(...))) == fp`, with a unicode sample. `from_dict` makes floats from ints and copies the samples. It rejects missing or extra traits, NaN and inf (`ValueError`), non-numbers and bool (`TypeError`), and bad samples.

### TDD evidence

**RED**, before any implementation existed (the brief's 5 tests alone):

```
$ cd ml && uv run --no-sync pytest tests/test_personality.py -v
tests/test_personality.py:4: in <module>
    from airace_ml.personality.fingerprint import (
E   ModuleNotFoundError: No module named 'airace_ml.personality.fingerprint'
ERROR tests/test_personality.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.08s ===============================
```

That failure was expected: the module did not exist.

**GREEN**, after writing `probes.json` and `fingerprint.py`:

```
tests/test_personality.py::test_probes PASSED
tests/test_personality.py::test_traits_respond_to_behavior PASSED
tests/test_personality.py::test_precision_with_answering_fake PASSED
tests/test_personality.py::test_empty_model_finite PASSED
tests/test_personality.py::test_describe_and_distance PASSED
5 passed in 0.18s
```

**The added tests were written after the implementation**, so they never ran red against a missing module. To show they actually bite, I mutation-tested the implementation:

- About 30 targeted mutation runs against the personality tests. They covered the empty-set Jaccard value, the register and warmth formulas, the boldness comparisons and the 3-word minimum, slip counting right answers, `any` vs `all` for precision, case folding, ddof, the std floor, both strict `describe` comparisons, the population minimum, repetitiveness skipping empties, confidence counting empties, char-based verbosity, distinct-1 vs distinct-2, prefix matching vs whole word, no strip, the `k` check, per-probe generate calls, a dropped seed, and off-by-one or lost sample fields.
- 2 survivors were genuine test gaps and are now fixed:
  - Boldness had no 2-word reply, so the 3-word minimum was untested.
  - The samples test only looked at an "open" probe, so a lost `kind` was not seen.
- 4 mutants that first appeared to survive were invalid (the replacement hit a docstring, or the edit was a no-op), and I redid them properly. Their replacements, and the 2 gaps above, are now all killed.
- Pooled vs per-probe steadiness is an equivalent mutant with a uniform `k`, so no test can separate them.
- The mutation script is in the scratchpad, not the repo. The file was restored after each run and compared byte for byte.

**Full suite and lint:**

```
$ uv run --no-sync pytest -q
1044 passed, 1 skipped, 6 deselected in 116.60s (0:01:56)   (an earlier run: 131.55s)
$ uv run --no-sync ruff check .
All checks passed!
```

## Files changed

- `ml/src/airace_ml/personality/__init__.py` (new)
- `ml/src/airace_ml/personality/fingerprint.py` (new)
- `ml/src/airace_ml/personality/probes.json` (new)
- `ml/tests/test_personality.py` (new)

No other file was touched.

## Self-review

- **Complete.** Every interface in the brief exists with the exact signatures, and every ruling is implemented.
- **YAGNI.** I removed a redundant `if pairs` filter. Beyond the brief I added only the `describe` band validation and `from_dict` validation.
- **Test pace.** The suite stays under 3 minutes (see concern 6).
- **No ML jargon** in the probes or the trait phrases, and every probe is fit for all ages. The 45 non-factual probes are hand-written, not copied from a dataset.

## Concerns (decisions to confirm)

1. **Ruling 1 conflicts with itself.**
   - It says to use only relations with `mc_safe` and `exact_safe`, and also gives "languages and animal homes" as examples.
   - But `country_language` and `animal_home` have `mc_safe=False` (overlapping answers, so a correct reply could be flagged as a slip). I followed the flag rule and excluded them.
   - `baby_animal` is flag-eligible but its objects include everyday words ("kid", "calf", "chick"), which would inflate `slip_rate`, so it is out too.
   - The replacement relations are capitals, landmark countries and continents. If you want languages or animal homes, the KB flags would have to allow it.
2. **A silent model has steadiness 1.0**, because the brief says Jaccard(empty, empty) = 1 and its own test needs `"ok"` replies to give 1.0. Every other trait is 0 for it, and `boldness` 0 and `verbosity` 0 are what mark it. If steadiness should read 0 for a model that says nothing, the brief's rule would have to change.
3. **"ok" and "okay" are in the casual lexicon**, so a terse model that always says "ok" reads as register -0.99. The brief's tests allow it. Dropping "ok" would make that model read as register 0.
4. **`warmth` has no upper bound** (a reply of only "!!!" gives 30.0). It is finite, as the brief's formula requires.
5. **Word-regex duplication.** `words()` copies creativity's private `_words` regex, and a test pins them identical. Making creativity's helper public would remove the copy, but that file belongs to Task 17.
6. **Suite time.** The full suite took 116 to 132 s on this restarted container, against the roughly 70 s noted in the dispatch. My 59 tests take about 2 s, and the slowest ones are in other modules, so the slowdown is the machine. It is still under the 3-minute limit, but with less margin than before.
7. **Small additions beyond the brief:**
   - `describe` raises `ValueError` for a backwards or out-of-range band.
   - `from_dict` raises `TypeError` for a non-numeric trait value (ruff's TRY004 asks for that) and `ValueError` otherwise.
   - A probe set whose text changes is a new measurement. The set is frozen, but a rename would silently change fingerprints; nothing records a probe-set version yet.
8. **Probe edit during work.** I shortened `creative-09` from 16 words to "Name a pet fish and tell me why you picked that name." so that all probes are 14 words or fewer. It was before the final commit, so the commit holds the final text.
9. **Process.** The container restarted once mid-task. The controller committed my in-progress files as `7e8ffb5` and `7f38f5b`; I committed the finished work on top as `d29e0b7` without rewriting either.
