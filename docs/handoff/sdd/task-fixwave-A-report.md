# Fix wave group A: report

Branch `claude/upbeat-franklin-k1192h`. Base (commit just before the first `fix(A)` commit): `d5b12ba`.
Head: `fee3b02`. Nothing pushed. All commands run from `/home/user/AI-Race/ml` with `uv run --no-sync`.

| Finding | Status | Commits |
|---|---|---|
| F1 function prompts end before the body's newline | FIXED | 40381f0, c410d81 (docs) |
| F5 knowledge and consistency answers follow KB base rates | FIXED (see the shape_sides limit) | c8dea8b, fee3b02 (test/docs) |
| F24 knowledge free answers 8-token budget | FIXED | 48d3305 |
| F44 two brief tests check nothing | FIXED | 288702e |
| F53 animal-class plurals | FIXED | 9024daf |
| F54 planet-order bare digit | FIXED | c3e74f5 |
| F25 pin the bench-v1 item set | FIXED (last, after every item change) | c3f4145 |

Suite: default `1262 passed, 1 skipped, 8 deselected in 117 s` (baseline before the group:
`1247 passed, 1 skipped in 127 s`; +15 tests), no warnings. `ruff check .`: clean. Slow suite
`pytest -m slow -q` (run at c3f4145, before the two docs/test-only follow-ups): `5 passed, 1266
deselected in 289 s`.

## F1: function-completion prompts

**Reproduced.** Under a byte-level BPE trained with `train_tokenizer` (default 4096 cap, 1482 entries
reached) on 3000 `code_train_docs`, the merge `'ĊĠĠĠ'` (id 276 vs bare newline 214)
exists, and for 50 of 50 function items the prompt tokens were not a prefix of the training-text tokens.
No non-chat prompt other than the 50 function prompts ended in whitespace.

RED (before the fix):
```
uv run --no-sync pytest tests/test_bench.py -q -k "no_plain_bench_prompt or prefix_of_the_training or answer_key_replies_to_a_function"
E  assert ['coding-0002...ng-0017', ...] == []      (50 prompts end in whitespace)
E  assert [1, 302, 540,... 653, 23, ...] == [1, 302, 540,... 653, 23, ...]
E    At index 21 diff: 276 != 214                    (bare newline vs merged newline+indent token)
E  assert '    return m + n\n' == '\n    return m + n\n'
3 failed
```
GREEN: same command, `3 passed` (`tests/test_bench.py` 94 passed, 1 skipped; `test_code_instr_grammar.py` 88 passed).

**Fix (as ruled).** `code.py::_bench_functions`: `CheckItem.prompt = v.prompt.rstrip("\n")`;
`check_args["prompt"]` keeps the full header. `_Variant.prompt` and all training docs are unchanged (no
corpus rebuild). `function_body` is unchanged: it already accepts a reply starting with a newline.
`tests/fakes.py`: new `reference_reply(item)` (`"\n" + reference` for a non-chat check item);
`answer_key_lm` uses it.

New tests (tests/test_bench.py):
- `test_no_plain_bench_prompt_ends_in_whitespace` (every non-chat, non-pair item; asserts more than 800 such items).
- `test_function_prompt_tokens_are_a_prefix_of_the_training_text_tokens`: a tokenizer trained in the test with the
  build's own `train_tokenizer` settings on generator code text; asserts the merge exists (`id_of`) and occurs in
  `'"""\n    return a'`, then for all 50 function items `encode_doc(prompt)` is a token prefix of the training text
  (`check_args["prompt"] + reference`), the first reply token is the merge, and the rest of the text is `"\n" + reference`.
- `test_answer_key_replies_to_a_function_prompt_with_a_newline_and_the_body`.
- In tests/test_code_instr_grammar.py `test_function_reference_passes_and_wrong_fails` now proves all 50 references
  pass both bare and with a leading newline, and `"\n"`, `"\n    return None\n"` fail.

Existing tests changed because they encoded the old (defective) prompt, not weakened: `test_coding_bench_shape`
(prompt regex has no trailing newline; `check_args["prompt"] == prompt + "\n"`), the training-vs-bench header
overlap checks (now use `check_args["prompt"]`, otherwise they would have become vacuous),
`test_checkers_get_the_full_reply_and_untouched_check_args` and `_reply_budget_overruns` (use `reference_reply`).

Not done (not mine): the finder's "record the deviation from plan line 1176 in ADR 0001". The controller writes ADR text.
Gate caches refresh by themselves (code digest changed).

## F5: answers follow the KB's base rates

**Reproduced.** The new relation-prior model (MC: option most common in the KB for the item's relation; exact: the
relation's most common KB object) through `run_benchmarks`:

RED (before the fix):
```
uv run --no-sync pytest tests/test_bench.py -q -k relation_priors
E  AssertionError: CategoryScore(score=18.76923076923077, raw=0.34, n=200)
E  assert 18.76923076923077 <= 7
1 failed
```
(Finder's number: 18.8. Old frozen draw: animal_legs 11/21 'four', animal_class 10/21 'mammal', vehicle_travel 7/15 'by water'.)

GREEN: same command, `1 passed`. With the fix the model scores knowledge 0.997 (raw 0.255, chance 0.2475) and
consistency 0.0 (raw 0.100, chance 0.25).

**Fix (as ruled).**
- `facts.py`: `_balanced_facts(rng, n, pool)` takes a relation's distinct objects in a seeded order, round-robin, a random
  unused fact of each, skipping used-up objects; `_sample_bench_facts` = `fair_quota` over relations + `_balanced_facts`.
  `knowledge_bench_items` (exact and MC) and `consistency_groups` use it. Distractors untouched. `_sample_facts` is
  kept for `plan_false_facts` (its draw feeds training text, so it must not change).
- Exact-item chance is a measured floor: `scoring.relation_prior_floor(items)` and `suite._run_category` for category
  `knowledge` (ExactItems only; coding output items keep chance 0; no free answers scored, e.g. `max_items_per_category=150`,
  means floor 0). Module docstring of suite.py updated.
- Reading of the formula: "most common answer" counts every accepted form (`ExactItem.answers`, compared as
  `exact_match` does), so a constant reply that two items accept counts twice. On the frozen suite floor = 0.24 (12/50);
  counting only `answers[0]` would give 0.22 (the difference is baby_animal's `pup`/`puppy`). Easy to switch if you want the literal reading.

New tests: relation-prior model (above); `relation_prior_floor` arithmetic; floor wired as chance for knowledge only
and following the items run; `_balanced_facts` (turns, skipping, determinism, error); frozen-draw spread of the four
named relations (each at most 30% of its items); across 20 seeds every relation except shape_sides has its most common
answer at most 35% of its items (the old sampler broke this 53 times). No existing test pinned the old frozen draw.

**Residual (honest limit).** `shape_sides` has 14 eligible facts, 7 of them 'four', so its answers cannot be evened out
(MC 'four' is 5 to 7 of 10 across seeds; frozen draw 7). Over 60 other seeds the KB-frequency picker averages 0.31 on
MC (not 0.25), mostly from this relation. Worth about +3 normalized points on knowledge for such a model; the frozen
draw stays within the 7-point tolerance. Removing it needs a quota cap by distinct objects (moves items between
relations), which the ruling does not ask for.

## F24: knowledge reply budget

RED: after changing the pinned value in `test_knowledge_bench_composition` to 12:
`AssertionError: assert ('first_line' == 'first_line' and 8 == 12)` (item `knowledge-0150`), 1 failed.
Fix: `facts.KNOWLEDGE_EXACT_TOKENS = 12`, passed to the knowledge `ExactItem` (still one length bucket: the
`generate_calls` test still sees one generate call for knowledge). GREEN: `tests/test_kb_facts.py tests/test_bench.py`
174 passed, 1 skipped. The default `ExactItem` budget stays 8 (`test_item_and_doc_defaults`). The real-tokenizer
budget test is still skipped here (no tok-v1 on this machine); it will use 12.

## F44: vacuous tests

Evidence (scratch probe): `run_benchmarks(ScriptedLM, max_items_per_category=10).tag_breakdown()` is `{}`; with 60 items it has
18 tags. The old sampler assertion `any(k in msg for k in weights)` is true for a message that names the wrong dataset.
`test_report_serializable_and_tags` now runs 60 items, asserts the breakdown is not empty and each tag has
at least `MIN_TAG_ITEMS` items, then the 0..100 range. `test_invalid_mixtures` uses `alpha`/`beta`, `pytest.raises(match=...)`
on the offending quoted name, `"sum"` for the all-zero case, and checks no other name is in the message.
Mutation check: with the invalid-weight message rewritten to name `'other'`, the old test passes and the new one fails
(`1 failed, 4 passed`). GREEN: `6 passed`.

## F53 and F54: accepted forms

RED (new tests on the old data):
```
assert ['mammal'] == ['mammal', 'mammals']                         (animal_class)
assert ['first', '1st'] == ['first', '1st', '1']                   (planet_order, test_free_answers_accept_the_listed_forms)
assert (accepted[0] == 'first' and '1' in ['first', '1st'])        (test_planet_order_free_answers_accept_the_bare_digit)
```
Fix: `animals.json` animal_class `also_accepted` = the seven plurals; `space.json` planet_order adds the bare digit after
each ordinal form. GREEN: `tests/test_kb_facts.py` 78 passed. Existing assertion updated to the ruled value:
`test_free_answers_accept_the_listed_forms` now pins `["first", "1st", "1"]`. Plurals and digits are not objects, so no
draw, distractor or false fact changed (F53's and F54's commits do not move the digest).

## F25: the pin

`tests/test_bench.py::test_the_items_of_the_suite_version_are_pinned` hashes `{"version", "categories": [[name, [asdict(item)...]]...]}`
(`sort_keys=True`, compact separators) and compares with `PINNED_DIGESTS["bench-v1"]`:
`4b0e0bd856c96c5d45b80d27141550bc6284803549ba5b3ffde046887f16da68` (285,777 bytes of JSON; the same under
`PYTHONHASHSEED` 0, 987654 and random). Failure message: an intended change to the items needs a new `SUITE_VERSION`
(`airace_ml/evals/suite.py`) and a new digest in `PINNED_DIGESTS`. A second test shows the digest reacts to a prompt, an
option, an answer index, a good text, check_args, a tag, a reply budget, item order, category order and the version.
Mutation check (RED for "nothing pins it"): one function-family docstring reworded in `code.py` ("return the answer" ->
"return the total"): 301 other bench/KB/code/reasoning tests still passed, only the pin failed. Reverted.
**Later groups must not change bench items**; if one does, this test fails by design.

## Self-review (adversarial reread of the whole diff)
- Looked for loops that cannot end (`_balanced_facts`: n <= pool, every pass takes a fact or all are used up), changes that move the false-fact or
  training draws (none: `_sample_facts` untouched, only bench callers moved), tests weakened (none; the five existing
  assertions that encoded the old behaviour are listed above), vacuity (the prefix test asserts the merge; the spread tests
  were run against the old sampler; the F44 tests against a mutation).
- A first draft of the prefix test passed on the old code because it built `prompt + "\n" + reference` (double newline
  tokenizes as '\n' + '\n   '); rewritten to compare against the training text (`check_args["prompt"] + reference`).
- `git stash`/restore and temporary mutations were used in the working tree and all were reverted (working tree clean at the end).

## Concerns and other observations
- The knowledge exact floor is high-ish by construction: 0.24 on 50 items with ~12 relations of 3 to 5 items each (an oracle constant
  per relation). A real relation-prior model gets about 0.11 on those items, so such a model normalizes below 0 and clips to 0. That is the
  ruled definition; players who truly know facts are unaffected at the top (100 stays 100), but mid scores on the exact part compress
  by a factor 1/(1-0.24).
- The real tok-v1 is not available here: F1's token claims use a tokenizer trained in the test, the F24 budget is checked only by
  the (skipped) real-tokenizer test. Run `pytest tests/test_bench.py -k real_tokenizer` on the owner's PC after `airace-content build`.
- ADR 0001 should record: the function-prompt boundary (deviation from plan line 1176), the knowledge answer-evenness rule, the
  relation-prior floor, the 12-token budget, and the digest pin with its "bump the version" rule.
- `shape_sides` residual skew (see F5).
- Pre-existing: `ruff format --check src tests` reports 9 files that would be reformatted (9 at the base commit too). Of the files I
  changed, only `tests/test_sampler.py` is among them (it was unformatted at the base commit, in a compact one-line style that my
  edit follows); the other ten are format-clean. `ruff check .` is clean.
