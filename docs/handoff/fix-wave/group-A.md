# Fix wave group A: bench items (before bench-v1 freezes)

Read `docs/handoff/fix-wave/fix-wave-rules.md` first; it binds you. Rulings below are the controller's and are binding.

- **Where you work:** ml/src/airace_ml/skills/, ml/src/airace_ml/evals/suite.py, ml/tests/fakes.py and the bench/KB tests.
- **CLAUDE.md edits allowed:** none.
- **Commit subject prefix:** `fix(A): <Fxx> ...`. Report file: `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-A-report.md`.
- **Before reporting:** full default suite, ruff, and the slow suite: `cd ml && uv run --no-sync pytest -m slow -q   (the slow suite, including the quick feasibility gate; a few minutes)`.
- **Order:** work the findings in the order listed.

## F1: Function-completion bench prompts end on a bare newline token that training never follows with indentation, so no trained model can pass the 50 function items (a third of coding in bench-v1)
- Finder severity Critical; verifier severity Critical (fix size small).
- Where: `ml/src/airace_ml/skills/code.py` (878 (_Variant.prompt ends with the docstring's closing quotes plus a newline), 1004-1012 (CheckItem prompt = v.prompt), 927 (training text = prompt + body); consumed at ml/src/airace_ml/evals/suite.py)
- Full finding: `docs/handoff/final-review/verify/s-bench.json` (id F1); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-bench`.
- **Ruling:** Show the model the function header WITHOUT its final newline: in code.py _bench_functions build CheckItem(prompt=v.prompt.rstrip('\n'), check_args keep the FULL v.prompt for the checker). function_body must accept a reply that starts with a newline (it already keeps a leading empty line; prove it with the 50 references). Leave _Variant.prompt and all training docs unchanged (no corpus rebuild). Update tests/fakes.py answer_key_lm so it replies '\n' + reference for these items (or keys on the stripped prompt). Regression tests: (1) no non-chat bench prompt ends in whitespace; (2) under a byte-level BPE that HAS the newline+indent merge ('\u010a\u0120\u0120\u0120'), encode(prompt) is a token prefix of encode(prompt + reference) for every function item. For (2) train a small tokenizer in the test with the build's own train_tokenizer settings on generator code text, and assert the merge exists so the test cannot pass vacuously.

## F5: Knowledge (and consistency) bench answers follow the KB's skewed base rates while chance is set to 1/options (MC) and 0 (exact), so a model that ignores the subject scores about 15-19 on knowledge
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_ml/skills/facts.py` (46-55 (_sample_facts: uniform over facts within a relation), 68-79 (_bench_pools), 144-165 (knowledge_bench_items), 175 (consistency_groups); ml/src/airace_ml/skills/kb.py:161-178 (distractors uniform)
- Full finding: `docs/handoff/final-review/verify/s-bench.json` (id F5); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-bench`.
- **Ruling:** Make the answer distribution uniform within each relation, for knowledge_bench_items AND consistency_groups. Per relation, cycle round-robin through its distinct objects in a seeded order, skipping objects whose bench-eligible facts are used up, and take a random unused fact for each. Distractors stay as they are (uniform over the other objects), so the MC option set is exchangeable and any subject-blind rule gets exactly 1/4. Exact-item chance becomes a MEASURED floor, as the instruction category does: floor = (sum over relations of the count of that relation's most common answer among the scored exact items) / n_exact. Regression test: a relation-prior model (MC: per relation, pick the option with the highest KB object frequency; exact: reply the relation's most common object) normalizes to within about 7 points of 0 on knowledge and on consistency. Update any test that pins the old frozen draw.

## F24: Knowledge exact items keep the default 8-token reply budget, but bench-v1 asks for 'Bandar Seri Begawan', which a 4096-entry BPE likely splits into 9-10 tokens
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/skills/types.py` (116 (ExactItem.max_new_tokens = 8); used by ml/src/airace_ml/skills/facts.py:164 (item knowledge-0168); answer from skills/kb_data/countries.json:88; guard test tests/test_bench.py:737-739)
- Full finding: `docs/handoff/final-review/verify/m-bench-items.json` (id F24); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-bench-items`.
- **Ruling:** Knowledge exact items get max_new_tokens=12 (facts.py, one length bucket in suite._ask). Update tests/test_kb_facts.py, which asserts 8.

## F44: Two brief tests check nothing: an empty tag breakdown, and dataset names that are single letters found in every error message
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/tests/test_bench.py` (76-78 (test_report_serializable_and_tags); ml/tests/test_sampler.py:23-28 (test_invalid_mixtures))
- Full finding: `docs/handoff/final-review/verify/m-bench-items.json` (id F44); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-bench-items`.
- **Ruling:** test_bench.py: use max_items_per_category=60 (or enough for tags to qualify), assert tag_breakdown() is non-empty, then the 0..100 range check. test_sampler.py test_invalid_mixtures: names 'alpha'/'beta', pytest.raises(match=...) on the offending name, and 'sum' for the all-zero case.

## F53: Free-answer animal-class items reject the plural answers people usually give ('mammals', 'birds')
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/skills/kb_data/animals.json` (9 (animal_class also_accepted {}); facts at 64, 90, 100, 125)
- Full finding: `docs/handoff/final-review/verify/m-bench-items.json` (id F53); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-bench-items`.
- **Ruling:** kb_data/animals.json animal_class also_accepted = {mammal:[mammals], bird:[birds], fish:[fishes], reptile:[reptiles], amphibian:[amphibians], insect:[insects], arachnid:[arachnids]}.

## F54: Planet-order free answers accept '5th' but not the bare digit '5'
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/skills/kb_data/space.json` (9-18 (planet_order also_accepted); facts at 47 (Venus), 50 (Jupiter))
- Full finding: `docs/handoff/final-review/verify/m-bench-items.json` (id F54); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-bench-items`.
- **Ruling:** kb_data/space.json planet_order also_accepted: add the bare digit to each ordinal (first ['1st','1'] ... eighth ['8th','8']).

## F25: Nothing pins the bench-v1 item set: it is regenerated from code, KB data and numpy streams at runtime, so it can drift silently under the same version label
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/evals/suite.py` (87 (SUITE_VERSION), 116-129 (_bench_items regenerated from the skills generators); skills/types.py:21-28 (skill_rng); ml/pyproject.toml:5 (numpy>=2.5, no upper bound))
- Full finding: `docs/handoff/final-review/verify/m-bench-items.json` (id F25); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-bench-items`.
- **Ruling:** LAST in group A, after every other item change: a test that hashes canonical JSON of build_suite() (categories and items in order; per item dataclasses.asdict with sort_keys: prompts, options, answers, check_args, tags, budgets) against a recorded sha256. The failure message must say that an intended item change needs a SUITE_VERSION bump and a new digest. Later groups must not change bench items.
