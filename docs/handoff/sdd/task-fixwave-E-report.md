# Fix wave group E (evals): report

Branch `claude/upbeat-franklin-k1192h`. Base `e1a94c6e33556e74c225d923718790ff2f071328` (the commit just before the first `fix(E)` commit). Head `a0ded66b587e0d66c3b34d63761053da14fdbf99`. Nothing pushed.

An earlier attempt of this group was interrupted after committing F4 (`a64f5d4`) and leaving an untracked `ml/tests/test_data_identity.py` (the F46 test). I audited both (the F4 diff in full, the identity tests and mutation checks against them) and kept them; the rest was done in the order of the brief. Commit `0de3987` (a controller snapshot patch under `docs/handoff/fix-wave/`) is not mine and was left alone.

| Finding | Status | Commits |
|---|---|---|
| F4 judge / novelty index record their data | FIXED (earlier attempt, audited, behaviour re-demonstrated below) | a64f5d4 |
| F6 creativity: variety is a real multiplier | FIXED | 801d333 |
| F33 non-finite logits crash the sampler | FIXED | 08c4f79 |
| F34 duplicated definitions | FIXED (docstrings + one constant, no behaviour change) | 21094a7 |
| F42 slow test pollutes `tiny_data_root` | FIXED | ebd7de2 |
| F43 GPU test compared greedy tokens fp32 vs bf16 | FIXED (GPU test not runnable here; its logic runs in the default suite against a CPU bf16 stand-in) | b9feee3 |
| F46 identity never tested to change | FIXED | 80c44e9, a0ded66 (format only) |

All commands run from `/home/user/AI-Race/ml` with `uv run --no-sync`.

## F4: the judge and the novelty index record their data (commit a64f5d4)

Audit against the ruling, item by item: `gate._data_identity` moved to `airace_ml/data/identity.py::data_identity` (same digest, the gate and the CLI call it); `build_judge` writes `data_identity` and `full_recipe` (budget equals `judge_train_config().token_budget`) into `train_config.json`; `NoveltyIndex` carries `data_identity` and `build_novelty_index` writes it into the `.json`; `judge._trained` compares the whole record; `build_judge` loads a reused checkpoint on the CPU and trains again on a damaged one; `gate._load_judge/_load_novelty` and `cli._creativity_tools` compare the identity with the data root and raise the existing "damaged or out of date (...). Rebuild it with: airace-ml build-judge / build-novelty-index" error, also for a `--budget-tokens` judge; the identity is in the gate's benchmark cache key (`"data"`); CLAUDE.md has the one line. The CLI `bench` has no cache, so there is no key to change there.

**RED** (probe test in the scratchpad, run against the source of `e1a94c6` with `PYTHONPATH=.../oldsrc/ml/src`: build a stubbed-training judge, append a byte to the tokenizer, run `build-judge` again, then truncate the weights and run it again):
```
TRAINING RUNS AFTER THE REBUILD: 1 (1 = old weights reused)
JUDGE TOKENIZER == NEW TOKENIZER: True
TRAIN_CONFIG KEYS: [... no data_identity]
DAMAGED WEIGHTS: build_judge crashed: SafetensorError Error while deserializing header: invalid header length
```
**GREEN** (same probe, current source):
```
TRAINING RUNS AFTER THE REBUILD: 2
TRAIN_CONFIG KEYS: [... data_identity] True
DAMAGED WEIGHTS: build_judge rebuilt it, training runs 3
```
Tests (in the commit): `tests/test_creativity.py` (record, retrain after corpus/tokenizer rebuild, earlier record, 3 kinds of damaged weights, busy GPU, provenance checks, index record), `tests/test_gate.py` (loaders, the real `_load_inputs` on a copy of the tiny root, bench cache key), `tests/test_cli.py` (bench with stale, quick, early, rebuilt-corpus judge/index). Existing assertion changed to the ruled behaviour (as the commit says): the record assertion of `test_build_judge_writes_a_self_contained_judge`; the `test_cli`/`conftest` judge fixtures build on a recipe whose own budget is the small one so they are full-recipe judges.

## F6: variety is a real multiplier (commit 801d333)

**RED** (`pytest tests/test_creativity.py -k "one_answer or novel_story_told or half_as_varied"`, field `creative_distinct2` added to the dataclass first so the failures are about the score; old formula):
```
test_one_answer_for_every_prompt_scores_about_zero[novel]   assert 52.083333333333336 <= 5
test_one_answer_for_every_prompt_scores_about_zero[copied]  assert 20.83333333333334 <= 5
test_one_answer_for_every_prompt_scores_about_zero[opening] assert 20.83333333333334 <= 5
test_one_answer_for_every_prompt_scores_about_zero[stock]   assert 52.083333333333336 <= 5
test_stories_half_as_varied_as_real_ones_keep_half          assert 75.0 == 50.0
```
(before that, the very first run failed at collection: `JudgeCalibration.__init__() got an unexpected keyword argument 'creative_distinct2'`.)

**Fix.** `score = 100 * mean(story) * min(1, d / d_ref)`; `d` = distinct-2 of the run's stories; `d_ref = calibration.creative_distinct2`, measured by `judge.calibrate` over the held-out creative openings it already uses and written to `calibration.json`. A calibration without the key is refused like any other out-of-date file: `Judge.load` raises `TypeError` ("missing ... creative_distinct2"), the gate and `bench` turn that into "damaged or out of date ... Rebuild it with: airace-ml build-judge", and `build-judge` re-calibrates a judge built after F4 without training (its weights and record are good; a judge built before this wave has no data record, so `build-judge` trains it again, see the review fix round). Openings with no word pair make `calibrate` raise (`ValueError`, "no word pairs"). A calibration whose `d_ref` is not above 0 (hand-built) is a plain `d > 0` test, like `_coherence`'s no-spread case. Novelty floor 0.4, repetition penalty, prompt-echo rule unchanged; no length term. `ADR 0001` is left to the controller.

**Deliberate refinement of "the same distinct-2".** `d_ref` is the mean distinct-2 over sets of 24 openings (= number of stories; a seeded shuffle of the openings cut into consecutive sets, leftovers dropped; fewer than 24 openings: all of them), not the distinct-2 pooled over all (up to 500) openings. Distinct-2 falls as more text is pooled, so a pooled reference is lower than what 24 real stories reach; a constant model would then score `100/24/d_ref`, i.e. 8.3 at a pooled 0.5 instead of 4.6 at 0.9, and a half-varied model would be let off. Tests pin it (60 identical openings give 1/24, not 1/60). If the controller wants the literal pooled value it is a one-line change in `_reference_distinct_2`.

**GREEN** (`pytest tests/test_creativity.py`, 58 passed). Scores with `creative_distinct2 = 0.9`, fluent judge (coherence 1), 24 prompts:
```
constant novel story            4.63   (was 52.08)
constant corpus story           1.85   (was 20.83)
constant "Once upon a time,"    1.85   (was 20.83)
constant stock sentence         4.63   (was 52.08)
24 different novel stories    100.00   (old formula: 100.00)
12 novel stories, each twice   55.56   (old formula: 75.00)
```
Tests: `test_one_answer_for_every_prompt_scores_about_zero` (4 constants, <= 5), `test_a_novel_story_told_to_every_prompt_keeps_only_its_variety`, `test_stories_as_varied_as_real_ones_lose_nothing_to_variety`, `test_stories_half_as_varied_as_real_ones_keep_half`, the mix test now checks `100*raw*d/d_ref` with `d < d_ref`, calibration tests (few openings, grouping, determinism, no word pairs), `Judge.load` / `build_judge` / gate (`test_structurally_wrong_judge_or_index_files_ask_for_a_rebuild`, extended with "from before variety was measured") / CLI (`_calibrated_early`). The constant "novel" exemplars (`test_creativity_ordering`, the echo test, the endless-repetition test) are per-prompt stories now (`per_prompt`, `novel_story` helpers). In `test_gate.py` the invalid-calibration cases were rewritten as `{**good, ...}` with the fourth field, because without it they would all have failed on the missing key and no longer tested their own value check (no assertion weakened).

Files: `ml/src/airace_ml/evals/creativity.py`, `ml/src/airace_ml/evals/judge.py`, tests above.

## F33: non-finite logits (commit 08c4f79)

**RED** (`pytest tests/test_infer.py tests/test_creativity.py tests/test_personality.py -k "non_finite or without_a_distribution"` with the old `lm.py`): 7 tests, 5+2 failed with
`RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0` at `infer/lm.py:125`.

**Fix** (`infer/lm.py::_sample`): rows whose softmax is not all finite take `logits.argmax(-1)` (as temperature 0 does); they get a placeholder distribution so one `multinomial` call serves the batch, and their draw is discarded; healthy rows use the same operations as before (seeded output unchanged).
**GREEN**: `7 passed`; whole files `169 passed`. Tests: `nan_lm` fixture (conftest; all embeddings NaN): `generate` at (0.9, 0.9), (0.9, 1.0), greedy returns the greedy tokens, full length; `_sample` with NaN, one `+inf` and all `-inf` rows leaves the other rows drawing what they would; `chat_reply`/`complete`; `score_creativity` and `measure_fingerprint` finite.

## F34: shared definitions (commit 21094a7)

**RED** (`pytest tests/test_creativity.py -k "g2_counts or defined_once"`): `'skills.checkers.words' in "The words of text: runs of letters, lowercased. Everything that counts words of a model's reply ..."` false; `assert ['creativity.py', 'judge.py'] == ['creativity.py']`.
**Fix:** docstrings of `creativity.words` and `gate.garble_rate` say G2's `is_well_formed` uses `skills.checkers.words`; the 96-token length is defined once. Because F6 made the judge's `calibrate` use creativity's `distinct_2` and `STORY_PROMPTS`, `judge.py` imports `creativity`, so creativity cannot import `judge.CREATIVE_TOKENS` (cycle): `STORY_TOKENS = 96` stays in `creativity.py` and `judge.CREATIVE_TOKENS = STORY_TOKENS`. **GREEN:** the two tests pass; they also pin the intended counting difference (`"I don't know."` is 3 words for G2, 4 for creativity). No behaviour change.

## F42: slow test pollutes `tiny_data_root` (commit ebd7de2)

**RED** (`pytest "tests/test_creativity.py::test_build_judge_smoke" tests/test_gate.py::test_gate_refuses_to_start_without_its_inputs tests/test_cli.py::test_bench_prints_a_table_and_notes_that_creativity_is_missing tests/test_cli.py::test_bench_without_creativity_does_not_call_it_missing tests/test_cli.py::test_gate_without_its_inputs_names_what_to_build -m "slow or not slow"`): `3 failed, 2 passed in 64s` (e.g. "the reference judge in .../data_root0/judge/v1 is damaged or out of date (it was trained on a shortened budget ...)").
**Fix:** `test_build_judge_smoke` uses `data_root_copy`. **GREEN:** same command, `5 passed in 60s`. Other slow tests: `test_build_judge_resume_smoke` copies, the two gate tests use `gate_data_root`, the reasoning test uses no root; the GPU tests only read `tiny_data_root` (the trainer writes nothing into the data root).

## F43: GPU bf16 test (commit b9feee3)

Evidence (CPU bf16 autocast as stand-in for CUDA bf16, 12 sharp-model seeds, the old exact-greedy assertion): `DIVERGED 10 of 12`; seed 1 (the fixture's) passes by luck, its step-0 top-2 margin is 0.014 against a bf16 logit error up to 0.19. Over 120 seeds: per-position mean absolute logit error median 0.18, max 0.594; argmax flips by bf16 up to a fp32 margin of 0.964.
**Fix** (`tests/test_infer.py`): `test_cuda_bf16_path_tracks_the_cpu_path` teacher-forces the CPU greedy reply through both models (`_assert_logits_track`) and requires the mean absolute logit error of every position < 1.0 and the same argmax wherever the fp32 top-2 gap exceeds 2.0; seeded reproducibility, probability range and `score_continuations` (rel 0.05) checks kept. **Deviation from the ruling's numbers:** "about 0.5" (and margin 1.0) fail on measured data (max 0.594, flip at 0.964), so I used twice the worst seen (1.0 and 2.0); a different model or mangled weights still fail it. **The GPU test itself was NOT run (no CUDA).** The same helper runs in the default suite against `_Bf16OnTheCpu` (8 seeds) with two negative controls (`test_bf16_logits_track_...`, `test_the_logit_check_notices_a_model_that_is_not_the_same`), and passed for 120 seeds in a probe.

## F46: the data identity is tested to change (commit 80c44e9)

`tests/test_data_identity.py` (36 tests, on copies of the tiny root): stable across calls and copies; changes when any dataset's `tags.npz`, `offsets.npy`, `info.json` or `tokens.bin` size changes (32 cases) or the tokenizer does; a change in one dataset differs from the same change in another; missing data is `FileNotFoundError`. `tests/test_gate.py`: `fake_gate` exposes `inputs`; `test_reuse_follows_the_data_the_runs_were_trained_on` (identity changed between two runs into the same `--out`: 25 runs trained again, 15 benchmarks and 15 fingerprints again, nothing reused; unchanged: everything reused); `test_growth_sequences_take_one_held_out_document_of_each_dataset_in_turn`. The real `_load_inputs` and the bench cache key tests are in the F4 commit.
**RED by mutation** (the code was right, the tests were missing): an identity that ignores `tags.npz` and the token count fails 16 tests; `GateRuns` keyed on a constant fails the fake-gate test (`assert (0, 15, 0) == (25, 15, 15)`); a bench cache key that ignores the data fails `test_the_benchmark_cache_follows_the_data` (`assert 1 == 2`). **GREEN:** all pass on the real code (`99 passed` for the two files). Sources restored after each mutation (checked with `git diff -- src`).

## Suite

- Default suite (`pytest -q`): **1497 passed, 1 skipped, 9 deselected in 162.5 s**, no warnings. (My new tests are each under ~1.5 s.)
- `ruff check .`: **clean** (All checks passed).
- Slow suite (`pytest -m slow -q`, includes the quick feasibility gate, the judge smoke test on its copy, the judge resume smoke test): **5 passed, 1502 deselected in 327 s**.
- The last commit (a0ded66) only re-wrapped one def line of `tests/test_data_identity.py` after the full-suite run; that file's 36 tests and `ruff check .` were re-run after it.

## Self-review, concerns, other observations

- **Not in the ruling, noticed:** an interrupted judge build leaves a resume state keyed on the recipe only. If the corpora are rebuilt between the interruption and the next `build-judge`, `can_resume` is true, training continues from old-data weights on the new data, and `train_config.json` then records the new data's identity. That is F4's failure in the resume case. A fix needs the data identity in the trainer's resume state or a marker written before training starts (group D territory); I did not change it. (Fixed in the review fix round below, with a marker in `judge.py`.)
- `Generation.token_probs` / `top1_probs` of a row with non-finite logits stay NaN (only the token choice was ruled). The fingerprint's confidence already ignores non-finite probabilities (tested elsewhere); nothing else reads them for such rows.
- F6 changes creativity numbers for every model (lower when stories are samey), so gate G3/G6 numbers and any stored bench results change; a judge built before this wave has to be built again with `airace-ml build-judge`, which trains it again (about an hour on the reference GPU): its `train_config.json` records no data identity, so it does not count as trained (corrected in the review fix round; only a judge built after F4 is re-calibrated without retraining). The controller records the rule in ADR 0001 (not edited by me).
- `ruff format --check` reports 9 files that were already unformatted at the base (e.g. `tests/test_infer.py`); none of the files I created is among them (checked against a `git archive` of `e1a94c6`), and I did not reformat files I did not own.

## Review fix round

Base `a0ded66b587e0d66c3b34d63761053da14fdbf99` (the last commit of the first round). Head `815a8e5` (see `git log`). Nothing pushed. The review reported three issues; the commits' trailers name Sonnet 5.5 as the harness's attribution reminder for this session says, where the first round's commits say Opus 5.5 (the rules file's text).

| Finding | Status | Commits |
|---|---|---|
| F4 (blocking): a resumed judge build ignores the data identity | FIXED | 7cc3208, 815a8e5 |
| F43 (minor): bf16 test thresholds 1.0 / 2.0 instead of 0.5 / 1.0 | ACCEPTED as is, no code change (first round: b9feee3) | none |
| F6 (minor): wording that pre-wave judges need only re-calibration | FIXED in the places I own; two places left to the controller | 7cc3208 (docstring) |

### F4: a judge build resumes only on the data it began on (7cc3208, 815a8e5)

**Defect (reproduced by the reviewer, and by the new tests).** `build_judge` passed `resume=can_resume(out, cfg)`; the trainer's resume state knows the recipe, not the data. Interrupt a build, rebuild the corpora or the tokenizer, run `build-judge`: it continued from the old data's weights on the new data and then wrote `train_config.json` with the NEW data identity and `full_recipe`, so the gate and `bench` accepted old weights for rebuilt data.

**RED** (`cd ml && uv run --no-sync pytest tests/test_creativity.py -k "does_not_resume_a_build_that_began or marks_what_it_is_building"`, tests written first, the existing helper unchanged):
```
FAILED test_build_judge_does_not_resume_a_build_that_began_on_other_data[_rebuild_a_corpus]       assert [True] == [False]
FAILED test_build_judge_does_not_resume_a_build_that_began_on_other_data[_rebuild_the_tokenizer]  assert [True] == [False]
FAILED test_build_judge_does_not_resume_a_build_that_began_on_other_data[_rebuild_both]           assert [True] == [False]
FAILED test_build_judge_marks_what_it_is_building_before_it_trains_and_clears_it_after   AttributeError: no attribute 'BUILD_MARKER_NAME'
4 failed
```
A second RED, found while self-reviewing the reviewer's recipe: with the marker written before `train_run`, a build killed after writing the new marker but before the trainer cleared the old state (the trainer clears it only after loading the data) leaves the new marker beside the old data's state, which then looks resumable. `test_a_build_killed_before_the_trainer_cleared_the_old_state_still_does_not_resume_it` failed with `assert [True] == [False]` on the first implementation.

**Fix (all in `ml/src/airace_ml/evals/judge.py`).**
- `BUILD_MARKER_NAME = "building.json"`: `build_judge` writes the record it is about to produce (`_run_record`: recipe, `data_identity`, `full_recipe`) before training starts, and removes it once `train_config.json` is written.
- `_may_resume(out, cfg, record)`: resume only if the marker equals the current record and `can_resume(out, cfg)`. No marker (a build from before markers), an unreadable one, or one of other data or budget: fresh start (`resume=False`).
- `_discard_resume_states(out)`: when not resuming, the old `resume`, `resume.tmp.*`, `resume.old.*` directories are deleted before the new marker is written (best effort, like the trainer's own delete). Uses the trainer's public `RESUME_DIR`; the trainer is not touched.
- The module and `build_judge` docstrings say all of this, and that a judge built before the data record is trained again.

**GREEN.** `pytest tests/test_creativity.py`: `67 passed, 2 deselected`. Tests added: the rebuilt-corpus / rebuilt-tokenizer / both cases (resume flag `False`, progress steps `[1, 6]`, `result` record's identity is the new one, no resume directory left), the no-marker case, the kill-window case, and the marker-content-while-training and gone-afterwards case. `_interrupted_judge_build` now writes the marker through `judge_module._run_record`, as `build_judge` would (parameter `marked=False` leaves it out); the assertions of the existing resume tests (`test_build_judge_resumes_an_interrupted_build`, `..._starts_fresh_over_another_builds_state`, the slow `..._resume_smoke`) are unchanged. Without the helper change `test_build_judge_resumes_an_interrupted_build` fails (`assert [False] == [True]`), which is the marker doing its job.

**Mutation checks** (source restored after each, `diff` against a backup): marker ignored entirely: the three rebuilt-data tests and the kill-window test fail; the no-marker branch returning `can_resume`: the no-marker test fails (`assert [True] == [False]`); no discard before the marker: the kill-window test fails; marker never removed: the marker-content test fails.

**Residual.** The delete of an old resume state is best effort (`ignore_errors`), as the trainer's own: a file a virus scanner holds open on Windows could leave an old state beside a new marker if the build is then killed before the trainer's own cleanup. The window is a few seconds wide and needs both a held file and a kill. Not tested (no way to hold a file here).

### F43: bf16 thresholds (no change)

The reviewer reproduced my justification on CPU bf16 over 60 seeded models (worst mean logit error 0.594, 62 argmax flips, largest flipping margin 0.964): the ruled 0.5 / 1.0 would fail the CPU stand-in, so the looser 1.0 / 2.0 stand. Nothing to change in code. **The CUDA numerics are still unmeasured and the GPU test has never been run.** For the controller: accept the deviation, and have the owner run `cd ml && python -m uv run pytest -m gpu tests/test_infer.py` once on the CUDA PC.

### F6: wording (7cc3208 for the docstring)

A judge built before this wave has no `data_identity` or `full_recipe` in `train_config.json`, so `_trained` is false and `build-judge` trains it again (about an hour on the reference GPU); `test_a_judge_that_records_nothing_about_its_data_is_trained_again` pins it. Corrected: the `build_judge` docstring (new sentence in its first bullet), and in this report the F6 paragraph and the concern in "Self-review" above. **Left to the controller (rules: I do not edit `docs/handoff/`):** I found no sentence saying "no retraining" in `docs/handoff/sdd/ledger.md` or `docs/handoff/README.md` in this checkout (the ledger's line 106 is an unrelated Task 10 entry), so if the controller's own text says that a pre-wave judge only needs re-calibration, it should say instead that `build-judge` retrains judges built before this wave. The message of commit 801d333 says the same and cannot be changed without rewriting history. No real judge has been built yet (that is left for the GPU PC), so nobody is affected in practice.

### Suite (after the last commit)

- Default suite (`cd ml && uv run --no-sync pytest -q`): **1503 passed, 1 skipped, 9 deselected in 170.25 s**, no warnings (six more tests than the first round's 1497; each of the new tests takes 0.5 to 0.8 s; the first round's 162.5 s and this 170 s differ by machine load more than by these tests).
- `ruff check .`: clean. `ruff format --check` on `judge.py` and `test_creativity.py`: already formatted.
- Slow suite (`pytest -m slow -q`): **5 passed, 1508 deselected in 218.65 s**.
- After the last commit (815a8e5, a one-line `list(...)`) I re-ran `tests/test_creativity.py` (67 passed), ruff check and ruff format --check on `judge.py`; the full suites ran just before it.
