# Task 17 report: novelty index, reference judge, creativity

**Status:** DONE_WITH_CONCERNS (all requirements met; the concerns below are product/robustness notes for the controller, not defects)
**Commit:** `ad94045 feat(ml): novelty index, reference judge, creativity scoring`

## What I implemented

### `ml/src/airace_ml/evals/novelty.py`
- `ngram_hashes(ids, n=8)`: polynomial hash, base 1,000,003, mod 2^64. It is vectorized over every window, with a loop over the `n` positions only, never over tokens. It uses numpy uint64 *array* arithmetic, which wraps silently; only scalar-scalar ops would warn. Fewer than `n` ids give an empty uint64 array. A bad `n` or a 2-D input raises ValueError.
- `NoveltyIndex(hashes, n=8, sample_mod=8)`:
  - The constructor sorts and uniques the hashes if they aren't already (an O(n) check otherwise).
  - `build(token_arrays, n, sample_mod)`: no window spans two arrays.
  - `novelty(ids)`: `np.searchsorted` on the sorted index. Every occurrence counts. Returns 0.0 with no sampled windows, and 1.0 against an empty index.
  - `save(path)`/`load(path)`: the `.npy` is written via a temp file and `os.replace`. A sidecar `.json` beside it (`index.json`) records `hash_base`, `n`, `sample_mod` and `count`. Reason: a bare `.npy` can't carry `sample_mod`, and the brief's own test saves a `sample_mod=1` index and expects identical novelty after load. `load` refuses a different hash base.
- `build_novelty_index(data_root)`, per ruling 5:
  - For each of the 8 corpora, one vectorized pass over the concatenated token array.
  - A window-start validity mask comes from a difference array over the document offsets. Only non-held-out documents of at least `n` tokens open a run, so no window crosses a document boundary or touches a held-out document.
  - Hashing runs in 4M-window chunks so memory stays bounded. The result is unique-sorted and saved to `novelty_path(root)`.
  - There is no per-document Python loop.

### `ml/src/airace_ml/evals/judge.py`
- `JudgeCalibration(creative_p10, creative_p90, conv_reply_p90)`.
- `Judge(lm, tok, calibration=None)`:
  - `Judge.load(dir, device=None)`: reads the checkpoint, `tokenizer.json` and `calibration.json` from `dir` only (ruling 1). It checks that the vocabulary matches. `device=None` means `pick_device()`, as in `train_run`.
  - `nll_per_token(lists)` (ruling 2):
    - special ids (< 16) are stripped, then each list is cut to `ctx_len - 1` tokens;
    - one batched `score_continuations` call with `[bos]` contexts;
    - the result is `-sum_logprob / n_tokens`;
    - an empty list, or a NaN log-prob, gives `inf`.
- `judge_train_config(seed=1234)`: exactly the brief's values (shape (8, 384, 256), about 15.7M params at vocab 4096).
- `calibrate(judge, root)` (ruling 3):
  - Creative: the first 96 tokens after `<|bos|>` of up to 500 held-out creative docs. Above 500, a fixed-seed sample is used.
  - Conversations: AI-reply segments, the tokens between each `<|ai|>` and its closing `<|end|>`, from held-out conversation docs. Docs are visited in a fixed-seed order until 500 segments are collected. Empty or unclosed segments are skipped.
  - Segments are scored exactly as `is_well_formed` scores a reply. The test checks that `encode(decode(seg).strip()) == seg`.
  - Percentiles use finite values only; with none, it raises ValueError.
- `build_judge(root, *, device, on_event, token_budget)`:
  1. Deletes any previous `calibration.json`.
  2. Calls `train_run(cfg, out_dir=judge_dir(root), data_root=root, device=device, on_event=on_event)`.
  3. Copies the tokenizer in.
  4. Loads the checkpoint and calibrates it.
  5. Writes `calibration.json` last, atomically.

  A directory that has `calibration.json` is therefore a complete judge.
- `is_well_formed(reply, judge)`:
  - Requires ≥ 4 words, using `skills.checkers.words` (the project's checker word: letters and digits) compared case-folded.
  - Rejects any word 3-gram that occurs ≥ 3 times.
  - Requires `nll_per_token([judge.tok.encode(reply.strip())]) <= conv_reply_p90`. NaN and inf fail.
  - The judge is only called when the word checks pass.

### `ml/src/airace_ml/evals/creativity.py`
- `STORY_PROMPTS`: 24 unique, simple-English "Write a short story about …." prompts, 4 each for animals, places, feelings, objects, people and nature. Each item is tagged `("fmt:story", "topic:<t>")`.
- `score_creativity(lm, tok, judge, novelty, *, seed=0, max_new_tokens=96)`:
  - **Generation (ruling 6):** one batched `generate` call, temperature 0.9, top_p 0.95, `seed=seed`. Each story is `tok.decode(tokens).strip()`. The judge scores `judge.tok.encode(story)` in one batched call; novelty uses `tok.encode(story)`.
  - **Coherence:** `c = clip((p90 - nll)/(p90 - p10), 0, 1)`. A NaN gives 0, and a calibration with no spread becomes a plain `nll <= p90` test.
  - **Story score:** `c * (0.4 + 0.6 * nov)`. A story with no words (runs of letters) scores 0.
  - **Diversity:** `d` is distinct-2 over lowercase letter-run words, pooled across stories without crossing stories (ruling 7).
  - **Category score:** `100 * mean * (0.5 + 0.5 d)`, with `raw = mean` and `n = 24`. Items are `story-00` … `story-23`, and each output is its stripped story.

### Other changes
- **`suite.py`:** the hook already matched the contract, so the only changes are the stale "# arrives with Task 17" comment and a docstring paragraph. The lazy import stays: the existing `test_creativity_hook` still pins the wiring with a fake module, and the real module doesn't make it redundant, so I kept it unchanged.
- **`data/corpus.py`:** a new read-only `Corpus.tokens` property, the concatenated memory-mapped token array. Ruling 5's per-corpus vectorized pass needs it, and the alternative was reaching into the private `_tokens`. This file is outside the brief's file list; the change adds six lines and is purely additive.

## Measurements (requested)

**Slow test** (`uv run --no-sync pytest -m slow tests/test_creativity.py -v`):
- **PASSED in 84.1 s** (call time; 87 s wall) on this 4-core cloud CPU. That covers 4 steps × 32768 tokens at the judge shape, plus final eval, calibration and the tokenizer copy.
- Smoke judge: status `completed`. Calibration: `creative_p10 = 5.458`, `creative_p90 = 7.405`, `conv_reply_p90 = 5.741`.

**Creativity scores, the brief's cases** (stub judges: fluent NLL 2.5, so c = 0.875; gibberish NLL 9.0, so c = 0; index of `CORPUS_TEXT * 3`, `sample_mod = 1`):

| Case | Score | Raw | Notes |
|---|---|---|---|
| novel | **45.57** | 0.875 | novelty 1.0; d = 10/240 = 0.042 because all 24 stories share their words |
| copy | **18.23** | 0.350 | novelty 0.0, so it keeps 40%; d = 15/360 = 0.042 |
| gibberish | **0.00** | 0 | coherence 0 |
| empty | **0.00** | 0 | — |

**Full-scale novelty index estimate.** I built a real data root of 8 synthetic corpora at the full-scale targets in `airace_content.assemble.SCALES`, **137.0M tokens** in total:
- token ids drawn as Zipf(1.2) over the 4080 text ids;
- documents of 20–480 tokens;
- 10% held out.

Then I timed `build_novelty_index`:
- **Build: 20.8 s** on 4 CPU cores. Peak memory rose by about 0.77 GB during the build (1.28 GB process peak, including the synthetic write).
- **Index: 14,974,369 hashes = 119.8 MB** on disk. That is about 137M × 0.9 × 1/8 sampled windows, nearly all unique.
- **Load: 0.05 s.** Novelty for 24 stories of 96 tokens: **0.7 ms** in total.
- Synthetic ids make nearly every 8-gram unique, so 120 MB is an upper bound. Real text, with duplicates and templated generators, will give a smaller index. The script is in my scratchpad (`novelty_scale.py`), not committed.

## Tests and results
- **RED:**
  - Command: `cd ml && uv run --no-sync pytest tests/test_creativity.py -v`
  - Output: `ModuleNotFoundError: No module named 'airace_ml.evals.creativity'` (1 error during collection).
  - Expected, because none of the three modules existed yet.
- **GREEN (brief tests):** the same command gave `4 passed, 1 deselected` (the slow test deselected). The slow test then passed separately, as above.
- **Added tests** (17 more in `tests/test_creativity.py`; 21 fast tests + 1 slow in total):
  - **Novelty:**
    - `ngram_hashes` against a slow Python-int reference for n ∈ {1, 2, 3, 8}, with ids up to 65535, under `warnings.simplefilter("error")`; fewer than n tokens gives an empty uint64 array; a bad n or 2-D input raises.
    - Index contents equal the exact reference set (sampled, sorted, unique); windows never span arrays; per-occurrence novelty is exact.
    - Empty index, unsorted input, and bad parameters.
    - Save/load keeps `n` and `sample_mod`, writes the sidecar, and rejects a different hash base.
  - **`build_novelty_index` on a custom 8-corpus root** (random docs, some shorter than 8 tokens, one empty, every fourth held out):
    - the index equals exactly the in-document, non-held-out sampled windows;
    - boundary-spanning and held-out-only windows exist and are absent;
    - the result is identical with `_CHUNK = 3`.
  - **Judge:**
    - The `nll_per_token` contract: one call, `[bos]` contexts, specials stripped, cut to `ctx_len - 1`; empty, special-only and NaN give inf.
    - Real `TorchLM`: matches direct scoring; an over-long list equals its first 63 tokens.
    - `Judge.load` from a moved directory alone; a vocabulary mismatch raises ValueError; a missing calibration raises FileNotFoundError.
    - `calibrate` records exactly the held-out creative openings (`doc[1:97]`) and the reference-parsed AI replies, and its percentiles match.
    - Seeded sub-sampling (`CALIBRATION_DOCS = 2`) is deterministic and capped; a judge that only returns NaN raises.
    - `build_judge` with a fake `train_run`:
      - the config is the judge config with the budget overridden; the keyword arguments pass through;
      - the stale calibration is gone before training starts;
      - the directory holds exactly `calibration.json`, `meta.json`, `model.safetensors` and `tokenizer.json`;
      - after a move, `Judge.load` works and the stored calibration equals a fresh `calibrate`.
    - The judge config recipe.
  - **`is_well_formed` edge cases:**
    - exactly 4 words passes, and the reply is judged stripped;
    - a number counts as a word;
    - a 3-gram said twice passes; three times, in any case, fails;
    - empty and punctuation-only replies fail;
    - the judge is not called for replies that fail the word checks;
    - nll equal to p90 passes; NaN, inf, and anything above p90 fail.
  - **Creativity:**
    - Exact per-story scores and the mix formula against an independent distinct-2.
    - Exactly one batched `generate` call with the right prompts and settings; `max_new_tokens` and the seed pass through.
    - Empty and digits-only stories score 0; outputs are stripped; ids and tags are correct (6 topics).
    - **Degenerate outputs (Review Focus 4):**
      - special-token-only generations score exactly 0 with empty outputs;
      - punctuation-only scores 0;
      - repeated `"the " * 60` and copy outputs stay finite and in range against judges returning NaN, inf, −inf, 0 and 1e9, and against a zero-spread calibration;
      - a NaN loss scores 0.
    - `STORY_PROMPTS`: 24 unique, the expected form, at most 14 words, and none of them (nor their subjects) appears in any `airace_content` or `airace_ml.skills` source or KB file.
  - **`run_benchmarks`** with a real `Judge` (tiny `TorchLM`) and a real `NoveltyIndex`:
    - creativity is scored (n = 24, finite, between 0 and 100) and `missing == []`;
    - it equals a direct `score_creativity` call;
    - it is reproducible across runs.
- **Mutation sanity check:** I applied 10 deliberate bugs one at a time (boundary mask removed, off-by-one in the mask, held-out not excluded, specials not stripped, opening includes bos, `>=` in the 3-gram rule, reply not stripped, d factor dropped, NaN guard removed, story not stripped). All 10 were caught.
- **Full suite:**
  - Command: `cd ml && uv run --no-sync pytest -q -W error::RuntimeWarning`
  - Result: **972 passed, 1 skipped, 5 deselected in 75.9 s**, no warnings. The new file adds about 2 s.
- **Lint:** `uv run --no-sync ruff check .` reports All checks passed; ruff format is clean.
- **Brief test formatting changes** (formatting only; no assertion changed):
  - `ruff format` split the one-line statements;
  - `ruff check --fix` split `import numpy as np, pytest` and sorted the imports;
  - I moved the `# Review Focus 4` comment onto its own line, because the formatter otherwise split `[0].score` across three lines;
  - the import block grew for the added tests.

## Files changed
- `ml/src/airace_ml/evals/novelty.py` (new)
- `ml/src/airace_ml/evals/judge.py` (new)
- `ml/src/airace_ml/evals/creativity.py` (new)
- `ml/src/airace_ml/evals/suite.py` (comment and docstring)
- `ml/src/airace_ml/data/corpus.py` (`Corpus.tokens` property)
- `ml/tests/test_creativity.py` (new)

## Self-review
- **Complete** against the brief and rulings 1–10:
  - one batched generate;
  - special ids stripped and lists cut to `ctx_len - 1`;
  - a self-contained judge directory;
  - calibration matched to `is_well_formed`;
  - vectorized hashing with no warnings;
  - per-corpus, boundary-masked index builds;
  - distinct-2 over lowercase letter-run words.
- **Degenerate outputs are finite everywhere** (Review Focus 4).
- **YAGNI:** no resume logic and no extra API beyond `Corpus.tokens`. The private helpers are small, one per responsibility.
- **R1:** the story text is only stripped of surrounding whitespace for measuring and display (ruling 6); the model's words are never changed.
- **R2:** the judge and the index are global instruments that hold no player data.
- **Word definitions differ on purpose.** `is_well_formed` uses `skills.checkers.words`, so "I have 3 cats" counts 4 words. Creativity uses runs of letters (ruling 7). I did not reuse `airace_content.textproc.words`, so that the runtime package `airace_ml` doesn't import the build-time `airace_content`.

## Concerns (for the controller)
1. **Creativity can be gamed by repetition (R5 measurement quality).** A story that repeats one novel phrase gets novelty ≈ 1, since every occurrence counts. A real judge with in-context copying may also give repetition a low NLL, so high coherence. The only penalty is distinct-2, pooled across stories, which costs at most half the score. A possible ruling: also zero a story that fails `is_well_formed`'s 3-gram rule, or use per-story distinct-n. I did not change the brief's formula.
2. **`build_judge` always starts fresh** (`train_run(resume=False)`). An interrupted ~1 h build on the owner's PC restarts from zero, even though the trainer can resume. A small follow-up (auto-resume when `judge_dir` holds a resume state for the same config) could go into Task 19's CLI or a ruling here. Adding it would have meant going beyond the brief's signature, so I left it out.
3. **The hash's low bits are linear in the token ids** (note; the brief's hash is followed verbatim). Because 1,000,003 ≡ 3 (mod 8), `hash % 8` depends only on each token id mod 8. Sampling is still content-defined, consistent between the index and queries, and about 1/8 of windows. Every constant window (for example "the the … the") is always sampled. This does no harm to correctness.
4. **The judge directory also keeps the trainer's `telemetry.jsonl` and `result.json`.** `result.json` records the training status. `build_judge` does not raise on `unstable_stopped`: the calibration is relative to whatever judge was trained, and the status shows in `result.json` and the `Done` event.
5. **The novelty index for the M1 corpus is about 120 MB** and must be held in memory by any process that scores creativity. That is fine on the reference PC; flagged only for M3 worker sizing.

---

## Pre-review changes (rulings 1, 2)

**Commit:** `9098708 fix(ml): mark down repetitive stories in creativity; resume an interrupted judge build`

### What changed

**Ruling 1: repetition factor (`evals/creativity.py`).**
- New public `repetitiveness(text) = 1 − unique word 3-grams / all word 3-grams`. It is 0 when the text has no 3-gram, and it uses the same `_words` extraction as distinct-2.
- Each story is now `c * (0.4 + 0.6 * nov) * (1 − repetitiveness)`. Distinct-2 across stories is unchanged.
- The module docstring documents the factor and why it is there: a loop reads as easy to the judge and counts as novel, so without the factor it scores about 50. With it, a phrase said k times keeps about 1/k.

**Ruling 2: resuming a judge build (`train/trainer.py`, `evals/judge.py`).**
- `train_run` raises a plain `ValueError` for a mismatched state, which is the same type it uses for config errors. So instead of catching it, I decide before training:
  - New public `trainer.can_resume(out_dir, cfg) -> bool`: True only when `out_dir` holds a complete resume state saved by a run of exactly `cfg`.
  - It shares the state selection with `_read_resume`, which I refactored into `_newest_resume_state` plus a validator. `_read_resume` behaves exactly as before.
- `build_judge` passes `resume=can_resume(out, cfg)` to `train_run`. Nothing is caught. A missing, incomplete or mismatched state means `resume=False`, and `train_run`'s fresh path already deletes stale resume states. So a build never errors on a stale state.
- `trainer.py` is a Task 8 file; the change is additive plus that one refactor.

### Tests
- **RED first:**
  - `test_can_resume_…` failed with an `ImportError` (no `can_resume`).
  - `test_build_judge_resumes_an_interrupted_build` failed with `assert [False] == [True]`.
  - The exact-formula test failed: story 7 differed by 0.25, the missing factor.
  - `test_endless_repetition_scores_about_zero` failed: the loop scored 43.77 (`assert 43.77 < 1`).
- **Added or updated tests:**
  - **`test_trainer.py::test_can_resume_only_a_complete_state_of_the_same_config`:**
    - no directory gives False;
    - an interrupted run gives True;
    - another seed or another budget gives False;
    - a complete state renamed to `resume.tmp.*` still gives True;
    - an incomplete state (no `state.json`, or unreadable JSON) gives False.
  - **`test_creativity.py::test_creativity_scores_every_story_and_the_mix`:** now includes
    - a story with repeated 3-grams, "the red cat sat, the red cat sat" (rep = 1/3);
    - a story with no 3-gram, "hello friend" (factor 1).

    The exact expected scores include `(1 − ref_repetitiveness)`, computed by an independent reference.
  - **`test_endless_repetition_scores_about_zero`:**
    - `"the the the " * 30` scores **0.50**, and every story is under 0.01;
    - `"purple robots dance on frozen moons " * 16` (about the 96-token budget) scores **2.80**, against **45.57** for the brief's novel sentence. Each story is about 0.875 × 6/94.
  - **`test_build_judge_resumes_an_interrupted_build`** (fast; `judge_train_config` patched to a (2, 64, 64) shape with 1024-token steps, every other recipe value real):
    1. A real `train_run` is interrupted with `TrainHooks(stop_after_steps=3)` in `judge_dir`.
    2. `build_judge` then passes `resume=True` (checked with a spy around the real `train_run`).
    3. Only step 6 emits Progress, so steps 1–3 were not re-run; `Done` reports completed.
    4. No resume directories are left; `result.json` shows 6 steps, completed.
    5. `Judge.load` works, and its calibration equals a fresh `calibrate`.
  - **`test_build_judge_starts_fresh_over_another_builds_state`** (fast): a 6-step interrupted state followed by a 4-step build gives `resume=False`, Progress at [1, 4], completion, and a loadable judge.
  - **`@slow test_build_judge_resume_smoke`**: the real judge recipe (8×384×256), interrupted after 1 of 2 steps of 32768 tokens. `build_judge` resumes (spy shows True; Progress only at step 2) and leaves a loadable judge with a finite calibration.
  - **`test_build_judge_writes_a_self_contained_judge`**: the expected `train_run` keyword arguments now include `resume: False`.
- **Mutation check of the new logic:** I applied 5 deliberate bugs one at a time (`can_resume` ignoring the config, checked from both test files; `build_judge` never resuming; the repetition factor dropped; bigrams in place of 3-grams). All 5 were caught.

### Commands and output
- `cd ml && uv run --no-sync pytest tests/test_creativity.py tests/test_bench.py tests/test_trainer.py -q -W error::RuntimeWarning` gave **137 passed, 1 skipped, 3 deselected in 28.2 s**.
- `uv run --no-sync pytest -m slow tests/test_creativity.py -v --durations=4` gave **2 passed in 123.5 s**:
  - `test_build_judge_smoke` **83.9 s**;
  - `test_build_judge_resume_smoke` **39.4 s**.
- Full suite, `uv run --no-sync pytest -q -W error::RuntimeWarning`: **976 passed, 1 skipped, 6 deselected in 75.2 s**. After that I changed only docstrings: "three ways" became "four ways", and "set aside" became "deleted", which is accurate because a fresh run removes stale states. I then re-ran `test_creativity` and `test_bench`: 115 passed.
- `uv run --no-sync ruff check .` reported All checks passed; ruff format is clean on the touched files.
  - Process note: an accidental `ruff format tests/` reformatted 7 unrelated test files that were never format-clean. I reverted them with `git checkout` before committing, so only my files are in the commit.

### Brief's cases after ruling 1

| Case | Score | Change |
|---|---|---|
| novel | 45.57 | unchanged; no repeated 3-gram |
| copy | 18.23 | unchanged |
| gibberish | 0.00 | — |
| empty | 0.00 | — |
| `"the the the " * 30` | **0.50** | was about 43.8 |
| novel phrase × 16 | **2.80** | — |

The brief's ordering test still passes.

### Remaining notes
- With the 1 − rep form, repetition scores *about* 0, not exactly 0: a phrase said k times keeps about 1/k of its score. That fits the ruling's "0 (or ≈0)".
- Concern 3 (hash residues) is deferred per the ruling; no change.
