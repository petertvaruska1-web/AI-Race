# Fix wave group D (trainer): report

Branch `claude/upbeat-franklin-k1192h`. Base `afee1120bfd12ec9553220c28e1fdbab26bbb807` (the commit just before the first `fix(D)` commit). Head `e1a94c6e33556e74c225d923718790ff2f071328`. Nothing pushed.

| Finding | Status | Commits |
|---|---|---|
| F2 micro-batching | FIXED (GPU memory test written, not runnable here) | 9b588ca, e1a94c6 |
| F7 replay of notebook / coaching | FIXED | f107f6d, e1a94c6 |
| F22 batch_tokens < ctx_len | FIXED | dde9652 |
| F23 lone surrogates | FIXED | b3fa351 |
| F47 untested resume branches | FIXED (tests only; the code was already right) | 34fd5f8, e1a94c6 |

All commands run from `/home/user/AI-Race/ml` with `uv run --no-sync`.

## F2: micro-batches of at most 8192 tokens

**Defect, reproduced** (probe in the scratchpad, unmodified trainer, spy on `Transformer.forward`):
`FORWARD CALLS OF STEP 1 (batch, ctx): [(128, 256)]` and `judge: batch_size 128 ctx 256 tokens/step 32768`. One forward and one backward over all 128 rows.

**RED** (`pytest tests/test_trainer.py -k "micro or judge_recipe or bit_identical"`): 7 failed, with
`AttributeError: module 'airace_ml.train.trainer' has no attribute 'MICRO_BATCH_TOKENS'` (the first run also hit a spy signature slip in my test, fixed).

**Fix** (`trainer.py`): `MICRO_BATCH_TOKENS = 8192`. `_train_step` keeps the single `next_batch(cfg.batch_size)` draw, then calls the new `_Run._forward_backward(x, y)`: rows per pass = `max(1, 8192 // ctx_len)`; per pass forward under autocast, `cross_entropy(logits.float().flatten(0,1), y_i.flatten(), reduction="sum") / tokens_per_step`, `del logits`, `backward()`, `total += loss.detach()`. After the loop: `after_backward` hook, ONE clip, ONE host sync, one spike check, one optimizer step, exactly as before. `judge_train_config` is untouched (32768 tokens).

**GREEN** (`pytest tests/test_trainer.py -k "micro or bit_identical"`): `5 passed`. Tests added to `tests/test_trainer.py`:
- `test_micro_batches_give_the_same_step_as_the_whole_batch` (3 cases: caps of 512 -> forwards `[8, 8]`, 192 -> `[3,3,3,3,3,1]`, 10 -> `[1]*16`): rows per forward asserted; loss at steps 1-3 `approx(rel=1e-4)`; step-1 gradients `assert_close(rtol=1e-3, atol=1e-6)`; final weights `assert_close(atol=2e-3)`.
- `test_a_batch_in_one_micro_batch_is_bit_identical_to_the_plain_step`: runs 3 steps with the old formulation (whole batch, mean cross-entropy, one backward) patched in as `_forward_backward`, and asserts losses `==`, gradients and final weights `torch.equal` to the new code's single-chunk run.
- `test_micro_batches_do_not_change_what_a_resumed_run_draws`: with a 192-token cap, an interrupted run resumes to the telemetry and weights of an uninterrupted one.
- `test_a_judge_recipe_step_fits_the_8gb_reference_card` (`@pytest.mark.gpu`, skipped without CUDA): one step of `judge_train_config()` on CUDA, `torch.cuda.max_memory_allocated() < 6 GiB`. NOT RUN here (no GPU). Its body was dry-run on the CPU with a toy shape (config route, tiny data root, prep thorough + dedup + fact_check) and it works up to the memory assertion.
- Mutation check: dividing each chunk by its own token count instead of the batch's makes the equivalence tests fail.

Bit-identity argument for CUDA (not testable here): for one chunk, `sum / N` and `mean` run the same float32 operations (nll_loss mean is `acc / total_weight`; its backward is `-(grad / total_weight)`, the same as div-backward of `1/N` then sum-backward), and `0 + x == x`. Pinned on the CPU by the bit-identical test.

Side effects: gate runs (batch_tokens 16384, ctx 256 -> 64 rows) now take two passes of 32 rows, so their numerics differ from before by float summation order. The gate's run reuse is keyed on code, so a full gate rerun redoes its runs.

Files: `ml/src/airace_ml/train/trainer.py`, `ml/tests/test_trainer.py`.

## F7: replay covers base datasets only

**RED** (`pytest tests/test_trainer.py -k "replay_covers or its_own_share or custom_data"`): 3 failed:
- `MixtureError: dataset 'notebook' has weight but no usable documents` (child without a notebook, and the grandchild),
- `MixtureError: dataset 'coaching' has weight but no usable documents` (parent trained on notebook + coaching alone),
- `assert {'code': 0.60..} == approx({'code': 0.6, 'notebook': 0.2, 'creative': 0.2})`: obtained notebook 0.25, creative 0.15 (the parent's notebook share blended in on top of the child's own).
Re-run against the pre-fix source after I shortened the tests: same 3 failures.

**Fix** (`trainer.py`, `config.py`): new `_stable_mixture(cfg, parent)` used by `_prepare`. With replay > 0 and a parent it filters the parent's `last_mixture` to `DATASET_IDS`, records the dropped ids (positive-weight, non-base, sorted), and if no base dataset has positive weight left it drops replay for the run (`effective_mixture(cfg.mixture, 0.0, None)`, i.e. normalized `mixture`). Otherwise `effective_mixture` renormalizes the filtered parent mix as before. `_Setup.replay_dropped` -> `"replay_dropped"` in `_Run._summary` (so in `meta.runs[-1]`, `result.json` and the `Done` event). It is always present: `[]` when nothing was dropped or no replay happened. Contract documented in the new `TrainRunConfig` class docstring (the class had none; the `replay` paragraph is the ruled "replay docstring").

**GREEN** (same command): `3 passed`. Tests:
- `test_replay_covers_base_datasets_only_down_a_lineage`: notebook parent a (last_mixture creative .75 / notebook .25) -> child b (no notebook, replay .3) passes with `{code .7, creative .3}`, `replay_dropped == ["notebook"]`, no notebook tokens -> grandchild c (replay .3) passes with `{web .7, code .21, creative .09}`; run history is a, b, c; a's `replay_dropped == []`.
- `test_a_child_with_a_notebook_gets_only_its_own_share_for_it`: mixture code 3 / notebook 1, replay .2, own notebook -> `{code .6, notebook .2, creative .2}`.
- `test_replay_of_a_parent_with_nothing_but_custom_data_is_dropped`: parent notebook + coaching only; child replay .5 trains on its own mixture alone (`{code .25, web .75}`), `replay_dropped == ["coaching", "notebook"]`, sums to 1.
- The existing `test_replay_notebook_and_coaching` (parent without custom data) passes unchanged.

Files: `ml/src/airace_ml/train/trainer.py`, `ml/src/airace_ml/train/config.py`, `ml/tests/test_trainer.py`.

## F22: batch_tokens below the attention span

**RED** (`pytest tests/test_train_config.py -q`): `Failed: DID NOT RAISE ValueError` x5 for `test_validate_rejects_a_batch_smaller_than_the_attention_span` with (ctx, batch_tokens) in (256,100), (256,255), (1024,512), (64,63), (64,1).

**Fix** (`config.py`, `validate()`): after the `batch_tokens` integer check, `if batch_tokens < shape.ctx_len: raise ValueError("batch_tokens must be at least the attention span (ctx_len), got N for a span of M")`. `steps` and `batch_size` are unchanged (the `max(1, ...)` floor stays; `test_config_steps_and_batch_size_floor_at_one` still passes, as it never calls `validate`). The message begins with the ruled text and adds the two numbers.

**GREEN** (`pytest tests/test_train_config.py`): 148 passed at that commit. Added `test_validate_rejects_a_batch_smaller_than_the_attention_span` (5 params, message matched with `re.escape` of the ruled text) and `test_validate_accepts_a_batch_of_at_least_one_row` ((256,256), (256,257), (384,16384), (640,16384), (1024,1024): divisibility is not required, so the default 16384 keeps accepting spans such as 384/640/832).

Files: `ml/src/airace_ml/train/config.py`, `ml/tests/test_train_config.py`.

## F23: lone surrogates

**RED**: `test_train_config.py::test_validate_rejects_a_lone_surrogate_in_the_players_text` 5 x `DID NOT RAISE ValueError` (notebook[1], notebook[0], coaching[1][0], coaching[0][1], probe_prompts[2]). End to end on the old source, `test_trainer.py::test_a_lone_surrogate_in_the_players_text_fails_before_training` x3: `TypeError: TextInputSequence must be str` (as in the finding; the probe-prompt case only after the run had started).

**Fix** (`config.py`): `_check_text(where, text)` does `text.encode("utf-8")` and on `UnicodeEncodeError` raises `ValueError(f"{where} contains a lone surrogate, not valid text")`. `validate()` runs it for every notebook entry (`notebook[i]`), coaching part (`coaching[i][j]`) and probe prompt (`probe_prompts[i]`), after the existing type checks.

**GREEN** (`pytest tests/test_train_config.py tests/test_trainer.py -k "attention_span or one_row or surrogate"`): `19 passed`. Tests: the 5 rejections above, each also through `from_json(json.dumps(...))` (surrogates arrive as `\ud800` escapes); a test that real emoji, accents and CJK still validate; and the 3 trainer-level cases (`ValueError`, `out_dir` never created).

Files: `ml/src/airace_ml/train/config.py`, `ml/tests/test_train_config.py`, `ml/tests/test_trainer.py`.

## F47: three untested resume branches

The code was correct; these are regression guards. All pass on the current source. Each was checked to FAIL when the code it guards is broken (mutations applied to `trainer.py`, then restored):

| Test | Guards | Mutation that makes it fail |
|---|---|---|
| `test_resume_inside_the_finishing_phase_trains_the_rest_on_the_finishing_mix` (30 steps, decay from step 25, stop at 27) | `_load_resume` re-deriving the finishing mixture (old line 617-618) | replace the condition with `if False:` -> telemetry differs (a Progress loss of 5.68 instead of 4.65, in the first run of this mutation) |
| `test_resume_after_a_rollback_at_the_stop_step_keeps_rolling_back` (NaN at 30 and 35, stop at 30, resume) | the snapshot rebuilt when none was saved (old line 609-611) | build the snapshot from live references -> telemetry differs |
| `test_training_goes_on_when_windows_refuses_to_rename_the_resume_folder` x2 (`Path.rename` raises `PermissionError`: every rename of a resume folder, and moving the previous state aside only) | `contextlib.suppress(OSError)` swap (old line 599-604) and `_newest_resume_state` | drop the suppress -> `PermissionError` propagates; prefer the folder named `resume` over a newer one -> resumes from the older state (the first-Progress assertion fails) |

The rename tests assert a `resume.tmp.*` folder was left (the refused path really ran), `can_resume`, the first `Progress` after the resume (state of step 12 was used, not 6), final weights and telemetry equal to the uninterrupted run, and that no `resume*` folder remains at the end. Reference runs are module-scoped fixtures (`plain_run`, `finishing_run`, `rolled_back_run`), so each reference trains once.

**GREEN** (`pytest tests/test_trainer.py -k "finishing_phase or stop_step or refuses"`): `4 passed`.

Files: `ml/tests/test_trainer.py` (one import added: `Path`).

## Suite

- Default suite (`uv run --no-sync pytest -q`), final tree: **1403 passed, 1 skipped, 9 deselected, 159.24 s**. No warnings in the output.
- Same machine, same time of day, base commit `afee112` extracted with `git archive` and run against the same venv: **1372 passed, 1 skipped, 8 deselected, 146.52 s**. So group D adds 31 tests and about 13 s. The cloud box is slower than the owner's PC (the verifier measured 114 s for 1247 tests), and its load average was 2-2.6 during both runs, so the absolute figure carries noise. Group C's additions are in both numbers.
- Ruff (`uv run --no-sync ruff check .`): **clean**.
- Slow suite (`uv run --no-sync pytest -m slow -q`, includes the quick feasibility gate): **5 passed, 189.81 s**.
- GPU: `pytest -m gpu tests/test_trainer.py`: 2 skipped (no CUDA): `test_starter_speed_gpu` and the new judge-memory test.

## Self-review (adversarial read of the whole diff)

- Accidentally ran `ruff format` over `src tests` once, which reformatted 9 unrelated files; reverted them with `git checkout --` before any commit. The repo as a whole is not `ruff format`-clean (only `ruff check` is the gate), so I formatted only the files I touched, and checked the diffs were limited to my lines.
- F2 test tolerances: gradients are compared only at step 1 (identical weights). Later steps and the final weights diverge by float noise that AdamW's first steps can amplify on near-zero gradients, hence the loose weight tolerance (`atol=2e-3`); the strong checks are the step-1 gradients, the three logged losses, the forward row counts and the bit-identity test.
- The micro-batch probe tests replace `_Run._evaluate` with a no-op (they look at steps, and the eval dominates their time). No other test does.
- The F7 filter keeps zero-weight base keys of the parent's mix (as before) but drops non-base keys regardless of weight from the blend; only positive-weight dropped ids are reported.
- Split F22 / F23 into separate commits by staging intermediate file versions; both commits were tested on their own (F22 alone: 148 passed in `test_train_config.py`).

## Concerns

1. **The GPU memory test has not run.** The 6 GiB bound and the "about a third of ~7 GiB" estimate come from the finder's CPU probe (203 KiB/token for the judge shape). The owner should run `pytest -m gpu tests/test_trainer.py` on the RTX 3050 before the 1 h `build-judge`.
2. **8192 tokens may still be too much for the largest legal shapes.** Scaling the probe by layers x width, a 16-layer x 640-wide model needs roughly 0.6 MiB/token, so one 8192-token micro-batch is around 5 GiB of activations plus about 1.3 GiB of weights, gradients and Adam state: tight on an 8 GB card that also drives the desktop. `MICRO_BATCH_TOKENS` is a single constant; the judge shape and the gate shapes are comfortably inside. If the owner's card needs less, lower it (it changes only summation order, not the result).
3. Gate numerics shift slightly (see F2): results already stored under `runs/` are keyed on the old code and will be redone on the next `airace-ml gate`.

## Other observations (not fixed; outside the notes)

- `config.py:133` (a docstring line in `_build_nested`) and `train/__init__.py:1` exceed 100 characters; ruff does not flag them.
- `test_config_steps_and_batch_size_floor_at_one` still builds a config with `batch_tokens` (100) below `ctx_len` (1024) to test the `max(1, ...)` floor. It never calls `validate()`, so it is unaffected; the floor is now unreachable for a validated config.
- Nothing else noticed.
