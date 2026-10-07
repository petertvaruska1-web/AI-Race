# Task 8 report: trainer

**Status:** DONE_WITH_CONCERNS. All tests pass. The concerns are design decisions I made where the brief was silent; they are listed for a ruling.
**Commit:** `a0ea8e5 feat(ml): trainer with growth, replay, finishing mix, rollback, resume, telemetry` (with the `Co-Authored-By` trailer). Only the two new files are staged. LF line endings.

## What I implemented

`ml/src/airace_ml/train/trainer.py`:

- **Public API (as the brief specifies):**
  - `PROBE_PROMPTS` (exact value)
  - `TrainHooks(loss_override, stop_after_steps)`
  - `TrainResult` with the brief's fields
  - `train_run(cfg, *, out_dir, data_root, device, on_event, resume, checkpoint_every, _hooks)`

- **Validation before any side effect** (`_prepare`), in the brief's order:
  1. `cfg.validate()`
  2. Load the tokenizer.
  3. Build the custom docs:
     - Notebook entries are stripped, blank ones are dropped, then `encode_doc`.
     - Coaching pairs: `encode_chat([("user", u), ("ai", a)])`.
  4. `load_pools` → `MixtureSampler`, which raises `MixtureError`.
  5. Load the parent and grow it, which raises `GrowthError`.

  More checks before `out_dir` is created:
  - The parent's `meta.json` is read before step 4, because replay needs the parent's `last_mixture`.
  - The finishing mixture is checked up front (`set_weights(finishing)`, then back to stable), so a bad finishing mix fails before training, not at the decay start.
  - `checkpoint_every` is checked (`TypeError` if not an int, `ValueError` if < 1).
  - On resume, the saved state must exist and must have been written by the same config.

  On any failure, nothing is created.

- **Model and optimizer:**
  - A root run gets a fresh `Transformer` seeded with `cfg.seed` inside `fork_rng`, so the global RNG is untouched.
  - A continued run gets the parent, or `grow(parent, cfg.shape, seed=cfg.seed)` when the shape differs.
  - The model is built on CPU, then `.to(device)`. `device=None` → `pick_device()`.
  - AdamW: betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on params with ≥2 dims and 0 on the rest. `fused=True` on CUDA.
  - bf16 autocast on CUDA. Gradients are clipped to `learning_style(boldness).clip_norm`.

- **Per step:**
  - `lr = wsd_lr(i, steps, style) * lr_scale`.
  - Forward and backward, then one `loss.item()` (the only per-step host sync).
  - `loss_override` (if set) replaces the logged loss, which then feeds the `SpikeDetector`.
  - The optimizer step is skipped when the loss is a spike.

- **Rollback (R4):**
  - An in-memory CPU snapshot (weights plus optimizer state) is taken at step 0 and after every good step that is a multiple of 25.
  - On a spike: restore the snapshot (the bad batch's update is never applied), halve `lr_scale`, emit `Instability(step, "rollback", lr_scale)`. The step counter and sampler keep going forward.
  - On the 3rd spike: restore the snapshot, emit `Instability(step, "stopped", lr_scale)`, run a final held-out eval and samples on the kept weights, save the final checkpoint, and finish with `Done("unstable_stopped")`.
  - `Optimizer.load_state_dict` keeps CPU tensors as passed, and AdamW updates them in place. Without a copy, the next steps would silently overwrite the snapshot. So a restore always loads a fresh copy of the snapshot.

- **Finishing mix:** from the decay start of the WSD schedule (0-based step `steps - max(1, int(DECAY_FRAC * steps))`), the trainer calls `sampler.set_weights(finishing_mixture)`.

- **Telemetry:**
  - `Progress` on step 1, every 10 steps, and on the last step.
  - Every `max(25, steps // 20)` steps and on the last step: a `HeldoutEval` over the 8 base datasets, then one `Sample` per probe prompt.
    - The held-out batch is 8 sequences per dataset from `heldout_docs`, built once per run with a per-dataset rng (seed 0). Datasets missing from `data_root` are skipped.
    - Samples cover `PROBE_PROMPTS` plus `cfg.probe_prompts`: 32 tokens, temperature 0.8, seed = step, via `TorchLM(model, tok)` on the training device. The text is the decoded new text only.
  - Each event is written to `telemetry.jsonl` as `json.dumps(event_to_dict(e), allow_nan=False)`, flushed, then passed to `on_event`.
  - `Done` is the last event. It is emitted after the checkpoint and `result.json` are on disk.

- **Resume (R5):**
  - Every `checkpoint_every` steps, and always at `stop_after_steps`, the trainer writes `out_dir/resume/`:
    - `state.pt`: model, optimizer, and the snapshot only when it differs from the current weights.
    - `state.json`: config, step, `lr_scale`, spike count, last-10 losses, sampler rng state, detector state and warmup, lineage id, elapsed wall time, telemetry byte offset.
  - The state is written to `resume.tmp/` and then swapped in.
  - `resume=True` restores all of it:
    - It re-derives the phase's sampler weights from the step.
    - It truncates `telemetry.jsonl` back to the saved offset, so events after that point are not duplicated.
    - It continues the run and deletes `resume/` when the run completes.
  - A run cut short by `stop_after_steps` returns status `"interrupted"`. It writes no final checkpoint, no `result.json`, and no `Done`.

- **Metadata:** as in the brief.
  - `version_id = run_id`.
  - `lineage_id` is the parent's, or a new uuid4 hex (persisted across a resume).
  - `parent_version_id` and `tokens_trained_total` (parent total + this run).
  - `last_mixture` is the effective stable-phase mixture (normalized, replay included).
  - `runs` = the parent's runs plus `{run_id, steps, tokens, mixture, prep, shape, status}`.
- **Results:**
  - `final_loss` = the mean loss of the last 10 applied (non-spike) steps.
  - `result.json` holds the `TrainResult` fields except `out_dir`, as strict JSON.

## Tests and results

`ml/tests/test_trainer.py` holds the brief's 6 tests and the GPU test, plus 6 added tests. An autouse fixture sets `AIRACE_DEVICE=cpu`, so the default suite runs on CPU on this CUDA machine. The GPU test passes `device=cuda` explicitly.

The added tests:

| Test | What it checks |
|---|---|
| `test_resume_continues_where_it_stopped` | The resumed run's first `Progress` is at step 30, not step 1. A resume that silently restarted from scratch would pass the brief's weight-equality test, but not this one. The interrupted-and-resumed `telemetry.jsonl` is byte-identical to the uninterrupted run's. `resume/` is gone after completion. A resume with a different config raises `ValueError`. Resuming a finished run raises `FileNotFoundError`. |
| `test_resume_keeps_the_rollback_snapshot` | Interrupt at step 30, when the snapshot (step 25) is older than the weights, then a NaN at step 35 after resume. The weights and telemetry equal an uninterrupted run with the same spike. |
| `test_rollback_halves_lr_and_stop_keeps_last_good_weights` | The instability sequence is exactly `[(20, rollback, 0.5), (30, rollback, 0.25), (40, stopped, 0.25)]`. The held-out losses at the stop equal those at step 25, which proves the kept weights are the last good snapshot. The checkpoint's last run has status `unstable_stopped`. |
| `test_finishing_mix_loss_level_is_not_instability` | A loss level 10 higher after the switch to the finishing mix completes instead of stopping. See decision 1. |
| `test_replay_notebook_and_coaching` | With replay 0.5 from a creative parent, `last_mixture` is `{notebook: .25, coaching: .25, creative: .5}`. All three datasets get tokens. Blank notebook entries and coaching pairs with a blank side are dropped. A coaching set made only of blank pairs fails with `MixtureError` naming `coaching`, before `out_dir` is created. |
| `test_attention_span_cannot_shrink` | A parent with ctx 128 and a child with ctx 64 raises `GrowthError`, and nothing is created. |

Results:
- `python -m uv run pytest tests/test_trainer.py -v`: **12 passed**, 1 deselected (gpu), 22.2 s.
- Full suite `python -m uv run pytest`: **279 passed, 3 deselected in 24.78s**, with no warnings.
- `python -m uv run ruff check .`: **All checks passed!**
- `python -m uv run pytest -m gpu -v`: **3 passed** (including `test_starter_speed_gpu`).

### GPU throughput (RTX 3050, 8 GB)

`ModelShape(4, 128, 256)`, 1,048,576 tokens, `batch_tokens=16384` (the config default; 64 steps):

| Measure | Value |
|---|---|
| Cold process (includes ~4.4 s of CUDA/cuBLAS init before step 1) | wall 10.03 s → **104,553 tok/s** |
| Warm (second run in the same process) | wall 5.51 s → **190,290 tok/s** |
| Threshold | 30,000 tok/s |

Profile of one steady-state step:
- The full step takes 49–65 ms (it varies with GPU clocks).
- GPU work alone, on a fixed batch with no sync, is about the same, so the loop is GPU-bound.
- `sampler.next_batch(64)` takes 1.1 ms.
- A snapshot copy takes 10 ms, once every 25 steps.
- The probe samples (3 prompts × 32 tokens) take about 0.85 s per eval.

No prefetching is needed.

Also checked on CUDA (a scratch test, not committed): two rollbacks to the same snapshot (NaN at 30 and 35), plus interrupt-at-32 and resume, give weights **bitwise equal** (max diff 0.0) to the uninterrupted run. This exercises fused-AdamW `load_state_dict` from the CPU snapshot.

## TDD evidence

**RED** (expected: the module does not exist yet):
```
$ python -m uv run pytest tests/test_trainer.py -v
E   ModuleNotFoundError: No module named 'airace_ml.train.trainer'
ERROR tests/test_trainer.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

**GREEN** (the brief's tests, first implementation):
```
$ python -m uv run pytest tests/test_trainer.py -v --durations=10
tests/test_trainer.py::test_loss_decreases_and_events PASSED
tests/test_trainer.py::test_continue_and_grow PASSED
tests/test_trainer.py::test_resume_equivalence PASSED
tests/test_trainer.py::test_spike_rollback_and_stop PASSED
tests/test_trainer.py::test_invalid_mixture_fails_before_training PASSED
tests/test_trainer.py::test_finishing_mixture PASSED
====================== 6 passed, 1 deselected in 13.69s =======================
```

**RED for the added tests.** These were written after the implementation, so I mutation-checked the two that guard my own design decisions. Each mutation was applied temporarily, the tests were run, and the file was restored:
- Removing the detector re-warm at the phase switch fails `test_finishing_mix_loss_level_is_not_instability` with `AssertionError: assert 'unstable_stopped' == 'completed'`.
- Never persisting the snapshot in the resume state fails `test_resume_keeps_the_rollback_snapshot` (the telemetry differs).
- After restoring: `2 passed`.

**GREEN (final):** 12 passed (trainer), 279 passed (full suite), 3 passed (gpu).

## Files changed

- `ml/src/airace_ml/train/trainer.py` (new, 639 lines including docstrings)
- `ml/tests/test_trainer.py` (new)

## Changes to the brief's test code (formatting and lint only; no assertion changed)

The repo's ruff 0.16 default rule set is wider than plain E/F. The brief's code failed it, so:
- Applied `ruff format`.
- Removed the unused `json` import (F401).
- Split the `;` statements (E702).
- Renamed the lambda parameter `l` to `loss` (E741).
- Rewrote `cfg()`'s `dict(...)` as a dict literal (C408).

I also added the autouse `AIRACE_DEVICE=cpu` fixture, which the controller ruling allows, and the GPU test passes `device=torch.device("cuda")`.

## Decisions where the brief was silent (please rule if you disagree)

1. **The spike detector re-learns the loss level when the finishing mix starts.** At the switch, the detector is replaced by `SpikeDetector(decay_start + warmup)`. NaN or inf is still always a spike.
   - Why: the detector never folds spikes into its EMA. In a long run with a small EMA std, switching to harder finishing data (for example stories → code) puts the loss above `1.3 × mean` and `mean + 6σ` on every step. That would roll back three times and stop the run, punishing a legitimate player choice as "instability".
   - The detector's warmup (the run start and the re-warm) equals the LR warmup length, `max(1, int(warmup_frac * steps))`.
2. **The finishing phase starts at the WSD decay start**, `steps - max(1, int(DECAY_FRAC * steps))`, per the ruling's "decay start" wording. This equals the brief's `i >= steps * (1 - DECAY_FRAC)` for every run of 5 or more steps. For runs under 5 steps, the brief's formula would never use the finishing mix, while the LR still decays on the last step. With this choice, the last step does use it.
3. **Player probe prompts (`cfg.probe_prompts`) are sampled as chat** (`chat_reply([("user", p)])`), because the player is talking to their AI. This path also reserves room for the reply in the context, while `complete()` does not (Task 5 minor). `top_p` is TorchLM's default of 0.95 (the brief gives only temperature 0.8).
4. **Coaching pairs are stripped, and a pair is dropped when either side is blank.** This mirrors the notebook rule. A blank reply would teach empty answers. A coaching set of only blank pairs therefore raises `MixtureError` (Review Focus 3).
5. **A `ctx_len` shrink raises `GrowthError`** in the trainer. `grow()` itself only checks layers and width, and the global constraint says shape only grows.
6. **Parent vocabulary vs tokenizer mismatch** raises `ValueError` before training.
7. **`tokens`** counts every step drawn, rolled-back steps included. That matches the sampler's `tokens_per_dataset`, and `tokens_trained_total` adds the same number.
8. **`final_loss` when no step was ever applied** (for example, spikes on every step before any good one): it falls back to the mean held-out loss of the kept weights, then 0.0. It is always finite.
9. **Fresh vs resumed runs in an existing `out_dir`:**
   - A fresh run into an existing `out_dir` truncates `telemetry.jsonl` and deletes any stale `resume/`.
   - `resume=True` with no saved state raises `FileNotFoundError`. It never silently restarts.
10. **Probe samples are generated one prompt per `generate` call**, so each prompt's sample is reproducible regardless of the other prompts.

## Self-review findings

- **Completeness:** every brief bullet and ruling (R4, R5, notebook stripping, phase weights on resume, device, validation order, `final_loss`, strict telemetry, probe samples on the training device) is implemented and covered by a test.
- **Resume swap is not fully atomic.** The new state goes to `resume.tmp/`; then the old `resume/` is removed and the tmp dir renamed. A crash between those two steps loses the resume state. This is minor, the same class as the deferred `save_checkpoint` / `write_corpus` non-atomic notes.
- **Private import:** `_json_safe` comes from `train/events.py`, for `result.json`, instead of duplicating it.
- **Duplicate meta read:** the parent's `meta.json` is read twice (once early for replay, once by `load_checkpoint`). It's trivial.
- **Probe sampling cost:** about 9 ms per generated token on this GPU (TorchLM's Python decode loop), about 0.85 s per eval. That is fine at the current eval cadence, but worth knowing for larger shapes.
- **File size:** 639 lines with docstrings, one responsibility (the run loop and its state). I kept it as one module per the plan's file structure.

## Concerns

- Decisions 1–5 above go slightly beyond the brief's letter. Decision 1 (re-warm at the phase switch) is the one with real product impact. Without it, long runs with a harder finishing mix would end as `unstable_stopped`.

---

# Fix round 1

**Commit:** `d7123d2 fix(ml): trainer lets the attention span change freely, rolls back on non-finite gradients with validated snapshots, survives a refused resume swap; bad mixtures raise MixtureError` (with the `Co-Authored-By` trailer). It is on top of `a0ea8e5`.

**Files:**
- `ml/src/airace_ml/train/trainer.py`
- `ml/src/airace_ml/train/config.py`
- `ml/src/airace_ml/train/events.py`
- `ml/tests/test_trainer.py`

## What changed

1. **(Critical) The attention span changes freely within a lineage, per spec §4.3.**
   - Removed the trainer's `ctx_len` shrink check (and the now-unused `GrowthError` import). A shape change goes through `grow()`, which still raises `GrowthError` for fewer layers or a narrower width and handles a context change in either direction.
   - `test_attention_span_cannot_shrink` is replaced by `test_attention_span_changes_freely_within_a_lineage`. The chain is a (2, 96, ctx 128) → b (ctx 64) → c (ctx 128). The test asserts:
     - Both continuations complete.
     - One lineage id throughout.
     - `meta.shape.ctx_len` is 64 then 128.
     - c's parent is b, and c has 3 runs.
     - From c, fewer layers `(1, 96, 128)` and a narrower width `(2, 64, 128)` each raise `GrowthError`, with no `out_dir` created.
2. **(Important) Every unsampleable mixture raises `MixtureError` (Review Focus 3).**
   - `config.py` imports `MixtureError` from `airace_ml.data.sampler`. There is no import cycle: sampler imports nothing from `train`, so nothing had to move.
   - The mixture weight and id checks (`_normalized`, `_check_mixture`, used for `mixture` and `finishing_mixture`, and by `effective_mixture`) now raise it.
   - It subclasses `ValueError`, so `validate()`'s contract holds, and the messages still start with the field name.
   - Container checks (a mixture that is not an object) stay plain `ValueError`, as structural errors. The `validate()` docstring now says so.
   - New parametrized `test_unsampleable_mixtures_fail_before_training`. Each case raises `MixtureError` with no `out_dir`:
     - all-zero weights
     - a negative weight
     - an unknown id
     - a positive weight on a blank notebook pool
     - a finishing mixture on a blank notebook
     - an unknown finishing id
   - Task 7's tests pass unchanged.
3. **(Important) Non-finite gradients, and snapshots only of validated weights.**
   - `_train_step` keeps `clip_grad_norm_`'s total norm and fetches it with the loss in the one host sync: `torch.stack([loss.detach().float(), norm.float()]).tolist()`.
   - A non-finite norm is a spike: the update is skipped and the run rolls back. It short-circuits, so the step stays out of the detector's statistics.
   - **Snapshot timing.** The rollback snapshot is now taken *before* `optimizer.step()` on step 26, 51, 76, …, and only when that step's loss and gradient are both good. It therefore holds the weights after 25, 50, … steps, which the following step has validated, and its optimizer state matches them. The step-0 initial snapshot remains.
   - New test hook `TrainHooks.after_backward(step, model)`, which runs between backward and clipping. I chose it over a `grad_norm_override` so the test corrupts *real* gradients and the "final weights finite" assertion actually depends on the fix.
   - New parametrized `test_bad_updates_never_become_the_kept_weights`, injecting at snapshot-interval step 25:
     - `inf` gradients: the only instability is `(25, rollback)`.
     - A finite update that wrecks the model (`norm_f.weight = 1e4`, huge but finite loss): the only instability is `(26, rollback)`; step 26 never became a snapshot.
     - Both cases complete, with finite checkpoint weights and finite held-out losses.
4. **(Minor, must-fix) The resume swap survives Windows file locks.**
   - Each save writes `resume.tmp.<id>/` (`state.json` last).
   - The swap: if `resume/` exists, rename it to `resume.old.<id>`; rename the tmp dir to `resume/`; then best-effort delete every other resume directory. Unique suffixes mean a leftover directory that cannot be deleted never blocks a rename.
   - An `OSError` during the swap is suppressed, so a long run never crashes on a refused rename: a complete state still exists under some name, and the next save tries again.
   - `_read_resume` takes the **newest complete** state (highest `step`, ties preferring `resume/`) among `resume/`, `resume.tmp.*` and `resume.old.*`. A state is complete when `state.json` exists and parses.
   - Fresh runs and finished runs best-effort delete all of them.
   - New `test_resume_recovers_from_an_interrupted_swap`:
     - After an interrupt at step 20, the test renames `resume/` to `resume.tmp.crashed` and adds an incomplete `resume.tmp.partial/` (no `state.json`).
     - `resume=True` completes with weights equal to the uninterrupted run (atol 1e-6) and byte-identical telemetry.
     - No resume directory is left over.
5. **Cleanup:** `_json_safe` is now public `json_safe` in `events.py`; the trainer imports it.

## TDD evidence

**RED.** I wrote the four new tests before any fix:
```
$ python -m uv run pytest tests/test_trainer.py -q -k "attention_span or unsampleable or non_finite_gradient or interrupted_swap"
FAILED test_attention_span_changes_freely_within_a_lineage    GrowthError: attention span 128 cannot become 64
FAILED test_unsampleable_mixtures_fail_before_training[over0] ValueError: mixture: weights must sum to more than 0
FAILED test_unsampleable_mixtures_fail_before_training[over1] ValueError: mixture: dataset 'creative' has an invalid weight -1.0
FAILED test_unsampleable_mixtures_fail_before_training[over2] ValueError: mixture: unknown dataset 'bogus'
FAILED test_unsampleable_mixtures_fail_before_training[over5] ValueError: finishing_mixture: unknown dataset 'bogus'
FAILED test_non_finite_gradient_rolls_back_and_keeps_weights_finite  TypeError: unexpected keyword argument 'after_backward'
FAILED test_resume_recovers_from_an_interrupted_swap          FileNotFoundError: no resume state in ...\p\resume
7 failed, 2 passed, 12 deselected
```
The two cases that already passed (blank pools) were already raising `MixtureError` from the sampler.

**Behavioral RED for the gradient fix.** With only the hook added and no norm check: `RuntimeError: probability tensor contains either inf, nan or element < 0`. The NaN update at snapshot step 25 poisoned the weights, and the run crashed in probe sampling.

**Mutation checks**, run on the finished code and then restored (`cmp` confirmed the file is identical):
- **Old post-update snapshot timing:** the finite-wreck case fails with `[(26, rollback), (27, rollback), (28, stopped)] != [(26, rollback)]`. The run rolled back into the wrecked snapshot.
- **Norm check removed:** the `inf`-gradient case fails (`RuntimeError: probability tensor contains either inf, nan ...`).

**GREEN (final):**
```
$ python -m uv run ruff check .                                             -> All checks passed!
$ python -m uv run pytest tests/test_trainer.py tests/test_train_config.py -q -> 159 passed, 1 deselected
$ python -m uv run pytest                                                   -> 288 passed, 3 deselected (no warnings)
$ python -m uv run pytest -m gpu -q                                         -> 3 passed
```

**CUDA check** (a scratch test, not committed):
- `inf` gradients at steps 25 and 33 give `[(25, rollback, 0.5), (33, rollback, 0.25)]`, the run completes, and the weights are finite.
- Interrupt at 30 plus resume is bitwise equal to the uninterrupted run (max diff 0.0), on the fused AdamW path.
- Starter-shape throughput after the change is 169,249–181,530 tok/s (CUDA already initialized in-process); the threshold is 30,000.

## Notes

- **Resume size.** Because the rollback target is now always older than the current weights, a resume save normally stores the snapshot as well. For a large model, `state.pt` is about twice the size of model plus optimizer state. That is the price of exact resume with validated snapshots.
- **NaN probe crash.** `TorchLM` sampling raises on NaN logits (Task 5 behavior). After this fix, training cannot produce NaN weights: a finite clipped gradient gives a finite AdamW update, and non-finite gradients are rejected. That is why the finite-wreck test case uses huge finite weights rather than NaN.
- **Formatting slip.** During the fix, a broad `ruff format src tests` reformatted six unrelated test files. I reverted them with `git checkout` before committing; only the four files above are in the commit.
- **Long commit subject.** It is longer than usual. I made a new commit rather than amending.
