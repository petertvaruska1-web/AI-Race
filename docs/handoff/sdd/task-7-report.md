# Task 7 report: training config, learning style, WSD schedule, spike detection, events

**Status:** DONE (one note for the controller, see Concerns)
**Commit:** `256ae5b feat(ml): training run config, learning-style mapping, WSD schedule, spike detection` (with the `Co-Authored-By` trailer). The commit's `config.py` is LF; a CRLF working copy produced by my own refactor script was normalized back to LF and re-staged after the commit (no content change, so no amend was needed; the blob in HEAD is LF and the tree is clean).

## What was implemented
All under `ml/src/airace_ml/train/`:

- `__init__.py`: docstring only (same style as the other packages).
- `config.py`:
  - `LearningStyle` (frozen) and `learning_style(boldness)` with the exact mapping from the brief: `peak_lr = 10 ** lerp(log10(6e-4), log10(1.2e-2), t)`, `clip_norm = lerp(0.5, 2.5, t)`, `warmup_frac = lerp(0.06, 0.02, t)`. ValueError outside [0, 1], and also for NaN.
  - `effective_mixture(mixture, replay, parent_mixture)`: `(1 - replay) * norm(mixture) + replay * norm(parent)`. ValueError if `replay` is outside [0, 1], if `replay > 0` and the parent is None, or if either mixture has a negative or non-finite weight or a zero sum. Keys from the parent are only added when `replay > 0`, and keys from `mixture` only when `replay < 1`. Zero-weight keys inside a contributing mixture are kept with weight 0.0.
  - `TrainRunConfig` (non-frozen dataclass, `field(default_factory=...)` for every mutable default, including `prep`) with `steps` and `batch_size` properties, `validate()`, `to_json()` and `from_json()`.
- `schedule.py`: `DECAY_FRAC = 0.2`, `MIN_LR_RATIO = 0.1`, `wsd_lr`, implemented exactly per the clarification. Warmup `W = max(1, int(warmup_frac * T))` with `lr = peak * (step + 1) / W`. Decay `D = max(1, int(0.2 * T))` starting at peak at `T - D` and reaching `0.1 * peak` at `T - 1`. For `D == 1` the single decay step is the floor. Where warmup and decay overlap (runs of 1 to 2 steps) the lower rate wins. Steps past the end hold the floor. ValueError for `total_steps < 1` or `step < 0`.
- `stability.py`: `SpikeDetector(warmup_steps, alpha=0.02, z=6.0, ratio=1.3)`. Non-finite loss is always a spike. After warmup (and once at least one loss has been folded in), a spike needs `loss > mean + z * std` **and** `loss > ratio * mean`. Spikes are never folded into the EMA mean or variance. During warmup, losses are folded and never flagged, apart from non-finite ones. `state()` returns `{"mean", "var", "count"}` (JSON-safe) and `load_state()` restores it exactly.
- `events.py`: `Progress`, `HeldoutEval`, `Sample`, `Instability`, `Done` as frozen dataclasses, `TrainEvent` union, `event_to_dict` (`{"type": <lowercase class name>, **asdict(e)}`).

### `TrainRunConfig` details
- `to_json` serializes `shape` and `prep` through their own `to_dict`. `from_json` rebuilds them via `from_dict` and converts `coaching` back to a list of tuples. `from_json(c.to_json()) == c` holds, including coaching tuples and a non-default `PrepConfig`.
- `from_json` raises ValueError on a non-object document, on unknown fields and on missing required fields (naming them). It does not call `validate()`.
- `validate()` checks exactly what the brief lists: `shape.validate()` (ShapeError is a ValueError), `token_budget >= batch_tokens`, mixture keys in `DATASET_IDS + CUSTOM_DATASET_IDS` with weights >= 0 and sum > 0, `0 <= replay <= 0.9`, purchases in (0, 1], boldness in [0, 1]. Every message names the offending field.

### Additions beyond the brief (small, each defensive)
Please check these are acceptable:
1. `validate()` also requires `batch_tokens >= 1` (otherwise `steps` would divide by zero).
2. `validate()` applies the same key/weight check to `finishing_mixture` when it is not None, so a bad finishing mixture fails at validate time instead of mid-run.
3. `validate()` requires purchases keys to be known dataset IDs (a misspelled key would otherwise silently mean "buy everything" because missing = 1.0), and checks that `notebook` entries are strings and `coaching` entries are `(str, str)` pairs, since a config arrives as JSON from outside Python.
4. `from_json` strictness (unknown / missing field errors) is not in the brief.
5. `event_to_dict` does not sanitize non-finite floats (see Concerns).

## Tests and results
`ml/tests/test_train_config.py`: the six brief tests verbatim, plus 29 extra cases (35 total):
- WSD: runs of 1, 2, 3 and 5 steps, the floor past the end, input errors, decay starting exactly at peak at `T - 200`, linear warmup value.
- `learning_style`: NaN and negative rejected.
- `effective_mixture`: full replay, zero-weight keys, a hand-computed three-key blend, invalid weights on either side.
- Validation: 16 parametrized cases asserting the error message **names the field** (the brief's loop only proves "some ValueError"), plus a fully populated valid config.
- Config JSON: round trip with every field set (tuples and nested `ModelShape` / `PrepConfig` restored), defaults not shared between instances, unknown / missing / non-object JSON, `steps` and `batch_size` floors.
- `SpikeDetector`: no spike during warmup, spikes not folded (state unchanged after a spike), both conditions required (noisy series and zero-variance series), and a resume test (state saved through JSON at step 150, then the remaining 150 flags and final state are identical to an uninterrupted run).
- Events: all five types give the right `"type"` string, and every dict survives a `json.dumps`/`loads` round trip.

Lint/format note: ruff 0.16.10 flagged C408 (`dict()` call) 21 times in the test file, including four in the brief's `test_config_roundtrip_and_validation`. I applied `ruff check --select C408 --fix --unsafe-fixes` to that file only (pure `dict(k=v)` to `{"k": v}` rewrites; no assertion changed) and `ruff format` to the new files. TRY004 flagged two `isinstance` + `raise ValueError` spots in `config.py`; I restructured them (an `all(isinstance(...))` check and `type(d) is not dict`) rather than switching the exception type, because the brief requires ValueError. Pre-existing test files in the repo are not ruff-formatted; I left them alone.

## TDD evidence
RED (before any `train/` code existed):
```
cd ml && python -m uv run pytest tests/test_train_config.py -v
E   ModuleNotFoundError: No module named 'airace_ml.train'
ERROR tests/test_train_config.py
1 error in 0.10s
```
Expected: the module under test did not exist yet.

GREEN, brief tests only (first implementation): `6 passed in 0.04s`.

GREEN, final state:
```
python -m uv run pytest tests/test_train_config.py -q -W error   ->  35 passed in 0.10s
python -m uv run ruff check .                                     ->  All checks passed!
python -m uv run ruff format --check src/airace_ml/train tests/test_train_config.py -> 6 files already formatted
python -m uv run pytest                                           ->  164 passed, 2 deselected in 4.40s
```
No warnings in test output.

## Files changed
Created (all in the commit):
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\__init__.py`
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\config.py`
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\schedule.py`
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\stability.py`
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\events.py`
- `C:\Users\petko\Desktop\AI Race\ml\tests\test_train_config.py`

## Self-review findings
- Import direction is clean: `schedule.py` imports `LearningStyle` from `config.py`; `config.py` imports nothing from `train/`. No cycles.
- `_check_mixture` reuses `_normalized` for the weight/sum rules, so `validate()` and `effective_mixture` share one definition of a valid mixture (found and removed a duplicate during review).
- Tests exercise real behavior (the resume test replays actual loss sequences) and none of the brief's assertions were touched.
- All file I/O in the new code: none; the new tests do not touch the filesystem.

## Concerns
1. **`event_to_dict` and non-finite floats.** It returns plain Python values only; it does not turn `nan`/`inf` into `None`. `json.dumps` writes bare `NaN`/`Infinity` for them, which is not valid JSON and which `JSON.parse` on the Node side rejects. If Task 8 ever puts a non-finite loss into `Progress`, the event stream would break at the bridge. Suggest Task 8 only emits `Progress` for finite losses (the `Instability` event covers the non-finite case) or serializes with `allow_nan=False` so it fails loudly. I left the brief's contract unchanged.
2. **A persistent loss-level shift.** Because spikes are never folded into the EMA, a loss that jumps and stays high flags every following step. That is the intended input for Task 8's rollback / stop logic (it must cap the number of rollbacks), but Task 8 should know the detector never adapts to a new level on its own.
3. `TrainRunConfig.validate()` does not require `parent_dir` when `replay > 0` (the brief does not list it). Task 8 will hit `effective_mixture`'s ValueError if the parent mixture cannot be found; it may want to check this up front.

---

# Fix round 1 (commit `bbc8f99 fix(ml): TrainRunConfig rejects malformed JSON with ValueError; event_to_dict emits strict JSON`)

## Issue 1: malformed external JSON must give ValueError and never be mis-parsed

### `from_json` (structure only; it still does not call `validate()`)
- `coaching` is now converted by `_coaching_from_json`: each entry must be a JSON **list** of exactly two strings, else `ValueError("coaching[i] must be a [prompt, reply] pair of strings ...")`. Dict pairs and 2-character strings are no longer coerced by `tuple(...)`.
- New `_check_containers` runs on the raw document (and again from `validate()`): `mixture` and `purchases` must be objects, `finishing_mixture` an object or null, and `notebook`, `coaching`, `probe_prompts` lists. Each error names the field.
- `shape` and `prep` go through `_build_nested`: not an object (including null) is a ValueError; unknown keys are a ValueError (so a stray key in `prep` or `shape` is no longer a TypeError or silently ignored); a missing key (`KeyError`) and any `TypeError`/`ValueError` from `from_dict` are re-raised as `ValueError("shape: ...")` / `ValueError("prep: ...")`.
- `json.loads` errors on malformed text are already a `ValueError` subclass (tested).

### `validate()` (now also the type gate)
Every check raises `ValueError` naming the field; nothing else may escape:
- `seed`: int (not bool), >= 0.
- `run_id`: full match of `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` (so `""`, `"../../x"`, `"a/b"`, `"a\b"`, `"a b"`, `".hidden"`, a trailing newline and 65+ characters are rejected).
- `token_budget` and `batch_tokens`: int (not bool, not float), so `32768.5` and `32768.0` are rejected.
- `replay`, `boldness`, every purchase fraction and every mixture weight (`mixture` and `finishing_mixture`): real number (int or float, not bool), finite, in range. This also fixed the `TypeError` from comparing a string weight.
- `shape` must be a `ModelShape` (a dict or null is a ValueError naming `shape`); the existing `shape.validate()` already rejects non-int fields.
- `prep` must be a `PrepConfig` and `prep.dedup` / `prep.fact_check` must be real bools (`"dedup": "no"` was previously truthy and silently turned dedup on).
- `probe_prompts` must be strings; `parent_dir` must be null or a non-empty string.
- `replay > 0` with `parent_dir` None is a ValueError naming `parent_dir`.
- Containers are type-checked before use, so null `mixture` / `purchases` / `notebook` / `coaching` give a ValueError instead of `AttributeError` / `TypeError`.
- Public helpers hardened the same way: `learning_style` and `effective_mixture` reject non-numbers with ValueError instead of `TypeError`.

Implementation note: ruff's TRY004 flags `if not isinstance(...): raise ValueError`, but ValueError is the required contract. A small helper `_expect(value, kind, message)` (which returns early on success and then raises ValueError) centralizes those type checks with the explanation in its docstring; no `noqa` and no config change.

### Tests added (all in `ml/tests/test_train_config.py`)
- `test_validate_rejects_malformed_values_with_a_value_error`: 68 parametrized cases (seed, run_id, int fields, float fields, weights, containers, coaching, probe_prompts, parent_dir, replay-without-parent, shape/prep objects), each asserting `ValueError` with the field name in the message.
- `test_from_json_rejects_malformed_documents_with_a_value_error`: 29 parametrized documents (the dict-pair and `"ab"` coaching cases, null containers, null / short / extra-key shape, null / unknown-key / bad-enum prep, ...).
- `test_from_json_then_validate_catches_what_the_structure_check_lets_through`: end-to-end `from_json` then `validate()` for float/str `token_budget`, negative seed, traversal run_id, replay without parent, non-bool dedup, string n_layer, string weight.
- `test_from_json_accepts_nulls_only_where_allowed_and_defaults_when_absent`, `test_validate_accepts_replay_with_a_parent_and_int_valued_numbers` (valid inputs still pass, including boundary run_id lengths and int-valued numbers), `test_from_json_malformed_json_is_a_value_error`.

## Issue 2: `event_to_dict` must emit strict JSON
`events.py`: `event_to_dict` now runs the dict through `_json_safe`, which maps every non-finite float to `None`, recursing into dicts, lists and tuples (so `losses` and `summary`, including nested containers). The event object itself is not mutated. Tests: `test_event_to_dict_maps_non_finite_floats_to_null_so_the_json_is_strict` (NaN/inf in `Progress.loss`, `Progress.lr`, `HeldoutEval.losses`, `Done.summary` incl. nested list/dict, `Instability.lr_scale`; asserts `json.dumps(..., allow_nan=False)` succeeds and the exact `None` placement) and `test_event_to_dict_does_not_mutate_the_event`.

## Minor: resume test no longer vacuous
`test_spike_detector_state_roundtrip_resumes_exactly` now injects a real spike (30.0) at step 200 and a NaN at step 250, asserts the uninterrupted run flags exactly `[200, 250]`, and asserts the resumed detector (saved at step 150 through JSON) reproduces those flags and the final state.

## RED / GREEN
RED: I restored the previous `config.py` and `events.py` from commit `256ae5b` and ran the new test file against them: `86 failed, 52 passed` (29 from-JSON documents, 55 validate cases, the strict-JSON event test and the end-to-end test failed, e.g. `TypeError` / `AttributeError` / `KeyError` escaping, or no error at all for float `token_budget`, dict coaching pairs, `"../../x"` run ids). The fixed files were then put back (diff-verified by re-running tests below).

GREEN:
```
python -m uv run pytest tests/test_train_config.py -q -W error  ->  138 passed in 0.26s
python -m uv run ruff check .                                    ->  All checks passed!
python -m uv run ruff format --check src/airace_ml/train tests/test_train_config.py -> 6 files already formatted
python -m uv run pytest                                          ->  267 passed, 2 deselected in 5.31s
```
I also ran a scratch probe replaying every input listed in the review (dict/"ab" coaching, float and string `token_budget`, string `replay`/`boldness`/weight, null `coaching`/`notebook`/`purchases`/`mixture`/`prep`/`shape`, shape missing key, unknown prep key, traversal and empty `run_id`, negative seed, replay without parent) through `from_json(...).validate()`: all 19 raise `ValueError` naming the field, none accepted, none with another exception type.

## Files changed in this round
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\config.py`
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\train\events.py`
- `C:\Users\petko\Desktop\AI Race\ml\tests\test_train_config.py`

## Notes for later tasks
- Contract change from the ruling: `validate()` now requires `parent_dir` when `replay > 0`, so earlier concern 3 is resolved. Concern 1 (non-finite floats in events) is resolved inside `event_to_dict`; a `Progress` with `loss: null` now means "non-finite loss", which Task 8 / the TS consumer should treat as such.
- `from_json` is strict about numeric types by design: `token_budget: 32768.0` is rejected (JSON written by `JSON.stringify` from an integer is `32768`, so this only affects a producer that writes floats).
