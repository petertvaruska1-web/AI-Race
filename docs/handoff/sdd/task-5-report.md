# Task 5 report: inference

**Status:** DONE (one note under Concerns)
**Commit:** `6e5546c feat(ml): batched KV-cached generation, continuation scoring, chat prompts`

## What I implemented

`ml/src/airace_ml/infer/lm.py` (+ empty-docstring `infer/__init__.py`):

- `ContinuationScore(sum_logprob, n_tokens)` and `Generation(tokens, token_probs, top1_probs, stopped)` dataclasses.
- `LanguageModel` Protocol (`ctx_len`, `score_continuations`, `generate`), signatures exactly as in the brief.
- `build_chat_prompt(tok, history, ctx_len, reserve)`:
  - Format is `encode_chat(..., add_generation_prompt=True)`. Each turn is encoded with `encode_chat(tok, [turn])[1:]`, so the role mapping and validation stay in one place.
  - Turns are taken newest-first and encoded lazily, so a very long history never pays to tokenize the dropped turns.
  - If the newest turn alone is too long, it keeps `[role, *tail(text), <|end|>]`, so the prompt still starts with `bos` and ends with `ai`.
  - `ValueError` if `reserve < 0` or `ctx_len - reserve < 4`.
- `TorchLM(model, tok, device=None, batch_size=32)` with public `model`, `tok`, `device`, `batch_size`, and a `ctx_len` property (reads `model.shape.ctx_len`).
  - `device=None` means `pick_device()`. The model is moved there with `.to()` and put in `.eval()`.
  - **Generation:**
    - Prompts are left-truncated to `ctx_len - 1`, keeping a leading `bos`.
    - Rows are left-padded with `positions = cumsum(mask) - 1` clamped at 0.
    - One prefill, then KV-cached decode. Each step feeds per-row positions `len_i + step` and extends `key_padding_mask` by one True column.
    - One fresh cache per chunk, never reused.
    - Chunks of `batch_size` share one `torch.Generator(device)` seeded per call.
    - Greedy at `temperature == 0`, otherwise nucleus sampling over `softmax(logits / T)`. The top token always survives top-p.
    - A row stops at a stop id (default `(end, bos)`; `[]` means never) or when prompt plus reply reach `ctx_len`. The stop token is excluded from `tokens`, `token_probs` and `top1_probs`.
    - Finished rows keep decoding with positions clamped to `ctx_len - 1`, and their output is ignored. The loop breaks as soon as every row is done.
    - `top1_probs` and `token_probs` come from the raw `softmax(logits)` at T=1 before top-p. `token_probs` is clamped to float32 `tiny` so it stays in (0, 1] even if the softmax underflows.
  - **Scoring:** right-padded chunks through the causal model, with no mask needed. Log-softmax is taken only over the continuation positions. An empty continuation gives `(0.0, 0)` without running the model.
  - **Runtime:** everything runs under `torch.inference_mode()`, with bf16 autocast on CUDA only.
  - `chat_reply` and `complete` are thin wrappers over `generate` and `decode(skip_special=True)`. `complete` returns only the new text.

`ml/tests/conftest.py`: added the `tiny_lm` fixture, per the brief (function-scoped, `torch.manual_seed(0)`, CPU).

## Tests and results

`ml/tests/test_infer.py` has the 5 brief tests plus 26 CPU tests and 1 `gpu`-marked test.

- Focused: `python -m uv run pytest tests/test_infer.py -q` gives 31 passed, 1 deselected (the gpu test).
- GPU: `python -m uv run pytest tests/test_infer.py -m gpu -v` gives 1 passed. The test is `test_cuda_bf16_path_tracks_the_cpu_path`. Greedy tokens equal CPU, seeded sampling is reproducible, probabilities are in (0, 1], and scores are within 5% of CPU.
- Full suite: `python -m uv run pytest` gives 79 passed, 1 deselected in about 3.7s. `-W error` finds 0 warnings.
- Lint: `python -m uv run ruff check .` passes. `ruff format --check src` passes, so all of `src` is formatted.

### TDD evidence
- RED: `python -m uv run pytest tests/test_infer.py -q` failed with this output. It is expected because `airace_ml.infer` did not exist yet, and the new `tiny_lm` conftest fixture imports `TorchLM`:
  ```
  ImportError while loading conftest '...\ml\tests\conftest.py'.
  tests\conftest.py:7: in <module>
      from airace_ml.infer.lm import TorchLM
  E   ModuleNotFoundError: No module named 'airace_ml.infer'
  ```
- GREEN: after implementing, `python -m uv run pytest tests/test_infer.py -v` showed 30 passed, then 31 after the sharp-model test below was added.
- **Mutation check.** All 30 tests passed on the first run, so I tried to break `lm.py` in 15 targeted ways and confirmed the tests catch each one. Covered mutations: padding positions, key mask in prefill and decode, mask column, position off by one, position clamp, stop token kept, top-p dropping the top token, top1 after temperature, length cap, bos kept on truncation, scoring off by one, chat dropping the newest turn, seeding, and chunk dropping rows.
  - The first round let two survive (padding positions, key-padding mask). An untrained model has near-uniform attention, and RoPE is relative, so context barely matters there.
  - I added a `sharp_lm` fixture (weights x20, so attention is peaked) and `test_ragged_batch_matches_naive_when_attention_is_sharp`. It compares tokens and `top1_probs` to naive decoding for a ragged batch.
  - After that, all 15 mutations are caught, and `lm.py` was restored byte-identical (`cmp`).

## Files changed
- `ml/src/airace_ml/infer/__init__.py` (new)
- `ml/src/airace_ml/infer/lm.py` (new, 376 lines)
- `ml/tests/conftest.py` (imports + `tiny_lm` fixture)
- `ml/tests/test_infer.py` (new)

## Self-review
- **Requirements:** every brief item is covered, and the brief's tests are verbatim apart from the note below.
- **Review Focus 2** (long history): `build_chat_prompt` fits the budget, starts with `bos`, ends with `ai`, and keeps the newest question. `test_length_cap_and_no_position_overflow` asserts that no position fed to the model exceeds `ctx_len - 1`. A default `chat_reply` (`max_new_tokens=96`) on a 64-token model works, because the reserve is capped at `ctx_len // 2`.
- **YAGNI:** I did not drop finished rows from the batch, the simpler option the rulings allowed. No length-sorting of batches.
- **Source style:** `lm.py` is `ruff format`ted. Tests keep the brief's compact style, like `test_model.py`.

## Concerns
1. **One edit to a brief test.** `test_score_continuations_matches_manual` raised a `UserWarning` ("Converting a tensor with requires_grad=True to a scalar"), because the brief calls `float()` on a graph tensor. I wrapped only the manual `logp` computation in `with torch.no_grad():`. The assertion is unchanged.
2. **Choices the brief left open** (tell me if you disagree):
   - `device=None` uses `pick_device()` and moves the model in place.
   - An empty prompt or context raises `ValueError`.
   - `score_continuations` left-truncates over-long contexts. A continuation too long to leave even one context token is cut, and `n_tokens` reports how many tokens were scored.
   - `complete()` returns only the generated continuation.
   - `chat_reply` reserves `min(max_new_tokens, ctx_len // 2)`.
3. **Prefill memory:** `Transformer.forward` returns logits for every position, so prefill builds `[B, W, V]`. At batch 32, ctx 1024 and vocab 4096, that is about 512 MB in fp32 (about 256 MB in bf16). A last-token-only option on the model would fix it. It is outside this task's files, so I left it.
4. **Seeded sampling depends on batch composition.** Each call draws from one generator, so a prompt's samples depend on the other prompts in the call. Reproducibility is per call, as the brief says.
5. **Numerics:** on CUDA the bf16 autocast gives scores about 0.5% off CPU, and greedy tokens could flip on near-ties.
6. **For Task 19:** decoded text from an untrained byte-level model can contain U+FFFD. Printing it raised `UnicodeEncodeError` on the cp1252 console in my smoke script.

---

# Fix report, round 1

**Finding (Important):** `TorchLM` mutated the caller's model state. It set eval mode once in `__init__`, never at call time. With `device=None` it moved the model in place to `pick_device()`.

## What changed
`ml/src/airace_ml/infer/lm.py`:
1. **Eval mode at call time.**
   - New private context manager `TorchLM._eval_mode()` records `was_training`, calls `model.eval()`, and restores `model.train(was_training)` in a `finally`.
   - `generate` and `score_continuations` run their model work inside it, after argument validation. `chat_reply` and `complete` go through `generate`, so they are covered.
   - `__init__` no longer calls `.eval()`, so building a `TorchLM` leaves the model's mode alone.
2. **Device.**
   - `device=None` now uses the model's own device (`next(model.parameters()).device`), and nothing is moved. Only an explicit `device` calls `model.to(device)`.
   - The `pick_device` import is gone.
3. The class docstring now describes this contract.

`ml/tests/test_infer.py`:
- Rewrote two of my own earlier tests that encoded the old behavior. These were not brief tests.
  - `test_lm_exposes_...` no longer asserts `not model.training`.
  - `test_default_device_follows_pick_device_...` became `test_default_device_is_the_models_own_and_nothing_is_taken_over`. It deliberately sets no `AIRACE_DEVICE`, so on this CUDA machine the old behavior would pick CUDA. It checks:
    - a CPU model stays on CPU;
    - construction leaves train and eval mode alone, both ways;
    - `batch_size` is 32.
- Added `test_calls_run_in_eval_mode_and_restore_the_callers_mode` (parametrized over starting in train and eval mode):
  - spies on `forward` to prove every forward in `generate` and `score_continuations` ran in eval mode;
  - checks the caller's mode is restored, including after `chat_reply` and `complete`;
  - checks the outputs equal those produced in eval mode.
- Added `test_mode_is_restored_even_when_a_call_fails` (parametrized the same way): a failing forward still restores the mode, for both `generate` and `score_continuations`.
- Added a `gpu`-marked `test_cuda_device_is_inherited_or_moved_only_when_explicit`. An explicit `cuda` moves a CPU model, and no device inherits the model's CUDA placement.

## TDD evidence
- RED: new tests run against the committed round-0 `lm.py` (restored from HEAD):
  `python -m uv run pytest tests/test_infer.py -q` gave `2 failed, 33 passed, 2 deselected`.
  - `test_default_device_is_the_models_own_and_nothing_is_taken_over` failed with `assert device(type='cuda') == device(type='cpu')`. This is the review's bug reproduced: on this machine a CPU model was moved to CUDA.
  - `test_calls_run_in_eval_mode_and_restore_the_callers_mode[True]` failed with `assert ([True, True, ...] and not True)`: forwards ran in train mode when the caller had train mode.
  - The other new tests pass on the old code, because old code never touched mode at call time. They are guarded by the mutation check below.
- GREEN:
  - `python -m uv run pytest tests/test_infer.py -q` gave `35 passed, 2 deselected`.
  - `python -m uv run pytest tests/test_infer.py -m gpu -q` gave `2 passed, 35 deselected`.
  - Full suite `python -m uv run pytest -W error` gave `83 passed, 2 deselected` with 0 warnings.
  - `ruff check .` is clean, and `ruff format --check src` is clean.
- Mutation check on the fix: 8 mutations (restore removed, restore outside `finally`, eval never set, always restore to eval, `device=None` moving the model, explicit device not moving it, `generate` unwrapped, `score_continuations` unwrapped). All 8 were caught, and `lm.py` was restored byte-identical (`cmp`).

## Notes
- A trainer can now build a `TorchLM` on a model that is mid-training, or probe it between steps, and the model comes back in the same train or eval mode it had.
- The model has no dropout or batch-norm today, so the mode only matters for future layers. Hence the spy-based test rather than an output comparison.
- `model.train(was_training)` is applied recursively. If a caller had deliberately mixed modes across submodules, the restore would flatten them. Nothing in this codebase does that.
