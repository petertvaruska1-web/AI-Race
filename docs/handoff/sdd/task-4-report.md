# Task 4 report: function-preserving growth

**Status:** DONE

**Commit:** `17207f2 feat(ml): exact function-preserving depth/width growth`

## What was implemented

`ml/src/airace_ml/model/growth.py`:

- `GrowthError(ValueError)`.
- `insertion_layout(n_old, n_new) -> list[int | None]`.
  - Fresh block `j` of `k = n_new - n_old` goes after the first `floor(j*n_old/k + 0.5)` old blocks.
  - Computed in exact integer arithmetic as `(2*j*n_old + k) // (2*k)`.
  - `k = 0` yields the identity layout without dividing.
  - A shrinking request raises `GrowthError`.
- `grow(model, target, *, seed) -> Transformer`:
  - Raises `GrowthError` if `target.n_layer < old` or `target.d_model < old`. `ctx_len` is deliberately not checked: spec §4.3 says context length may go up or down freely.
  - Builds a **fresh** `Transformer(target, vocab_size)` on the CPU inside `torch.random.fork_rng(devices=[])` with `torch.manual_seed(seed)`. The new parameters therefore depend on the seed alone, whatever the device, and the global RNG state is left untouched. The model is then moved to the input model's device and dtype.
  - `vocab_size` comes from `model.tok_emb.num_embeddings`.
  - The embedding is zeroed and the old one copied into its leading slice. The head is tied, so it follows.
  - Per copied block:
    - `wq/wk/wv` and `w_gate/w_up` get the old weights in the leading slice, and the new rows and columns keep their random init.
    - `wo` and `w_down` are zeroed, then the old weights are copied into the leading slice. This zeroes the new rows (new residual dims), the new columns (new heads, new hidden units) and everything else new.
    - Both norms and `norm_f`: `weight` is filled with 1.0, then `old * sqrt(d/d')` goes into `[:d]`. `eps = old_eps * d/d'`, computed in float64 and cast back to the old buffer dtype.
  - Fresh blocks: `wo` and `w_down` are zeroed entirely, everything else keeps its random init.
  - The result keeps the input model's train/eval flag.

`ml/tests/test_growth.py`: the brief's 4 tests, verbatim assertions, plus 2 extra tests for the clarification points (see below).

## Tests and results

Full suite: `cd ml && python -m uv run pytest`
```
collected 48 items
tests\test_growth.py .........   tests\test_model.py .................
tests\test_paths_device.py ....   tests\test_tokenizer.py ..................
============================= 48 passed in 6.47s =============================
```
No warnings. `python -m uv run ruff check .` gives `All checks passed!`, and `ruff format --check` is clean on both new files.

Extra verification (scratch script, not committed). Max logit diff, grown vs. original, with perturbed weights, fp32 CPU:

| Case | Max logit diff |
|---|---|
| (2,64,64) to (2,128,64) | 3.3e-6 |
| (2,64,64) to (4,64,64) | 0.0 |
| (2,64,64) to (5,96,128) | 3.5e-6 |
| (3,64,64) to (7,192,64) | 4.1e-6 |
| (1,64,64) to (6,64,64) | 0.0 |
| (1,64,64) to (1,64,64) | 0.0 |
| (2,64,64) to (3,96,64), on CUDA | 2.3e-6 |

The tolerance is 1e-4. Also confirmed: `lm_head.weight is tok_emb.weight` on the grown model, including on CUDA and in bf16.

Mutation check: dropping the `eps` rescale gives a 4.7e-3 error, and dropping the gain rescale gives 4.06. The brief's 1e-4 tolerance catches both, so the function-preservation test is not vacuous.

## TDD evidence

RED: `cd ml && python -m uv run pytest tests/test_growth.py -v`
```
ERROR tests/test_growth.py
E   ModuleNotFoundError: No module named 'airace_ml.model.growth'
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!
```
This was expected: the module did not exist yet.

GREEN: same command after implementing.
```
tests/test_growth.py::test_growth_preserves_function[src0-dst0] PASSED
tests/test_growth.py::test_growth_preserves_function[src1-dst1] PASSED
tests/test_growth.py::test_growth_preserves_function[src2-dst2] PASSED
tests/test_growth.py::test_growth_preserves_function[src3-dst3] PASSED
tests/test_growth.py::test_growth_rejects_shrinking PASSED
tests/test_growth.py::test_insertion_layout PASSED
tests/test_growth.py::test_new_parameters_receive_gradient PASSED
============================== 7 passed in 5.72s ==============================
```
The 2 extra tests were added afterwards, and the suite is now 9 passed in `test_growth.py`.

## Ruff adjustments to the brief's test code

The brief's test code was reformatted with `ruff format`, and `ruff check --fix` split `import pytest, torch` into two imports and sorted them (I001). One non-formatting lint, RUF015, was a pure rewrite: `[... if v is None][0]` became `next(... if v is None)` in `test_new_parameters_receive_gradient`. No assertion was changed or weakened.

## Extra tests (beyond the brief)

1. `test_seed_controls_new_parameters_and_leaves_global_rng_alone`: same seed gives identical parameters, a different seed gives different new weights, and the global RNG is not advanced.
2. `test_grown_model_keeps_dtype_tying_and_mode`: a bf16 model grows into a bf16 model, the `eps` buffer dtype is kept, the head stays tied, and the train/eval flag is kept.

## Files changed

- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\model\growth.py` (new)
- `C:\Users\petko\Desktop\AI Race\ml\tests\test_growth.py` (new)

## Self-review

- The new heads, new FFN units and new layers have zeroed writers, and their input-side weights (`wq/wk/wv` columns, `w_gate/w_up` columns) stay random because they read exact zeros.
- The new dims stay exactly zero through the whole forward pass. New embedding columns are 0, the zero-padded input is normalised to 0 (the new gain slots are 1.0, but 0 * 1.0 = 0), and `wo` and `w_down` write zeros into the new rows.
- Gradients reach every new parameter group, as the brief's gradient test asserts. The new embedding columns get gradient through the random `wq/wk/wv` columns of the old blocks, via the new-dim path of the norm.
- No modules are swapped in place. `n_head` is always recomputed by the fresh `Attention` constructor.
- Nothing speculative was added. The only extra surface is the `insertion_layout` shrink guard and the dtype and mode handling the clarification asked for.

## Concerns

- `insertion_layout` follows the brief's formula literally. When `k` is much larger than `n_old`, the rounding can put a fresh block before old block 0. For example, `insertion_layout(1, 6)` gives `[None, None, 0, None, None, None]`. Function preservation still holds, because fresh blocks are identities. The brief does not forbid this, so I left it.
- Working-tree files are CRLF, consistent with the existing files. `.gitattributes` (`* text=auto eol=lf`) normalises them to LF in the index. Git printed a harmless CRLF-to-LF warning.
