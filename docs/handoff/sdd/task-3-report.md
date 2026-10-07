# Task 3 report: model shape, transformer, checkpoint

Status: DONE (with minor notes under Concerns)
Commit: `69c03d4 feat(ml): decoder-only transformer with RoPE, SwiGLU, KV cache, checkpoints`

## What I implemented
All under `ml/src/airace_ml/model/`:

- `__init__.py`: docstring only.
- `shape.py`: `HEAD_DIM=32`, `MAX_PARAMS=85_000_000`, `ShapeError(ValueError)`, frozen `ModelShape(n_layer, d_model, ctx_len)`.
  - `n_head = d_model // 32`.
  - `ffn_hidden = 64 * ceil(8*d_model/3/64)`, computed in exact integer arithmetic (no float rounding risk).
  - `param_count(vocab_size=VOCAB_SIZE)`: embedding (tied head counted once) + per block (4d^2 + 3*d*ffn + 2d) + final norm d.
  - `validate(vocab_size)`: n_layer in [1,24]; d_model a multiple of 32 in [64,1024]; ctx_len a multiple of 64 in [64,1024]; integer types; `param_count(vocab) <= MAX_PARAMS`.
  - `to_dict` / `from_dict`.
- `transformer.py`:
  - `RMSNorm(dim, eps=1e-5)`: parameter `weight`, persistent float32 scalar buffer `eps` (it is in the state dict, so it survives save/load). The math is done in fp32.
  - `KVCache` dataclass (`k`, `v` lists of per-layer `[B,H,T,32]`, `length`).
  - `Attention` (`wq/wk/wv/wo`), `SwiGLU` (`w_gate/w_up/w_down`), `Block` (`attn_norm, attn, mlp_norm, mlp`), `Transformer` (`tok_emb`, `blocks`, `norm_f`, `lm_head` tied to `tok_emb.weight`). All linears are bias-free, matching the growth contract.
  - Pre-norm blocks, RoPE (base 10000) applied to q and k from explicit `positions`, SwiGLU, SDPA attention.
  - Init: normal std 0.02; `wo` and `w_down` use 0.02/sqrt(2*n_layer).
  - `Transformer.__init__` calls `shape.validate(vocab_size)`, so an invalid shape cannot be built.
  - `forward(idx, *, positions=None, key_padding_mask=None, kv_cache=None)`.
    - Default positions are `cache.length + arange(T)`.
    - `key_padding_mask` is `[B, T_total]` (cached + new keys), True = real.
    - The cache is updated in place and `cache.length` is advanced once at the end of the forward pass.
    - Shape mismatches of `positions` and `key_padding_mask` raise `ValueError`.
  - Mask logic (`_attention_mask`):
    - No mask and no past: `is_causal=True`.
    - No mask, a past, and T==1: no mask needed.
    - No mask, a past, and T>1: causal mask with the cache offset.
    - Key padding given: boolean mask `(causal & key_padding) | self-diagonal`.
    - The self-diagonal term means a fully padded left-pad query row still has one valid key, so it cannot produce NaN. Real queries are unaffected, because their own key is already allowed. Padded keys stay masked for real queries, so the garbage in padded rows never leaks into real positions.
  - `new_kv_cache()`, `num_params()` (unique parameters via `parameters()`).
- `checkpoint.py`: `CheckpointMeta` dataclass (fields as specified, `format=1`) with `to_json` / `from_json` (rejects an unknown `format`). `save_checkpoint` writes `model.safetensors` (via `safetensors.torch.save_model`) and `meta.json` (utf-8). `load_checkpoint(ckpt_dir, device="cpu")` builds the model from `meta.shape` and the vocab size read from the weights file, then calls `load_model` and `.to(device)`.
- `ml/tests/test_model.py`: the brief's 6 tests (11 cases with the parametrize) plus 5 extra contract tests (below).

## TDD evidence
RED (before any implementation):
```
$ cd ml && python -m uv run pytest tests/test_model.py -v
ERROR collecting tests/test_model.py
tests\test_model.py:2: in <module>
    from airace_ml.model.shape import ModelShape, ShapeError
E   ModuleNotFoundError: No module named 'airace_ml.model'
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 3.66s ===============================
```
Expected: the `model` package did not exist yet.

GREEN (brief tests only, right after implementing):
```
$ python -m uv run pytest tests/test_model.py -v
... 11 passed in 5.71s
```
GREEN (final, with the extra tests, full suite):
```
$ python -m uv run pytest
tests\test_model.py ................   tests\test_paths_device.py ....   tests\test_tokenizer.py ..................
38 passed in 5.94s
$ python -m uv run ruff check .
All checks passed!
```
No warnings in the test output.

## Extra checks beyond the brief (also kept as tests)
- Shape derived values (`n_head`, `ffn_hidden` 384 and 704) and the `to_dict`/`from_dict` round trip.
- Block layout contract for growth: all linears are bias-free; norm weights are `[d_model]`.
- `eps` buffers are in the state dict, and a non-default eps (3e-4 on one norm) survives a checkpoint round trip while the others stay at 1e-5.
- Cached batched generation with left padding: prefill a 2-row left-padded batch with `positions` + `key_padding_mask` + cache, then one decode step with the mask extended to `[B, T_total]`. The outputs are finite (no NaN from pad rows) and each row matches the unpadded single-row full forward to 1e-5.
- Prefill continuation with a cache (T>1 after a cache) matches the full forward; a wrong-shaped `key_padding_mask` raises `ValueError`.

Ad hoc, not in the test suite (scratch script): on CUDA, `load_checkpoint(..., "cuda")` keeps `lm_head.weight is tok_emb.weight`, moves the `eps` buffers to the GPU, and the output matches CPU to about 3.6e-7.

## Files changed
- `ml/src/airace_ml/model/__init__.py`, `shape.py`, `transformer.py`, `checkpoint.py` (new)
- `ml/tests/test_model.py` (new)

## Self-review findings
- I compared the diff with the brief's interface list; every name, signature and default matches.
- `ruff check` initially flagged the brief's test file only for import order (I001). I applied `ruff check --fix`. No assertion was changed. I did NOT run `ruff format` on the test file, since `ruff check` passes without it and that keeps the brief's tests verbatim. The source files are `ruff format`-clean.
- The reference counts (1,377,408 and 7,475,456) match the brief.
- `save_model` keeps the tied weight under `lm_head.weight` and drops `tok_emb.weight` (the file metadata says `{'tok_emb.weight': 'lm_head.weight'}`). `_vocab_size` therefore checks both names, and this is covered by the round-trip test with vocab 512.
- My first multi-file bash heredoc failed to parse and wrote nothing. I recreated the files with the Write tool. One small Python rewrite of `checkpoint.py` briefly produced CRLF; I normalized it to LF, and the committed blob is LF.

## Concerns / notes for later tasks
1. **Context-length enforcement is not in the model.** RoPE has no hard limit, so `forward` does not raise when positions or `T_total` exceed `shape.ctx_len`. The brief is silent here, and left-padded batches can legitimately have `T_total > ctx_len` with real positions below it. Task 5 owns the truncation/no-position-overflow behaviour (Review Focus 2).
2. **`Transformer.__init__` validates the shape against `vocab_size`.** Any later test or tool that builds a model must therefore use a legal shape (d_model >= 64, ctx_len a multiple of 64, and so on). The brief's tests all do.
3. **Brief deviation: `is_causal` fast path.** The cache-with-T==1 and no-mask case passes no mask at all (a single query sees every cached key). This is mathematically identical to the specified explicit mask. The pure-causal fast path is taken only when there is no mask and no cached past.
4. **Cache lists are filled lazily.** `KVCache.k`/`v` start empty and each layer appends on its first call, because B is unknown at `new_kv_cache()` time. Task 5 should create one cache per batch and not reuse it across batches.
5. **Git status oddity (cosmetic).** After normalizing line endings, `git status` briefly showed `checkpoint.py` as modified with no content diff. I re-staged it, the tree is clean, and the commit is correct.

---

## Fix round 1

Finding (Important): `save_checkpoint` did not verify `meta.shape == model.shape`. `ctx_len` has no footprint in the weights, so a stale or ctx_len-only-grown meta saved and loaded silently with the wrong `ctx_len`.

Change (`ml/src/airace_ml/model/checkpoint.py`): `save_checkpoint` now raises `ValueError("meta.shape ... does not match model.shape ...")` as its first statement, before `mkdir` or any file write, when `model.shape != meta.shape`.

Test added (`ml/tests/test_model.py::test_save_checkpoint_rejects_mismatched_meta_shape`): a `ModelShape(2,64,64)` model with `meta.shape = ModelShape(2,64,128)` raises `ValueError`. Into an existing directory, nothing is written (directory stays empty). Into a non-existent directory, the directory is not created.

RED (test written first, check not yet added):
```
$ python -m uv run pytest tests/test_model.py -q -k mismatched
E       Failed: DID NOT RAISE ValueError
FAILED tests/test_model.py::test_save_checkpoint_rejects_mismatched_meta_shape
1 failed, 16 deselected
```
GREEN (after the fix):
```
$ python -m uv run pytest tests/test_model.py -v
... 17 passed in 5.82s
$ python -m uv run pytest -q
39 passed in 5.95s
$ python -m uv run ruff check .
All checks passed!
```
