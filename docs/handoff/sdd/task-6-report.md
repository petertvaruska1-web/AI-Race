# Task 6 report: corpus format, preparation filters, mixture sampler

Status: DONE. Commit `3d827c5 feat(ml): corpus shards with tags, preparation filters, mixture sampler`.

## What was implemented

- `ml/src/airace_ml/data/corpus.py`
  - Constants `DATASET_IDS`, `CUSTOM_DATASET_IDS`, `TOPICS`, `NOISE_KINDS` (verbatim from the brief).
  - `DocTags`, with every field coerced to its canonical dtype and length-checked on write and on open.
  - `write_corpus` streams uint16 tokens to `tokens.bin`, then writes `offsets.npy` (int64, N+1), `tags.npz` and `info.json`.
    - It rejects token ids outside the uint16 range and tag arrays whose length is not N. The error names the offending tag.
  - `Corpus.open` memory-maps the tokens and checks the file size against the offsets.
    - An empty corpus is handled, because `np.memmap` cannot map an empty file.
    - `doc(i)` raises `IndexError` out of range (including negative `i`) and returns a uint16 memmap view.
    - I added a public `offsets` attribute so the pool can compute document lengths without touching the memmap.
- `ml/src/airace_ml/data/prep.py`
  - `CLEANING_THRESHOLDS`.
  - `PrepConfig` (frozen). `__post_init__` raises `ValueError` for an unknown `cleaning` or `variety`. `to_dict` / `from_dict` are provided.
  - `eligible_docs`: `quality >= threshold`, not held out, optional dedup (unique or canonical), optional fact-check, and `purchase_rank < purchase_fraction`.
    - The threshold is cast to float16 before comparing. Without that, a document stored at exactly 0.40 would fail `>= 0.40`, because float16(0.4) = 0.39990.
    - A test covers this.
  - `heldout_docs`.
- `ml/src/airace_ml/data/sampler.py`
  - `MixtureError`.
  - `DocPool`: `from_corpus` (lazy, keeps ids and fetches from the memmap; drops empty documents; balanced pools group positions by the topics present) and `from_docs` (drops empty documents, unbalanced).
    - `sample_doc` draws a topic uniformly among those present, then a document, when balanced.
    - Sampling an empty pool raises `MixtureError`.
  - `MixtureSampler`:
    - Validation raises `MixtureError` naming the dataset or "sum" for: an unknown key, a negative or non-finite weight, a sum that is not positive and finite, and a positive-weight pool with no non-empty documents.
    - `set_weights` validates everything before mutating, so a failed switch keeps the previous mixture.
    - A row is built from one dataset chosen by weight. A document longer than seq_len+1 contributes a random window. Otherwise documents from the same pool are appended and the row is truncated.
    - `next_batch` returns contiguous int64 `(B, seq_len)` inputs and targets.
    - `stats()` counts `seq_len` tokens per row and lists every pool, at 0 if undrawn.
    - `rng_state()` is `{"bit_generator": <PCG64 state>, "stats": {...}}`. It is JSON-safe and tested through a JSON round trip. `set_rng_state` restores both.
  - `load_pools` opens only positive-weight datasets.
    - It first validates ids and weights for all keys. An unknown id, or a negative or NaN weight, raises `MixtureError` naming it, rather than the "unknown dataset" error the sampler would give for a skipped key.
    - Custom ids come from `custom[...]` via `from_docs`. A positive weight on a custom id missing from `custom` raises `MixtureError` naming it.
- `ml/tests/conftest.py`
  - Added `tiny_data_root` (session fixture). Layout: the tiny tokenizer JSON at `tokenizer_path(root)` and all 8 corpora under `corpus_dir(root)`.
    - Each corpus has 50 documents cycled and rotated (offset 7 x dataset index) from its source fixture files. Sources: web = facts+unicode+stories, books = stories, educational = facts, conversations = chats, code = code, reasoning = code+facts, facts = facts, creative = stories+unicode.
    - `conversations` are parsed from the `User:` / `AI:` lines into turns and encoded with `encode_chat`. Unprefixed continuation lines, such as the multi-line poem reply, are folded into the previous turn.
    - Tags use a fixed seed (1000 + dataset index):
      - stratified qualities (one per 1/50 band, so min < 0.15 and max > 0.65 hold);
      - 3 dup clusters of 3 (members repeat the canonical's tokens verbatim);
      - 4 `false_fact`;
      - 4 topics;
      - 5 held out (10%);
      - `purchase_rank` from a permutation / n;
      - `noise_kind` consistent with the other tags.
    - Setup is about 0.1 s (excluding tokenizer training), and the fixture does not depend on test order.
  - Added `tiny_tok_path`, and refactored `tiny_tok` to `Tok.load(tiny_tok_path)`.
    - `Tok` has no public `save`, and I did not want to reach into `_tk` or retrain a possibly different tokenizer.
    - Behaviour of `tiny_tok` is the same (session scope, same vocab 512). The full suite is green.
  - Extracted `_file_paragraphs`, which `fixture_texts` now uses. Its output is unchanged.

## Tests

Command: `cd ml && python -m uv run pytest -q` gives **125 passed, 2 deselected in 4.3s**. It also passes with `-W error`, so there are no warnings.
`python -m uv run ruff check .` gives "All checks passed!".

`test_corpus.py` has 14 tests and `test_sampler.py` has 28 (including parametrized cases). Together they run in about 0.5 s.

- The brief's tests are verbatim: `test_roundtrip`, `test_filters`, `test_dedup_factcheck_heldout`, `test_purchase_nested`, `test_proportions_and_shapes`, `test_invalid_mixtures` (5 params), `test_zero_weight_empty_pool_is_fine`, `test_short_docs_pack`, `test_set_weights_and_rng_state`.
- `test_balanced_variety_equalizes_topics` uses a 90/10 topic corpus. Natural sampling gives a minor-topic share in (0.03, 0.2), and balanced gives [0.4, 0.6].
- Added tests:
  - `eligible_docs`: independence of the dedup and fact-check switches, the float16 threshold boundary, `purchase_fraction = 0`.
  - `heldout_docs` and `PrepConfig` validation and round trip.
  - Corpus edge cases: out-of-range `doc`, an empty corpus, `write_corpus` rejecting bad input, and tag dtypes and `info` round trip, including non-ASCII.
  - Sampler behaviour: same seed gives the same batches, contiguous random windows from long documents, packing never mixing datasets within a row (with the stats token total), a failed `set_weights` keeping the old mixture (negative, unknown, zero-sum and NaN weights), `seq_len` and `batch_size` validation, a JSON-safe `rng_state` that restores the stats, and an empty pool being unsampleable.
  - `load_pools`: real corpora plus a custom pool, prep and purchase applied, only positive-weight datasets opened, bad weights and ids named, and the unsampleable cases (purchase 0 and a blank notebook).
  - `tiny_data_root`: layout and tag spread (all of the brief's requirements), conversations being real chats, and all 8 datasets poolable under thorough + dedup + fact-check.
- Mutation sanity checks: ignoring `balanced` makes the two variety tests fail. Dropping the held-out exclusion makes `test_dedup_factcheck_heldout` fail.

## TDD evidence

RED: `python -m uv run pytest tests/test_corpus.py tests/test_sampler.py -q` (before any implementation).
Output: `ERROR tests/test_corpus.py` and `ERROR tests/test_sampler.py`, with `ModuleNotFoundError: No module named 'airace_ml.data'`, "2 errors during collection". Expected, because the package did not exist.

GREEN: after implementing, the same command gives `42 passed in 0.52s`. The full suite then gave `125 passed, 2 deselected`.

## Files changed

Created:
- `ml/src/airace_ml/data/__init__.py`
- `ml/src/airace_ml/data/corpus.py`
- `ml/src/airace_ml/data/prep.py`
- `ml/src/airace_ml/data/sampler.py`
- `ml/tests/test_corpus.py`
- `ml/tests/test_sampler.py`

Modified: `ml/tests/conftest.py`.

## Self-review and deviations

- Brief test code was not weakened. The only edit to it was in the `_tags` helper in `test_corpus.py`: `dict(...)` became a dict literal, because `ruff check` flags C408. No assertions changed. Other brief-style test code (semicolons, dense lines) passes `ruff check` and was left as written, matching the existing test files. The repo does not enforce `ruff format` (9 files fail `--check`, including pre-existing ones). I formatted only the new `src/airace_ml/data` files and `conftest.py`.
- `test_tiny_data_root_layout` compares `purchase_rank` in float32 (`np.sort(rank)` against `arange(n, float32) / n`), because float32 does not equal python-double `i / n`.
- Line endings: Python edits on Windows briefly produced CRLF in three files. The repo has `eol=lf` and the committed blobs are LF. I confirmed the working copy is LF too and the tree is clean.

## Concerns / decisions to confirm

1. **Missing `purchases` key means the full dataset (1.0)**, matching `eligible_docs`'s default `purchase_fraction = 1.0`. A dataset the player never bought is not excluded unless the caller passes 0.0. If Task 8 or the game layer wants "absent = nothing owned", this one line in `load_pools` changes (`purchases.get(name, 1.0)`).
2. `PrepConfig.from_dict` is strict, so an unknown key raises `TypeError`. A missing key falls back to the default.
3. `rng_state()` has the shape `{"bit_generator": ..., "stats": ...}`. The brief says "bit_generator.state plus the stats" without specifying nesting, so downstream checkpoint code should treat it as opaque.
4. A missing corpus directory for a positive-weight dataset raises `FileNotFoundError` from `Corpus.open`, not `MixtureError`.

---

# Fix report, round 1

Two Important review findings fixed. One deviation from the ruling's literal wording on finding 2 is flagged below.

## 1. `load_pools` output did not compose with `MixtureSampler` for zero-weight entries

Changes in `ml/src/airace_ml/data/sampler.py`, `MixtureSampler.set_weights` (the constructor goes through it):
- For every key it first checks the weight, so a negative or non-finite weight raises `MixtureError` naming the id even when no pool for it was loaded.
- A key with weight 0 and no pool is now ignored.
- A key with positive weight and no pool raises `MixtureError`: "dataset 'x' has weight but was not loaded (no pool)".
- A failed `set_weights` still leaves the previous mixture untouched.

New tests in `tests/test_sampler.py`:
- `test_load_pools_output_composes_with_zero_weight_entries` builds `load_pools(tiny_data_root, {"web": .7, "books": .3, "code": 0.0, "notebook": 0.0}, ...)` and passes the same weights to `MixtureSampler`. It draws batches, then `set_weights` with other zero-weight ids.
- `test_missing_pool_only_matters_with_positive_weight` covers:
  - zero weight and no pool is fine;
  - positive weight and no pool names the id with "not loaded";
  - negative, NaN and inf weights on an unloaded id raise naming it;
  - `set_weights` behaves the same.

## 2. Blank notebook entry escaped the empty-pool check

`DocPool.from_docs` now drops documents with no content token as well as empty ones. Before this, `encode_doc(tok, "")` = `[bos]` was kept and trained on as repeated `<|bos|>`.

Tests added:
- `test_blank_documents_are_dropped_from_custom_pools`:
  - A pool of only blank documents has length 0 and `MixtureSampler` raises `MixtureError` naming "notebook". The blank documents are `[]`, `encode_doc(tok, "")`, `encode_chat` with empty user and ai turns, and `[bos, end]`.
  - A pool with one real document plus blank ones keeps only the real document, and sampled batches contain no pad.
- `test_blank_notebook_through_load_pools_is_unsampleable` runs the same path through `load_pools`.

### Deviation from the ruling (please confirm)

The ruling said to drop documents made only of ids `< len(SPECIAL_TOKENS)` (16). That literal rule cannot be used, for two reasons:

- **Brief tests would fail.** The brief's own test helper `_pool` builds docs `[1, 2, ..., 10]` and `test_short_docs_pack` uses `[1, 9, 9]`; every id in them is below 16. I implemented the literal rule first. 10 of the brief's tests (`test_proportions_and_shapes`, `test_short_docs_pack`, `test_set_weights_and_rng_state`, `test_zero_weight_empty_pool_is_fine` and others) and `test_load_pools_real_corpora` failed with `MixtureError`. The brief's tests are binding and I did not touch them.
- **Whitespace is not a "special" token.** `encode_doc(tok, "   ")` is `[1, 282, 236]`, ordinary tokens (there is no normalizer), so no id-based rule can drop it. The test you described, `from_docs([encode_doc(tok, "   "), encode_doc(tok, "")])`, would therefore not raise. I used `""` instead. Whitespace-only entries must be stripped before encoding (Task 8, as you said).

What I implemented: a document is dropped when it has no content token, meaning every id is a framing token (`<|pad|> <|bos|> <|end|> <|user|> <|ai|>`, ids 0-4, derived as `SPECIAL_TOKENS.index("<|ai|>") + 1`).
- This catches the reproduced bug: `[bos]` from a blank `encode_doc`.
- It also catches blank chats: `encode_chat` with empty turns, or `[bos, end]`.
- It keeps every document in the brief's tests, because those contain `<|sep|>`, `<|sys|>` or `<|r*|>` ids (5 and above).
- Documents of only `<|sep|>`, `<|sys|>` or `<|r*|>` ids are still kept. Nothing produces those from blank text.

If you want the literal 16-id rule, the brief's `_pool` helper and `test_short_docs_pack` docs would need to change to ids >= 16. Say so and I will do that.

Not changed: `DocPool.from_corpus` only drops zero-length documents, not framing-only ones. The ruling named `from_docs` only, and scanning every corpus document would be costly. Corpus documents come from the content pipeline, which should not emit them.

## Test evidence

RED (new tests against the previous `sampler.py` from HEAD): `python -m uv run pytest tests/test_sampler.py -q` gives 4 failed, 28 passed. The four failures are the four new tests above, as expected.

GREEN:
- `python -m uv run pytest tests/test_sampler.py tests/test_corpus.py -q` gives 46 passed.
- Full suite `python -m uv run pytest -q -W error` gives 129 passed, 2 deselected.
- `python -m uv run ruff check .` passes. `ruff check --fix` fixed one import-order error in `sampler.py`.
- `ruff format --check` is clean on the new `data/` files and `conftest.py`.

Files changed: `ml/src/airace_ml/data/sampler.py`, `ml/tests/test_sampler.py`.

