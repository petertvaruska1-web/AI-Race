## Global Constraints

**Toolchain**
- Python `>=3.13,<3.14` via uv. Pins: `torch==2.14.1`, `numpy>=2.5`, `tokenizers>=0.23`, `safetensors>=0.8`. Content extra: `datasets>=5.1`. Dev: `pytest>=9`, `ruff>=0.16`.

**Tokenizer**
- Vocabulary 4096. 16 special tokens with IDs 0–15 in this exact order:
  `<|pad|> <|bos|> <|end|> <|user|> <|ai|> <|sep|> <|sys|> <|r0|> … <|r8|>`
- Tokenizer version `tok-v1`, frozen once built. Corpus version `v1`. Benchmark suite `bench-v1`.

**Model**
- Head dim 32. `d_model` is a multiple of 32 in [64, 1024]. `n_layer` in [1, 24]. `ctx_len` is a multiple of 64 in [64, 1024]. Max parameters 85,000,000.
- Shape only grows within a lineage.

**Data**
- Dataset IDs are fixed: `web, books, educational, conversations, code, reasoning, facts, creative`. Custom datasets: `notebook, coaching`.

**Safety and determinism**
- R1: no scripted or templated model output anywhere.
- Model-written code runs only in MiniPy. `exec`, `eval` and `compile` are never applied to model output.
- All randomness flows from explicit seeds (`numpy.random.Generator`, `torch.Generator` / `torch.manual_seed`).

**Testing and I/O**
- Tests are offline and CPU-only by default, and the default suite finishes in under 3 minutes. Use `@pytest.mark.slow` and `@pytest.mark.gpu` for heavy tests.
- Generated data goes under `data_root()` (`AIRACE_DATA` env, default `<repo>/data`) and is never committed.
- All text file I/O is explicit `encoding="utf-8"`.

**Licensing**
- Dataset sources must be permissively licensed and recorded in `manifest.json` and `docs/data-sources.md`.

## Review Focus

1. **Literal special-token strings in text.** If a player types `<|end|>` or `<|ai|>`, or a dataset contains them, they are tokenized as plain text, never as control tokens. Owner: Task 2.
2. **Chat history longer than the attention span.** The reply still generates. The oldest turns are dropped, the prompt starts with `<|bos|>` and ends with `<|ai|>`, and there is no position overflow. Owner: Task 5.
3. **Unsampleable mixtures.** These fail with `MixtureError` naming the dataset, before any training step:
   - all-zero or negative weights
   - unknown dataset IDs
   - a positive weight on a dataset whose pool is empty after prep or purchase
   - a notebook of only blank entries

   Owners: Tasks 6, 8.
4. **Degenerate model outputs** (empty generations, only special tokens, endless repetition) in benchmarks, creativity scoring and fingerprints. Scores come out finite (0 where appropriate), never NaN or a ZeroDivisionError. Owners: Tasks 16, 17, 18.
5. **Non-ASCII text** (accents, emoji, CJK, and the Windows cp1252 console) through tokenizer round-trip, corpus build and CLI printing. Round-trip is lossless and nothing crashes. Owners: Tasks 2, 19.


## Environment note
- uv is not on PATH: run every uv command as `python -m uv ...` (e.g. `cd ml && python -m uv run pytest`).
