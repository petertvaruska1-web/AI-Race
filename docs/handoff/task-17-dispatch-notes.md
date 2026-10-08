# Task 17 controller notes (binding alongside the brief)

## Interfaces from earlier tasks
- **Training:**
  - `airace_ml.train.trainer.train_run(cfg, *, out_dir, data_root=None, device=None, on_event=None, resume=False, checkpoint_every=200) -> TrainResult`
  - `TrainRunConfig(run_id, seed, shape, token_budget, mixture, ..., prep=PrepConfig(), boldness=0.5, batch_tokens=16384, ...)`
  - `ModelShape(n_layer, d_model, ctx_len)`
  - `PrepConfig(cleaning="standard", dedup=False, fact_check=False, variety="natural")`
  - `cfg.validate()` raises ValueError
- **Checkpoints and inference:**
  - `airace_ml.model.checkpoint.load_checkpoint(dir, device) -> (Transformer, CheckpointMeta)`
  - `airace_ml.infer.lm.TorchLM(model, tok, device=None, batch_size=32)`, with `.score_continuations(contexts, continuations) -> [ContinuationScore(sum_logprob, n_tokens)]` and `.generate(...)`
- **Data** (`airace_ml.data.corpus`, `airace_ml.data.prep`):
  - `Corpus.open(dir)`, with `n_docs`, `tags`, and per-document token access (read corpus.py)
  - `heldout_docs(corpus) -> int64 indices`
  - `DATASET_IDS`
- **Paths:** `airace_ml.paths.tokenizer_path(root)`, `corpus_dir(root)`, `novelty_path(root)`, `judge_dir(root)`.
- **Tokenizer:** `Tok.load(path)`, `tok.encode`, `tok.decode`, `tok.bos_id`, `tok.ai_id`, `tok.end_id`.
- **Benchmarks** (Task 16, `airace_ml.evals.suite`): `run_benchmarks(lm, tok, *, suite=None, judge=None, novelty=None, ...)` already has the hook. With both `judge` and `novelty` given, it must call `evals.creativity.score_creativity` and put the result under `"creativity"`. Otherwise `"creativity"` stays in `missing`. `CategoryScore(score, raw, n)` and `ItemResult(item_id, category, score, tags, output)` live in `evals.scoring` or `suite` (read the code).
- **Fakes:** `tests/fakes.py` `ScriptedLM` (Task 16).

## Rulings
1. **The judge directory is self-contained.** `build_judge` copies the frozen tokenizer file into `judge_dir(data_root)` next to the checkpoint. `Judge.load(dir, device=None)` then needs nothing else: it loads the checkpoint, `tokenizer.json` and `calibration.json` from `dir`.
2. **`nll_per_token(token_lists)`** scores each list as a continuation of `[bos]`, giving `-sum_logprob / n_tokens`; an empty list gives `inf`.
   - Lists longer than `ctx_len - 1` are cut to their first `ctx_len - 1` tokens.
   - Batch all lists in one `score_continuations` call; TorchLM chunks internally.
   - Strip special-token ids (< 16) from the lists before scoring. Text replies never contain them, and a degenerate model's special tokens must not crash anything.
3. **Calibration matches `is_well_formed`.** The conversation AI-reply segments are the tokens between each `<|ai|>` and its `<|end|>` in held-out conversation documents. Each segment is scored exactly as `is_well_formed` scores a reply (bos + segment, no chat context). Creative docs: the first 96 tokens after bos. Use seeded selection (or the first N) when more than 500 are available, deterministically.
4. **`ngram_hashes`** uses numpy uint64 arithmetic with natural mod-2^64 wraparound. Make sure no overflow RuntimeWarning is emitted; test output must be pristine. Vectorize: no Python loop over tokens. Fewer than n tokens give an empty array.
5. **`build_novelty_index`** covers all non-held-out documents of the 8 corpora, without n-grams spanning document boundaries. At full scale this is about 150M tokens, so vectorize over each corpus's concatenated token array and mask out windows that cross document offsets. Never loop per document in Python. The index stores sorted unique sampled hashes; `novelty()` uses `np.searchsorted` / `np.isin` on the sorted array.
6. **Creativity sampling.** Generate all 24 stories in ONE batched `lm.generate` call with `seed=seed`, rather than 24 calls seeded `seed + i`. The batch is always the same 24 prompts, so it is deterministic. The spec's ≤ 30 s GPU target for the full benchmark suite makes 24 sequential calls a poor trade. Prompts are `encode_chat(tok, [("user", p)], True)` with temperature 0.9 and top_p 0.95. The story text is `tok.decode(generation.tokens)`, stripped. For judging and novelty, re-encode the stripped text with `judge.tok` / `tok`, which drops any special tokens. An empty story (no words) scores 0.
7. **Distinct-2** = unique word bigrams / total word bigrams, pooled over all stories; 0 when there are none. Words are lowercase alphabetic tokens (reuse an existing helper if one fits).
8. **`STORY_PROMPTS`** are exactly 24, each "Write a short story about …", in simple English, varied across topics (animals, places, feelings, objects, people). None of them may appear verbatim in training data generators. These are chat prompts only.
9. **`judge_train_config`:** `ModelShape(8, 384, 256)`, mixture `{ds: 1.0 for ds in DATASET_IDS}`, and the remaining values from the brief verbatim. `build_judge(..., token_budget=N)` overrides only the budget. `on_event` passes through to `train_run`.
10. **The slow test `test_build_judge_smoke`** must actually pass on CPU in reasonable time. Run it once with `uv run --no-sync pytest -m slow tests/test_creativity.py -v` and report the time. Its 4 steps of 32768 tokens at 15.7M params may take a minute or two on CPU, which is fine for a slow test. If it takes far longer, report it rather than changing the brief's budget.
