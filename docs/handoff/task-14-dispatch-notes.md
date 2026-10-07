# Task 14 controller notes (binding alongside the brief)

## Interfaces from earlier tasks
- **Tokenizer** (`airace_ml.tokenizer`):
  - `train_tokenizer(texts, out_path, vocab_size=VOCAB_SIZE) -> Tok` (`VOCAB_SIZE = 4096`)
  - `encode_doc(tok, text)` → `[bos, ...]`
  - `encode_chat(tok, turns, add_generation_prompt=False)` → `bos (<|user|>|<|ai|>) text <|end|> ...`
  - `Tok.load(path)`
- **Corpus** (`airace_ml.data.corpus`):
  - `DATASET_IDS`, `TOPICS`, `NOISE_KINDS`
  - `DocTags(quality f16, dup_cluster i32, dup_canonical bool, false_fact bool, topic u8, noise_kind u8, heldout bool, purchase_rank f32)`
  - `write_corpus(out_dir, docs: Sequence[Sequence[int]], tags: DocTags, info: dict)`
  - `Corpus.open(dir)`
- **Paths** (`airace_ml.paths`): `tokenizer_path(root)`, `corpus_dir(root)`, `data_root()`.
- **Generators** (all return `list[TextDoc]`):
  - `gen:fact_prose` → `airace_ml.skills.facts.fact_prose_docs(kb, rng, n)`
  - `gen:fact_chat` → `fact_chat_docs(kb, rng, n)`
  - `gen:instructions` → `airace_ml.skills.instructions.instruction_train_docs(kb, rng, n)`
  - `gen:code` → `airace_ml.skills.code.code_train_docs(rng, n)`
  - `gen:reasoning` → `airace_ml.skills.reasoning.reasoning_train_docs(rng, n)`
  - `gen:patterns` → `airace_ml.skills.patterns.pattern_train_docs(rng, n)`

  `TextDoc(kind: "plain"|"chat", text: str, turns: list[(role, text)] | None, topic: str)`. The generators exclude bench-reserved worlds themselves, so any build-seeded rng is fine.
- **KB:** `airace_ml.skills.kb.load_kb()`. **False facts:** `airace_ml.skills.facts.plan_false_facts(kb, rng, n)` → `FalseFactPlan`, which has `to_json()` and `from_json()`.
- **Content processing** (Task 13, `airace_content`):
  - `textproc.normalize_text`. It keeps leading indentation, so it is safe for code.
  - `textproc`: `words`, `build_vocab`, `simplicity`, `quality_score`
  - `noise.NoiseRates`, `noise.inject_noise(texts, rates, rng, kb, plan) -> list[NoisyDoc(text, noise_kind, false_fact)]`. Rates are fractions of the output count, and duplicates are appended copies.
  - `dedup.cluster_near_duplicates(texts, threshold=0.7) -> (cluster int32, canonical bool)` (MinHash; singletons −1/True)
  - `topics.tag_topic(text) -> int`

  Read the modules for details.
- **Preparation filters** (Task 6, `airace_ml.data.prep.eligible_docs`):
  - keep `quality >= threshold`
  - keep `~heldout`
  - with dedup, keep `(dup_cluster < 0) | dup_canonical`
  - with fact-check, drop `false_fact`
  - keep `purchase_rank < fraction`

  Held-out docs are excluded **before** dedup.

## Rulings
1. **Hold out whole duplicate clusters.** Never hold out a cluster's canonical document while its copies stay in training. Draw held-out units from a seeded permutation, where a unit is a whole dup cluster (every member) or a singleton, until at least max(8, 1% of docs) documents are held out. This also keeps near-copies of evaluation text out of training.
2. **Topic tag.** For generator documents whose `TextDoc.topic` is not "other", use the generator's topic. Otherwise use `tag_topic` on the document text (for chats, the joined turn texts).
3. **Quality, dedup and topic for chats** use the turn texts joined with newlines. Every document is normalized with `normalize_text` (each chat turn separately).
4. **Noise on chat documents.**
   - `conversations` has typo noise only. Apply it per chat document: a selected document gets `add_typos` on each turn text, keeps its chat structure, and is tagged "typo".
   - Spam replaces a document, and boilerplate and garble rewrite it as text, so in any dataset they apply only to plain documents. Chats are never turned into plain text.
   - Recipes put them only on plain-text datasets anyway; enforce it with an explicit check, so a recipe that would need them on chats raises a clear error.
   - Tags must always describe the document.
5. **Partial rebuilds** (`datasets=[...]` is a subset) reuse the shared artifacts already in `corpus_dir(out_root)` when present: `false_facts.json` (the same falsehoods told everywhere), `simple_vocab.txt` and `known_vocab.txt`, so rebuilt datasets are scored like the others. `manifest.json` is merged: entries for datasets not rebuilt are kept. On a fresh build, compute everything per the brief's pipeline.
6. **Full-scale token estimate before the tokenizer exists.**
   - When `tokenizer_path(out_root)` exists (frozen), count tokens with it.
   - Otherwise estimate tokens as UTF-8 bytes / `BYTES_PER_TOKEN_ESTIMATE = 3.6` (a named, documented constant).
   - The summary reports actual token counts after encoding.
7. **Vocab size.** After training the tokenizer at `full` scale, assert `tok.vocab_size == 4096`. Tiny fixture text may be too small to reach 4096 merges, so the tiny scale doesn't assert it, but it must still train with `vocab_size=4096`.
8. **Fixture cycling must terminate.** Tiny scale cycles fixture rows to reach 150 docs. If a full pass over a source's rows yields no accepted document (every row filtered), stop for that component instead of looping forever. Full scale stops when the source is exhausted.
9. **Licenses.** huggingface.co is not reachable from the cloud session, so take `license` values from the dataset cards as known:
   - TinyStories `cdla-sharing-1.0`
   - FineWeb and FineWeb-Edu `odc-by`
   - Cosmopedia `apache-2.0`
   - SODA `cc-by-4.0`
   - everyday-conversations `apache-2.0`
   - gutenberg_english `mit`
   - gsm8k `mit`
   - mbpp `cc-by-4.0`
   - wikimedia/wikipedia `cc-by-sa-3.0`

   Task 15 (local, with network) re-verifies all of these and decides on any non-permissive source.
10. **Code documents.**
    - Do not run the prose simplicity filter on code. None of the brief's code components have a simplicity minimum anyway.
    - Normalize code with `normalize_text`, which keeps indentation.
    - mbpp is real Python and gen:code is MiniPy; both stay as the brief says.
11. **Generators and rng.** Seed generator calls and the noise rng from `build_corpus`'s `seed` through `np.random.default_rng` (derive per-dataset/per-component child seeds, e.g. via `np.random.SeedSequence` or `skill_rng`-style hashing). The same seed yields byte-identical outputs.
12. **`hf_fetch`** imports `datasets` lazily inside the function. The `datasets` extra is NOT installed in the cloud venv, and tests must never import it. Do not install it.
13. **Fixture rows** (5–20 per source, with the real field names):
    - Write them so the tiny build exercises every adapter rule: Gutenberg header/footer markers and a non-children subject that must be dropped, an `int_score` below 3, a soda dialogue with fewer than 2 turns, an everyday_conv system turn, gsm8k `<<…>>` and `####`, and mbpp tabs and CRLF.
    - Each source still needs enough accepted rows for every dataset to get documents after its simplicity filter.
    - Use simple, original English text: do not paste real copyrighted text into fixtures.
14. **`purchase_rank`** comes from a second seeded permutation: `rank / n` as float32, in [0, 1).
15. **CLI:** `airace-content build --scale {tiny,full} [--datasets a,b] [--out PATH] [--retrain-tokenizer]`.
    - `--out` defaults to `data_root()`.
    - `main(argv=None) -> int` returns 0 on success.
    - A bad argument or an unknown dataset id prints `error: ...` to stderr and returns 2.
    - Full scale uses `hf_fetch`. Tiny scale uses `hf_fetch` too unless a `--fixtures DIR` flag is given; add that flag, since it lets the tiny build run offline.
