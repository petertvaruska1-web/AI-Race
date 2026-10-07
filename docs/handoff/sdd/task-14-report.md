# Task 14 report: sources, recipes and the build pipeline (offline-tested)

**Status:** DONE_WITH_CONCERNS (everything in the brief and the 15 rulings is implemented and tested. Concern 1, code-quality scoring, needs a ruling before the full build.)
**Commit:** `965d3ab feat(content): source adapters, dataset recipes, corpus build pipeline and CLI`, on `claude/upbeat-franklin-k1192h`. Not pushed. The trailers are exactly as dispatched.

## What I implemented
- **`airace_content/sources.py`**
  - `SourceSpec`, `SOURCES` (the brief's 13 IDs, paths, configs, splits; licenses per ruling 9; homepage `https://huggingface.co/datasets/<path>`), and `Fetch`.
  - `hf_fetch`: a generator that imports `datasets` only when iterated. A missing package gives an ImportError that names the fix.
  - `fixture_fetch`: reads `<dir>/<id>.jsonl`, one pass, up to `max_rows`.
  - `row_to_content` implements every adapter rule:
    - **gutenberg:**
      - The subjects filter is case-insensitive. METADATA can be a JSON string or a dict, with `subjects` as a string or a list.
      - It drops the header (the START marker, or the old `*END*THE SMALL PRINT` line, plus leading "Produced by…" credit paragraphs) and the footer (the END marker or the "End of the Project Gutenberg…" line).
      - Hard-wrapped prose paragraphs are unwrapped; verse keeps its lines.
      - Paragraphs are packed into passages of at most 400 words, and an over-long paragraph is cut between sentences. Passages under 20 words are dropped.
    - **fineweb_edu:** keeps `int_score >= 3`.
    - **soda:** `speakers[0]` is the user and every other speaker the AI; dialogues with fewer than 2 turns are dropped.
    - **everyday_conv:** maps user and assistant roles, drops system turns, and also requires at least 2 turns.
    - **gsm8k:** strips `<<…>>` and splits on `####`. A trailing period on the final answer is removed; a row without `####` is dropped.
    - **mbpp:** `# {text}\n{code}`, with tabs as 4 spaces and CRLF as LF. The problem text is collapsed to one line so the comment stays one line.
    - **simplewiki:** the title line, then whole paragraphs up to 150 words. A first paragraph longer than that is cut at a sentence end.
- **`airace_content/assemble.py`**
  - `Component`, `Recipe`, `RECIPES` and `SCALES`, verbatim. `GENERATORS` maps the 6 `gen:` names to the Task 10–12 functions.
  - `validate_recipe` checks:
    - known names
    - shares > 0 and summing to 1
    - no simplicity minimum on code (ruling 10)
  - `split_count` uses largest remainders, which gives tiny quotas such as 120/30 and 83/23/22/22.
  - `prepare_content` normalizes every document with `normalize_text`; for chats, each turn separately.
  - Per-component collection: tiny scale counts documents and cycles passes; full scale uses a token target (see rulings 6 and 8).
  - `apply_noise`: plain datasets go through `inject_noise`. Duplicate copies are traced to their source (`source_of_copies`) and record `copy_of` and the source's topic. Chats get typo noise only.
  - `AssembledDoc(content, noise_kind, false_fact, topic=None, copy_of=None)`: the last two fields are additions with defaults.
  - `assemble_dataset(...)` follows the brief's signature, plus keyword-only `count_tokens=` and `stats=`.
- **`airace_content/build.py`**
  - `build_corpus` runs the 12 pipeline steps.
  - `heldout_mask` holds out whole units (ruling 1, plus copy links).
  - `purchase_ranks`; `tokenizer_texts` (20 MB budget, an equal share per dataset, held-out documents excluded).
  - `stream(seed, name)` gives named rng streams.
  - `BuildError` is raised for builds that cannot run as asked; `BuildSummary`.
- **`airace_content/manifest.py`**
  - The manifest is built and merged.
  - `sources` lists every source feeding a built dataset, with license, homepage and `feeds`.
  - Each dataset entry records:
    - components with their docs, rows read and passes
    - noise rates and counts
    - held-out and duplicate counts
    - topics
    - scale and seed
  - Readers and writers for `false_facts.json` (`FalseFactPlan.to_json`) and the vocab text files (sorted, one word per line).
- **`airace_content/cli.py`**
  - `airace-content build --scale {tiny,full} [--datasets a,b] [--out PATH] [--retrain-tokenizer] [--fixtures DIR]`, through `main(argv) -> int`.
  - Prints a per-dataset summary.
- **Fixtures:** `ml/tests/fixtures/sources/*.jsonl`, 13 files of 6–14 rows each, with the real field names. All text is original simple English, written for these fixtures.

## Tiny build summary
Command: `airace-content build --scale tiny --fixtures tests/fixtures/sources` (0.9 s; the tokenizer trained to 4,096 entries).

| dataset | docs | tokens | components (docs) | noise counts | held out | dup clusters | quality ≥ light / standard / thorough |
|---|---|---|---|---|---|---|---|
| web | 163 | 12,499 | fineweb 150 (14 passes) | typo 24, spam 10, boilerplate 16, garbled 8, duplicate 13, false_fact 10 | 14 | 11 | 90% / 87% / 79% |
| books | 153 | 36,756 | gutenberg 150 (30 passes) | typo 3, garbled 2, duplicate 3 | 32 | 5 | 99% / 99% / 99% |
| educational | 153 | 10,911 | cosmo_khan 38, cosmo_wikihow 37, cosmo_openstax 30, fineweb_edu 45 | duplicate 3, false_fact 2 | 11 | 25 | 100% / 100% / 100% |
| conversations | 150 | 6,746 | soda 83, everyday_conv 23, fact_chat 22, instructions 22 | typo 5 | 8 | 13 | 98% / 95% / 88% |
| code | 155 | 6,977 | code 120, mbpp 30 | duplicate 5 | 8 | 10 | **13% / 5% / 0%** |
| reasoning | 150 | 22,165 | reasoning 90, patterns 45, gsm8k 15 | none | 8 | 7 | 74% / 36% / 8% |
| facts | 153 | 8,658 | fact_prose 90, simplewiki 60 | duplicate 3 | 12 | 11 | 100% / 100% / 100% |
| creative | 155 | 13,166 | tinystories 128, cosmo_stories 22 | typo 3, duplicate 5 | 11 | 18 | 100% / 100% / 100% |

Notes on these numbers:
- Docs exceed 150 by the appended duplicate copies.
- Books holds out 32 because fixture cycling turns 5 passages into clusters of about 30, and a held-out unit is a whole cluster.
- Simplicity and adapter filters at work in the fixtures:
  - fineweb: 3 of 14 rows fail simplicity.
  - fineweb_edu: 2 rows have `int_score` < 3 and 1 fails simplicity.
  - gutenberg: 2 of 6 books are not children's books.
  - soda and everyday_conv: 1 row each has fewer than 2 turns.

**Full-scale rehearsal (offline).** Real targets ÷100 and ÷20, endless synthetic sources made from the fixtures, frozen tokenizer:
- Every dataset reached 98–119% of its token target. Web is over because of its 8% appended duplicates.
- Run time: 41 s at ÷100 and 119 s at ÷20; about 21 s of each is the fixed simple-vocab pass over 200k TinyStories rows.
- Peak RSS was 122 MB at ÷100 and 135 MB at ÷20.
- Extrapolated real build: about 35 min of local processing, plus streaming time. Memory should stay well under a few GB.
- Generators at full-scale counts all succeed: code 232k docs in 31 s, instructions 61k in 28 s, the rest in under 8 s.

## How each ruling was implemented
1. **Whole clusters held out.**
   - Units are dup clusters or single documents. They are taken in a seeded permutation (`stream(seed, "heldout/<ds>")`) until `max(8, ceil(1% × n))` documents are held out.
   - Addition: an injected duplicate copy is joined to its source's unit through `copy_of`, even when MinHash missed it. Task 13 measured a 42% miss rate for 2-typo copies of short documents, so a held-out text could otherwise keep a near-copy in training.
   - Guard: a unit that would push the held-out total past half the dataset is skipped (`HELDOUT_MAX_SHARE = 0.5`).
   - `dup_cluster` itself stays the pure MinHash measurement.
2. **Topics.**
   - A generator `TextDoc.topic` other than "other" is used as is, and a copy inherits its source's topic. Everything else gets `tag_topic(joined text)`.
   - A spam document loses its original's topic, because spam replaced the text.
3. **Chats.**
   - `content_text` joins turn texts with `\n` for quality, dedup, topic and tokenizer text.
   - `prepare_content` normalizes each turn separately and drops empty turns.
4. **Chat noise.**
   - `apply_noise` raises `ValueError` ("dataset 'conversations' has chat documents, but its recipe adds ['spam'] noise …") for any non-typo kind on a dataset containing chats.
   - A typo-selected chat gets `add_typos` on every turn, at one log-uniform rate per document, keeps its turns, and is tagged typo only if it changed.
5. **Partial rebuilds.**
   - These reuse `false_facts.json`, `simple_vocab.txt` and `known_vocab.txt` when present, and merge `manifest.json`, keeping the entries of datasets that were not rebuilt.
   - When `known_vocab.txt` is absent, the four vocab datasets are assembled in memory and not written.
   - Every random stream is named per dataset, so a partial build without artifacts reproduces a full build's artifacts and web corpus byte for byte. A test checks this.
   - A build that must train the tokenizer (it is missing, or `retrain_tokenizer`) raises `BuildError` unless it builds all datasets. A tokenizer trained on a subset would be frozen forever, and retraining would orphan the other corpora.
6. **Token counts.**
   - With a frozen tokenizer: `len(tok.encode)`. Otherwise `BYTES_PER_TOKEN_ESTIMATE = 3.6` (named and documented), plus the special-token overhead.
   - The summary and manifest report actual token counts after encoding.
7. **Vocab size.**
   - At full scale, a `RuntimeError` is raised if `vocab_size != 4096`.
   - Hardening: the tokenizer is trained to `tokenizer.json.tmp` and moved into place only after the check. Otherwise a rejected short tokenizer would be reused as "frozen" on the next run.
   - Tiny scale trains with `vocab_size=4096` and does not assert it (the fixtures reach 4,096 anyway).
8. **Fixture cycling.**
   - Tiny scale starts another pass while the quota is unmet, but stops a component after a pass that accepts no document. Full scale never cycles and stops at exhaustion.
   - `MAX_ROWS_PER_PASS` is 20k per pass at tiny scale and 10⁹ at full scale.
9. **Licenses:** exactly as listed. `simplewiki` is `cc-by-sa-3.0`; Task 15 decides.
10. **Code.**
    - `validate_recipe` refuses a simplicity minimum on code.
    - Code is normalized with `normalize_text`, which keeps indentation. mbpp stays real Python and gen:code stays MiniPy.
11. **Seeds.**
    - `stream(seed, name)` = `default_rng([seed, blake2b(name)])` for:
      - `false_facts`
      - `assemble/<ds>`
      - `heldout/<ds>`
      - `purchase/<ds>`
      - `tokenizer/<ds>`
    - Components and noise get child seeds drawn from the dataset stream.
    - The same seed gives byte-identical `tokens.bin`, `offsets.npy`, `info.json`, manifest, artifacts and `tokenizer.json`, and equal tag arrays. `tags.npz` bytes differ only by zip timestamps; see concern 4.
12. **`datasets` is never imported in tests.** A test checks that `datasets` is not in `sys.modules` after importing every module and calling `hf_fetch`; the generator body never runs.
13. **Fixtures** exercise every rule, and a test asserts that they do. Every dataset gets documents after its filters.
14. **`purchase_rank`** = `rank / n` from a second permutation, as float32, clipped below 1.
15. **CLI**
    - `--out` defaults to `data_root()`.
    - Exit codes:
      - 0 on success
      - 2 with `error: …` on stderr for a bad argument, an unknown or empty dataset list, `--fixtures` with full scale, or a missing fixtures directory
      - 1 with `error: …` when the build cannot run (`BuildError`, or a missing `datasets` package)
    - `--fixtures DIR` is tiny scale only. `--help` returns 0.

## Decisions and interpretations (each also recorded in docstrings and comments)
- **Educational row, "(simplicity ≥ 0.35)".** I read it as binding to `fineweb_edu` only, the component it follows. That matches facts, where it binds to simplewiki and not the generator, and leaves the synthetic Cosmopedia text unfiltered. If all four components were meant, it is a one-line recipe change.
- **Generator components at full scale.** A 200-document sample comes from a separate child stream and sets the document count. The kept documents then come from one generator call, so per-call constraints (no repeated world in a document, and so on) hold.
- **Tokenizer sample:** an equal byte share per dataset, in a seeded order, skipping documents that would overflow the share. Held-out documents are excluded.
- **Encoded documents** are held as uint16 arrays, because 30M Python ints would take about 1 GB.
- **Copy matching** is keyed by a 16-byte digest of each candidate's non-letter characters, with candidates re-split only when matching. This keeps memory small at full scale.

## Tests and results
- `ml/tests/content/test_build.py` holds 48 tests: the brief's 3 verbatim plus 45 added; 7.8 s with the rest of `tests/content`. A module-scoped shared tiny build serves the read-only tests. What the added tests cover:
  - the source table, recipes and scales exactly
  - every adapter rule, including the Gutenberg header, footer, credits, wrapping, verse and passage bounds, and `chunk_passages` cuts
  - that the fixtures exercise every rule, and the lazy `datasets` import
  - tiny cycling to quota, and the cycling termination guard (all rows filtered, and an empty source)
  - full-scale token targets for sources (past the 200-document estimate) and generators, and stopping at exhaustion
  - generator topics and copy topics/`copy_of`; `source_of_copies`
  - chat typo noise keeping structure, every non-typo kind refused on chats, and a recipe with spam on conversations refused
  - held-out units are whole clusters, follow missed copies, and keep the larger part for training
  - purchase-rank permutation, and the tokenizer sample's budget and held-out exclusion
  - in the tiny build: tags describe documents, checked against decoded text:
    - spam markers in spam documents
    - page furniture in boilerplate documents
    - mojibake in garbled documents
    - a planned subject and wrong object in false-fact documents
    - typo documents differ from every clean document
    - duplicates are at least 95% similar to a clean document
    - one canonical document per cluster
  - also in the tiny build: chats keep role/end structure; manifest, info and shared artifacts
  - same seed gives an identical corpus, and a different seed differs
  - a partial rebuild reuses artifacts (a tampered `known_vocab` is honored), merges the manifest and leaves other corpora untouched
  - a partial rebuild without artifacts equals the full build
  - builds that cannot run (`BuildError`), and a full-scale build from fixtures with a too-small tokenizer refused and not left behind
  - CLI: success, 9 bad-argument cases returning 2, a build error returning 1, and `--help`
- **Full suite:** `cd ml && uv run --no-sync pytest` → **843 passed, 4 deselected in 64.07 s** with no warnings (it was about 61 s before this task). `uv run --no-sync ruff check .` → All checks passed. `ruff format --check` is clean for the new files.
- **Mutation checks** (the added tests were written after the code, so I verified that they bite). Each mutation was applied, the selected tests run, and the file restored byte-identical:
  - held-out ignoring clusters
  - ignoring copy links
  - removing the chat-noise check
  - disabling partial-build reuse
  - removing the cycling guard (the test hangs and was killed by timeout)
  - dropping copy topics
  - letting held-out documents into the tokenizer sample
  - keeping the Gutenberg header
  - leaving a rejected tokenizer in place

  All 9 were caught.

## TDD evidence
- **RED:** `cd ml && uv run --no-sync pytest tests/content/test_build.py -v`
  ```
  ImportError while importing test module '/home/user/AI-Race/ml/tests/content/test_build.py'.
  tests/content/test_build.py:4: in <module>
      from airace_content.build import build_corpus
  E   ModuleNotFoundError: No module named 'airace_content.build'
  1 error in 0.12s
  ```
  This was expected: the modules did not exist yet.
- **GREEN** (brief tests): `uv run --no-sync pytest tests/content/test_build.py -v` → `test_adapters PASSED`, `test_tiny_build_end_to_end PASSED`, `test_tokenizer_is_frozen_on_rebuild PASSED` (3 passed in 1.83 s).
- **Final:** `uv run --no-sync pytest tests/content -q` → 122 passed in 7.8 s; full suite as above.
- **Brief test code:** formatting only, the way ruff format lays it out. A blank line between the stdlib and first-party imports, two blank lines between functions, and the first soda assertion wrapped at 100 columns. No assertion was changed.

## Files changed (all new)
- `ml/src/airace_content/sources.py`, `assemble.py`, `manifest.py`, `build.py`, `cli.py`
- `ml/tests/content/test_build.py`
- `ml/tests/fixtures/sources/{tinystories,fineweb,fineweb_edu,cosmo_khan,cosmo_wikihow,cosmo_openstax,cosmo_stories,soda,everyday_conv,gutenberg,gsm8k,mbpp,simplewiki}.jsonl`

## Self-review
- **Completeness:** every brief interface, pipeline step and ruling is covered and tested. `pyproject.toml` and `uv.lock` are untouched, and nothing outside `ml/` was changed.
- **Additions beyond the brief**, all small and documented:
  - the `AssembledDoc.topic` and `copy_of` fields
  - `count_tokens=` and `stats=` on `assemble_dataset`
  - held-out copy links and the cap at half a dataset
  - held-out documents kept out of the tokenizer sample
  - the trial-file tokenizer
  - `BuildError`
  - CLI exit code 1 for build errors
  - the extra adapter niceties: Gutenberg credits, unwrapping and the 20-word minimum; everyday_conv needing at least 2 turns; mbpp one-line text
- **Size:** `assemble.py` is 521 lines, larger than the others. It holds recipes, collection, noise and copy matching, which is cohesive with "assemble a dataset". The test file is 733 lines because it covers many rules.

## Concerns
1. **Clean code scores near-zero quality, so the default cleaning removes most of the code dataset (decision needed).**
   - Implemented as the brief says: `quality_score` with the prose `known_vocab` for every document.
   - Tiny build: only 13% / 5% / 0% of code documents pass light / standard / thorough. Reasoning passes 74% / 36% / 8%, because patterns lines are mostly digits.
   - Full-scale estimate: a 30k vocabulary from real books, educational, facts and creative text would know "print", "return", "total" and so on, but not identifiers such as `def`, `xs`, `nums`, `lst`, `seq` or `arr`. Code would then average about 0.45: roughly half of clean code dropped at the default "standard" level and almost all at "thorough".
   - With the code dataset's own words added to `known_vocab`, code averages 0.62 (100% / 100% / 36%) and patterns 0.66.
   - Code has no defect noise (only duplicates, which dedup handles), so this filtering is a mis-measurement, not a real consequence of cleaning. It would also shrink code-heavy mixes (gate G3).
   - Suggested fix: score the code dataset with `known_vocab ∪` the top words of the code dataset, or exempt code from the cleaning threshold. It is a one-line change in `build.tag_documents`. I did not deviate from the brief without a ruling.
2. **Generator diversity at full-scale counts.** `fact_chat` gives only 24% distinct chats when asked for about 89k (15% of conversations), and `fact_prose` 86%. Task 13 also noted that 41% of fact paragraphs are near-duplicates. With "remove duplicates", these shares shrink a lot. This is a data-design question for Task 10/15; it does not block this task.
3. **The Gutenberg METADATA format is unverified offline.** The adapter accepts a JSON string or a dict, with `subjects` as a string or a list (or `subject`), matched case-insensitively. Task 15 should confirm in the manifest that `gutenberg` docs > 0 relative to `rows_read`.
4. **`tags.npz` is not byte-identical between builds.** `np.savez` (in Task 6's `write_corpus`) stamps zip entries with the write time. The arrays are identical and every other output is byte-identical. The determinism test compares tag arrays for this reason.
5. **Housekeeping for the controller:** `CLAUDE.md` "Commands" could now list `airace-content build --scale tiny --fixtures tests/fixtures/sources`. I left docs alone because the brief's commit scope is `ml/`.

## Pre-review change (Ruling A)
**Commit:** `69fa501 fix(content): score code and reasoning with structured_quality, the language-free quality factors` (not pushed; the same two trailers).

### What changed
- **`airace_content/textproc.py`**
  - New `structured_quality(text) -> float` = `g * p * (0.5 + 0.5 r)`. It returns 0 for empty or whitespace-only text, the same emptiness test as `quality_score`.
  - The garble, spam and repeated-line factors now live in one private helper, `_language_free_quality(text)`. Both `quality_score` (`k * (0.5 + 0.5 a) * _language_free_quality(text)`) and `structured_quality` call it, so there is no duplicated factor code.
  - This is a pure refactor of `quality_score`: on the tiny build, the quality arrays of the six prose datasets are bit-for-bit identical to before, and all of Task 13's quality tests pass unchanged.
- **`airace_content/build.py`**
  - New documented constant `STRUCTURED_DATASETS = frozenset({"code", "reasoning"})`, with the ruling's rationale in a comment.
  - `tag_documents` scores every document of those datasets with `structured_quality`; all other datasets keep `quality_score(text, known_vocab)`.
  - The module docstring's step 5 mentions this.

### Tests (added; nothing existing changed)
- `tests/content/test_textproc.py`:
  - `test_structured_quality_scores_clean_code_and_puzzles_high`:
    - 300 generated code docs (MiniPy programs and functions), 100 reasoning docs, 100 pattern docs, the existing `MBPP_STYLE` function and the existing MiniPy `NESTED_PROGRAM` all score ≥ the Thorough threshold.
    - `MBPP_STYLE` scores exactly 1.0, while `quality_score` with the prose vocabulary gives it less than the Light threshold.
  - `test_structured_quality_scores_spam_mojibake_and_repeats_low`:
    - `make_spam` scores 0.
    - Garbled code (rate 0.1) scores below Light.
    - One line repeated 10 times scores exactly 0.55.
  - `test_structured_quality_of_empty_text_is_zero`: `""`, spaces, and newlines with a tab all score 0.
  - `test_structured_quality_shares_the_language_free_factors`: on text with every word known and over 90% letters (so `k = a = 1`), `structured_quality == quality_score`, including a repeated-line case (`0.5 + 0.5 · 2/3`).
- `tests/content/test_build.py`:
  - `test_code_and_reasoning_are_scored_as_structured_text`:
    - `STRUCTURED_DATASETS == {"code", "reasoning"}`.
    - In the tiny build, at least 98% of clean (`noise_kind` none) code and reasoning documents reach `np.float16(CLEANING_THRESHOLDS["thorough"])`. Float16 is how `prep.eligible_docs` compares.
    - Measured: 100% for both; the minimum quality is 0.95 for code and 0.91 for reasoning. The 98% bound expresses "nearly all" with a margin.
  - `test_web_quality_still_separates_noise_kinds`:
    - The mean web quality of typo, boilerplate and garbled documents is each more than 0.2 below that of clean documents: clean 0.93, typo 0.65, boilerplate 0.61, garbled 0.32.
    - Every spam document scores below Light.
- Note: my first draft of the unit tests defined a second module-level `MBPP_STYLE`, which shadowed Task 13's constant of the same name and broke `test_normalized_mbpp_style_function_is_unchanged_and_still_runs`. I removed my copy and reused the existing constant; that Task 13 test passes again unchanged.

### Commands and output
- **RED 1.** `cd ml && uv run --no-sync pytest tests/content/test_textproc.py -q` gave `ImportError: cannot import name 'structured_quality' from 'airace_content.textproc'`. `tests/content/test_build.py` gave `ImportError: cannot import name 'STRUCTURED_DATASETS' from 'airace_content.build'`. Both were expected: the names did not exist yet.
- **RED 2 (behaviour).** With `structured_quality` implemented and `STRUCTURED_DATASETS` defined but not yet used, `uv run --no-sync pytest tests/content -q` gave `FAILED test_code_and_reasoning_are_scored_as_structured_text` with `assert np.float64(0.0) >= 0.98`. That is the code pass rate at Thorough before wiring: expected.
- **GREEN.** `uv run --no-sync pytest tests/content -q` → `128 passed in 9.33s`.
- **Mutation check.** Scoring web with `structured_quality` makes `test_web_quality_still_separates_noise_kinds` fail (`assert 1.0 < 1.0 - 0.2`). Restored byte-identical.
- **Full suite.** `uv run --no-sync pytest -q` → `849 passed, 4 deselected in 61.84s` (no warnings). `uv run --no-sync ruff check .` → `All checks passed!`; `ruff format --check` → `14 files already formatted`.

### Pass rates in the tiny build, re-measured
`airace-content build --scale tiny --fixtures tests/fixtures/sources`; a document passes a level when `quality >= np.float16(threshold)`, with Light 0.15, Standard 0.40 and Thorough 0.65.

| dataset | docs | scorer | before (light / standard / thorough) | after (light / standard / thorough) | after, clean docs only | mean quality after |
|---|---|---|---|---|---|---|
| web | 163 | quality_score | 90% / 87% / 79% | 90% / 87% / 79% | 100% / 100% / 100% | 0.77 |
| books | 153 | quality_score | 99% / 99% / 99% | 99% / 99% / 99% | 100% / 100% / 100% | 0.99 |
| educational | 153 | quality_score | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% / 100% | 1.00 |
| conversations | 150 | quality_score | 98% / 95% / 88% | 98% / 95% / 88% | 99% / 98% / 90% | 0.78 |
| code | 155 | **structured_quality** | 13% / 5% / 0% | **100% / 100% / 100%** | 100% / 100% / 100% | 1.00 |
| reasoning | 150 | **structured_quality** | 74% / 36% / 8% | **100% / 100% / 100%** | 100% / 100% / 100% | 0.99 |
| facts | 153 | quality_score | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% / 100% | 1.00 |
| creative | 155 | quality_score | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% / 100% | 1.00 |

- Web still separates noise by quality. By kind: clean 0.93, duplicate 0.97, false_fact 0.91, typo 0.65, boilerplate 0.61, garbled 0.32, spam 0.00.
- In conversations, about 10% of clean chats fail Thorough. This predates Ruling A and is unchanged by it. It is a tiny-scale artifact: the tiny `known_vocab` is only the fixture text of four datasets, so some knowledge-base words in `fact_chat` are unknown (for example "Which letters mean manganese in chemistry?" scored 0). At full scale the 30,000-word vocabulary should cover them; Task 15 can confirm on real data.
