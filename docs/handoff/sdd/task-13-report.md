# Task 13 report: content text processing (normalize, quality, noise, dedup, topics)

Status: DONE_WITH_CONCERNS (concerns are hand-off notes for Task 14, none block this task).
Commit: `155e999 feat(content): normalization, quality scoring, tagged noise injection, near-dup clustering, topics`
(on `claude/upbeat-franklin-k1192h`, not pushed, trailers exactly as dispatched).

## What was implemented

All in `ml/src/airace_content/`:

### `textproc.py`
- `normalize_text`: line endings to `\n`, control characters (category Cc, except `\n` and `\t`) removed, then NFC, runs of spaces collapsed, spaces at the end of every line and of the text stripped, 3+ newlines to 2. Control removal runs before NFC so a removed control character that separated a base letter from a combining mark still composes. Non-ASCII text (accents, CJK, emoji, ZWJ sequences, NBSP) is untouched apart from NFC. Idempotent (fuzz-tested).
- `words`: literal `[a-z']+` on the lowercased text.
- `build_vocab(texts, top_n)`: most frequent words; ties broken alphabetically (so the result does not depend on arrival order); negative `top_n` raises ValueError.
- `simplicity`, `quality_score`: exactly the brief's formulas, constants named. `MOJIBAKE_CHARS` and `SPAM_MARKERS` copied verbatim (checked programmatically against the brief: identical 33 characters).
- Interpretations where the brief is silent (all pinned by tests):
  - spam `hits` = every occurrence of every marker (case-insensitive), not distinct markers;
  - `garble_ratio` = mojibake characters over all characters of the text;
  - `alpha_ratio` uses `str.isalpha` (Unicode-aware) over non-whitespace characters; text with no non-space character scores 0;
  - `repeated_line_ratio` = share of non-empty (stripped) lines that repeat an earlier line;
  - a sentence ends at a `.`, `!` or `?` run; `simplicity` counts only segments that contain a word;
  - tokens made only of apostrophes are not words (stray quote marks do not count as unknown words); `words()` itself is the literal regex.

### `topics.py`
- `TOPIC_KEYWORDS`: 10 topics (not "other"), 57 to 117 keywords each, every keyword in exactly one topic (checked at import and in a test).
- `tag_topic`: argmax of keyword hits, the earlier topic in `TOPICS` wins a tie, "other" when there are no hits. A word also matches through simple endings (`s`, `es`, `ies`, `ed`, `ing`), so "foxes", "berries", "smiled" count.
- Measured agreement with the knowledge-base paragraphs' own topic labels: 90% on non-"other" paragraphs (food 95%, places 98%, science 100%, school 92%, animals 89%, technology 75%, nature 61%; nature is low because colour facts are about everyday objects).

### `dedup.py` (ruling R1: MinHash, no SimHash)
- `minhash_signature(text, num_perm=64) -> uint64 ndarray`: word 3-shingles of the normalized, case-folded, whitespace-collapsed text. Shingle hash = blake2b (8 bytes); the permutations are `murmur3 fmix64(hash XOR salt_i)` with salts = blake2b("minhash-v1:i"). No Python `hash()`. A golden test pins the first three values; a subprocess test (PYTHONHASHSEED 1 and 31337) pins that the signature, `inject_noise` and the clusters are identical across hash seeds.
- Estimator check (300 pairs per row, unique-word docs): mean estimate equals true Jaccard within 0.003 and the spread equals the binomial value (J 0.90 -> 0.900/0.032, 0.75 -> 0.748/0.051, 0.59 -> 0.591/0.060, 0.32 -> 0.322/0.060).
- `cluster_near_duplicates(texts, threshold=0.7) -> (cluster int32, canonical bool)`: exact matches on the blake2b hash of the canonical text, plus LSH candidates (16 bands x 4 rows) verified by estimated Jaccard >= threshold, merged with union-find (smallest index is the root). Singletons: cluster -1, canonical True. Cluster ids are dense from 0 in order of first occurrence; first occurrence is canonical, other members are not. `(cluster < 0) | canonical` keeps exactly one text per cluster (tested against the way `prep.py` consumes it).
- Within an LSH bucket each member is compared only with the representatives found so far, so a bucket of N costs about N comparisons, not N squared (guards against the large-bucket case).
- Non-ASCII: tokens are `\w+` runs plus one token per kana/CJK/Hangul character (those scripts have no spaces), so CJK near-duplicates are found and distinct CJK documents are not merged. Text with no word characters (emoji, punctuation) falls back to whitespace tokens, so `"!!!"` and `"???"` are not merged. Text with fewer than 3 tokens is one shingle. Empty text gets an all-max signature, is never an LSH candidate, and only matches other empty texts through the exact rule.
- Threshold outside (0, 1] raises ValueError; empty input returns empty typed arrays.

### `noise.py`
- `NoiseRates`, `NoisyDoc`, and the four primitives (`add_typos`, `make_spam`, `add_boilerplate`, `garble`), plus `inject_noise`.
- `inject_noise` semantics:
  - Output = the input texts in order (each tagged) followed by the duplicate copies. Rates are fractions of the output count: `n_copies = round(n * d / (1 - d))`, other quotas `round(rate * (n + n_copies))`, capped so they never exceed the input count.
  - Each document gets at most one kind. Selection is by a seeded permutation.
  - Validation: every rate in [0, 1] (NaN rejected), total <= 1, `duplicate < 1`, `false_fact > 0` needs both `kb` and a non-empty `plan` (clear ValueError naming `false_fact`), duplicates need at least one clean document to copy.
  - "Tags mean what they say": if a noise function leaves a document unchanged (typos on a text with no letters) the document is tagged "none". `false_fact` is True exactly for false_fact documents.
  - Duplicate copies are copies of documents that received no other noise, so each copy has a twin in the output (a copy of a spam or false-fact document would carry the wrong tags). Copies come in groups of 2-6 (a lone copy only when exactly 1 copy is requested); about half carry 1-2 typos.
  - `false_fact`: 1-3 `false_fact_sentence`s inserted at sentence boundaries (start, between sentences, end); the original text is otherwise unchanged.
- Primitive semantics: `add_typos(rate)` = per-letter probability, at least one typo guaranteed when there is a letter, only letters are touched (swap with next, drop, double, nearby-key substitute). `garble(rate)` = per-letter probability of replacing a letter, apostrophe, quote or dash with real mojibake (UTF-8 read as Windows-1252, e.g. the "e acute" becomes "A-tilde copyright"), at least one replacement guaranteed, text with nothing to replace gets a junk run; every replacement contains a `MOJIBAKE_CHARS` character, so `quality_score` can see it. `make_spam` always contains 3+ spam markers (quality 0). `add_boilerplate` wraps the unchanged text in 1-4 page-furniture lines (some containing spam markers such as "Subscribe", "Click here", "https://").
- Design decision (flagged): in `inject_noise` the per-document strength of typo and garbled noise is drawn log-uniformly (`TYPO_RATE_RANGE = (0.01, 0.15)`, `GARBLE_RATE_RANGE = (0.005, 0.12)` per letter) instead of a fixed rate. With a fixed rate every garbled document scores 0 and every typo document scores about 0.95, so the Light/Standard/Thorough cleaning levels would remove exactly the same garbled documents and none of the typo documents. With the spread, cleaning is a graded filter. Measured on 3,000 knowledge-base paragraphs (typo 0.2, spam 0.1, boilerplate 0.2, garbled 0.2, false_fact 0.1; vocab built from the clean text), share of each kind that SURVIVES each cleaning level (thresholds from `prep.py`):

| kind | mean quality | light 0.15 | standard 0.40 | thorough 0.65 |
|---|---|---|---|---|
| none | 1.000 | 1.00 | 1.00 | 1.00 |
| typo | 0.774 | 0.95 | 0.85 | 0.74 |
| spam | 0.000 | 0.00 | 0.00 | 0.00 |
| boilerplate | 0.737 | 1.00 | 0.91 | 0.66 |
| garbled | 0.332 | 0.55 | 0.44 | 0.26 |
| false_fact | 1.000 | 1.00 | 1.00 | 1.00 |

  False-fact documents are invisible to cleaning (only fact-checking removes them), as the design intends. The two constants are the single place to retune.

## Tests

`ml/tests/content/test_textproc.py` (27 tests) and `ml/tests/content/test_noise_dedup.py` (37 tests): the brief's 6 tests plus 58 more. The brief's assertions are unchanged.

- normalize: line endings, each control class, spaces, newline runs, non-ASCII/ZWJ/NBSP intact, NFC, removed-control-exposes-combining-mark, idempotence fuzz.
- words / build_vocab / simplicity exact values; quality: one pinned test per factor (a, k, r, g, p), float in [0, 1] fuzz, zero for empty/symbol/CJK-only text, high on real prose, lower with a narrow vocabulary.
- topics: counts, lowercase, disjoint keywords, each topic recognised from its own keywords, argmax and tie rule, plurals and verb forms, "other" for empty, symbols, CJK and emoji, and 90%-level agreement with the knowledge base's own labels (asserted above 85%).
- noise primitives: rate monotonic and proportional, only letters touched, always changes something, deterministic per seed, spam always overt, boilerplate keeps the text and varies header/footer, garble proportional to rate and always garbles (including digits-only, CJK, accented text).
- inject_noise: exact quotas as fractions of the output; each kind verified against its source (typo similar but different with no mojibake or spam; spam 3+ markers and quality 0; boilerplate contains the source; garbled has mojibake and lower quality; false_fact contains 1-3 planned sentences and removing them restores the source); insertions at start, inside and end, inside ones between a sentence end and a space; duplicates are exact or 1-2-typo copies of a clean document, 2-6 per source, each in a cluster with exactly one original that stays canonical; group sizes; tiny corpora; determinism; no-noise identity; validation errors; unchangeable documents are untagged; cleaning levels are graded real filters (table above).
- dedup: shape/dtype/golden values/hash-seed independence/Jaccard estimate; dense ids, canonical flags, prep consumption; exact match ignores case, spacing and CRLF; 300 identical documents; chaining through a middle document; threshold behaviour; recall >= 98% (1 typo) and >= 95% (2 typos) on documents of 60+ words with no merging of different documents; 60+ real hand-written paragraphs (stories, facts, unicode, chats fixtures) produce no clusters; CJK, accented text and emoji / punctuation-only / empty texts.
- Mutation check: ten deliberate bugs (salted `hash()`, no verification step, everything canonical, wrong alpha range, no spam penalty, garble allowed to do nothing, duplicates copied from noisy documents, false_fact flag always False, no "other", control removal after NFC) were each caught by the suite; sources restored afterwards.

### TDD evidence
RED (tests written first; the four modules were then moved aside to reproduce the state before implementation):
```
$ cd ml && uv run --no-sync pytest tests/content -q
E   ModuleNotFoundError: No module named 'airace_content.dedup'
E   ModuleNotFoundError: No module named 'airace_content.textproc'
ERROR tests/content/test_noise_dedup.py
ERROR tests/content/test_textproc.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
2 errors in 0.23s
```
(expected: the modules did not exist). First run after implementing showed three failures that were errors in my own new tests, fixed on the test side without touching the brief's tests: SequenceMatcher on a repeated sentence gives a meaningless ratio (switched to non-repetitive prose); planned false sentences that are substrings of longer planned sentences were found twice (locate longest first and mask); the recall test used 43-word documents where 2 typos legitimately drop below the threshold (see concerns).

GREEN:
```
$ cd ml && uv run --no-sync pytest tests/content -q -W error
................................................................         [100%]
64 passed in 2.25s
$ cd ml && uv run --no-sync pytest -q
785 passed, 4 deselected in 57.32s
$ cd ml && uv run --no-sync ruff check .
All checks passed!
```
Suite time 57 s (limit 3 minutes). Content tests alone: 2.3 s.

Adjustments to the brief's test code (no assertion changed): `ruff format` applied to both test files; ruff's SIM905 fix turned `"cat dog ...".split()` in the brief's `_texts` into a list literal; import order fixed by `ruff check --fix`.

## Files
Created: `ml/src/airace_content/textproc.py`, `topics.py`, `dedup.py`, `noise.py`; `ml/tests/content/test_textproc.py`, `test_noise_dedup.py`. No existing file changed. `docs/progress.md` and the ledger were not touched (controller's).

## Self-review
- Completeness against the brief and ruling R1: all listed functions and signatures exist; no `simhash64`/`max_hamming`; brief tests pass unchanged.
- Determinism: no `hash()`, no iteration over string sets feeding the RNG; verified across PYTHONHASHSEED values.
- Performance (21,700 documents of about 200 characters): inject 0.5 s, quality 0.7 s, topics 0.6 s, normalize 0.2 s, cluster 2.6 s (about 0.12 ms/document; cost grows with document length: 0.67 ms per document at about 1,400 characters).
- Source files contain literal UTF-8 characters only where readable (accented letters, mojibake sample); invisible characters (ZWJ, NBSP, combining marks) are written as escapes.
- Known limitation of `inject_noise` I did not "fix": the module docstring and tests say what a "duplicate" tag means; whether the clustering finds each copy depends on the document length (see concern 2).

## Concerns (for Task 14 / the controller)
1. **`normalize_text` collapses leading indentation** (the brief says runs of spaces collapse, and I implemented that literally). It must not be applied to code documents (`gen:code`, `mbpp`: Python blocks would lose their indentation). `quality_score` is also English-prose-oriented (letter share, English vocabulary): code and non-English text score low or 0. Task 14 should decide per dataset what code documents use (for example skip normalization and score code with a code-aware rule).
2. **Dedup recall on short documents.** Word 3-shingle MinHash at threshold 0.7 finds a 1-typo copy of any document (>= 97% at every length measured) but misses many 2-typo copies of short documents: recall of 2-typo copies is 42% at 25-35 words, 75% at 35-45, 92% at 45-60 and 99% at 60+. `inject_noise` gives about half of its copies 1-2 typos, so for datasets of short documents some "duplicate"-tagged copies will not be clustered (the noise_kind tag stays true, the dup_cluster measurement is blind to them). Fixable only by lowering the threshold or shortening shingles (Task 14/owner decision); I kept the ruling's 0.7.
3. **The facts generator is full of natural near-duplicates.** Of 5,958 distinct `fact_prose_docs` paragraphs, 41% land in dup clusters (631 clusters, up to 16 paragraphs), because paragraphs about one subject reuse the same sentences in a different order. The clusters are real (every clustered paragraph has a member at exact Jaccard >= 0.6; of 315 true pairs at >= 0.7 in a 1,500 sample the clustering found 303). With the "remove duplicates" choice on, the facts dataset (and conversations if fact chats behave the same) will shrink a lot. Task 14 may want fewer paragraphs per subject at full scale.
4. **Single-linkage chaining.** Union-find merges transitively, so a cluster can contain members with low pairwise similarity to the canonical one (minimum 0.11 seen in the facts data). That follows the ruling; `canonical` is always the first occurrence.
5. **Quality scale on real web text is unmeasured.** The thorough threshold (0.65) equals a known-word share of about 76% for an otherwise clean document. With the plan's `known_vocab` (top 30,000 words from books/educational/facts/creative) real web text with many proper nouns or numbers might be cut hard by Thorough. Worth checking the quality histogram on real fineweb rows in Task 15.
6. Rounding details: a lone duplicate copy (total of exactly 1) violates the "2-6" group size; `inject_noise` raises ValueError when the rates leave no clean document to copy (for example typo 0.5 + duplicate 0.5).
