# Fix wave group B: content data quality and quota sizing

Read `docs/handoff/fix-wave/fix-wave-rules.md` first; it binds you. Rulings below are the controller's and are binding.

- **Where you work:** ml/src/airace_content/ and ml/tests/content/ (fixtures: add rows only in new files or copies; other tests pin the shared fixture counts).
- **CLAUDE.md edits allowed:** none.
- **Commit subject prefix:** `fix(B): <Fxx> ...`. Report file: `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-B-report.md`.
- **Before reporting:** full default suite, ruff, and the slow suite: `cd ml && uv run --no-sync pytest -m slow -q   (the slow suite, including the quick feasibility gate; a few minutes)`.
- **Order:** work the findings in the order listed.

## F12: Real sedthh/gutenberg_english TEXT puts a blank line after every wrapped line, so every books passage has blank lines mid-sentence and real paragraph breaks are lost
- Finder severity Important; verifier severity Important (fix size small).
- Where: `ml/src/airace_content/sources.py` (131 (_PARAGRAPH_BREAK), 160-172 (strip_gutenberg_boilerplate rejoins with blank lines), 175-180, 207-231 (_unwrap/chunk_passages), 334-337 (_gutenberg); fixture ml/tests/fixtures/sources/gutenberg.jso)
- Full finding: `docs/handoff/final-review/verify/s-data.json` (id F12); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-data`.
- **Ruling:** Restore plain-text layout for the real sedthh Gutenberg rows before chunking (in _gutenberg or at the start of strip_gutenberg_boilerplate): CRLF->LF; then, if runs of exactly two newlines outnumber all other newline runs, replace every run of exactly 2 newlines with one newline and every other run (1 or 3+) with a blank line. The existing _unwrap then joins wrapped prose. The classic-layout fixture must be unaffected. Add fixture rows in the real layout (both paragraph styles: CRLF CRLF CRLF paragraphs and lone-CRLF paragraphs) and a test that no passage has a blank line between a line without an end mark and a line starting lowercase.

## F14: About 7.5% of the TinyStories rows the build reads contain real mojibake ('â€œ', 'â€™', 'â€'), which enters the creative corpus tagged noise 'none' and survives even Thorough cleaning
- Finder severity Important; verifier severity Important (fix size small).
- Where: `ml/src/airace_content/sources.py` (347-353 (_text adapter used for tinystories); ml/src/airace_content/assemble.py:220-228 (prepare_content); ml/src/airace_content/textproc.py:44-58 (normalize_text); plan check docs/superpowers/plans/2)
- Full finding: `docs/handoff/final-review/verify/s-data.json` (id F14); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-data`.
- **Ruling:** Add a dedicated tinystories adapter in sources.py (replace _text for 'tinystories' in _ADAPTERS; row_to_content also feeds build._simple_vocab) that repairs the cp1252 mojibake to ASCII, matching the clean stories' straight quotes: 'â€œ'->'"', 'â€™' and 'â€˜'->"'", 'â€“' and 'â€”'->'-', then any remaining 'â€'->'"'. Clean rows must be byte-identical. Add a fixture row with the mojibake and a test.

## F26: Code dataset copies carry typos that break programs, no cleaning level can remove them, and about 44% also survive dedup, contradicting the Task 14 ruling's premise that code gets no typos
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_content/noise.py` (386-393 (inject_noise misspells half of all copies); assemble.py:106 (code duplicate=0.03); build.py:83-87, 179-180 (structured_quality); textproc.py:124-134)
- Full finding: `docs/handoff/final-review/verify/m-content-text.json` (id F26); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-text`.
- **Ruling:** Add a keyword parameter copy_typo_share (default COPY_TYPO_SHARE) to noise.inject_noise and use it where copies are misspelled; assemble.apply_noise passes 0.0 for structured datasets (move STRUCTURED_DATASETS to assemble.py or textproc.py and import it in build.py). Do NOT add a NoiseRates field. Test: every code duplicate copy equals its source exactly. Correct the build.py comment.

## F27: Boilerplate noise barely moves the quality score on real-length documents, so cleaning almost never removes it
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_content/textproc.py` (98-121, 137-151 (quality_score); noise.py:175-204 (add_boilerplate); tests/content/test_build.py:735-741)
- Full finding: `docs/handoff/final-review/verify/m-content-text.json` (id F27); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-text`.
- **Ruling:** Add a page-furniture factor to textproc.quality_score, counted like spam hits: non-empty lines containing a '|' separator or a phrase from a short furniture list ('privacy policy', 'skip to', 'sign in', 'log in', 'back to top', 'share this', 'read more', 'cookies', 'follow us', 'powered by', 'leave a comment', 'older posts', 'sitemap'; not the generic 'terms of'); factor = 1 - min(1, furniture_lines/3). Tests at realistic lengths (40, 150, 400, 800 words): boilerplate docs mostly fall below Thorough while clean docs keep their score. Re-run the Task 14 quality-gap test and update the docstring.

## F45: No test runs non-ASCII text through the corpus build (Review Focus 5): every source fixture is pure ASCII
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/tests/fixtures/sources/*.jsonl` (all 13 files (exercised by ml/tests/content/test_build.py))
- Full finding: `docs/handoff/final-review/verify/m-content-text.json` (id F45); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-text`.
- **Ruling:** One test in tests/content/test_build.py: copy the fixture dir to tmp_path, append non-ASCII text (accents, emoji, CJK, curly quotes, a literal '<|end|>') to rows of fineweb, tinystories, soda and simplewiki, run build_corpus('tiny', ...) with fixture_fetch, and assert the decoded docs contain the strings and the vocab files and manifest/info.json reload. Do not edit the shared fixtures.

## F52: Topic keywords mis-tag real text on dates and false friends: month and weekday names alone make about 7% of web docs 'school'
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_content/topics.py` (61-73 (school keywords; calendar words at 69-70), 109-121 ('market' via 'marketing', 'place'), 157-177 (_forms); consumer ml/src/airace_ml/data/sampler.py:48-50 (balanced variety))
- Full finding: `docs/handoff/final-review/verify/m-content-text.json` (id F52); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-text`.
- **Ruling:** topics.py: remove month and weekday names (and 'calendar') from 'school' so they belong to no topic; drop 'addition', 'number', 'place' and 'data' from their topics; add 'marketing' to _NOT_INFLECTED.

## F16: Full-scale token quotas are extrapolated from the first 200 accepted docs at a fixed 3.6 bytes/token and never checked against real totals, so datasets land far from target (overall about +15-17%, web up to 1.25-1.47x)
- Finder severity Minor; verifier severity Minor (fix size medium).
- Where: `ml/src/airace_content/assemble.py` (142-147 (EST_SAMPLE_DOCS, BYTES_PER_TOKEN_ESTIMATE 3.6), 283-299 (_Goal), 338-345, 424-434 (generator estimate); ml/src/airace_content/build.py:341; cli format_warnings (warns only for exhausted sourc)
- Full finding: `docs/handoff/final-review/verify/m-content-build.json` (id F16); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-build`.
- **Ruling:** Together with F30. Quotas use a deterministic, tokenizer-independent byte estimate with per-component constants (prose about 3.2 bytes/token, code and patterns about 2.7, creative about 3.6) and a RUNNING SUM over every accepted doc: at full scale a component stops when its estimated tokens reach the quota (drop the 200-doc extrapolation). _GeneratorCollector uses the same constants. cli.format_warnings also warns when a component's real manifest tokens end below 90% of planned. Correct the BYTES_PER_TOKEN_ESTIMATE comment.

## F30: Running the same full build twice gives a different corpus, because quota counting switches from the byte estimate to the frozen tokenizer
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_content/build.py` (23-25 (determinism claim), 341; assemble.py:283-299, 424-434)
- Full finding: `docs/handoff/final-review/verify/m-content-build.json` (id F30); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-build`.
- **Ruling:** Always size quotas with the estimate from F16, never with the frozen tokenizer, so a second build gives the same corpus. Fix the build.py docstring. Test: a second full-scale fixture build into the same root, with the tokenizer already present, gives an identical corpus.
