# Fix wave group B: report (content data quality and quota sizing)

Range: `fee3b02` (base, the commit before the first `fix(B)` commit) to `5e14018f0580b00322199234ad1cf3fc44854312` (head).
Order worked: F12, F14, F26, F27, F45, F52, F16 together with F30, as listed.

An earlier attempt of this group had been interrupted: it left F12 and F14 committed and the F26 change uncommitted. I audited all three against the rulings (diffs read in full, tests rerun, RED reproduced against the pre-fix source), kept them as they were, and committed F26 as it stood. Nothing was redone.

| Finding | Status | Commit |
|---|---|---|
| F12 Gutenberg real layout | FIXED | `4cfef01` |
| F14 TinyStories mojibake | FIXED | `999f7e3` |
| F26 code duplicates exact | FIXED | `460ff1a` |
| F27 page-furniture factor | FIXED | `bc4c198` |
| F45 non-ASCII through the build | FIXED | `b9f136d` |
| F52 topic false friends | FIXED | `d2e4af8` |
| F16 quotas: running sum, per-kind estimate | FIXED | `5e14018` |
| F30 same corpus on a second full build | FIXED | `5e14018` |

## F12 (FIXED, `4cfef01`, earlier attempt, audited)
- Fix: `sources.restore_plain_layout` (called by the `_gutenberg` adapter before `strip_gutenberg_boilerplate`). CRLF to LF, then, only when runs of exactly two newlines outnumber all other runs, every run of two becomes one newline and every other run a blank line. The classic-layout fixture is returned unchanged (asserted for every fixture book). `_unwrap` then joins wrapped prose.
- Fixture rows: new directory `ml/tests/fixtures/sources_real/gutenberg.jsonl` (three books: CRLF CRLF CRLF paragraphs, lone-CRLF paragraphs with a verse, LF LF wraps). The shared fixtures are untouched.
- Tests: `test_gutenberg_in_the_real_layout_keeps_sentences_whole` (no blank line between a line without an end mark and a line starting lowercase, whole paragraphs, the verse keeps its lines), `test_restore_plain_layout`.
- RED (probe running the pre-fix `sources.py` from `fee3b02` on the new rows): `cd ml && uv run --no-sync python .../red_f12_f14.py .../old`
  ```
  F12 OLD: blank-line-separated chunks: 23
  F12 OLD: a chunk break mid-sentence: 'f sitting by her sister on the' | 'bank, and of having nothing to'
  ```
- GREEN: `uv run --no-sync pytest tests/content/test_build.py -k "real_layout or restore_plain or tinystories_mojibake"` gives `3 passed`.

## F14 (FIXED, `999f7e3`, earlier attempt, audited)
- Fix: dedicated `_tinystories` adapter (`repair_tinystories_mojibake`): left double quote to `"`, right and left single quote to `'`, en and em dash to `-`, any remaining leftover of the closing double quote to `"`. Clean rows (and rows with real curly quotes) are returned unchanged. `build._simple_vocab` goes through `row_to_content`, so it gets the repair too.
- Fixture: `ml/tests/fixtures/sources_real/tinystories.jsonl` (garbled row, clean row, row with real curly quotes and an accent).
- Test: `test_tinystories_mojibake_is_repaired_to_ascii` (including `didn't` and not `didn` in the simple vocab).
- RED (same probe): `F14 OLD: mojibake still in the corpus text: True '. â€œLook at my kite!â€ she'`. GREEN: the same 3-test command above.

## F26 (FIXED, `460ff1a`, change by the earlier attempt, committed by me after verifying)
- Fix: `noise.inject_noise(..., *, copy_typo_share=COPY_TYPO_SHARE)` (validated to 0..1, NaN refused); `assemble.apply_noise` passes `0.0` for `STRUCTURED_DATASETS`; `STRUCTURED_DATASETS` moved from `build.py` to `assemble.py` (build.py imports it). The comment now says the duplicates of a structured dataset are exact copies. No `NoiseRates` field. The rng is still consumed with 0.0, so the stream is unchanged.
- Tests: `test_duplicates_of_structured_datasets_are_exact_copies` (600 code programs, every copy equals its source; prose keeps typo'd copies), `test_copy_typo_share_sets_how_many_copies_carry_typos`; `test_generator_topics_and_copies` strengthened from "similarity > 0.95" to exact equality.
- RED (source changes stashed, tests kept): `uv run --no-sync pytest tests/content/test_build.py::test_generator_topics_and_copies ...`
  ```
  E   assert 'Progrram:\ny...)\nOutput: 72' == 'Program:\ny ...)\nOutput: 72'
  FAILED test_duplicates_of_structured_datasets_are_exact_copies   (assert False where False = all(...))
  FAILED test_copy_typo_share_sets_how_many_copies_carry_typos     (TypeError: inject_noise() got an unexpected keyword argument 'copy_typo_share')
  ```
- GREEN: same three tests plus the rest of `test_noise_dedup.py`: `40 passed`.

## F27 (FIXED, `bc4c198`)
- Fix, `textproc.quality_score`: new factor `f = 1 - min(1, furniture_lines/3)`, counted like spam hits, so the score is `k * f * g * p * (0.5+0.5a) * (0.5+0.5r)`. `furniture_lines(text)`: non-empty lines with a `|` separator, or a phrase from `FURNITURE_PHRASES` (privacy policy, skip to, sign in, log in, back to top, share this, read more, cookies, follow us, powered by, leave a comment, older posts, sitemap; not "terms of"). Docstring and module docstring updated. `structured_quality` has no `f`: a `|` is ordinary code (tested).
- Three refinements the ruling does not spell out, each tested (see Concerns): phrases match whole words (so "design in" is not "sign in"), phrase matches count only on short lines (at most 20 words; "they made cookies in the kitchen ..." is a 69-word story paragraph in the TinyStories fixture), and a literal `<|end|>` is not a separator.
- Tests (`tests/content/test_textproc.py`): `test_boilerplate_is_found_by_cleaning_at_realistic_document_lengths` at 40, 150, 400, 800 words (200 docs each; clean docs all stay at or above Thorough with mean above 0.9, under half of the boilerplate survives Thorough, Light > Standard > Thorough survivors, boilerplate mean 0.25 below clean); `test_furniture_lines_counts_menus_share_buttons_and_footers` (13 of the 17 furniture lines `add_boilerplate` writes are found; the 4 that are not are named: the breadcrumb, the copyright line, and the two that are spam hits); `test_furniture_lines_does_not_count_prose_or_special_token_text`; `test_quality_loses_a_third_for_each_furniture_line` (exact factors 1, 2/3, 1/3, 0, 0, and furniture times spam); `test_structured_quality_ignores_furniture_because_a_pipe_is_code`.
- RED: `uv run --no-sync pytest tests/content/test_textproc.py -k realistic_document` (before the change), boilerplate share at or above Thorough: `0.88`, `0.955`, `0.895`, `0.78` for 40, 150, 400, 800 words (`assert np.float64(0.88) < 0.5`, and so on).
- GREEN: `tests/content/test_textproc.py` `49 passed` (at that commit). Measured with the test's seeds, boilerplate surviving Light/Standard/Thorough, clean mean:
  ```
   40 words: 0.81 / 0.49 / 0.40  (clean 1.000)      400 words: 0.83 / 0.50 / 0.31  (clean 0.975)
  150 words: 0.80 / 0.54 / 0.45  (clean 0.991)      800 words: 0.81 / 0.52 / 0.09  (clean 0.944)
  ```
  (before: Thorough 0.83 / 0.98 / 0.91 / 0.72). The Task 14 quality-gap test (`test_web_quality_still_separates_noise_kinds`, the 0.2 gap) and `test_cleaning_levels_are_real_filters_over_the_noise_tags` still pass (`tests/content`: `152 passed`).

## F45 (FIXED, `b9f136d`)
- One test, `test_non_ascii_text_goes_through_the_whole_build` in `tests/content/test_build.py`: copies the fixture dir to `tmp_path`, appends a non-ASCII string (accents, curly quotes, emoji, CJK, a literal special token each) to every row of fineweb, tinystories, soda (every turn) and simplewiki, builds the tiny corpus with `fixture_fetch`, and asserts: the decoded noise-free docs contain the strings (every one of them for web); no control id inside plain documents and roles and ends still alternate in chats (so `<|end|>`/`<|ai|>` stay plain text); `simple_vocab.txt` equals `build_vocab` of the modified stories and both vocab files are `[a-z']+`; manifest equals the summary; every `info.json` equals `Corpus.info`. Shared fixtures not edited.
- There was no defect to RED against (the verifier said so: test gap). Instead I proved the test bites: with `normalize_text` mutated to drop non-ASCII (`encode("ascii","ignore")`), `... -k non_ascii` fails (`AssertionError: web`, `assert []`); with the mutation reverted, `1 passed`.

## F52 (FIXED, `d2e4af8`)
- Fix (`topics.py`): month names, weekday names and `calendar` removed from `school` (they belong to no topic; "march" and "may" were never keywords); `addition` and `number` out of `school`, `place` out of `places`, `data` out of `technology`; `marketing` added to `_NOT_INFLECTED`. Module docstring says why.
- Tests: `test_dates_and_false_friends_are_no_topic_keywords` (each word alone and in date sentences tags "other"; "the teacher wrote the date, Monday 5 January" is still school; "data ... database ... computer" is still technology), `test_marketing_is_not_the_market` (`market`/`markets` still places).
- RED: `uv run --no-sync pytest tests/content/test_textproc.py -k "false_friends or marketing"`: `AssertionError: january` (`assert 'january' not in {...}`) and `assert 8 == 10 where 8 = tag_topic('Marketing marketing marketing')`; `2 failed`. GREEN: `155 passed` for `tests/content`.

## F16 and F30 (FIXED, `5e14018`, one commit: the two share the code)
- F16: full-scale quotas are a running sum of a deterministic byte estimate. `assemble._Goal` keeps `tokens`, adding every accepted document's estimate, and `met` is `tokens >= quota` (the 200-document extrapolation and `EST_SAMPLE_DOCS` are gone). Per-component constants: `BYTES_PER_TOKEN_ESTIMATE = 3.2` (prose, the default) and `BYTES_PER_TOKEN_BY_COMPONENT` = code, patterns, mbpp 2.7; tinystories, cosmo_stories 3.6 (`component_bytes_per_token`). `estimate_tokens` and `doc_tokens` take the constant. `_GeneratorCollector` uses the same constants: it asks the generator for `GENERATOR_BATCH_DOCS = 200` documents at a time and keeps them, in order, up to the document that reaches the quota (the rng stream of the tiny scale is unchanged). `cli.format_warnings` also warns, at full scale, when a non-exhausted component's real manifest tokens end below 90% of `planned_tokens` (`SHORT_OF_PLAN`). The `BYTES_PER_TOKEN_ESTIMATE` comment is rewritten (what it is, why not the tokenizer, the measured ratios, indicative).
- F30: `assemble_dataset` no longer takes `count_tokens`; `build_corpus` no longer builds a tokenizer-based counter. Quotas never depend on the frozen tokenizer. The `build.py` module docstring now states the determinism claim correctly (same seed, same source rows, any dataset subset, tokenizer present or not).
- Tests: `test_a_second_full_build_into_the_same_root_gives_the_same_corpus` (full-scale fixture build, copy, second build with the frozen tokenizer, `tokenizer_trained` False, summaries equal, every corpus file and tag array identical via the new helper `_assert_same_corpus`; and a `--datasets code` rebuild matches too), `test_token_estimates_use_a_constant_per_kind_of_text`, `test_a_full_scale_source_stops_at_the_document_that_reaches_its_quota` (250 short documents then long ones, two fills), `test_full_scale_quotas_are_met_whatever_the_first_documents_are` (long head and long tail both land within 1%), `test_full_scale_generators_stop_at_the_document_that_reaches_their_quota` (code, patterns, fact_chat, several batches, resume, determinism), `test_the_cli_warns_when_a_component_lands_below_90_percent_of_its_plan` (85% warns, exactly 90% does not, an exhausted one warns once, tiny scale never).
- RED F30 (old code, `uv run --no-sync pytest tests/content/test_build.py -k "second_full_build"`):
  ```
  E  Differing items:
  E  {'conversations': {... 'docs': 831, 'tokens': 29853, ...}} != {... 'docs': 761, 'tokens': 27717, ...}
  E  {'reasoning': {... 'docs': 226 ...}} != {... 'docs': 271 ...}
  E  {'facts': {... 'docs': 604 ...}} != {... 'docs': 527 ...}
  ```
  (the same numbers the verifier recorded.)
- RED F16 (old code, probe `.../probe_quota.py`): a fineweb-like source with 250 short documents and then 100/900-word ones, quota 50,000: `collected 2858 docs, 1826088 estimated tokens (36.52x)`. GREEN (new code, same probe): `collected 308 docs, 50430 estimated tokens (1.01x)`.
- GREEN: `tests/content` `161 passed`; full suite below.
- Existing tests changed because they encoded the old behaviour (all tightened, none weakened): the `_tokens` helper counts with the component's constant (`doc_tokens`); `test_full_scale_collects_to_the_token_target` (drops `n_stories > 200`, the old sample boundary; the stories now land in `[assigned, assigned + last story)`), `test_full_scale_generators_reach_their_targets` (from a 0.8 to 1.2 band to the exact running-sum bound, total within 1% of 40,000), `test_full_scale_gives_an_exhausted_sources_quota_to_the_others` (0.95 to 1.05 to `[20,000, 1.02 * 20,000]`). `test_same_seed_same_corpus` now calls the extracted helper with the same assertions.
- Sanity probe (not a test), full-scale fixture build with 300k targets, real tokens against assigned (the tokenizer trained on fixture text only, so indicative): code 1.01x, patterns 1.03x, reasoning 0.92x, but the templated generators fact_prose 0.74x, fact_chat 0.79x, instructions 0.82x (see Concerns).

## Suite
- Default: `cd ml && uv run --no-sync pytest -q`: `1284 passed, 1 skipped, 8 deselected in 139.33s (0:02:19)`, no warnings (baseline after group A: `1262 passed, 1 skipped` in 117 s; +22 tests; the time is within machine noise, group A saw 117 s to 127 s on the same base).
- Ruff: `cd ml && uv run --no-sync ruff check .`: `All checks passed!` (also run on the final HEAD).
- Slow: `cd ml && uv run --no-sync pytest -m slow -q`: `5 passed, 1288 deselected in 290.74s (0:04:50)`, run at `5e14018` (includes the quick feasibility gate).
- `git status` clean at the end (the scratchpad, `data/` and `runs/` never staged).

## Self-review (what a reviewer might reject)
- F27 refinements beyond the ruling. The ruling says "lines containing a `|` separator or a phrase". Plain matching would flag ordinary prose: "cookies" is in a 69-word TinyStories paragraph and in gsm8k, "design in" contains "sign in". So phrases match whole words and count only on lines of at most 20 words; a `|` always counts, except inside a `<|...|>` special-token string (Review Focus 1). If the controller wants the literal rule, it is two constants (`_FURNITURE_MAX_WORDS`, `_SPECIAL_TOKEN_LITERAL`) and the tests that name them.
- F16 generator trimming discards the rest of the last 200-document batch per `fill` call (at most 199 documents of generator time per call, deterministic).
- `GENERATOR_BATCH_DOCS`, `component_bytes_per_token`, `SHORT_OF_PLAN` are new public names; `EST_SAMPLE_DOCS` and the `count_tokens` parameter are removed.
- The F52 commit also contains `ruff format` re-wraps of three lines of the F27 tests (same file, same assertions).

## Concerns and other observations (for the controller)
1. Constants for the templated generators. The finding measured prose 3.1 to 3.2, code and patterns 2.6 to 2.7, creative 3.66; it did not measure fact_prose, fact_chat and instructions. My fixture-trained probe suggests highly templated text tokenizes into fewer tokens than 3.2 bytes each, so those three may land under their plan on a real build (0.74 to 0.82x assigned in the probe). The new 90% warning will say so on the first real full build; if it does, the fix is a one-line entry in `BYTES_PER_TOKEN_BY_COMPONENT` (a controller or product decision; I kept to the ruled constants).
2. `|` lines in prose. A markdown table with three or more rows in an educational text (cosmopedia openstax/khan may have them) now counts as three furniture lines, so quality 0 and removal even at Light. The ruling asks for exactly this rule; I did not add a table exception. Worth a look on a real sample.
3. ADR text for the controller: the Task 14 ruling (estimate only before a tokenizer exists, `BYTES_PER_TOKEN_ESTIMATE = 3.6`, `count_tokens=` on `assemble_dataset`, `docs/handoff/sdd/task-14-report.md`) is superseded by F16/F30: tokenizer-independent per-component estimate, running sum, no `count_tokens`. I edited no ADR or handoff doc.
4. F27 reduces but does not remove surviving boilerplate: documents whose only furniture line is one of the four lines the ruled list does not catch ("You are here: ...", "Copyright (c) ... All rights reserved.", and the two spam footers, which have their own factor), and one-line boilerplate documents at 40 to 400 words (2/3 of a score near 1.0 is just above Thorough 0.65). Measured Thorough survival: 0.40, 0.45, 0.31, 0.09 at 40, 150, 400, 800 words, against 0.83 to 0.98 before.
5. F12: the real layout is only inferred from the finding (no network); the fixture rows are modelled on its description. `restore_plain_layout` leaves anything where exact-two runs do not outnumber the rest alone.
