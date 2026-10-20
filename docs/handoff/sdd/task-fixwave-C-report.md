# Fix wave group C: report (content build operations)

Branch `claude/upbeat-franklin-k1192h`. Base (commit just before the first `fix(C)` commit): `5e14018`.
Nothing pushed. All commands run from `/home/user/AI-Race/ml` with `uv run --no-sync`.
No earlier attempt of this group existed (`git log` showed no `fix(C)` commits, clean tree).

| Finding | Status | Commits |
|---|---|---|
| F3 full build reuses a tiny tokenizer / shared files | FIXED | 63b4386, b73286b (test speed) |
| F11 no retry, resume or source pre-flight | FIXED | c6d9277, 805361e (flaky-hub corpus test) |
| F17 silent full build | FIXED | 3ef2bf6, b73286b |
| F18 CLAUDE.md omits the content extra | FIXED | 6be19d3 |
| F28 airace-content does not switch to UTF-8 | FIXED | 1988e37 |
| F29 sources unpinned, revision not in the manifest | FIXED | 164803f, 1212564 |

Suite and checks: see the end of this file.

Real-hub caveat for the whole group: this sandbox has no Hub access and `datasets` is not installed, so
nothing here ran against the real Hub. Everything is tested offline with a fake `datasets` module and a
fake `HfApi` (the real `huggingface_hub` and `httpx` error classes are used for the error tests). I read the
`datasets` 5.1.0 wheel (downloaded to the scratchpad, not installed) to check `load_dataset(revision=...)`,
`DatasetNotFoundError`, and the `ConnectionError("Server Disconnected")` it raises.

## F3: a full build never reuses a tiny tokenizer, a tiny build never overwrites a full root

**Reproduced.** `tests/content/test_scale_guard.py` (new). RED, before the fix:
```
uv run --no-sync pytest tests/content/test_scale_guard.py -q -x -k "does_not_reuse_a_tiny_tokenizer or does_not_overwrite"
E       Failed: DID NOT RAISE BuildError          (tiny build, then build_corpus("full") into the same root)
uv run --no-sync pytest tests/content/test_scale_guard.py -q -k "does_not_overwrite or refuses_a_tiny_build_into"
E       Failed: DID NOT RAISE BuildError          (full build, then a tiny one into the same root)
E       AssertionError: assert 0 == 1  where 0 = main(['build', '--scale', 'tiny', '--fixtures', ..., '--out', <full root>])
```
Fix, as ruled:
- `manifest.py`: `tokenizer/tok-v1/build.json` (written when the tokenizer is promoted, after `trial.replace`):
  `tokenizer_version, corpus_version, scale, seed, sample_bytes, vocab_size, sha256`. I added the file's
  digest beyond the ruling: a record that does not match the tokenizer file beside it (a hand-copied
  tokenizer, a half-finished promotion) reads as "no record", so a crash between the two writes can never
  make a wrong tokenizer look full. `corpus/v1/shared.json`: `corpus_version, scale, seed`, written after
  the three shared artifacts; a partial rebuild that reuses artifacts keeps the earlier record (they were
  made by that build, with that seed).
- `build.py` `_refuse_other_scale`, called before any source is read. At full: `BuildError` naming
  `--retrain-tokenizer` and another `--out` when the frozen tokenizer is not recorded as full (also when
  the record is missing), when a reused tokenizer does not have 4096 entries (the vocab check now covers
  a reused tokenizer), or when a reused shared artifact is not recorded as full (names the files). At
  tiny: `BuildError` ("a tiny build needs its own --out") when the tokenizer record, the shared record or
  any manifest dataset entry says full, also with `--datasets` and `--retrain-tokenizer`.
- `airace_ml/cli.py`: `BUILD_DATA_HINT = "airace-content build --scale full"`.
- `CLAUDE.md`: the build line notes that a tiny build needs its own `--out` (e.g. `--out ../data/tiny`).

GREEN: `uv run --no-sync pytest tests/content/test_scale_guard.py -q` -> `11 passed in 6.4s`. Includes the
tiny-then-full test (`test_a_full_build_does_not_reuse_a_tiny_tokenizer`), full-then-tiny (build, CLI),
wrong-size, stale-record, missing-record, shared-artifact and partial-rebuild-keeps-record tests.

Existing test updated to the ruled behaviour (not weakened): `test_a_failed_build_leaves_no_new_tokenizer`
listed the tokenizer folder as exactly `["tokenizer.json"]`; it is now `["build.json", "tokenizer.json"]`,
and it also asserts that the failed retrain left `build.json` byte-identical.

Legacy roots (built before this fix, no records) are refused at full and need `--retrain-tokenizer` (all
datasets) or another `--out`; at tiny they are allowed unless their manifest says full.

## F11: retry and exact resume inside hf_fetch, plus a source pre-flight

**Reproduced.** Probe `scratchpad/fixwave/C/probe_f11.py` (fake `datasets`; a fetch failing in the last source):
```
hf_fetch mid-stream: ConnectionError Server Disconnected
cli main: uncaught ConnectionError Server Disconnected
```
and the new `tests/content/test_hub_fetch.py` failed at import (no `SourceUnavailable`).

Fix, as ruled:
- `sources.py`: `hf_fetch` retries `load_dataset` and iteration. Transient = `ConnectionError`, `TimeoutError`,
  `httpx.TransportError`, `HfHubHTTPError`/`httpx.HTTPStatusError` with status None, 408, 429 or 5xx; a wrapped
  error is judged by its first `__cause__` that has an opinion. Never retried: 401/403/404/400, `FileNotFoundError`
  (`DatasetNotFoundError`), `ValueError`, `SourceFormatError`, `KeyError`, `OfflineModeIsEnabled` (a
  `ConnectionError` subclass, excluded on purpose: offline mode is asked for), and `BaseException`.
  Back-off 30 s, doubling, capped at 300 s; 8 tries in a row (7 waits: 30, 60, 120, 240, 300, 300, 300 = 22.5 min);
  a row delivered in between starts the count again; one flushed stderr line per retry; then `SourceUnavailable`
  (a `ConnectionError`). Resume: reopen and read past the rows already given (I chose this over
  `state_dict()/load_state_dict`: it is exact whatever the library does inside, and the pinned revision of F29
  makes the order stable).
- `airace_content/cli.py`: any transient error that reaches `main` becomes
  `error: network problem while reading source 'x' (...): <reason>. Check the internet connection, then run the same command again.` (exit 1).
- `build.py` `check_sources`, before anything is assembled: opens every source the selected recipes need (plus
  tinystories when the simple vocabulary is to be built), reads rows until one passes `row_has_format` (at most
  `FORMAT_CHECK_ROWS`), raises `SourceFormatError` as before, wraps any other open error in a `BuildError` that
  names the source (a plain message instead of a traceback), passes network errors and `DatasetsUnavailable`
  through. An empty source passes (the build already warns when a source runs dry).
- `CLAUDE.md`: the optional `hf auth login` line (with F18's edit).

GREEN: `pytest tests/content/test_hub_fetch.py -q` -> `48 passed`. Probe after the fix
(`probe_f11_after.py`, every open fails with `httpx.ConnectError`): seven `network problem while reading ... waiting 30 s ... (try 2 of 8)`
lines (30, 60, 120, 240, 300, 300, 300 s), then `error: network problem while reading source 'tinystories' (roneneldan/TinyStories): ConnectError: no route to host. Check the internet connection, then run the same command again.` and exit 1.
Key tests: the stream that drops twice gives exactly the rows of the steady one; `test_a_build_over_a_flaky_hub_gives_the_corpus_of_a_steady_one`
builds the whole tiny corpus through `hf_fetch` with a connection that drops at every open and compares every
byte with a steady build (verified to fail if the skip is changed to restart from row 0); a late source in
the wrong format or that cannot be opened stops the build before any source is read for real and writes nothing.

## F17: progress lines

**Reproduced.** Probe `probe_f17.py`: a successful `main([... build --scale tiny --fixtures ...])` writes nothing
to stderr (only uv's own env warning); the new `tests/content/test_progress.py` failed at import (no `progress` module).

Fix, as ruled: new `airace_content/progress.py` (`Progress` callback, `silent` default, `Pace`).
`build_corpus(..., progress=)` and `assemble_dataset(..., progress=)` pass it down; the CLI passes a flushed-stderr printer.
Lines: `checking source ...`, `pinning ...`, `simple vocabulary: opening ...`, `<ds>: assembling from ...`,
`<ds>: opening <source> (<repo>)` (and `, pass N` when a tiny build cycles), `<ds>: <source>: N rows read, N documents kept[, ~N tokens at full scale]`
every `EVERY_ROWS = 20,000` rows, `<ds>: finished with <source>: ...`, `<ds>: N documents assembled`,
`known vocabulary: ...`, `<ds>: scoring and finding near-duplicates among N documents`, `training the tokenizer on X MB of text`,
`<ds>: encoding and writing N documents`, `writing the manifest`. Beyond the ruling I added a time rule to the row rule
(a source also reports when 30 s passed since its last line), because Gutenberg rows are whole books and 20,000 rows
could be hours. The `airace-content` docstring says: if no new line appears for 15 minutes, press Ctrl+C and run
the same command again.

GREEN: `pytest tests/content/test_progress.py -q` -> `8 passed`: order of the stages on a fixture build, the row lines
(multiples of N, docs <= rows), estimated tokens at full scale, the clock rule, the "finished" line, a build without a
callback prints nothing and gives byte-identical corpora, CLI: progress on stderr, summary on stdout, warning last.

Existing tests updated to the ruled behaviour (progress now shares stderr): `test_cli_warns_about_exhausted_sources` picks the lines that start
with `warning:` (still exactly one), and `test_cli_reports_a_source_in_an_unexpected_format` checks the last stderr
line starts with `error: source 'gutenberg'` instead of the first.

## F18: CLAUDE.md commands, hint

**Reproduced** (docs/config check). `grep -n "uv sync\|uv run" CLAUDE.md` showed `python -m uv sync` and
`python -m uv run airace-content build ...` without the extra; `pytest tests/content/test_build.py -k missing_datasets_package`
with the stricter assertion failed: `assert 'python -m uv sync --extra content' in "...(uv sync --extra content)\n"`.
Fix: setup `cd ml && python -m uv sync --extra content` (keep it on every later sync), build line
`python -m uv run --extra content airace-content build --scale {tiny,full}`, a `pytest -m gpu` line, the optional
`python -m uv run --extra content hf auth login` line (checked: the `hf auth login` command exists in the installed
huggingface_hub 1.33), and the `DatasetsUnavailable` hint now reads `python -m uv sync --extra content`.
GREEN: `1 passed` (and the whole content suite).

## F28: UTF-8 console in airace-content

**Reproduced.** The new test (cp1252-like strict stdout and stderr, `--out` under `Петко_data`) before the fix:
```
src/airace_content/cli.py:183: in main     (print(format_summary(...)))
E       UnicodeEncodeError: 'charmap' codec can't encode characters in position 81-85: character maps to <undefined>
```
Fix: the switch moved to a shared helper `airace_ml/console.py::use_utf8_output` (`airace_ml.cli` imports it under its old
name `_use_utf8_output`, so its callers and tests are unchanged); `airace_content.cli.main` calls it first (without importing
`airace_ml.cli`, which would load torch). Test:
`test_the_summary_and_warnings_survive_a_cp1252_redirect_and_a_non_ascii_path` (also an error line naming a non-ASCII path).
GREEN: passes; the airace-ml cp1252 tests still pass.

## F29: pinned Hub revision, recorded in the manifest

**Reproduced.** `SourceSpec` had no `revision` field (`['id','hf_path','hf_config','split','license','homepage']`), `hf_fetch` passed
none, `tests/content/test_revisions.py` failed at import (`resolve_revision`).
Fix, as ruled: `SourceSpec.revision` (optional, default None); `hf_fetch` passes `revision=` on every (re)open: the spec's, else the
head sha from `HfApi().dataset_info(repo, timeout=60).sha`, looked up once per read (`resolve_revision`, retried like the reads, and
checking for the content extra first so a missing extra is reported before any network call). `build_corpus(..., pin=)`
(the CLI passes `resolve_revision` for hub reads, nothing for `--fixtures`): asked once per source the build needs, before
the pre-flight, so the checks and every later read of that source use one commit; the manifest's `sources[*].revision` records it
(a source whose datasets were not rebuilt keeps its earlier revision; fixture builds record null). A lookup that fails for a
non-network reason (401, missing repo) becomes a `BuildError` naming the source (second commit).
GREEN: `pytest tests/content/test_revisions.py -q` -> `16 passed`, including an end-to-end `main(["build", "--scale", "tiny", ...])` through the fake
hub that checks every `load_dataset` call carried its repo's sha and the manifest says so.
Existing test updated to the ruled behaviour: `test_hf_fetch_imports_datasets_only_when_it_runs` now has a fake `load_dataset(..., revision)`
and a fake `HfApi` (it asserted the call without `revision`).

## Suite (at head b73286b)

- Default: `cd ml && uv run --no-sync pytest -q` -> `1368 passed, 1 skipped, 8 deselected in 149 s` (no warnings).
  The first full run, before I made the new tests share their builds, took 168 s; `tests/content` is 245 tests in 46 s (it was 161 in 25 s).
- `uv run --no-sync ruff check .` -> `All checks passed!`
- Slow: `uv run --no-sync pytest -m slow -q` -> `5 passed, 1372 deselected in 310.60s (0:05:10)` (includes the quick feasibility gate,
  which builds its data with `build_corpus("tiny", ...)`, so it also exercises the new records and pre-flight).

## Self-review (adversarial)

- A reviewer could say the pre-flight doubles the first-row cost of 14 sources: it is one open and one row each, and it is the whole point.
- `check_sources` wraps every non-network, non-format error from opening a source into a `BuildError` (plain message, exit 1) rather than letting the traceback through; the original is chained (`from e`).
  This goes a little beyond "fail within minutes", chosen because the finding is about a non-technical owner meeting tracebacks.
- Re-entrancy of `_Retries`: one object per `hf_fetch` call / per lookup, no module state; tests patch `sources._pause`.
- The sidecar digest means a tokenizer replaced by hand needs `--retrain-tokenizer` again; that is the intent.
- I accidentally ran `ruff format` over `src/` once, which reformatted two unrelated files (`airace_ml/minipy/*`); I reverted them with `git checkout` before committing, so no commit touches them.

## Other observations (not fixed, outside my notes)

- `airace-content` still prints a Python traceback on Ctrl+C (`airace-ml` handles `KeyboardInterrupt`); F17's handoff note says to press Ctrl+C.
- The Hub tree listing inside `load_dataset` has no timeout (F17's "indefinitely hung metadata call"); the progress lines make a hang visible, the optional watchdog was not built. Only my own `dataset_info` lookup has a timeout (60 s).
- `ruff format --check src tests` reports 9 files outside this group (e.g. `airace_ml/minipy/*`) as not formatted; `ruff check .` is clean.
- When `revision` is a sha, `datasets` 5.1 skips the parquet-export dataset infos (it only uses them for `revision` None/"main"); only `ds.features` metadata is affected for streaming, not the rows. Unverified against the real Hub.
- `docs/data-sources.md` and the plan's Task 15 text may want the new facts: `--extra content`, the `build.json`/`shared.json` records, a tiny build needs its own `--out`, the `hf auth login` option, and the revisions in `manifest.json`.
- Task 15 can pin `SourceSpec.revision` to verified shas in `SOURCES` instead of relying on the head lookup.

## Review fix round

An adversarial review of the group's commits (base `b73286b`, head at the start of the round) reported two blocking findings
and two minor ones. All four were handled. None needed a source change: the two blocking ones were tests that did not pin
the behaviour they were written for, so the proof of the fix is a mutation check (remove the guarded line, watch the new
test fail, restore it), done for each. Commits of this round: `cc89c38`, `1015c2b`, `62614a2`, `0f7846d`, `afee112`.
Base of the round: `b73286b`. Head: `afee112`. Nothing pushed.

| Finding | Severity | Status | Commits |
|---|---|---|---|
| F3 the tokenizer record's sha256 check is not pinned | blocking | FIXED | cc89c38 |
| F29 the head lookup's 60 s timeout is not pinned | blocking | FIXED | 1015c2b |
| F28 second half of the cp1252 test pins nothing | minor | FIXED | 62614a2, 0f7846d |
| F29 a source's revision is that of the latest build that read it | minor | FIXED (docstring) | afee112 |

### F3 (blocking): the digest check in `read_tokenizer_record` had no test

**Reproduced by mutation.** With `return record if record.get("sha256") == digest else None` replaced by `return record`,
all 11 tests in `tests/content/test_scale_guard.py` still passed. The test meant to cover it
(`test_a_record_of_another_tokenizer_does_not_count`) copied the 512-entry tiny tokenizer over the full one, so the
`vocab_size == 4096` check raised a `BuildError` that already contains `--retrain-tokenizer`; the digest was never the
reason it passed.

Fix (tests only; the digest check stays, because the docstring and the first report describe it and it closes a real
hole: a 4096-entry tokenizer replaced after its record was written would otherwise be reused as full). In
`ml/tests/content/test_scale_guard.py` the old test is replaced by three:
- `test_a_full_size_tokenizer_put_there_after_the_record_does_not_count`: rewrites the full root's `tokenizer.json` as the same
  4096 entries in other bytes (`json.dumps(..., indent=1)`, as a second full build's file would differ), asserts that the file
  still loads with `vocab_size == 4096` and that the record still says `scale == "full"`, then asserts `build_corpus("full", ...)`
  raises `BuildError` matching `no record` (and naming `--retrain-tokenizer`) and that the root is byte-identical afterwards.
  Only the digest can tell this file from the recorded one.
- `test_a_record_whose_digest_is_not_the_tokenizers_does_not_count`: the other way round (file untouched, record's `sha256`
  set to the digest of other bytes); asserts the same `no record` refusal.
- `test_a_tiny_tokenizer_put_over_a_full_one_is_refused_at_full`: the old scenario, kept as what it really tests (the tiny
  file has no matching record and is too small), matching `--retrain-tokenizer`.

RED (mutation: `return record` in `read_tokenizer_record`):
```
uv run --no-sync pytest tests/content/test_scale_guard.py -q
E       Failed: DID NOT RAISE BuildError     (test_a_full_size_tokenizer_put_there_after_the_record_does_not_count)
E       Failed: DID NOT RAISE BuildError     (test_a_record_whose_digest_is_not_the_tokenizers_does_not_count)
2 failed, 11 passed in 9.84s
```
GREEN (source restored): `uv run --no-sync pytest tests/content/test_scale_guard.py -q` -> `13 passed in 6.84s`.
Files: `ml/tests/content/test_scale_guard.py`.

### F29 (blocking): the `timeout=` of the head lookup had no test

**Reproduced by mutation.** With `timeout=INFO_TIMEOUT_S` removed from `HfApi().dataset_info(...)` in `resolve_revision`, all 64 tests
in `test_revisions.py` and `test_hub_fetch.py` still passed: the fake `dataset_info` accepted `timeout` and ignored it.

Fix (test only): `FakeHub` in `ml/tests/content/test_revisions.py` records the keywords of every `dataset_info` call
(`info_kwargs`, `{"timeout": timeout, **kwargs}`), and `test_the_head_lookup_cannot_hang_for_ever` asserts
`0 < sources.INFO_TIMEOUT_S <= 300` and that a lookup made by `resolve_revision` passes `timeout == sources.INFO_TIMEOUT_S` both
on the first try and on the retry after a transient error, and that the lookup `hf_fetch` makes does too.

RED (mutation: `dataset_info(spec.hf_path)` without `timeout=`):
```
uv run --no-sync pytest tests/content/test_revisions.py tests/content/test_hub_fetch.py -q
E       assert [{'timeout': ...meout': None}] == [{'timeout': ...meout': 60.0}]
FAILED tests/content/test_revisions.py::test_the_head_lookup_cannot_hang_for_ever
1 failed, 64 passed in 12.50s
```
GREEN (source restored): `uv run --no-sync pytest tests/content/test_revisions.py -q` -> `17 passed in 6.81s`.
Files: `ml/tests/content/test_revisions.py`.

### F28 (minor): the second half of the cp1252 test pinned nothing

The error-line half used `Ünï`, which cp1252 can encode, on streams the first `main()` call had already switched to UTF-8.
It is now a test of its own, `test_an_error_naming_a_non_ascii_path_prints_on_a_cp1252_redirect`: fresh cp1252 streams, a
Cyrillic `--out` (`Петко_data`), a partial build with no tokenizer yet (so the error names the tokenizer path), asserting exit 1,
that the last stderr line starts `error: the tokenizer is trained on` and that the path is in it. The first test keeps its
summary and warnings assertions.

RED (mutation: `use_utf8_output()` in `airace_content.cli.main` replaced by `pass`):
```
uv run --no-sync pytest tests/content/test_build.py -q -k cp1252
FAILED ...::test_the_summary_and_warnings_survive_a_cp1252_redirect_and_a_non_ascii_path
FAILED ...::test_an_error_naming_a_non_ascii_path_prints_on_a_cp1252_redirect
2 failed, 72 deselected in 1.45s
```
GREEN (source restored): `2 passed, 72 deselected in 1.57s`.
Process slip: my first commit of this (`62614a2`) went in with a ruff `RUF059` (unused `out_bytes`) because I piped ruff into
`tail` and the `&&` saw `tail`'s exit code. `0f7846d` fixes it; history is not rewritten. Files: `ml/tests/content/test_build.py`.

### F29 (minor): one revision per source

`make_manifest`'s docstring now says that a source's recorded revision is that of the latest build that read it, and that a
partial rebuild which re-reads a source also feeding datasets it did not rebuild leaves those datasets at an earlier revision
the entry no longer names. This is the limit of one revision per source (a per-dataset record would be a manifest format change,
which nobody asked for). Docstring only. Files: `ml/src/airace_content/manifest.py`.

### Suite after the review fixes (head `afee112`)

- Default: `cd ml && uv run --no-sync pytest -q` -> `1372 passed, 1 skipped, 8 deselected in 162.32s (0:02:42)`, no warnings
  (before the round: 1368 passed in 149 s; the 4 new tests are the +4. The new tests cost under 0.3 s together, measured with
  `--durations=0` on an idle machine: they copy a small root and build nothing; the 1.8 s `setup` line there is the module's
  shared `built_full` fixture, which the module already paid for. So the 13 s difference is run-to-run variance on this
  machine, not the tests).
- `uv run --no-sync ruff check .` -> `All checks passed!` (exit 0, checked by exit code).
- Slow: `uv run --no-sync pytest -m slow -q` -> `5 passed, 1376 deselected in 316.65s (0:05:16)`, exit 0 (includes the quick
  feasibility gate, which builds its data with `build_corpus("tiny", ...)`).

### Notes for the controller

- The default suite is at 2:42 here, under the 3 minute bar with 18 s to spare (it was 2:29 before this round, for the same
  code apart from four sub-second tests). The margin is thin, so if another group's tests land on top of this one the 3 minute
  bar may need a look; that is for the controller, not something this round changes.
- Nothing in this round changes behaviour, so the real-Hub caveat and "Other observations" above stand unchanged.
