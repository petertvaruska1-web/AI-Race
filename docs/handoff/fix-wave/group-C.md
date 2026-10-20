# Fix wave group C: content build operations

Read `docs/handoff/fix-wave/fix-wave-rules.md` first; it binds you. Rulings below are the controller's and are binding.

- **Where you work:** ml/src/airace_content/, airace_ml/cli.py BUILD_DATA_HINT, ml/tests/content/.
- **CLAUDE.md edits allowed:** CLAUDE.md 'Commands' lines for setup, the corpus build, the gpu tests and the optional hf login (F3, F11, F18).
- **Commit subject prefix:** `fix(C): <Fxx> ...`. Report file: `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-C-report.md`.
- **Before reporting:** full default suite, ruff, and the slow suite: `cd ml && uv run --no-sync pytest -m slow -q   (the slow suite, including the quick feasibility gate; a few minutes)`.
- **Order:** work the findings in the order listed.

## F3: A full corpus build silently reuses (and freezes as tok-v1) a tokenizer and shared vocab files left by an earlier tiny build in the same data root; the airace-ml CLI itself suggests the tiny build
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_content/build.py` (311 (train_tok = retrain_tokenizer or not tok_path.exists()), 322 (loaded as frozen), 324-337 (reusable shared artifacts), 387-390 (vocab-size check only for a freshly trained tokenizer); ml/src/airac)
- Full finding: `docs/handoff/final-review/verify/s-build.json` (id F3); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-build`.
- **Ruling:** Write a provenance sidecar when the tokenizer is promoted (scale, sample bytes, seed, corpus version), and a scale record beside simple_vocab/known_vocab/false_facts. At --scale full, raise BuildError naming --retrain-tokenizer or another --out when the tokenizer or a reused shared artifact is not recorded as full. Apply the vocab==4096 check to reused tokenizers at full. At --scale tiny, refuse when the root holds a full-scale tokenizer or manifest. Drop 'tiny' from airace_ml.cli BUILD_DATA_HINT, and note in CLAUDE.md that a tiny build needs its own --out. Add a tiny-then-full test.

## F11: The hour-long full build has no retry, resume or source pre-flight: a Hub error when any of the 14 sources is opened, a 500/502/503 lasting over ~25 s, or a late format/config error discards all streamed work with a traceback
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_content/sources.py` (82-93 (hf_fetch; load_dataset at 92); assemble.py:338-345, 356-378, 619-636; build.py:222-232, 344-358, 385-417; cli.py:140-146)
- Full finding: `docs/handoff/final-review/verify/s-build.json` (id F11); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-build`.
- **Ruling:** Inside hf_fetch, retry load_dataset and iteration on transient errors: ConnectionError, TimeoutError, httpx.TransportError, and HfHubHTTPError with status None, 408, 429 or 5xx. Never retry not-found, 401/403/404 or SourceFormatError. Back off from 30 s, doubling, for about 8 attempts, with one flushed stderr line per retry. Resume exactly (ds.state_dict()/load_state_dict, or reopen and skip the rows already yielded) so the same seed gives the same corpus. In airace_content/cli.py, turn the final transient error into a plain 'network problem, run the same command again' line (exit 1). Before assembling, open every source the selected recipes need (plus tinystories), read a few rows and run row_has_format, so config and format errors fail within minutes. Offline tests with a fake datasets module. CLAUDE.md may add an optional 'python -m uv run --extra content hf auth login' line.

## F17: The full build prints nothing until it finishes, so a stall, a rate-limit loop or an indefinitely hung Hub metadata call (no timeout) looks exactly like progress
- Finder severity Minor; verifier severity Minor (fix size medium).
- Where: `ml/src/airace_content/cli.py` (133-149 (only prints after build_corpus returns); build.py:285-418; assemble.py (no output); sources.py:92 (load_dataset's tree listing uses huggingface_hub paginate with timeout=None))
- Full finding: `docs/handoff/final-review/verify/m-content-build.json` (id F17); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-build`.
- **Ruling:** A flushed-stderr progress callback passed through build_corpus and assemble: one line when each source opens, then every N rows (rows read, docs kept, estimated tokens), then lines for dedup, tokenizer training and writing each corpus. The CLI passes it. Test the lines on a fixture build.

## F18: CLAUDE.md's setup and corpus-build commands omit the 'content' extra, so the owner's next step (full build) fails at once; the error's own fix uses bare `uv`, and a later plain `uv sync` uninstalls `datasets` again
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `CLAUDE.md` (32-33 (ML setup: python -m uv sync), 37 (build command); hint at ml/src/airace_content/sources.py:87-91)
- Full finding: `docs/handoff/final-review/verify/m-content-build.json` (id F18); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-build`.
- **Ruling:** CLAUDE.md: setup 'cd ml && python -m uv sync --extra content' (keep the extra on every later sync); build line 'cd ml && python -m uv run --extra content airace-content build --scale {tiny,full}'; add 'cd ml && python -m uv run pytest -m gpu'. The DatasetsUnavailable hint names 'python -m uv sync --extra content'.

## F28: airace-content does not switch stdout to UTF-8 (unlike airace-ml), so the final summary crashes when stdout is redirected on Windows and the path has non-cp1252 characters, losing the exhausted-source warnings
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_content/cli.py` (147-149 (also 77-96))
- Full finding: `docs/handoff/final-review/verify/m-content-build.json` (id F28); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-build`.
- **Ruling:** airace_content/cli.py main() first switches stdout and stderr to UTF-8 with errors='replace' (reuse airace_ml.cli._use_utf8_output or move it to a shared helper). Add a test with a non-ASCII --out path through a cp1252-like stream.

## F29: Sources are streamed from an unpinned Hugging Face revision and the manifest does not record which revision was used
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_content/sources.py` (92 (load_dataset without revision); manifest.py:202-226)
- Full finding: `docs/handoff/final-review/verify/m-content-build.json` (id F29); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-content-build`.
- **Ruling:** SourceSpec gains an optional revision; hf_fetch passes revision=spec.revision; when it is None, resolve the head sha once per source (HfApi().dataset_info(...).sha) and use it for the whole read. The manifest's sources entries record the revision actually used. Offline tests with fakes.
