# Final-review fix wave: plan (draft of 2026-10-14, written before the verdicts)

**Status: tentative.**
- Confirm each item against its verdict in `verify/verdicts-<group>.json`.
- The controller re-checks the verifier's reasoning on every refuted serious finding (F1–F15) before dropping it. A refuted minor is dropped with one ledger line.
- The ledger records each decision below as a `Ruling:` line with its cost if wrong.

## Model tiers (owner suggestion; ruling in the ledger)
- **Controller (top model):**
  - rulings and measurement-rule decisions
  - ADR text
  - re-checking refuted serious findings
- **Sonnet:**
  - every implementer, working from a precise controller-written brief
  - the scoped re-review
  - Escalate to the top model only if a Sonnet worker reports BLOCKED, or the re-review shows its work is wrong.
- **Haiku:** pure copy jobs.
- Ultracode is off. Use Agent-tool agents, not the Workflow tool, unless the owner opts in again.

## Process
1. **One fix wave = six sequential Sonnet implementer dispatches**, groups A–F below, in this order:
   - A, B and C are independent.
   - D comes before F (G5 relies on F8's trainer change).
   - E comes before F (F4 moves the shared data-identity helper that the gate then uses).
2. **Each dispatch gets the controller's dispatch notes:** the group's verdicts plus the rulings below.
3. **Each implementer:**
   - Writes a failing test that reproduces each bug first. If it cannot reproduce one, it reports `NOT REPRODUCED` with evidence and leaves that code alone.
   - Makes one commit per finding or per small cluster.
   - Runs the full default suite plus `ruff` before reporting.
   - Cloud commands: `cd ml && uv run --no-sync pytest`; never `uv sync`.
   - Must not touch `docs/decisions/*`; the controller writes the ADR text. CLAUDE.md edits are allowed only where a group's items require them (C: build commands; F: gate `--out` guidance).
4. **After each group,** the controller reads the report and checks the test count, writes the ledger lines, and syncs.
5. **After all groups,** run one scoped re-review: one Sonnet agent per group, adversarial, over that group's commits. Rule on any residuals.
6. **Then the controller:**
   - Writes the ADR text: addenda to `0001-bench-v1-measurement-rules.md`, and a new `0002-feasibility-gate-measurement-rules.md`.
   - Updates `docs/progress.md` and `docs/handoff/README.md`.
   - Syncs and pushes.
   - Sends the final owner message with every ruling and its cost if wrong.

## A: bench items, before bench-v1 freezes (F1, F5, F24, F44, F49, F53–F56, then F25 last)
- **F1 (Critical).** Function-completion prompts end at the docstring's closing quotes, without the trailing newline: `CheckItem.prompt = v.prompt.rstrip("\n")`, with `check_args` keeping the full prompt.
  - `function_body` must accept a reply that starts with a newline.
  - Update `fakes.answer_key_lm`.
  - Tests:
    - no non-chat prompt ends in whitespace
    - prompt tokens are a prefix of prompt+reference tokens under a BPE trained with the build's pre-tokenizer
  - Training docs are unchanged, so no rebuild is needed.
  - ADR 0001 addendum: deviation from plan line 1176.
- **F5.** Knowledge and consistency: within each relation, draw the *answer object* uniformly (without replacement), then a subject with that object. A subject-blind rule then gets exactly 1/4 on MC.
  - Exact items use a measured chance floor: the best per-relation constant reply rate, like the instruction floor.
  - Regression test: a relation-prior ("always the most common answer") model normalizes to about 0 on knowledge and consistency.
  - This changes the bench/train fact partition, which is fine before Task 15's build.
- **F24.** Knowledge exact items: `max_new_tokens = max(8, 4 + len(answer)//2)`.
- **F49.** Every item gets `topic:`, `skill:` and `fmt:` tags. They are metadata only; scores don't change.
- **F53.** Plural class answers (mammals, birds, …).
- **F54.** Bare digits for ordinals.
- **F55.** `never_false` for every bird → reptile pair.
- **F56.** Drop the hen → chick row.
- **F44.** Make the two vacuous tests meaningful: non-empty tag breakdown; multi-letter dataset names matched by regex.
- **F25 (last, after every item change).** A digest test over canonical `build_suite()` content. Its message says that any intended change must bump `SUITE_VERSION`.

## B: content data quality (F12, F14, F26, F27, F31, F45, F52) and quota sizing (F16, F30)
- **F12.** Convert the real sedthh Gutenberg layout back to plain-text layout before chunking:
  - CRLF becomes LF.
  - When exactly-double newlines outnumber lone newlines, a lone double newline becomes a single newline, and a run of three or more becomes a double newline.
  - Add a fixture row in the real layout (Alice excerpt), and a test that no passage has a blank line between a line without an end mark and a lowercase line.
- **F14.** Repair cp1252 mojibake in *source* text before normalization, never in deliberately injected garble noise:
  - `â€œ`→“, `â€™`→’, `â€˜`→‘, `â€“`→–, `â€”`→—, and any remaining `â€`→”.
  - Add a fixture row and a test.
- **F26.** No misspellings on copies of structured datasets (code; check reasoning and patterns too), so programs stay runnable. Correct the `build.py` comment.
- **F27.** Add a page-furniture factor to `quality_score`, covering lines with 2+ `|` separators and short title-case navigation lines. Alternatively, make boilerplate graded like typos. Either way, test separation at realistic lengths (400–800 words).
- **F31.** One word rule: `textproc.words` becomes Unicode-aware (letter runs incl. accents, inner apostrophes, curly-apostrophe handling kept), so `known_vocab.txt` matches the gate's garble reader.
- **F52.** Calendar words leave the "school" topic. Drop or qualify `addition`, `number`, `place`, `data`, `practice`, `goals`, `training`. Add `marketing` to `_NOT_INFLECTED`.
- **F16 + F30.** Quotas are always sized with the deterministic byte estimate, never the frozen tokenizer, so repeated builds agree.
  - Per-source bytes/token constants: prose ≈3.2, code and patterns ≈2.7, creative ≈3.6.
  - Use a running sum over all accepted docs.
  - Warn when a component ends below 90% of its quota.
  - Fix the docstring.
- **F45.** Non-ASCII fixture rows (accents, emoji, CJK, curly quotes) run end to end through build, vocab files and manifest.

## C: content build operations (F3, F11, F17, F18, F28, F29)
- **F3.** Tokenizer and shared-vocab provenance sidecar recording scale, sample bytes, seed and corpus version.
  - `--scale full` refuses a tokenizer or shared artifacts made at another scale or with no record. It raises a plain BuildError naming `--retrain-tokenizer` or another `--out`.
  - The `VOCAB_SIZE` check also applies to reused tokenizers.
  - A tiny build refuses to overwrite a full corpus.
  - `BUILD_DATA_HINT` stops suggesting a tiny build into the default root.
  - Add a tiny-then-full test.
- **F11.** Retry and exact resume inside `hf_fetch`, using the datasets `state_dict`/`load_state_dict`.
  - Transient: ConnectionError, TimeoutError, httpx transport errors, and HfHubHTTPError with status None or 408/429/5xx.
  - Never retry not-found, 401/403/404 or format errors.
  - Back off 30 s, 60 s, 120 s…, about 8 attempts over about 30 minutes, with one flushed stderr line per retry.
  - Final failure → a plain "network problem, run the same command again" line.
  - Pre-flight before assembling: open every needed source and check a few rows.
  - Test offline with a fake `datasets` module.
- **F17.** Flushed stderr progress lines per stage and source: opening, rows/docs/estimated tokens every N rows, dedup, tokenizer, writes.
- **F18.** CLAUDE.md:
  - `python -m uv sync --extra content`
  - `python -m uv run --extra content airace-content build --scale {tiny,full}`
  - the GPU test line `python -m uv run pytest -m gpu`
  - The DatasetsUnavailable hint uses `python -m uv sync --extra content`.
- **F28.** UTF-8 stdout/stderr in `airace-content` (reuse `_use_utf8_output`).
- **F29.** `SourceSpec.revision`, passed to `load_dataset` and written to the manifest. Task 15 fills in the verified commit shas.

## D: trainer (F2, F7, F8, F22, F23, F47, F48)
- **F2.** Gradient accumulation in `_train_step`, with micro-batches capped at about 8192 tokens.
  - Scale the loss by rows_in_micro/batch_size.
  - Keep one clip, one spike check and one optimizer step per step.
  - Drop `logits` before backward.
  - The sampler draws the same rows, so resume equivalence holds.
  - Add an `@pytest.mark.gpu` test: one step of `judge_train_config()` on CUDA with `max_memory_allocated` well under 8 GiB.
- **F7.** Replay drops the custom datasets (notebook, coaching) from the parent's mixture and renormalizes the rest; the run summary records what was dropped. Tests: a notebook parent, then a replay child without a notebook, then a grandchild.
- **F8.** A continuation run's sampler stream is derived from (seed, the parent's total tokens trained), so "train more" draws fresh data. G5's two arms stay paired, because they share the parent.
- **F22.** `validate()` requires `batch_tokens % ctx_len == 0`. Steps come from the real tokens per step.
- **F23.** `validate()` rejects notebook, coaching and probe strings that don't round-trip UTF-8. Probe prompts are encoded once in `_prepare`.
- **F47.** Three resume-branch tests:
  - resume inside the finishing phase
  - resume after a rollback at the stop step
  - a monkeypatched refused rename
- **F48.** A module-scoped reference-run fixture (copytree per test), folding the two near-identical resume tests together.

## E: evals (F4, F6, F33, F34)
- **F4.** Move the gate's `_data_identity` to a shared helper. The judge's `train_config.json` and the novelty index's `.json` record it at build time.
  - `build_judge` retrains when the identity differs, or when the reused checkpoint fails to load.
  - The gate and `bench` refuse stale instruments with the existing "out of date … rebuild" message. The identity goes into the bench cache key.
  - A quick-budget judge is labelled as a test judge and refused as the reference.
- **F6.** Creativity:
  - Diversity becomes a real multiplier, `min(1, d/d_ref)`, with d_ref the distinct-2 of 24 held-out creative openings at calibration.
  - Add a per-story length factor.
  - Windows shared with another story in the same run count as not new.
  - Tests: a constant reply and a constant short reply score about 0. Replace the constant "novel" exemplar with per-prompt replies.
  - ADR 0001 addendum.
- **F33.** NaN-safe sampling: a row with no finite logit stops, with no text emitted.
- **F34.** Fix the docstrings and define the 96-token opening length once.

## F: gate and CLI (F9, F10, F13, F19, F20, F21, F35–F43, F46)
- **F9.** G4 measures its own noise floor.
  - Re-fingerprint the three balanced models at MEASURE_SEED+1.
  - Pass requires seed_mean > noise_mean (one population including the re-measures) and mix_mean > seed_mean.
  - Traits that are constant across the population are excluded from the distance, so there is no 1e-6 blow-up.
  - ADR 0002.
- **F10.** Follow spec §4.12's "3 seeds each" for G2, G5, G6 and G7. G1 is a timing and stays single-seed; record that.
  - G5, G6 and G7 decide like G3: the mean effect must exceed SPREAD_FACTOR × seed std.
  - G2 must pass on every seed.
  - About 12 extra runs, roughly 30–40 GPU minutes more.
- **F13 + F36.** G7(a) measures the model's learned-noise signature instead of all unknown words: unknown words within edit distance 1 of a known word of length ≥4, plus words carrying mojibake marks.
  - The thorough arm's word volume must be ≥ 0.8 × light's, with looped 3-grams excluded.
  - The ≤ 0.8 ratio stays.
  - ADR 0002.
  - A real-data FAIL after this is a legitimate step-8 signal (strengthen noise or adjust cleaning), not a measurement bug.
- **F35.** G2 counts a reply as well-formed only if it is neither a near-copy of its prompt nor a near-duplicate of another reply. The report shows all 20 exchanges.
- **F19.** Retry `os.replace` in the judge and gate record writers. Recover "completed" from the trainer's outputs when the final record is missing.
- **F20.** Refuse when `--out` holds finished runs of another scale, device or seed set. `gate` refuses a silent CPU fallback. CLAUDE.md: use a separate `--out` for `--quick`.
- **F21.** Print the device in plain words. `build-judge` and `gate` refuse the CPU unless `--device cpu` is given.
- **F37.** Reuse `unstable_stopped` runs whose key matches.
- **F38.** No tracebacks for:
  - a damaged corpus
  - an OSError in `build-novelty-index`
  - a corrupt tokenizer.json (re-raise as ValueError naming the file)
- **F39.** `--resume` on a finished run says it is finished and ready to use.
- **F40.** Escape C0/C1 control characters at print time only; stored text is unchanged (R1).
- **F41.** Carry an existing Iterations section across report rewrites.
- **F42.** `test_build_judge_smoke` works on a copy of the root.
- **F43.** The GPU parity test uses tolerances.
- **F46.** Data-identity invalidation tests.

## Deferred with rulings (no code in this wave)
- **F15 (simplicity thresholds):** calibrate on real data during Task 15, *before* the full build.
  - Measure per-source simplicity quantiles on the first ~2000 rows with the real `simple_vocab`, pick a keep-rate, and set `simplicity_min`.
  - Record the numbers. The finding has the cloud measurements: 0.6 keeps 92% of FineWeb, 0.7 keeps 79%.
- **F32 (notebook novelty, coaching vs bench screening):** an M2 obligation, recorded in the handoff.
- **F50 (mini-eval during training for live charts):** deferred to M2, where the live-training UI lives.
- **F51:** done on 2026-10-14.
