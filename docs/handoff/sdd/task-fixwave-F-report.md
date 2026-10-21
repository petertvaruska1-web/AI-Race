# Fix wave group F (gate and CLI): report

Branch `claude/upbeat-franklin-k1192h`. Base `815a8e55dcdaacf27cd5f4e16afdc89c1d82b1f7` (the commit just before the first `fix(F)` commit). Head `9d803cdaaf844e3616b56408baca278f8f66e4a3`. Nothing pushed. No earlier attempt of this group existed (`git log | grep 'fix(F)'` was empty, the tree was clean).

| Finding | Status | Commit |
|---|---|---|
| F8 G5 continuations replayed the starter's batches | FIXED | d1fa3a9 |
| F9 G4 had no noise floor | FIXED | 0612e56 |
| F10 G2/G5/G6/G7 were single-seed | FIXED | 377857c |
| F13 G7(a) counted every unknown word | FIXED | 5010620 |
| F19 refused rename lost a finished training | FIXED | ef634d3 |
| F20 a gate folder was overwritten by another setup | FIXED | 648656c |
| F21 silent CPU fallback | FIXED | cb930a7 |
| F38 tracebacks for damaged data | FIXED | 09b436c |
| F39 `train --resume` on a finished run | FIXED | fc0d72e |
| F41 rerun lost the Iterations notes | FIXED | 9d803cd |

All commands run from `/home/user/AI-Race/ml` with `uv run --no-sync`.

## Suite

- Default suite: `pytest -q` gave **1560 passed, 1 skipped, 9 deselected in 179.5 s** (baseline before my first commit: 1503 passed, 1 skipped in 172.4 s on this machine; the three files I touched most take 18 s together, so the difference is machine noise plus 57 new fast tests). No new warnings.
- `ruff check .`: **All checks passed**. (`ruff format --check` was already failing on 9 unrelated files before I started, minipy etc.; I formatted only files that were already formatted, and reverted an accidental `ruff format tests/` of 7 unrelated test files before committing.)
- Slow suite: `pytest -m slow -q` gave **5 passed, 1565 deselected in 188.9 s**, including `test_run_gate_quick` and `test_quick_gate_through_the_cli_reports_and_then_reuses_its_runs` (the quick gate runs 14 runs, G1 cannot pass on CPU as before, the rerun trains nothing).

## F8: G5 continuation arms train on fresh tokens (d1fa3a9)

Files: `experiments/gate.py`, `tests/test_gate.py`.

**RED** `pytest tests/test_gate.py -k g5_continuations` (fake gate, configs recorded):
```
>       assert grown.seed != starter.seed  # and not the ones the first model trained on
E       AssertionError: assert 1 != 1
```
**Fix.** `CONTINUATION_SEED_OFFSET = 10_000`; both arms use `starter seed + 10_000` (same for both, so identical fresh data and still paired). Trainer unchanged (M2 obligation recorded in the commit message and a comment: a continuation must never reuse its parent's sampler stream).
**GREEN** `-k g5_continuations`: `1 passed` (after F10 the test checks all three seeds: arms share a seed, it differs from the starter's, three distinct continuation seeds, each arm's parent is that seed's starter).

## F9: G4 measures its own noise floor (0612e56)

Files: `experiments/gate.py`, `tests/test_gate.py`.

**RED** (old gate code with the new tests; the noise is made larger than the seed difference):
```
>       assert not g4.passed and g4.data["noise_mean"] > g4.data["seed_mean"] > 0.05
E        +  where True = GateCriterion(id='G4', ..., passed=True, detail='same mix, different seeds: mean distance 0.245 (ne...
```
(plus `TypeError: eval_seed_variation() takes 2 positional arguments but 3 were given` for the evaluator tests, `AttributeError: NOISE_MEASURE_SEED`.)

**Fix, as ruled.** Each balanced model is fingerprinted again at `MEASURE_SEED + 1` (`NOISE_MEASURE_SEED`), cached in its own file `gate_fingerprint_again.json` under a key that holds the seed, recorded in `raw["fingerprints_remeasured"]`. The same-model distances (original vs second measurement), the seed pairs and the mix pairs are all computed against ONE population: the 15 first measurements plus the 3 second ones. `population_distance` (new, gate-local; `personality/` is outside my files) leaves out every trait whose population std is below `STD_FLOOR` (1e-6). `eval_seed_variation(seed, mix, noise)` passes only if `seed_mean > noise_mean` and `seed_mean < mix_mean` and `seed_mean > 0.05` (kept as the absolute sanity bound); equal is not "more". `noise_mean`, `seed_mean`, `mix_mean` (and the three distance lists) are in `data`, so in the report, and in the detail with the reason when one fails ("different seeds differ no more than the noise of measuring one model"). The module docstring row for G4 follows.
**GREEN** `-k "g4 or seed_variation or noise or nobody_varies"`: `7 passed`. The fake-gate test recomputes the expected distances from the raw traits with the 18-fingerprint population (`unit = sqrt(10)/std(first + again)`), so a population that left out the re-measures would fail it.
Existing tests changed: `test_simple_evaluators` and `test_seed_variation_edges` pass the third argument; their verdicts are the same (the `[0.05, 0.05]` and `[1.0, 1.0]` cases still fail for the same reasons, and `[0.06, 0.06]` passes where there is no noise). The fake-gate fingerprint counts are 18 (15 + 3).

## F10: G2, G5, G6, G7 take every seed (377857c)

Files: `experiments/gate.py`, `tests/test_gate.py`.

**RED** (old gate code, new tests; 8 failed, among them):
```
test_g2_g5_g6_and_g7_run_every_seed_and_g1_one          FAILED
test_a_quick_gate_runs_one_seed_and_decides_on_it_alone FAILED
test_g5_fails_when_the_gain_of_growing_is_swamped_by_the_seed_spread FAILED
test_g2_fails_when_one_seeds_first_model_does_not_answer_well        FAILED
  GateCriterion(id='G2', ..., passed=True, detail='20 of 20 replies well-formed (100%; needs 70%)' ...
```
**Fix, as ruled.**
- G2: the first model of every seed answers the 20 probes. Seeds 2 and 3 need starters, trained in G2 and named `starter-s<seed>`; the first seed's starter is G1's own `starter`. G2 passes only if every seed reaches 70%. The transcript and the per-reply column in the report are the first seed's.
- G5: per seed `g5-grown-s<seed>` and `g5-ungrown-s<seed>` (each pair on its own fresh seed, F8), plus the growth check on every seed's starter (G1's timing run is the first seed's, the others are computed in G5). Pass: worst logit difference at most 1e-4 and the mean gain (ungrown loss minus grown loss) more than `SPREAD_FACTOR` x its sample std over the seeds.
- G6: per seed, code-only children of that seed's own G3 creative-heavy model (`g6-code-replay0-s<seed>`, `g6-code-replay30-s<seed>`). The drop and what replay wins back are averaged; the spec thresholds (drop >= 3 points, recovery >= 50%) stay on the means and each mean must also exceed twice its seed spread.
- G7: per seed the four preparation runs (`g7-web-light-s<seed>`, `g7-web-thorough-s<seed>`, `g7-facts-unchecked-s<seed>`, `g7-facts-checked-s<seed>`). The 80% bar stays on the mean rates, and each fall must exceed twice its seed spread.
- G1 stays single-seed. Quick mode: one seed, which decides alone (spread 0, so the effect only has to be there).
- Per-seed numbers: every criterion's `data` has `<name>_by_seed` lists and a `seeds` list, so the report's tables show them (rendered and checked with the fake gate).
- Run names: all per-seed runs end in `-s<seed>`; only G1's `starter` and its CPU twin do not.
- Time estimate: a full gate with three seeds trains 43 runs (25 before). `run_gate`'s docstring says so ("about 75 to 105 minutes on an RTX 3050": the earlier 45 to 65 plus the ruled 30 to 40). CLAUDE.md states no estimate and neither the CLI nor the gate prints one, so there was nothing else to update; the old estimate is in `.superpowers/sdd/.../task-20-report.md` and `progress.md` (the controller's documents).

The evaluators `eval_legibility`, `eval_growth`, `eval_forgetting`, `eval_prep` now take one measurement per seed (lists). **Existing tests changed**: every call wrapped its single values in one-seed lists (same verdicts, same messages where the message did not change), `got 0` still appears in the empty-replies detail, and the fake-gate counts and run names follow (25 runs -> 43, 15 benchmarks -> 19, because G6 benchmarks 3 x 2 children).
**GREEN** `-k "every_seed or own_seeds or quick_gate_runs or swamped or does_not_answer or must_hold_on_every_seed or must_beat or legibility or growth_edges or forgetting or prep"`: `17 passed`; the whole of `tests/test_gate.py` 101 passed; `pytest -m slow tests/test_gate.py` 2 passed. The stopped-early test now has five cases, including a second seed's starter (only G2 and G5 fail) and a seed-3 G6 child plus a seed-2 G7 run.

## F13: G7(a) counts garble-like words (5010620)

Files: `experiments/gate.py`, `experiments/configs.py`, `tests/test_gate.py`.

**RED** (old code):
```
test_garble_rate FAILED: assert garble_rate(["xq zz"], {"the"}) == 0   (assert 1.0 == 0)
```
and `GateScale.__init__() got an unexpected keyword argument 'garble_samples'`.
**Fix, as ruled.** `garble_rate`: a word is garble-like if it holds a character of `airace_content.textproc.MOJIBAKE_CHARS` (the set `noise.garble()` writes from; a test checks every output of `garble()` at 20 strengths is caught), or it is not in `known_vocab`, has at least 4 letters and is within edit distance 1 of a word that is (insert, delete or substitute; implemented by trying every neighbour from the vocabulary's alphabet). Words are runs of letters and mojibake characters in the original case (the uppercase `Ã` is what marks mojibake; `words()` lowercases). `garble_rate` = garble-like / all words. Full mode `garble_samples` 200 (`GateScale.garble_samples`; quick 60, unchanged); `_web_prefixes` and `_load_inputs` take it from the scale.
Information beside the verdict: `prepared_web_text(root, tok, prep, n)` takes a seeded sample of up to `DATA_SAMPLE_DOCS` = 2000 documents of the web pool each arm may train on (`eligible_docs` under `LIGHT_CLEANING` / `THOROUGH_CLEANING`, the constants the runs now use too). `data_garble_light`, `data_garble_thorough`, `data_ratio` (None when light has nothing garble-like), `data_docs_light`, `data_docs_thorough` go into the criterion data, and a sentence into the detail; above 0.7 it adds "strengthen the web noise or the cleaning thresholds before trusting this verdict". The 0.8 bar stays. The cost of the data-level measure (synthetic 800k-word text, 30k-word vocabulary, 7% unknown): 7.4 s per arm.
**GREEN** `-k "garble or prepared_web or g7_ or web_prefixes or configs_are"`: `11 passed`.
Existing tests changed (they encoded the unknown-word share): `test_garble_rate` (`["xq zz"]` with vocab `{"the"}` is now 0, was 1), `test_garble_rate_counts_words_as_model_text_is_counted` (the pooled `4/6` and the `café naïve` `0.5` became 0 and a mojibake case), and the fake-gate completions now contain mojibake.

## F19: a refused rename is tried again (ef634d3)

Files: `evals/judge.py`, `experiments/runs.py`, `tests/test_gate.py`.

**RED** `pytest tests/test_gate.py -k rename`: `assert (1 == 7)` (one `os.replace` call, no retry) and the retry test failing with `PermissionError`.
**Fix.** `_replace(tmp, path)` in both modules: `os.replace`, and on `PermissionError` up to six more tries with sleeps 0.1, 0.2, 0.4, 0.8, 1.6, 3.2 s, then the error is raised unchanged; other errors are not retried. `_write_json` of both modules uses it. (`novelty.py` has the same two renames; it is outside this group's files.)
**GREEN** `-k "rename or refusal"`: `6 passed` (judge and gate-runs writer each: two refusals then success with sleeps `[0.1, 0.2]` and the tmp file gone; always refused -> 7 calls, six sleeps, `PermissionError`; `FileNotFoundError` -> no sleep).

## F20: a gate folder remembers its setup (648656c)

Files: `experiments/gate.py`, `cli.py` (the `--out` help), `CLAUDE.md` (gate example), `tests/test_gate.py`.

**RED** `-k gate_folder`: `FileNotFoundError` for `gate_setup.json` (never written) and no refusal.
**Fix.** After the inputs are checked (so a gate that cannot start creates nothing) and before anything is trained, `run_gate` writes `out/gate_setup.json` = `{quick, device (type), seeds}` (the effective seeds; quick keeps the first) on first use. If it exists and differs, or cannot be read, `GateSetupError` (the CLI's exit 2) says what the folder holds and what this run is, "Running here would train over its finished runs; use another --out folder." CLAUDE.md: the gate example has no `--quick`, and a second sentence gives `--quick` its own folder (`runs/gate-quick`) and says why.
**GREEN** `-k gate_folder`: `6 passed` (record written; rerun of the same setup reuses; quick, CPU, other first seed, fewer seeds each refused with nothing trained and the record and runs untouched, and a folder of its own works; an unreadable record is refused).
Note: a folder from before this fix has no record, so its first run records the setup without looking at the runs in it.

## F21: the device line and no silent CPU fallback (cb930a7)

Files: `cli.py`, `tests/test_cli.py`.

**RED** `-k "device or gpu or cpu"`: `assert (0 == 2)` for `build-judge` and for the full `gate` with no usable GPU, and `AttributeError: _announce_device`.
**Fix, as ruled.** `build-judge`, `bench` and `chat` print `using the GPU (<name>)`, `using the CPU` (when `--device cpu` or `AIRACE_DEVICE=cpu` asked for it) or `no usable GPU found: using the CPU`. The line comes after the checks that can refuse the command (the existing contract is that a command that cannot start prints nothing on stdout, and 30 existing tests said so), just before the work. `build-judge` and a full `gate` that would run on the CPU without being asked refuse with exit 2: "no usable GPU was found, so building the judge / the full gate would run on the CPU, which would take days. To run it on the CPU anyway, add --device cpu (or set AIRACE_DEVICE=cpu)." (the gate's adds that `--quick` may use the CPU). `train` is unchanged. CLI docstring and `--device` help say so.
**GREEN** `-k "device or gpu or cpu"`: `7 passed`.

## F38: damaged data is an error line (09b436c)

Files: `tokenizer.py`, `cli.py`, `tests/test_tokenizer.py`, `tests/test_cli.py`, `tests/test_gate.py`.

**RED** (before the fix): `Exception: EOF while parsing a value at line 635 column 14` out of `Tok.load` (bare `Exception`), 12 new tests failing, among them every `build-novelty-index`/`build-judge` on a half `tokens.bin` and every command on a half `tokenizer.json`.
**Fix, as ruled.** `Tok.load` re-raises a parse failure as `ValueError("the tokenizer at <path> is damaged: <why>")` (with `from e`). I also turned a file that is not there into a `FileNotFoundError` (the library raises the same bare `Exception` for that; "damaged" would be untrue and no caller could tell), checked only in the failure path. `_require_tokenizer` turns the `ValueError` into a usage error with `BUILD_DATA_HINT`. `_cmd_build_novelty_index` catches `OSError` via `_files_problem` (after `FileNotFoundError`) and `ValueError` as "the training data under <root> is damaged (<why>); rebuild it with airace-content build ..." (`_damaged_data`); `_cmd_build_judge` gets the `ValueError` catch (which covers calibrate's). The gate's `_LOAD_ERRORS` already held `ValueError`, so its damaged-data message works now (tested for the tokenizer and for a half `tokens.bin`); `Judge.load` of a judge folder with a damaged `tokenizer.json` now reaches `_creativity_tools`' "damaged or out of date ... Rebuild it with: airace-ml build-judge" (added to the existing parametrized test).
**GREEN** `-k "damaged or cut_short or novelty_folder or cannot_be_read or tokenizer_file or value_error_out"` over the three files: `60 passed`.
**Existing test changed**: `test_an_unexpected_value_error_out_of_the_judge_build_is_not_hidden` asserted that a bare `ValueError` out of `build_judge` propagates as a traceback, which is exactly what the ruling reverses; it is now `test_a_value_error_out_of_the_judge_build_is_a_plain_error_that_shows_its_reason` (exit 2, the error's own words are in the message). See concerns.

## F39: `train --resume` on a finished run (fc0d72e)

Files: `cli.py`, `tests/test_cli.py`.

**RED** `-k "resume_on or resume_with"`: the message still said "nothing saved to resume ... Run the command again without --resume to start the run." (2 of 4 failing; the other 2 are no-regression guards).
**Fix, as ruled.** When `--resume` finds no saved state and `--out` holds a `result.json` of a completed run of this config's `run_id`, or weights with their `meta.json`, it raises a usage error: the run already finished (or the folder "already holds a trained model" when no result of this run says which run made it), nothing to resume, "it is ready to use (airace-ml chat --model <out>). To train again, delete that folder or use another --out." A folder with weights alone keeps the old message. Nothing in the folder is touched (the test compares every file's bytes).
**GREEN** `-k "resume_on or resume_with"`: `4 passed`; all 8 resume tests pass.

## F41: a rerun keeps the Iterations notes (9d803cd)

Files: `experiments/report.py`, `tests/test_gate.py`.

**RED** `-k iterations`: after editing the Iterations section and writing the report again, the notes were gone (`text.rstrip().endswith(notes)` false, the placeholder back).
**Fix.** `write_gate_report` reads the target first; if it has a `## Iterations` heading (the last one) whose body is anything but empty, that body is written back instead of the placeholder; a missing, non-UTF-8 or foreign file starts from the placeholder. Everything above is regenerated. Model text cannot pose as the heading (every table cell starts with `| ` and has its `#` escaped; tested).
**GREEN** `-k "iterations or placeholder"`: `2 passed` (notes with `###` subsections and a `##` heading survive two reruns without doubling; removed notes bring the placeholder back).

## Self-review (adversarial read of the whole diff)

Found and fixed during the work:
- `ruff format tests/` reformatted 7 unrelated test files (and `test_tokenizer.py`) before a commit; I reverted all of them and re-applied only my tests. The final diff touches only the files named above plus nothing outside `ml/` except `CLAUDE.md`.
- The device line first went before the loading, which broke 30 tests of the contract "a failed command prints nothing on stdout"; moved to just before the work.
- `test_a_word_one_edit_from_a_known_word_is_a_learned_typo` first asserted wrong things about a digit inside a word (`fin3`) and a swap; rewritten and each case reasoned out.
- A `G5` edge test claimed "exactly twice the spread" with numbers that were not; now binary-exact numbers (1/16, 1/8, 3/16).

## Concerns and open points for the controller (ADR 0002 text)

1. **F10 rule as implemented.** "Decide as G3 does" is applied as: spec thresholds stay on the seed means (G6: drop >= 3 points and recovery >= 50%; G7: both rates fall to <= 80%; G5: logit diff <= 1e-4 on every seed) AND the mean effect must exceed `SPREAD_FACTOR` x its sample std over the seeds (G5: the gain; G6: the drop and what replay wins back, each; G7: each fall). G2 must pass on every seed (no spread rule). If the intended rule is only the spread rule, drop the threshold conjunct in `eval_forgetting`/`eval_prep`.
2. **F38 versus an earlier ruling.** The old test I replaced encoded "a ValueError out of build_judge is a bug and must propagate". The new ruling makes it "training data damaged ... rebuild it", so a genuine bug that raises `ValueError` inside `build_judge` is now reported as exit 2 with the bug's own message and the rebuild hint (nothing is hidden: the message is in the line).
3. **F13 edit distance** is the literal one (insert, delete, substitute): an adjacent swap (about a quarter of the noise module's typos) is two edits and not counted. The mojibake set includes the lowercase `â`, `º`, `ª` and the degree sign, so a legitimate French or Spanish word with them counts as garble-like; rare in English web text and equal in both arms.
4. **F13 data-level ratio** is only computed in the full-scale sense on the real corpora; on the real data run it is the number to read before trusting G7(a) (above about 0.7: strengthen web noise or thresholds). It adds about 15 s at the end of the gate.
5. **F9** needs the real run to say whether the proxy finding (seed effect about equal to the noise at k=3) holds; if G4 fails for exactly that reason, raising `fingerprint_k` in `FULL_SCALE` is the next lever (the noise floor scales down with k).
6. **F20** records the effective seeds, so `--seeds 1,2` after `--seeds 1,2,3` is also refused (a different gate by the ruling's wording).
7. Quick mode: G4 still fails (one balanced model, no seed pairs) and G3 fails for one seed per mix, as before.

## Other observations (not fixed, outside the notes)

- `evals/novelty.py` writes the index and its record with two unretried `os.replace` calls (the F19 hazard, smaller: the index build is minutes, not an hour).
- G6's continuations keep the parent's seed with a different mixture (code only, 30% replay of the parent's mix), so they do not replay the parent's batches the way G5 did; the replay part draws old-mix documents on purpose. The M2 obligation of F8 (the trainer deriving the sampler seed from the lineage position) is still open.
- `docs/handoff/` and `.superpowers/sdd/.../progress.md` still say the real gate takes 45 to 65 minutes (24 GPU runs and the CPU twin). It is now 42 GPU runs plus the CPU twin: roughly 75 to 105 minutes. The controller owns those documents.
- The CLI's `--device` default means `build-judge` on the cloud Linux session (no GPU) now needs `--device cpu`; CLAUDE.md's cloud line says tests only, so nothing there breaks.
