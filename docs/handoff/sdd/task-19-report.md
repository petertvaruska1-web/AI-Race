# Task 19 report: `airace-ml` CLI

**Status:** DONE_WITH_CONCERNS (design decisions to confirm, listed under Concerns; nothing is blocked)
**Commit:** `cedcfa4` feat(ml): airace-ml CLI for train, chat, bench, fingerprint, judge and index
**Branch:** current branch, not pushed.

## What I implemented

`ml/src/airace_ml/cli.py` (new) and `ml/tests/test_cli.py` (new). `pyproject.toml` already had the script; I did not touch it.

`main(argv) -> int` never calls `sys.exit` and never raises for an expected problem. argparse's own exits are turned into return values by a small `_Parser` subclass (`error()` raises an internal usage error, `exit()` raises an internal exit marker). Each subcommand is a small `_cmd_*` function.

Subcommands, all with `--data-root PATH` (default `data_root()`) and `--device {cpu,cuda}` (default `pick_device()`):

| Command | Behaviour |
|---|---|
| `train --config FILE --out DIR [--parent DIR] [--resume]` | Prints a one-line header, then `step {s}/{n}  loss {l:.3f}  lr {lr:.2e}` per Progress event, `  heldout [step N] name loss ...` per HeldoutEval, `  sample [{prompt}]: {text}` per Sample, `  warning [step N]: ...` per Instability, then a summary (status, run, steps, tokens, final loss, time, out dir). `train_run` writes `result.json`. `--parent` overrides the file's `parent_dir` before validation. `--resume` is passed through. |
| `chat --model DIR [--temperature F] [--max-tokens N]` | REPL on stdin, `> ` prompt on stdout. `/reset` clears history and restarts the seed counter, `/raw` toggles `lm.complete`, `/quit` or EOF exits 0. Calls `chat_reply(history, max_new_tokens, temperature, seed=turn_index)`. Typed text, including literal `<\|end\|>`, goes through unchanged. |
| `bench --model DIR [--out FILE] [--no-creativity] [--max-items N]` | Loads judge + novelty index when both exist; otherwise prints a note naming what is missing and the command that builds it, and creativity lands in `missing`. `--no-creativity` drops the category (so it is not reported missing). Prints category/score/items table, overall, not-measured line, time. `--out` writes `report.to_dict()` as strict JSON (UTF-8, parent dirs created, non-finite floats become null). |
| `fingerprint --model DIR [--k N] [--seed N]` | One trait per line, `TRAITS` order. |
| `build-novelty-index` | Calls `build_novelty_index`, prints the path. |
| `build-judge [--budget-tokens N]` | Same progress lines as `train`, then `reference judge ready in <dir>`. |
| `gate --out DIR [--seeds 1,2,3] [--cpu-speed] [--quick]` | Arguments parsed and validated now (`--seeds` must be whole numbers). Exits 2 with `error: gate not available yet: ...`. Task 20 swaps the body of `_cmd_gate` (marked with a comment). |

Exit codes follow ruling 2: 0 success; 2 usage/config/missing model, tokenizer or data/gate unavailable; 1 `JudgeBuildError`; 130 KeyboardInterrupt (message on stderr; for train it says to rerun with `--resume`, for build-judge to rerun the same command). Every handled error is one `error: <message>` line on stderr, no traceback.

UTF-8 (ruling 1): at startup stdout and stderr are reconfigured to `utf-8, errors="replace"` only if the stream has `reconfigure`.

Plain-language errors (examples of real output):
```
error: no trained model found in /nonexistent (it should hold model.safetensors and meta.json). Train one first with: airace-ml train --config FILE --out /nonexistent
error: cannot find the tokenizer at /empty/tokenizer/tok-v1/tokenizer.json. Build the training data first with airace-content build --scale tiny (a quick test set) or --scale full, or point --data-root at the folder that holds it.
error: the following arguments are required: --config, --out (run 'airace-ml train --help' to see the options)
error: gate not available yet: the feasibility gate has not been installed.
```

## TDD evidence

**RED** (brief's two tests, verbatim apart from ruff formatting; command `cd ml && uv run --no-sync pytest tests/test_cli.py -v`):
```
ImportError while importing test module '/home/user/AI-Race/ml/tests/test_cli.py'.
tests/test_cli.py:5: in <module>
    from airace_ml.cli import main
E   ModuleNotFoundError: No module named 'airace_ml.cli'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```
Expected: the module did not exist yet. (Ruff changes to the brief's tests: `ruff format` and import-order fix only; no assertion touched.)

**GREEN** after implementing: `tests/test_cli.py`: `2 passed`, then after adding the ruling-10 and error-path tests: `48 passed in 6.42s`.

**Mutation checks** (to prove the new tests have teeth; each was restored afterwards, `diff` confirmed identical):
| Mutation | Result |
|---|---|
| remove stdout `reconfigure` call from `main` | cp1252 end-to-end test fails with `UnicodeEncodeError: 'charmap' codec can't encode '\U0001f642'` |
| `errors="strict"` instead of `"replace"` | lone-surrogate test fails with `UnicodeEncodeError ... surrogates not allowed` |
| remove stdin reconfigure | piped-input test fails with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81` |
| `/reset` does not clear history | `/reset` test fails (history assertion) |
| `--parent` ignored | `--parent` test fails (config's missing parent is reported) |
| `resume=False` always | `--resume` test fails (`step 1/6` reappears) |

## Tests (48 in `test_cli.py`, 6.4 s)

Brief's two: `test_train_bench_chat`, `test_bad_config_exit_2`.
Ruling 10 cases: chat `/reset` (history, seeds, option pass-through, literal `<|end|>` unchanged), non-ASCII through cp1252 (`TextIOWrapper(BytesIO, encoding="cp1252")` for stdout and stderr; shows the stream raises without reconfigure; lone surrogate prints as `?`; non-ASCII path in an error message; train sample lines with accents/emoji/CJK), `--resume` (interrupted via `TrainHooks(stop_after_steps=3)`, then CLI output has `step 6/6` and no `step 1/6`; and no-state resume is exit 2), `gate` returns 2, `fingerprint` runs (all traits, in order, finite), missing tokenizer returns 2 (parametrized over train, chat, bench, fingerprint, build-judge), bench with creativity (real tiny judge and novelty index built through the CLI in a module-scoped copy of the tiny root, judge recipe shrunk to 2x64 as the judge tests do; creativity appears in the table) and bench without them (note printed, `missing == ["creativity"]`).
Extras: shared module-scoped trained model; train output format regexes; `--parent` lineage; six bad-config shapes including a BOM; missing config / parent / corpus; train KeyboardInterrupt (130, `--resume` hint) and chat KeyboardInterrupt; model/tokenizer vocabulary mismatch; `--device cuda` without GPU (skipped on a GPU box); usage errors and `--help` return instead of exiting; chat raw toggle, blank lines, typo'd command, EOF, empty reply; piped cp1252-labelled stdin carrying UTF-8 plus a stray byte; `build-judge` budget pass-through and progress; `JudgeBuildError` exits 1; `--seeds` validation; `build-novelty-index` without a corpus.

No `@pytest.mark.slow` test was needed: everything, including the real tiny judge build, fits in the default budget, so there was nothing heavy to run once.

## Full suite and lint (final tree)

```
cd ml && uv run --no-sync pytest -q
1101 passed, 1 skipped, 6 deselected in 121.89s (0:02:01)      (wall 2m06s)
uv run --no-sync ruff check .
All checks passed!
uv run --no-sync airace-ml --help      -> usage: airace-ml [-h] command ...   (exit 0)
```
The suite was ~115 s before this task; `test_cli.py` adds about 6.4 s. The one skip is an existing CUDA/tokenizer-dependent test, not mine. My `--device cuda` test runs on this CPU-only machine.

## Sample output (tiny data, CPU; paths shortened)

`train` (20 steps, 2x64x64 model, creative + conversations):
```
training demo: 20 steps, 20,480 tokens, on cpu, into <scratch>/m
step 1/20  loss 6.255  lr 2.68e-03
step 10/20  loss 5.470  lr 2.68e-03
step 20/20  loss 5.072  lr 2.68e-04
  heldout [step 20] web 5.350  books 5.155  educational 5.203  conversations 5.046  code 6.105  reasoning 5.749  facts 5.316  creative 5.538
  sample [Once upon a time]: o c hy p youor pve s fce isuet t ite par a pa? The, toitesenp
  sample [Hello! How are you?]: o f withrer isle iturnohakbr ted yousw a theoringteredy.opes cen
  sample [The capital of France is]: lrheied aiin yingarellangk cinar in, andigh s aboutar w ped a,len

training finished
  run         demo
  steps       20
  tokens      20,480
  final loss  5.197
  time        1.9 s
  saved to    <scratch>/m
```

`bench` (no judge/index built yet, so creativity is left out with a note):
```
note: creativity was left out of this run. Still to build:
  - the reference judge: build it with 'airace-ml build-judge'
  - the novelty index: build it with 'airace-ml build-novelty-index'
category         score  items
language          33.3      3
reasoning          0.0      3
pattern            0.0      3
knowledge         11.1      3
coding             0.0      3
consistency        0.0      1
instruction        0.0      3
overall            6.3
(score: 0 is no better than guessing, 100 is perfect)
not measured: creativity
took 1.0 s
```

`chat` (untrained-ish model; input `What is a cat?`, `Tell me more`, `/reset`, `/raw`, `Once upon`, `/quit`):
```
Talking to the model in <scratch>/m. Type /reset to start the conversation over, /raw to switch between chat and plain text continuation, /quit to leave.
>  to and w e l ailyow ander d wwpy to p the l
>  of" the gffes al are the,,p the andightowhid
> (conversation cleared)
> (raw mode on: the model now continues what you type as plain text)
> TheW is ofeseslyo it wr p theoren lit m w and
>
```

`fingerprint --k 1` prints ten rows, e.g. `verbosity 13.733`, `confidence 0.024`, ... `repetitiveness 0.000`.

Real-process check of Review Focus 5: `printf 'héllo 🙂 <|end|>\nWhat is a cat?\n/quit\n' | PYTHONIOENCODING=cp1252 airace-ml chat ...` exits 0 and writes valid UTF-8 (the model's U+FFFD output shows as bytes `ef bf bd`). The same `print` of an emoji under `PYTHONIOENCODING=cp1252` in plain Python raises `UnicodeEncodeError`.

## Files changed
- `/home/user/AI-Race/ml/src/airace_ml/cli.py` (new)
- `/home/user/AI-Race/ml/tests/test_cli.py` (new)

I briefly ran `ruff format src tests` (a mistake: it reformatted 7 files from earlier tasks) and reverted exactly those with `git checkout --`. The commit contains only the two files above, and the full suite was re-run on the clean tree.

## Self-review
- Completeness: every subcommand, option and ruling 1-10 item is covered. `missing` is printed only when non-empty (ruling 5 says "plus overall and missing"; an always-printed "nothing" line seemed like noise).
- `cli.py` is ~600 lines after formatting, mostly argparse wiring and plain-language messages; one responsibility, no restructuring needed.
- Tests drive real training/inference for the brief's cases and for the shared model; fakes (`chat_reply`, `complete`, `build_judge`, `train_run`) are used only where the thing under test is the CLI's own wiring (history handling, printing, exit codes), never to stand in for a model result.
- No `exec`/`eval`; all file I/O has explicit encoding (`utf-8`, and `utf-8-sig` for reading the config so a Windows editor's BOM does not break it).

## Concerns / decisions for the controller to confirm
1. **`--device cuda` without a GPU is exit 2**, not a silent CPU fallback. `pick_device("cuda")` would quietly return the CPU; for a multi-hour run on the owner's PC an explicit request should be honored or refused. Default (no flag) still falls back to CPU.
2. **A run that ends `unstable_stopped` exits 0.** The model was saved (last good weights) and the summary says "training stopped early"; `result.json` has the status. Ruling 2 reserves 1 for failures that are not the user's input and lists only `JudgeBuildError`. `build-judge` does exit 1 in that case (it raises).
3. **Additions beyond the rulings:** (a) for `chat`, piped/redirected (non-tty) stdin is reconfigured to UTF-8 with `errors="replace"` so a stray byte cannot crash it (ruling 1 only named stdout/stderr); (b) unknown `/word` commands in chat print a hint instead of being sent to the model (a literal `/word` is still sendable by adding any other character or space); (c) an empty reply prints `(the model said nothing)` as a CLI annotation while the raw empty reply stays in the history, so model output is never rewritten (R1).
4. **Docs not touched:** CLAUDE.md's "Commands" section says more commands are added as they exist, and `docs/progress.md` needs a Task 19 entry. I left both to the controller since the brief's file list is `cli.py` and `test_cli.py`. Suggested CLAUDE.md lines: `cd ml && python -m uv run airace-ml --help` and the seven subcommands above.
5. **Task 20 hook:** `_cmd_gate` currently raises the "not available" usage error; the replacement is a call to `experiments.gate.run_gate` using the already-parsed `args.out`, `args.seeds`, `args.cpu_speed`, `args.quick`. A `gate` test asserting exit 2 will need to change at that point.

---

# Fix round 1

**Commits:** `e489538` fix(ml): CLI keeps an unfinished run safe and turns bad paths, damaged files and bad budgets into plain errors; then a small follow-up test commit (see the last note below).
**Files:** `ml/src/airace_ml/cli.py`, `ml/tests/test_cli.py` only. Not pushed.
**Not done, as instructed:** M3 (narrowing the broad `except ValueError` around `train_run`).

## TDD evidence

RED (tests written first; `cd ml && uv run --no-sync pytest tests/test_cli.py -q`): `26 failed, 55 passed`. I checked each failure was for the intended reason before implementing, e.g.:
```
test_train_refuses_to_start_over_an_unfinished_run      assert (0 == 2)            (it silently restarted)
test_an_output_path_that_is_a_file...[inside-a-file]    NotADirectoryError: [Errno 20] Not a directory: '.../taken/run'
test_a_report_path_that_is_a_folder...                  AssertionError: the benchmark must not start
test_a_report_that_cannot_be_written...                 PermissionError: [Errno 13] Permission denied: '.../bench.json'
..._damaged...[_damage_index_record]                    ValueError: ... holds 1953 hashes but index.json records a count of 1954 ... build the index again
..._damaged...[_damage_judge_weights], model garbage    SafetensorError: Error while deserializing header: header too large
..._damaged_model_file...[truncated-*]                  SafetensorError: Error while deserializing header: invalid header length
test_a_judge_budget_below_one_step...                   AssertionError: no training may start
test_a_judge_build_that_hits_a_config_error             ValueError: token_budget (5) must be at least batch_tokens (32768)
test_interrupting_train_before_any_save...              hint said "...again with --resume" (wrong: nothing was saved)
test_options_with_a_minimum...[bench --max-items 0]     accepted
test_build_judge_says_when_it_moves_on_to_calibrating   StopIteration (no "calibrating" line)
```
GREEN: `tests/test_cli.py`: `82 passed in 9.2s` (was 48). Full suite on the final tree: `1135 passed, 1 skipped, 6 deselected in 121.37s (0:02:01)`, wall 2m06. `uv run --no-sync ruff check .` -> `All checks passed!`. `uv run --no-sync airace-ml --help` works.

Mutation checks (each reverted, `diff` confirmed identical afterwards): disabling the unfinished-run refusal, the judge-budget check, the report-path check, `SafetensorError` or `TypeError` in the damaged-file tuple, `--max-items` minimum, the lazy header, the calibrating line, the accurate interrupt hint, or the output-folder check each makes the covering test fail (10 mutations, 10 caught; an 11th, dropping `EOFError`, is caught by the zero-byte-index case added afterwards).

## What changed, per finding

**I1 (silent loss of resume state).**
- `_cmd_train` now asks `can_resume(args.out, cfg)` before training (wrapped so a broken state counts as "nothing saved"). Saved progress and no `--resume`: exit 2, `error: <out> holds an unfinished run of this config; add --resume to carry on, or delete the folder to start over`. Nothing is touched; `resume/` is intact.
- `--resume` with nothing saved: exit 2, `there is nothing saved to resume in <out> for this config (a run saves its progress every 200 steps, so one stopped sooner has none). Run the command again without --resume to start the run.` The folder is not created.
- The interrupt hint is now computed after Ctrl-C from what is actually on disk (new `_Interrupted` exception carries it to `main`, still exit 130): with saved progress it says to run again with `--resume`; without, it says no progress had been saved yet (every 200 steps) and to run again without `--resume`. `build-judge`'s fixed hint, which had the same inaccuracy, now says it carries on from the last saved point if there was one and starts over if not. The 200 is one constant, `CHECKPOINT_EVERY`, also passed to `train_run(checkpoint_every=...)`, so the message and behavior cannot drift.
- Covering tests: `test_train_refuses_to_start_over_an_unfinished_run` (refused, `resume/` file list unchanged, no output, then the suggested `--resume` works and skips step 1), `test_train_resume_without_a_run_to_resume_says_to_run_without_resume`, `test_train_into_a_folder_with_a_finished_run_is_not_refused` (guards against over-refusing), `test_interrupting_train_after_a_save_says_to_add_resume`, `test_interrupting_train_before_any_save_says_to_run_again_without_resume`.

**I2 (`build-judge --budget-tokens` below one step).** `_cmd_build_judge` compares the budget with `judge_module.judge_train_config().batch_tokens` (looked up through the module, so the tests that shrink the recipe still see their own value) before anything else. Real output: `error: --budget-tokens 2048 is too small: the judge learns in steps of 32,768 tokens, so the budget must be at least 32,768.` exit 2. As a second line of defense, a `ValueError` out of `build_judge` is also a plain exit-2 error. Tests: `test_a_judge_budget_below_one_step_of_the_real_recipe_is_refused_up_front` (real recipe, `build_judge` replaced by a function that fails the test if called, so it proves nothing starts), `test_a_judge_budget_of_exactly_one_step_is_accepted`, `test_a_judge_build_that_hits_a_config_error_is_a_plain_error`. One older test used a budget of 1234 with a fake builder; it now uses 65536 because 1234 is (correctly) refused.

**I3 (tracebacks).**
- `train --out <file>` and `--out <inside a file>`: `_require_folder_path` checks the path and every parent before training; message names the offending file. Any other `OSError` out of `train_run` is mapped by `_os_problem`: a path problem (permission, not-a-directory, is-a-directory, exists) is exit 2, anything else (full disk) is exit 1.
- `bench --out <existing folder>` / inside a file: `_require_report_path` runs before the model loads or the benchmark starts. The write itself is wrapped, so a failure (permission) prints the table and then `error: cannot write the report to <file>: Permission denied`, exit 2.
- Stale or half-saved novelty index, bad judge dir (`ValueError`, `KeyError`, `TypeError`, `EOFError`, `SafetensorError`, other `OSError` that is not "not found"): `_creativity_tools` now raises `error: the novelty index at <path> is damaged or out of date (<reason>). Rebuild it with: airace-ml build-novelty-index` or the same for the judge with `airace-ml build-judge`, exit 2, before the benchmark runs. A merely missing judge or index still gives the note and exit 0. `--no-creativity` never touches them.
- Corrupt or truncated `model.safetensors` (`SafetensorError`): `_load_lm` catches the shared `_DAMAGED` tuple and says `cannot load the model in <dir> (<reason>). Is it a folder made by 'airace-ml train', and is the file complete?`, exit 2, for `chat`, `bench` and `fingerprint`.
- Tests: `test_an_output_path_that_is_a_file_is_a_plain_error[file|inside-a-file]`, `test_a_report_path_that_is_a_folder_is_refused_before_the_benchmark`, `test_a_report_path_inside_a_file_is_refused_before_the_benchmark`, `test_a_report_that_cannot_be_written_is_a_plain_error_after_the_table`, `test_a_damaged_judge_or_index_is_an_error_that_names_the_rebuild` (6 damages: index record count, index array garbage, index array zero bytes, judge calibration `{}`, judge calibration half a file, judge weights; each also checks `--no-creativity` still works), `test_a_damaged_model_file_is_a_plain_error` (garbage and truncated, for chat, bench, fingerprint).

**M1 (unstable stop).** Test: `test_a_run_that_goes_unstable_stops_early_and_says_what_to_try` (`TrainHooks(loss_override=...)` through a wrapped `train_run`; exit 0; three warning lines; `result.json` status `unstable_stopped`, 4 steps). The summary now reads: `training stopped early because it became unstable, after 4 of 6 steps.` / `The last good weights were kept, so the model can still be used.` / `To try again, use a more careful learning style: lower boldness in the config, which also makes the learning steps smaller.` (There is no separate learning-rate field in the config; boldness is the real knob and it scales the peak learning rate, so the message names it rather than inventing one.)

**M2 (header).** The `training X: N steps...` line now prints just before the first training event, via `_announcing`, which comes after the data has loaded and the model is set up. A missing corpus prints nothing but the error (test asserts stdout is empty). Trade-off: the header no longer appears during data loading, so on a full-size corpus there is a silent wait before it; `train_run` has no hook between setup and the first step to do better without editing the trainer.

**M4 / M5.** `_whole_number(minimum)` replaces `_positive_int` and `_non_negative_int`; `--max-items`, `--max-tokens`, `--k` and `--budget-tokens` need 1 or more, `--seed` 0 or more. `bench --max-items 0` is now `error: argument --max-items: '0' is not a whole number of 1 or more (run 'airace-ml bench --help' to see the options)`. Tests: `test_options_with_a_minimum_reject_values_below_it` (7 cases) and `test_a_seed_of_zero_and_a_max_items_of_one_are_fine`.

**M6 (calibrating line).** Done without touching `judge.py`: `build_judge` already passes the trainer's `Done` event to `on_event`, so the CLI's judge handler prints `training finished; now calibrating the judge on real held-out text (this takes a little while)` after a `Done("completed")`. Not printed after `unstable_stopped` (the build then fails with exit 1). Limit: when `build_judge` finds the judge already trained and goes straight to calibration, no training event occurs, so no line is printed. Tests: `test_build_judge_says_when_it_moves_on_to_calibrating` (order: last step line, then calibrating, then ready), `test_build_judge_does_not_say_calibrating_when_training_did_not_complete`, and the real tiny judge build in `test_build_judge_shows_training_progress_and_where_the_judge_is`.

## Commands and output (real entry point, tiny data)

```
$ airace-ml train --config ucfg.json --out u     # u holds a run stopped at step 3 of 6
error: .../u holds an unfinished run of this config; add --resume to carry on, or delete the folder to start over     (exit 2; u/resume still there)
$ airace-ml train ... --out u --resume
resuming unfinished: 6 steps, 6,144 tokens, on cpu, into .../u
step 6/6  loss 5.718  lr 2.68e-04                 (exit 0; no step 1)
$ airace-ml train ... --out fresh --resume
error: there is nothing saved to resume in .../fresh for this config (a run saves its progress every 200 steps, so one stopped sooner has none). Run the command again without --resume to start the run.     (exit 2)
$ airace-ml build-judge --budget-tokens 2048
error: --budget-tokens 2048 is too small: the judge learns in steps of 32,768 tokens, so the budget must be at least 32,768.     (exit 2)
$ airace-ml train --config ucfg.json --out <an existing file>
error: cannot use .../f for --out: it is a file, not a folder. Choose another folder name.     (exit 2)
$ airace-ml bench --model m --out <an existing folder>
error: --out .../rep is a folder; give a file name instead, such as .../rep/bench.json     (exit 2)
$ airace-ml fingerprint --model <truncated weights>
error: cannot load the model in .../bad (Error while deserializing header: invalid header length). Is it a folder made by 'airace-ml train', and is the file complete?     (exit 2)
$ airace-ml bench --model m --max-items 0
error: argument --max-items: '0' is not a whole number of 1 or more (run 'airace-ml bench --help' to see the options)     (exit 2)
```
Final checks: `cd ml && uv run --no-sync pytest -q` -> `1135 passed, 1 skipped, 6 deselected in 121.37s (0:02:01)`; `uv run --no-sync ruff check .` -> `All checks passed!`.

## Notes for the controller
- The unstable-stop path still exits 0 (decision unchanged from round 0); the message now makes the early stop and next step plain.
- `_DAMAGED` also lists `EOFError` and `TypeError`/`KeyError`. I checked rather than assumed: a truncated `.npy` raises `ValueError`, but a zero-byte `.npy` raises `EOFError`, so `EOFError` is needed and now has its own test case (removing it makes that case fail); `TypeError`/`KeyError` come from a calibration or meta record with the wrong fields. They are caught only where a file written by an earlier build is being read.
- The zero-byte-index test case was added after the main fix commit, as its own small commit (history is not rewritten): `test(ml): cover a zero-byte novelty index in the CLI's damaged-file errors`.
- In this round I ran `ruff format` only on the two task files (the round-0 mistake of formatting the whole tree was not repeated); `git status` after the commit is clean.
