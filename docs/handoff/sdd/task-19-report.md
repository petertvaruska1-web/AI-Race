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
