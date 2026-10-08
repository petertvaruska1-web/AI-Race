# Task 19 controller notes (binding alongside the brief)

## Interfaces from earlier tasks (read the modules for details)
- **Config:** `TrainRunConfig.from_json(s)` raises `ValueError` on a malformed document; `.validate()` raises `ValueError` (with `MixtureError` as a subclass).
- **Training:** `train_run(cfg, *, out_dir, data_root=None, device=None, on_event=None, resume=False, checkpoint_every=200) -> TrainResult`. It writes `result.json` itself.
- **Events** (`airace_ml.train.events`):
  - `Progress(step, total_steps, tokens, loss, lr)`
  - `HeldoutEval(step, losses)`
  - `Sample(step, prompt, text)`
  - `Instability(step, action, lr_scale)`
  - `Done(status, summary)`
- **Models and inference:**
  - `load_checkpoint(dir, device) -> (Transformer, CheckpointMeta)`
  - `TorchLM(model, tok, device=None)`
  - `.chat_reply(history, *, max_new_tokens=96, temperature=0.8, top_p=0.95, seed=0) -> str`
  - `.complete(text, *, max_new_tokens=64, temperature=0.8, top_p=0.95, seed=0) -> str`
- **Benchmarks:**
  - `airace_ml.evals.suite.run_benchmarks(lm, tok, *, suite=None, judge=None, novelty=None, categories=CATEGORIES, max_items_per_category=None, seed=0) -> BenchReport` (`.scores`, `.overall`, `.missing`, `.to_dict()`)
  - `Judge.load(judge_dir(root), device)`
  - `NoveltyIndex.load(novelty_path(root))`
  - `build_novelty_index(root) -> Path`
  - `build_judge(root, *, device=None, on_event=None, token_budget=None) -> Path`; it raises `JudgeBuildError` when training does not complete
- **Fingerprint:** `airace_ml.personality.fingerprint.measure_fingerprint(lm, tok, ...) -> Fingerprint(traits, samples)` and `TRAITS`.
- **Paths:** `airace_ml.paths.data_root()`, `tokenizer_path(root)`, `judge_dir(root)`, `novelty_path(root)`.
- **Device:** `airace_ml.device.pick_device(prefer=None)`. Read its signature.

## Rulings
1. **UTF-8 output (Review Focus 5).** At startup, reconfigure `sys.stdout`/`sys.stderr` to `encoding="utf-8", errors="replace"` only when the stream has a `reconfigure` method. pytest's capture objects may not. An untrained model emits U+FFFD and arbitrary bytes; printing must never crash, including on the Windows cp1252 console.
2. **Exit codes.**
   - 0: success.
   - 2: usage or config errors. That covers an argparse error, a malformed or invalid config, a missing model directory or tokenizer, and a gate that is not available yet. The error is printed as `error: <message>` on stderr.
   - 1: a runtime failure that is not the user's input, such as `JudgeBuildError`. It is printed as `error: <message>`.
   - 130: KeyboardInterrupt during train or chat, after printing that the run was interrupted. For train, also print that `--resume` continues the run.
   - Never print a traceback for the expected error types; let anything else propagate.
3. **`train`.**
   - Print `step {s}/{n}  loss {l:.3f}  lr {lr:.2e}` on each `Progress` event (two spaces between fields, exactly).
   - Print each `Sample` as `  sample [{prompt}]: {text}`, and each `HeldoutEval` as one compact line.
   - Print `Instability` as a warning line.
   - Print a final summary from the `TrainResult`: status, steps, tokens, final loss, wall seconds and out dir.
   - Add `--resume` (passed to `train_run(resume=True)`), since a long run interrupted on the owner's PC should continue.
   - `--parent DIR` sets the config's `parent_dir` (overriding the file's value) before validation.
4. **`chat`.**
   - A REPL over stdin lines, with the `>` prompt written to stdout. `/reset` clears history. `/raw` toggles completion mode (`lm.complete`). `/quit` or EOF exits 0.
   - Chat mode keeps a history list and calls `chat_reply(history, max_new_tokens=--max-tokens, temperature=--temperature, seed=turn_index)`.
   - Literal special-token text typed by the user is plain text (Review Focus 1). Nothing to do beyond passing it through.
5. **`bench`.**
   - By default it loads the judge and novelty index when both exist under the data root; if either is missing it prints a note, and creativity lands in `missing`.
   - `--no-creativity` skips them. `--max-items N` maps to `max_items_per_category`.
   - It prints a category table (category, score, n) plus overall and missing, and writes `report.to_dict()` JSON to `--out` when given (utf-8).
6. **`fingerprint`** prints each trait with its value, one per line, in `TRAITS` order. `--k` and `--seed` options are fine to add.
7. **`build-judge`** prints progress lines from training events (the same format as `train`), then the judge directory. `--budget-tokens N` maps to `token_budget`.
8. **`gate`** returns 2 with `error: gate not available` until Task 20 wires it. Accept its documented arguments now (`--out DIR [--seeds 1,2,3] [--cpu-speed] [--quick]`), so Task 20 only swaps in the call.
9. **Common options.**
   - `--data-root PATH` defaults to `data_root()`.
   - `--device {cpu,cuda}` defaults to `pick_device()`.
   - The model directory's tokenizer comes from `tokenizer_path(data_root)`. A missing tokenizer is a usage error (exit 2) with a message naming the path and saying to build the corpus first.
10. **Testability.** Keep `main(argv)` pure: no `sys.exit` inside it. Each subcommand is a small function. Tests run the brief's two tests plus the cases below on tiny data (`tiny_data_root`), CPU only, and keep the suite under 3 minutes:
    - chat `/reset`
    - non-ASCII output through a cp1252-like stream: wrap a BytesIO in `TextIOWrapper(encoding="cp1252")` and show reconfigure plus `errors="replace"` prevent a crash
    - `--resume`
    - `gate` returns 2
    - `fingerprint` runs
    - a missing tokenizer returns 2
    - bench with creativity: build a tiny judge and novelty index only if it fits in the time budget, otherwise test that a missing judge lands in `missing`
