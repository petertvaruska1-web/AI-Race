# Task 20 controller notes (binding alongside the brief). Cloud scope: steps 1–5 (code and tests)

Steps 6–9 (the real novelty index and judge builds, the real gate run and the report) need the real corpus and the GPU. They stay with the owner's local session. Here you build `configs.py`, `gate.py`, `report.py`, the CLI `gate` wiring and the tests, and prove the whole path end to end on tiny data.

## Interfaces from earlier tasks (read the modules for details)
- **Training:** `train_run(cfg, *, out_dir, data_root, device, on_event, resume, checkpoint_every) -> TrainResult`.
  - `TrainResult` has `status`, `heldout_losses: dict[str, float]` and `wall_seconds`.
  - `TrainRunConfig(..., replay, prep, parent_dir, batch_tokens, ...)`. Growth to a bigger shape happens inside `train_run` when `parent_dir` is set and the shape grows. `replay > 0` requires `parent_dir`.
- **Model and inference:**
  - Growth: `airace_ml.model.growth.grow(model, target, *, seed)`.
  - Loading: `load_checkpoint(dir, device)`.
  - `TorchLM(model, tok, device)` provides `.chat_reply(...)`, `.complete(...)` and `.score_continuations(...)`.
- **Evaluation:**
  - `run_benchmarks(lm, tok, *, judge, novelty, categories, ...)`
  - `Judge.load(judge_dir(root))`, `is_well_formed(reply, judge)`
  - `NoveltyIndex.load(novelty_path(root))`
  - `measure_fingerprint(lm, tok, ...)`, `fingerprint_distance(a, b, population)`, `describe(fp, population)`, `load_probes()` (open 20 / help 10)
- **Data and KB:**
  - `load_kb()`, `FalseFactPlan.from_json(text)`
  - `corpus_dir(root)` holds `false_facts.json` and `known_vocab.txt` from the Task 14 build.
  - `Relation.train_templates` are `str.format` templates with `{s}` and `{o}`.
  - `Corpus.open`, `heldout_docs`.
- **CLI:** the `gate` subcommand from Task 19 accepts `--out DIR [--seeds 1,2,3] [--cpu-speed] [--quick]` and returns 2 until wired.

## Rulings
1. **Device and G1.**
   - G1 times the starter run on the gate's device. With `include_cpu_speed`, it also times a CPU run.
   - If the gate's device is not CUDA, G1 fails with the detail "no CUDA device; GPU speed target not measured", plus the measured time. It never passes a CPU time off as a GPU time.
   - `eval_speed` itself stays as the brief says.
2. **Evaluators on degenerate input.** Every evaluator returns a `GateCriterion` with finite numbers in `data`, never raising and never NaN, for:
   - empty lists, e.g. G4 with a single seed has no seed pairs
   - a single seed (std 0)
   - zero drop in G6

   In those cases the criterion fails with an explanatory detail. Exception: in G6, drop = 0 fails by the brief's own rule.
3. **G2 probes:** the first 10 `open` and all 10 `help` probes, in probes.json order. Use `lm.chat_reply([("user", text)], max_new_tokens=64, temperature=0.8, top_p=0.95, seed=i)`. Keep all 20 (prompt, reply) pairs as the G2 transcript.
4. **Model reuse.** Several criteria reuse the same trained models rather than retraining:
   - G2 and G5 reuse G1's starter run, which uses seed 1.
   - G6's base is G3's `creative`-heavy seed 1 run.
   - G4's mix distances use G3's four seed-1 runs.
5. **Resumable gate.** The real gate runs 1–2 h on the owner's PC.
   - Each run goes in `out_dir/runs/<name>`.
   - After a run completes, write `gate_run.json` (the `cfg.to_json()` plus the result status).
   - On re-entry, a run whose `gate_run.json` matches the config and says "completed" is reused, not retrained.
   - A run interrupted mid-way resumes via `train_run(resume=True)` when its resume state matches. Use `trainer.can_resume(out_dir, cfg)`.
   - Bench and fingerprint results per run are cached the same way (JSON beside the run).
6. **`false_fact_rate`.**
   - The context is `relation.train_templates[0]` rendered with the subject and cut just before `{o}`. Strip trailing spaces; the continuation carries the leading space.
   - The fact counts as false-preferred when the false object's **mean log-prob per token** beats the true object's, matching the MC rule. The brief says "score"; mean per token is the fair comparison for objects of different lengths.
   - Skip, and count, pairs whose template does not end its sentence with `{o}` (text after `{o}` is fine; the cut is at `{o}`).
   - Return 0.0 when there are no pairs.
7. **`garble_rate`.** Words are lowercase alphabetic tokens; reuse `airace_ml.evals.creativity.words`.
   - G7(a) samples: 60 held-out web documents, chosen deterministically (seeded).
   - Each prefix is the first 32 tokens, decoded.
   - Use `lm.complete(prefix, max_new_tokens=48, temperature=0.8, top_p=0.95, seed=i)`.
   - The texts measured are the completions only.
8. **G5.**
   - The logit diff is measured with `grow()` directly on the loaded starter model, comparing logits on 4 held-out sequences (any datasets, deterministic) at `ctx_len` or shorter.
   - Continue training the grown model with `parent_dir` = the starter run and shape `EARLY_SHAPE`.
   - Continue training the ungrown model with `parent_dir` = the starter run and shape `STARTER_SHAPE`.
   - Both add `+STARTER_BUDGET` on `BALANCED_MIX` with the same seed.
   - Compare the mean of `TrainResult.heldout_losses` values.
9. **Quick mode.**
   - Every experiment uses shape `(2, 64, 64)`, `batch_tokens=1024`, a budget of 1024 × 20 tokens per run, and a single seed (seed 1). Continuation runs add another 1024 × 20.
   - For any parent/child pair, the quick "grown" shape is `(3, 96, 64)`, so growth is still exercised.
   - Fingerprints use `k=1`; bench uses the full suite.
   - The quick gate must run in a few minutes on CPU, so the slow test stays practical. Report its time.
10. **Tiny data for the slow quick-gate test (plan ruling R6).**
    - Extend the `tiny_data_root` fixture in `ml/tests/conftest.py` to also write `known_vocab.txt` (the words of its corpora) and `false_facts.json` (a `plan_false_facts` from the real KB, seeded) into each tiny corpus dir. This must not change any existing document or tag, since earlier tests depend on them.
    - Add a session fixture, e.g. `gate_data_root`, that copies `tiny_data_root` and lazily builds a tiny novelty index (`build_novelty_index`) and a tiny judge (`build_judge(token_budget=small)`). Use it only from slow tests.
    - Alternatively, if it is simpler and faster, build a real tiny corpus with `airace_content.build.build_corpus("tiny", root, fixture_fetch(FIX))`. That produces `known_vocab.txt` and `false_facts.json` itself, after which you build the judge and index. Either is acceptable. Pick the faster one that exercises the whole path, and report the choice.
11. **Report (`write_gate_report`).**
    - Markdown, utf-8.
    - Escape `|` and newlines inside table cells, because transcripts contain arbitrary model text.
    - Cap each transcript exchange at a reasonable length, since an untrained model can emit long garbage.
    - Include an "Iterations" section heading with a placeholder line, for step 8 to fill locally.
12. **CLI.** Wire `gate` to `run_gate`:
    - print progress lines through `on_progress`
    - write the report to `out_dir/report.md`
    - print a PASS/FAIL summary table
    - exit 0 if every criterion passed, 1 if any failed, 2 on a usage error
    - `--seeds 1,2,3` is parsed strictly (positive ints)
    - a missing judge or novelty index is a usage error (exit 2) that names the CLI command that builds it
13. **CLAUDE.md "Commands".** Add the commands that now exist:
    - `airace-content build --scale {tiny,full}`
    - `airace-ml train|chat|bench|fingerprint|build-novelty-index|build-judge|gate`
    - the cloud `uv run --no-sync` note

    Keep it concise; the existing style is short bullets.
14. **Suite time.** The default suite takes about 115 s of 180 s on this machine. Default-suite tests for Task 20 must be pure and fast (evaluators, configs, report, CLI arg parsing). The quick gate is `@pytest.mark.slow`.
