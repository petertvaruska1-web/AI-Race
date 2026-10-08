# Task 20 report: feasibility gate, steps 1–5 (cloud)

**Status:** DONE_WITH_CONCERNS (the concerns are an interpretation to confirm and the file size; see the end)
**Commit:** `8d50f5d feat(ml): feasibility gate experiments, evaluators and report` (on `claude/upbeat-franklin-k1192h`, not pushed)
**Out of scope, left for the owner's GPU machine:** steps 6–9 (real novelty index and judge, real gate run, `docs/reports/m1-feasibility.md`, iterations). `docs/progress.md` not touched.

## What was implemented

- `ml/src/airace_ml/experiments/configs.py`
  - The brief's constants, verbatim: `STARTER_SHAPE`, `EARLY_SHAPE`, `STARTER_BUDGET`, `EARLY_BUDGET`, `BALANCED_MIX`, `TARGET_CATEGORY` and `mix_heavy` (validates the target and share).
  - `GateScale`, with `FULL_SCALE` and `QUICK_SCALE`. Quick mode uses shape (2,64,64), grown shape (3,96,64), 1024 × 20 tokens per run, `batch_tokens=1024` and fingerprint `k=1`. The full scale keeps the trainer's default `batch_tokens` (16384).
- `ml/src/airace_ml/experiments/gate.py`
  - `GateCriterion`, `GateOutcome` (with a `.passed` property), `GateSetupError`, and `TITLES`.
  - Pure evaluators `eval_speed`, `eval_legibility`, `eval_differentiation`, `eval_seed_variation`, `eval_growth`, `eval_forgetting` and `eval_prep`. None of them raise on degenerate input (ruling 2), and `data` holds only finite numbers, with `None` for anything not measured.
  - `speed_criterion` implements ruling 1.
  - Measurements: `garble_rate`, `false_fact_probes`, `false_fact_rate` and `max_logit_diff`.
  - `GateRuns` reuses and resumes runs (ruling 5). `run_gate` follows the brief's signature exactly.
- `ml/src/airace_ml/experiments/report.py`: `write_gate_report`, covering every item in ruling 11.
- `ml/src/airace_ml/cli.py`: `gate` is wired to `run_gate` (ruling 12).
  - It prints progress lines and a PASS/FAIL table, and writes `out/report.md` and `out/outcome.json` (every criterion, the transcripts and the raw numbers).
  - Exit codes: 0 when everything passed, 1 when any criterion failed (stdout only, no `error:` line), 2 for usage errors. A `GateSetupError` or a `MixtureError` also exits 2.
  - `--seeds` is parsed strictly: whole numbers of 1 or more, separated by commas only, and no repeats.
  - Interrupting prints how to carry on. The module docstring documents the new exit codes.
- `ml/tests/conftest.py`: one new session fixture, `gate_data_root`. `tiny_data_root` is unchanged.
- `ml/tests/test_gate.py`: the brief's tests, plus 21 default-suite tests and 1 more slow test.
- `ml/tests/test_cli.py`: the 2 old gate tests (the placeholder `test_gate_is_not_available_yet` and the 1-case seeds check) are replaced by 8 gate test functions, 21 test cases in all.
- `CLAUDE.md` "Commands" (ruling 13):
  - the cloud `--no-sync` rule
  - `pytest -m slow`
  - `airace-content build --scale {tiny,full}`
  - `airace-ml train|chat|bench|fingerprint|build-novelty-index|build-judge|gate`
  - the gate command, noting that it reuses runs and needs a fresh `--out` after a change to the training code

### How `run_gate` works (summary)
1. **Inputs are checked before any training.** These are: the tokenizer, all 8 corpora, `known_vocab.txt`, `false_facts.json`, the judge (`calibration.json`) and the novelty index.
   - Anything missing raises one `GateSetupError` whose message names every missing piece and the command that builds it.
   - Damaged files raise an error naming the command that rebuilds them.
   - The same step also loads the 60 web prefixes and the 4 growth sequences, and computes a digest of the tokenizer and corpora.
2. **G1.** Trains `starter` (seed 1, balanced, at the starter shape and budget) on the gate's device.
   - `speed_criterion` turns that time into G1. On a non-CUDA device G1 fails with "no CUDA device; GPU speed target not measured" plus the measured time.
   - With `--cpu-speed` on CUDA, it also trains `starter-cpu` on the CPU. On a CPU gate, the starter's own time is the CPU time.
3. **G2.** Takes the first 10 open and 10 help probes, in order, and asks each with `chat_reply(max_new_tokens=64, temperature=0.8, top_p=0.95, seed=i)`.
   - Each reply is checked with `is_well_formed`.
   - All 20 exchanges are kept as the transcript, together with their well-formed flags.
4. **G3.** Trains `g3-<target>-s<seed>` for each of the 4 targets and each seed, using `mix_heavy(target)` at the early shape and budget. Each run gets the full bench with creativity.
5. **G4.** Trains `g4-balanced-s<seed>` and fingerprints all G3 and G4 models.
   - The population is every fingerprint the gate takes (15 in the full gate, 5 in quick mode).
   - It measures the pairwise distances between the G4 seeds and between the four first-seed G3 models.
   - It records each model's `describe` words.
   - The G3 transcript is each first-seed mix model's reply to probes `creative-10` and `factual-01`, taken from its fingerprint samples: 8 exchanges.
6. **G5.**
   - It loads the starter and grows it to `grown_shape` with `grow(seed=seed 1)`.
   - It measures `max_logit_diff` on 4 held-out sequences, taken round-robin from the datasets, at full precision.
   - It trains `g5-grown` (grown shape) and `g5-ungrown` (starter shape), each continuing from the starter with +starter_budget on the balanced mix. It then compares their mean `heldout_losses`.
   - The time of the growth step is recorded as `growth_seconds`.
7. **G6.** The base is G3's creative-heavy first-seed run.
   - It trains `g6-code-replay0` and `g6-code-replay30` with mix `{"code": 1}`, replay 0 / 0.3, and +starter_budget at the early shape.
   - It benches language and creativity, and reads the base's numbers from its cached full bench.
8. **G7.**
   - (a) Trains `g7-web-light` (`PrepConfig("light")`) and `g7-web-thorough` (`PrepConfig("thorough", dedup=True)`) on `mix_heavy("web", 0.7)`. Each writes 60 `complete()` samples (48 new tokens, seed=i), and `garble_rate` is measured over the completions only.
   - (b) Trains `g7-facts-unchecked` and `g7-facts-checked` on `{"web": .5, "facts": .5}` and measures `false_fact_rate`.
   - Both pairs train at the starter shape on `prep_budget` (2 × the starter budget).
   - G7's data also records the number of completions, the number of distinct prompts, the number of fact pairs, and the number of fact pairs skipped.

**Reuse and resume (ruling 5, extended).**
- Each run's `runs/<name>/gate_run.json` holds a key made of four parts: the config (with the parent recorded by name), a digest of the tokenizer and corpora, the device type, and the parent run's *token*.
- A run is reused only when the key matches, the status is `completed`, and the checkpoint exists.
- A "started" record whose resume state matches `can_resume` resumes with `train_run(resume=True)`. Anything else retrains fresh.
- Every training gets a new random token. As a result, the children of a retrained parent and the cached measurements of a retrained run are invalidated automatically.
- Bench results are cached in `gate_bench.json`, keyed by the run token, the suite version, the categories, the seed, the judge calibration and the novelty-index size. Fingerprints are cached in `gate_fingerprint.json`, keyed by the run token, `k` and the seed.
- One case is not detected: a change to the training code itself. This is documented in the module docstring and CLAUDE.md: use a fresh `--out`.

## Tests and results

| Run | Result |
|---|---|
| Default suite, before (baseline, this machine) | 1171 passed, 1 skipped, 6 deselected in **127.56 s** |
| Default suite, after: `cd ml && uv run --no-sync pytest` | **1217 passed, 1 skipped, 8 deselected in 128.68 s** (+46 tests, about +1 s, no warnings) |
| `uv run --no-sync ruff check .` | All checks passed |
| `uv run --no-sync pytest -m slow tests/test_gate.py -v` | **2 passed in 103.30 s**: `test_run_gate_quick` 45.19 s (+3.81 s fixture setup), `test_quick_gate_through_the_cli_reports_and_then_reuses_its_runs` 54.22 s |
| Other existing slow tests (`-m slow --deselect tests/test_gate.py`) | 3 passed (judge smoke, judge resume smoke, reasoning at scale) |
| `tests/test_gate.py tests/test_cli.py` on the committed tree | 161 passed, 2 deselected |

### TDD evidence
- **RED.** `cd ml && uv run --no-sync pytest tests/test_gate.py -v`, with the brief's tests written verbatim and only ruff formatting and import-order fixes applied:
  ```
  E   ModuleNotFoundError: No module named 'airace_ml.experiments'
  ERROR tests/test_gate.py
  !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
  ```
  This failure was expected: the package did not exist yet.
- **GREEN.**
  - The brief's six fast tests: `6 passed, 1 deselected in 0.31s`.
  - The brief's slow test: `uv run --no-sync pytest -m slow tests/test_gate.py -v` gave `1 passed ... in 52.75s`, on the first run.
- **Extra tests.** These were partly written after the code, so I checked by mutation that each one bites. I applied 19 single-line mutations (11 to `gate.py`, 3 to `report.py`, 5 to `cli.py`), ran the covering test each time, and restored the file. **All 19 were caught.**
  - One test initially missed: the "subject after `{o}`" skip case. Its template put `{o}` first, so the context was empty and skipped for another reason. I fixed the test to use `"In {o} you find the {s}."`, and the mutation is now caught.
  - Mutations in `gate.py`:
    - the best rival's spread in G3
    - garble's vocabulary normalisation
    - the per-token mean against the summed total in `false_fact_rate`
    - the parent token in the run key
    - the CPU-as-GPU guard
    - resuming only with a matching key
    - reusing only `completed` runs
    - `max_logit_diff` with no sequences
    - G7's nothing-to-lower case
    - the `{s}`-before-`{o}` skip
    - the single-seed G3 fail
  - Mutations in `report.py`:
    - the 10-exchange cap
    - `<br>` for newlines
    - escaping `|`
  - Mutations in `cli.py`:
    - seeds ≥ 1
    - no repeated seeds
    - exit 1 on a failed criterion
    - `GateSetupError` mapped to exit 2
    - the `--cpu-speed` pass-through
- While writing the extra tests I made three test-construction mistakes and fixed them: a tie that made the wrong rival "best", a false object that was another pair's true object, and a miscounted call total. None of them changed the code.

### What the extra tests cover
- **Configs:** exact values, both scales' shapes are valid, and `mix_heavy` shares and errors.
- **Evaluators:** every boundary, including 90 / 480 s, 14 of 20, a seed distance of exactly 0.05, seed distance equal to mix distance, a logit diff of exactly 1e-4, drop 3 with recovery 0.5, and exactly 0.8 ×. Degenerate inputs (empty, a single seed, NaN or inf, a missing category or target, zero or negative drop, a zero rate before preparation) give fail details with finite data. G3 uses the best rival and the larger of the two spreads, with sample std.
- **`speed_criterion`:** never passes a CPU time as a GPU time.
- **`garble_rate`:** case, punctuation, digits, pooling over all texts (not a mean of per-text rates), contractions in the vocabulary, and non-ASCII text.
- **`false_fact_rate` (with `ScriptedLM`):** the exact contexts and continuations, the mean-per-token rule winning where the totals disagree, a mixed outcome giving 0.5, and an empty plan giving 0.
- **`false_fact_probes`:** skips and counts an unknown subject, an unknown relation, a "false" object equal to the true one, and a subject after `{o}`.
- **`max_logit_diff`:** under 1e-4 after real growth, larger for a different model, train/eval mode restored, and `inf` with no sequences.
- **The report:** escapes `|`, newlines and HTML/Markdown; caps long text with `…`; shows at most 10 exchanges with a note; every table row has its header's cell count; shows the "Iterations" heading, the wide per-mix table, the side-by-side bench table, `describe` words, the quick banner and the well-formed column.
- **`GateRuns` (with a fake `train_run`):** reuse versus retrain when the config, the parent token, the device or the data digest changes; `unstable_stopped` is retrained; a missing checkpoint is retrained; a run interrupted mid-way leaves "started", resumes only when `can_resume` holds for the same key, and is never resumed from another config.
- **`_cached`:** the cache behaves as intended.
- **`run_gate`:** refuses to start without its inputs, naming all three build commands, and writes nothing. It also rejects duplicate seeds.
- **CLI:**
  - wiring and argument pass-through, `report.md` and `outcome.json`, progress lines, and the summary table
  - exit 1 on a failed criterion, with an empty stderr
  - the defaults, and the note that `--quick` uses only the first seed
  - 11 malformed `--seeds` values
  - a real missing-inputs run on `tiny_data_root`, giving exit 2 with one line naming `airace-content build`, `airace-ml build-judge` and `airace-ml build-novelty-index`, and no output folder
  - `GateSetupError`, `MixtureError` and `FileNotFoundError` exit 2; disk full exits 1
  - Ctrl-C exits 130 with how to carry on
  - `--out` pointing at a file
- **The second slow test:** runs the quick gate through `main()` (exit 1).
  - It checks that every criterion's data is finite, that the G1 detail is the no-CUDA one, that there are 20 G2 and 8 G3 exchanges, 14 runs and 5 described models, and that all four report headings are present.
  - It then runs the same command again with a spy on `train_run`. **The spy sees zero training calls**, every run is marked reused, and the verdicts, details, benches and fingerprints are identical.

## The quick gate on tiny data (CPU)

Command:

```
airace-ml gate --out <scratch>/gate-quick --quick --data-root <tiny build + small judge + index> --device cpu
```

The data root is built exactly as the `gate_data_root` fixture builds it.

**Wall time:**
- **49.2 s** fresh, on the committed code. An earlier identical run took 53.0 s.
- **12.7 s** on a rerun into the same folder, with all 14 runs and every bench and fingerprint reused, and an identical report apart from `growth_seconds`.
- In pytest, the brief's slow test takes 45.2 s, plus 3.8 s to build the fixture: a tiny corpus build of 1.3 s, the novelty index, and the judge.

The run trains 14 runs:
- `starter`
- 4 × `g3-*`
- `g4-balanced-s1`
- `g5-grown` and `g5-ungrown`
- 2 × `g6-*`
- 4 × `g7-*`

Each takes about 1.2–2.6 s. The run then takes 4 full benches, 2 partial benches and 5 fingerprints.

Per-criterion output (exit 1):

```
G1  Speed            FAIL  no CUDA device; GPU speed target not measured (the first model trained in 2.5 s on the CPU)
G2  Legibility       PASS  16 of 20 replies well-formed (80%; needs 70%)
G3  Differentiation  FAIL  one seed per mix cannot show a margin over the seed spread. creative-heavy creativity 33.1 vs conversations-heavy 10.0 (lead +23.1, needs more than 0.0); code-heavy coding 0.0 vs creative-heavy 0.0 (lead +0.0, needs more than 0.0); facts-heavy knowledge 3.4 vs conversations-heavy 4.6 (lead -1.2, needs more than 0.0); conversations-heavy instruction 0.0 vs creative-heavy 0.0 (lead +0.0, needs more than 0.0)
G4  Seed variation   FAIL  no pairs of seeds to compare (needs at least 2 seeds). same mix, different seeds: mean distance 0.000 (needs more than 0.05); different mixes: 4.370 (seeds must differ less)
G5  Growth           PASS  growing moved no output by more than 2.1e-06 (allowed 0.0001); after the same extra training the held-out loss is 6.523 grown vs 6.591 not grown
G6  Forgetting       FAIL  creativity + language 33.1 for the story-heavy model, 37.1 after code-only training (a drop of -4.1; needs 3 or more), 75.5 with 30% replay (no drop to win back)
G7  Preparation      FAIL  garbled words 17.2% after thorough cleaning vs 15.9% after light cleaning; false facts preferred 42.0% with fact-checking vs 44.7% without (each must fall to 80% or less)

gate FAILED: 2 of 7 criteria passed
```

What these results mean (every number is finite and explained; tiny data only has to run end to end):
- **G1** fails by design on a CPU (ruling 1). On the CPU the 20-step first model took 2.5 s.
- **G2 passes, but it means nothing here.** The replies are word salad. The tiny judge (2×64×64, 8 steps) is so weak that its 90th-percentile threshold on real replies is very loose. The real judge decides this criterion.
- **G3 and G4 fail by design in quick mode.** With one seed there is no seed spread and no seed pair (ruling 2).
  - G3 still shows a real signal even at 20 steps. The creative-heavy model's creativity is 33.1, against 1.7–10.0 for the other mixes.
  - The fingerprints of different mixes are 4.37 standard deviations apart on average.
- **G5 passes for real.** Growth to (3,96,64) moved no logit by more than 2.1e-06, and the grown model reached a lower held-out loss (6.523 against 6.591).
- **G6** shows no drop: creativity at 20 steps is noise.
- **G7** shows no effect at 20 steps. The tiny web corpus has only 18 held-out documents, and they decode to just 5 distinct 32-token prefixes, because the fixture rows repeat. The 60 completions cycle through them. The fact pairs are all 300 of the plan, with 0 skipped.

### Sample of the generated report (`report.md`, 177 lines; excerpts)

````markdown
# M1 feasibility gate

> **Quick run.** Tiny models on a smoke-test budget: this checks that the gate runs end to end, and its results say nothing about feasibility.

Seeds: 1. Device: cpu.

## Summary

**2 of 7 criteria passed.**

| ID | Criterion | Result | Detail |
|---|---|---|---|
| G1 | Speed | FAIL | no CUDA device; GPU speed target not measured (the first model trained in 2.5 s on the CPU) |
| G2 | Legibility | PASS | 16 of 20 replies well-formed (80%; needs 70%) |
| ... |

## Measurements

### G3 Differentiation: FAIL

one seed per mix cannot show a margin over the seed spread. creative-heavy creativity 33.1 vs ...

|  | category | mean | std | seeds | rival | rival\_mean | rival\_std | lead | needed\_lead | leads |
|---|---|---|---|---|---|---|---|---|---|---|
| creative | creativity | 33.05 | 0 | 1 | conversations | 9.962 | 0 | 23.09 | 0 | yes |
| code | coding | 0 | 0 | 1 | creative | 0 | 0 | 0 | 0 | no |
| facts | knowledge | 3.385 | 0 | 1 | conversations | 4.615 | 0 | -1.231 | 0 | no |
| conversations | instruction | 0 | 0 | 1 | creative | 0 | 0 | 0 | 0 | no |

### G5 Growth: PASS

| Measure | Value |
|---|---|
| max\_logit\_diff | 2.146e-06 |
| max\_allowed | 0.0001 |
| grown\_loss | 6.523 |
| ungrown\_loss | 6.591 |
| growth\_seconds | 0.02218 |

## Transcripts

### G2: the first model's replies to probe prompts

The first 10 of 20 exchanges.

| # | Prompt | Reply | Well-formed |
|---|---|---|---|
| 1 | Hello! How are you today? |  that\])) fruit a's begins His?: they How\_ to Lome is isYes Local up to ... | yes |
| 2 | What did you do this morning? |  course dog rocks Earth play find Write fresh blue " farm in does ... | no |

### G3: each contrasting mix's model on the same prompts

| # | Prompt | Reply |
|---|---|---|
| 1 | \[creative-heavy\] Tell me a tiny story about a lost balloon. | cow built some came dog down! still brother On the morning, outside him ... |
| 2 | \[code-heavy\] Tell me a tiny story about a lost balloon. | boatQuestion your make come Write s early frogmer the?, justvOutputProgram to m \* and her<br> fourReturn |

## Differentiation models: benchmark scores

0 is no better than guessing, 100 is perfect.

| Category | creative-heavy (g3-creative-s1) | code-heavy (g3-code-s1) | facts-heavy (g3-facts-s1) | conversations-heavy (g3-conversations-s1) |
|---|---|---|---|---|
| language | 0 | 2 | 1 | 0 |
| knowledge | 0.3077 | 4 | 3.385 | 4.615 |
| creativity | 33.05 | 0 | 1.656 | 9.962 |
| consistency | 10 | 16.67 | 13.33 | 10 |
| ... |

## Personalities

| Model | Described as |
|---|---|
| g3-creative-s1 | talkative, predictable, bold |
| g3-code-s1 | hesitant, reserved |
| g3-facts-s1 | self-assured |
| g3-conversations-s1 | terse, inventive, timid, casual, warm |
| g4-balanced-s1 | nothing unusual |

## Iterations

_None yet. Each attempt to fix a failing criterion is recorded here: the hypothesis, the change and the result._
````

## Estimate for the real gate on the RTX 3050 (`airace-ml gate --out runs/gate-1 --cpu-speed`)

**Run count:** 24 GPU training runs, plus 1 CPU run with `--cpu-speed`, for 25 in all.

| Group | Runs | Shape | Tokens per run |
|---|---|---|---|
| G1 (shared by G2 and G5) | 1 `starter` | starter shape | 4.19 M |
| G3 | 12 (4 mixes × 3 seeds) | early shape | 8.39 M |
| G4 | 3 | early shape | 8.39 M |
| G5 grown | 1 | early shape | +4.19 M |
| G5 ungrown | 1 | starter shape | +4.19 M |
| G6 | 2 | early shape | +4.19 M |
| G7 | 4 | starter shape | 8.39 M |
| `--cpu-speed` | 1 `starter-cpu` | starter shape, on the CPU | 4.19 M |

**Measurements:**
- 12 full benches with creativity (G3), and 2 benches of language and creativity only (G6).
- 15 fingerprints with k=3.
- 20 G2 replies.
- 2 × 60 G7 completions and 2 × 300 false-fact pair scorings.

**Tokens:**
- Starter shape: 41.9 M tokens.
- Early shape: 138.4 M tokens.

**Throughput basis:**
- Task 8 measured the starter shape at **170–190k tokens/s warm** on this RTX 3050. Cold, it needs about 4.4 s extra for CUDA/cuBLAS start-up, which falls inside G1's timed run.
- The early shape costs about 2.5× the starter's compute per token (about 4.0 M against 1.6 M multiply-adds). Small models on this GPU are partly launch-bound, so I assume **70–100k tokens/s**.
- On this cloud CPU the early/starter time ratio was 1.9×. This machine is a 4-core Xeon: starter 7.9k tokens/s, early 4.1k tokens/s, measured over 8 steps.

**Wall time:**

| Part | Estimate |
|---|---|
| Training, starter shape | about 4 min |
| Training, early shape | about 23–33 min |
| Held-out evals and samples inside the runs (about 20 evals per 512-step run) | about 3 min |
| Benches (full bench target for the first model is 30 s; early models assumed 30–60 s each) | about 7–13 min |
| Fingerprints, G2, G5 and G7 measurements | about 2–3 min |
| `--cpu-speed` run | about 4–9 min |
| Setup (judge load, data digest) | under 1 min |
| **Total** | **about 45–65 min (call it about an hour)** |

- The total is lower than the brief's guess of 1–2 h.
- If early-shape throughput on the GPU turns out nearer 50k tokens/s, the total is about 75–85 min.
- **G1 margin:** the starter run should take roughly 25–35 s against the 90 s target.
- **CPU run:** the same run takes 8.8 min on this 4-core Xeon; a Ryzen 5 5500 (6 cores, 12 threads) should land around 4–7 min against the 8-minute target. This is the least certain number.
- **Resumability:** Ctrl-C at any point loses at most the unfinished part of one run, up to 200 steps.
- **Judge build:** step 6's judge build is separate, about 1 h per the brief.

## Files changed
- `ml/src/airace_ml/experiments/__init__.py`: new
- `ml/src/airace_ml/experiments/configs.py`: new, 88 lines
- `ml/src/airace_ml/experiments/gate.py`: new, 1216 lines
- `ml/src/airace_ml/experiments/report.py`: new, 200 lines
- `ml/src/airace_ml/cli.py`: gate wiring, strict `--seeds`, an interrupt hint, docstring exit codes
- `ml/tests/conftest.py`: adds the `gate_data_root` session fixture; nothing existing changed
- `ml/tests/test_gate.py`: new, 617 lines
- `ml/tests/test_cli.py`: the gate section rewritten, imports added
- `CLAUDE.md`: the Commands section

## Decisions and interpretations (recorded here for the controller)
1. **The brief's slow test uses `gate_data_root`, not `tiny_data_root`** (ruling 10: "Use it only from slow tests"). This is the only change to the brief's test code besides ruff formatting and import merging.
2. **Ruling 10 choice: a real tiny build.** I used `build_corpus("tiny", root, fixture_fetch(tests/fixtures/sources), seed=0)`, which takes 1.3 s and writes `known_vocab.txt` and `false_facts.json` with KB false facts that really appear in the data.
   - After the corpus come `build_novelty_index` and `build_judge(token_budget=8192)`.
   - The judge recipe is shrunk to 2×64×64 with batch 1024, monkeypatched exactly as the existing CLI `judged` fixture does. The real recipe would need at least one 32k-token step of an 8×384 model, about 30 s or more on CPU.
   - `tiny_data_root` is not touched at all.
3. **G3 seed spread = sample std (n−1),** the conservative choice. With one seed the criterion fails with an explanation (ruling 2). The rival u* is the target category's best other mix; ties go to the first in `TARGET_CATEGORY` order.
4. **G7 fails when the unprepared rate is 0** (nothing garbled, or no false fact preferred), with the detail "nothing could fall". The brief's formula would pass 0 ≤ 0.8 × 0, but the spec asks for a measurable reduction, so this is the ruling-2 degenerate case.
5. **`garble_rate` reads the known vocabulary with the same `words()` rule.** The build's vocabulary keeps apostrophes (`don't`), but `words()` splits them, so without this every contraction would count as garbled.
6. **G7(a) sampling.** It takes 60 held-out web documents in a seeded permutation order. With fewer than 60 (the tiny build has 18), they are cycled, and the `seed=i` sampling still varies each completion. Prefixes are the first 32 tokens after `<|bos|>`, decoded.
7. **"Seed 1" means the first of `--seeds`** (1 by default). Quick mode keeps `seeds[:1]`.
8. **Quick-mode shapes.** Only G5's grown child uses (3,96,64). G6's children keep the base's shape, as in the full gate, where early grows to early.
9. **Run reuse is stricter than ruling 5's minimum.** The key also includes a digest of the data, the device type and the parent's training token. A run that ends `unstable_stopped` is retrained on re-entry, because the ruling reuses only "completed" runs.
10. **G1 on a CPU gate with `--cpu-speed`** uses the starter's own time as the CPU time, rather than training it twice.
11. **G3 transcripts** come from the fingerprints' replies to `creative-10` and `factual-01`, sample 0, for each mix's first-seed model: 8 exchanges and no extra generation.
12. **The CLI's exit 1 for a failed criterion prints no `error:` line.** It is a verdict, not an error, and the module docstring says so. Strict `--seeds` also rejects repeated seeds, because a repeat would silently reuse one run as two "seeds" and fake a std of 0.

## Self-review
- **Every ruling is applied** (1–14), and the binding values match the brief, including signatures, the 20 probes, the 60 × 32 → 48 tokens, the replay share of 0.3, the web share of 0.7, the 2× preparation budget and the thresholds.
- **R1 holds:** nothing scripts or edits model output; the gate only measures.
- **All file I/O is UTF-8,** and JSON is written strictly (`allow_nan=False`) via a temp file and `os.replace`.
- **`data` stays finite:** non-finite values become `None`, verified by tests and by the slow test on real output.
- **Default tests are light:** the suite grew by 46 tests (27 in `test_gate.py`, a net 19 in `test_cli.py`) for about 1 s in total. Heavy paths are `@pytest.mark.slow`.
- **The code is cohesive but large.** The sections are evaluators, measurements, inputs, runs and caches, experiments, and `run_gate`.

## Concerns
1. **Ruling 6 wording (please confirm).** The ruling says: "Skip, and count, pairs whose template does not end its sentence with `{o}` (text after `{o}` is fine; the cut is at `{o}`)."
   - I read the parenthesis as allowing templates like `"The {s} has {o} legs."`, whose context `"The spider has"` is a fair probe.
   - So I skip, and count, only pairs whose cut would not contain the subject (`{s}` not before `{o}`), plus unknown relations or subjects and false objects that equal the true one.
   - With the current KB this skips 0 of the 300 planned pairs.
   - Under the stricter reading (skip anything with words after `{o}`), `animal_legs`, `shape_sides` and `planet_order` would be dropped, roughly a quarter of the pairs.
   - It is a one-line change in `_object_context` if you want the strict reading.
2. **`gate.py` is 1216 lines.** It is cohesive and follows the brief's file plan. If you prefer, `GateRuns`, `_cached` and the input loading could move to `experiments/runs.py` without other changes. I did not restructure on my own.
3. **A training-code change is not detected by run reuse;** only changes to the config, data, device or parent are. This matters during step 8 iterations on, for example, the boldness mapping. It is documented in the `gate.py` docstring and CLAUDE.md (use a fresh `--out`).
4. **Quick-mode G2 "passes" on word salad** because the tiny judge is lenient. This is expected at tiny scale and has no bearing on the real gate.
