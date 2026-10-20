# Fix wave group F: gate and CLI

Read `docs/handoff/fix-wave/fix-wave-rules.md` first; it binds you. Rulings below are the controller's and are binding.

- **Where you work:** ml/src/airace_ml/experiments/, cli.py, tokenizer.py (F38), evals/judge.py (F19 only), and their tests.
- **CLAUDE.md edits allowed:** the CLAUDE.md gate example (separate --out for --quick, F20) and the real-gate time estimate if CLAUDE.md states one (F10).
- **Commit subject prefix:** `fix(F): <Fxx> ...`. Report file: `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-F-report.md`.
- **Before reporting:** full default suite, ruff, and the slow suite: `cd ml && uv run --no-sync pytest -m slow -q   (the slow suite, including the quick feasibility gate; a few minutes)`.
- **Order:** work the findings in the order listed.

## F8: G5 continuation runs retrain on the starter's exact token stream, so the 'extra tokens' are a verbatim second epoch
- Finder severity Important; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/experiments/gate.py` (1130-1136 (growth(): both arms seed=self.seeds[0], BALANCED_MIX); root cause ml/src/airace_ml/train/trainer.py:194 (MixtureSampler seeded only by cfg.seed))
- Full finding: `docs/handoff/final-review/verify/s-train.json` (id F8); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-train`.
- **Ruling:** Gate-only fix: both G5 continuation arms (g5-grown and g5-ungrown) use the same seed, different from the starter's (seed + 10_000), so they train on identical FRESH data and stay paired. The trainer is unchanged. Add a test that the continuation configs share a seed different from the starter's. M2 obligation, recorded by the controller: a continuation must never reuse its parent's sampler stream.

## F9: G4's 'measurably different' clause (seed distance > 0.05) sits about 40x below the fingerprint's own sampling noise, so it can never fail and G4 passes on pure measurement noise about half the time
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_ml/experiments/gate.py` (123 (SEED_MIN_DISTANCE), 140 (MEASURE_SEED), 410-437 (eval_seed_variation, pass rule at 436), 1076-1104 (G4 population and pairs); ml/src/airace_ml/personality/fingerprint.py:363-375 (fingerprint_dist)
- Full finding: `docs/handoff/final-review/verify/s-gate.json` (id F9); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-gate`.
- **Ruling:** G4 measures its own noise floor. Also fingerprint each balanced seed model at MEASURE_SEED+1 (cached under its own key), and compute same-model distances (original vs re-measure) against ONE population that includes the re-measures. Pass only if seed_mean > noise_mean AND seed_mean < mix_mean; keep seed_mean > 0.05 as an absolute sanity bound. Traits whose population std is below 1e-6 are excluded from every distance (no 66,666 blow-ups). Show noise_mean, seed_mean and mix_mean in the detail, data and report. The controller writes ADR 0002.

## F10: Spec §4.12 says '3 seeds each', but G2, G5, G6 and G7 (and G1) are decided on seed 1 alone with zero-margin thresholds, so verdicts can flip by seed
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_ml/experiments/gate.py` (1001, 1020, 1034-1054, 1126-1143, 1145-1165, 1167-1223 (only 1061 and 1089 loop over self.seeds); spec docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md:210; plan:1868-1873)
- Full finding: `docs/handoff/final-review/verify/s-gate.json` (id F10); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-gate`.
- **Ruling:** Follow spec §4.12 '3 seeds each' for G2, G5, G6 and G7. G1 stays single-seed: it is a timing and does not depend on the seed; the controller records this. G2 needs starters for seeds 2-3 and must pass on every seed. G5 runs its paired arms per seed. G6 uses the existing g3 creative base for each seed. G7 runs its prep arms per seed. Suffix run names with the seed. G5, G6 and G7 decide as G3 does: the mean effect must exceed SPREAD_FACTOR x the seed std. Per-seed numbers go into criterion data and the report. Quick mode stays single-seed and decides on that seed alone. Update the real-gate time estimate wherever it is printed or documented (about +30-40 GPU minutes).

## F13: G7(a) cannot be expected to pass on real data: thorough cleaning lowers the unknown-word share of real web training text only to 79-82% of light's, at or above the 0.8 bar, before the model adds non-words of its own
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_ml/experiments/gate.py` (128 (PREP_MAX_RATIO = 0.8), 512-544 (eval_prep), 550-560 (garble_rate), 1167-1210 (preparation runs); noise strengths ml/src/airace_content/noise.py:23-24; web rates ml/src/airace_content/assemble.py:)
- Full finding: `docs/handoff/final-review/verify/s-gate.json` (id F13); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-gate`.
- **Ruling:** G7(a) counts only GARBLE-LIKE words, not all unknown words. A word is garble-like if it contains a character from the noise module's mojibake/garble character set, OR it is not in known_vocab, has length >= 4 and is within edit distance 1 of a known_vocab word (a learned typo). garble_rate = garble-like words / all words. Full-mode GARBLE_SAMPLES rises from 60 to 200 (quick stays small). Report, as information next to the verdict, the same measure on each arm's prepared WEB training text (a deterministic sample of about 2,000 docs from the pools each arm trained on): the data-level ratio. The 0.8 bar stays. The controller writes ADR 0002, and the local checklist says: if the data-level ratio is above about 0.7, strengthen web noise or thresholds before trusting G7(a).

## F19: A refused Windows rename of the post-training record (judge train_config.json, gate gate_run.json) throws away a finished training: os.replace has no retry and the trainer has already deleted the resume state
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/evals/judge.py` (228-232 (_write_json), 235-243 (_trained), 277-292; ml/src/airace_ml/experiments/runs.py:90-95 (_write_json), 181, 211; trainer.py:565-604, 687-690 (_finish removes resume states))
- Full finding: `docs/handoff/final-review/verify/m-gate-runs.json` (id F19); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-gate-runs`.
- **Ruling:** A small retry around os.replace in both _write_json helpers (evals/judge.py and experiments/runs.py; optionally novelty.py): retry on PermissionError about 6 times with a growing sleep (0.1 s, doubling), then re-raise. Test by monkeypatching os.replace to fail twice and then succeed.

## F20: Gate reruns into the same --out with --quick, another device or another first seed silently overwrite the finished full-scale runs (same run names for both scales)
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/experiments/runs.py` (176-198 (key mismatch -> train_run(resume=False) into runs_dir/cfg.run_id); gate.py:996-1003, 1062-1069, 1132-1136, 1150-1159, 1172-1181, 1258; cli.py:356-362; CLAUDE.md:39)
- Full finding: `docs/handoff/final-review/verify/m-gate-runs.json` (id F20); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-gate-runs`.
- **Ruling:** On first use the gate writes out/gate_setup.json ({quick, device type, seeds}). If the file exists and differs, raise a usage error (exit 2) telling the user to pick another --out. CLAUDE.md: the --quick example uses its own --out.

## F21: build-judge and train silently fall back to the CPU when CUDA is unavailable; on the CPU the judge needs about 10.7 GB of RAM per step and takes days
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/device.py` (8-13 (silent fallback); ml/src/airace_ml/cli.py:356-362, 465-489, 772-779; ml/src/airace_ml/evals/judge.py:262; ml/src/airace_ml/train/trainer.py:347-350 (no autocast on the CPU))
- Full finding: `docs/handoff/final-review/verify/m-core-train.json` (id F21); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-core-train`.
- **Ruling:** build-judge, bench and chat print one plain-words device line before work ('using the GPU (<name>)' / 'no usable GPU found: using the CPU'). build-judge and the FULL gate refuse a silent CPU fallback (exit 2, saying it would take days on the CPU and how to force it with --device cpu); an explicit --device cpu or AIRACE_DEVICE=cpu is allowed, and --quick is allowed on the CPU. train keeps its existing header line.

## F38: CLI commands print Python tracebacks for a damaged corpus, a file-system error or a corrupt tokenizer.json (bare Exception from tokenizers), breaking the 'one error: line, never a traceback' contract
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/cli.py` (14-19 (contract), 89 (_DAMAGED), 374-381 (_require_tokenizer), 686-693, 762-769 (_cmd_build_novelty_index), 782-801 (_cmd_build_judge); ml/src/airace_ml/experiments/gate.py:150 (_LOAD_ERRORS), 812, 83)
- Full finding: `docs/handoff/final-review/verify/m-cli-tests.json` (id F38); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-cli-tests`.
- **Ruling:** tokenizer.Tok.load re-raises a parse failure as ValueError(f'the tokenizer at {path} is damaged: {e}'). cli._require_tokenizer turns that ValueError into a usage error with BUILD_DATA_HINT. _cmd_build_novelty_index catches OSError via _files_problem and ValueError as a usage error ('the training data under {root} is damaged ...; rebuild it with airace-content build'). _cmd_build_judge gets the same ValueError catch, which also covers calibrate's ValueError. Tests for a truncated tokens.bin and a truncated tokenizer.json.

## F39: `train --resume` on a folder holding a finished run tells the user to run without --resume, which overwrites the finished model
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/cli.py` (576-581)
- Full finding: `docs/handoff/final-review/verify/m-cli-tests.json` (id F39); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-cli-tests`.
- **Ruling:** In _cmd_train, when --resume finds no saved state but --out holds a finished model of this config (result.json with this run_id and status completed, or model.safetensors + meta.json), raise a usage error saying the run already finished and the model is ready to use, and that retraining needs the folder deleted or another --out. Test.

## F41: Every gate rerun regenerates report.md from scratch, discarding any 'Iterations' notes written into it
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/experiments/report.py` (259-270 (write_gate_report); cli.py:832-833)
- Full finding: `docs/handoff/final-review/verify/m-gate-runs.json` (id F41); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-gate-runs`.
- **Ruling:** write_gate_report keeps an existing '## Iterations' section: if the target file exists and its Iterations body differs from ITERATIONS_PLACEHOLDER, reuse that body. Test: write, edit Iterations, rewrite, and assert the text survives.
