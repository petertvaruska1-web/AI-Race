# Fix wave group E: evals

Read `docs/handoff/fix-wave/fix-wave-rules.md` first; it binds you. Rulings below are the controller's and are binding.

- **Where you work:** ml/src/airace_ml/evals/, infer/, personality/ (F33 test only), the shared identity helper, the gate and CLI only where F4 needs them, and their tests.
- **CLAUDE.md edits allowed:** one CLAUDE.md line on rebuilding the judge and novelty index after a corpus or tokenizer rebuild (F4).
- **Commit subject prefix:** `fix(E): <Fxx> ...`. Report file: `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-E-report.md`.
- **Before reporting:** full default suite, ruff, and the slow suite: `cd ml && uv run --no-sync pytest -m slow -q   (the slow suite, including the quick feasibility gate; a few minutes)`.
- **Order:** work the findings in the order listed.

## F4: The judge and the novelty index record nothing about the tokenizer and corpora they were built from: after a rebuild, build-judge reuses old weights with a new tokenizer, the gate silently measures with stale instruments, and the advised fix for a damaged judge crashes
- Finder severity Important; verifier severity Minor (fix size medium).
- Where: `ml/src/airace_ml/evals/judge.py` (235-243 (_trained compares only the recipe config and file presence), 276-297 (tokenizer copied at 293 even when weights are reused), 100-106 (Judge.load checks only vocab size); ml/src/airace_ml/eval)
- Full finding: `docs/handoff/final-review/verify/s-evals.json` (id F4); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-evals`.
- **Ruling:** Move gate._data_identity to a shared helper (e.g. airace_ml/data/identity.py) and use it from the gate. build_judge writes the identity, plus whether the budget is the full recipe, into the judge's train_config record; build_novelty_index writes it into the index's .json. judge._trained requires a match, and build_judge retrains if a reused checkpoint fails load_checkpoint (_LOAD_ERRORS/SafetensorError). The gate's _load_inputs and cli bench (_creativity_tools) compare the identity with the current data root and raise the existing 'out of date ... Rebuild it with: airace-ml build-novelty-index / build-judge' error; they also refuse a non-recipe-budget judge as the reference judge. Put the identity into the bench cache key. CLAUDE.md: one line saying the judge and novelty index are rebuilt after any corpus or tokenizer rebuild.

## F6: Creativity rewards degenerate output: the same story for all 24 prompts scores 21-52, and a short stock sentence gets copy-level credit that can beat real stories
- Finder severity Important; verifier severity Minor (fix size none).
- Where: `ml/src/airace_ml/evals/creativity.py` (129-143 (story = c*(0.4+0.6*nov)*(1-rep); score = 100*raw*(0.5+0.5*d)), 97-102 (_coherence); novelty.py:328-330; tests test_creativity.py:76-90, 80-82, 883-895)
- Full finding: `docs/handoff/final-review/verify/s-evals.json` (id F6); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-evals`.
- **Ruling:** OVERRULED REFUTATION, narrowed fix. Spec §4.7 defines creativity as coherence x novelty x DIVERSITY, and ADR 0001 says a model that gives the same answer to everything scores near 0; the plan's (0.5 + 0.5*d) lets a constant model keep half. Make diversity a real multiplier: score = 100 * raw * min(1, d / d_ref), where d is today's distinct-2 across the run's stories and d_ref is the same distinct-2 measured at judge calibration over the held-out creative openings already used there, stored in calibration.json (e.g. 'creative_distinct2'; a judge without it must be recalibrated, not guessed). Keep the 0.4 novelty floor, the repetition penalty and the prompt-echo rule. Do NOT add a length term. Tests: the same story for every prompt (novel or corpus) and a constant short reply score at most about 5; per-prompt varied stories score close to before; replace the constant 'novel' exemplar in test_creativity with per-prompt replies. The controller records the rule in ADR 0001.

## F33: A model with non-finite logits crashes creativity, the fingerprint and chat (torch.multinomial RuntimeError) instead of scoring finite
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/infer/lm.py` (115-126 (_sample); callers personality/fingerprint.py:314-324, evals/creativity.py:123-125)
- Full finding: `docs/handoff/final-review/verify/m-evals-infer.json` (id F33); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-evals-infer`.
- **Ruling:** In infer/lm.py _sample: rows whose probabilities are not all finite take logits.argmax(-1), exactly as temperature 0 already does. No text is invented (R1). Test: generate(temperature=0.9, top_p=0.9) on a NaN-weight model, plus creativity and fingerprint scoring, all run without raising.

## F34: Definitions that should be shared are duplicated: the G2 well-formed check counts words differently from creativity and the gate, and the 96-token story/opening length is defined twice
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/evals/judge.py` (34 (imports checkers.words), 45 (CREATIVE_TOKENS = 96), 141-150 (is_well_formed); creativity.py:44 (STORY_TOKENS = 96), 77-80; gate.py:554-559)
- Full finding: `docs/handoff/final-review/verify/m-evals-infer.json` (id F34); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-evals-infer`.
- **Ruling:** Reword the docstrings at creativity.py:77-80 and gate.py:554-556 to say that G2's is_well_formed uses skills.checkers.words, not these letter-run words. Define the 96-token opening length once (creativity reuses judge.CREATIVE_TOKENS, or a shared constant if importing would create a cycle). No behaviour change.

## F42: A slow test writes a judge into the shared session-scoped tiny_data_root, so running slow and default tests together fails other tests
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/tests/test_creativity.py` (119-125 (test_build_judge_smoke); affected ml/tests/test_gate.py:585-599, ml/tests/test_cli.py:596-613 and 1147-1153; fixture conftest.py:111)
- Full finding: `docs/handoff/final-review/verify/m-cli-tests.json` (id F42); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-cli-tests`.
- **Ruling:** test_build_judge_smoke copies tiny_data_root to tmp_path and builds and loads the judge there, as test_build_judge_resume_smoke does. Check that no other slow test writes into tiny_data_root.

## F43: The GPU test asserts exact greedy-token equality between CPU fp32 and CUDA bf16, but the fp32 top-2 margin (0.014 logits) is far below bf16 rounding error
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/tests/test_infer.py` (413-428 (test_cuda_bf16_path_tracks_the_cpu_path))
- Full finding: `docs/handoff/final-review/verify/m-evals-infer.json` (id F43); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-evals-infer`.
- **Ruling:** Replace the exact greedy-token equality in test_cuda_bf16_path_tracks_the_cpu_path with a tolerance check: teacher-force the CPU greedy sequence through both models and assert per-position logits agree within about 0.5 (or require the GPU argmax to match only where the fp32 top-2 margin exceeds 1.0). Keep the seeded-reproducibility, probability-range and score_continuations checks.

## F46: No test checks that the gate's data identity changes when the corpus is rebuilt, and the default suite never runs the gate's real input loading
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/experiments/gate.py` (721-735 (_data_identity), 764-775 (_growth_sequences), 836-852, 922-926; ml/src/airace_ml/experiments/runs.py:250-262)
- Full finding: `docs/handoff/final-review/verify/m-gate-runs.json` (id F46); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-gate-runs`.
- **Ruling:** After the F4 helper move, add default-suite tests on a tmp copy of tiny_data_root: the data identity is stable across two calls and changes when tags.npz, offsets.npy, the tokenizer, info.json or a tokens.bin size changes. In the fake-gate fixture, vary inputs.identity between runs and assert everything retrains.
