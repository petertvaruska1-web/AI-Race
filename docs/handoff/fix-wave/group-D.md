# Fix wave group D: trainer

Read `docs/handoff/fix-wave/fix-wave-rules.md` first; it binds you. Rulings below are the controller's and are binding.

- **Where you work:** ml/src/airace_ml/train/, ml/src/airace_ml/data/ (only if needed), ml/tests/test_trainer.py and test_train_config.py.
- **CLAUDE.md edits allowed:** none.
- **Commit subject prefix:** `fix(D): <Fxx> ...`. Report file: `.superpowers/sdd/2026-10-06-m1-it-learns/task-fixwave-D-report.md`.
- **Before reporting:** full default suite, ruff, and the slow suite: `cd ml && uv run --no-sync pytest -m slow -q   (the slow suite, including the quick feasibility gate; a few minutes)`.
- **Order:** work the findings in the order listed.

## F2: Trainer runs a whole batch_tokens batch in one forward/backward with no micro-batching, so the reference judge recipe (8x384x256, 32768 tokens/step) likely does not fit the RTX 3050's 8 GB, and the spec's largest shapes cannot train at all
- Finder severity Important; verifier severity Important (fix size medium).
- Where: `ml/src/airace_ml/train/trainer.py` (510-519 (_train_step: one forward of cfg.batch_size rows; `logits` stays referenced through backward); recipe at ml/src/airace_ml/evals/judge.py:123-135 (batch_tokens=32768))
- Full finding: `docs/handoff/final-review/verify/s-train.json` (id F2); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-train`.
- **Ruling:** In _train_step keep the single sampler draw (x, y = next_batch(cfg.batch_size)), so RNG use and resume are unchanged. Split the rows into micro-batches of at most MICRO_BATCH_TOKENS=8192 tokens (rows = max(1, 8192 // ctx_len)). For each: forward under autocast, loss_i = cross_entropy(logits.float().flatten(0,1), y_i.flatten(), reduction='sum') / (batch_size*ctx_len), backward, accumulate loss_i.detach(), del logits. Then the existing after-backward work: ONE clip, ONE host sync, one spike check, one optimizer step. A single-chunk batch must stay bit-identical to today. Keep judge_train_config at 32768 tokens. Tests: a CPU test that micro-batched grads and loss match the full batch (assert_close); a @pytest.mark.gpu test that one step of judge_train_config() on CUDA keeps torch.cuda.max_memory_allocated() under 6 GiB.

## F7: Replay ('Review what it knows') of a parent whose mix included Notebook or Coaching fails with a MixtureError, or silently trains the child's current notebook under the review share, and the share propagates down the lineage
- Finder severity Important; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/train/trainer.py` (186-197 (_prepare: effective_mixture over parent.last_mixture at 188, load_pools with this run's custom docs at 193), 203-211 (_custom_docs always returns the key), 666 (last_mixture = blended stable )
- Full finding: `docs/handoff/final-review/verify/s-train.json` (id F7); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:s-train`.
- **Ruling:** Replay covers base datasets only. In trainer._prepare, filter the parent's last_mixture to DATASET_IDS before effective_mixture. If nothing is left, drop replay for this run and use normalize(mixture). Always renormalize. Record the dropped ids in the run summary (replay_dropped). Document the contract in the TrainRunConfig.replay docstring. Tests: a notebook parent, then a replay child without a notebook (passes, no notebook weight), then that child's grandchild; and a child that sends a notebook gets only its own mixture weight for it.

## F22: Tokens actually trained per step differ from batch_tokens; validate() accepts batch_tokens < ctx_len, which trains up to 2-2.56x the paid token budget
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/train/config.py` (170-176 (steps = token_budget // batch_tokens; batch_size = max(1, batch_tokens // ctx_len)), 196-204 (validate); trainer.py:432 (tokens_per_step))
- Full finding: `docs/handoff/final-review/verify/m-core-train.json` (id F22); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-core-train`.
- **Ruling:** TrainRunConfig.validate() raises ValueError('batch_tokens must be at least the attention span (ctx_len)') when batch_tokens < shape.ctx_len. Add a parametrized test. Leave the steps/batch_size derivation unchanged.

## F23: Lone UTF-16 surrogates in notebook, coaching or probe_prompts pass validate() but crash the tokenizer with TypeError; a bad probe prompt only fails mid-run at the first eval, every time
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/train/config.py` (229-241 (only isinstance(str) checks); trainer.py:203-211 (_custom_docs), 551-561 (_evaluate encodes probe prompts mid-run); tokenizer.py:65)
- Full finding: `docs/handoff/final-review/verify/m-core-train.json` (id F23); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-core-train`.
- **Ruling:** validate() checks every notebook entry, coaching part and probe prompt with s.encode('utf-8'); on UnicodeEncodeError it raises ValueError naming the field and index ('contains a lone surrogate, not valid text'). Tests for all three fields.

## F47: Three trainer resume branches, including the Windows 'refused rename' path, are never executed by any test
- Finder severity Minor; verifier severity Minor (fix size small).
- Where: `ml/src/airace_ml/train/trainer.py` (599-604 (suppress(OSError) swap), 609-611 (rebuild snapshot when saved one is None), 617-618 (re-derive finishing mixture on resume))
- Full finding: `docs/handoff/final-review/verify/m-core-train.json` (id F47); verdict and its fix recommendation: `docs/handoff/final-review/verify/verify-results.json` → `verify3:m-core-train`.
- **Ruling:** Add default tests in tests/test_trainer.py using cfg(): (1) a resume from a stop inside the finishing phase equals the uninterrupted run (weights and telemetry); (2) a monkeypatched Path.rename raising PermissionError during saves, then resume and compare; (3) if cheap, NaN at a step that is also a checkpoint step, then resume. Keep the default suite well under 3 minutes.
