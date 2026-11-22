# Next level, plan B: brain design 2 (screening) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (the owner's instruction of 2026-11-13: bounded tasks and their tests go to lighter models, the lead developer keeps design, measurements and decisions; no task goes to a model that would do it worse). Steps use checkbox (`- [ ]`) syntax for tracking. Ledger: append to `docs/handoff/sdd/ledger-next-level.md` (working copy in the plan's `.superpowers/sdd/` workspace).

**Goal:** find out, by measurement, whether a second brain design trains better than today's on the RTX 3050; build it only if it earns its place.

**Architecture:** a scratch model class beside the product (`LabTransformer`, a subclass of `airace_ml.model.transformer.Transformer`) whose five design changes are switched by flags; a lab script trains the scorecard's shapes through the real trainer (the lab swaps in its class for the length of a run) at equal wall time against today's design, and measures held-out loss, the yardstick and bench-v1. Nothing in `ml/src/` changes in this plan. If the gate passes, a plan B2 builds design 2 into the product.

**Tech Stack:** Python 3.13, PyTorch (CUDA, bf16 autocast), the existing trainer, scorecard and benchmark code.

**Spec:** `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`, section 5.5 (screening and gate), section 7 (targets), section 8 (compatibility). Results of plan A: `docs/decisions/0010-next-level.md`.

## Global Constraints
- Run uv as `python -m uv` from `ml/`; never `uv sync` without `--extra content`.
- Never write into `data/` or `data2/`; the screen reads `data2/` (text 2, the release's text) through `AIRACE_DATA=../data2`.
- Nothing in `ml/src/` changes: the lab lives in `docs/handoff/next-level/arch_lab.py` with its test in `ml/tests/test_arch_lab.py` (the pattern of `growth_lab.py` and `ml/tests/test_growth_lab.py`).
- The five changes (spec section 5.5), each a flag of `ArchOptions`: output projections that start at zero (`zero_out`); normalised queries and keys with a learned scale (`qk_norm`); a norm right after the embedding (`emb_norm`); a two-matrix squared-ReLU feed-forward of the same size in place of SwiGLU (`relu2`); RoPE base matched to spans of at most 512 (`rope_base`: 500.0 instead of 10,000, so the slowest rotation is about 2,100 tokens, four times the longest span).
- Sizes and budgets (spec section 5.5): Tiny (4x128) 12,582,912 tokens, Small (6x192) 12,582,912, Medium (8x256) 25,165,824; seeds 1 and 2; the scorecard's even mix (`BALANCED_MIX`).
- **The gate (spec section 5.5, fixed before measuring):** design 2 is built only if all five changes together are better by at least 0.04 held-out loss at equal time on all three sizes (mean of two seeds), or by at least 3 bench-v1 points on all three; otherwise it is dropped and recorded. A single change that passes the gate on its own is reported, and plan B2 may build only it.
- GPU timing runs with nothing else running (no tests, no builds, no other GPU work).

## Review Focus
1. A finished run reused for another design: every run's cache record names its options, size, seed and token budget, and a mismatch trains again (test in Task 2).
2. The lab's class left in the trainer after a run, so a later run silently trains design 2: the swap is a context manager that restores the trainer whatever happens, and the lab asserts it is restored after every run (test in Task 2).
3. A lab checkpoint read as a product checkpoint: lab runs write a `lab_arch.json` beside the weights, and `load_lab_model` refuses a folder without it; the product `load_checkpoint` on a lab folder whose state dict does not fit fails loudly, never loads part of it (test in Task 1).
4. Timing taken while something else used the card: the timing phase measures the baseline twice per size and refuses to go on if the two differ by more than 5%; the equal-time budget is never below 50% or above 150% of the baseline's tokens (tests in Task 2).
5. The KV cache: a design-2 model answering with the cache must give the same tokens as without it, or every reply in the game would differ from what was measured (test in Task 1).

---

### Task 1: the scratch model class
**Files:**
- Create: `docs/handoff/next-level/arch_lab.py` (this task: the model part only)
- Test: `ml/tests/test_arch_lab.py` (loads the lab file by path, as `test_growth_lab.py` does)

**Interfaces — produces:**
- `@dataclass(frozen=True) class ArchOptions: zero_out: bool = False; qk_norm: bool = False; emb_norm: bool = False; relu2: bool = False; rope_base: float = 10000.0` with `ArchOptions.all() -> ArchOptions` (every change on, `rope_base=500.0`) and `name -> str` ("base", "zero", "qk", "emb", "relu2", "rope", "all" for the seven variants the screen uses; `VARIANTS: dict[str, ArchOptions]`).
- `class LabTransformer(Transformer)`: `__init__(self, shape: ModelShape, vocab_size: int, options: ArchOptions)`; parameter names stay the product's where a module exists in both (`tok_emb`, `blocks[i].attn.wq/wk/wv/wo`, `blocks[i].mlp.w_up/w_down`, norms); new parameters: `blocks[i].attn.q_norm.weight`, `blocks[i].attn.k_norm.weight` (shape `[HEAD_DIM]`, start at 1), `emb_norm.weight` (shape `[d_model]`); with `relu2` the feed-forward has `w_up` (`d_model x h2`) and `w_down` (`h2 x d_model`) with `h2` = 1.5 x `shape.ffn_hidden` rounded to a multiple of 32, and no `w_gate`.
- RoPE base: the lab builds its own RoPE tables with `options.rope_base` (the product's module constant `ROPE_BASE` is not patched).
- `save_lab_arch(out_dir: Path, options: ArchOptions) -> None` (writes `lab_arch.json`) and `load_lab_model(out_dir: Path, device) -> LabTransformer` (reads `lab_arch.json`, the run's meta for the shape and the weights; refuses a folder without `lab_arch.json`).

- [x] **Step 1: failing tests** in `ml/tests/test_arch_lab.py`:
  - `test_base_options_give_the_product_model`: with `ArchOptions()` and the same `torch.manual_seed`, `LabTransformer` has exactly the product `Transformer`'s parameter names and shapes, equal initial values and equal logits on random tokens (`torch.equal`).
  - `test_each_change_does_what_it_says`: `zero_out` → every `wo` and `w_down` is all zero at the start; `qk_norm` → logits unchanged when `wq` and `wk` of one block are multiplied by 3 (to 1e-5); `emb_norm` → the hidden state after the embedding has root-mean-square 1 per position (to 1e-4) before the first block; `relu2` → no `w_gate`, parameter count within 1% of the base model's; `rope_base=500.0` → the lab's cos table at position 1 equals `cos(1 * 500 ** (-2i/HEAD_DIM))`.
  - `test_the_optimizer_takes_every_new_parameter_once`: `trainer._make_optimizer(LabTransformer(..., ArchOptions.all()), style)` puts every 2-D weight but the embedding in the Muon group, the embedding alone in its group, every 1-D gain (the new norms included) in the gains group; the union is every parameter, each once.
  - `test_the_cache_gives_the_same_reply` for `ArchOptions.all()`: greedy generation of 20 tokens through `TorchLM.generate` equals greedy decoding by repeated full forward passes without a cache.
  - `test_a_lab_folder_is_never_a_product_checkpoint`: a saved all-on lab model loads with `load_lab_model` to equal logits; `load_lab_model` on a folder without `lab_arch.json` raises; the product `load_checkpoint` on the lab folder raises (no partial load).
- [x] **Step 2: run them, see them fail:** `cd ml && python -m uv run pytest tests/test_arch_lab.py -q -p no:cacheprovider` → errors (no lab file).
- [x] **Step 3: implement** the model part of `arch_lab.py` (subclass `Attention`, the feed-forward and `Block` as needed; reuse the product's `RMSNorm`, `_apply_rope`, `_attention_mask`, KV cache).
- [x] **Step 4: pass:** the same command → all pass; `python -m uv run ruff check ../docs/handoff/next-level/arch_lab.py tests/test_arch_lab.py` clean.
- [x] **Step 5: commit** `docs: the brain-design lab, its model class`.

### Task 2: the screen at equal time
**Files:**
- Modify: `docs/handoff/next-level/arch_lab.py` (the runs, the timing, the measures, the CLI)
- Test: `ml/tests/test_arch_lab.py`

**Interfaces — consumes:** Task 1's `ArchOptions`, `VARIANTS`, `LabTransformer`, `save_lab_arch`, `load_lab_model`; the scorecard's `_Card.config`, `TINY`, `SMALL`, `MEDIUM`, `STEP_TOKENS`, `BALANCED_MIX`, `yardstick_loss`, and the bench path `growth_lab.py` uses (`run_benchmarks` with the reference judge and novelty index, `MEASURE_SEED`).
**Produces:** `python ../docs/handoff/next-level/arch_lab.py --out runs/arch-lab [--sizes tiny small medium] [--seeds 1 2] [--variants base zero qk emb relu2 rope all] [--bench base all]`, writing `--out/results.json` and printing the tables.

- `lab_design(options) -> ContextManager`: inside it, the trainer builds `LabTransformer(shape, vocab, options)` where it builds `Transformer` today (patch the name the trainer calls at `ml/src/airace_ml/train/trainer.py:287`); restored on exit whatever happens; `assert_restored()` checks it.
- Timing phase, per size: 100 training steps of each variant (a throw-away run in `--out/timing/`), tokens per second from the trainer's own progress events after the first 20 steps; the baseline twice, and the screen stops with a clear message if its two speeds differ by more than 5%.
- Equal-time budget of a variant: `round(base_tokens * variant_speed / base_speed)` to a whole number of steps, clamped to 50%–150% of the baseline's tokens (and a warning when clamped).
- Runs: `--out/runs/<size>-<variant>-s<seed>/`, each with a `lab_run.json` key of options, size, seed, token budget, data identity and training-code digest; reused only when the key matches.
- Measures per run: the trainer's held-out mean (data2's own held-out text), `yardstick_loss` (text 1's conversations, `--yardstick-root ../data`), training seconds, the budget; bench-v1 overall for the variants named in `--bench`.
- Tables: per size, each variant's mean over seeds and its difference to `base`; then the gate line, word for word: "all together: better by ≥ 0.04 held-out loss on every size: yes/no; by ≥ 3 bench points on every size: yes/no", and the same line for each single change.

- [x] **Step 1: failing tests:** `test_the_screen_runs_on_the_tiny_fixture` (the CLI with hidden `--tiny-root DIR --device cpu` options on the trainer tests' tiny data and shapes, one seed, variants `base` and `all`: exit 0, `results.json` holds both, the trainer is restored afterwards); `test_a_run_of_one_design_is_never_reused_for_another` (a second call with `all` replaced by `qk` trains `qk` and reuses `base`); `test_budgets_follow_speed_and_stay_in_bounds` (`base` 1,000 tokens/s and a variant at 800 get 80% of the steps; a variant at 300 gets 50%, at 2,000 gets 150%); `test_unsteady_timing_stops_the_screen` (two baseline speeds 6% apart raise the lab's error).
- [x] **Step 2: run, see them fail.**
- [x] **Step 3: implement.**
- [x] **Step 4: pass:** `python -m uv run pytest tests/test_arch_lab.py -q -p no:cacheprovider`; ruff clean.
- [x] **Step 5: commit** `docs: the brain-design screen at equal time`.

### Task 3: run the screen and decide (the lead)
- [x] **Step 1:** nothing else running; from `ml/`: `AIRACE_DATA=../data2 python -m uv run python ../docs/handoff/next-level/arch_lab.py --out runs/arch-lab --bench base all > runs/arch-lab.log 2>&1` (about five hours; it keeps the PC awake).
- [x] **Step 2:** apply the gate in Global Constraints. Write ADR 0010 section 7 "Brain design 2": the timing table, the loss and bench table per size, the gate lines, and the decision (built, built for one change only, or dropped) with the reason in one sentence.
- [x] **Step 3:** if dropped: plan C (the release) is next. If built: write plan B2 (design 2 in the product: `arch: 2` in the checkpoint meta; brand-new AIs use it everywhere and an AI that continues an earlier one keeps its family; growth, the KV cache and exact resume proven again by the tests of design 1; the helper carries both; the engine raised) from spec section 5.5's last paragraph.
- [x] **Step 4:** ledger, `docs/progress.md`, push.

**Outcome (2026-11-22):** dropped; ADR 0010 section 7. Plan C (the release) is next.
