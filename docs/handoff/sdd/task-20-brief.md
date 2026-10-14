### Task 20: Feasibility gate (spec §4.12)

**Files:**
- Create: `ml/src/airace_ml/experiments/__init__.py`, `configs.py`, `gate.py`, `report.py`
- Create (from the real run): `docs/reports/m1-feasibility.md`
- Modify: `ml/src/airace_ml/cli.py` (wire up `gate`), `docs/progress.md`, `CLAUDE.md` (Commands)
- Test: `ml/tests/test_gate.py`

**Interfaces:**
- Consumes: everything above.
- Produces in `configs.py`:
  - `STARTER_SHAPE = ModelShape(4, 128, 256)`, `EARLY_SHAPE = ModelShape(6, 192, 256)`, `STARTER_BUDGET = 4_194_304`, `EARLY_BUDGET = 8_388_608`
  - `BALANCED_MIX`: 1.0 on each of the 8 datasets
  - `mix_heavy(target: str, share: float = 0.6) -> dict[str, float]`: target = share, the other 7 = (1 − share) / 7
  - `TARGET_CATEGORY = {"creative": "creativity", "code": "coding", "facts": "knowledge", "conversations": "instruction"}`
- Produces in `gate.py`:
  - `@dataclass class GateCriterion(id: str, title: str, passed: bool, detail: str, data: dict)`
  - Pure evaluators:

    | Evaluator | id | Passes when |
    |---|---|---|
    | `eval_speed(gpu_seconds: float, cpu_seconds: float \| None)` | `G1` | gpu ≤ 90 and (cpu is None or cpu ≤ 480) |
    | `eval_legibility(flags: list[bool])` | `G2` | mean ≥ 0.7, with exactly 20 flags |
    | `eval_differentiation(scores: dict[str, list[dict[str, float]]])` | `G3` | for every target t, with category k = `TARGET_CATEGORY[t]` and the per-seed score lists: `mean(t,k) > max_{u≠t} mean(u,k) + 2 * max(std(t,k), std(u*,k))`, where u* is the best rival |
    | `eval_seed_variation(seed_dists: list[float], mix_dists: list[float])` | `G4` | `mean(seed) > 0.05` and `mean(seed) < mean(mix)` |
    | `eval_growth(max_logit_diff: float, grown_loss: float, ungrown_loss: float)` | `G5` | diff ≤ 1e-4 and grown < ungrown |
    | `eval_forgetting(base: dict, no_replay: dict, replay: dict)` | `G6` | drop = Σ(base − no_replay) over creativity+language ≥ 3, and (Σreplay − Σno_replay) / drop ≥ 0.5 |
    | `eval_prep(garble_dirty, garble_clean, false_unchecked, false_checked)` | `G7` | clean ≤ 0.8 × dirty and checked ≤ 0.8 × unchecked |

  - `garble_rate(texts: list[str], known_vocab: set[str]) -> float`: fraction of words not in `known_vocab`; 0 if there are no words.
  - `false_fact_rate(lm, tok, kb, plan: FalseFactPlan) -> float`: for each planned (subject, relation) pair, render `relation.train_templates[0]` up to `{o}` as the context; the fact counts as false-preferred when `score(" " + false) > score(" " + true)`. Returns the fraction that are false-preferred.
  - `@dataclass class GateOutcome(criteria: list[GateCriterion], transcripts: dict[str, list[tuple[str, str]]], raw: dict)`
  - `run_gate(data_root: Path, out_dir: Path, *, seeds: Sequence[int] = (1, 2, 3), device=None, include_cpu_speed: bool = False, quick: bool = False, on_progress: Callable[[str], None] = print) -> GateOutcome`. `quick=True` swaps in shape `(2, 64, 64)`, a budget of 1024 × 20 tokens, `batch_tokens=1024` and a single seed, for smoke tests. Experiments:

    | Criterion | Experiment |
    |---|---|
    | G1 speed | `BALANCED_MIX`, `STARTER_SHAPE`, `STARTER_BUDGET`, timed, on GPU (and CPU when `include_cpu_speed`) |
    | G2 legibility | balanced starter, seed 1; 20 open/help probe replies via `chat_reply`; `is_well_formed` each; keep the transcript |
    | G3 differentiation | 4 `mix_heavy` targets × seeds, at `EARLY_SHAPE` / `EARLY_BUDGET`; full bench with creativity |
    | G4 seed variation | balanced `EARLY_SHAPE` × seeds; fingerprints; pairwise distances using all the gate's fingerprints as the population, compared with distances between the G3 mixes (seed 1) |
    | G5 growth | starter model (seed 1); max abs logit difference after `grow` to `EARLY_SHAPE` on 4 held-out sequences; continue +`STARTER_BUDGET` grown vs. ungrown on `BALANCED_MIX`; compare the final mean held-out loss |
    | G6 forgetting | `mix_heavy("creative")` at `EARLY_SHAPE` (seed 1) = base; continue +`STARTER_BUDGET` with mixture `{"code": 1}` and replay 0 vs 0.3; bench language + creativity |
    | G7 prep | (a) `mix_heavy("web", 0.7)` with `PrepConfig("light")` vs `PrepConfig("thorough", dedup=True)`, measuring `garble_rate` of 60 `complete()` samples from web held-out prefixes; (b) `{"web": 0.5, "facts": 0.5}` with `fact_check` False vs True, measuring `false_fact_rate`. Each at `STARTER_SHAPE` with 2× `STARTER_BUDGET` |

    All runs go under `out_dir/runs/<name>`.
- Produces in `report.py`:
  - `write_gate_report(outcome: GateOutcome, path: Path) -> None`: Markdown with a summary table (id, title, PASS/FAIL, detail); per-criterion data tables; G2 and G3 transcripts (≤ 10 exchanges each); the 4 differentiation models' bench tables side by side; and each model's `describe` words.

- [ ] **Step 1: Write the failing tests:**

```python
import pytest
from airace_ml.experiments.configs import TARGET_CATEGORY, mix_heavy
from airace_ml.experiments.gate import (GateCriterion, GateOutcome, eval_differentiation, eval_forgetting, eval_growth,
                                        eval_legibility, eval_prep, eval_seed_variation, eval_speed, garble_rate)
from airace_ml.experiments.report import write_gate_report

def test_mix_heavy():
    m = mix_heavy("code"); assert m["code"] == 0.6 and sum(m.values()) == pytest.approx(1) and len(m) == 8

def test_simple_evaluators():
    assert eval_speed(60, None).passed and not eval_speed(120, None).passed and not eval_speed(60, 900).passed
    assert eval_legibility([True] * 14 + [False] * 6).passed and not eval_legibility([True] * 13 + [False] * 7).passed
    assert eval_growth(1e-5, 2.0, 2.2).passed and not eval_growth(1e-3, 2.0, 2.2).passed
    assert eval_seed_variation([0.3, 0.4], [1.0, 1.2]).passed and not eval_seed_variation([0.0, 0.01], [1.0]).passed
    assert eval_prep(0.10, 0.05, 0.4, 0.1).passed and not eval_prep(0.10, 0.095, 0.4, 0.1).passed

def test_differentiation():
    def runs(best):
        return [{c: (60 if TARGET_CATEGORY[best] == c else 30) + d for c in TARGET_CATEGORY.values()} for d in (-1, 0, 1)]
    assert eval_differentiation({t: runs(t) for t in TARGET_CATEGORY}).passed
    noisy = {t: [{c: v for c in TARGET_CATEGORY.values()} for v in (10, 50, 90)] for t in TARGET_CATEGORY}
    assert not eval_differentiation(noisy).passed

def test_forgetting():
    base = {"creativity": 40, "language": 60}
    assert eval_forgetting(base, {"creativity": 20, "language": 50}, {"creativity": 32, "language": 56}).passed
    assert not eval_forgetting(base, {"creativity": 20, "language": 50}, {"creativity": 22, "language": 51}).passed
    assert not eval_forgetting(base, {"creativity": 39, "language": 60}, {"creativity": 40, "language": 60}).passed

def test_garble_rate():
    assert garble_rate(["the cat sat"], {"the", "cat", "sat"}) == 0 and garble_rate(["xq zz"], {"the"}) == 1
    assert garble_rate([""], {"a"}) == 0

def test_report(tmp_path):
    o = GateOutcome([GateCriterion("G1", "Speed", True, "60s", {})], {"G2": [("Hi", "Hello there friend.")]}, {})
    write_gate_report(o, tmp_path / "r.md"); text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "G1" in text and "PASS" in text and "Hello there friend." in text

@pytest.mark.slow
def test_run_gate_quick(tiny_data_root, tmp_path):
    from airace_ml.experiments.gate import run_gate
    out = run_gate(tiny_data_root, tmp_path, quick=True, on_progress=lambda s: None)
    assert {c.id for c in out.criteria} == {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}
```

`test_run_gate_quick` needs a tiny judge and novelty index. Make `tiny_data_root` lazily build them with `build_judge(token_budget=small)` and `build_novelty_index`, or skip creativity-dependent criteria with an explicit "skipped" detail when they're missing. Choose the lazy build, so the smoke test exercises the whole path.

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/test_gate.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `configs.py`, `gate.py` and `report.py`, and wire up the CLI `gate` command.**

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/test_gate.py -v && uv run pytest -m slow tests/test_gate.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(ml): feasibility gate experiments, evaluators and report"`.

- [ ] **Step 6: Build the real novelty index and judge.** Run `uv run airace-ml build-novelty-index`, then `uv run airace-ml build-judge`. The judge takes about 1 hour on the RTX 3050; run it in the background and monitor it. Expected: `data/novelty/v1/index.npy`, and `data/judge/v1/{model.safetensors,meta.json,calibration.json}` with finite calibration values. Chat with the judge (`airace-ml chat --model data/judge/v1`) to sanity-check its coherence.

- [ ] **Step 7: Run the gate.** Run `uv run airace-ml gate --out runs/gate-1 --cpu-speed` in the background (about 1–2 hours), then copy the report to `docs/reports/m1-feasibility.md`.

- [ ] **Step 8: If any criterion fails, iterate.**
  - Use superpowers:systematic-debugging to find the root cause (corpus content, recipe shares, simplicity filters, the boldness mapping, budgets, eval sensitivity).
  - Make the smallest principled change. Never special-case the gate.
  - Rebuild only what changed, then rerun just the failing criteria and the ones that depend on them.
  - Record each iteration (hypothesis, change, result) in an "Iterations" section of the report.
  - If a spec number has to change (a performance target, a shape, a budget), update the spec §4 table and say why in the report.

- [ ] **Step 9: Verify and close out the milestone.** Use superpowers:verification-before-completion:
  - all 7 criteria show PASS in the committed report
  - `uv run pytest -v` passes
  - `ruff` is clean

  Then update `docs/progress.md` (results, what's next: M2 starts by asking the D1 visual-direction question with mockups) and the `CLAUDE.md` Commands section. Run `git add docs CLAUDE.md ml && git commit -m "docs: M1 feasibility report — gate passed"`.
