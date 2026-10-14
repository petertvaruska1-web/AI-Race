### Task 16: Benchmark suite (bench-v1)

**Files:**
- Create: `ml/src/airace_ml/evals/__init__.py`, `scoring.py`, `suite.py`; `ml/tests/fakes.py`
- Test: `ml/tests/test_bench.py`

**Interfaces:**
- Consumes: `LanguageModel`, `ContinuationScore`, `Generation` (Task 5); all skill item builders (Tasks 10–12); `CHECKERS`.
- Produces:
  - `CATEGORIES = ("language", "reasoning", "pattern", "knowledge", "coding", "creativity", "consistency", "instruction")`
  - `normalize(acc: float, chance: float) -> float`: `clip((acc - chance) / (1 - chance), 0, 1) * 100`
  - `@dataclass class ItemResult(item_id: str, category: str, score: float, tags: tuple[str, ...], output: str | None = None)`
  - `@dataclass class CategoryScore(score: float, raw: float, n: int)`
  - `@dataclass class BenchReport`:
    - fields `suite: str`, `scores: dict[str, CategoryScore]`, `overall: float`, `missing: list[str]`, `items: list[ItemResult]`, `seconds: float`
    - `to_dict() -> dict` (JSON-safe)
    - `tag_breakdown() -> dict[str, tuple[float, int]]`: mean item score × 100 and the count per tag
  - `@dataclass class Suite(version: str, items: dict[str, list])`
  - `build_suite() -> Suite`, deterministic, with exactly these counts:

    | Category | Items |
    |---|---|
    | language | 200 `PairItem`s |
    | reasoning | 200 |
    | pattern | 150 |
    | knowledge | 200 |
    | coding | 150 |
    | consistency | 120 (40 groups × 3) |
    | instruction | 120 |

    Every item is built with `skill_rng(<category>, "bench")`.
  - `run_benchmarks(lm: LanguageModel, tok: Tok, *, suite: Suite | None = None, judge=None, novelty=None, categories: Sequence[str] = CATEGORIES, max_items_per_category: int | None = None, seed: int = 0) -> BenchReport`. Per item kind:

    | Item kind | Method | Item score | Chance |
    |---|---|---|---|
    | `PairItem` | compare the sum log-prob of `encode_doc(good)` vs `encode_doc(bad)` (empty context = `[bos]`) | 1 if good > bad | 0.5 |
    | `MCItem` | context `encode_doc(prompt)` (or `encode_chat([("user", prompt)], True)` if chat); continuations `encode(" " + option)`; argmax of mean log-prob per token (first index on ties) | 1 if correct | `1/len(options)` |
    | `ExactItem` | greedy generation; extract the text before the first `","`/`"\n"` (`first_item`) or before the first `"\n"` (`first_line`); strip; case-insensitive match | 1 if matched | 0 |
    | `CheckItem` | greedy generation; `CHECKERS[check](text, args)` | 1 if passed | 0 |
    | consistency | per group, the MC choice per member; agreement = fraction of equal pairs (option *text*) | per group | mean of `1/len(options)` |

    - Creativity is delegated to `evals.creativity.score_creativity` (Task 17) when both `judge` and `novelty` are given; otherwise `"creativity"` goes into `missing`.
    - `overall` = mean of the present category scores; 0 if none.
    - The suite version string is `"bench-v1"`.
  - `tests/fakes.py`: `class ScriptedLM(tok, ctx_len=256, reply: Callable[[list[int]], str] | None = None, score: Callable[[list[int], list[int]], float] | None = None)`. It implements `LanguageModel`: `generate` returns `tok.encode(reply(prompt))` with `top1_probs = token_probs = 0.5`; `score_continuations` returns `score(ctx, cont)`. Defaults are an empty reply and a 0.0 score. Also `answer_key_lm(tok, suite) -> ScriptedLM`, which scores the correct option or good sentence at 0.0 and everything else at −10, and replies with the item's answer or reference.

- [ ] **Step 1: Write the failing tests:**

```python
import json, math
import pytest
from airace_ml.evals.scoring import normalize
from airace_ml.evals.suite import CATEGORIES, build_suite, run_benchmarks
from tests.fakes import ScriptedLM, answer_key_lm

def test_normalize():
    assert normalize(0.5, 0.5) == 0 and normalize(1.0, 0.25) == 100 and normalize(0.1, 0.25) == 0
    assert normalize(0.625, 0.25) == pytest.approx(50)

def test_suite_deterministic_and_sized():
    a, b = build_suite(), build_suite()
    sizes = {k: len(v) for k, v in a.items.items()}
    assert sizes == {"language": 200, "reasoning": 200, "pattern": 150, "knowledge": 200, "coding": 150,
                     "consistency": 120, "instruction": 120}
    assert [i.id for i in a.items["coding"]] == [i.id for i in b.items["coding"]]

def test_answer_key_scores_100(tiny_tok):
    suite = build_suite(); rep = run_benchmarks(answer_key_lm(tiny_tok, suite), tiny_tok, suite=suite)
    for c in CATEGORIES:
        if c not in ("creativity", "consistency"):
            assert rep.scores[c].score == pytest.approx(100), c
    assert rep.missing == ["creativity"]

def test_constant_model_near_zero(tiny_tok):
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, max_items_per_category=40)
    assert all(s.score <= 25 for s in rep.scores.values())  # ties resolve to the first option → chance level

@pytest.mark.parametrize("reply", [lambda p: "", lambda p: "<|end|><|bos|>", lambda p: "the the the " * 30])
def test_degenerate_outputs(tiny_tok, reply):  # Review Focus 4
    rep = run_benchmarks(ScriptedLM(tiny_tok, reply=reply), tiny_tok, max_items_per_category=20)
    assert all(math.isfinite(s.score) for s in rep.scores.values()) and math.isfinite(rep.overall)

def test_report_serializable_and_tags(tiny_tok):
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, max_items_per_category=10)
    json.dumps(rep.to_dict()); assert all(0 <= v[0] <= 100 for v in rep.tag_breakdown().values())

def test_real_tiny_model_end_to_end(tiny_lm, tiny_tok):
    rep = run_benchmarks(tiny_lm, tiny_tok, max_items_per_category=5)
    assert rep.seconds > 0 and math.isfinite(rep.overall)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/test_bench.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `scoring.py`, `suite.py` and `tests/fakes.py`.** Batch all scoring and generation calls per category through the `LanguageModel` interface.

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/test_bench.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(ml): bench-v1 suite with normalized category scores and tag breakdowns"`.

---

