### Task 17: Novelty index, reference judge, creativity

**Files:**
- Create: `ml/src/airace_ml/evals/novelty.py`, `judge.py`, `creativity.py`
- Modify: `ml/src/airace_ml/evals/suite.py` (wire in creativity)
- Test: `ml/tests/test_creativity.py`

**Interfaces:**
- Consumes: `Corpus`, `heldout_docs`; `train_run`, `TrainRunConfig`; `load_checkpoint`; `TorchLM`; `CategoryScore`, `ItemResult`.
- Produces in `novelty.py`:
  - `ngram_hashes(ids: Sequence[int], n: int = 8) -> np.ndarray[uint64]`: polynomial hash over the window with base 1,000,003, mod 2⁶⁴, vectorized.
  - `class NoveltyIndex(hashes: np.ndarray, n: int = 8, sample_mod: int = 8)`:
    - `build(token_arrays: Iterable[np.ndarray], n=8, sample_mod=8) -> NoveltyIndex` (classmethod)
    - `save(path)`, `load(path)` (classmethod)
    - `novelty(ids) -> float`: the fraction of the text's sampled hashes (`hash % sample_mod == 0`) absent from the sorted index; 0.0 when there are no sampled hashes.
  - `build_novelty_index(data_root: Path) -> Path`: all non-held-out documents of all 8 corpora, written to `novelty_path(data_root)`.
- Produces in `judge.py`:
  - `@dataclass class JudgeCalibration(creative_p10: float, creative_p90: float, conv_reply_p90: float)`
  - `class Judge`:
    - `load(dir: Path, device=None) -> Judge` (classmethod)
    - `nll_per_token(token_lists: list[list[int]]) -> list[float]`: mean NLL over the given tokens, conditioned on a leading `bos`; `inf` for an empty list.
    - attributes `calibration`, `tok`
  - `judge_train_config(seed: int = 1234) -> TrainRunConfig`: shape `(8, 384, 256)`, `token_budget=120_000_000`, equal mixture over the 8 datasets, `PrepConfig("thorough", dedup=True, fact_check=True)`, `boldness=0.4`, `batch_tokens=32768`, `run_id="judge-v1"`.
  - `calibrate(judge: Judge, data_root: Path) -> JudgeCalibration`: 10th/90th percentile NLL of the first 96 tokens of up to 500 held-out creative documents, plus the 90th percentile NLL of up to 500 held-out conversation AI-reply segments.
  - `build_judge(data_root: Path, *, device=None, on_event=None, token_budget: int | None = None) -> Path`: trains into `judge_dir(data_root)` and writes `calibration.json`.
  - `is_well_formed(reply: str, judge: Judge) -> bool`: ≥ 4 words, no word 3-gram occurring ≥ 3 times, and `nll_per_token(encode(reply)) <= conv_reply_p90`.
- Produces in `creativity.py`:
  - `STORY_PROMPTS: tuple[str, ...]`: exactly 24 chat prompts of the form "Write a short story about …".
  - `score_creativity(lm, tok, judge, novelty, *, seed: int = 0, max_new_tokens: int = 96) -> tuple[CategoryScore, list[ItemResult]]`:
    1. One sampled story per prompt (temperature 0.9, top_p 0.95, seed + i).
    2. `c = clip((p90 - nll) / (p90 - p10), 0, 1)` using the judge's creative calibration.
    3. `nov = novelty.novelty(ids)`.
    4. `story = c * (0.4 + 0.6 * nov)`.
    5. `d` = distinct-2 over all stories' word bigrams (0 if none).
    6. **Score:** `100 * mean(story) * (0.5 + 0.5 * d)`. `raw` = `mean(story)`. Empty stories score 0.

- [ ] **Step 1: Write the failing tests:**

```python
import math
import numpy as np, pytest
from airace_ml.evals.creativity import STORY_PROMPTS, score_creativity
from airace_ml.evals.judge import JudgeCalibration, is_well_formed, judge_train_config
from airace_ml.evals.novelty import NoveltyIndex
from tests.fakes import ScriptedLM

class StubJudge:
    def __init__(self, tok, nll): self.tok, self._nll = tok, nll
    calibration = JudgeCalibration(creative_p10=2.0, creative_p90=6.0, conv_reply_p90=5.0)
    def nll_per_token(self, lists): return [self._nll(l) for l in lists]

CORPUS_TEXT = "once upon a time a little bunny hopped to the big green hill and ate a carrot"

def test_novelty_index(tiny_tok, tmp_path):
    ids = np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)
    idx = NoveltyIndex.build([ids], sample_mod=1)
    assert idx.novelty(ids.tolist()) == 0.0 and idx.novelty([5, 6]) == 0.0
    rnd = np.random.default_rng(0).integers(20, 500, 200).tolist(); assert idx.novelty(rnd) > 0.95
    idx.save(tmp_path / "i.npy"); assert NoveltyIndex.load(tmp_path / "i.npy").novelty(rnd) == idx.novelty(rnd)

def test_creativity_ordering(tiny_tok):
    idx = NoveltyIndex.build([np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)], sample_mod=1)
    fluent = StubJudge(tiny_tok, lambda l: 2.5); gib = StubJudge(tiny_tok, lambda l: 9.0)
    novel = ScriptedLM(tiny_tok, reply=lambda p: f"a curious robot {len(p)} painted purple stars across the quiet ocean sky")
    copy = ScriptedLM(tiny_tok, reply=lambda p: CORPUS_TEXT)
    s_novel = score_creativity(novel, tiny_tok, fluent, idx)[0].score
    s_copy = score_creativity(copy, tiny_tok, fluent, idx)[0].score
    s_gib = score_creativity(novel, tiny_tok, gib, idx)[0].score
    s_empty = score_creativity(ScriptedLM(tiny_tok), tiny_tok, fluent, idx)[0].score  # Review Focus 4
    assert s_novel > s_copy > s_gib >= 0 and s_empty == 0 and len(STORY_PROMPTS) == 24

def test_well_formed(tiny_tok):
    j = StubJudge(tiny_tok, lambda l: 3.0)
    assert is_well_formed("I like to play in the park.", j)
    assert not is_well_formed("hi there", j) and not is_well_formed("go go go go go go go go go", j)
    assert not is_well_formed("I like to play in the park.", StubJudge(tiny_tok, lambda l: 8.0))

def test_judge_config_valid():
    c = judge_train_config(); c.validate(); assert c.shape.n_layer == 8 and c.token_budget == 120_000_000

@pytest.mark.slow
def test_build_judge_smoke(tiny_data_root):
    from airace_ml.evals.judge import Judge, build_judge
    d = build_judge(tiny_data_root, token_budget=32768 * 4); j = Judge.load(d)
    assert math.isfinite(j.calibration.creative_p90)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/test_creativity.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement the modules, and wire creativity into `run_benchmarks`.**

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/test_creativity.py tests/test_bench.py -v` and `uv run pytest -m slow tests/test_creativity.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(ml): novelty index, reference judge, creativity scoring"`.

---

