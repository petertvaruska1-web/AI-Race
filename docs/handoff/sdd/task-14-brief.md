### Task 14: Sources, recipes and the build pipeline (offline-tested)

**Files:**
- Create: `ml/src/airace_content/sources.py`, `assemble.py`, `manifest.py`, `build.py`, `cli.py`, and fixtures `ml/tests/fixtures/sources/<source_id>.jsonl` (5–20 rows each, with the real field names below)
- Test: `ml/tests/content/test_build.py`

**Interfaces:**
- Consumes: Tasks 2, 6, 10–13.
- Produces:
  - `@dataclass(frozen=True) class SourceSpec(id: str, hf_path: str, hf_config: str | None, split: str, license: str, homepage: str)`
  - `SOURCES: dict[str, SourceSpec]`, using exactly these IDs, paths and fields:

    | ID | HF path / config / split | Fields used |
    |---|---|---|
    | `tinystories` | `roneneldan/TinyStories` / None / train | `text` |
    | `fineweb` | `HuggingFaceFW/fineweb` / sample-10BT / train | `text` |
    | `fineweb_edu` | `HuggingFaceFW/fineweb-edu` / sample-10BT / train | `text`, `int_score` |
    | `cosmo_khan` | `HuggingFaceTB/cosmopedia` / khanacademy / train | `text` |
    | `cosmo_wikihow` | `HuggingFaceTB/cosmopedia` / wikihow / train | `text` |
    | `cosmo_openstax` | `HuggingFaceTB/cosmopedia` / openstax / train | `text` |
    | `cosmo_stories` | `HuggingFaceTB/cosmopedia` / stories / train | `text` |
    | `soda` | `allenai/soda` / None / train | `dialogue`, `speakers` |
    | `everyday_conv` | `HuggingFaceTB/everyday-conversations-llama3.1-2k` / None / train_sft | `messages[{role, content}]` |
    | `gutenberg` | `sedthh/gutenberg_english` / None / train | `TEXT`, `METADATA` (JSON string) |
    | `gsm8k` | `openai/gsm8k` / main / train | `question`, `answer` |
    | `mbpp` | `google-research-datasets/mbpp` / full / train | `text`, `code` |
    | `simplewiki` | `wikimedia/wikipedia` / 20231101.simple / train | `title`, `text` |

    `license` takes the value from the dataset card, re-verified in Task 15.
  - `Fetch = Callable[[SourceSpec, int], Iterator[dict]]`
  - `hf_fetch(spec: SourceSpec, max_rows: int) -> Iterator[dict]`: `datasets.load_dataset(..., streaming=True)`, imported lazily.
  - `fixture_fetch(fixture_dir: Path) -> Fetch`
  - `row_to_content(source_id: str, row: dict) -> list[str | list[tuple[Role, str]]]` (0 or more pieces):
    - **gutenberg:** keep books whose METADATA subjects mention "Juvenile", "Fairy tales", "Fables" or "Children"; drop the Project Gutenberg header and footer; chunk into passages of ≤ 400 words.
    - **fineweb_edu:** keep rows with `int_score >= 3`.
    - **soda:** `speakers[0]` maps to "user" and every other speaker to "ai"; drop dialogues with fewer than 2 turns.
    - **everyday_conv:** role "user" maps to "user" and "assistant" to "ai"; drop system turns.
    - **gsm8k:** `"Question: {q}\nAnswer: {reasoning} The answer is {final}."`, with `<<…>>` spans removed and `####` handled.
    - **mbpp:** `"# {text}\n{code}"`, with tabs converted to 4 spaces and CRLF converted to LF.
    - **simplewiki:** `"{title}\n{first paragraphs up to 150 words}"`.
  - `@dataclass class Component(kind: Literal["source", "generator"], name: str, share: float, simplicity_min: float | None = None)`
  - `@dataclass class Recipe(dataset_id: str, components: list[Component], noise: NoiseRates)`
  - `RECIPES: dict[str, Recipe]`, with exactly these shares and noise rates:

    | Dataset | Components (share) | Noise |
    |---|---|---|
    | `web` | fineweb 1.0 (simplicity ≥ 0.35) | typo 0.15, spam 0.06, boilerplate 0.10, garbled 0.05, duplicate 0.08, false_fact 0.06 |
    | `books` | gutenberg 1.0 (simplicity ≥ 0.30) | typo 0.02, garbled 0.01, duplicate 0.02 |
    | `educational` | cosmo_khan 0.25, cosmo_wikihow 0.25, cosmo_openstax 0.20, fineweb_edu 0.30 (simplicity ≥ 0.35) | duplicate 0.02, false_fact 0.01 |
    | `conversations` | soda 0.55, everyday_conv 0.15, gen:fact_chat 0.15, gen:instructions 0.15 | typo 0.03 |
    | `code` | gen:code 0.80, mbpp 0.20 | duplicate 0.03 |
    | `reasoning` | gen:reasoning 0.60, gen:patterns 0.30, gsm8k 0.10 | none |
    | `facts` | gen:fact_prose 0.60, simplewiki 0.40 (simplicity ≥ 0.30) | duplicate 0.02 |
    | `creative` | tinystories 0.85, cosmo_stories 0.15 | typo 0.02, duplicate 0.03 |

  - `SCALES = {"tiny": {"docs_per_dataset": 150}, "full": {"target_tokens": {"web": 30e6, "books": 15e6, "educational": 20e6, "conversations": 20e6, "code": 10e6, "reasoning": 10e6, "facts": 12e6, "creative": 20e6}}}`. Full scale estimates tokens/doc from the first 200 accepted docs and keeps fetching until the target is met or the source is exhausted.
  - `@dataclass class AssembledDoc(content: str | list[tuple[Role, str]], noise_kind: int, false_fact: bool)`
  - `assemble_dataset(dataset_id: str, scale: str, fetch: Fetch, kb: KB, simple_vocab: set[str], plan: FalseFactPlan, rng: np.random.Generator) -> list[AssembledDoc]`
  - `@dataclass class BuildSummary(datasets: dict[str, dict], tokenizer_trained: bool, seconds: float)`
  - `build_corpus(scale: Literal["tiny", "full"], out_root: Path, fetch: Fetch, *, seed: int = 0, datasets: Sequence[str] = DATASET_IDS, retrain_tokenizer: bool = False) -> BuildSummary`. Pipeline:
    1. `simple_vocab` = top 6,000 words of tinystories text.
    2. `plan_false_facts(n=300)`.
    3. Assemble each dataset.
    4. `known_vocab` = top 30,000 words over the books, educational, facts and creative text.
    5. `quality_score` for every document.
    6. Dedup per dataset.
    7. `tag_topic`.
    8. Held-out: 1% of documents by seeded permutation, at least 8 per dataset.
    9. `purchase_rank` from a second permutation.
    10. Train the tokenizer only if `tokenizer_path(out_root)` is missing or `retrain_tokenizer` is set. It trains on up to 20 MB of text sampled evenly across datasets, with chats rendered as plain text.
    11. Encode with `encode_doc` / `encode_chat` and `write_corpus`.
    12. Write `manifest.json`, `false_facts.json`, `simple_vocab.txt` and `known_vocab.txt` into `corpus_dir(out_root)`.
  - CLI: `airace-content build --scale {tiny,full} [--datasets a,b] [--out PATH] [--retrain-tokenizer]` → `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests:**

```python
import json
from pathlib import Path
from airace_content.build import build_corpus
from airace_content.sources import fixture_fetch, row_to_content
from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.paths import corpus_dir, tokenizer_path

FIX = Path(__file__).parents[1] / "fixtures" / "sources"

def test_adapters():
    assert row_to_content("soda", {"dialogue": ["Hi", "Hey", "Bye"], "speakers": ["A", "B", "A"]}) == [
        [("user", "Hi"), ("ai", "Hey"), ("user", "Bye")]]
    g = row_to_content("gsm8k", {"question": "Q?", "answer": "2+2 = <<2+2=4>>4\n#### 4"})[0]
    assert "<<" not in g and g.endswith("The answer is 4.")
    assert row_to_content("fineweb_edu", {"text": "x", "int_score": 2}) == []
    assert "\t" not in row_to_content("mbpp", {"text": "t", "code": "def f():\r\n\treturn 1"})[0]

def test_tiny_build_end_to_end(tmp_path):
    s = build_corpus("tiny", tmp_path, fixture_fetch(FIX), seed=0)
    assert s.tokenizer_trained and tokenizer_path(tmp_path).exists()
    for ds in DATASET_IDS:
        c = Corpus.open(corpus_dir(tmp_path) / ds)
        assert c.n_docs > 0 and len(c.tags.quality) == c.n_docs and c.tags.heldout.sum() >= 1
    man = json.loads((corpus_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert all(src["license"] for src in man["sources"])
    assert json.loads((corpus_dir(tmp_path) / "false_facts.json").read_text(encoding="utf-8"))
    assert Corpus.open(corpus_dir(tmp_path) / "web").tags.noise_kind.any()

def test_tokenizer_is_frozen_on_rebuild(tmp_path):
    build_corpus("tiny", tmp_path, fixture_fetch(FIX), seed=0)
    before = tokenizer_path(tmp_path).read_bytes()
    s2 = build_corpus("tiny", tmp_path, fixture_fetch(FIX), seed=0, datasets=["web"])
    assert not s2.tokenizer_trained and tokenizer_path(tmp_path).read_bytes() == before
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/content/test_build.py -v`. Expected: FAIL.

- [ ] **Step 3: Write the fixture rows and implement the modules.** `hf_fetch` must not be imported by tests. With `fixture_fetch`, the tiny scale cycles through fixture rows as needed to reach 150 documents.

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/content -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(content): source adapters, dataset recipes, corpus build pipeline and CLI"`.

---

