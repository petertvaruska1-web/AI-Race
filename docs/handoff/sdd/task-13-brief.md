### Task 13: Content text processing: normalize, quality, noise, dedup, topics

**Files:**
- Create: `ml/src/airace_content/textproc.py`, `noise.py`, `dedup.py`, `topics.py`
- Test: `ml/tests/content/test_textproc.py`, `ml/tests/content/test_noise_dedup.py`

**Interfaces:**
- Consumes: `TOPICS`, `NOISE_KINDS` (Task 6); KB and `FalseFactPlan` (Task 10).
- Produces in `textproc.py`:
  - `normalize_text(s: str) -> str`: NFC; `\r\n` and `\r` become `\n`; control characters other than `\n` and `\t` are removed; runs of spaces collapse; trailing spaces are stripped; 3+ newlines become 2.
  - `words(s: str) -> list[str]`: lowercase alphabetic words via regex `[a-z']+`.
  - `build_vocab(texts: Iterable[str], top_n: int) -> set[str]`
  - `simplicity(text: str, simple_vocab: set[str]) -> float`: `in_vocab_ratio * min(1, 18 / mean_sentence_words)`; 0 for empty text.
  - `quality_score(text: str, known_vocab: set[str]) -> float` in [0, 1]:
    - `a = clip((alpha_ratio - 0.6) / 0.3)`, where alpha_ratio is the share of letters among non-space characters
    - `k = clip((known_ratio - 0.5) / 0.4)`, over words of length ≥ 2
    - `r = 1 - repeated_line_ratio`
    - `g = 1 - clip(20 * garble_ratio)`, where garble_ratio is the share of characters in `MOJIBAKE_CHARS` = set("ÃÂâ€™œ¢�¤¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾")
    - `p = 1 - min(1, spam_hits / 3)`, counting case-insensitive `SPAM_MARKERS` = ("click here", "buy now", "free!!!", "!!!", "$$$", "http://", "https://", "subscribe", "limited offer", "act now")
    - **Result:** `k * g * p * (0.5 + 0.5 a) * (0.5 + 0.5 r)`; 0 for empty text.
- Produces in `noise.py`:
  - `@dataclass class NoiseRates(typo=0.0, spam=0.0, boilerplate=0.0, garbled=0.0, duplicate=0.0, false_fact=0.0)`
  - `@dataclass class NoisyDoc(text: str, noise_kind: int, false_fact: bool)`
  - Primitives: `add_typos(text, rng, rate=0.03) -> str`, `make_spam(rng) -> str`, `add_boilerplate(text, rng) -> str`, `garble(text, rng, rate=0.08) -> str`
  - `inject_noise(texts: list[str], rates: NoiseRates, rng, kb=None, plan: FalseFactPlan | None = None) -> list[NoisyDoc]`. Rates are fractions of the output count:
    - `duplicate` appends 2–6 copies of a random document, some with 1–2 typos
    - `spam` replaces a document with `make_spam`
    - `false_fact` inserts 1–3 `false_fact_sentence`s at sentence boundaries and sets `false_fact=True`
    - each document gets at most one noise kind
- Produces in `dedup.py`:
  - `simhash64(text: str) -> int` (word 3-shingles, blake2b 8-byte hashes)
  - `cluster_near_duplicates(texts: Sequence[str], max_hamming: int = 3) -> tuple[np.ndarray[int32], np.ndarray[bool]]`: exact matches on normalized-text hash, plus 4-band × 16-bit LSH candidates verified by Hamming distance, merged with union-find. Singletons get −1; canonical = the first occurrence.
- Produces in `topics.py`:
  - `TOPIC_KEYWORDS: dict[str, frozenset[str]]`: at least 15 keywords for every topic except "other".
  - `tag_topic(text: str) -> int`: index into `TOPICS`; argmax of keyword hits; "other" when there are none.

- [ ] **Step 1: Write the failing tests:**

```python
# test_textproc.py
from airace_content.textproc import normalize_text, quality_score, simplicity, build_vocab
from airace_content.topics import tag_topic
from airace_ml.data.corpus import TOPICS

CLEAN = "The little dog ran to the park. It played with a red ball. Then it went home to sleep."
VOCAB = build_vocab([CLEAN * 3, "the a an and to of in it is was he she they ball park home sleep"], top_n=5000)

def test_normalize():
    assert normalize_text("a\r\nb\x07  c\n\n\n\nd  ") == "a\nb c\n\nd"

def test_quality_ordering():
    q_clean = quality_score(CLEAN, VOCAB)
    q_typo = quality_score("Teh littel dgo rna to teh prak. It plyaed wiht a rde blal.", VOCAB)
    q_spam = quality_score("CLICK HERE!!! buy now $$$ http://x.y limited offer act now", VOCAB)
    q_garb = quality_score("ThÃ© littlÃ© dÃ¶g rÃ¤n tÃ¶ thÃ© pÃ¤rk Â¤Â¦Â§", VOCAB)
    assert q_clean > 0.8 and q_clean > q_typo and q_spam < 0.3 and q_garb < 0.3 and quality_score("", VOCAB) == 0

def test_simplicity():
    assert simplicity(CLEAN, VOCAB) > simplicity("Notwithstanding epistemological considerations, the hermeneutic paradigm persists.", VOCAB)

def test_topics():
    assert TOPICS[tag_topic("The cat and the dog chased a bird near the cow.")] == "animals"
    assert TOPICS[tag_topic("zzz qqq")] == "other"
```

```python
# test_noise_dedup.py
import numpy as np
from airace_content.dedup import cluster_near_duplicates
from airace_content.noise import NoiseRates, inject_noise
from airace_ml.data.corpus import NOISE_KINDS
from airace_ml.skills.facts import plan_false_facts
from airace_ml.skills.kb import load_kb

def _texts(n):
    rng = np.random.default_rng(0); words = "cat dog sun tree red blue run jump big small happy sad".split()
    return [" ".join(rng.choice(words, 40)) + "." for _ in range(n)]

def test_noise_rates_and_tags():
    kb = load_kb(); plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    out = inject_noise(_texts(2000), NoiseRates(typo=0.1, spam=0.05, garbled=0.05, duplicate=0.05, false_fact=0.05),
                       np.random.default_rng(1), kb, plan)
    kinds = np.array([d.noise_kind for d in out])
    for name, rate in [("typo", 0.1), ("spam", 0.05), ("garbled", 0.05), ("false_fact", 0.05)]:
        share = (kinds == NOISE_KINDS.index(name)).mean(); assert 0.7 * rate < share < 1.3 * rate
    assert all(d.false_fact == (d.noise_kind == NOISE_KINDS.index("false_fact")) for d in out)

def test_dedup_finds_injected_duplicates_without_false_positives():
    base = _texts(500); cl, canon = cluster_near_duplicates(base)
    assert (cl == -1).mean() > 0.99
    dup = base + [base[3], base[3].replace("cat", "cta", 1), base[7]]
    cl, canon = cluster_near_duplicates(dup)
    assert cl[500] == cl[3] == cl[501] != -1 and cl[502] == cl[7] != -1 and canon[3] and not canon[500]
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/content -v`. Expected: FAIL.

- [ ] **Step 3: Implement the four modules to the interfaces above.**

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/content -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(content): normalization, quality scoring, tagged noise injection, near-dup clustering, topics"`.

---

