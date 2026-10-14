### Task 18: Personality fingerprint

**Files:**
- Create: `ml/src/airace_ml/personality/__init__.py`, `fingerprint.py`, `probes.json`
- Test: `ml/tests/test_personality.py`

**Interfaces:**
- Consumes: `LanguageModel`, `Tok`, `encode_chat`; KB (for factual probes).
- Produces:
  - `TRAITS = ("verbosity", "confidence", "inventiveness", "steadiness", "precision", "boldness", "slip_rate", "register", "warmth", "repetitiveness")`
  - `@dataclass class Probe(id: str, text: str, kind: Literal["open", "factual", "creative", "help", "opinion"], answers: list[str] | None = None, wrong_answers: list[str] | None = None)`
  - `load_probes() -> list[Probe]`: from `probes.json`, exactly 60 probes (open 20, factual 15, creative 10, help 10, opinion 5). Factual probes carry KB answers, plus `wrong_answers` = the other objects of the same relation.
  - `FORMAL_WORDS`, `CASUAL_WORDS`, `POSITIVE_WORDS`: frozensets of 30–60 words each.
  - `@dataclass class Fingerprint(traits: dict[str, float], samples: list[dict])`, with `to_dict` and `from_dict`.
  - `measure_fingerprint(lm, tok, *, probes=None, k=3, seed=0, temperature=0.8, top_p=0.95, max_new_tokens=64) -> Fingerprint`. Traits, all finite and 0 for empty input:

    | Trait | Formula |
    |---|---|
    | verbosity | mean word count |
    | confidence | mean of per-reply mean `top1_probs` |
    | inventiveness | distinct-2 of word bigrams over all replies |
    | steadiness | mean over probes of mean pairwise Jaccard of word sets (Jaccard(∅, ∅) = 1) |
    | precision | fraction of factual probes where any sample contains an answer (whole word, case-insensitive) |
    | boldness | fraction of replies with ≥ 3 words and repetitiveness < 0.5 |
    | slip_rate | fraction of factual replies containing a wrong answer |
    | register | `(formal - casual) / (formal + casual + 1)` |
    | warmth | `10 * (positive hits + "!" count) / max(1, total words)` |
    | repetitiveness | mean per-reply fraction of repeated word 3-grams |

  - `TRAIT_PHRASES: dict[str, tuple[str, str]]`: (low phrase, high phrase) per trait, e.g. verbosity → ("terse", "talkative").
  - `describe(fp: Fingerprint, population: Sequence[Fingerprint], low_pct: float = 15, high_pct: float = 85) -> list[str]`: phrases for traits outside the percentile band; empty when the population has fewer than 5 entries.
  - `fingerprint_distance(a: Fingerprint, b: Fingerprint, population: Sequence[Fingerprint]) -> float`: Euclidean distance over traits, each divided by the population std (floor 1e-6).

- [ ] **Step 1: Write the failing tests:**

```python
import math
from collections import Counter
from airace_ml.personality.fingerprint import (TRAITS, Fingerprint, describe, fingerprint_distance, load_probes,
                                              measure_fingerprint)
from tests.fakes import ScriptedLM

def test_probes():
    ps = load_probes(); c = Counter(p.kind for p in ps)
    assert len(ps) == 60 and c == {"open": 20, "factual": 15, "creative": 10, "help": 10, "opinion": 5}
    assert all(p.answers and p.wrong_answers for p in ps if p.kind == "factual")

def test_traits_respond_to_behavior(tiny_tok):
    talky = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "well " * 40), tiny_tok, k=2)
    terse = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "ok"), tiny_tok, k=2)
    assert talky.traits["verbosity"] > terse.traits["verbosity"] and terse.traits["steadiness"] == 1.0
    formal = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "Therefore, however, additionally."), tiny_tok, k=1)
    casual = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "yeah lol gonna hey"), tiny_tok, k=1)
    assert formal.traits["register"] > 0 > casual.traits["register"]

def test_precision_with_answering_fake(tiny_tok):
    probes = {p.text: p for p in load_probes()}
    def reply(ids):
        text = tiny_tok.decode(ids); hit = next((p for t, p in probes.items() if t in text), None)
        return f"It is {hit.answers[0]}." if hit and hit.kind == "factual" else "Hello friend."
    fp = measure_fingerprint(ScriptedLM(tiny_tok, reply=reply), tiny_tok, k=1)
    assert fp.traits["precision"] == 1.0 and fp.traits["slip_rate"] == 0.0

def test_empty_model_finite(tiny_tok):  # Review Focus 4
    fp = measure_fingerprint(ScriptedLM(tiny_tok), tiny_tok, k=2)
    assert set(fp.traits) == set(TRAITS) and all(math.isfinite(v) for v in fp.traits.values())

def test_describe_and_distance():
    pop = [Fingerprint({t: float(i) for t in TRAITS}, []) for i in range(10)]
    hi = Fingerprint({t: 100.0 for t in TRAITS}, [])
    assert "talkative" in describe(hi, pop) and describe(hi, pop[:3]) == []
    assert fingerprint_distance(hi, hi, pop) == 0 and fingerprint_distance(hi, pop[0], pop) == fingerprint_distance(pop[0], hi, pop)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/test_personality.py -v`. Expected: FAIL.

- [ ] **Step 3: Author `probes.json` and implement `fingerprint.py`.**

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/test_personality.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(ml): measured personality fingerprint with population-relative descriptions"`.

---

