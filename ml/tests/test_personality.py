import math
from collections import Counter

from airace_ml.personality.fingerprint import (
    TRAITS,
    Fingerprint,
    describe,
    fingerprint_distance,
    load_probes,
    measure_fingerprint,
)
from tests.fakes import ScriptedLM


def test_probes():
    ps = load_probes()
    c = Counter(p.kind for p in ps)
    assert len(ps) == 60 and c == {
        "open": 20,
        "factual": 15,
        "creative": 10,
        "help": 10,
        "opinion": 5,
    }
    assert all(p.answers and p.wrong_answers for p in ps if p.kind == "factual")


def test_traits_respond_to_behavior(tiny_tok):
    talky = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "well " * 40), tiny_tok, k=2)
    terse = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "ok"), tiny_tok, k=2)
    assert (
        talky.traits["verbosity"] > terse.traits["verbosity"] and terse.traits["steadiness"] == 1.0
    )
    formal = measure_fingerprint(
        ScriptedLM(tiny_tok, reply=lambda p: "Therefore, however, additionally."), tiny_tok, k=1
    )
    casual = measure_fingerprint(
        ScriptedLM(tiny_tok, reply=lambda p: "yeah lol gonna hey"), tiny_tok, k=1
    )
    assert formal.traits["register"] > 0 > casual.traits["register"]


def test_precision_with_answering_fake(tiny_tok):
    probes = {p.text: p for p in load_probes()}

    def reply(ids):
        text = tiny_tok.decode(ids)
        hit = next((p for t, p in probes.items() if t in text), None)
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
    assert fingerprint_distance(hi, hi, pop) == 0 and fingerprint_distance(
        hi, pop[0], pop
    ) == fingerprint_distance(pop[0], hi, pop)
