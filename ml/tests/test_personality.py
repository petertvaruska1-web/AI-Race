import json
import math
from collections import Counter

import numpy as np
import pytest

from airace_ml.evals import creativity
from airace_ml.evals.creativity import STORY_PROMPTS, words
from airace_ml.infer.lm import Generation
from airace_ml.personality.fingerprint import (
    CASUAL_WORDS,
    FORMAL_WORDS,
    POSITIVE_WORDS,
    TRAIT_PHRASES,
    TRAITS,
    Fingerprint,
    Probe,
    answer_words,
    describe,
    fingerprint_distance,
    load_probes,
    measure_fingerprint,
)
from airace_ml.skills.kb import load_kb
from airace_ml.tokenizer import encode_chat
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


# ---------------------------------------------------------------------------------------------
# Tests added beyond the brief: what the brief and the controller's rulings state but the
# brief's tests do not pin.
# ---------------------------------------------------------------------------------------------


class FixedLM:
    """Answers the nth prompt with the nth given generation, and records every call."""

    ctx_len = 256

    def __init__(self, generations):
        self.generations = list(generations)
        self.calls = []

    def generate(self, prompts, *, max_new_tokens, temperature, top_p, seed, stop_ids=None):
        self.calls.append(
            {
                "prompts": prompts,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }
        )
        return list(self.generations)

    def score_continuations(self, contexts, continuations):
        raise NotImplementedError


def gen(tok, text, top1=0.5):
    ids = tok.encode(text)
    return Generation(ids, [0.5] * len(ids), [top1] * len(ids), True)


def lm_saying(tok, *texts):
    return FixedLM([gen(tok, t) for t in texts])


def open_probe(n=1):
    return Probe(f"open-{n}", f"Tell me thing number {n}.", "open")


def factual_probe(n, answers, wrong):
    return Probe(f"fact-{n}", f"What is fact number {n}?", "factual", answers, wrong)


def traits_of(tok, texts, probes, k):
    """The traits of a model that gives ``texts`` in order (probe by probe, k samples each)."""
    return measure_fingerprint(lm_saying(tok, *texts), tok, probes=probes, k=k).traits


# -- probes ------------------------------------------------------------------------------------


def test_probe_set_is_clean(tiny_tok):
    ps = load_probes()
    assert len({p.id for p in ps}) == 60 and len({p.text for p in ps}) == 60
    for p in ps:
        assert p.text.isascii() and p.text == p.text.strip() and 2 <= len(p.text.split()) <= 14
        assert tiny_tok.decode(tiny_tok.encode(p.text)) == p.text
        assert (p.answers is None) == (p.kind != "factual")
        assert not any(
            q.text != p.text and p.text in q.text for q in ps
        )  # the fakes find probes by text
    assert not {p.text for p in ps} & set(STORY_PROMPTS)


def test_load_probes_returns_fresh_copies_and_ignores_cwd(monkeypatch, tmp_path):
    first = load_probes()
    first[0].text = "changed"
    first.pop()
    monkeypatch.chdir(tmp_path)
    again = load_probes()
    assert len(again) == 60 and again[0].text != "changed"


def test_probes_json_ships_beside_the_module():
    from airace_ml.personality import fingerprint

    path = fingerprint.PROBES_PATH
    assert path.name == "probes.json" and path.parent == path.with_name("x").parent
    assert path.is_file() and path.parent.joinpath("__init__.py").is_file()
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 60


def _fact_of(kb, probe):
    """The one KB fact whose question template, filled in, is the probe's text."""
    found = [
        (relation, fact)
        for relation, spec in kb.relations.items()
        for fact in kb.facts_for(relation)
        for template in spec.question_templates
        if template.format(s=fact.subject) == probe.text
    ]
    assert len(found) == 1, probe.text
    return found[0]


def test_factual_probes_agree_with_the_kb():
    kb = load_kb()
    for probe in (p for p in load_probes() if p.kind == "factual"):
        relation, fact = _fact_of(kb, probe)
        spec = kb.relations[relation]
        assert spec.mc_safe and spec.exact_safe  # only relations whose answers cannot overlap
        assert probe.answers == kb.accepted_answers(fact)
        assert set(probe.wrong_answers) <= set(kb.wrong_objects(fact))
        assert not set(probe.wrong_answers) & set(probe.answers)
        assert len(set(probe.wrong_answers)) == len(probe.wrong_answers)
        # a true answer can never read as a slip, whatever form it takes
        said = [tuple(answer_words(a)) for a in probe.answers]
        for wrong in map(answer_words, probe.wrong_answers):
            assert not any(
                wrong == list(a[i : i + len(wrong)]) for a in said for i in range(len(a))
            )
        assert not set(words(fact.subject)) & {
            w for a in probe.wrong_answers for w in answer_words(a)
        }


def test_every_right_form_scores_and_every_wrong_answer_slips(tiny_tok):
    factual = [p for p in load_probes() if p.kind == "factual"]
    for probe in factual:
        for right in probe.answers:
            t = traits_of(tiny_tok, [f"The answer is {right}."], [probe], 1)
            assert (t["precision"], t["slip_rate"]) == (1.0, 0.0), right
        for wrong in probe.wrong_answers:
            t = traits_of(tiny_tok, [f"I think it is {wrong}!"], [probe], 1)
            assert (t["precision"], t["slip_rate"]) == (0.0, 1.0), wrong


def test_lexicons(tiny_tok):
    for lexicon in (FORMAL_WORDS, CASUAL_WORDS, POSITIVE_WORDS):
        assert isinstance(lexicon, frozenset) and 30 <= len(lexicon) <= 60
        assert all(w.isalpha() and w.isascii() and w == w.lower() for w in lexicon)
    assert not FORMAL_WORDS & CASUAL_WORDS
    assert {"therefore", "however", "additionally"} <= FORMAL_WORDS
    assert {"yeah", "lol", "gonna", "hey"} <= CASUAL_WORDS
    assert {"love", "happy", "great", "kind"} <= POSITIVE_WORDS


def test_once_upon_a_time_is_not_formal(tiny_tok):
    t = traits_of(tiny_tok, ["Once upon a time there was a small dragon."], [open_probe()], 1)
    assert t["register"] == 0.0
    # a model that opens only its stories this way is not formal either
    kinds = {p.text: p.kind for p in load_probes()}

    def reply(ids):
        text = tiny_tok.decode(ids)
        creative = any(kind == "creative" and t in text for t, kind in kinds.items())
        return "Once upon a time, a little bird sang to the moon." if creative else "Hello."

    fp = measure_fingerprint(ScriptedLM(tiny_tok, reply=reply), tiny_tok, k=1)
    assert sum(w.startswith("Once upon") for w in (s["text"] for s in fp.samples)) == 10
    assert fp.traits["register"] == 0.0


def test_formal_words_are_not_everyday_words():
    # words that turn up in children's stories and plain speech say nothing about register
    everyday = {"upon", "indeed", "shall", "unfortunately", "although", "otherwise", "concerning"}
    everyday |= {"provide", "ensure", "indicate", "require", "requires", "required", "appropriate"}
    everyday |= {"significant", "essentially"}
    assert not FORMAL_WORDS & everyday


def test_casual_words_have_no_everyday_sense():
    everyday = {"cool", "stuff", "guys", "buddy", "pal", "totally", "whatever", "awesome", "wow"}
    everyday |= {"okay", "ok", "oops", "yay", "yikes", "ugh", "hmm"}
    assert not CASUAL_WORDS & everyday


def test_positive_words_are_affective_only():
    everyday = {"good", "best", "friend", "friends", "warm", "bright", "sweet", "fine", "nice"}
    everyday |= {"fun", "brave", "lucky", "generous", "gentle", "please", "welcome"}
    everyday |= {"cozy", "hope", "beautiful"}
    assert not POSITIVE_WORDS & everyday
    assert {"love", "loved", "loves", "happy", "great", "kind", "glad", "thanks"} <= POSITIVE_WORDS


def test_register_lexicons_stay_silent_on_plain_text(fixture_texts):
    # the fixture corpus is plain stories, chats, facts and code: no formal or casual markers
    said = set(words(" ".join(fixture_texts)))
    assert not said & (FORMAL_WORDS | CASUAL_WORDS)


def test_a_model_that_only_restates_the_question_reads_neutral(tiny_tok):
    fp = measure_fingerprint(ScriptedLM(tiny_tok, reply=tiny_tok.decode), tiny_tok, k=1)
    assert [s["text"] for s in fp.samples] == [p.text for p in load_probes()]
    said = Counter(w for s in fp.samples for w in words(s["text"]))
    assert not set(said) & (FORMAL_WORDS | CASUAL_WORDS)
    # the one warm word a probe forces on anyone repeating it is "a new kind of fruit"
    assert {w: n for w, n in said.items() if w in POSITIVE_WORDS} == {"kind": 1}
    assert fp.traits["register"] == 0.0
    total = sum(said.values())
    assert fp.traits["warmth"] == pytest.approx(10 * 2 / total)  # "kind" and the "!" of "Hello!"
    assert fp.traits["warmth"] < 0.1


def test_trait_phrases_are_plain_language():
    assert list(TRAIT_PHRASES) == list(TRAITS)
    phrases = [p for pair in TRAIT_PHRASES.values() for p in pair]
    assert len(phrases) == 20 and len(set(phrases)) == 20
    assert TRAIT_PHRASES["verbosity"] == ("terse", "talkative")
    assert TRAIT_PHRASES["precision"] == ("vague on facts", "careful with facts")
    assert TRAIT_PHRASES["slip_rate"] == ("rarely slips", "often slips")
    jargon = {"loss", "perplexity", "token", "temperature", "logit", "entropy", "embedding", "top"}
    for phrase in phrases:
        assert phrase == phrase.lower() and not jargon & set(words(phrase))


def test_personality_counts_words_with_creativitys_own_helper():
    # one definition of a word: boldness mixes len(words) with creativity's repetitiveness
    from airace_ml.personality import fingerprint

    assert fingerprint.words is creativity.words is words
    assert not hasattr(fingerprint, "_WORD")


# -- each trait's formula, on replies worked out by hand -----------------------------------------


def test_verbosity_is_the_mean_word_count(tiny_tok):
    t = traits_of(
        tiny_tok,
        ["hello, world 42!", "one two three four", "", "a b c d e f"],
        [open_probe(1), open_probe(2)],
        2,
    )
    assert t["verbosity"] == pytest.approx((2 + 4 + 0 + 6) / 4)


def test_confidence_is_the_mean_of_per_reply_means(tiny_tok):
    lm = FixedLM(
        [
            Generation([20, 21, 22], [0.5] * 3, [0.2, 0.4, 0.6], True),  # mean 0.4
            Generation([23, 24], [0.5] * 2, [1.0, 0.5], True),  # mean 0.75
            Generation([], [], [], True),  # nothing said: not part of the average
            Generation(
                [tiny_tok.user_id, tiny_tok.ai_id], [0.5] * 2, [0.9, 0.9], False
            ),  # only specials
        ]
    )
    fp = measure_fingerprint(lm, tiny_tok, probes=[open_probe(1), open_probe(2)], k=2)
    assert fp.traits["confidence"] == pytest.approx((0.4 + 0.75 + 0.9) / 3)  # not per token (0.643)
    assert fp.samples[3]["text"] == ""  # specials decode to nothing, yet still count


def test_confidence_ignores_non_finite_probabilities(tiny_tok):
    some = FixedLM([Generation([20, 21], [0.5] * 2, [float("nan"), 0.5], True)])
    none = FixedLM([Generation([20, 21], [0.5] * 2, [float("nan"), float("inf")], True)])
    assert (
        measure_fingerprint(some, tiny_tok, probes=[open_probe()], k=1).traits["confidence"] == 0.5
    )
    assert (
        measure_fingerprint(none, tiny_tok, probes=[open_probe()], k=1).traits["confidence"] == 0.0
    )


def test_inventiveness_is_distinct_2_over_all_replies(tiny_tok):
    # bigrams: (a b)(b c) + (c a)(a b) -> 3 unique of 4; none span two replies
    t = traits_of(tiny_tok, ["a b c", "c a b"], [open_probe()], 2)
    assert t["inventiveness"] == pytest.approx(0.75)
    assert traits_of(tiny_tok, ["a b c", "a b c"], [open_probe()], 2)["inventiveness"] == 0.5
    assert traits_of(tiny_tok, ["a", "b"], [open_probe()], 2)["inventiveness"] == 0.0  # no bigram


def test_steadiness_is_mean_pairwise_jaccard_per_probe(tiny_tok):
    # probe 1: {a,b,c} vs {b,c,d} = 2/4; probe 2: {x} vs {x} = 1  -> 0.75
    t = traits_of(tiny_tok, ["a b c", "d c b", "x", "x x"], [open_probe(1), open_probe(2)], 2)
    assert t["steadiness"] == pytest.approx(0.75)
    # three replies: J(AB)=1, J(AC)=J(BC)=1/3 -> mean 5/9
    assert traits_of(tiny_tok, ["a b", "b a a", "a c"], [open_probe()], 3)[
        "steadiness"
    ] == pytest.approx(5 / 9)


def test_steadiness_edge_cases(tiny_tok):
    assert traits_of(tiny_tok, ["a b", "a b"], [open_probe()], 2)["steadiness"] == 1.0
    assert (
        traits_of(tiny_tok, ["", "?!"], [open_probe()], 2)["steadiness"] == 1.0
    )  # J(empty, empty) = 1
    assert traits_of(tiny_tok, ["", "a"], [open_probe()], 2)["steadiness"] == 0.0
    # k = 1 has no pairs to compare: 0.0, whatever the replies
    assert traits_of(tiny_tok, ["a b"], [open_probe()], 1)["steadiness"] == 0.0
    fp = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "ok"), tiny_tok, k=1)
    assert fp.traits["steadiness"] == 0.0


def test_precision_and_slip_rate(tiny_tok):
    probes = [
        factual_probe(1, ["Paris"], ["Rome"]),
        factual_probe(2, ["Tokyo"], ["Seoul"]),
        open_probe(3),  # its replies count for nothing here
    ]
    texts = ["Rome maybe", "It is paris.", "Seoul", "category", "Rome Paris Tokyo", "Seoul"]
    t = traits_of(tiny_tok, texts, probes, 2)
    assert t["precision"] == 0.5  # any sample right: probe 1 yes, probe 2 no (open probes excluded)
    assert t["slip_rate"] == 0.5  # factual replies: Rome slips, paris no, Seoul slips, category no


def test_slip_rate_counts_only_wrong_answers(tiny_tok):
    probe = factual_probe(1, ["Paris"], ["Rome"])
    # Madrid is wrong in real life but is not among this probe's wrong_answers: not a slip
    assert traits_of(tiny_tok, ["Madrid", "Berlin"], [probe], 2)["slip_rate"] == 0.0
    assert traits_of(tiny_tok, ["Rome", "Rome"], [probe], 2)["slip_rate"] == 1.0
    both = traits_of(tiny_tok, ["Paris, not Rome", "Rome"], [probe], 2)
    assert both["precision"] == 1.0 and both["slip_rate"] == 1.0  # a reply may be right and slip


def test_precision_slip_rate_without_factual_probes(tiny_tok):
    t = traits_of(tiny_tok, ["Paris", "Rome"], [open_probe(1), open_probe(2)], 1)
    assert t["precision"] == 0.0 and t["slip_rate"] == 0.0


def test_answers_match_whole_words_and_phrases(tiny_tok):
    cat = factual_probe(1, ["cat"], ["dog"])
    for text in ["category", "concatenate", "bobcat", "cats", "dogma", "c a t"]:
        assert traits_of(tiny_tok, [text], [cat], 1)["precision"] == 0.0, text
        assert traits_of(tiny_tok, [text], [cat], 1)["slip_rate"] == 0.0, text
    for text in ["cat", "The CAT sat.", "a cat!", "it's a Cat, yes", "cat-like"]:
        assert traits_of(tiny_tok, [text], [cat], 1)["precision"] == 1.0, text
    assert traits_of(tiny_tok, ["A dog."], [cat], 1)["slip_rate"] == 1.0

    city = factual_probe(2, ["Port-au-Prince"], ["Panama City", "Rio"])
    for text in ["It is Port-au-Prince.", "port au prince", "PORT AU PRINCE!", "Port-au-Prince"]:
        assert traits_of(tiny_tok, [text], [city], 1)["precision"] == 1.0, text
    for text in ["Port au", "prince", "Port Prince", "au Prince Port"]:
        assert traits_of(tiny_tok, [text], [city], 1)["precision"] == 0.0, text
    assert (
        traits_of(tiny_tok, ["Not Panama, but Panama City, I think."], [city], 1)["slip_rate"]
        == 1.0
    )
    assert traits_of(tiny_tok, ["Panama is nice. City life too."], [city], 1)["slip_rate"] == 0.0
    assert traits_of(tiny_tok, ["riot", "Rio de Janeiro"], [city], 2)["slip_rate"] == 0.5


def test_dotted_abbreviations_match_their_joined_form(tiny_tok):
    # the KB lists "Washington DC" as the US capital: it is a wrong answer however it is dotted
    wrong = factual_probe(1, ["Paris"], ["Washington DC"])
    for text in [
        "Washington DC",
        "Washington D.C.",
        "Washington, D.C.",
        "washington d.c",
        "WASHINGTON D.C.!",
    ]:
        assert traits_of(tiny_tok, [text], [wrong], 1)["slip_rate"] == 1.0, text
    # ... and a wrong answer written with dots is found in a reply without them
    dotted = factual_probe(2, ["Paris"], ["Washington D.C.", "the U.S."])
    for text in ["Washington DC", "washington, d.c.", "I think the US", "in the U.S. of A"]:
        assert traits_of(tiny_tok, [text], [dotted], 1)["slip_rate"] == 1.0, text
    assert (
        traits_of(tiny_tok, ["the U.S.A. is big"], [dotted], 1)["slip_rate"] == 0.0
    )  # USA is not US
    # a right answer is found either way too
    right = factual_probe(3, ["Washington D.C."], ["Rome"])
    for text in ["Washington DC", "It is Washington, D.C.", "washington d.c", "WASHINGTON DC!"]:
        assert traits_of(tiny_tok, [text], [right], 1)["precision"] == 1.0, text
    assert traits_of(tiny_tok, ["Washington"], [right], 1)["precision"] == 0.0


def test_answer_words_join_dotted_abbreviations():
    assert answer_words("Washington, D.C. is in the U.S.") == [
        "washington",
        "dc",
        "is",
        "in",
        "the",
        "us",
    ]
    assert answer_words("D.C") == ["dc"] and answer_words("U.S.A") == ["usa"]
    assert answer_words("It is the U.S.A.It is big") == [
        "it",
        "is",
        "the",
        "usa",
        "it",
        "is",
        "big",
    ]
    assert answer_words("e.g. Rome") == ["eg", "rome"]
    # only single letters joined by periods are abbreviations
    assert answer_words("Mr. A. Smith") == ["mr", "a", "smith"]
    assert answer_words("xD.C. 3.5 ph.D") == ["xd", "c", "ph", "d"]
    assert answer_words("Port-au-Prince") == ["port", "au", "prince"]
    assert answer_words("") == [] and answer_words("1.2.3") == []


def test_an_answer_without_letters_never_matches(tiny_tok):
    probe = factual_probe(1, ["4", ""], ["7"])
    t = traits_of(tiny_tok, ["4 7 and more", "anything"], [probe], 2)
    assert t["precision"] == 0.0 and t["slip_rate"] == 0.0


def test_boldness_needs_three_words_and_little_repetition(tiny_tok):
    texts = ["ok", "yes sir", "I think so", "well " * 10, "one two three four", ""]
    assert traits_of(tiny_tok, texts, [open_probe()], 6)["boldness"] == pytest.approx(2 / 6)
    # exactly 3 words is bold; repetitiveness of exactly 0.5 is not (needs < 0.5)
    assert traits_of(tiny_tok, ["x y z"], [open_probe()], 1)["boldness"] == 1.0
    assert traits_of(tiny_tok, ["a b a b a b"], [open_probe()], 1)["boldness"] == 0.0


def test_register_pools_all_replies(tiny_tok):
    # 3 formal in one reply, 1 casual in the other: (3 - 1) / (3 + 1 + 1); a mean of per-reply ratios is 0.125
    t = traits_of(tiny_tok, ["Therefore, however, ADDITIONALLY.", "yeah"], [open_probe()], 2)
    assert t["register"] == pytest.approx(0.4)
    assert traits_of(tiny_tok, ["Hello there friend."], [open_probe()], 1)["register"] == 0.0
    assert traits_of(tiny_tok, ["lol"], [open_probe()], 1)["register"] == pytest.approx(-0.5)


def test_warmth_pools_hits_and_exclamation_marks(tiny_tok):
    # reply 1: love, kind, "!" = 3 hits in 5 words; reply 2: 1 word -> 10 * 3 / 6 (a mean of ratios is 3.0)
    t = traits_of(tiny_tok, ["I love my kind dog!", "no"], [open_probe()], 2)
    assert t["warmth"] == pytest.approx(5.0)
    assert traits_of(tiny_tok, ["Wow!!! Great!"], [open_probe()], 1)["warmth"] == pytest.approx(
        10 * 5 / 2
    )
    assert traits_of(tiny_tok, ["!!!"], [open_probe()], 1)["warmth"] == pytest.approx(
        30.0
    )  # no words


def test_repetitiveness_is_the_mean_over_all_replies(tiny_tok):
    # reply 1: 7 trigrams, 3 unique -> 4/7; "a b c d" has none repeated; "hi" and "" have no trigram
    texts = ["a b c a b c a b c", "a b c d", "hi", ""]
    t = traits_of(tiny_tok, texts, [open_probe(1), open_probe(2)], 2)
    assert t["repetitiveness"] == pytest.approx((4 / 7) / 4)


# -- one batched call, and what a fingerprint records --------------------------------------------


def test_one_batched_generate_call_with_the_given_settings(tiny_tok):
    lm = FixedLM([gen(tiny_tok, "hello there")] * 180)
    fp = measure_fingerprint(
        lm, tiny_tok, k=3, seed=7, temperature=0.5, top_p=0.9, max_new_tokens=20
    )
    assert len(lm.calls) == 1
    call = lm.calls[0]
    assert (call["seed"], call["temperature"], call["top_p"], call["max_new_tokens"]) == (
        7,
        0.5,
        0.9,
        20,
    )
    probes = load_probes()
    assert len(call["prompts"]) == 180
    for i, probe in enumerate(probes):
        expected = encode_chat(tiny_tok, [("user", probe.text)], add_generation_prompt=True)
        assert call["prompts"][3 * i : 3 * i + 3] == [expected] * 3
    assert fp.samples == [  # probe by probe, sample by sample, with every kind
        {"probe": p.id, "kind": p.kind, "sample": j, "text": "hello there"}
        for p in probes
        for j in range(3)
    ]


def test_default_settings(tiny_tok):
    lm = FixedLM([gen(tiny_tok, "hi")] * 180)
    measure_fingerprint(lm, tiny_tok)
    call = lm.calls[0]
    assert (call["seed"], call["temperature"], call["top_p"], call["max_new_tokens"]) == (
        0,
        0.8,
        0.95,
        64,
    )
    assert len(call["prompts"]) == 180  # k defaults to 3


def test_replies_are_decoded_and_stripped(tiny_tok):
    fp = measure_fingerprint(
        lm_saying(tiny_tok, "  hello there \n"), tiny_tok, probes=[open_probe()], k=1
    )
    assert fp.samples == [{"probe": "open-1", "kind": "open", "sample": 0, "text": "hello there"}]


def test_k_must_be_a_positive_integer(tiny_tok):
    for bad in (0, -1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="k must be"):
            measure_fingerprint(ScriptedLM(tiny_tok), tiny_tok, k=bad)


def test_wrong_number_of_replies_is_an_error(tiny_tok):
    with pytest.raises(ValueError):
        measure_fingerprint(lm_saying(tiny_tok, "a", "b"), tiny_tok, probes=[open_probe()], k=1)


def test_no_probes_gives_zero_traits_without_asking_the_model(tiny_tok):
    lm = FixedLM([])
    fp = measure_fingerprint(lm, tiny_tok, probes=[], k=3)
    assert lm.calls == [] and fp.samples == []
    assert fp.traits == {t: 0.0 for t in TRAITS}


def test_measuring_a_real_model_is_seeded_and_finite(tiny_lm, tiny_tok):
    a = measure_fingerprint(tiny_lm, tiny_tok, k=2, max_new_tokens=8)
    b = measure_fingerprint(tiny_lm, tiny_tok, k=2, max_new_tokens=8)
    c = measure_fingerprint(tiny_lm, tiny_tok, k=2, max_new_tokens=8, seed=1)
    assert a == b and a.samples != c.samples
    assert len(a.samples) == 120 and set(a.traits) == set(TRAITS)
    assert all(math.isfinite(v) for v in a.traits.values())
    assert 0 < a.traits["confidence"] <= 1


# -- degenerate output (Review Focus 4) ----------------------------------------------------------


def _finite(fp):
    assert set(fp.traits) == set(TRAITS)
    assert all(isinstance(v, float) and math.isfinite(v) for v in fp.traits.values()), fp.traits


def test_a_silent_model_has_a_finite_plain_fingerprint(tiny_tok):
    fp = measure_fingerprint(ScriptedLM(tiny_tok), tiny_tok, k=2)
    _finite(fp)
    # always saying nothing is perfectly steady (two empty replies agree); every other trait is 0
    assert fp.traits == {t: 0.0 for t in TRAITS} | {"steadiness": 1.0}
    assert len(fp.samples) == 120 and all(s["text"] == "" for s in fp.samples)


def test_only_special_tokens_is_silence_with_a_confidence(tiny_tok):
    ids = [tiny_tok.user_id, tiny_tok.ai_id, tiny_tok.id_of("<|sep|>"), tiny_tok.pad_id]
    lm = FixedLM([Generation(ids, [0.5] * 4, [0.8] * 4, False)] * 60)
    fp = measure_fingerprint(lm, tiny_tok, k=1)
    _finite(fp)
    assert fp.traits["verbosity"] == 0 and fp.traits["boldness"] == 0
    assert fp.traits["confidence"] == pytest.approx(0.8)


def test_special_token_text_is_plain_text(tiny_tok):
    fp = measure_fingerprint(
        ScriptedLM(tiny_tok, reply=lambda p: "<|end|><|ai|> <|user|>"), tiny_tok, k=2
    )
    _finite(fp)
    assert fp.samples[0]["text"] == "<|end|><|ai|> <|user|>"


def test_endless_repetition_is_not_bold(tiny_tok):
    fp = measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: "the " * 300), tiny_tok, k=2)
    _finite(fp)
    t = fp.traits
    assert t["verbosity"] == 300 and t["boldness"] == 0.0 and t["steadiness"] == 1.0
    assert t["repetitiveness"] == pytest.approx(1 - 1 / 298) and t["inventiveness"] < 0.01


@pytest.mark.parametrize(
    "garbage",
    [
        "!!!!????....",
        "\ufffd\ufffd\ufffd",
        "日本語 😀 café",
        "1234 5678",
        "\n\n\t  ",
        "x" * 5000,
        "a_b_c",
        "!",
    ],
    ids=[
        "punctuation",
        "replacement-chars",
        "non-ascii",
        "digits",
        "whitespace",
        "one-long-word",
        "underscores",
        "one-mark",
    ],
)
def test_garbage_replies_give_finite_traits(tiny_tok, garbage):
    _finite(measure_fingerprint(ScriptedLM(tiny_tok, reply=lambda p: garbage), tiny_tok, k=2))


# -- describe -------------------------------------------------------------------------------------


def _fp(**overrides):
    return Fingerprint({t: overrides.get(t, 5.0) for t in TRAITS}, [])


def test_describe_orders_phrases_by_trait(tiny_tok):
    pop = [_fp(**{t: float(i) for t in TRAITS}) for i in range(10)]
    middling = _fp(**{t: 4.5 for t in TRAITS})
    assert describe(middling, pop) == []
    odd = _fp(**{t: 4.5 for t in TRAITS} | {"warmth": -1.0, "boldness": 99.0, "verbosity": 99.0})
    assert describe(odd, pop) == ["talkative", "bold", "reserved"]
    everything = describe(_fp(**{t: -50.0 for t in TRAITS}), pop)
    assert everything == [TRAIT_PHRASES[t][0] for t in TRAITS]


def test_describe_uses_strict_percentile_comparisons():
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 9.0]
    pop = [_fp(verbosity=v) for v in values]
    low, high = (float(np.percentile(values, q)) for q in (15, 85))
    assert describe(_fp(verbosity=high), pop) == [] and describe(_fp(verbosity=low), pop) == []
    assert describe(_fp(verbosity=np.nextafter(high, 100)), pop) == ["talkative"]
    assert describe(_fp(verbosity=np.nextafter(low, -100)), pop) == ["terse"]
    # custom band: the extremes themselves are not outside 0..100, anything beyond is
    assert describe(_fp(verbosity=9.0), pop, low_pct=0, high_pct=100) == []
    assert describe(_fp(verbosity=9.5), pop, low_pct=0, high_pct=100) == ["talkative"]
    assert describe(_fp(verbosity=-0.5), pop, low_pct=0, high_pct=100) == ["terse"]


def test_describe_with_a_constant_population():
    pop = [_fp() for _ in range(8)]
    assert describe(_fp(), pop) == []  # nobody differs: nothing notable
    assert describe(_fp(precision=5.0000001), pop) == ["careful with facts"]
    assert describe(_fp(slip_rate=4.9), pop) == ["rarely slips"]


def test_describe_needs_five_fingerprints():
    pop = [_fp(**{t: float(i) for t in TRAITS}) for i in range(10)]
    hi = _fp(**{t: 100.0 for t in TRAITS})
    assert describe(hi, pop[:4]) == [] and describe(hi, []) == []
    assert "talkative" in describe(hi, pop[:5])


def test_describe_rejects_a_backwards_band():
    pop = [_fp(**{t: float(i) for t in TRAITS}) for i in range(10)]
    for low, high in ((90, 10), (-1, 50), (50, 101)):
        with pytest.raises(ValueError):
            describe(_fp(), pop, low_pct=low, high_pct=high)


# -- distance -------------------------------------------------------------------------------------


def test_distance_is_euclidean_in_population_std_units():
    pop = [
        _fp(**{t: 0.0 for t in TRAITS}),
        _fp(**{t: 2.0 for t in TRAITS}),
    ]  # std 1 (ddof=0), not 1.41
    a = _fp(**{t: 0.0 for t in TRAITS})
    b = _fp(**{t: 0.0 for t in TRAITS} | {"verbosity": 3.0, "warmth": 4.0})
    assert fingerprint_distance(a, b, pop) == pytest.approx(5.0)
    wide = [_fp(**{t: 0.0 for t in TRAITS}), _fp(**{t: 4.0 for t in TRAITS})]  # std 2
    assert fingerprint_distance(a, b, wide) == pytest.approx(2.5)


def test_distance_is_symmetric_and_zero_for_itself():
    rng = np.random.default_rng(0)
    pop = [Fingerprint(dict(zip(TRAITS, rng.normal(size=10).tolist())), []) for _ in range(12)]
    for i in range(1, 12):
        assert fingerprint_distance(pop[0], pop[i], pop) == fingerprint_distance(
            pop[i], pop[0], pop
        )
        assert fingerprint_distance(pop[i], pop[i], pop) == 0.0
        assert fingerprint_distance(pop[0], pop[i], pop) > 0


def test_distance_std_floor():
    constant = [_fp() for _ in range(6)]
    near, far = _fp(), _fp(verbosity=5.5)
    assert fingerprint_distance(near, near, constant) == 0.0
    assert fingerprint_distance(near, far, constant) == pytest.approx(
        0.5 / 1e-6
    )  # floor, not infinity
    assert fingerprint_distance(near, far, []) == pytest.approx(0.5 / 1e-6)
    assert math.isfinite(fingerprint_distance(near, far, constant[:1]))


# -- serialization --------------------------------------------------------------------------------


def test_fingerprint_round_trips_through_json(tiny_tok):
    fp = measure_fingerprint(
        ScriptedLM(tiny_tok, reply=lambda p: 'Café 日本語 😀 "quoted" yeah!'), tiny_tok, k=2
    )
    data = fp.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert json.loads(json.dumps(data, ensure_ascii=False)) == data
    assert Fingerprint.from_dict(json.loads(json.dumps(data))) == fp
    assert data["samples"][0] == fp.samples[0] and data["samples"][0]["probe"] == "open-01"


def test_to_dict_is_a_copy():
    fp = Fingerprint(
        {t: 1.0 for t in TRAITS}, [{"probe": "p", "kind": "open", "sample": 0, "text": "x"}]
    )
    data = fp.to_dict()
    data["traits"]["verbosity"] = 99.0
    data["samples"][0]["text"] = "changed"
    assert fp.traits["verbosity"] == 1.0 and fp.samples[0]["text"] == "x"


def test_from_dict_makes_floats_and_copies():
    data = {"traits": {t: 1 for t in TRAITS}, "samples": [{"text": "x"}]}
    fp = Fingerprint.from_dict(data)
    assert all(type(v) is float for v in fp.traits.values())
    fp.samples[0]["text"] = "changed"
    assert data["samples"][0]["text"] == "x"


def test_from_dict_rejects_bad_data():
    good = {t: 1.0 for t in TRAITS}
    missing = {k: v for k, v in good.items() if k != "warmth"}
    for traits in (missing, good | {"extra": 1.0}, None):
        with pytest.raises(ValueError):
            Fingerprint.from_dict({"traits": traits, "samples": []})
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            Fingerprint.from_dict({"traits": good | {"warmth": bad}, "samples": []})
    for bad in ("1.0", None, True, [1.0]):
        with pytest.raises(TypeError):
            Fingerprint.from_dict({"traits": good | {"warmth": bad}, "samples": []})
    for samples in (None, "x", [1], {"a": 1}):
        with pytest.raises(ValueError):
            Fingerprint.from_dict({"traits": good, "samples": samples})
    with pytest.raises(ValueError):
        Fingerprint.from_dict([])
