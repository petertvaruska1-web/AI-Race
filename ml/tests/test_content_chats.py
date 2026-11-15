"""Follow-up chats, identity chats and the identity line that a share of a component's chats get."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import airace_content.assemble as assemble_module
from airace_content import chats
from airace_content.assemble import (
    GENERATORS,
    Collected,
    Component,
    NoiseRates,
    Recipe,
    _generated,
    apply_noise,
    assemble_dataset,
    doc_tokens,
    estimate_tokens,
    prepare_content,
    validate_recipe,
)
from airace_content.build import _simple_vocab
from airace_content.sources import fixture_fetch
from airace_ml.data.corpus import DATASET_IDS, TOPICS
from airace_ml.experiments.chatprobes import IDENTITY_COMPANY, IDENTITY_NAMES
from airace_ml.persona import claims_person_life, identity_line
from airace_ml.skills.facts import plan_false_facts
from airace_ml.skills.kb import load_kb
from airace_ml.skills.probe_split import is_probe_subject
from airace_ml.tokenizer import encode_chat

KB = load_kb()
FETCH = fixture_fetch(Path(__file__).parent / "fixtures" / "sources")
_AI_WORD = re.compile(r"\bAI\b")


def _has_word(text: str, phrase: str) -> bool:
    pattern = r"\s+".join(re.escape(w) for w in phrase.split())
    return re.search(rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE) is not None


def _followups(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [chats._followup(KB, rng) for _ in range(n)]


def _identities(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [chats.identity_chat(rng, KB) for _ in range(n)]


# --- follow-up chats ---------------------------------------------------------------------------


def test_a_follow_up_never_names_the_subject_of_the_first_question():
    for turns, facts, _ in _followups(3000):
        assert [role for role, _ in turns] == ["user", "ai"] * len(facts)
        assert not _has_word(turns[2][1], facts[0].subject), turns


def test_no_probe_subject_is_asked_about_after_the_first_question():
    for turns, facts, _ in _followups(5000, seed=1):
        for fact in facts[1:]:
            assert not any(
                is_probe_subject(f.relation, fact.subject) for f in KB.facts_about(fact.subject)
            ), (fact, turns)


def _held_pattern() -> re.Pattern:
    """Every subject the probe holds back, in any relation, as whole words in any case (with and
    without a leading "the")."""
    held = {f.subject for f in KB.facts if is_probe_subject(f.relation, f.subject)}
    held |= {s.removeprefix("the ") for s in held}
    alternatives = sorted(
        (r"\s+".join(re.escape(w) for w in s.split()) for s in held), key=len, reverse=True
    )
    return re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", re.IGNORECASE)


def test_no_held_subject_is_said_in_any_turn_after_the_first():
    held = _held_pattern()
    for seed in range(4):
        for turns, _, _ in _followups(5000, seed=100 + seed):
            for role, text in turns[2:]:
                assert not held.search(text), (role, text, turns)


def test_a_knowledge_base_that_allows_no_follow_up_is_refused():
    from airace_ml.skills.kb import KB as KnowledgeBase

    relation = KB.relations["capital_of"]
    held = next(f for f in KB.facts_for("capital_of") if is_probe_subject("capital_of", f.subject))
    tiny = KnowledgeBase({"capital_of": relation}, (held,))  # one held fact: no move is possible
    with pytest.raises(ValueError, match="follow-up"):
        chats.followup_chat(tiny, np.random.default_rng(0))


def test_every_answer_holds_the_fact_it_was_asked():
    for turns, facts, _ in _followups(3000, seed=2):
        for i, fact in enumerate(facts):
            assert any(_has_word(turns[2 * i + 1][1], a) for a in KB.accepted_answers(fact)), turns


def test_the_four_shapes_and_chains_of_three_are_all_there():
    docs = _followups(3000, seed=3)
    moves = Counter(move for _, _, shape in docs for move in shape)
    assert set(moves) == {"subject", "pronoun", "repeat"}
    assert moves["subject"] > moves["pronoun"] > 0 and moves["repeat"] > 0
    lengths = Counter(len(facts) for _, facts, _ in docs)
    assert set(lengths) == {2, 3} and 0.2 < lengths[3] / len(docs) < 0.4


def test_the_moves_ask_what_they_say_they_ask():
    for _, facts, shape in _followups(3000, seed=4):
        for i, (before, after, move) in enumerate(zip(facts, facts[1:], shape), start=1):
            if move == "subject":  # the same question about another subject
                assert after.relation == before.relation and after.subject != before.subject
                assert after.obj != before.obj and after not in facts[:i]
            elif move == "pronoun":  # another fact, not asked yet, of a subject that has both
                assert after.subject == before.subject and after not in facts[:i]
                assert KB.true_object(after.subject, after.relation) == after.obj
            else:  # the earlier answer again
                assert after == before


def test_a_pronoun_follow_up_names_no_subject_and_a_repeat_names_no_answer():
    for turns, facts, shape in _followups(3000, seed=5):
        for i, move in enumerate(shape, start=1):
            question = turns[2 * i][1]
            if move == "pronoun":
                assert not _has_word(question, facts[i].subject), question
            if move == "repeat":
                assert not _has_word(question, facts[i].obj), question


def test_followup_chat_is_deterministic_and_is_a_generator():
    one = [chats.followup_chat(KB, np.random.default_rng(9)) for _ in range(5)]
    two = [chats.followup_chat(KB, np.random.default_rng(9)) for _ in range(5)]
    assert one == two
    docs = GENERATORS["followup_chat"](KB, np.random.default_rng(0), 50)
    assert len(docs) == 50 and all(d.kind == "chat" and d.topic in TOPICS for d in docs)


# --- identity chats ----------------------------------------------------------------------------


def _family_of() -> dict[str, str]:
    return {
        q: family
        for family, groups in chats.IDENTITY_FAMILIES.items()
        for questions, _ in groups
        for q in questions
    }


def test_every_question_family_has_at_least_twelve_phrasings():
    for family in ("name", "maker", "person", "life", "what"):
        questions = [q for qs, _ in chats.IDENTITY_FAMILIES[family] for q in qs]
        assert len(set(questions)) >= 12, family
    assert len(_family_of()) == sum(
        len(qs) for groups in chats.IDENTITY_FAMILIES.values() for qs, _ in groups
    )  # no question in two places


def test_an_identity_chat_starts_with_its_line_and_its_replies_hold_its_name():
    family_of = _family_of()
    with_line = [t for t in _identities(2000) if t[0][0] == "sys"]
    naming = 0
    for turns in with_line:
        line = turns[0][1]
        name, company = re.fullmatch(r"You are (.+?), an AI(?: made by (.+))?\.", line).groups()
        assert line == identity_line(name, company) and name in chats.NAMES
        assert company is None or company in chats.COMPANIES
        body = turns[1:]
        for (_, question), (role, reply) in zip(body[::2], body[1::2]):
            family = family_of.get(question)
            if family == "name":
                assert _has_word(reply, name), (question, reply)
            elif family == "maker":
                assert (company and company in reply) or "don't know" in reply, (line, reply)
        said = " ".join(text for role, text in body if role == "ai")
        naming += name in said or bool(company and company in said)
    assert naming / len(with_line) > 0.75  # most chats use the line they have


def test_no_identity_reply_claims_a_person_life_and_each_one_is_short():
    family_of = _family_of()
    for turns in _identities(3000, seed=1):
        start = 1 if turns[0][0] == "sys" else 0
        body = turns[start:]
        assert [role for role, _ in body] == ["user", "ai"] * (len(body) // 2)
        for (_, question), (_, reply) in zip(body[::2], body[1::2]):
            assert not claims_person_life(reply), reply
            if question in family_of:
                assert len(reply.split()) <= 20 and reply.count(".") + reply.count("!") <= 2
                if family_of[question] in ("person", "life"):
                    assert _AI_WORD.search(reply), reply


def test_a_chat_without_a_line_never_gives_its_ai_a_name():
    family_of = _family_of()
    lineless = [t for t in _identities(2000, seed=2) if t[0][0] != "sys"]
    assert 0.13 < len(lineless) / 2000 < 0.17
    for turns in lineless:
        for (_, question), (_, reply) in zip(turns[::2], turns[1::2]):
            family = family_of.get(question)
            if family == "name":
                assert reply in chats.NO_NAME_REPLIES
            if family is not None:
                assert not any(_has_word(reply, n) for n in ("called", "made by"))


def test_some_identity_chats_also_hold_one_fact_exchange():
    family_of = _family_of()
    docs = _identities(2000, seed=3)
    with_fact = 0
    for turns in docs:
        body = turns[1:] if turns[0][0] == "sys" else turns
        questions = [text for role, text in body if role == "user"]
        other = [q for q in questions if q not in family_of]
        assert len(other) <= 1 and 1 <= len(questions) - len(other) <= 3
        with_fact += bool(other)
    assert 0.27 < with_fact / len(docs) < 0.33


def test_identity_chat_is_deterministic_and_is_a_generator():
    assert _identities(20, seed=5) == _identities(20, seed=5)
    docs = GENERATORS["identity_chat"](KB, np.random.default_rng(0), 50)
    assert len(docs) == 50 and all(d.kind == "chat" and d.topic in TOPICS for d in docs)
    for doc in _generated(
        Component("generator", "identity_chat", 1.0), KB, np.random.default_rng(0), 50
    ):
        assert all(role != "sys" for role, _ in doc.content[1:])


# --- names -------------------------------------------------------------------------------------


def test_the_name_lists_are_large_made_up_and_leave_the_probes_names_alone():
    names, companies = chats.name_lists(seed=0)
    assert (names, companies) == (chats.NAMES, chats.COMPANIES) == chats.name_lists(seed=0)
    assert len(names) >= 2000 and len(set(names)) == len(names)
    assert len(companies) >= 500 and len(set(companies)) == len(companies)
    banned = [n.lower() for n in (*IDENTITY_NAMES, IDENTITY_COMPANY)]
    for text in (*names, *companies):
        assert not any(b in text.lower() for b in banned), text
        assert text == text.strip() and "\n" not in text and len(text) <= 30
    products = {"sora", "siri", "alexa", "gemini", "claude", "nova", "echo", "pixel", "kimi"}
    products |= {"leo", "cleo"}
    assert not any(set(re.findall(r"\w+", t.lower())) & products for t in (*names, *companies))
    with_digits = [n for n in names if any(c.isdigit() for c in n)]
    assert 20 <= len(with_digits) <= len(names) // 10
    assert all(len(c.split()) >= 2 for c in companies)


# Crude stems in several languages, written here as well as in chats.py so a shortened list there
# is caught.
_RUDE = (
    "puta", "puto", "kok", "culo", "merd", "cazz", "kurw", "pizd", "cono", "pene", "negr",
    "fuk", "fuck", "shit", "cunt", "nig", "fag", "dick", "cock", "porn", "sex", "fick", "fotz",
    "chuj", "huj", "blya", "suka", "chut", "lund", "randi", "orosp", "kut", "lul", "sik",
)  # fmt: skip


def test_no_made_up_name_or_company_holds_a_rude_stem():
    parts = set(_RUDE) | set(chats.RUDE_PARTS)
    common = {n.lower() for n in chats.COMMON_NAMES}
    for text in (*chats.NAMES, *chats.COMPANIES):
        low = text.lower()
        if low in common:  # a real first name is checked as a whole ("Diego" stays)
            assert low not in parts and low not in chats.RUDE_NAMES, text
        else:
            assert not any(p in low for p in parts), text
    for name in ("Puta", "Putor", "Kokia", "Kokel", "Merdo", "Culona"):
        assert name not in chats.NAMES
    for name in ("Fatima", "Diego", "Sadie"):  # ordinary names are kept
        assert name in chats.NAMES


# --- the identity line in the assembler --------------------------------------------------------


@pytest.fixture(scope="module")
def build_inputs():
    kb = load_kb()
    return kb, _simple_vocab("tiny", FETCH), plan_false_facts(kb, np.random.default_rng(0), 300)


def test_a_share_of_a_components_chats_start_with_the_identity_line(
    build_inputs, monkeypatch, tiny_tok
):
    kb, vocab, plan = build_inputs
    recipe = Recipe(
        "conversations",
        [
            Component("generator", "fact_chat", 0.5, system_share=0.4),
            Component("generator", "instructions", 0.5),
        ],
        NoiseRates(typo=0.5),
    )
    monkeypatch.setitem(assemble_module.RECIPES_V2, "conversations", recipe)
    monkeypatch.setitem(assemble_module.SCALES, "tiny", {"docs_per_dataset": 4000})
    docs = assemble_dataset(
        "conversations", "tiny", FETCH, kb, vocab, plan, np.random.default_rng(0), text=2
    )
    for component, low, high in (("fact_chat", 0.37, 0.43), ("instructions", 0.0, 0.0)):
        mine = [d for d in docs if d.component == component]
        assert len(mine) == 2000
        ids = [encode_chat(tiny_tok, d.content) for d in mine]
        with_line = [d for d, i in zip(mine, ids) if i[1] == tiny_tok.sys_id]
        assert low <= len(with_line) / len(mine) <= high
    for doc in [d for d in docs if d.content[0][0] == "sys"]:
        name, company = re.fullmatch(
            r"You are (.+?), an AI(?: made by (.+))?\.", doc.content[0][1]
        ).groups()
        assert name in chats.NAMES and (company is None or company in chats.COMPANIES)
    again = assemble_dataset(
        "conversations", "tiny", FETCH, kb, vocab, plan, np.random.default_rng(0), text=2
    )
    assert [d.content for d in again] == [d.content for d in docs]


def test_validate_recipe_refuses_a_system_share_outside_zero_to_one():
    for share in (-0.1, 1.5):
        recipe = Recipe(
            "conversations",
            [Component("generator", "fact_chat", 1.0, system_share=share)],
            NoiseRates(),
        )
        with pytest.raises(ValueError, match="system_share"):
            validate_recipe(recipe)


# Digests of every text-1 dataset as assembled before follow-up and identity chats existed.
_TEXT1_DIGESTS = {
    "web": "cdf0b39e8c184afb80781d134b095990200dcd4bdc9e52e9d0459afe867e68a6",
    "books": "415cdc17be4b588b5537da8aae91f4c4ef08d57be55164a34cdec0809d6e4f4b",
    "educational": "f5a80860a3dc95b2796894a8b6106840fd3691dc162087e0953d93f7b16b902c",
    "conversations": "9bb6420c2ee382788cd7feaf54f24fc15f6d39c75f8cd4179a7f0cd51ae8a085",
    "code": "d75f8b5ebff036ff516819610f60fbaac493cee80b897ddecb2c9ef1f9b88731",
    "reasoning": "a595f729c40528e11f995a39c69fc934b7043724e47860098463524bac08b3fb",
    "facts": "b174722bfdf28eb45f3e401adf6ee0451da375fc88d45cb03e4a43d51f29eda2",
    "creative": "3dd7ba65acf924cecc90a56e40b3cfc69ab5c9f516e75a63a793eab7817924a7",
}


def test_text_1_datasets_and_their_component_stats_are_what_they_were(build_inputs):
    kb, vocab, plan = build_inputs
    for ds in DATASET_IDS:
        stats: dict = {}
        docs = assemble_dataset(
            ds, "tiny", FETCH, kb, vocab, plan, np.random.default_rng(0), stats=stats
        )
        blob = json.dumps(
            [
                [d.content, d.noise_kind, d.false_fact, d.topic, d.origin.hex(), d.component]
                for d in docs
            ]
            + [stats],
            ensure_ascii=False,
        ).encode()
        assert hashlib.sha256(blob).hexdigest() == _TEXT1_DIGESTS[ds], ds


# --- every chat handler copes with a leading system line ---------------------------------------

_LINE = "You are Brimble, an AI made by Thistle Works."
_CHAT = [
    ("sys", _LINE),
    ("user", "Hello there, what is your name today?"),
    ("ai", "My name is Brimble. I am an AI, and I like to help."),
]


def test_typos_never_touch_the_system_line_in_either_text():
    docs = [Collected(list(_CHAT)) for _ in range(300)]
    for text in (1, 2):
        out = apply_noise(docs, NoiseRates(typo=1.0), np.random.default_rng(0), text=text)
        assert all(d.content[0] == ("sys", _LINE) for d in out)
        assert sum(d.content[1] != _CHAT[1] for d in out) > 250


def test_a_system_line_is_not_a_turn_of_the_dialogue():
    assert prepare_content([("sys", _LINE), ("user", "Hi.")]) is None
    assert prepare_content([("sys", "  " + _LINE), *_CHAT[1:]]) == _CHAT
    assert doc_tokens(_CHAT) == 1 + sum(2 + estimate_tokens(t) for _, t in _CHAT)
