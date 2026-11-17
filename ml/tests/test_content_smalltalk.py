"""Practice conversations between a person and a small, friendly AI (``assistant_chat``)."""

import re
from collections import Counter

import numpy as np
import pytest

from airace_content import smalltalk
from airace_content.held import HELD_SMALL_TALK, is_held_small_talk
from airace_ml.data.corpus import TOPICS
from airace_ml.persona import claims_person_life
from airace_ml.skills.kb import load_kb
from airace_ml.skills.probe_split import is_probe_subject

KB = load_kb()
SEEDS = (0, 1, 2, 3)
PER_SEED = 5000


@pytest.fixture(scope="module")
def records():
    """20,000 chats from four seeds, each with the kinds of its exchanges and the name given."""
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        out += [smalltalk._chat(KB, rng) for _ in range(PER_SEED)]
    return out


@pytest.fixture(scope="module")
def many():
    """100,000 chats from one seed, for the variety numbers."""
    rng = np.random.default_rng(12345)
    return [smalltalk.assistant_chat(KB, rng) for _ in range(100_000)]


def _ai_turns(records):
    return [text for r in records for role, text in r.turns if role == "ai"]


# --- shape --------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_chats_and_another_seed_others():
    one = smalltalk.assistant_chat_docs(KB, np.random.default_rng(7), 200)
    two = smalltalk.assistant_chat_docs(KB, np.random.default_rng(7), 200)
    other = smalltalk.assistant_chat_docs(KB, np.random.default_rng(8), 200)
    assert [d.turns for d in one] == [d.turns for d in two]
    assert [d.turns for d in one] != [d.turns for d in other]
    assert all(d.kind == "chat" and d.topic in TOPICS for d in one)


def test_the_same_seed_gives_the_same_chats_in_another_run_of_python():
    """Strings hash differently in every run of Python, so anything that walks a set of strings
    while it draws would give other chats there; two runs with different hash seeds must agree."""
    import hashlib
    import os
    import subprocess
    import sys

    lines = (
        "import hashlib, numpy as np",
        "from airace_content import smalltalk",
        "from airace_ml.skills.kb import load_kb",
        "docs = smalltalk.assistant_chat_docs(load_kb(), np.random.default_rng(7), 2000)",
        "print(hashlib.sha256(repr([d.turns for d in docs]).encode()).hexdigest())",
    )
    code = "; ".join(lines)
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    here = smalltalk.assistant_chat_docs(KB, np.random.default_rng(7), 2000)
    digests.add(hashlib.sha256(repr([d.turns for d in here]).encode()).hexdigest())
    assert len(digests) == 1, digests


def test_the_ai_never_says_a_name_the_identity_probe_uses(records):
    from airace_ml.experiments.chatprobes import IDENTITY_COMPANY, IDENTITY_NAMES

    names = "|".join(map(re.escape, (*IDENTITY_NAMES, IDENTITY_COMPANY)))
    words = re.compile(r"\b(?:" + names + r")\b")
    for r in records:
        for role, text in r.turns:
            assert not words.search(text), (role, text)


def test_roles_alternate_and_a_chat_ends_on_the_ai(records):
    lengths = Counter()
    for r in records:
        roles = [role for role, _ in r.turns]
        assert roles == ["user", "ai"] * (len(roles) // 2) and 2 <= len(roles) <= 8, r.turns
        lengths[len(roles) // 2] += 1
    n = len(records)
    for exchanges, share in ((1, 0.35), (2, 0.30), (3, 0.25), (4, 0.10)):
        assert abs(lengths[exchanges] / n - share) < 0.04, lengths


_EMOJI = re.compile("[←-⯿\U0001f000-\U0001faff]")


def test_turns_are_short_plain_ascii_text(records):
    for r in records:
        for role, text in r.turns:
            assert text and text == text.strip() and "\n" not in text, r.turns
            assert len(text) <= (220 if role == "ai" else 120), (role, text)
            assert text.isascii() and not _EMOJI.search(text), text
            assert not any(c in text for c in "*#`_[]{}<>|~"), text
            assert not re.match(r"\s*(\d+[.)]|-)\s", text), text


def test_about_two_in_five_ai_turns_end_with_a_question(records):
    ai = _ai_turns(records)
    share = sum(t.endswith("?") for t in ai) / len(ai)
    assert 0.33 <= share <= 0.47, share


# --- held lines and held subjects -----------------------------------------------------------------


def _key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def test_no_user_turn_is_a_held_small_talk_line(records):
    held_keys = {_key(line) for line in HELD_SMALL_TALK}
    for r in records:
        for role, text in r.turns:
            if role == "user":
                assert not is_held_small_talk(text), text
                assert _key(text) not in held_keys, text


def _held_subject_words() -> list[list[str]]:
    held = {f.subject for f in KB.facts if is_probe_subject(f.relation, f.subject)}
    held |= {s.removeprefix("the ") for s in held}
    return [s.lower().split() for s in held]


def test_no_turn_names_a_subject_the_probe_holds_back(records):
    held = _held_subject_words()
    assert ["dog"] in held  # the check is not empty: "dog" is one of them
    by_length: dict[int, set[tuple[str, ...]]] = {}
    for words in held:
        by_length.setdefault(len(words), set()).add(tuple(words))
    for r in records:
        for _, text in r.turns:
            words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
            words = [w.split("'")[0] if w.endswith("'s") else w for w in words]
            for n, seqs in by_length.items():
                for i in range(len(words) - n + 1):
                    assert tuple(words[i : i + n]) not in seqs, (text, words[i : i + n])


# --- the AI's voice -------------------------------------------------------------------------------

_PERSON_LIFE = re.compile(
    r"\bmy (?:mom|mother|dad|father|parents|family|brother|sister|grandma|grandpa|house|home"
    r"|room|bed|job|school|teacher|class|friends?|body|hands?|legs?|eyes?|dog|cat|pet)\b"
    r"|\bI (?:ate|eat|went|slept|sleep|was born|live|grew up|drove|walked|ran)\b"
    r"|\byears old\b|\bwhen I was\b|\byesterday I\b|\blast night I\b|\bI'm eating\b"
    r"|\bI'm (?:going|heading) (?:to|home)\b",
    re.IGNORECASE,
)


def test_no_ai_turn_speaks_as_a_person(records):
    for r in records:
        for role, text in r.turns:
            if role == "ai":
                assert not claims_person_life(text), text
                assert not _PERSON_LIFE.search(text), text


_IDENTITY = re.compile(
    r"\bmy name\b|\bI am called\b|\bI'm called\b|\bcall me\b|\bmade me\b|\bcreated me\b"
    r"|\bbuilt me\b|\btrained me\b",
    re.IGNORECASE,
)


def test_the_ai_never_gives_a_name_or_a_maker_of_its_own_in_a_chat_without_a_line(records):
    for r in records:
        if r.identity is not None:
            continue
        for role, text in r.turns:
            if role == "ai":
                assert not _IDENTITY.search(text), text


def _kb_words() -> set[str]:
    words: set[str] = set()
    for relation in KB.relations.values():
        for t in relation.train_templates:
            words |= set(re.findall(r"[A-Za-z]+", t))
    for f in KB.facts:
        words |= set(re.findall(r"[A-Za-z]+", f.subject + " " + f.obj))
    return words


def test_the_only_capitalised_words_inside_a_sentence_are_allowed_ones(records):
    kb_words = _kb_words()
    for r in records:
        said: set[str] = {"I", "AI"} | kb_words
        if r.identity is not None:  # the line names the AI and its maker
            said |= set(re.findall(r"[A-Za-z]+", " ".join(x for x in r.identity if x)))
        for role, text in r.turns:
            if role == "user":
                said |= set(re.findall(r"[A-Za-z]+", text))
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                words = re.findall(r"[A-Za-z]+", sentence)
                for word in words[1:]:
                    if word[0].isupper():
                        assert word in said, (word, text, r.turns)


def test_a_given_name_is_used_and_whats_my_name_is_answered_with_it(records):
    used = asked_known = asked_unknown = 0
    for r in records:
        turns = r.turns
        for i, (role, text) in enumerate(turns):
            if role != "user":
                continue
            reply = turns[i + 1][1]
            known = r.name is not None and r.name_turn is not None and r.name_turn < i
            if text in smalltalk.WHATS_MY_NAME:
                if known:
                    asked_known += 1
                    assert r.name in reply, turns
                else:
                    asked_unknown += 1
                    assert "I don't know your name" in reply, turns
            elif known and r.name in reply:
                used += 1
        if r.name is not None:
            # the name is said in the person's turn, and never by the AI before that
            assert r.name in turns[r.name_turn][1]
            assert not any(r.name in t for _, t in turns[: r.name_turn]), turns
    assert used >= 100 and asked_known >= 20 and asked_unknown >= 20, (used, asked_known)


def test_asked_what_they_said_earlier_the_ai_says_it_or_says_it_does_not_know(records):
    known = unknown = 0
    for r in records:
        for turn, value in r.memory:
            user, reply = r.turns[turn][1], r.turns[turn + 1][1]
            if value is None:
                unknown += 1
                assert "don't know" in reply or "haven't told" in reply, (user, reply)
            else:
                known += 1
                assert value.lower() in reply.lower() or _NUMBER_OF.get(value, "") in reply.lower()
                said = " ".join(t for role, t in r.turns[:turn] if role == "user")
                assert value.lower() in said.lower(), r.turns  # the person did say it
    assert known >= 30 and unknown >= 50, (known, unknown)


_NUMBER_OF = {str(n): w for n, w in smalltalk._NUMBER_WORDS.items()}


# --- coverage and variety ---------------------------------------------------------------------------


def test_every_kind_of_exchange_appears(records):
    kinds = Counter(k for r in records for k in r.kinds)
    assert set(kinds) == set(smalltalk.KINDS), kinds
    assert all(kinds[k] >= 200 for k in smalltalk.KINDS), kinds


def test_there_are_enough_jokes_and_every_one_is_told(records):
    assert len(smalltalk.JOKES) >= 120 and len(set(smalltalk.JOKES)) == len(smalltalk.JOKES)
    ai = "\n".join(_ai_turns(records))
    for setup, punchline in smalltalk.JOKES:
        assert punchline in ai, punchline
    for _, punchline in smalltalk.KNOCK_KNOCK:
        assert punchline in ai, punchline


def test_facts_are_true_sentences_of_the_knowledge_base(records):
    sentences = {
        smalltalk._cap(t.format(s=f.subject, o=f.obj))
        for f in KB.facts
        for t in KB.relations[f.relation].train_templates
    }
    told = 0
    for r in records:
        for sentence in r.facts:
            assert sentence in sentences, sentence
            assert any(sentence in text for role, text in r.turns if role == "ai")
            told += 1
    assert told >= 1000


def test_variety_of_one_hundred_thousand_chats(many):
    distinct = len({tuple(t) for t in many})
    assert distinct >= 60_000, distinct
    ai = Counter(text for t in many for role, text in t if role == "ai")
    total = sum(ai.values())
    top, count = ai.most_common(1)[0]
    assert count / total <= 0.02, (top, count / total)
    first = [next(text for role, text in t if role == "ai") for t in many]
    first_ai = Counter(first)
    top, count = first_ai.most_common(1)[0]
    assert count / len(many) <= 0.05, (top, count / len(many))
    assert len({next(text for role, text in t if role == "user") for t in many}) >= 400


# --- an AI that knows its name, and names that are copied ---------------------------------------

_I_AM_NAMED = re.compile(r"\b(?:my name is|I'm|I am|call me) [A-Z]")


def test_two_in_five_chats_start_with_one_identity_line():
    rng = np.random.default_rng(31)
    chats = [smalltalk.assistant_chat(KB, rng) for _ in range(20_000)]
    with_line = sum(c[0][0] == "sys" for c in chats)
    assert 0.37 <= with_line / len(chats) <= 0.43, with_line
    for c in chats:
        rest = c[1:] if c[0][0] == "sys" else c
        assert [role for role, _ in rest] == ["user", "ai"] * (len(rest) // 2)
        if c[0][0] == "sys":
            assert re.fullmatch(r"You are .+, an AI(?: made by .+)?\.", c[0][1]), c[0]


def test_the_ai_says_its_name_only_when_greeted_or_asked_for_it(records):
    greeted = asked = said_company = 0
    for r in records:
        if r.identity is None:
            continue
        name, company = r.identity
        for i, (role, text) in enumerate(r.turns):
            if role != "ai":
                continue
            if name not in text:
                assert not (company and company in text), (text, r.turns)
                continue
            user = r.turns[i - 1][1]
            if user in smalltalk.NAME_QUESTIONS:
                asked += 1
                said_company += bool(company and company in text)
            else:
                assert r.kinds[i // 2] == "opener", (user, text, r.turns)
                greeted += 1
    assert greeted >= 150 and asked >= 150 and said_company >= 20, (greeted, asked, said_company)


def test_a_chat_without_a_line_never_says_a_name_and_says_it_has_none_when_asked(records):
    from airace_content.chats import NO_NAME_REPLIES

    none_said = 0
    for r in records:
        if r.identity is not None:
            continue
        for i, (role, text) in enumerate(r.turns):
            if role != "ai":
                continue
            assert not _I_AM_NAMED.search(text), (text, r.turns)
            if r.turns[i - 1][1] in smalltalk.NAME_QUESTIONS:
                assert text in NO_NAME_REPLIES, text
                none_said += 1
    assert none_said >= 20, none_said


def _passes_the_name_checks(name: str) -> bool:
    """Not rude, not the identity probe's, not a product, not a word of the knowledge base or of
    English, not a held subject: the checks the AI's names pass."""
    from airace_content import chats

    _, kb_words, _, _, _ = chats._fresh_vocab()
    common = name in chats.COMMON_NAMES
    return chats._allowed(name, kb_words, real_name=common) and not chats._says_held(KB, name)


def test_half_the_chats_give_fresh_names_to_the_person_and_their_pets(records):
    listed_people, listed_pets = set(smalltalk._user_names()), set(smalltalk.SLOTS["pname"])
    people = fresh_people = pets = fresh_pets = 0
    for r in records:
        if r.name:
            people += 1
            if r.name not in listed_people:
                fresh_people += 1
                assert _passes_the_name_checks(r.name), r.name
        pet = r.slots.get("pname")
        if pet and any(pet in t for role, t in r.turns if role == "user"):
            pets += 1
            if pet not in listed_pets:
                fresh_pets += 1
                assert _passes_the_name_checks(pet), pet
    assert people >= 500 and fresh_people / people >= 0.3, (people, fresh_people)
    assert pets >= 80 and fresh_pets / pets >= 0.3, (pets, fresh_pets)
