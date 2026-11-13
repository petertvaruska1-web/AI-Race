"""Talk probes: how a trained AI talks, measured with short chats.

Every probe takes a model with the chat interface of :class:`~airace_ml.infer.lm.TorchLM`
(``chat_reply`` and ``chat_generation``), so it can be tested with a scripted one. Replies are
drawn the way the game draws them (:data:`SERVING_TEMPERATURE`, :data:`SERVING_TOP_P`, loops
ended) unless a probe says it is greedy. Nothing here writes or changes what an AI says (R1):
the probes only read replies and score them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airace_ml.infer.serving import SERVING_TEMPERATURE, SERVING_TOP_P
from airace_ml.persona import claims_person_life, identity_line
from airace_ml.skills.probe_split import is_probe_subject

if TYPE_CHECKING:
    from airace_ml.infer.lm import TorchLM
    from airace_ml.skills.kb import KB, Fact, Relation

SMALL_TALK = [
    "Hello!", "Hi there.", "How are you?", "Good morning.", "What is your name?", "Thank you.",
    "Can you help me?", "Tell me a story.", "What do you like to do?", "I am happy today.",
    "Tell me a joke.", "What is your favorite color?", "Do you like dogs?", "I feel sad.",
    "What should I eat for lunch?", "Goodbye.", "Tell me about cats.", "How was your day?",
    "Can we be friends?", "What can you do?",
]  # fmt: skip


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:]


STYLES = {
    "clean": lambda s: s,
    "lower": lambda s: s.lower(),
    "lower_nopunct": lambda s: s.lower().rstrip(".?!"),
    "nopunct": lambda s: s.rstrip(".?!"),
    "cap_first(lower_nopunct)": lambda s: _cap_first(s.lower().rstrip(".?!")),
}

IDENTITY_NAMES = ("Zula", "Pip", "Orbit", "Mako", "Tenby", "Quill")
IDENTITY_COMPANY = "Pengu"
NAME_QUESTIONS = ("What is your name?", "Who are you?", "what's your name")
KIND_QUESTIONS = ("Are you a person?", "Are you human?")

FOLLOWUP_TEMPLATES = ("And {b}?", "What about {b}?", "And for {b}?")

# Made-up facts no corpus holds, each answer ending in its key word.
LESSONS: tuple[tuple[str, str], ...] = (
    ("What colour is a glimmerfin?", "A glimmerfin is teal."),
    ("Where does a brindlewump live?", "A brindlewump lives in a cave."),
    ("What does a snorfle eat?", "A snorfle eats pebbles."),
    ("How many legs does a zibblet have?", "A zibblet has seven legs."),
    ("What sound does a plonkabee make?", "A plonkabee makes a hum."),
    ("What is a wuzzle made of?", "A wuzzle is made of copper."),
    ("When does a mirrowisp wake up?", "A mirrowisp wakes up at midnight."),
    ("What shape is a doppleflit?", "A doppleflit is shaped like a star."),
    ("Who looks after the frumblebush?", "The frumblebush is looked after by a badger."),
    ("What does a tinkerlark collect?", "A tinkerlark collects buttons."),
)

_CASUAL_TOKENS = 24
_FOLLOWUP_TRIES = 50  # draws per wanted item before giving up on finding more
_CHAT_TOKENS = 96
_EXAMPLES_PER_SIDE = 4
_ENDINGS = (".", "!", "?", '"', "'", "”", "’")
_AI_WORD = re.compile(r"\bAI\b")


@dataclass
class ProbeScore:
    """How many items of a probe were right, with a few ``(prompt, reply, right)`` examples."""

    passed: int
    total: int
    examples: list[tuple[str, str, bool]]

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def _score(items: list[tuple[str, str, bool]]) -> ProbeScore:
    """A score from every item; the examples are the first failures, then the first passes."""
    failed = [i for i in items if not i[2]][:_EXAMPLES_PER_SIDE]
    right = [i for i in items if i[2]][:_EXAMPLES_PER_SIDE]
    return ProbeScore(sum(i[2] for i in items), len(items), failed + right)


def _holds_word(text: str, phrase: str) -> bool:
    """Whether ``phrase`` appears in ``text`` as whole words, ignoring case and the amount of
    space between its words ("four" is not in "fourteen", "7" is not in "17")."""
    words = phrase.split()
    if not words:
        return False
    pattern = r"\s+".join(re.escape(w) for w in words)
    return re.search(rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE) is not None


def _holds(reply: str, answers: list[str]) -> bool:
    return any(_holds_word(reply, a) for a in answers)


def _greedy(lm: TorchLM, history: list, max_new_tokens: int, system: str | None = None) -> str:
    return lm.chat_reply(
        history, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, seed=0, system=system
    )


def _served(lm: TorchLM, history: list, seed: int, system: str | None = None) -> str:
    return lm.chat_reply(
        history,
        max_new_tokens=_CHAT_TOKENS,
        temperature=SERVING_TEMPERATURE,
        top_p=SERVING_TOP_P,
        seed=seed,
        stop_on_loop=True,
        system=system,
    )


def _first_question(relation: Relation, subject: str) -> str:
    return relation.chat_templates[0][0].replace("{s}", subject)


def casual_facts(
    lm: TorchLM, kb: KB, *, per_relation: int = 5, seed: int = 0
) -> dict[str, ProbeScore]:
    """Facts asked in the training phrasing, typed five ways (greedy). Right when the reply holds
    an accepted answer. The keys are the names in :data:`STYLES`."""
    rng = np.random.default_rng(seed)
    chosen: list[tuple[Relation, Fact]] = []
    for name, relation in kb.relations.items():
        facts = kb.facts_for(name)
        for k in rng.permutation(len(facts))[:per_relation]:
            chosen.append((relation, facts[int(k)]))
    out: dict[str, ProbeScore] = {}
    for style, fn in STYLES.items():
        items = []
        for relation, fact in chosen:
            prompt = fn(_first_question(relation, fact.subject))
            reply = _greedy(lm, [("user", prompt)], _CASUAL_TOKENS)
            items.append((prompt, reply, _holds(reply, kb.accepted_answers(fact))))
        out[style] = _score(items)
    return out


def followups(lm: TorchLM, kb: KB, *, n: int = 60, seed: int = 0) -> ProbeScore:
    """Two-turn chats: a fact about subject A, the true answer, then "And B?" (greedy). Right
    when the reply holds B's accepted answer. B is always a held-back probe subject
    (:func:`~airace_ml.skills.probe_split.is_probe_subject`), so no AI was taught it in a second
    turn. Pairs whose answers could be mistaken for each other are left out."""
    rng = np.random.default_rng(seed)
    names = [name for name in kb.relations if len(kb.facts_for(name)) >= 2]
    items: list[tuple[str, str, bool]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for _ in range(_FOLLOWUP_TRIES * n):
        if len(items) >= n or not names:
            break
        relation = kb.relations[names[int(rng.integers(len(names)))]]
        facts = kb.facts_for(relation.name)
        a = facts[int(rng.integers(len(facts)))]
        held = [f for f in facts if is_probe_subject(relation.name, f.subject)]
        if not held:
            continue
        b = held[int(rng.integers(len(held)))]
        a_answers, b_answers = kb.accepted_answers(a), kb.accepted_answers(b)
        if a.subject == b.subject or any(
            x.lower() in y.lower() or y.lower() in x.lower() for x in a_answers for y in b_answers
        ):
            continue
        ask_b = FOLLOWUP_TEMPLATES[int(rng.integers(len(FOLLOWUP_TEMPLATES)))].format(b=b.subject)
        key = (relation.name, a.subject, b.subject, ask_b)
        if key in seen:
            continue
        seen.add(key)
        history = [
            ("user", _first_question(relation, a.subject)),
            ("ai", relation.chat_templates[0][1].format(s=a.subject, o=a.obj)),
            ("user", ask_b),
        ]
        reply = _greedy(lm, history, _CASUAL_TOKENS)
        items.append((ask_b, reply, _holds(reply, b_answers)))
    if not items:
        raise ValueError(
            "no follow-up item can be built: it needs a relation with two subjects, one of them "
            "a probe subject, whose answers differ"
        )
    return _score(items)


def identity(
    lm: TorchLM, *, with_line: bool, names: tuple[str, ...] = IDENTITY_NAMES
) -> ProbeScore:
    """Whether the AI knows who it is. With the identity line (once per name, company
    :data:`IDENTITY_COMPANY`): name questions pass when the reply holds the name, "are you a
    person" questions when it holds "AI" as a word and claims no human life. Without the line every
    question passes only when the reply says it is an AI ("AI" as a word) and claims no human life
    (avoiding a human life alone is no answer: an AI that says nothing about itself passes that)."""
    items = []
    for seed, name in enumerate(names):
        system = identity_line(name, IDENTITY_COMPANY) if with_line else None
        for question in (*NAME_QUESTIONS, *KIND_QUESTIONS):
            reply = _served(lm, [("user", question)], seed, system)
            human = claims_person_life(reply)
            if not with_line:
                ok = _AI_WORD.search(reply) is not None and not human
            elif question in NAME_QUESTIONS:
                ok = _holds_word(reply, name)
            else:
                ok = _AI_WORD.search(reply) is not None and not human
            items.append((question, reply, ok))
    return _score(items)


def _cut_off(text: str, hit_limit: bool) -> bool:
    return hit_limit and not text.rstrip().endswith(_ENDINGS)


def small_talk(
    lm: TorchLM, well_formed: Callable[[str], bool], *, samples: int = 3
) -> dict[str, float]:
    """Small talk over :data:`SMALL_TALK`, ``samples`` seeds each. The rates of replies that are
    ``well_formed`` (as the caller judges), claim a ``person_life``, are ``cut_off`` (used every
    allowed token and stop mid-sentence) and ``looped``."""
    counts = {"well_formed": 0, "person_life": 0, "cut_off": 0, "looped": 0}
    total = 0
    for seed in range(samples):
        for prompt in SMALL_TALK:
            reply, gen = lm.chat_generation(
                [("user", prompt)],
                max_new_tokens=_CHAT_TOKENS,
                temperature=SERVING_TEMPERATURE,
                top_p=SERVING_TOP_P,
                seed=seed,
                stop_on_loop=True,
            )
            total += 1
            counts["well_formed"] += bool(well_formed(reply))
            counts["person_life"] += claims_person_life(reply)
            counts["cut_off"] += _cut_off(reply, gen.hit_limit)
            counts["looped"] += gen.looped
    return {k: v / total if total else 0.0 for k, v in counts.items()}


def _key_word(answer: str) -> str:
    """The last word of a lesson's answer, lower case, without punctuation."""
    return answer.rstrip(" .!?").split()[-1].lower()


def lessons(train: Callable[[list[tuple[str, str]]], TorchLM], *, n: int = 10) -> ProbeScore:
    """Whether coaching sticks. ``train`` continues the model with the first ``n`` of
    :data:`LESSONS`, as the game trains coaching, and returns it; each question is then asked
    (greedy) and is right when the reply holds the answer's key word."""
    pairs = list(LESSONS[:n])
    lm = train(pairs)
    items = []
    for question, answer in pairs:
        reply = _greedy(lm, [("user", question)], _CASUAL_TOKENS)
        items.append((question, reply, _holds_word(reply, _key_word(answer))))
    return _score(items)
