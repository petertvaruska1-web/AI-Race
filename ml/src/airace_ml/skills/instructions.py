"""Instruction following as chats: training conversations and benchmark items.

A user gives a short instruction and the AI follows it: "What is the capital of France? Answer
with one word." / "Paris." Seven kinds of instruction, each judged by the checker of the same
name in ``checkers.CHECKERS`` (benchmark items are tagged ``fam:<kind>``):

- ``one_word``: a question from the knowledge base, to be answered with one word (a "yes" or
  "no" is not an answer to it: ``args["not"]``)
- ``yes_no``: a statement about a fact (or "is 7 bigger than 4?"), to be answered yes or no
- ``list_n``: "List three mammals.", "Name two words that start with the letter B.": 2-6 items
  that are animals of a class, foods of a group, countries of a continent, things of a color,
  animals by what they eat, where they live or how many legs they have, vehicles by how they
  travel, or words by their first or last letter
- ``starts_with``: a question, to be answered in a reply that starts with a given word
- ``all_caps``: a question, to be answered with a sentence in capital letters (at least 3 words:
  ``args["min_words"]``)
- ``repeat_word``: "Say the word banana and nothing else."
- ``contains_any``: "Write one sentence that uses the word dog."

The benchmark measures whether the reply has the asked *form*, not whether its content is right.
It is built so that no one constant reply passes more than one kind: "Yes" is not a one-word
answer, "YES" is not a sentence in capitals, and the other kinds ask for a word, a prefix or a
number of items that changes from item to item.

Training replies are written to be right as well: the facts come from the knowledge base (the
statements from the relations' ``train_templates``, the questions from their ``chat_templates``,
never the benchmark-only phrasings of ``facts``), a claim in a yes-or-no question is true or
clearly false by the knowledge base and the reply says which, and every reply passes its checker,
which :func:`_render` makes sure of.

Partition. An instruction belongs to a *world*: its kind and what it is about (the fact, the word,
the group of things and how many), whatever its wording. Benchmark items use only reserved worlds
and prompts (:func:`reserved_for_bench`), training text uses neither: so no benchmark instruction
occurs in training, nor the same instruction about the same thing in other words. The worlds are
listed from the knowledge base up front, which lets the benchmark share its items evenly over the
kinds as far as each kind's reserved worlds allow.

Nothing here is a model output; these are data generators.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from airace_ml.skills.checkers import CHECKERS
from airace_ml.skills.kb import KB, Fact
from airace_ml.skills.types import CheckItem, TextDoc, fair_quota, reserved_for_bench
from airace_ml.tokenizer import Role

KINDS = (
    "one_word",
    "yes_no",
    "list_n",
    "starts_with",
    "all_caps",
    "repeat_word",
    "contains_any",
)
RENDER_TRIES = 40  # wordings tried for an accepted world
MAX_ATTEMPTS_PER_EXCHANGE = 50  # renderings, pooled over a request
EXCHANGES_PER_CHAT = (1, 2, 3)  # how many instructions a training chat holds ...
EXCHANGE_WEIGHTS = (0.5, 0.3, 0.2)  # ... and how often
MAX_NEW_TOKENS = 64
MIN_CAPS_WORDS = 3  # a sentence in capitals is at least this many words
MIN_MEMBERS_OVER_N = 2  # a list world needs this many more members than items asked
LIST_SIZES = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PERIOD_SHARE = 0.7  # share of short replies that end with a period
NUMBER_RANGE = 20  # "is a bigger than b" uses 1..20
WORD_PATTERN = re.compile(r"[A-Za-z]{3,12}")
PREFIXES = ("Sure", "Okay", "Hello", "Well", "Great", "Hi", "Alright", "Certainly")

# What a list can ask for, by (relation, object). Written out, so plurals and articles are right
# ("fish" does not take an "s"; "in a nest", "in the ocean").
LIST_THINGS = {
    ("animal_class", "mammal"): "mammals",
    ("animal_class", "bird"): "birds",
    ("animal_class", "fish"): "fish",
    ("animal_class", "reptile"): "reptiles",
    ("animal_class", "amphibian"): "amphibians",
    ("animal_class", "insect"): "insects",
    ("food_group", "fruit"): "fruits",
    ("food_group", "vegetable"): "vegetables",
    ("food_group", "grain"): "grains",
    ("food_group", "dairy"): "dairy foods",
    ("food_group", "protein"): "protein foods",
    ("animal_food", "insects"): "animals that eat insects",
    ("animal_food", "grass"): "animals that eat grass",
    ("animal_food", "meat"): "animals that eat meat",
    ("animal_food", "seeds"): "animals that eat seeds",
    ("animal_food", "fish"): "animals that eat fish",
    ("animal_food", "fruit"): "animals that eat fruit",
    ("animal_home", "ocean"): "animals that live in the ocean",
    ("animal_home", "jungle"): "animals that live in the jungle",
    ("animal_home", "nest"): "animals that live in a nest",
    ("animal_home", "burrow"): "animals that live in a burrow",
    ("animal_home", "savanna"): "animals that live in the savanna",
    ("animal_legs", "four"): "animals with four legs",
    ("animal_legs", "two"): "animals with two legs",
    ("animal_legs", "six"): "animals with six legs",
    ("animal_legs", "zero"): "animals with no legs",
    ("vehicle_travel", "by water"): "vehicles that travel by water",
    ("vehicle_travel", "by road"): "vehicles that travel by road",
    ("vehicle_travel", "by air"): "vehicles that travel by air",
}

ONE_WORD_WORDINGS = (
    "{q} Answer with one word.",
    "{q} Reply with one word only.",
    "{q} Use just one word.",
    "{q} Give a one-word answer.",
    "Answer in one word: {q}",
    "One word only: {q}",
)
YES_NO_WORDINGS = (
    "Is this true? {st} Answer yes or no.",
    "{st} Is that right? Answer yes or no.",
    "Is this statement true? {st} Reply with yes or no.",
    "Yes or no: is this true? {st}",
)
COMPARE_WORDINGS = (
    "Is {a} bigger than {b}? Answer yes or no.",
    "Is {a} greater than {b}? Reply with yes or no.",
    "Is {a} larger than {b}? Answer with yes or no.",
    "Yes or no: is {a} more than {b}?",
)
LIST_WORDINGS = (
    "List {n} {things}.",
    "Name {n} {things}.",
    "Give me {n} {things}.",
    "Write down {n} {things}.",
    "Tell me {n} {things}.",
    "Can you list {n} {things}?",
    "Think of {n} {things} and list them.",
    "I need {n} {things}. List them.",
)
STARTS_WITH_WORDINGS = (
    "{q} Start your answer with the word {p}.",
    "{q} Begin your reply with the word {p}.",
    "{q} Your reply must start with the word {p}.",
    "Start your reply with the word {p}. {q}",
)
ALL_CAPS_WORDINGS = (
    "{q} Answer in a sentence, in all capital letters.",
    "{q} Reply with a sentence in ALL CAPS.",
    "{q} Write your answer as a sentence in capital letters only.",
    "{q} Use only capital letters and answer in a full sentence.",
    "In all capital letters, answer in a sentence: {q}",
    "Use capital letters only, and answer in a sentence: {q}",
)
REPEAT_WORDINGS = (
    "Say the word {w}.",
    "Repeat the word {w}.",
    "Say {w} and nothing else.",
    "Just say {w}.",
    "Reply with the word {w} only.",
    "Repeat after me: {w}",
    "Please repeat the word {w}.",
    "Write the word {w} and nothing else.",
)
CONTAINS_WORDINGS = (
    "Write one sentence that uses the word {w}.",
    "Use the word {w} in a sentence.",
    "Make a sentence with the word {w} in it.",
    "Write a short sentence. It must include the word {w}.",
    "Say something that includes the word {w}.",
    "Please write a sentence with the word {w}.",
    "Give me a sentence that has the word {w} in it.",
    "Write a sentence about {w}. Use the word {w}.",
)


@dataclass(frozen=True)
class _World:
    """What an instruction is about, apart from its wording. ``key`` is hashed by the partition."""

    kind: str
    key: str
    topic: str
    data: tuple


@dataclass(frozen=True)
class _Exchange:
    """One instruction and a reply that passes its checker."""

    world: str  # the world's key
    kind: str
    topic: str
    user: str
    reply: str
    check_args: dict


def _pick(rng: np.random.Generator, n: int) -> int:
    return int(rng.integers(n))


def _choice[T](rng: np.random.Generator, options: tuple[T, ...] | list[T]) -> T:
    return options[_pick(rng, len(options))]


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def _short_reply(rng: np.random.Generator, text: str) -> str:
    """A short answer, with or without a period."""
    return text + ("." if rng.random() < PERIOD_SHARE else "")


def _statement(kb: KB, fact: Fact, rng: np.random.Generator) -> str:
    templates = kb.relations[fact.relation].train_templates
    return _choice(rng, templates).format(s=fact.subject, o=fact.obj)


def _question(kb: KB, fact: Fact, rng: np.random.Generator) -> str:
    users = [user for user, _ in kb.relations[fact.relation].chat_templates]
    return _choice(rng, users).format(s=fact.subject)


def _topic(kb: KB, fact: Fact) -> str:
    return kb.relations[fact.relation].topic


# --- the worlds of each kind ---------------------------------------------------------------------


def _fact_worlds(kb: KB, kind: str, facts: Collection[Fact]) -> list[_World]:
    return [_World(kind, f"{kind}|{f.subject}|{f.relation}", _topic(kb, f), (f,)) for f in facts]


def _one_word_worlds(kb: KB) -> list[_World]:
    facts = [f for f in kb.facts if CHECKERS["one_word"](f.obj, {})]
    return _fact_worlds(kb, "one_word", facts)


def _yes_no_worlds(kb: KB) -> list[_World]:
    claimable = [
        f
        for f in kb.facts
        if kb.relations[f.relation].falsifiable and any(o != f.subject for o in kb.wrong_objects(f))
    ]
    worlds = _fact_worlds(kb, "yes_no", claimable)
    for a in range(1, NUMBER_RANGE + 1):
        for b in range(1, NUMBER_RANGE + 1):
            if a != b:
                worlds.append(_World("yes_no", f"yes_no|num|{a}|{b}", "school", (a, b)))
    return worlds


def _list_groups(kb: KB) -> list[tuple[str, str, str, list[str]]]:
    """(id, the things asked for, topic, members) for every group a list can be about."""
    groups: list[tuple[str, str, str, list[str]]] = []

    def members(relation: str, obj: str) -> list[str]:
        return sorted(
            f.subject
            for f in kb.facts_for(relation)
            if f.obj == obj
            and not f.subject.startswith("the ")
            and "," not in f.subject
            and len(f.subject.split()) <= 2
        )

    things = dict(LIST_THINGS)
    things.update(
        {("continent_of", o): f"countries in {o}" for o in kb.objects_for("continent_of")}
    )
    things.update({("color_of", o): f"things that are {o}" for o in kb.objects_for("color_of")})
    for (relation, obj), what in things.items():
        topic = kb.relations[relation].topic
        groups.append((f"{relation}:{obj}", what, topic, members(relation, obj)))
    starts: dict[str, list[str]] = {}
    ends: dict[str, list[str]] = {}
    for subject in kb.subjects:
        if subject.islower() and WORD_PATTERN.fullmatch(subject):
            starts.setdefault(subject[0], []).append(subject)
            ends.setdefault(subject[-1], []).append(subject)
    for letter, found in sorted(starts.items()):
        what = f"words that start with the letter {letter.upper()}"
        groups.append((f"starts:{letter}", what, "school", sorted(found)))
    for letter, found in sorted(ends.items()):
        what = f"words that end with the letter {letter.upper()}"
        groups.append((f"ends:{letter}", what, "school", sorted(found)))
    return groups


def _list_n_worlds(kb: KB) -> list[_World]:
    worlds = []
    for group, things, topic, found in _list_groups(kb):
        for n in LIST_SIZES:
            if len(found) >= n + MIN_MEMBERS_OVER_N:
                key = f"list_n|{group}|{n}"
                worlds.append(_World("list_n", key, topic, (things, n, tuple(found))))
    return worlds


def _starts_with_worlds(kb: KB) -> list[_World]:
    return _fact_worlds(kb, "starts_with", kb.facts)


def _all_caps_worlds(kb: KB) -> list[_World]:
    return _fact_worlds(kb, "all_caps", kb.facts)


def _words(kb: KB, *, subjects_only: bool) -> dict[str, str]:
    """Each plain word the knowledge base has (as a subject, or also as an object), with a topic."""
    found: dict[str, str] = {}
    for fact in kb.facts:
        for word in (fact.subject, *(() if subjects_only else (fact.obj,))):
            if WORD_PATTERN.fullmatch(word):
                found.setdefault(word, _topic(kb, fact))
    return found


def _repeat_word_worlds(kb: KB) -> list[_World]:
    return [
        _World("repeat_word", f"repeat_word|{word}", topic, (word,))
        for word, topic in sorted(_words(kb, subjects_only=False).items())
    ]


def _contains_any_worlds(kb: KB) -> list[_World]:
    return [
        _World("contains_any", f"contains_any|{word}", topic, (word,))
        for word, topic in sorted(_words(kb, subjects_only=True).items())
    ]


_WORLD_BUILDERS = {
    "one_word": _one_word_worlds,
    "yes_no": _yes_no_worlds,
    "list_n": _list_n_worlds,
    "starts_with": _starts_with_worlds,
    "all_caps": _all_caps_worlds,
    "repeat_word": _repeat_word_worlds,
    "contains_any": _contains_any_worlds,
}


@lru_cache(maxsize=2)
def _pools(kb: KB) -> tuple[dict[str, tuple[_World, ...]], dict[str, tuple[_World, ...]]]:
    """Every world of every kind, split into the benchmark's (reserved) and training's."""
    bench: dict[str, tuple[_World, ...]] = {}
    train: dict[str, tuple[_World, ...]] = {}
    for kind in KINDS:
        worlds = _WORLD_BUILDERS[kind](kb)
        bench[kind] = tuple(w for w in worlds if reserved_for_bench(w.key))
        train[kind] = tuple(w for w in worlds if not reserved_for_bench(w.key))
    return bench, train


# --- rendering a world as an instruction and a reply ---------------------------------------------


def _one_word(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    (fact,) = world.data
    user = _choice(rng, ONE_WORD_WORDINGS).format(q=_question(kb, fact, rng))
    return user, _short_reply(rng, _capitalized(fact.obj)), {"not": ["yes", "no"]}


def _yes_no(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    if world.data and isinstance(world.data[0], Fact):
        (fact,) = world.data
        true = bool(rng.integers(2))
        wrong = [o for o in kb.wrong_objects(fact) if o != fact.subject]
        claimed = Fact(fact.subject, fact.relation, fact.obj if true else _choice(rng, wrong))
        statement = _statement(kb, claimed, rng)
        user = _choice(rng, YES_NO_WORDINGS).format(st=statement)
    else:
        a, b = world.data
        true = a > b
        user = _choice(rng, COMPARE_WORDINGS).format(a=a, b=b)
    return user, _short_reply(rng, "Yes" if true else "No"), {}


def _list_n(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    things, n, found = world.data
    user = _choice(rng, LIST_WORDINGS).format(n=LIST_SIZES[n], things=things)
    items = [found[int(i)] for i in rng.choice(len(found), size=n, replace=False)]
    match _pick(rng, 10):
        case 0 | 1 | 2 | 3:  # "a, b and c"
            reply = ", ".join(items[:-1]) + " and " + items[-1]
        case 4 | 5 | 6:  # "a, b, c"
            reply = ", ".join(items)
        case 7 | 8:  # "a, b, and c"
            reply = ", ".join(items[:-1]) + (", and " if n > 2 else " and ") + items[-1]
        case _:  # one a line
            reply = "\n".join(items)
    return user, reply, {"n": n}


def _starts_with(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    (fact,) = world.data
    prefix = _choice(rng, PREFIXES)
    user = _choice(rng, STARTS_WITH_WORDINGS).format(q=_question(kb, fact, rng), p=prefix)
    return user, f"{prefix}! {_statement(kb, fact, rng)}", {"prefix": prefix}


def _all_caps(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    (fact,) = world.data
    user = _choice(rng, ALL_CAPS_WORDINGS).format(q=_question(kb, fact, rng))
    return user, _statement(kb, fact, rng).upper(), {"min_words": MIN_CAPS_WORDS}


def _repeat_word(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    (word,) = world.data
    user = _choice(rng, REPEAT_WORDINGS).format(w=word)
    return user, _short_reply(rng, word), {"word": word}


def _contains_any(kb: KB, world: _World, rng: np.random.Generator) -> tuple[str, str, dict]:
    (word,) = world.data
    fact = _choice(rng, kb.facts_about(word))
    user = _choice(rng, CONTAINS_WORDINGS).format(w=word)
    return user, _statement(kb, fact, rng), {"words": [word]}


_RENDERERS = {
    "one_word": _one_word,
    "yes_no": _yes_no,
    "list_n": _list_n,
    "starts_with": _starts_with,
    "all_caps": _all_caps,
    "repeat_word": _repeat_word,
    "contains_any": _contains_any,
}


def _render(kb: KB, world: _World, rng: np.random.Generator, *, bench: bool) -> _Exchange | None:
    """An instruction about the world with a reply that passes its checker, in the given split.

    The wording is drawn at random until the prompt is reserved (``bench``) or not; ``None`` if no
    wording within ``RENDER_TRIES`` fits.
    """
    for _ in range(RENDER_TRIES):
        user, reply, args = _RENDERERS[world.kind](kb, world, rng)
        if reserved_for_bench(user) == bench and CHECKERS[world.kind](reply, args):
            return _Exchange(world.key, world.kind, world.topic, user, reply, args)
    return None


# --- public generators --------------------------------------------------------------------------


def _train_chats(kb: KB, rng: np.random.Generator, n: int) -> list[list[_Exchange]]:
    """``n`` training chats as lists of exchanges, all in training worlds and wordings."""
    pools = _pools(kb)[1]
    sizes = rng.choice(EXCHANGES_PER_CHAT, size=n, p=EXCHANGE_WEIGHTS)
    budget = MAX_ATTEMPTS_PER_EXCHANGE * int(sizes.sum())
    chats: list[list[_Exchange]] = []
    for size in sizes:
        chat: list[_Exchange] = []
        while len(chat) < size:
            if budget <= 0:
                raise RuntimeError(
                    "instructions: could not draw enough exchanges within the budget"
                )
            budget -= 1
            pool = pools[KINDS[_pick(rng, len(KINDS))]]
            world = pool[_pick(rng, len(pool))]
            if any(world.key == done.world for done in chat):
                continue
            exchange = _render(kb, world, rng, bench=False)
            if exchange is not None:
                chat.append(exchange)
        chats.append(chat)
    return chats


def instruction_train_docs(kb: KB, rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` chats of 1-3 instructions, each followed by a reply that passes its checker.

    Instructions are of every kind in turn at random, about random facts and words of the
    knowledge base. No benchmark instruction occurs, nor the same one about the same thing in
    other words, and no world occurs twice in a chat.
    """
    docs = []
    for chat in _train_chats(kb, rng, n):
        turns: list[tuple[Role, str]] = []
        for exchange in chat:
            turns += [("user", exchange.user), ("ai", exchange.reply)]
        docs.append(TextDoc("chat", turns=turns, topic=chat[0].topic))
    return docs


def instruction_bench_items(kb: KB, rng: np.random.Generator, n: int = 120) -> list[CheckItem]:
    """``n`` instructions to follow (category ``"instruction"``, ``instruction-0000``...).

    The kinds share the items as evenly as their reserved worlds allow and take turns, so any
    prefix is as balanced as it can be; a kind appears as ``fam:<kind>`` and its topic as
    ``topic:<topic>``. Every item is a user message (``chat=True``) judged by the kind's checker,
    with ``reference`` a reply that passes it. No world occurs twice, and every world and prompt
    is reserved for the benchmark. ``ValueError`` if ``n`` is more than the worlds there are.
    """
    pools = _pools(kb)[0]
    quota = fair_quota({kind: len(pool) for kind, pool in pools.items()}, n)
    chosen: dict[str, list[_Exchange]] = {}
    for kind in KINDS:
        pool = pools[kind]
        exchanges: list[_Exchange] = []
        for i in rng.permutation(len(pool)):
            if len(exchanges) == quota[kind]:
                break
            exchange = _render(kb, pool[int(i)], rng, bench=True)
            if exchange is not None:
                exchanges.append(exchange)
        if len(exchanges) < quota[kind]:
            raise RuntimeError(f"instructions: could not draw enough {kind} items")
        chosen[kind] = exchanges
    ordered = [
        chosen[kind][k]
        for k in range(max(quota.values()))
        for kind in KINDS
        if k < len(chosen[kind])
    ]
    return [
        CheckItem(
            f"instruction-{i:04d}",
            "instruction",
            e.user,
            e.kind,
            e.check_args,
            e.reply,
            (f"fam:{e.kind}", f"topic:{e.topic}"),
            chat=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for i, e in enumerate(ordered)
    ]
