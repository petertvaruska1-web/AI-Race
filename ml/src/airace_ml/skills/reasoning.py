"""Reasoning puzzles as text: training blocks and benchmark items.

Four families of tiny puzzles, all in the frame ``Question: ...`` / ``Answer: ...``:

- ``compare``: three people in a chain ("Tom is taller than Ben. Ben is taller than Sam.") and a
  question about the top, the bottom or the middle of it. A fourth name, the odd one out among the
  options, is in a sentence that says nothing about the order.
- ``syllogism``: made-up category words ("All blicks are fenks. No zorps are glorps. Tom is a blick.
  Is Tom a fenk?"), answered yes or no. Every puzzle has the same quantifiers whatever its answer;
  one premise is a decoy about words the question does not mention.
- ``word_problem``: one addition or subtraction story, with numbers up to 20.
- ``count``: how many times a word is in a short list.

Partition. A puzzle belongs to a *world*: the facts it is about, whatever their wording (for
``compare`` the chain of names and the adjective; for ``syllogism`` the set of premises and the
question; for ``word_problem`` who has what and the operation and numbers; for ``count`` the word
and the multiset of list items). The key of a world is hashed by :func:`reserved_for_bench`.
Benchmark items use only worlds whose key is reserved, worded as a prompt (``Question:
...\\nAnswer:``) that is itself reserved; training blocks use only worlds and prompts that are not
reserved. So training text never contains a benchmark question, nor the same facts in other words
(other premise order, polarity, question or story). Word problems with the same operation and numbers
about other people and things do recur, which is fine for an arithmetic skill.

Worlds and wordings are drawn at random and accepted or rejected by that rule, with a budget of 50
world draws per requested item. Nothing here is a model output; these are data generators.
"""

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from airace_ml.skills.types import MCItem, TextDoc, reserved_for_bench

FAMILIES = ("compare", "syllogism", "word_problem", "count")
# Each family has a small control number that fixes one balanced property of a puzzle: the question
# asked (compare), the answer and the number of steps (syllogism), the operation (word_problem) and
# the answer (count). A benchmark steps through the controls so the property is balanced; training
# draws them at random.
N_CONTROLS = {"compare": 3, "syllogism": 4, "word_problem": 2, "count": 6}
TOPIC = "school"
MAX_ATTEMPTS_PER_ITEM = 50  # world draws
RENDER_TRIES = 40  # wordings tried for an accepted world
MIN_BLOCKS, MAX_BLOCKS = 3, 8  # puzzles in one training document
EXPLAIN_SHARE = 0.4  # share of training blocks followed by a one-sentence explanation
N_OPTIONS = 4
NUMBER_RANGE = {"word_problem": (0, 20), "count": (0, 8)}  # the numbers an option may be
NEARBY = 5  # how far a number option is from the answer, at most

BOYS = ("Tom", "Ben", "Sam", "Max", "Leo", "Jack", "Ned", "Tim", "Dan", "Joe", "Ray", "Luke")
GIRLS = ("Mia", "Ana", "Eve", "Amy", "Zoe", "Liz", "Kim", "Ivy", "Sue", "Meg", "Beth", "Cara")
NAMES = BOYS + GIRLS

# Made-up words for the syllogisms. Each starts with a consonant (so "a" is always right) and
# makes its plural with a plain "s".
# fmt: off
NONCE_WORDS = (
    "blick", "fenk", "zorp", "wug", "glorp", "nim", "plork", "skib", "trel", "vump",
    "yelk", "quib", "rund", "brab", "cleb", "drin", "frob", "gleb", "hup", "jorn",
    "kep", "lorm", "mib", "nerp", "pab", "rold", "sarn", "tuk", "vorn", "wib",
)

# Plurals made with a plain "s" (never "ch", "o" or consonant + "y" endings).
THINGS = (
    "apples", "pears", "cookies", "balls", "stickers", "marbles", "books", "coins", "cards",
    "shells", "stones", "pens", "hats", "kites", "toys", "cups", "eggs", "cakes",
)
# fmt: on
COUNT_GROUPS = (
    ("cat", "dog", "cow", "pig", "hen", "bird", "duck", "frog", "lion", "bear", "goat", "horse"),
    ("apple", "pear", "plum", "lemon", "grape", "melon", "banana", "fig", "lime", "kiwi"),
)

# (comparative, superlative, opposite comparative, opposite superlative)
ADJECTIVES = (
    ("taller", "tallest", "shorter", "shortest"),
    ("older", "oldest", "younger", "youngest"),
    ("faster", "fastest", "slower", "slowest"),
    ("heavier", "heaviest", "lighter", "lightest"),
)
# Sentences about the odd name out ({e}) and one of the three in the chain ({x}) that say nothing
# about the order.
ASIDES = (
    "{e} is a friend of {x}.",
    "{e} lives near {x}.",
    "{e} sits next to {x}.",
    "{e} and {x} are friends.",
    "{e} plays with {x}.",
    "{x} likes {e}.",
)

ADD_STORIES = (
    "{n} has {a} {t}. {s} gets {b} more. How many {t} does {n} have now?",
    "{n} has {a} {t}. A friend gives {o} {b} more. How many {t} does {n} have now?",
    "{n} has {a} {t} and finds {b} more. How many {t} does {n} have now?",
    "{n} had {a} {t}. Then {p} got {b} more. How many {t} does {n} have now?",
    "There are {a} {t} in a box. {n} puts {b} more in. How many {t} are in the box now?",
)
SUBTRACT_STORIES = (
    "{n} has {a} {t}. {s} gives {b} away. How many {t} does {n} have now?",
    "{n} has {a} {t}. {s} loses {b}. How many {t} does {n} have now?",
    "{n} has {a} {t} and gives {b} to a friend. How many {t} does {n} have now?",
    "{n} had {a} {t}. Then {p} lost {b}. How many {t} does {n} have now?",
    "There are {a} {t} in a box. {n} takes {b} out. How many {t} are in the box now?",
)


@dataclass(frozen=True)
class _World:
    """The facts of a puzzle, apart from how they are worded. ``key`` is what the partition hashes."""

    family: str
    key: str
    data: tuple


@dataclass(frozen=True)
class _Question:
    """One wording of a world: the question text, its answer and the options a benchmark offers."""

    family: str
    world: str  # the world's key
    text: str
    answer: str
    explanation: str
    options: tuple[str, ...] = ()  # names or yes/no; empty for number answers (see _number_options)
    shuffle: bool = True  # whether a benchmark mixes the order of the options

    @property
    def prompt(self) -> str:
        """What the model is shown before it answers."""
        return f"Question: {self.text}\nAnswer:"


def _pick(rng: np.random.Generator, n: int) -> int:
    return int(rng.integers(n))


def _flip(rng: np.random.Generator) -> bool:
    return bool(rng.integers(2))


def _sample(rng: np.random.Generator, pool: tuple[str, ...], k: int) -> list[str]:
    """``k`` different members of ``pool``."""
    return [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]


def _shuffled(rng: np.random.Generator, items: list[str]) -> list[str]:
    return [items[int(i)] for i in rng.permutation(len(items))]


def _feasible_ranks(answer: int, lo: int, hi: int) -> list[int]:
    """The ranks (0 = smallest) the answer can have among 4 options within ``NEARBY`` of it."""
    below = answer - max(lo, answer - NEARBY)
    above = min(hi, answer + NEARBY) - answer
    return [k for k in range(N_OPTIONS) if k <= below and N_OPTIONS - 1 - k <= above]


def _number_options(
    rng: np.random.Generator, answer: int, rank: int, lo: int, hi: int
) -> tuple[str, ...]:
    """The answer followed by 3 different numbers in ``[lo, hi]`` near it, ``rank`` of them smaller."""
    lower = range(max(lo, answer - NEARBY), answer)
    upper = range(answer + 1, min(hi, answer + NEARBY) + 1)
    below = [lower[int(i)] for i in rng.choice(len(lower), size=rank, replace=False)]
    above = [upper[int(i)] for i in rng.choice(len(upper), N_OPTIONS - 1 - rank, replace=False)]
    return tuple(str(x) for x in (answer, *below, *above))


# --- compare ---------------------------------------------------------------------------------


def _compare_world(rng: np.random.Generator, control: int) -> _World:
    top, middle, bottom, odd = _sample(rng, NAMES, 4)
    adjectives = ADJECTIVES[_pick(rng, len(ADJECTIVES))]
    key = f"compare|{adjectives[0]}|{top}|{middle}|{bottom}"
    return _World("compare", key, (adjectives, top, middle, bottom, odd))


def _compare_question(rng: np.random.Generator, world: _World, control: int) -> _Question:
    (more, most, less, least), top, middle, bottom, odd = world.data
    if _flip(rng):
        chain = [f"{top} is {more} than {middle}.", f"{middle} is {more} than {bottom}."]
    else:
        chain = [f"{bottom} is {less} than {middle}.", f"{middle} is {less} than {top}."]
    chain_name = (top, middle, bottom)[_pick(rng, 3)]
    aside = ASIDES[_pick(rng, len(ASIDES))].format(e=odd, x=chain_name)
    sentences = _shuffled(rng, [*chain, aside])
    if control == 0:
        ask, answer, finish = f"Who is the {most}?", top, f"is the {most}"
    elif control == 1:
        ask, answer, finish = f"Who is the {least}?", bottom, f"is the {least}"
    else:
        ask, answer, finish = "Who is in the middle?", middle, "is in the middle"
    explanation = (
        f"{top} is {more} than {middle} and {middle} is {more} than {bottom}, so {answer} {finish}."
    )
    others = [name for name in (top, middle, bottom) if name != answer]
    return _Question(
        "compare",
        world.key,
        f"{' '.join(sentences)} {ask}",
        answer,
        explanation,
        (answer, *others, odd),
    )


# --- syllogism -------------------------------------------------------------------------------


def _syllogism_world(rng: np.random.Generator, control: int) -> _World:
    yes, two_steps = control % 2 == 0, control >= 2
    words = _sample(rng, NONCE_WORDS, 5 if two_steps else 4)
    (name,) = _sample(rng, NAMES, 1)
    a, b, c, d, *rest = words
    if two_steps:
        e = rest[0]
        premises = (
            f"All {a}s are {b}s.",
            f"All {b}s are {c}s." if yes else f"No {b}s are {c}s.",
            f"No {d}s are {e}s." if yes else f"All {d}s are {e}s.",
        )
        question = f"Is a {a} a {c}?"
        explanation = (
            f"All {a}s are {b}s and all {b}s are {c}s, so all {a}s are {c}s."
            if yes
            else f"All {a}s are {b}s and no {b}s are {c}s, so no {a}s are {c}s."
        )
    else:
        premises = (
            f"All {a}s are {b}s." if yes else f"No {a}s are {b}s.",
            f"No {c}s are {d}s." if yes else f"All {c}s are {d}s.",
            f"{name} is a {a}.",
        )
        question = f"Is {name} a {b}?"
        explanation = (
            f"All {a}s are {b}s and {name} is a {a}, so {name} is a {b}."
            if yes
            else f"No {a}s are {b}s and {name} is a {a}, so {name} is not a {b}."
        )
    key = "syllogism|" + "|".join(sorted(premises)) + "|" + question
    return _World("syllogism", key, (premises, question, "yes" if yes else "no", explanation))


def _syllogism_question(rng: np.random.Generator, world: _World, control: int) -> _Question:
    premises, question, answer, explanation = world.data
    text = " ".join(_shuffled(rng, list(premises))) + " " + question
    return _Question(
        "syllogism", world.key, text, answer, explanation, ("yes", "no"), shuffle=False
    )


# --- word_problem ----------------------------------------------------------------------------


def _word_problem_world(rng: np.random.Generator, control: int) -> _World:
    boy = _flip(rng)
    (name,) = _sample(rng, BOYS if boy else GIRLS, 1)
    thing = THINGS[_pick(rng, len(THINGS))]
    subtract = control == 1
    if subtract:  # the result is 1-19, something is left
        result = int(rng.integers(1, 20))
        start = int(rng.integers(result + 1, 21))
        change = start - result
    else:  # the result is 3-20 and at least 2 things were there at the start
        result = int(rng.integers(3, 21))
        start = int(rng.integers(2, result))
        change = result - start
    key = f"word_problem|{name}|{thing}|{'subtract' if subtract else 'add'}|{start}|{change}"
    return _World("word_problem", key, (name, boy, thing, subtract, start, change))


def _word_problem_question(rng: np.random.Generator, world: _World, control: int) -> _Question:
    name, boy, thing, subtract, start, change = world.data
    stories = SUBTRACT_STORIES if subtract else ADD_STORIES
    text = stories[_pick(rng, len(stories))].format(
        n=name,
        a=start,
        b=change,
        t=thing,
        s="He" if boy else "She",
        p="he" if boy else "she",
        o="him" if boy else "her",
    )
    if subtract:
        result, explanation = start - change, f"{start} minus {change} is {start - change}."
    else:
        result, explanation = start + change, f"{start} plus {change} is {start + change}."
    return _Question("word_problem", world.key, text, str(result), explanation)


# --- count -----------------------------------------------------------------------------------


def _count_world(rng: np.random.Generator, control: int) -> _World:
    group = COUNT_GROUPS[_pick(rng, len(COUNT_GROUPS))]
    target = group[_pick(rng, len(group))]
    others = [w for w in group if w != target]
    length = int(rng.integers(max(4, control + 1), 9))
    words = [target] * control + [others[_pick(rng, len(others))] for _ in range(length - control)]
    key = f"count|{target}|{','.join(sorted(words))}"
    return _World("count", key, (target, control, tuple(words)))


def _count_question(rng: np.random.Generator, world: _World, control: int) -> _Question:
    target, times, words = world.data
    listed = ", ".join(_shuffled(rng, list(words)))
    text = f"How many {target}s are in this list: {listed}?"
    explanation = f"The list has {times} {target}{'' if times == 1 else 's'}."
    return _Question("count", world.key, text, str(times), explanation)


_WORLDS = {
    "compare": _compare_world,
    "syllogism": _syllogism_world,
    "word_problem": _word_problem_world,
    "count": _count_world,
}
_QUESTIONS = {
    "compare": _compare_question,
    "syllogism": _syllogism_question,
    "word_problem": _word_problem_question,
    "count": _count_question,
}


class _Drawer:
    """Draws puzzles until one is accepted, from a budget of 50 world draws per requested puzzle."""

    def __init__(self, rng: np.random.Generator, n_items: int) -> None:
        self.rng = rng
        self.left = MAX_ATTEMPTS_PER_ITEM * n_items

    def draw(
        self, family: str, control: int, *, reserved: bool, avoid: Collection[str] = ()
    ) -> _Question:
        """A puzzle of ``family`` whose world and wording are (``reserved``) for the benchmark.

        Its world's key is not in ``avoid``. ``control`` fixes the balanced property of the family.
        """
        while self.left > 0:
            self.left -= 1
            world = _WORLDS[family](self.rng, control)
            if reserved_for_bench(world.key) != reserved or world.key in avoid:
                continue
            for _ in range(RENDER_TRIES):
                question = _QUESTIONS[family](self.rng, world, control)
                if reserved_for_bench(question.prompt) == reserved:
                    return question
        raise RuntimeError(f"reasoning: could not draw enough {family} puzzles within the budget")


def reasoning_train_docs(rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` documents of 3-8 puzzle blocks, ``Question: ...\\nAnswer: ...``, never a benchmark one.

    Each block is of a random family (and a random question, answer and so on within it); about
    40% are followed by a one-sentence explanation on the next line. Blocks are separated by a
    blank line, and no world occurs twice in a document. Worlds and prompts that are reserved for
    the benchmark are never used.
    """
    sizes = [int(s) for s in rng.integers(MIN_BLOCKS, MAX_BLOCKS + 1, size=n)]
    drawer = _Drawer(rng, sum(sizes))
    docs: list[TextDoc] = []
    for size in sizes:
        taken: set[str] = set()
        blocks: list[str] = []
        for _ in range(size):
            family = FAMILIES[_pick(rng, len(FAMILIES))]
            control = _pick(rng, N_CONTROLS[family])
            question = drawer.draw(family, control, reserved=False, avoid=taken)
            taken.add(question.world)
            block = f"{question.prompt} {question.answer}"
            if rng.random() < EXPLAIN_SHARE:
                block += f"\n{question.explanation}"
            blocks.append(block)
        docs.append(TextDoc("plain", "\n\n".join(blocks), topic=TOPIC))
    return docs


def reasoning_bench_items(rng: np.random.Generator, n: int = 200) -> list[MCItem]:
    """``n`` multiple-choice puzzles (category ``"reasoning"``), all in worlds reserved for the benchmark.

    Families take turns (``fam:compare``, ``fam:syllogism``, ``fam:word_problem``, ``fam:count``),
    so any prefix of the list is as balanced as it can be, and within a family the items step
    through the balanced property: compare asks for the top, the bottom and the middle in turn,
    syllogisms take turns between yes and no and between one and two steps, word problems between
    addition and subtraction, and count answers go through 0-5. No world repeats.

    Options: the 4 names of a compare puzzle (the 3 in the chain and the odd one out), ``["yes",
    "no"]``, or for numbers the answer and 3 different numbers within 5 of it that are possible
    answers (0-20 for word problems, 0-8 for counts). The answer's rank among the 4 numbers is
    spread as evenly as the answer allows, and option order is shuffled except for yes/no.
    """
    drawer = _Drawer(rng, n)
    seen: set[str] = set()
    ranks: dict[str, Counter[int]] = {family: Counter() for family in NUMBER_RANGE}
    items: list[MCItem] = []
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        control = (i // len(FAMILIES)) % N_CONTROLS[family]
        question = drawer.draw(family, control, reserved=True, avoid=seen)
        seen.add(question.world)
        if family in NUMBER_RANGE:
            lo, hi = NUMBER_RANGE[family]
            feasible = _feasible_ranks(int(question.answer), lo, hi)
            fewest = min(ranks[family][k] for k in feasible)
            least_used = [k for k in feasible if ranks[family][k] == fewest]
            rank = least_used[_pick(rng, len(least_used))]
            ranks[family][rank] += 1
            options = list(_number_options(rng, int(question.answer), rank, lo, hi))
        else:
            options = list(question.options)
        if question.shuffle:
            options = _shuffled(rng, options)
        items.append(
            MCItem(
                f"reasoning-{i:04d}",
                "reasoning",
                question.prompt,
                options,
                options.index(question.answer),
                (f"fam:{family}",),
            )
        )
    return items
