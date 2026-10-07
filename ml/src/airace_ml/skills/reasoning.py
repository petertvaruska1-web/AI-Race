"""Reasoning puzzles as text: training blocks and benchmark items.

Four families of tiny puzzles, all in the frame ``Question: ...`` / ``Answer: ...``:

- ``compare``: three people in a chain ("Tom is taller than Ben. Ben is taller than Sam.") and a
  question about the top or the bottom of it.
- ``syllogism``: made-up category words ("All blicks are fenks. Tom is a blick. Is Tom a fenk?"),
  answered yes or no.
- ``word_problem``: one addition or subtraction story, with numbers up to 20.
- ``count``: how many times a word is in a short list.

The canonical key of a puzzle is its benchmark prompt, ``Question: ...\\nAnswer:``. Benchmark items
use only keys for which :func:`reserved_for_bench` is true and training blocks only keys for which
it is false, so training text never contains a benchmark question. Puzzles are drawn at random and
accepted or rejected by that rule; a draw budget of 50 per requested item keeps the search finite.
Nothing here is a model output; these are data generators.
"""

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from airace_ml.skills.types import MCItem, TextDoc, reserved_for_bench

FAMILIES = ("compare", "syllogism", "word_problem", "count")
TOPIC = "school"
MAX_ATTEMPTS_PER_ITEM = 50
MIN_BLOCKS, MAX_BLOCKS = 3, 8  # puzzles in one training document
EXPLAIN_SHARE = 0.4  # share of training blocks followed by a one-sentence explanation
N_OPTIONS = 4

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

THINGS = (
    "apples", "pears", "cookies", "balls", "stickers", "marbles", "books", "coins", "cards",
    "shells", "stones", "pens", "hats", "kites", "toys", "cups", "eggs", "cakes",
)
# fmt: on
COUNT_GROUPS = (
    ("cat", "dog", "cow", "pig", "hen", "bird", "duck", "frog", "lion", "bear", "goat", "horse"),
    ("apple", "pear", "plum", "lemon", "grape", "peach", "mango", "melon", "banana", "cherry"),
)

# (comparative, superlative, opposite comparative, opposite superlative)
ADJECTIVES = (
    ("taller", "tallest", "shorter", "shortest"),
    ("older", "oldest", "younger", "youngest"),
    ("faster", "fastest", "slower", "slowest"),
    ("heavier", "heaviest", "lighter", "lightest"),
)

ADD_STORIES = (
    "{n} has {a} {t}. {s} gets {b} more. How many {t} does {n} have now?",
    "{n} has {a} {t}. A friend gives {o} {b} more. How many {t} does {n} have now?",
    "{n} has {a} {t} and finds {b} more. How many {t} does {n} have now?",
)
SUBTRACT_STORIES = (
    "{n} has {a} {t}. {s} gives {b} away. How many {t} does {n} have now?",
    "{n} has {a} {t}. {s} loses {b}. How many {t} does {n} have now?",
    "{n} has {a} {t} and gives {b} to a friend. How many {t} does {n} have now?",
)


@dataclass(frozen=True)
class _Question:
    """One puzzle: the question text, its answer and the options a benchmark offers for it."""

    family: str
    text: str
    answer: str
    options: tuple[str, ...]  # contains the answer; all different
    explanation: str
    shuffle: bool = True  # whether a benchmark mixes the order of the options

    @property
    def prompt(self) -> str:
        """The canonical key: what the model is shown before it answers."""
        return f"Question: {self.text}\nAnswer:"


def _pick(rng: np.random.Generator, n: int) -> int:
    return int(rng.integers(n))


def _flip(rng: np.random.Generator) -> bool:
    return bool(rng.integers(2))


def _sample(rng: np.random.Generator, pool: tuple[str, ...], k: int) -> list[str]:
    """``k`` different members of ``pool``."""
    return [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]


def _number_options(rng: np.random.Generator, answer: int) -> tuple[str, ...]:
    """The answer followed by 3 different numbers within 3 of it (never negative)."""
    nearby = [answer + d for d in (-3, -2, -1, 1, 2, 3) if answer + d >= 0]
    wrong = [nearby[int(i)] for i in rng.choice(len(nearby), size=N_OPTIONS - 1, replace=False)]
    return tuple(str(x) for x in (answer, *wrong))


def _compare(rng: np.random.Generator) -> _Question:
    top, middle, bottom, stranger = _sample(rng, NAMES, 4)  # the stranger is only ever an option
    more, most, less, least = ADJECTIVES[_pick(rng, len(ADJECTIVES))]
    if _flip(rng):
        premises = [f"{top} is {more} than {middle}.", f"{middle} is {more} than {bottom}."]
    else:
        premises = [f"{bottom} is {less} than {middle}.", f"{middle} is {less} than {top}."]
    if _flip(rng):
        premises.reverse()
    superlative, answer, others = (most, top, (middle, bottom))
    if _flip(rng):
        superlative, answer, others = (least, bottom, (top, middle))
    text = f"{' '.join(premises)} Who is the {superlative}?"
    explanation = (
        f"{top} is {more} than {middle} and {middle} is {more} than {bottom}, "
        f"so {answer} is the {superlative}."
    )
    return _Question("compare", text, answer, (answer, *others, stranger), explanation)


def _syllogism(rng: np.random.Generator) -> _Question:
    a, b, c = _sample(rng, NONCE_WORDS, 3)
    (name,) = _sample(rng, NAMES, 1)
    yes, chain = _flip(rng), _flip(rng)
    if chain:
        second = f"All {b}s are {c}s." if yes else f"No {b}s are {c}s."
        premises = [f"All {a}s are {b}s.", second]
        question = f"Is a {a} a {c}?"
        explanation = (
            f"All {a}s are {b}s and all {b}s are {c}s, so all {a}s are {c}s."
            if yes
            else f"All {a}s are {b}s and no {b}s are {c}s, so no {a}s are {c}s."
        )
    else:
        rule = f"All {a}s are {b}s." if yes else f"No {a}s are {b}s."
        premises = [rule, f"{name} is a {a}."]
        question = f"Is {name} a {b}?"
        explanation = (
            f"All {a}s are {b}s and {name} is a {a}, so {name} is a {b}."
            if yes
            else f"No {a}s are {b}s and {name} is a {a}, so {name} is not a {b}."
        )
    if _flip(rng):
        premises.reverse()
    text = f"{' '.join(premises)} {question}"
    return _Question(
        "syllogism", text, "yes" if yes else "no", ("yes", "no"), explanation, shuffle=False
    )


def _word_problem(rng: np.random.Generator) -> _Question:
    boy = _flip(rng)
    (name,) = _sample(rng, BOYS if boy else GIRLS, 1)
    thing = THINGS[_pick(rng, len(THINGS))]
    if _flip(rng):  # subtraction: something is left
        start = int(rng.integers(2, 21))
        change = int(rng.integers(1, start))
        result, stories = start - change, SUBTRACT_STORIES
        explanation = f"{start} minus {change} is {result}."
    else:  # addition: the total stays within 20
        start = int(rng.integers(2, 20))
        change = int(rng.integers(1, 21 - start))
        result, stories = start + change, ADD_STORIES
        explanation = f"{start} plus {change} is {result}."
    text = stories[_pick(rng, len(stories))].format(
        n=name, a=start, b=change, t=thing, s="He" if boy else "She", o="him" if boy else "her"
    )
    options = _number_options(rng, result)
    return _Question("word_problem", text, options[0], options, explanation)


def _count(rng: np.random.Generator) -> _Question:
    group = COUNT_GROUPS[_pick(rng, len(COUNT_GROUPS))]
    target = group[_pick(rng, len(group))]
    others = [w for w in group if w != target]
    length = int(rng.integers(4, 9))
    times = int(rng.integers(0, min(5, length - 1) + 1))
    words = [target] * times + [others[_pick(rng, len(others))] for _ in range(length - times)]
    words = [words[int(i)] for i in rng.permutation(length)]
    text = f"How many {target}s are in this list: {', '.join(words)}?"
    explanation = f"The list has {times} {target}{'' if times == 1 else 's'}."
    options = _number_options(rng, times)
    return _Question("count", text, options[0], options, explanation)


_DRAWERS = {
    "compare": _compare,
    "syllogism": _syllogism,
    "word_problem": _word_problem,
    "count": _count,
}


class _Drawer:
    """Draws puzzles until one is accepted, from a budget of 50 draws per requested puzzle."""

    def __init__(self, rng: np.random.Generator, n_items: int) -> None:
        self.rng = rng
        self.left = MAX_ATTEMPTS_PER_ITEM * n_items

    def draw(
        self,
        family: str,
        *,
        reserved: bool,
        avoid: Collection[str] = (),
        answer: str | None = None,
    ) -> _Question:
        """A puzzle of ``family`` whose key is (``reserved``) for the benchmark, not in ``avoid``."""
        while self.left > 0:
            self.left -= 1
            question = _DRAWERS[family](self.rng)
            if (
                reserved_for_bench(question.prompt) == reserved
                and question.prompt not in avoid
                and (answer is None or question.answer == answer)
            ):
                return question
        raise RuntimeError(f"reasoning: could not draw enough {family} puzzles within the budget")


def reasoning_train_docs(rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` documents of 3-8 puzzle blocks, ``Question: ...\\nAnswer: ...``, never a benchmark one.

    Each block is of a random family; about 40% are followed by a one-sentence explanation on
    the next line. Blocks are separated by a blank line.
    """
    sizes = [int(s) for s in rng.integers(MIN_BLOCKS, MAX_BLOCKS + 1, size=n)]
    drawer = _Drawer(rng, sum(sizes))
    docs: list[TextDoc] = []
    for size in sizes:
        taken: set[str] = set()
        blocks: list[str] = []
        for _ in range(size):
            question = drawer.draw(FAMILIES[_pick(rng, len(FAMILIES))], reserved=False, avoid=taken)
            taken.add(question.prompt)
            block = f"{question.prompt} {question.answer}"
            if rng.random() < EXPLAIN_SHARE:
                block += f"\n{question.explanation}"
            blocks.append(block)
        docs.append(TextDoc("plain", "\n\n".join(blocks), topic=TOPIC))
    return docs


def reasoning_bench_items(rng: np.random.Generator, n: int = 200) -> list[MCItem]:
    """``n`` multiple-choice puzzles (category ``"reasoning"``), every key reserved for the benchmark.

    Families take turns (``fam:compare``, ``fam:syllogism``, ``fam:word_problem``, ``fam:count``), so
    any prefix of the list is as balanced as it can be, and syllogisms alternate between the
    answers yes and no. Compare and syllogism options are the names or ``["yes", "no"]``; number
    options are the answer and 3 different numbers within 3 of it. No prompt repeats.
    """
    drawer = _Drawer(rng, n)
    seen: set[str] = set()
    items: list[MCItem] = []
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        answer = ("yes", "no")[(i // len(FAMILIES)) % 2] if family == "syllogism" else None
        question = drawer.draw(family, reserved=True, avoid=seen, answer=answer)
        seen.add(question.prompt)
        options = list(question.options)
        if question.shuffle:
            options = [options[int(j)] for j in rng.permutation(len(options))]
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
