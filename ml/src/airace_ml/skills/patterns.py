"""Pattern lines as text: "what comes next" for training and benchmark items.

A line shows a few terms of a pattern and then the next one: ``Next: 2, 4, 6, 8, 10``. Five
families, each tagged ``fam:<name>`` in benchmarks:

- ``step``: counting up by 1-9 from 1-20 (2, 4, 6, 8)
- ``letters``: letters up by 1-3, staying within A-Z (A, C, E, G)
- ``cycle``: 2-3 colors or 2-3 animals over and over (red, blue, red, blue)
- ``double``: times 2 or times 3 from 1-9, up to 10,000 (3, 6, 12, 24)
- ``countdown``: counting down by 1-3 from up to 40, never below 0 (9, 8, 7, 6)

A benchmark item is the line without its last term, ``Next: 2, 4, 6, 8,``, which is also its
canonical key; 4-7 terms are shown. Items use only keys for which :func:`reserved_for_bench` is
true. A training line also contains the shorter prompts that are its first 4 to 7 terms followed
by a comma, and it is kept only if none of those is reserved, so no benchmark prompt occurs in
training text, not even inside a longer line.

The patterns of every family are listed up front instead of drawn and rejected: the spaces are
small (``double`` has only 67 prompts), so listing keeps a benchmark free of repeats, lets it
spread evenly over the families and makes the partition exact. Training lines choose a family
with probability proportional to the square root of its number of patterns, so the small
families are not repeated far more often than the large ones. Nothing here is a model output;
these are data generators.
"""

import itertools
import string
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache

import numpy as np

from airace_ml.skills.types import ExactItem, TextDoc, fair_quota, reserved_for_bench

FAMILIES = ("step", "letters", "cycle", "double", "countdown")
SHOWN_TERMS = (4, 5, 6, 7)  # how many terms a prompt shows
MIN_LINES, MAX_LINES = 3, 8  # lines in one training document
MAX_ATTEMPTS_PER_LINE = 50
MAX_DOUBLE = 10_000  # the largest term of a ``double`` pattern
TOPIC = "school"
# fmt: off
CYCLE_WORDS = {
    "colors": (
        "red", "blue", "green", "yellow", "pink", "purple",
        "orange", "black", "white", "brown", "gray", "gold",
    ),
    "animals": (
        "cat", "dog", "cow", "pig", "hen", "fox",
        "frog", "duck", "bear", "lion", "fish", "bird",
    ),
}
# fmt: on


@dataclass(frozen=True)
class _Pattern:
    """A pattern as shown (``shown`` terms) and the term that comes next."""

    family: str
    shown: tuple[str, ...]
    answer: str

    def prefix(self, terms: int) -> str:
        """The prompt that shows the first ``terms`` terms."""
        return "Next: " + ", ".join(self.shown[:terms]) + ","

    @property
    def key(self) -> str:
        """The canonical key, which is also the benchmark prompt."""
        return self.prefix(len(self.shown))

    @property
    def line(self) -> str:
        return f"{self.key} {self.answer}"


def _steps(n: int) -> Iterator[list[str]]:
    for start in range(1, 21):
        for step in range(1, 10):
            yield [str(start + i * step) for i in range(n)]


def _letters(n: int) -> Iterator[list[str]]:
    for step in (1, 2, 3):
        for start in range(26 - (n - 1) * step):
            yield [string.ascii_uppercase[start + i * step] for i in range(n)]


def _cycles(n: int) -> Iterator[list[str]]:
    for words in CYCLE_WORDS.values():
        for size in (2, 3):
            for cycle in itertools.permutations(words, size):
                yield [cycle[i % size] for i in range(n)]


def _doubles(n: int) -> Iterator[list[str]]:
    for start in range(1, 10):
        for factor in (2, 3):
            terms = [start * factor**i for i in range(n)]
            if terms[-1] <= MAX_DOUBLE:
                yield [str(t) for t in terms]


def _countdowns(n: int) -> Iterator[list[str]]:
    for step in (1, 2, 3):
        for start in range((n - 1) * step, 41):  # the n-th term is never below 0
            yield [str(start - i * step) for i in range(n)]


_SEQUENCES: dict[str, Callable[[int], Iterator[list[str]]]] = {
    "step": _steps,
    "letters": _letters,
    "cycle": _cycles,
    "double": _doubles,
    "countdown": _countdowns,
}


@cache
def _space() -> dict[str, tuple[_Pattern, ...]]:
    """Every pattern of every family: each sequence of ``k + 1`` terms for ``k`` in ``SHOWN_TERMS``."""
    return {
        family: tuple(
            _Pattern(family, tuple(terms[:-1]), terms[-1])
            for k in SHOWN_TERMS
            for terms in _SEQUENCES[family](k + 1)
        )
        for family in FAMILIES
    }


def _blocked(pattern: _Pattern) -> bool:
    """Whether a line of this pattern would show a prompt that belongs to the benchmark."""
    return any(
        reserved_for_bench(pattern.prefix(m)) for m in SHOWN_TERMS if m <= len(pattern.shown)
    )


@cache
def _train_pools() -> dict[str, tuple[_Pattern, ...]]:
    return {f: tuple(p for p in ps if not _blocked(p)) for f, ps in _space().items()}


@cache
def _bench_pools() -> dict[str, tuple[_Pattern, ...]]:
    return {f: tuple(p for p in ps if reserved_for_bench(p.key)) for f, ps in _space().items()}


def pattern_train_docs(rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` documents of 3-8 different lines like ``Next: 2, 4, 6, 8, 10``, never a benchmark one.

    Each line is of a random family (more often the larger ones, in proportion to the square root
    of their number of patterns) and then a random pattern of it (4-7 terms shown, then the next
    one), among the patterns that show no benchmark prompt.
    """
    pools = _train_pools()
    weights = np.sqrt([len(pools[family]) for family in FAMILIES])
    weights /= weights.sum()
    docs: list[TextDoc] = []
    for _ in range(n):
        wanted = int(rng.integers(MIN_LINES, MAX_LINES + 1))
        lines: list[str] = []
        for _ in range(MAX_ATTEMPTS_PER_LINE * wanted):
            if len(lines) == wanted:
                break
            pool = pools[FAMILIES[int(rng.choice(len(FAMILIES), p=weights))]]
            line = pool[int(rng.integers(len(pool)))].line
            if line not in lines:
                lines.append(line)
        if len(lines) < wanted:
            raise RuntimeError(f"pattern: could not draw {wanted} different lines")
        docs.append(TextDoc("plain", "\n".join(lines), topic=TOPIC))
    return docs


def pattern_bench_items(rng: np.random.Generator, n: int = 150) -> list[ExactItem]:
    """``n`` free-answer patterns (category ``"pattern"``), every key reserved for the benchmark.

    The prompt is the shown terms, ``Next: 2, 4, 6, 8,``, and the answer is the next term
    (``extract="first_item"``). Items are spread over the families as evenly as each family's
    reserved prompts allow (``double`` has only a few), none repeats, and the order is shuffled.
    Raises ``ValueError`` if ``n`` is more than the number of reserved prompts.
    """
    pools = _bench_pools()
    quota = fair_quota({f: len(p) for f, p in pools.items()}, n)
    chosen: list[_Pattern] = []
    for family in FAMILIES:
        pool = pools[family]
        chosen += [pool[int(i)] for i in rng.choice(len(pool), size=quota[family], replace=False)]
    chosen = [chosen[int(i)] for i in rng.permutation(len(chosen))]
    return [
        ExactItem(
            f"pattern-{i:04d}",
            "pattern",
            p.key,
            [p.answer],
            (f"fam:{p.family}",),
            extract="first_item",
        )
        for i, p in enumerate(chosen)
    ]
