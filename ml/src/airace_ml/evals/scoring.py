"""How benchmark answers become scores: pure arithmetic on what a model already said or scored.

Scores are measurements (R5). A category's score is its accuracy normalized so that 0 is what
answering at random would get and 100 is every item right (:func:`normalize`). Degenerate model
output (empty replies, only special tokens, endless repetition, NaN log-probabilities from a
diverged model) always yields a finite score, never NaN or a division by zero.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal

from airace_ml.infer.lm import ContinuationScore

MIN_TAG_ITEMS = 10
"""A tag needs at least this many items in a category before :meth:`BenchReport.tag_breakdown`
reports it, so a handful of items never becomes a discovery."""

_QUOTES = "\"'‘’“”"
_SENTENCE_END = ".!?;:,"
_SPACES = re.compile(r"\s+")
_FIRST_ITEM_END = re.compile(r"[,\n]")


def normalize(acc: float, chance: float) -> float:
    """``clip((acc - chance) / (1 - chance), 0, 1) * 100``: 0 at chance level, 100 when perfect.

    A chance of 1 or more leaves no room above chance and scores 0, as does a NaN.
    """
    if not chance < 1:
        return 0.0
    value = (acc - chance) / (1 - chance)
    if math.isnan(value):
        return 0.0
    return min(max(value, 0.0), 1.0) * 100


# -- free answers ----------------------------------------------------------------------------


def extract_answer(reply: str, extract: Literal["first_item", "first_line"]) -> str:
    """The part of ``reply`` that is the answer: up to the first ``,`` or new line for
    ``first_item`` (the next term of a list), up to the first new line for ``first_line``."""
    if extract == "first_item":
        return _FIRST_ITEM_END.split(reply, maxsplit=1)[0]
    if extract == "first_line":
        return reply.split("\n", 1)[0]
    raise ValueError(f"unknown extract mode {extract!r}")


def normalize_answer(text: str) -> str:
    """``text`` as free answers are compared: no surrounding spaces or quotes, no trailing
    sentence punctuation (``.!?;:,``), single spaces inside, case-folded.

    Nothing else is removed: a minus sign, brackets and inner commas are part of an answer
    (``-3`` is not ``3``; ``[1, 2]`` is not ``1, 2``).
    """
    previous = None
    while text != previous:  # until stable: '"Paris".' loses the period, then the quotes
        previous = text
        text = text.strip().strip(_QUOTES).rstrip(_SENTENCE_END)
    return _SPACES.sub(" ", text).casefold()


def exact_match(
    reply: str, answers: Sequence[str], extract: Literal["first_item", "first_line"]
) -> bool:
    """The extracted reply equals one of ``answers``, both normalized by :func:`normalize_answer`.

    An empty answer never matches, so saying nothing never scores.
    """
    said = normalize_answer(extract_answer(reply, extract))
    return bool(said) and said in {normalize_answer(answer) for answer in answers}


# -- log-probability comparisons -------------------------------------------------------------


def _total(score: ContinuationScore) -> float:
    """The summed log-probability, with NaN (a broken model) counted as impossible."""
    return -math.inf if math.isnan(score.sum_logprob) else score.sum_logprob


def mean_logprob(score: ContinuationScore) -> float:
    """Log-probability per token; ``-inf`` for an empty continuation or a NaN."""
    if score.n_tokens <= 0:
        return -math.inf
    return _total(score) / score.n_tokens


def mc_choice(option_scores: Sequence[ContinuationScore]) -> int:
    """The option with the highest mean log-probability per token (the first one on ties)."""
    means = [mean_logprob(score) for score in option_scores]
    return means.index(max(means))


def pair_correct(good: ContinuationScore, bad: ContinuationScore) -> bool:
    """The good text has the strictly higher total log-probability (a tie is not a win)."""
    return _total(good) > _total(bad)


# -- results ---------------------------------------------------------------------------------


@dataclass
class ItemResult:
    """One scored item: 1 if right, else 0 (a consistency group is right only when every one of
    its paraphrases is).

    ``output`` is what the model answered: its reply for a free answer, the option it chose for
    multiple choice, ``None`` when there is nothing to show (pairs, consistency groups).
    """

    item_id: str
    category: str
    score: float
    tags: tuple[str, ...]
    output: str | None = None


@dataclass
class CategoryScore:
    """``score`` is 0-100 above chance; ``raw`` the mean item score (0-1); ``n`` how many
    items (groups, for consistency) were scored."""

    score: float
    raw: float
    n: int


def category_score(item_scores: Sequence[float], chances: Sequence[float]) -> CategoryScore:
    """Mean item score, normalized against the mean chance of the same items."""
    raw = fmean(item_scores)
    return CategoryScore(normalize(raw, fmean(chances)), raw, len(item_scores))


@dataclass
class BenchReport:
    """Everything one benchmark run measured.

    ``missing`` lists the requested categories that could not be measured (creativity without a
    judge and novelty index, or a category with no items); they are not in ``scores`` and do not
    count towards ``overall``, the mean of the category scores (0 if there are none).
    """

    suite: str
    scores: dict[str, CategoryScore]
    overall: float
    missing: list[str]
    items: list[ItemResult]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Plain dicts, lists, strings and numbers only (``json.dumps`` safe)."""
        return {
            "suite": self.suite,
            "scores": {
                category: {"score": s.score, "raw": s.raw, "n": s.n}
                for category, s in self.scores.items()
            },
            "overall": self.overall,
            "missing": list(self.missing),
            "items": [
                {
                    "item_id": r.item_id,
                    "category": r.category,
                    "score": r.score,
                    "tags": list(r.tags),
                    "output": r.output,
                }
                for r in self.items
            ],
            "seconds": self.seconds,
        }

    def tag_breakdown(self) -> dict[str, tuple[float, int]]:
        """``"<category>/<tag>"`` -> (mean item score x 100, item count), for every tag with at
        least :data:`MIN_TAG_ITEMS` items in its category.

        Tags are counted per category: the same tag in two categories measures two different
        things (``fam:double`` is a number sequence in ``pattern`` and a function in ``coding``;
        a ``topic:`` tag counts single questions in ``knowledge`` but whole paraphrase groups in
        ``consistency``).
        """
        totals: dict[str, list[float]] = {}
        for result in self.items:
            for tag in dict.fromkeys(result.tags):
                totals.setdefault(f"{result.category}/{tag}", []).append(result.score)
        return {
            key: (100 * fmean(scores), len(scores))
            for key, scores in totals.items()
            if len(scores) >= MIN_TAG_ITEMS
        }
