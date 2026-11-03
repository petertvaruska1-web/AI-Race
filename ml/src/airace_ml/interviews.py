"""Interview prompt cards and how published interview answers are scored (game-layer design §10).

A player interviews their AI in front of the press with prompt cards or their own questions,
then publishes the answers they choose. Half the cards have a right answer, drawn from the skill
generators on their own ``skill_rng(<category>, "interviews")`` stream and never a benchmark
prompt; an answer to one is scored by the card's check. The open cards, and any question the
player types, have no right answer: their answers are scored for how fluently they read.
"""

import dataclasses
from functools import cache

from airace_ml.evals.creativity import _coherence, repetitiveness
from airace_ml.evals.judge import Judge, _has_reply_words
from airace_ml.evals.suite import build_suite, score_reply
from airace_ml.skills.code import code_bench_items
from airace_ml.skills.facts import knowledge_bench_items
from airace_ml.skills.instructions import instruction_bench_items
from airace_ml.skills.kb import load_kb
from airace_ml.skills.patterns import pattern_bench_items
from airace_ml.skills.types import CheckItem, ExactItem, skill_rng

CHECKED_CATEGORIES = ("knowledge", "pattern", "instruction", "coding")
CHECKED_PER_CATEGORY = 3
OPEN_PROMPTS = (
    "What would you like people to know about you?",
    "Tell us a joke.",
    "Describe your perfect day.",
    "What is your favorite word, and why?",
    "Write a short poem about the sea.",
    "Explain rain to a five-year-old.",
    "What makes a good friend?",
    "Tell us a short story about a robot.",
    "What do you think about cats?",
    "Give one tip for learning faster.",
    "What would you do with a million dollars?",
    "Describe the color blue.",
)


@dataclasses.dataclass(frozen=True)
class InterviewCard:
    id: str
    prompt: str
    checked: bool


def _generated(category: str) -> list:
    rng = skill_rng(category, "interviews")
    if category == "knowledge":
        return knowledge_bench_items(load_kb(), rng)
    if category == "instruction":
        return instruction_bench_items(load_kb(), rng)
    if category == "coding":
        return code_bench_items(rng)
    if category == "pattern":
        return pattern_bench_items(rng)
    raise ValueError(f"no interview cards for {category!r}")


@cache
def _checked_items() -> dict[str, ExactItem | CheckItem]:
    """The checked cards' items by card id: the first questions a chat can ask, in each
    category, that are not benchmark prompts."""
    bench = {getattr(i, "prompt", None) for items in build_suite().items.values() for i in items}
    cards: dict[str, ExactItem | CheckItem] = {}
    for category in CHECKED_CATEGORIES:
        asked = [
            item
            for item in _generated(category)
            if isinstance(item, (ExactItem, CheckItem)) and item.prompt not in bench
        ]
        for k, item in enumerate(asked[:CHECKED_PER_CATEGORY], start=1):
            card_id = f"interview-{category}-{k}"
            cards[card_id] = dataclasses.replace(item, id=card_id)
    return cards


def interview_cards() -> list[InterviewCard]:
    """The 24 prompt cards: 12 with a right answer, then the 12 open ones."""
    checked = [InterviewCard(i, item.prompt, True) for i, item in _checked_items().items()]
    open_cards = [
        InterviewCard(f"interview-open-{k:02d}", prompt, False)
        for k, prompt in enumerate(OPEN_PROMPTS, start=1)
    ]
    return checked + open_cards


def score_answer(card_id: str | None, reply: str, judge: Judge | None) -> tuple[float, bool]:
    """``(score 0-1, checked)`` for one published answer.

    A checked card's answer is scored by its check. Anything else (an open card, the player's
    own question, an unknown card) is scored for fluency: with the judge, how naturally it reads
    times how little it repeats itself; without one, 1 for a reply with real words and 0 for one
    without.
    """
    item = _checked_items().get(card_id or "")
    if item is not None:
        return float(score_reply(item, reply)), True
    if not _has_reply_words(reply):
        return 0.0, False
    if judge is None:
        return 1.0, False
    nll = judge.nll_per_token([judge.tok.encode(reply.strip())])[0]
    return float(_coherence(nll, judge.calibration) * (1 - repetitiveness(reply))), False
