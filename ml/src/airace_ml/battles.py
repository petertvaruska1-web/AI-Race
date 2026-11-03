"""Fresh challenges for battles between two AIs (game-layer design §10).

A battle asks both AIs the same few challenges in one skill. Checked skills draw from the same
generators as the benchmark, on their own ``skill_rng(<skill>, "battles")`` stream and only from the
benchmark's held-out keys, so no model was trained on them; any challenge whose prompt the
benchmark, the customer questions or the creativity eval also ask is left out, so a battle never
repeats a scored item. Creativity asks for a short story about a subject and a situation drawn from
word lists, scored like one story of the creativity eval (judge and novelty index).
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from airace_ml.evals.creativity import STORY_PROMPTS, score_story
from airace_ml.evals.scoring import extract_answer, normalize_answer
from airace_ml.evals.suite import build_suite, score_reply
from airace_ml.feedback import GROUPS, customer_pool
from airace_ml.skills.checkers import CHECKERS
from airace_ml.skills.code import code_bench_items
from airace_ml.skills.facts import knowledge_bench_items
from airace_ml.skills.instructions import instruction_bench_items
from airace_ml.skills.kb import load_kb
from airace_ml.skills.patterns import pattern_bench_items
from airace_ml.skills.types import CheckItem, ExactItem, skill_rng

if TYPE_CHECKING:
    from airace_ml.evals.judge import Judge
    from airace_ml.evals.novelty import NoveltyIndex
    from airace_ml.tokenizer import Tok

BATTLE_SKILLS = ("knowledge", "coding", "instruction", "pattern", "creativity")


@dataclass(frozen=True)
class StoryPrompt:
    id: str
    prompt: str


type Challenge = ExactItem | CheckItem | StoryPrompt

_SUBJECTS = (
    "a fox",
    "a shy robot",
    "an old lighthouse keeper",
    "a small dragon",
    "a lost puppy",
    "a brave mouse",
    "a sleepy bear",
    "a young painter",
    "a talking kettle",
    "a curious girl",
    "a grumpy goat",
    "a paper boat",
    "a tiny snail",
    "a busy bee",
    "a friendly ghost",
    "a clever crow",
    "a quiet boy",
    "a lonely moon",
    "a happy whale",
    "a forgetful wizard",
)
_SITUATIONS = (
    "who loses the way home",
    "who finds a door in a tree",
    "who wants to see the ocean",
    "who is afraid of the rain",
    "who makes a new friend",
    "who learns to whistle",
    "who wins a race by accident",
    "who finds a map in a bottle",
    "who misses the last train",
    "who builds a tiny house",
    "who hears a strange song",
    "who wakes up in the snow",
    "who tries to catch a star",
    "who plants a magic bean",
    "who loses a favorite hat",
    "who travels to the city",
    "who keeps a big secret",
    "who bakes a cake for the king",
    "who follows a rainbow",
    "who cannot fall asleep",
)


@cache
def _taken_prompts() -> frozenset[str]:
    """Prompts a battle must not ask: the benchmark's, the customers' and the stories'."""
    prompts = {
        p
        for items in build_suite().items.values()
        for item in items
        if (p := getattr(item, "prompt", None))
    }
    prompts |= {q.prompt for group in GROUPS for q in customer_pool(group)}
    return frozenset(prompts | set(STORY_PROMPTS))


@cache
def _pool(skill: str) -> tuple[Challenge, ...]:
    if skill not in BATTLE_SKILLS:
        raise ValueError(f"no battles in {skill!r}; expected one of {BATTLE_SKILLS}")
    rng = skill_rng(skill, "battles")
    if skill == "creativity":
        order = rng.permutation(len(_SUBJECTS) * len(_SITUATIONS))
        prompts = [
            f"Write a short story about {_SUBJECTS[k // len(_SITUATIONS)]} "
            f"{_SITUATIONS[k % len(_SITUATIONS)]}."
            for k in order
        ]
        return tuple(StoryPrompt(f"creativity-battle-{k:04d}", p) for k, p in enumerate(prompts))
    if skill == "knowledge":
        items = knowledge_bench_items(load_kb(), rng)
    elif skill == "instruction":
        items = instruction_bench_items(load_kb(), rng)
    elif skill == "coding":
        items = code_bench_items(rng)
    else:
        items = pattern_bench_items(rng)
    taken = _taken_prompts()
    asked = [
        item
        for item in items
        if isinstance(item, (ExactItem, CheckItem)) and item.prompt not in taken
    ]
    return tuple(
        dataclasses.replace(item, id=f"{skill}-battle-{k:04d}") for k, item in enumerate(asked)
    )


def battle_challenges(skill: str, n: int, seed: int) -> list[Challenge]:
    """``n`` different challenges in ``skill``; the same ones for the same seed."""
    pool = _pool(skill)  # refuses an unknown skill
    rng = np.random.default_rng([seed, BATTLE_SKILLS.index(skill)])
    return [pool[int(k)] for k in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]


def battle_prompt(item: Challenge) -> tuple[str, bool, int, float]:
    """How a battle asks ``item``: (prompt, as a chat?, reply length, temperature).

    A checked challenge is asked exactly as the benchmark asks its kind (its own chat flag and
    reply length, greedily), so each AI shows what it can really do; a story as the creativity
    eval asks one.
    """
    if isinstance(item, StoryPrompt):
        return item.prompt, True, 96, 0.9
    return item.prompt, item.chat, item.max_new_tokens, 0.0


def expected_answer(item: Challenge) -> str | None:
    """The right answer a battle shows under the two replies (none for a story)."""
    if isinstance(item, ExactItem):
        return item.answers[0]
    if isinstance(item, CheckItem):
        return item.reference
    return None


_WORD = re.compile(r"\w+")


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _overlap(said: str, right: str) -> float:
    """The F1 of the words two texts share (case-folded, punctuation ignored)."""
    a, b = Counter(_WORD.findall(said.casefold())), Counter(_WORD.findall(right.casefold()))
    common = sum((a & b).values())
    if common == 0:
        return 0.0
    precision, recall = common / sum(a.values()), common / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def _closeness_to(said: str, right: str) -> float:
    a, b = _number(said), _number(right)
    if a is not None and b is not None:
        return max(0.0, 1.0 - abs(a - b) / max(1.0, abs(b)))
    return _overlap(said, right)


def _tests_passed(item: CheckItem, reply: str) -> float:
    """The share of a function challenge's tests the reply passes (other checks have no parts)."""
    if item.check != "minipy_function_tests":
        return 0.0
    tests = item.check_args["tests"]
    passed = sum(
        CHECKERS["minipy_function_tests"](reply, {**item.check_args, "tests": [test]})
        for test in tests
    )
    return passed / len(tests)


def closeness(item: ExactItem | CheckItem, reply: str) -> float:
    """How near a reply that fails its check came to the right answer (0-1).

    The best of: number distance (both answers numbers), the words it shares with the right
    answer or the reference reply, and the share of a function's tests it passes. It measures the
    reply against the answer; the reply itself is never changed (R1).
    """
    if isinstance(item, ExactItem):
        said = normalize_answer(extract_answer(reply, item.extract))
        if not said:
            return 0.0
        return max(_closeness_to(said, normalize_answer(answer)) for answer in item.answers)
    if not reply.strip():
        return 0.0
    return max(_overlap(reply, item.reference), _tests_passed(item, reply))


def score_challenge(
    item: Challenge,
    reply: str,
    *,
    tok: Tok | None = None,
    judge: Judge | None = None,
    novelty: NoveltyIndex | None = None,
) -> float:
    """How well ``reply`` meets the challenge (0-1).

    A story: the judge. A checked challenge: 1 if it passes its check, otherwise half its
    :func:`closeness`, so of two wrong answers the nearer one wins the round.
    """
    if isinstance(item, StoryPrompt):
        if tok is None or judge is None or novelty is None:
            raise ValueError("creativity needs the judge and the novelty index")
        return score_story(reply, item.prompt, tok, judge, novelty)
    if score_reply(item, reply) == 1.0:
        return 1.0
    return 0.5 * closeness(item, reply)
