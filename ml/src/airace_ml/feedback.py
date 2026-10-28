"""Customer questions for the game's weekly feedback (game-layer design §7).

Each customer group asks questions from the skills it cares about. They come from the same
generators as the benchmark, on their own ``skill_rng(<category>, "customers")`` stream and only
from the benchmark's held-out keys, so no model was trained on them; every one has an automatic
check (an accepted answer or a checker), so a reply can be scored. Reasoning generates only
multiple-choice items, which a chat cannot ask, so groups that care about reasoning also draw on
another skill.
"""

import dataclasses
from functools import cache

import numpy as np

from airace_ml.skills.code import code_bench_items
from airace_ml.skills.facts import knowledge_bench_items
from airace_ml.skills.instructions import instruction_bench_items
from airace_ml.skills.kb import load_kb
from airace_ml.skills.patterns import pattern_bench_items
from airace_ml.skills.reasoning import reasoning_bench_items
from airace_ml.skills.types import CheckItem, ExactItem, in_turn, skill_rng

type CustomerQuestion = ExactItem | CheckItem

GROUPS = ("students", "creators", "developers", "businesses", "everyday", "researchers")
GROUP_CATEGORIES: dict[str, tuple[str, ...]] = {
    "students": ("knowledge", "reasoning"),
    "creators": ("instruction",),
    "developers": ("coding",),
    "businesses": ("instruction", "knowledge"),
    "everyday": ("knowledge", "instruction"),
    "researchers": ("reasoning", "pattern"),
}


@cache
def _category_questions(category: str) -> tuple[CustomerQuestion, ...]:
    """A category's questions that a chat can ask, renamed apart from the benchmark's items."""
    rng = skill_rng(category, "customers")
    if category == "knowledge":
        items = knowledge_bench_items(load_kb(), rng)
    elif category == "instruction":
        items = instruction_bench_items(load_kb(), rng)
    elif category == "coding":
        items = code_bench_items(rng)
    elif category == "pattern":
        items = pattern_bench_items(rng)
    elif category == "reasoning":
        items = reasoning_bench_items(rng)
    else:
        raise ValueError(f"no customer questions for {category!r}")
    asked = [item for item in items if isinstance(item, (ExactItem, CheckItem))]
    return tuple(
        dataclasses.replace(item, id=f"{category}-customer-{k:04d}") for k, item in enumerate(asked)
    )


@cache
def _pool(group: str) -> tuple[CustomerQuestion, ...]:
    if group not in GROUP_CATEGORIES:
        raise ValueError(f"unknown customer group {group!r}")
    return tuple(in_turn([_category_questions(c) for c in GROUP_CATEGORIES[group]]))


def customer_pool(group: str) -> list[CustomerQuestion]:
    """Every question ``group``'s customers may ask, its skills taking turns (a fresh list)."""
    return list(_pool(group))


def pick_questions(group: str, n: int, seed: int) -> list[CustomerQuestion]:
    """``n`` different questions from ``group``'s pool; the same ones for the same seed."""
    pool = _pool(group)  # refuses an unknown group
    rng = np.random.default_rng([seed, GROUPS.index(group)])
    return [pool[int(k)] for k in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
