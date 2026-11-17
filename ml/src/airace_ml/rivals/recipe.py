"""The eight rival lineages of the pack: two per business style, each a few versions that grow.

Shapes are the game's brain sizes at an attention span of 256; each version continues from the
one before (growing when the shape grows, R6) and reads another 12,582,912 tokens. The premium
rivals end on the Large brain and twice that text: players of the newer rules can train Large
and Huge brains, and the big-budget rivals must stay worth beating.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from airace_ml.data.corpus import DATASET_IDS
from airace_ml.experiments.configs import mix_heavy
from airace_ml.model.shape import ModelShape

STEP_TOKENS = 12_582_912
TINY = ModelShape(4, 128, 256)
SMALL = ModelShape(6, 192, 256)
MEDIUM = ModelShape(8, 256, 256)
LARGE = ModelShape(10, 384, 256)


@dataclass(frozen=True)
class VersionStep:
    shape: ModelShape
    tokens: int


@dataclass(frozen=True)
class LineageRecipe:
    id: str
    style: str
    focus: str
    mix: dict[str, float]
    steps: tuple[VersionStep, ...]


def rest_even(**shares: float) -> dict[str, float]:
    """``shares`` as given, the rest split evenly over the other datasets."""
    others = [ds for ds in DATASET_IDS if ds not in shares]
    rest = (1.0 - sum(shares.values())) / len(others)
    return {ds: shares.get(ds, rest) for ds in DATASET_IDS}


def _steps(*shapes: ModelShape) -> tuple[VersionStep, ...]:
    return tuple(VersionStep(s, STEP_TOKENS) for s in shapes)


# A Large brain shows what it can do only on more text (measured: on 12.6M tokens it ends level
# with the Medium brain it grew from), so the premium rivals' last version reads twice as much.
PREMIUM_STEPS = (*_steps(SMALL, MEDIUM), VersionStep(LARGE, 2 * STEP_TOKENS))

LINEAGES: tuple[LineageRecipe, ...] = (
    LineageRecipe("premium-a", "premium", "businesses", rest_even(), PREMIUM_STEPS),
    LineageRecipe(
        "premium-b",
        "premium",
        "researchers",
        rest_even(facts=0.35, educational=0.35),
        PREMIUM_STEPS,
    ),
    LineageRecipe(
        "budget-a",
        "budget",
        "everyday",
        rest_even(web=0.4, conversations=0.4),
        _steps(TINY, TINY, SMALL),
    ),
    LineageRecipe(
        "budget-b",
        "budget",
        "students",
        rest_even(educational=0.4, conversations=0.3),
        _steps(TINY, TINY, SMALL),
    ),
    LineageRecipe(
        "specialist-code",
        "specialist",
        "developers",
        mix_heavy("code"),
        _steps(TINY, SMALL, MEDIUM),
    ),
    LineageRecipe(
        "specialist-stories",
        "specialist",
        "creators",
        mix_heavy("creative"),
        _steps(TINY, SMALL, MEDIUM),
    ),
    LineageRecipe(
        "consumer-a",
        "consumer",
        "everyday",
        mix_heavy("conversations"),
        _steps(TINY, SMALL, SMALL),
    ),
    LineageRecipe(
        "consumer-b",
        "consumer",
        "creators",
        rest_even(books=0.3, creative=0.3, conversations=0.3),
        _steps(TINY, SMALL, SMALL),
    ),
)


# The product words of the game's rival names (``PRODUCT_WORDS`` in ``packages/sim/src/rivalCast.ts``).
PRODUCT_WORDS = {
    "premium": ("Titan", "Apex", "Sage"),
    "budget": ("Lite", "Go", "Spark"),
    "specialist": ("Forge", "Scholar", "Muse"),
    "consumer": ("Buddy", "Chat", "Pal"),
}


def product_name(recipe: LineageRecipe, k: int) -> str:
    """The name the game gives a rival's ``k``-th product (from 1), as ``rivalProductName`` does.

    The game draws a rival's word and company name per world, so the pack cannot know them: a
    lineage takes the word that its place among its style's lineages picks (the first or the
    second) and numbers its versions like the game numbers a rival's releases."""
    same_style = [r for r in LINEAGES if r.style == recipe.style]
    words = PRODUCT_WORDS[recipe.style]
    word = words[same_style.index(recipe) % len(words)]
    return word if k == 1 else f"{word} {k}"


def recipe_digest(recipe: LineageRecipe) -> str:
    return hashlib.sha256(json.dumps(asdict(recipe), sort_keys=True).encode()).hexdigest()
