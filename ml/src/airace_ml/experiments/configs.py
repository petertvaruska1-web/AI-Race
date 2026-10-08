"""The feasibility gate's model shapes, token budgets and data mixes (spec 4.11-4.12).

The real gate trains at :data:`FULL_SCALE`: a first model at :data:`STARTER_SHAPE` (about 1.4 M
parameters) on :data:`STARTER_BUDGET` tokens, and the comparison models at :data:`EARLY_SHAPE`
on :data:`EARLY_BUDGET`. :data:`QUICK_SCALE` swaps in tiny shapes and budgets so the whole gate
runs in minutes on a CPU; a quick run checks the plumbing and says nothing about feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from airace_ml.data.corpus import DATASET_IDS
from airace_ml.model.shape import ModelShape

STARTER_SHAPE = ModelShape(4, 128, 256)
EARLY_SHAPE = ModelShape(6, 192, 256)
STARTER_BUDGET = 4_194_304
EARLY_BUDGET = 8_388_608

BALANCED_MIX: dict[str, float] = {ds: 1.0 for ds in DATASET_IDS}

# Each contrasting mix and the benchmark category it should lead (criterion G3).
TARGET_CATEGORY: dict[str, str] = {
    "creative": "creativity",
    "code": "coding",
    "facts": "knowledge",
    "conversations": "instruction",
}

QUICK_SHAPE = ModelShape(2, 64, 64)
QUICK_GROWN_SHAPE = ModelShape(3, 96, 64)  # a quick child that grows still exercises growth
QUICK_BATCH_TOKENS = 1024
QUICK_BUDGET = QUICK_BATCH_TOKENS * 20  # 20 steps


def mix_heavy(target: str, share: float = 0.6) -> dict[str, float]:
    """A mix of all eight datasets that puts ``share`` on ``target`` and splits the rest evenly
    over the other seven."""
    if target not in DATASET_IDS:
        raise ValueError(f"unknown dataset {target!r}; expected one of {list(DATASET_IDS)}")
    if not 0.0 <= share <= 1.0:  # also rejects NaN
        raise ValueError(f"share must be in [0, 1], got {share!r}")
    rest = (1.0 - share) / (len(DATASET_IDS) - 1)
    return {ds: share if ds == target else rest for ds in DATASET_IDS}


@dataclass(frozen=True)
class GateScale:
    """The shapes and budgets of every run the gate trains.

    ``grown_shape`` is the shape G5 grows the first model to. A continuation run (G5, G6) adds
    ``starter_budget`` tokens to its parent; G7's runs train on ``prep_budget``. ``batch_tokens``
    of ``None`` keeps the trainer's default. ``fingerprint_k`` is how many replies per probe a
    personality fingerprint samples.
    """

    starter_shape: ModelShape
    early_shape: ModelShape
    grown_shape: ModelShape
    starter_budget: int
    early_budget: int
    prep_budget: int
    batch_tokens: int | None
    fingerprint_k: int


FULL_SCALE = GateScale(
    starter_shape=STARTER_SHAPE,
    early_shape=EARLY_SHAPE,
    grown_shape=EARLY_SHAPE,
    starter_budget=STARTER_BUDGET,
    early_budget=EARLY_BUDGET,
    prep_budget=2 * STARTER_BUDGET,
    batch_tokens=None,
    fingerprint_k=3,
)

QUICK_SCALE = GateScale(
    starter_shape=QUICK_SHAPE,
    early_shape=QUICK_SHAPE,
    grown_shape=QUICK_GROWN_SHAPE,
    starter_budget=QUICK_BUDGET,
    early_budget=QUICK_BUDGET,
    prep_budget=QUICK_BUDGET,
    batch_tokens=QUICK_BATCH_TOKENS,
    fingerprint_k=1,
)
