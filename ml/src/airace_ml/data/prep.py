"""Data preparation: the player's cleaning, dedup, fact-check and variety choices as real filters."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from airace_ml.data.corpus import Corpus

CLEANING_THRESHOLDS: dict[str, float] = {"light": 0.15, "standard": 0.40, "thorough": 0.65}
VARIETIES: tuple[str, ...] = ("natural", "balanced")


@dataclass(frozen=True)
class PrepConfig:
    cleaning: Literal["light", "standard", "thorough"] = "standard"
    dedup: bool = False
    fact_check: bool = False
    variety: Literal["natural", "balanced"] = "natural"

    def __post_init__(self) -> None:
        if self.cleaning not in CLEANING_THRESHOLDS:
            raise ValueError(
                f"unknown cleaning level {self.cleaning!r}; expected one of {sorted(CLEANING_THRESHOLDS)}"
            )
        if self.variety not in VARIETIES:
            raise ValueError(f"unknown variety {self.variety!r}; expected one of {list(VARIETIES)}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrepConfig":
        return cls(**d)


def eligible_docs(corpus: Corpus, prep: PrepConfig, purchase_fraction: float = 1.0) -> np.ndarray:
    """Indices (int64, ascending) of the documents a model may train on under ``prep``."""
    tags = corpus.tags
    # Qualities are stored as float16, so compare against the threshold at the same precision.
    keep = tags.quality >= np.float16(CLEANING_THRESHOLDS[prep.cleaning])
    keep &= ~tags.heldout
    if prep.dedup:
        keep &= (tags.dup_cluster < 0) | tags.dup_canonical
    if prep.fact_check:
        keep &= ~tags.false_fact
    keep &= tags.purchase_rank < np.float32(purchase_fraction)
    return np.flatnonzero(keep).astype(np.int64)


def heldout_docs(corpus: Corpus) -> np.ndarray:
    """Indices (int64, ascending) of the documents reserved for evaluation."""
    return np.flatnonzero(corpus.tags.heldout).astype(np.int64)
