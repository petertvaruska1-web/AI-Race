"""Shared contract of the skill generators: documents, benchmark items and seeding.

Every generator returns :class:`TextDoc` for training and one of the item dataclasses for
benchmarks. Randomness always comes from a :func:`skill_rng` stream, so a (name, split, version)
triple fixes a generator's output forever, and :func:`reserved_for_bench` partitions the
canonical keys of generated items so training text never contains a benchmark item.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from airace_ml.tokenizer import Role

Split = Literal["train", "bench"]


def skill_rng(name: str, split: Split, version: str = "v1") -> np.random.Generator:
    """The seeded stream for one generator and split.

    The seed is the first 8 bytes (big-endian) of ``blake2b(f"{version}:{split}:{name}")``, so
    train and bench streams of the same generator are unrelated.
    """
    digest = hashlib.blake2b(f"{version}:{split}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def reserved_for_bench(key: str) -> bool:
    """Whether the item with this canonical key belongs to the benchmark (about 10% of keys)."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10 == 0


@dataclass
class TextDoc:
    """One training document: running text (``plain``) or a list of chat turns (``chat``)."""

    kind: Literal["plain", "chat"]
    text: str = ""
    turns: list[tuple[Role, str]] | None = None
    topic: str = "other"


@dataclass
class MCItem:
    """Multiple choice: the model picks the option it finds most likely after ``prompt``."""

    id: str
    category: str
    prompt: str
    options: list[str]
    answer_index: int
    tags: tuple[str, ...]
    chat: bool = False
    group: str | None = None


@dataclass
class ExactItem:
    """Free answer: the extracted generation must match one of ``answers`` (case-insensitive)."""

    id: str
    category: str
    prompt: str
    answers: list[str]
    tags: tuple[str, ...]
    chat: bool = False
    max_new_tokens: int = 8
    extract: Literal["first_item", "first_line"] = "first_line"


@dataclass
class CheckItem:
    """Free answer judged by a checker; ``reference`` is a reply known to pass the check."""

    id: str
    category: str
    prompt: str
    check: str
    check_args: dict[str, Any]
    reference: str
    tags: tuple[str, ...]
    chat: bool = True
    max_new_tokens: int = 48


@dataclass
class PairItem:
    """Two texts; the model should find ``good`` more likely than ``bad``."""

    id: str
    category: str
    good: str
    bad: str
    tags: tuple[str, ...]
