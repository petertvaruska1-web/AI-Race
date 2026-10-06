"""Events a training run emits, and their JSON-safe dict form."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class Progress:
    step: int
    total_steps: int
    tokens: int
    loss: float
    lr: float


@dataclass(frozen=True)
class HeldoutEval:
    step: int
    losses: dict[str, float]


@dataclass(frozen=True)
class Sample:
    step: int
    prompt: str
    text: str


@dataclass(frozen=True)
class Instability:
    step: int
    action: Literal["rollback", "stopped"]
    lr_scale: float


@dataclass(frozen=True)
class Done:
    status: Literal["completed", "unstable_stopped"]
    summary: dict


TrainEvent = Progress | HeldoutEval | Sample | Instability | Done


def _json_safe(value):
    """``value`` with every non-finite float replaced by None, recursing into dicts and lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def event_to_dict(e: TrainEvent) -> dict:
    """The event as a plain dict that is valid strict JSON.

    ``"type"`` is the lowercase class name (e.g. ``"heldouteval"``). NaN and infinite floats, which
    ``JSON.parse`` rejects, become ``None`` (JSON ``null``), including inside ``losses`` and
    ``summary``.
    """
    return _json_safe({"type": type(e).__name__.lower(), **asdict(e)})
