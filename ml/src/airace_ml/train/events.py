"""Events a training run emits, and their JSON-safe dict form."""

from __future__ import annotations

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


def event_to_dict(e: TrainEvent) -> dict:
    """The event as a plain dict; ``"type"`` is the lowercase class name (e.g. ``"heldouteval"``)."""
    return {"type": type(e).__name__.lower(), **asdict(e)}
