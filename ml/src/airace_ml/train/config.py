"""Training run config and the plain-language "learning style" mapped to real optimizer settings."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields

from airace_ml.data.corpus import CUSTOM_DATASET_IDS, DATASET_IDS
from airace_ml.data.prep import PrepConfig
from airace_ml.model.shape import ModelShape

_KNOWN_DATASETS = DATASET_IDS + CUSTOM_DATASET_IDS
_MAX_REPLAY = 0.9


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(frozen=True)
class LearningStyle:
    """Real optimizer settings behind the Careful <-> Bold slider."""

    peak_lr: float
    clip_norm: float
    warmup_frac: float


def learning_style(boldness: float) -> LearningStyle:
    """Map boldness in [0, 1] (0 = Careful, 1 = Bold) to learning-rate settings.

    Bolder means a higher peak rate (log-interpolated), looser gradient clipping and less warmup:
    faster when it works, and a real chance of an unstable run when it does not.
    """
    if not 0.0 <= boldness <= 1.0:  # also rejects NaN
        raise ValueError(f"boldness must be in [0, 1], got {boldness!r}")
    t = boldness
    return LearningStyle(
        peak_lr=10 ** _lerp(math.log10(6e-4), math.log10(1.2e-2), t),
        clip_norm=_lerp(0.5, 2.5, t),
        warmup_frac=_lerp(0.06, 0.02, t),
    )


def _normalized(mixture: Mapping[str, float], what: str) -> dict[str, float]:
    for name, w in mixture.items():
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"{what}: dataset {name!r} has an invalid weight {w!r}")
    total = sum(mixture.values())
    if not total > 0:
        raise ValueError(f"{what}: weights must sum to more than 0")
    return {name: w / total for name, w in mixture.items()}


def _check_mixture(field_name: str, mixture: Mapping[str, float]) -> None:
    for name in mixture:
        if name not in _KNOWN_DATASETS:
            raise ValueError(
                f"{field_name}: unknown dataset {name!r}; expected one of {list(_KNOWN_DATASETS)}"
            )
    _normalized(mixture, field_name)  # finite, non-negative weights with a positive sum


def effective_mixture(
    mixture: Mapping[str, float], replay: float, parent_mixture: Mapping[str, float] | None
) -> dict[str, float]:
    """The normalized blend ``(1 - replay) * norm(mixture) + replay * norm(parent_mixture)``."""
    if not 0.0 <= replay <= 1.0:
        raise ValueError(f"replay must be in [0, 1], got {replay!r}")
    if replay > 0 and parent_mixture is None:
        raise ValueError("replay > 0 needs a parent mixture to replay")
    parts = [(1.0 - replay, _normalized(mixture, "mixture"))]
    if parent_mixture is not None and replay > 0:
        parts.append((replay, _normalized(parent_mixture, "parent mixture")))
    blend: dict[str, float] = {}
    for share, norm in parts:
        if share == 0:
            continue
        for name, w in norm.items():
            blend[name] = blend.get(name, 0.0) + share * w
    return blend


@dataclass
class TrainRunConfig:
    run_id: str
    seed: int
    shape: ModelShape
    token_budget: int
    mixture: dict[str, float]
    finishing_mixture: dict[str, float] | None = None
    replay: float = 0.0
    prep: PrepConfig = field(default_factory=PrepConfig)
    purchases: dict[str, float] = field(default_factory=dict)  # a missing dataset means 1.0
    boldness: float = 0.5
    notebook: list[str] = field(default_factory=list)
    coaching: list[tuple[str, str]] = field(default_factory=list)
    batch_tokens: int = 16384
    probe_prompts: list[str] = field(default_factory=list)
    parent_dir: str | None = None

    @property
    def steps(self) -> int:
        return max(1, self.token_budget // self.batch_tokens)

    @property
    def batch_size(self) -> int:
        return max(1, self.batch_tokens // self.shape.ctx_len)

    def validate(self) -> None:
        """Raise ``ValueError`` (naming the offending field) unless the config can be trained."""
        self.shape.validate()  # ShapeError is a ValueError
        if self.batch_tokens < 1:
            raise ValueError(f"batch_tokens must be at least 1, got {self.batch_tokens}")
        if self.token_budget < self.batch_tokens:
            raise ValueError(
                f"token_budget ({self.token_budget}) must be at least batch_tokens "
                f"({self.batch_tokens})"
            )
        _check_mixture("mixture", self.mixture)
        if self.finishing_mixture is not None:
            _check_mixture("finishing_mixture", self.finishing_mixture)
        if not 0.0 <= self.replay <= _MAX_REPLAY:  # also rejects NaN
            raise ValueError(f"replay must be in [0, {_MAX_REPLAY}], got {self.replay!r}")
        for name, frac in self.purchases.items():
            if name not in _KNOWN_DATASETS:
                raise ValueError(f"purchases: unknown dataset {name!r}")
            if not 0.0 < frac <= 1.0:
                raise ValueError(f"purchases[{name!r}] must be in (0, 1], got {frac!r}")
        if not 0.0 <= self.boldness <= 1.0:
            raise ValueError(f"boldness must be in [0, 1], got {self.boldness!r}")
        if not all(isinstance(entry, str) for entry in self.notebook):
            raise ValueError(f"notebook entries must all be strings, got {self.notebook!r}")
        for pair in self.coaching:
            if len(pair) != 2 or not all(isinstance(part, str) for part in pair):
                raise ValueError(f"coaching entries must be (str, str) pairs, got {pair!r}")

    def to_json(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["shape"] = self.shape.to_dict()
        d["prep"] = self.prep.to_dict()
        return json.dumps(d)  # coaching tuples become lists

    @classmethod
    def from_json(cls, s: str) -> TrainRunConfig:
        d = json.loads(s)
        if type(d) is not dict:  # json.loads yields exactly dict for an object
            raise ValueError("a TrainRunConfig JSON document must be an object")
        spec = {f.name: f for f in fields(cls)}
        unknown = sorted(set(d) - set(spec))
        if unknown:
            raise ValueError(f"unknown TrainRunConfig field(s): {unknown}")
        missing = sorted(
            name
            for name, f in spec.items()
            if name not in d and f.default is MISSING and f.default_factory is MISSING
        )
        if missing:
            raise ValueError(f"missing TrainRunConfig field(s): {missing}")
        d = dict(d)
        d["shape"] = ModelShape.from_dict(d["shape"])
        if "prep" in d:
            d["prep"] = PrepConfig.from_dict(d["prep"])
        if "coaching" in d:
            d["coaching"] = [tuple(pair) for pair in d["coaching"]]
        return cls(**d)
