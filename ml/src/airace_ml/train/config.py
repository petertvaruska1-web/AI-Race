"""Training run config and the plain-language "learning style" mapped to real optimizer settings."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields

from airace_ml.data.corpus import CUSTOM_DATASET_IDS, DATASET_IDS
from airace_ml.data.prep import PrepConfig
from airace_ml.data.sampler import MixtureError
from airace_ml.model.shape import ModelShape

_KNOWN_DATASETS = DATASET_IDS + CUSTOM_DATASET_IDS
_MAX_REPLAY = 0.9
# The run id ends up in file paths, so it is a short plain name: no separators, no leading dot.
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_OBJECT_FIELDS = ("mixture", "purchases")
_OPTIONAL_OBJECT_FIELDS = ("finishing_mixture",)
_ARRAY_FIELDS = ("notebook", "coaching", "probe_prompts")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect(value: object, kind: type | tuple[type, ...], message: str) -> None:
    """Raise ValueError unless ``value`` is a ``kind``. ValueError, not TypeError: a malformed
    config is bad input, and callers catch one exception type for every way it can be wrong."""
    if isinstance(value, kind):
        return
    raise ValueError(f"{message}, got {value!r}")


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
    if not _is_real(boldness) or not 0.0 <= boldness <= 1.0:  # the range test also rejects NaN
        raise ValueError(f"boldness must be a number in [0, 1], got {boldness!r}")
    t = boldness
    return LearningStyle(
        peak_lr=10 ** _lerp(math.log10(6e-4), math.log10(1.2e-2), t),
        clip_norm=_lerp(0.5, 2.5, t),
        warmup_frac=_lerp(0.06, 0.02, t),
    )


def _normalized(mixture: Mapping[str, float], what: str) -> dict[str, float]:
    _expect(mixture, Mapping, f"{what} must map dataset ids to weights")
    for name, w in mixture.items():
        if not _is_real(w) or not math.isfinite(w) or w < 0:
            raise MixtureError(f"{what}: dataset {name!r} has an invalid weight {w!r}")
    total = sum(mixture.values())
    if not total > 0:
        raise MixtureError(f"{what}: weights must sum to more than 0")
    return {name: w / total for name, w in mixture.items()}


def _check_mixture(field_name: str, mixture: Mapping[str, float]) -> None:
    _normalized(
        mixture, field_name
    )  # a mapping of finite, non-negative weights with a positive sum
    for name in mixture:
        if name not in _KNOWN_DATASETS:
            raise MixtureError(
                f"{field_name}: unknown dataset {name!r}; expected one of {list(_KNOWN_DATASETS)}"
            )


def effective_mixture(
    mixture: Mapping[str, float], replay: float, parent_mixture: Mapping[str, float] | None
) -> dict[str, float]:
    """The normalized blend ``(1 - replay) * norm(mixture) + replay * norm(parent_mixture)``."""
    if not _is_real(replay) or not 0.0 <= replay <= 1.0:
        raise ValueError(f"replay must be a number in [0, 1], got {replay!r}")
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


def _check_containers(values: Mapping[str, object]) -> None:
    """Objects must be dicts (``None`` only where allowed) and arrays must be lists."""
    for name in _OBJECT_FIELDS + _OPTIONAL_OBJECT_FIELDS:
        if name not in values or (values[name] is None and name in _OPTIONAL_OBJECT_FIELDS):
            continue
        _expect(values[name], dict, f"{name} must be an object")
    for name in _ARRAY_FIELDS:
        if name in values:
            _expect(values[name], list, f"{name} must be a list")


def _build_nested(name: str, cls: type, value: object):
    """``cls.from_dict(value)``, with every way it can fail reported as a ValueError naming ``name``."""
    _expect(value, dict, f"{name} must be an object")
    unknown = sorted(set(value) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(f"{name}: unknown field(s) {unknown}")
    try:
        return cls.from_dict(value)
    except KeyError as e:
        raise ValueError(f"{name}: missing field {e.args[0]!r}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: {e}") from e


def _coaching_from_json(value: list) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, pair in enumerate(value):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(p, str) for p in pair)
        ):
            raise ValueError(
                f"coaching[{i}] must be a [prompt, reply] pair of strings, got {pair!r}"
            )
        pairs.append((pair[0], pair[1]))
    return pairs


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
        """Raise ``ValueError`` (naming the offending field) unless the config can be trained.

        A bad mixture (unknown dataset, invalid weight, nothing to sample) raises the
        ``ValueError`` subclass :class:`~airace_ml.data.sampler.MixtureError`.

        This is the gate for configs that arrive as JSON from outside Python, so every field is
        type-checked first: nothing but a ValueError may escape for a malformed config.
        """
        if not isinstance(self.run_id, str) or not _RUN_ID.fullmatch(self.run_id):
            raise ValueError(
                f"run_id must be 1-64 letters, digits, '.', '_' or '-' (not starting with a "
                f"punctuation mark), got {self.run_id!r}"
            )
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {self.seed!r}")
        _expect(self.shape, ModelShape, "shape must be a ModelShape")
        self.shape.validate()  # ShapeError is a ValueError
        if not _is_int(self.batch_tokens) or self.batch_tokens < 1:
            raise ValueError(f"batch_tokens must be an integer >= 1, got {self.batch_tokens!r}")
        if not _is_int(self.token_budget):
            raise ValueError(f"token_budget must be an integer, got {self.token_budget!r}")
        if self.token_budget < self.batch_tokens:
            raise ValueError(
                f"token_budget ({self.token_budget}) must be at least batch_tokens "
                f"({self.batch_tokens})"
            )
        _check_containers({f.name: getattr(self, f.name) for f in fields(self)})
        _check_mixture("mixture", self.mixture)
        if self.finishing_mixture is not None:
            _check_mixture("finishing_mixture", self.finishing_mixture)
        if not _is_real(self.replay) or not 0.0 <= self.replay <= _MAX_REPLAY:  # also rejects NaN
            raise ValueError(f"replay must be a number in [0, {_MAX_REPLAY}], got {self.replay!r}")
        if self.parent_dir is not None and not (
            isinstance(self.parent_dir, str) and self.parent_dir
        ):
            raise ValueError(
                f"parent_dir must be a non-empty string or null, got {self.parent_dir!r}"
            )
        if self.replay > 0 and self.parent_dir is None:
            raise ValueError("replay > 0 needs parent_dir, the run whose data mix is replayed")
        for name, frac in self.purchases.items():
            if name not in _KNOWN_DATASETS:
                raise ValueError(f"purchases: unknown dataset {name!r}")
            if not _is_real(frac) or not 0.0 < frac <= 1.0:
                raise ValueError(f"purchases[{name!r}] must be a number in (0, 1], got {frac!r}")
        if not _is_real(self.boldness) or not 0.0 <= self.boldness <= 1.0:
            raise ValueError(f"boldness must be a number in [0, 1], got {self.boldness!r}")
        _expect(self.prep, PrepConfig, "prep must be a PrepConfig")
        for flag in ("dedup", "fact_check"):
            _expect(getattr(self.prep, flag), bool, f"prep.{flag} must be true or false")
        if not all(isinstance(entry, str) for entry in self.notebook):
            raise ValueError(f"notebook entries must all be strings, got {self.notebook!r}")
        for i, pair in enumerate(self.coaching):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(isinstance(part, str) for part in pair)
            ):
                raise ValueError(
                    f"coaching[{i}] must be a (prompt, reply) pair of strings, got {pair!r}"
                )
        if not all(isinstance(prompt, str) for prompt in self.probe_prompts):
            raise ValueError(f"probe_prompts must all be strings, got {self.probe_prompts!r}")

    def to_json(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["shape"] = self.shape.to_dict()
        d["prep"] = self.prep.to_dict()
        return json.dumps(d)  # coaching tuples become lists

    @classmethod
    def from_json(cls, s: str) -> TrainRunConfig:
        """Parse a config; a malformed document raises ValueError naming the offending field.

        This only checks that the document has the right structure. Call :meth:`validate` to check
        that the values describe a trainable run.
        """
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
        _check_containers(d)
        d["shape"] = _build_nested("shape", ModelShape, d["shape"])
        if "prep" in d:
            d["prep"] = _build_nested("prep", PrepConfig, d["prep"])
        if "coaching" in d:
            d["coaching"] = _coaching_from_json(d["coaching"])
        return cls(**d)
