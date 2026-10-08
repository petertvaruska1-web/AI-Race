"""The feasibility gate's runs and measurement caches: what is reused when the gate starts again.

Each run trains into ``runs_dir/<name>`` and records a key in ``gate_run.json``: its config (with
its parent named, not located), a digest of the training data, a digest of the training code
(:data:`TRAINING_SOURCES`), the device type and the token of the exact parent training it
continued from. A finished run is reused while its key matches; an interrupted one carries on
from its last saved point. Every training gets a fresh token, so whatever was measured on or
grown from an older training is never mistaken for this one.

Measurements cached beside a run (:func:`_cached`) are keyed by the run's token and by what
measured them; the gate adds a digest of the measuring code (:data:`MEASURING_SOURCES`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

import airace_ml
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import Instability, Progress, TrainEvent, json_safe
from airace_ml.train.trainer import can_resume, train_run

GATE_RUN_FILE = "gate_run.json"
PACKAGE_ROOT = Path(airace_ml.__file__).resolve().parent
# What a trained model depends on, and what a benchmark or fingerprint result depends on.
TRAINING_SOURCES: tuple[str, ...] = ("train", "model", "data", "tokenizer.py")
MEASURING_SOURCES: tuple[str, ...] = ("evals", "personality", "infer", "skills")


def source_digest(parts: Sequence[str], root: Path = PACKAGE_ROOT) -> str:
    """A sha256 digest of the source files of ``parts`` (files or folders, relative to
    ``root``; folders are read recursively, compiled caches skipped).

    Each file contributes its path relative to ``root`` and its bytes, in sorted path order, so
    the digest changes exactly when a file is added, removed, renamed or edited.
    """
    root = Path(root)
    files: list[Path] = []
    for part in parts:
        path = root / part
        if path.is_dir():
            files += [
                p
                for p in path.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.relative_to(root).parts
                and p.suffix != ".pyc"
            ]
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"no source file or folder at {path}")
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda p: p.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        for part in (name, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    """Write ``value`` as strict JSON to a temporary file, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), allow_nan=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _plain(value: Any) -> Any:
    """``value`` as it reads back from JSON (tuples become lists), so keys compare equal."""
    return json.loads(json.dumps(json_safe(value), allow_nan=False))


def _cached(path: Path, key: dict, compute: Callable[[], Any]) -> Any:
    """The value cached at ``path`` under ``key``, or ``compute()``'s, cached there."""
    key = _plain(key)
    record = _read_json(path)
    if isinstance(record, dict) and record.get("key") == key:
        return record["value"]
    value = _plain(compute())
    _write_json(path, {"key": key, "value": value})
    return value


@dataclass
class GateRun:
    """A trained run of the gate. ``token`` names this exact training: a run trained again gets a
    new one, so caches and children of the old training are not reused. ``steps`` is how many of
    the ``planned_steps`` it ran (fewer when it stopped early)."""

    name: str
    dir: Path
    token: str
    status: str
    device: str
    steps: int
    planned_steps: int
    wall_seconds: float
    heldout_mean: float | None
    reused: bool


def _heldout_mean(losses: Mapping[str, float | None]) -> float | None:
    finite = [
        float(v)
        for v in losses.values()
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    ]
    return math.fsum(finite) / len(finite) if finite else None


class GateRuns:
    """Trains the gate's runs under ``runs_dir``, reusing finished ones and resuming unfinished
    ones (see the module docstring). ``code`` is the digest of the training code."""

    def __init__(
        self,
        data_root: Path,
        runs_dir: Path,
        identity: str,
        on_progress: Callable[[str], None] = print,
        code: str = "",
    ) -> None:
        self.data_root = Path(data_root)
        self.runs_dir = Path(runs_dir)
        self.identity = identity
        self.on_progress = on_progress
        self.code = code

    def _key(self, cfg: TrainRunConfig, device: torch.device, parent: GateRun | None) -> dict:
        config = json.loads(cfg.to_json())
        config["parent_dir"] = parent.name if parent is not None else None
        return _plain(
            {
                "config": config,
                "data": self.identity,
                "code": self.code,
                "device": device.type,
                "parent": parent.token if parent is not None else None,
            }
        )

    def train(
        self, cfg: TrainRunConfig, device: torch.device, parent: GateRun | None = None
    ) -> GateRun:
        """The finished run of ``cfg`` on ``device`` (continuing from ``parent``, if given)."""
        out = self.runs_dir / cfg.run_id
        cfg = replace(cfg, parent_dir=str(parent.dir) if parent is not None else None)
        key = self._key(cfg, device, parent)
        record = _read_json(out / GATE_RUN_FILE)
        same = isinstance(record, dict) and record.get("key") == key
        if same and record.get("status") == "completed" and self._has_checkpoint(out):
            self.on_progress(f"  {cfg.run_id}: finished earlier, reusing it")
            return self._run(cfg.run_id, out, record, reused=True)
        resume = same and self._can_resume(out, cfg)
        _write_json(out / GATE_RUN_FILE, {"key": key, "status": "started"})
        action = "resuming" if resume else "training"
        self.on_progress(
            f"  {cfg.run_id}: {action} ({cfg.steps} steps, {cfg.token_budget:,} tokens, "
            f"on {device.type})"
        )
        result = train_run(
            cfg,
            out_dir=out,
            data_root=self.data_root,
            device=device,
            on_event=self._events(cfg.run_id, cfg.steps),
            resume=resume,
        )
        record = {
            "key": key,
            "status": result.status,
            "token": uuid.uuid4().hex,
            "result": {
                "steps": result.steps,
                "planned_steps": cfg.steps,
                "wall_seconds": result.wall_seconds,
                "final_loss": result.final_loss,
                "heldout_losses": result.heldout_losses,
            },
        }
        _write_json(out / GATE_RUN_FILE, record)
        run = self._run(cfg.run_id, out, _plain(record), reused=False)
        loss = "n/a" if run.heldout_mean is None else f"{run.heldout_mean:.3f}"
        self.on_progress(
            f"  {cfg.run_id}: {result.status} in {result.wall_seconds:.1f} s, held-out loss {loss}"
        )
        return run

    @staticmethod
    def _has_checkpoint(out: Path) -> bool:
        return (out / WEIGHTS_NAME).is_file() and (out / META_NAME).is_file()

    @staticmethod
    def _can_resume(out: Path, cfg: TrainRunConfig) -> bool:
        try:
            return can_resume(out, cfg)
        except (OSError, ValueError):
            return False  # a saved state too damaged to read is started over

    @staticmethod
    def _run(name: str, out: Path, record: dict, *, reused: bool) -> GateRun:
        result = record["result"]
        return GateRun(
            name=name,
            dir=out,
            token=record["token"],
            status=record["status"],
            device=record["key"]["device"],
            steps=result["steps"],
            planned_steps=result["planned_steps"],
            wall_seconds=result["wall_seconds"],
            heldout_mean=_heldout_mean(result["heldout_losses"]),
            reused=reused,
        )

    def _events(self, name: str, steps: int) -> Callable[[TrainEvent], None]:
        """Progress at each quarter of the run, and every instability."""
        marks = [steps * q // 4 for q in (1, 2, 3)]

        def on_event(event: TrainEvent) -> None:
            if isinstance(event, Progress) and marks and event.step >= marks[0]:
                while marks and event.step >= marks[0]:
                    marks.pop(0)
                self.on_progress(
                    f"  {name}: step {event.step}/{event.total_steps}, loss {event.loss:.3f}"
                )
            elif isinstance(event, Instability):
                if event.action == "rollback":
                    what = "went back to its last good weights"
                else:
                    what = "stopped, keeping its last good weights"
                self.on_progress(f"  {name}: unstable at step {event.step}; {what}")

        return on_event
