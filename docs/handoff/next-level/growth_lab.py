"""Lab: how much of its parent does a grown brain keep, and can it keep more?

A player's AI grows from one generation to the next: the trained brain is enlarged exactly
function-preserving (``airace_ml/model/growth.py``) and trained further. Measured with engine 2,
what a Tiny parent learned is worth only about 0.014 held-out loss to the Small grown from it.
This script measures whether changing how fast the inherited and the new parts of the grown brain
learn at first keeps more of the parent. Nothing under ``ml/src`` is changed: the variants patch
the trainer for one child run at a time and put everything back afterwards.

Run it from ``ml/`` (it uses the GPU):

    python -m uv run python ../docs/handoff/next-level/growth_lab.py --out runs/growth-lab \\
        [--seeds 1 2] [--arms A B C D E] [--bench]

For each seed ``s`` it trains (finished runs in ``--out`` are reused, never trained again):

- ``parent-tiny-s``: the scorecard's first AI (Tiny, 12,582,912 tokens, the even mix, seed ``s``),
  its config built by the scorecard's own ``_Card.config`` from the scorecard's constants.
- A (as today): a Small continuing ``parent-tiny-s`` for 12,582,912 tokens with seed
  ``s + CONTINUATION_SEED_OFFSET`` (the scorecard's ``lineage-small``).
- B (new parts move faster): as A, but during the first third of the child's steps the realised
  update of every parameter element that did not exist in the parent is multiplied by 3.
- C (old parts move slower): as A, but during the first third the realised update of every element
  copied from the parent is multiplied by 0.3.
- D (the most a perfect transfer could give): a new Small trained like ``parent-tiny-s`` (same
  tokens, mix and seed ``s``), then a Small continuing it for 12,582,912 tokens with seed
  ``s + CONTINUATION_SEED_OFFSET``; the second one is reported.
- E (no inheritance): a new Small, 12,582,912 tokens, seed ``s + CONTINUATION_SEED_OFFSET``.

How B and C work. "Realised update" is a parameter after the optimizer's step minus the same
parameter before it. Inside :func:`transfer_wrapper` (used for the B and C child runs only, and for
self-check 2), three names are replaced and restored when the block ends, however it ends:

- ``trainer.grow`` (the name the trainer calls): after the real ``grow(parent, target, seed=...)``
  it knows the parent and the grown model, and builds, once, which element of every grown
  parameter was copied from the parent (:func:`copied_elements`, from
  ``growth.insertion_layout`` and the real shapes). It also checks that the copied elements hold
  the parent's values, and that the new writers are zero, so a mask on the wrong elements fails.
- ``trainer._Run._train_step``: only records the step number the trainer is on.
- ``MuonAdamW.step`` (the class the trainer's optimizer is): at its first call it binds the masks to
  the parameters of the grown model and checks that they are exactly the optimizer's parameters;
  during the first third of the run's steps it clones every parameter, runs the real step and sets
  ``p = before + mask * (p - before)`` (``p`` itself where the mask is 1, so a scale of 1 changes no
  bit); after the first third it is the real step alone. The steps it acted on are recorded and
  must be exactly the first third, less any step a spike rolled back.

Self-checks, run before the children are trained (any failure stops the script):

1. Growth is function-preserving for the parent just trained: on the CPU, in float32, the grown
   model's logits equal the parent's on the trainer's held-out rows to 1e-4.
2. With both scales 1.0 the wrapped step leaves bit-identical parameters to the unwrapped step:
   two child runs of 8 steps from the same parent and seed, one plain and one wrapped, compared
   tensor by tensor. They run on the CPU, because two identical runs on a CUDA card are not
   guaranteed to be bit-identical (atomic additions in the embedding's backward pass), and the
   wrapper's arithmetic does not depend on the device. A second, one-step check applies the B and
   C scales to the grown model with the same gradients as a plain step and checks that copied
   elements moved by exactly ``old_scale`` times and new ones by ``new_scale`` times the plain
   update.
3. It prints how many parameter elements of the grown Small are copied from the parent (old) and
   how many are new.

Measured for every final model: the trainer's mean held-out loss (from the run's result), the
scorecard's yardstick loss (``scorecard.yardstick_loss``), the training seconds and, with
``--bench``, bench-v1 overall exactly as the scorecard measures it: the same ``run_benchmarks``
call as ``scorecard._Card.measure``, with the same suite, the reference judge and novelty index
loaded by the same loaders, and the same seed (the whole ``_Card`` is not needed). For the A, B and
C children also the held-out mean at every evaluation during training. A final table gives arm by
seed for each measure, the mean over seeds per arm and its difference to A, and everything goes to
``--out/results.json``. Measurements are cached beside each run under ``--out/measured``.

How long: about 90 s a Small run and less for a Tiny one on the RTX 3050, so per seed about
10 minutes for the seven runs (the Tiny parent, A, B, C, D's new Small, D and E) and about two
minutes for the self-checks on the CPU (first seed); about 22 minutes for two seeds, and about
10 minutes more with ``--bench`` (12 final models).

Hidden options for checks of the script itself: ``--tiny-root DIR`` trains on a tiny data root
(the trainer tests' fixture) at tiny shapes and budgets, and ``--device cpu``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import sys
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from airace_ml.awake import stay_awake
from airace_ml.data.identity import data_identity
from airace_ml.evals.suite import SUITE_VERSION, build_suite, run_benchmarks
from airace_ml.experiments import scorecard as sc
from airace_ml.experiments.gate import CONTINUATION_SEED_OFFSET, _load_judge, _load_novelty
from airace_ml.experiments.runs import (
    TRAINING_SOURCES,
    _cached,
    _heldout_mean,
    _plain,
    _read_json,
    _write_json,
    source_digest,
)
from airace_ml.infer.lm import TorchLM
from airace_ml.model import growth
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME, CheckpointMeta, load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import data_root as default_data_root
from airace_ml.paths import tokenizer_path
from airace_ml.tokenizer import Tok
from airace_ml.train import trainer
from airace_ml.train.config import TrainRunConfig, learning_style
from airace_ml.train.events import HeldoutEval, Instability, Progress, TrainEvent
from airace_ml.train.optim import MuonAdamW

LAB_VERSION = "growth-lab-1"
ARMS = ("A", "B", "C", "D", "E")
# The arms whose child runs through the wrapper, each with its (old_scale, new_scale).
TRANSFER_SCALES: dict[str, tuple[float, float]] = {"B": (1.0, 3.0), "C": (0.3, 1.0)}
TRANSFER_SHARE = 3  # The wrapper acts during the first third of the child's steps.
FUNCTION_TOLERANCE = 1e-4
KEEP_MARGIN = 0.02  # A variant is kept only if its held-out loss beats A's by this on every seed.
CHECK_STEPS = 8
RECORD_FILE = "lab_run.json"
RESULTS_FILE = "results.json"
RUNS_DIR = "runs"
MEASURED_DIR = "measured"
CHECK_DIR = "selfcheck"
REPO_ROOT = Path(__file__).resolve().parents[3]

# The tiny scale of ``--tiny-root``: the trainer tests' shapes, 12 steps of 1,024 tokens.
TINY_SCALE = sc.CardScale(
    "lab-tiny",
    shapes={sc.TINY: ModelShape(2, 64, 64), sc.SMALL: ModelShape(3, 96, 64)},
    tokens=1024 * 12,
    batch_tokens=1024,
)


class LabError(RuntimeError):
    """A self-check or a consistency check failed: the results would not be trustworthy."""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`LabError` with ``message`` unless ``condition`` holds (kept under ``-O``)."""
    if not condition:
        raise LabError(message)


# -- which elements of a grown model were copied from its parent ------------------------------


def _parent_name(name: str, layout: list[int | None]) -> str | None:
    """The parent's parameter that the grown parameter ``name`` was copied from, or ``None`` for a
    parameter of a fresh block (which holds nothing of the parent)."""
    parts = name.split(".")
    if parts[0] != "blocks":
        return name  # The token embedding and the final norm keep their names.
    source = layout[int(parts[1])]
    if source is None:
        return None
    return ".".join(["blocks", str(source), *parts[2:]])


def _is_norm_gain(name: str) -> bool:
    return name.endswith(("attn_norm.weight", "mlp_norm.weight")) or name == "norm_f.weight"


@torch.no_grad()
def copied_elements(parent: Transformer, grown: Transformer) -> dict[str, Tensor]:
    """For every parameter of ``grown`` (by name), a boolean tensor of its shape that is True where
    the element was copied from ``parent`` by :func:`growth.grow`.

    The token embedding and every norm gain: the leading slice of the parent's same parameter (the
    embedding's first ``d_old`` columns, a gain's first ``d_old`` entries). A block that
    ``growth.insertion_layout`` marks ``None`` is all new. A copied block: for each weight, the
    leading slice with the shape of the parent block's same weight (what ``growth._copy_into``
    copies).

    It checks, and raises :class:`LabError` otherwise, that every parent parameter is used exactly
    once, that each copied slice holds the parent's values (a norm gain scaled by
    ``sqrt(d_old / d_new)`` as growth scales it), and that the writers growth zeroes (the
    embedding's new columns, a fresh block's ``wo`` and ``w_down``, the new part of a copied
    block's ``wo`` and ``w_down``) are zero.
    """
    old_shape, new_shape = parent.shape, grown.shape
    layout = growth.insertion_layout(old_shape.n_layer, new_shape.n_layer)
    _require(len(layout) == len(grown.blocks), "the layout does not cover the grown blocks")
    _require(
        parent.tok_emb.num_embeddings == grown.tok_emb.num_embeddings,
        "the parent and the grown model have different vocabularies",
    )
    parent_params = dict(parent.named_parameters())
    gain = math.sqrt(old_shape.d_model / new_shape.d_model)
    used: list[str] = []
    old: dict[str, Tensor] = {}
    for name, p in grown.named_parameters():
        mask = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        source = _parent_name(name, layout)
        if source is not None:
            _require(source in parent_params, f"{name}: the parent has no parameter {source}")
            src = parent_params[source]
            _require(
                src.dim() == p.dim()
                and all(a <= b for a, b in zip(src.shape, p.shape, strict=True)),
                f"{name}: the parent's {source} {tuple(src.shape)} does not fit in "
                f"{tuple(p.shape)}",
            )
            region = tuple(slice(0, n) for n in src.shape)
            mask[region] = True
            expected = src.to(p.device, p.dtype) * gain if _is_norm_gain(name) else src
            _require(
                torch.allclose(p[region], expected.to(p.device, p.dtype), rtol=1e-6, atol=0.0),
                f"{name}: the elements taken as copied do not hold the parent's {source}",
            )
            used.append(source)
        writer = name == "tok_emb.weight" or name.endswith(("attn.wo.weight", "mlp.w_down.weight"))
        if writer:
            _require(
                bool((p[~mask] == 0).all()),
                f"{name}: an element taken as new is not zero, though growth zeroes new writers",
            )
        old[name] = mask
    _require(
        sorted(used) == sorted(parent_params),
        f"the parent's parameters were not each copied once: used {sorted(used)}",
    )
    return old


def scale_masks(old: dict[str, Tensor], old_scale: float, new_scale: float) -> dict[str, Tensor]:
    """Float masks: ``old_scale`` on copied elements, ``new_scale`` on new ones."""
    return {
        name: torch.where(mask, old_scale, new_scale).to(torch.float32)
        for name, mask in old.items()
    }


def element_counts(old: dict[str, Tensor]) -> dict[str, int]:
    copied = sum(int(mask.sum()) for mask in old.values())
    total = sum(mask.numel() for mask in old.values())
    return {"old": copied, "new": total - copied, "total": total}


# -- the wrapper ------------------------------------------------------------------------------


def apply_scaled(pairs: list[tuple[Tensor, Tensor]], real: Callable[[], object]) -> object:
    """Run ``real()`` (an optimizer step) and scale each parameter's realised update by its mask:
    ``p = before + mask * (p - before)``, and ``p`` unchanged where the mask is exactly 1."""
    with torch.no_grad():
        before = [p.detach().clone() for p, _ in pairs]
    result = real()
    with torch.no_grad():
        for (p, mask), b in zip(pairs, before, strict=True):
            p.copy_(torch.where(mask == 1.0, p, b + mask * (p - b)))
    return result


@dataclass
class Transfer:
    """One wrapped child run: its scales and what the wrapper saw."""

    old_scale: float
    new_scale: float
    parent_shape: ModelShape
    child: TrainRunConfig
    active_steps: int
    grow_calls: int = 0
    grown: Transformer | None = None
    old: dict[str, Tensor] | None = None
    optimizer: torch.optim.Optimizer | None = None
    pairs: list[tuple[Tensor, Tensor]] | None = None
    step: int = 0
    applied: list[int] = field(default_factory=list)


_REAL_GROW = trainer.grow
_REAL_TRAIN_STEP = trainer._Run._train_step
_MARK = "_growth_lab_wrapper"


def _has_lab_wrapper(func) -> bool:
    """Whether ``func`` is this lab's optimizer wrapper or wraps it. The optimizer's ``step`` is
    not a fixed object: PyTorch wraps a class's ``step`` (for its profiler and step hooks) the
    first time an instance is made, so the chain of ``__wrapped__`` functions is searched."""
    seen = 0
    while func is not None and seen < 16:
        if getattr(func, _MARK, False):
            return True
        func, seen = getattr(func, "__wrapped__", None), seen + 1
    return False


def assert_unpatched() -> None:
    """The trainer as it ships: no wrapper left behind."""
    _require(
        trainer.grow is _REAL_GROW is growth.grow
        and not _has_lab_wrapper(MuonAdamW.step)
        and trainer._Run._train_step is _REAL_TRAIN_STEP,
        "a wrapper of the trainer is still in place",
    )


@contextlib.contextmanager
def transfer_wrapper(t: Transfer) -> Iterator[Transfer]:
    """Wrap the trainer for one child run (see the module docstring); restored however it ends."""
    assert_unpatched()
    real_step = MuonAdamW.step  # whatever PyTorch's hooks made of it so far; restored as it is

    def lab_grow(model: Transformer, target: ModelShape, *, seed: int) -> Transformer:
        t.grow_calls += 1
        _require(t.grow_calls == 1, "the trainer grew a model twice in one wrapped run")
        _require(
            model.shape == t.parent_shape,
            f"the model grown is {model.shape}, not the parent's {t.parent_shape}",
        )
        _require(
            target == t.child.shape and seed == t.child.seed,
            f"growth to {target} with seed {seed}, not the child's {t.child.shape}, {t.child.seed}",
        )
        grown = _REAL_GROW(model, target, seed=seed)
        t.old = copied_elements(model, grown)
        t.grown = grown
        return grown

    def lab_train_step(run, step: int, factor: float):
        t.step = step
        return _REAL_TRAIN_STEP(run, step, factor)

    def lab_step(opt, closure=None):
        _require(t.step >= 1, "the optimizer stepped outside a training step")
        if t.step > t.active_steps:
            return real_step(opt, closure)
        if t.optimizer is None:
            _require(t.grown is not None and t.old is not None, "no grown model to mask")
            named = dict(t.grown.named_parameters())
            in_optimizer = [p for group in opt.param_groups for p in group["params"]]
            _require(
                len(in_optimizer) == len(named)
                and {id(p) for p in in_optimizer} == {id(p) for p in named.values()},
                "the optimizer's parameters are not the grown model's",
            )
            masks = scale_masks(t.old, t.old_scale, t.new_scale)
            t.pairs = []
            for name, p in named.items():
                mask = masks[name].to(p.device, p.dtype)
                _require(mask.shape == p.shape, f"{name}: mask {mask.shape} is not {p.shape}")
                t.pairs.append((p, mask))
            t.optimizer = opt
        _require(opt is t.optimizer, "a second optimizer stepped in the wrapped run")
        t.applied.append(t.step)
        return apply_scaled(t.pairs, lambda: real_step(opt, closure))

    setattr(lab_step, _MARK, True)
    trainer.grow = lab_grow
    MuonAdamW.step = lab_step
    trainer._Run._train_step = lab_train_step
    try:
        yield t
    finally:
        trainer.grow = _REAL_GROW
        MuonAdamW.step = real_step
        trainer._Run._train_step = _REAL_TRAIN_STEP
    assert_unpatched()


# -- runs -------------------------------------------------------------------------------------


@dataclass
class LabRun:
    name: str
    arm: str
    seed: int
    dir: Path
    cfg: TrainRunConfig
    token: str
    status: str
    steps: int
    seconds: float
    heldout_losses: dict[str, float]
    curve: list[list[float]]
    applied: list[int] | None
    reused: bool

    @property
    def heldout(self) -> float | None:
        return _heldout_mean(self.heldout_losses)


class Lab:
    """Trains the lab's runs under ``out/runs`` and measures them (see the module docstring)."""

    def __init__(
        self, root: Path, out: Path, device: torch.device, scale: sc.CardScale, bench: bool
    ) -> None:
        self.root, self.out, self.device, self.scale, self.bench = root, out, device, scale, bench
        self.identity = data_identity(root)
        self.code = source_digest(TRAINING_SOURCES)
        self.tok = Tok.load(tokenizer_path(root))
        self._instruments = None

    def config(self, name: str, shape: ModelShape, seed: int) -> TrainRunConfig:
        """The config of a run of ``shape`` (a scorecard shape), the scorecard's token budget and
        even mix, built by the scorecard's own ``_Card.config`` at this lab's scale."""
        step = sc.Step(name, shape, sc.STEP_TOKENS, sc.BALANCED_MIX, seed=seed)
        return sc._Card.config(SimpleNamespace(scale=self.scale), step)

    def shape(self, shape: ModelShape) -> ModelShape:
        """The shape that the scorecard shape ``shape`` trains at on this lab's scale."""
        return self.scale.shapes[shape] if self.scale.shapes is not None else shape

    def train(
        self,
        name: str,
        arm: str,
        seed: int,
        cfg: TrainRunConfig,
        parent: LabRun | None,
        scales: tuple[float, float] | None = None,
    ) -> LabRun:
        """The finished run ``name`` of ``arm`` and ``seed``; reused only when its record's key
        (which names the arm, the seed, the exact parent training and the scales) matches."""
        _require(cfg.run_id == name, f"run {name} has the run id {cfg.run_id}")
        out = self.out / RUNS_DIR / name
        if parent is not None:
            cfg = replace(cfg, parent_dir=str(parent.dir))
        config = json.loads(cfg.to_json())
        config["parent_dir"] = parent.name if parent is not None else None
        transfer = None
        if scales is not None:
            transfer = {"old_scale": scales[0], "new_scale": scales[1], "share": TRANSFER_SHARE}
        key = _plain(
            {
                "lab": LAB_VERSION,
                "name": name,
                "arm": arm,
                "seed": seed,
                "parent": parent.token if parent is not None else None,
                "parent_name": parent.name if parent is not None else None,
                "config": config,
                "data": self.identity,
                "code": self.code,
                "device": self.device.type,
                "transfer": transfer,
            }
        )
        record = _read_json(out / RECORD_FILE)
        reusable = (
            isinstance(record, dict)
            and record.get("key") == key
            and record.get("status") in ("completed", "unstable_stopped")
            and (out / WEIGHTS_NAME).is_file()
            and (out / META_NAME).is_file()
        )
        if reusable:
            rk = record["key"]
            _require(
                rk["arm"] == arm
                and rk["seed"] == seed
                and rk["name"] == name
                and rk["parent"] == (parent.token if parent is not None else None)
                and rk["transfer"] == key["transfer"],
                f"{name}: the cached record does not name this arm, seed, parent and scales",
            )
            print(f"  {name}: finished earlier, reusing it", flush=True)
        else:
            record = self._train_fresh(name, cfg, out, key, parent, scales)
        run = LabRun(
            name=name,
            arm=arm,
            seed=seed,
            dir=out,
            cfg=cfg,
            token=record["token"],
            status=record["status"],
            steps=record["result"]["steps"],
            seconds=record["result"]["wall_seconds"],
            heldout_losses=record["result"]["heldout_losses"],
            curve=record["curve"],
            applied=record["applied"],
            reused=reusable,
        )
        self._check_lineage(run, parent)
        return run

    def _train_fresh(
        self,
        name: str,
        cfg: TrainRunConfig,
        out: Path,
        key: dict,
        parent: LabRun | None,
        scales: tuple[float, float] | None,
    ) -> dict:
        _write_json(out / RECORD_FILE, {"key": key, "status": "started"})
        print(
            f"  {name}: training ({cfg.steps} steps, {cfg.token_budget:,} tokens, "
            f"on {self.device.type})",
            flush=True,
        )
        curve: list[list[float]] = []
        unstable: set[int] = set()
        marks = [cfg.steps * q // 4 for q in (1, 2, 3)]

        def on_event(event: TrainEvent) -> None:
            if isinstance(event, HeldoutEval):
                mean = _heldout_mean(event.losses)
                curve.append([event.step, mean])
            elif isinstance(event, Instability):
                unstable.add(event.step)
                print(f"  {name}: unstable at step {event.step} ({event.action})", flush=True)
            elif isinstance(event, Progress) and marks and event.step >= marks[0]:
                while marks and event.step >= marks[0]:
                    marks.pop(0)
                print(f"  {name}: step {event.step}/{event.total_steps}", flush=True)

        applied = None
        if scales is None:
            assert_unpatched()
            result = trainer.train_run(
                cfg, out_dir=out, data_root=self.root, device=self.device, on_event=on_event
            )
        else:
            _require(parent is not None, f"{name}: a wrapped run needs a parent")
            t = Transfer(
                old_scale=scales[0],
                new_scale=scales[1],
                parent_shape=parent.cfg.shape,
                child=cfg,
                active_steps=cfg.steps // TRANSFER_SHARE,
            )
            with transfer_wrapper(t):
                result = trainer.train_run(
                    cfg, out_dir=out, data_root=self.root, device=self.device, on_event=on_event
                )
            _require(t.grow_calls == 1, f"{name}: the child was grown {t.grow_calls} times")
            last = min(t.active_steps, result.steps)
            expected = [s for s in range(1, last + 1) if s not in unstable]
            _require(
                t.applied == expected,
                f"{name}: the wrapper acted on steps {t.applied}, expected {expected}",
            )
            applied = t.applied
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
            "curve": curve,
            "applied": applied,
        }
        _write_json(out / RECORD_FILE, record)
        record = _plain(record)
        mean = _heldout_mean(result.heldout_losses)
        print(
            f"  {name}: {result.status} in {result.wall_seconds:.1f} s, held-out loss "
            f"{'n/a' if mean is None else f'{mean:.3f}'}",
            flush=True,
        )
        return record

    def _check_lineage(self, run: LabRun, parent: LabRun | None) -> None:
        """The checkpoint is this run's, at its shape, and continues exactly ``parent``."""
        meta = CheckpointMeta.from_json((run.dir / META_NAME).read_text(encoding="utf-8"))
        _require(meta.version_id == run.name, f"{run.name}: the checkpoint is {meta.version_id}")
        _require(meta.shape == run.cfg.shape, f"{run.name}: the checkpoint is a {meta.shape}")
        if parent is None:
            _require(meta.parent_version_id is None, f"{run.name}: should have no parent")
            _require(len(meta.runs) == 1, f"{run.name}: should be a first generation")
            return
        pmeta = CheckpointMeta.from_json((parent.dir / META_NAME).read_text(encoding="utf-8"))
        _require(
            meta.parent_version_id == parent.name
            and meta.lineage_id == pmeta.lineage_id
            and meta.runs[:-1] == pmeta.runs,
            f"{run.name}: the checkpoint does not continue {parent.name}",
        )

    def measure(self, run: LabRun) -> dict[str, float | None]:
        """The held-out mean, the yardstick loss, the training seconds and, with ``--bench``,
        bench-v1 overall of ``run``; the last two measurements are cached under the run's token."""
        key = {
            "run": run.token,
            "code": source_digest(sc.CARD_SOURCES),
            "yardstick": self.identity,
            "rows": sc.YARDSTICK_SEQS,
        }
        folder = self.out / MEASURED_DIR
        model = None

        def model_() -> Transformer:
            nonlocal model
            if model is None:
                model, _ = load_checkpoint(run.dir, self.device)
            return model

        value = _cached(
            folder / f"{run.name}.json",
            key,
            lambda: {"yardstick_loss": sc.yardstick_loss(model_(), self.root)},
        )
        measures = {
            "heldout": run.heldout,
            "yardstick_loss": value["yardstick_loss"],
            "seconds": run.seconds,
        }
        if self.bench:
            judge, novelty = self.instruments()
            bench_key = {
                **key,
                "suite": SUITE_VERSION,
                "seed": sc.MEASURE_SEED,
                "judge": asdict(judge.calibration) if judge.calibration is not None else None,
                "novelty": len(novelty),
            }

            def bench() -> dict:
                print(f"  {run.name}: bench", flush=True)
                report = run_benchmarks(
                    TorchLM(model_(), self.tok, self.device),
                    self.tok,
                    suite=build_suite(),
                    judge=judge,
                    novelty=novelty,
                    max_items_per_category=None,
                    seed=sc.MEASURE_SEED,
                )
                return {"bench_overall": report.overall}

            measures["bench_overall"] = _cached(
                folder / f"{run.name}.bench.json", bench_key, bench
            )["bench_overall"]
        return measures

    def instruments(self):
        if self._instruments is None:
            self._instruments = (
                _load_judge(self.root, self.device, self.identity),
                _load_novelty(self.root, self.identity),
            )
        return self._instruments


# -- self-checks ------------------------------------------------------------------------------


FUNCTION_ROWS = 2  # Self-check 1 compares logits on this many held-out rows of each dataset.


def _rows(root: Path, ctx_len: int) -> Tensor:
    """The first rows of the trainer's held-out batch of every dataset (real text)."""
    batches = trainer._heldout_batches(root, ctx_len)
    _require(bool(batches), f"no held-out rows in {root}")
    return torch.cat([torch.from_numpy(x[:FUNCTION_ROWS]) for x, _ in batches.values()])


@torch.no_grad()
def check_function(lab: Lab, parent: LabRun, child: TrainRunConfig) -> float:
    """Self-check 1; returns the largest logit difference."""
    model, _ = load_checkpoint(parent.dir, "cpu")
    model.eval()
    grown = growth.grow(model, child.shape, seed=child.seed).eval()
    x = _rows(lab.root, model.shape.ctx_len)
    diff = float((grown(x) - model(x)).abs().max())
    _require(
        diff <= FUNCTION_TOLERANCE,
        f"growth of {parent.name} is not function-preserving: logits differ by {diff:.2e}",
    )
    print(f"  self-check 1 ({parent.name}): grown logits equal the parent's (max diff {diff:.1e})")
    return diff


def check_one_step(parent: LabRun, child: TrainRunConfig) -> None:
    """Self-check 2b: one optimizer step with the B and C scales against a plain step from the
    same weights and gradients; copied elements must move ``old_scale`` times and new ones
    ``new_scale`` times the plain update."""
    model, _ = load_checkpoint(parent.dir, "cpu")
    grown = growth.grow(model, child.shape, seed=child.seed)
    old = copied_elements(model, grown)
    style = learning_style(child.boldness)
    start = {k: v.clone() for k, v in grown.state_dict().items()}
    gen = torch.Generator().manual_seed(0)
    grads = {name: torch.randn(p.shape, generator=gen) for name, p in grown.named_parameters()}

    def stepped(scales: tuple[float, float] | None) -> dict[str, Tensor]:
        net = growth.grow(model, child.shape, seed=child.seed)
        net.load_state_dict(start)
        opt = trainer._make_optimizer(net, style)
        for group in opt.param_groups:
            group["lr"] = group["peak_lr"]
        named = dict(net.named_parameters())
        for name, p in named.items():
            p.grad = grads[name].clone()
        if scales is None:
            opt.step()
        else:
            masks = scale_masks(old, *scales)
            apply_scaled([(p, masks[n]) for n, p in named.items()], opt.step)
        return {n: p.detach().clone() for n, p in named.items()}

    plain = stepped(None)
    for arm, (old_scale, new_scale) in TRANSFER_SCALES.items():
        scaled = stepped((old_scale, new_scale))
        for name, mask in old.items():
            before = start[name]
            delta, want = scaled[name] - before, plain[name] - before
            for part, factor in ((mask, old_scale), (~mask, new_scale)):
                if factor == 1.0:
                    _require(
                        torch.equal(scaled[name][part], plain[name][part]),
                        f"self-check 2b ({arm}): {name} moved where its scale is 1",
                    )
                else:
                    _require(
                        # The absolute tolerance covers the rounding of a weight near 1 plus a
                        # small scaled update, minus the weight again.
                        torch.allclose(delta[part], factor * want[part], rtol=1e-4, atol=1e-6),
                        f"self-check 2b ({arm}): {name} did not move {factor} times the update",
                    )
            _require(bool((want != 0).any()), f"self-check 2b: the plain step did not move {name}")
    print("  self-check 2b: one step with the B and C scales moves old and new elements as set")


def check_identity(lab: Lab, parent: LabRun, child: TrainRunConfig) -> None:
    """Self-check 2: 8 steps wrapped with both scales 1.0 equal 8 plain steps, bit for bit."""
    budget = CHECK_STEPS * child.batch_tokens
    cfg = replace(child, run_id="selfcheck", token_budget=budget, parent_dir=str(parent.dir))
    folder = lab.out / CHECK_DIR
    shutil.rmtree(folder, ignore_errors=True)
    cpu = torch.device("cpu")
    assert_unpatched()
    trainer.train_run(cfg, out_dir=folder / "plain", data_root=lab.root, device=cpu)
    t = Transfer(1.0, 1.0, parent.cfg.shape, cfg, active_steps=cfg.steps // TRANSFER_SHARE)
    with transfer_wrapper(t):
        trainer.train_run(cfg, out_dir=folder / "wrapped", data_root=lab.root, device=cpu)
    _require(
        t.grow_calls == 1 and t.applied == list(range(1, cfg.steps // TRANSFER_SHARE + 1)),
        f"self-check 2: the wrapper acted on steps {t.applied}, not the first third",
    )
    a, _ = load_checkpoint(folder / "plain", "cpu")
    b, _ = load_checkpoint(folder / "wrapped", "cpu")
    sa, sb = a.state_dict(), b.state_dict()
    _require(sa.keys() == sb.keys(), "self-check 2: the two runs have different parameters")
    different = [k for k in sa if not torch.equal(sa[k], sb[k])]
    _require(not different, f"self-check 2: scales of 1.0 changed the training of {different[:5]}")
    shutil.rmtree(folder, ignore_errors=True)
    print(
        f"  self-check 2: {cfg.steps} steps wrapped with scales 1.0 (acting on steps "
        f"{t.applied}) equal the plain steps bit for bit"
    )


def check_counts(parent: LabRun, child: TrainRunConfig) -> dict[str, int]:
    """Self-check 3: how many elements of the grown model are old and how many new."""
    model, _ = load_checkpoint(parent.dir, "cpu")
    grown = growth.grow(model, child.shape, seed=child.seed)
    counts = element_counts(copied_elements(model, grown))
    _require(
        counts["total"] == grown.num_params() and counts["old"] == model.num_params(),
        f"self-check 3: counts {counts} do not match the models' sizes",
    )
    print(
        f"  self-check 3: {model.shape.n_layer}x{model.shape.d_model} -> "
        f"{grown.shape.n_layer}x{grown.shape.d_model}: {counts['old']:,} elements old, "
        f"{counts['new']:,} new ({counts['total']:,} in all)"
    )
    return counts


# -- the experiment ---------------------------------------------------------------------------


def _check_recipes() -> None:
    """The scorecard's recipes this lab copies have not moved."""
    first = sc.RECIPES["first"][0]
    lineage = sc.RECIPES["lineage"][0]
    _require(
        (first.shape, first.tokens, dict(first.mix), first.seed)
        == (sc.TINY, sc.STEP_TOKENS, dict(sc.BALANCED_MIX), 1),
        "the scorecard's first AI is no longer Tiny, STEP_TOKENS, the even mix, seed 1",
    )
    _require(
        (lineage.shape, lineage.tokens, dict(lineage.mix), lineage.seed, lineage.parent)
        == (
            sc.SMALL,
            sc.STEP_TOKENS,
            dict(sc.BALANCED_MIX),
            1 + CONTINUATION_SEED_OFFSET,
            "first-s1",
        ),
        "the scorecard's lineage-small no longer continues first-s1 as this lab assumes",
    )


def run_lab(lab: Lab, seeds: list[int], arms: list[str]) -> dict:
    _check_recipes()
    tiny, small = lab.shape(sc.TINY), lab.shape(sc.SMALL)
    finals: dict[str, dict[int, LabRun]] = {arm: {} for arm in arms}
    checks: dict = {}
    for seed in seeds:
        child_seed = seed + CONTINUATION_SEED_OFFSET
        print(f"seed {seed}", flush=True)
        parent = None
        if any(arm in arms for arm in ("A", "B", "C")):
            name = f"parent-tiny-s{seed}"
            parent = lab.train(name, "parent", seed, lab.config(name, sc.TINY, seed), None)
            probe = lab.config(f"A-s{seed}", sc.SMALL, child_seed)
            _require(parent.cfg.shape == tiny and probe.shape == small, "unexpected shapes")
            checks.setdefault("function_max_diff", {})[str(seed)] = check_function(
                lab, parent, probe
            )
            check_one_step(parent, probe)
            if "elements" not in checks:
                checks["elements"] = check_counts(parent, probe)
                check_identity(lab, parent, probe)
        for arm in arms:
            name = f"{arm}-s{seed}"
            if arm in ("A", "B", "C"):
                cfg = lab.config(name, sc.SMALL, child_seed)
                finals[arm][seed] = lab.train(
                    name, arm, seed, cfg, parent, TRANSFER_SCALES.get(arm)
                )
            elif arm == "D":
                first = f"D-parent-small-s{seed}"
                new_small = lab.train(first, "D", seed, lab.config(first, sc.SMALL, seed), None)
                cfg = lab.config(name, sc.SMALL, child_seed)
                finals[arm][seed] = lab.train(name, arm, seed, cfg, new_small)
            else:
                cfg = lab.config(name, sc.SMALL, child_seed)
                finals[arm][seed] = lab.train(name, arm, seed, cfg, None)
    models: dict[str, dict[str, dict]] = {}
    for arm in arms:
        models[arm] = {}
        for seed, run in finals[arm].items():
            entry = {"run": run.name, "status": run.status, **lab.measure(run)}
            if arm in ("A", "B", "C"):
                entry["curve"] = run.curve
            if run.applied is not None:
                entry["transfer_steps"] = run.applied
            models[arm][str(seed)] = entry
    return {
        "lab": LAB_VERSION,
        "data_root": str(lab.root),
        "device": lab.device.type,
        "scale": lab.scale.name,
        "seeds": seeds,
        "arms": arms,
        "transfer_scales": {arm: list(s) for arm, s in TRANSFER_SCALES.items()},
        "transfer_share": f"1/{TRANSFER_SHARE}",
        "self_checks": checks,
        "models": models,
        "summary": summarise(models, seeds),
    }


def _measures(models: dict) -> list[str]:
    names = ["heldout", "yardstick_loss", "seconds"]
    if any("bench_overall" in e for per in models.values() for e in per.values()):
        names.append("bench_overall")
    return names


def summarise(models: dict, seeds: list[int]) -> dict:
    out: dict = {}
    for measure in _measures(models):
        out[measure] = {}
        for arm, per in models.items():
            values = [per[str(s)].get(measure) for s in seeds]
            finite = [v for v in values if isinstance(v, int | float) and math.isfinite(v)]
            mean = math.fsum(finite) / len(finite) if len(finite) == len(values) else None
            diff = None
            if "A" in models and mean is not None:
                base = [models["A"][str(s)].get(measure) for s in seeds]
                if all(isinstance(v, int | float) for v in base):
                    diff = mean - math.fsum(base) / len(base)
            out[measure][arm] = {"per_seed": values, "mean": mean, "minus_A": diff}
    return out


def _fmt(value, measure: str) -> str:
    if value is None:
        return "n/a"
    if measure == "seconds":
        return f"{value:.1f}"
    if measure == "bench_overall":
        return f"{value:.1f}"
    return f"{value:.4f}"


def print_tables(results: dict) -> None:
    seeds, summary, models = results["seeds"], results["summary"], results["models"]
    for measure, per_arm in summary.items():
        print(f"\n{measure}: per seed ({', '.join(f's{s}' for s in seeds)}) | mean | minus A")
        for arm, row in per_arm.items():
            values = " ".join(_fmt(v, measure) for v in row["per_seed"])
            diff = "n/a" if row["minus_A"] is None else f"{row['minus_A']:+.4f}"
            if measure in ("seconds", "bench_overall") and row["minus_A"] is not None:
                diff = f"{row['minus_A']:+.1f}"
            print(f"  {arm}  {values} | {_fmt(row['mean'], measure)} | {diff}")
    curves = [arm for arm in ("A", "B", "C") if arm in models]
    for seed in seeds:
        if not curves:
            break
        print(f"\nheld-out mean during training, seed {seed} (step: {' '.join(curves)})")
        by_arm = {arm: {int(s): m for s, m in models[arm][str(seed)]["curve"]} for arm in curves}
        for step in sorted(set().union(*by_arm.values())):
            row = " ".join(_fmt(by_arm[arm].get(step), "heldout") for arm in curves)
            print(f"  {step:>5}: {row}")
    if "A" in models:
        print(
            f"\nheld-out loss better than A by at least {KEEP_MARGIN} on every seed "
            f"(lower is better):"
        )
        for arm in ("B", "C", "D", "E"):
            if arm not in models:
                continue
            gains = [
                models["A"][str(s)]["heldout"] - models[arm][str(s)]["heldout"]
                if models["A"][str(s)]["heldout"] is not None
                and models[arm][str(s)]["heldout"] is not None
                else None
                for s in seeds
            ]
            ok = all(g is not None and g >= KEEP_MARGIN for g in gains)
            shown = " ".join("n/a" if g is None else f"{g:+.4f}" for g in gains)
            line = f"  {arm}: A minus {arm} per seed {shown}: {'yes' if ok else 'no'}"
            if "bench_overall" in summary:
                bench = summary["bench_overall"][arm]["minus_A"]
                line += f"; bench mean minus A {'n/a' if bench is None else f'{bench:+.1f}'}"
            print(line)


def _outside_data(out: Path) -> bool:
    resolved = out.resolve()
    for name in ("data", "data2"):
        if resolved.is_relative_to((REPO_ROOT / name).resolve()):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--tiny-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error("seeds and arms must not repeat")
    if not _outside_data(args.out):
        parser.error("--out must not be inside data/ or data2/")
    arms = [arm for arm in ARMS if arm in args.arms]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = args.tiny_root if args.tiny_root is not None else default_data_root()
    scale = TINY_SCALE if args.tiny_root is not None else sc.FULL
    lab = Lab(Path(root), args.out, device, scale, args.bench)
    print(f"growth lab: arms {' '.join(arms)}, seeds {args.seeds}, {device.type}, data {root}")
    with stay_awake():
        results = run_lab(lab, list(args.seeds), arms)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / RESULTS_FILE).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print_tables(results)
    print(f"\nwritten to {args.out / RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
