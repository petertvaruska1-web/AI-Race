"""The trainer: one training run that turns a player's choices into a new model version.

A run continues from the parent checkpoint (grown first, function-preserving, when the shape got
bigger), trains on the player's data mix blended with a replay share of the parent's mix, and
switches to the finishing mix for the decay phase of the learning-rate schedule. It streams
telemetry as it goes (training loss, held-out loss per dataset, and samples from probe prompts) so
the player watches their AI learn.

Instability is real. A loss spike, a non-finite loss or a non-finite gradient rolls the weights
and optimizer state back to the last in-memory snapshot (the bad batch's update is never
applied), halves the learning rate and carries on. A snapshot only ever holds weights that a
later step has shown to give a good loss and gradient. The third spike stops the run and keeps
those last good weights. The step counter and the sampler never rewind: a run always moves
forward through its data.

A run can be cut short and resumed exactly. ``out_dir/resume/`` holds everything the remaining
steps depend on, and a resumed run ends with the same weights as an uninterrupted one.

``out_dir`` ends up with the checkpoint (``model.safetensors`` and ``meta.json``),
``telemetry.jsonl`` (one strict-JSON event per line) and ``result.json``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.prep import heldout_docs
from airace_ml.data.sampler import DocPool, MixtureSampler, load_pools
from airace_ml.device import pick_device
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import META_NAME, CheckpointMeta, load_checkpoint, save_checkpoint
from airace_ml.model.growth import grow
from airace_ml.model.transformer import Transformer
from airace_ml.paths import TOKENIZER_VERSION, corpus_dir, tokenizer_path
from airace_ml.paths import data_root as default_data_root
from airace_ml.tokenizer import Tok, encode_chat, encode_doc
from airace_ml.train.config import TrainRunConfig, effective_mixture, learning_style
from airace_ml.train.events import (
    Done,
    HeldoutEval,
    Instability,
    Progress,
    Sample,
    TrainEvent,
    event_to_dict,
    json_safe,
)
from airace_ml.train.schedule import DECAY_FRAC, wsd_lr
from airace_ml.train.stability import SpikeDetector

PROBE_PROMPTS: tuple[tuple[Literal["doc", "chat"], str], ...] = (
    ("doc", "Once upon a time"),
    ("chat", "Hello! How are you?"),
    ("doc", "The capital of France is"),
)

SNAPSHOT_EVERY = 25  # steps between the in-memory snapshots a spike rolls back to
MAX_SPIKES = 3  # the third spike stops the run
PROGRESS_EVERY = 10
MIN_EVAL_EVERY = 25  # held-out eval and samples every max(25, steps // 20) steps
EVAL_SEQS = 8  # held-out sequences per dataset
SAMPLE_TOKENS = 32
SAMPLE_TEMPERATURE = 0.8
FINAL_LOSS_WINDOW = 10  # final_loss is the mean loss of the last 10 applied steps
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.1

TELEMETRY_NAME = "telemetry.jsonl"
RESULT_NAME = "result.json"
RESUME_DIR = "resume"
_RESUME_INFO = "state.json"
_RESUME_TENSORS = "state.pt"
_TMP_TAG = ".tmp."  # resume.tmp.<id>: a save being written, or one whose swap was refused
_OLD_TAG = ".old."  # resume.old.<id>: the previous state, moved aside during a swap

RunStatus = Literal["completed", "unstable_stopped", "interrupted"]


@dataclass
class TrainHooks:
    """Test-only levers.

    ``loss_override(step, loss)`` replaces the logged loss, and with it what the spike detector
    sees (the model still learns from its real loss). ``after_backward(step, model)`` runs
    between the backward pass and gradient clipping, e.g. to corrupt the gradients.
    ``stop_after_steps`` interrupts the run after that step, leaving resume state behind.
    """

    loss_override: Callable[[int, float], float] | None = None
    after_backward: Callable[[int, Transformer], None] | None = None
    stop_after_steps: int | None = None


@dataclass
class TrainResult:
    run_id: str
    status: RunStatus
    steps: int  # steps run (a stopped or interrupted run ends early)
    tokens: int  # tokens drawn by this run (rolled-back steps included)
    final_loss: float
    heldout_losses: dict[str, float]  # from the last held-out eval
    tokens_per_dataset: dict[str, int]
    out_dir: Path
    meta: CheckpointMeta
    wall_seconds: float


def train_run(
    cfg: TrainRunConfig,
    *,
    out_dir: Path,
    data_root: Path | None = None,
    device: torch.device | None = None,
    on_event: Callable[[TrainEvent], None] | None = None,
    resume: bool = False,
    checkpoint_every: int = 200,
    _hooks: TrainHooks | None = None,
) -> TrainResult:
    """Train ``cfg`` into ``out_dir`` and return what came out.

    Everything that can be wrong with the request (the config, the data mix, the parent and its
    growth, the resume state) is checked before ``out_dir`` is touched, so a failed request
    leaves nothing behind. Every ``checkpoint_every`` steps the resume state is saved to
    ``out_dir/resume/``; ``resume=True`` continues from there and the directory is deleted once
    the run finishes. ``device=None`` picks one with :func:`~airace_ml.device.pick_device`.
    """
    started = time.perf_counter()
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int):
        raise TypeError(f"checkpoint_every must be an integer, got {checkpoint_every!r}")
    if checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be at least 1, got {checkpoint_every}")
    out_dir = Path(out_dir)
    setup = _prepare(cfg, Path(data_root) if data_root is not None else default_data_root())
    saved = _read_resume(out_dir, cfg) if resume else None
    run = _Run(
        cfg,
        setup,
        device=torch.device(device) if device is not None else pick_device(),
        out_dir=out_dir,
        on_event=on_event,
        hooks=_hooks or TrainHooks(),
        checkpoint_every=checkpoint_every,
        started=started,
    )
    return run.execute(saved)


# -- validation and setup (no side effects) -------------------------------------------------------


@dataclass
class _Setup:
    """Everything a run needs that can be built and checked before it touches the disk."""

    tok: Tok
    sampler: MixtureSampler  # set to the stable mixture
    stable_mixture: dict[str, float]  # normalized, replay included
    finishing_mixture: dict[str, float] | None
    parent: CheckpointMeta | None
    model: Transformer  # on the CPU: fresh, the parent, or the parent grown
    heldout: dict[str, tuple[np.ndarray, np.ndarray]]  # fixed eval batch per base dataset


def _prepare(cfg: TrainRunConfig, root: Path) -> _Setup:
    cfg.validate()
    tok = Tok.load(tokenizer_path(root))
    custom = _custom_docs(cfg, tok)
    parent = _read_meta(Path(cfg.parent_dir)) if cfg.parent_dir is not None else None
    stable = effective_mixture(cfg.mixture, cfg.replay, parent.last_mixture if parent else None)
    finishing = dict(cfg.finishing_mixture) if cfg.finishing_mixture is not None else None
    wanted = dict(stable)  # load a pool for every dataset either phase draws from
    for name, w in (finishing or {}).items():
        wanted[name] = max(wanted.get(name, 0.0), w)
    pools = load_pools(root, wanted, cfg.prep, cfg.purchases, custom)
    sampler = MixtureSampler(pools, stable, cfg.shape.ctx_len, cfg.seed)
    if finishing is not None:
        sampler.set_weights(finishing)  # fail now rather than at the decay start
        sampler.set_weights(stable)
    model = _start_model(cfg, tok, parent)
    heldout = _heldout_batches(root, cfg.shape.ctx_len)
    return _Setup(tok, sampler, stable, finishing, parent, model, heldout)


def _custom_docs(cfg: TrainRunConfig, tok: Tok) -> dict[str, list[list[int]]]:
    """The player's notebook and coaching as documents; blank entries teach nothing and go."""
    notebook = [encode_doc(tok, text) for entry in cfg.notebook if (text := entry.strip())]
    coaching = [
        encode_chat(tok, [("user", prompt), ("ai", reply)])
        for prompt, reply in ((p.strip(), r.strip()) for p, r in cfg.coaching)
        if prompt and reply
    ]
    return {"notebook": notebook, "coaching": coaching}


def _read_meta(ckpt_dir: Path) -> CheckpointMeta:
    return CheckpointMeta.from_json((ckpt_dir / META_NAME).read_text(encoding="utf-8"))


def _start_model(cfg: TrainRunConfig, tok: Tok, parent: CheckpointMeta | None) -> Transformer:
    """A fresh model seeded by ``cfg.seed``, or the parent, reshaped to ``cfg.shape`` if needed."""
    if parent is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            return Transformer(cfg.shape, tok.vocab_size)
    model, _ = load_checkpoint(Path(cfg.parent_dir))
    if model.tok_emb.num_embeddings != tok.vocab_size:
        raise ValueError(
            f"parent model has a {model.tok_emb.num_embeddings}-token vocabulary, the tokenizer "
            f"has {tok.vocab_size}"
        )
    # grow() raises GrowthError for fewer layers or a narrower width; the attention span may go
    # up or down freely, since it is not a parameter shape (spec 4.3).
    return model if cfg.shape == model.shape else grow(model, cfg.shape, seed=cfg.seed)


def _heldout_batches(root: Path, seq_len: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``EVAL_SEQS`` held-out sequences per base dataset, the same at every eval of the run.

    Each dataset draws with its own rng (seed 0), so its eval batch does not depend on which
    other datasets are present. Datasets missing from ``root`` (or with no held-out text) are
    skipped.
    """
    batches = {}
    for name in DATASET_IDS:
        path = corpus_dir(root) / name
        if not path.is_dir():
            continue
        corpus = Corpus.open(path)
        pool = DocPool.from_corpus(corpus, heldout_docs(corpus), balanced=False)
        if len(pool):
            sampler = MixtureSampler({name: pool}, {name: 1.0}, seq_len, seed=0)
            batches[name] = sampler.next_batch(EVAL_SEQS)
    return batches


def _resume_states(out_dir: Path) -> list[Path]:
    """Every resume-state directory in ``out_dir``, complete or not."""
    if not out_dir.is_dir():
        return []
    tags = (f"{RESUME_DIR}{_TMP_TAG}", f"{RESUME_DIR}{_OLD_TAG}")
    return [
        path
        for path in out_dir.iterdir()
        if path.is_dir() and (path.name == RESUME_DIR or path.name.startswith(tags))
    ]


def _remove_resume_states(out_dir: Path, keep: tuple[Path, ...] = ()) -> None:
    """Best-effort delete; a directory a scanner still holds open is left for the next try."""
    for path in _resume_states(out_dir):
        if path not in keep:
            shutil.rmtree(path, ignore_errors=True)


def _read_resume(out_dir: Path, cfg: TrainRunConfig) -> tuple[dict, dict]:
    """The newest complete resume state (JSON part, tensor part), checked against ``cfg``.

    Normally that is ``out_dir/resume/``. After a crash or a refused rename in the middle of a
    save it may sit under a ``resume.tmp.*`` or ``resume.old.*`` name instead. A directory whose
    ``state.json`` (written last) is missing or unreadable is incomplete and ignored.
    """
    best: tuple[Path, dict] | None = None
    for path in _resume_states(out_dir):
        try:
            info = json.loads((path / _RESUME_INFO).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        best_step = best[1]["step"] if best is not None else -1
        if info["step"] > best_step or (info["step"] == best_step and path.name == RESUME_DIR):
            best = (path, info)
    if best is None:
        raise FileNotFoundError(f"no resume state in {out_dir / RESUME_DIR}")
    path, info = best
    if info["config"] != json.loads(cfg.to_json()):
        raise ValueError(f"the resume state in {path} belongs to a different run config")
    tensors = torch.load(path / _RESUME_TENSORS, map_location="cpu", weights_only=True)
    return info, tensors


# -- training -------------------------------------------------------------------------------------


def _decay_start(steps: int) -> int:
    """First step (0-based) of the schedule's decay phase, exactly as :func:`wsd_lr` places it."""
    return steps - max(1, int(DECAY_FRAC * steps))


def _make_optimizer(model: Transformer, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices and embeddings only (not on norm gains)."""
    params = list(model.parameters())  # the tied head appears once
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=0.0,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        fused=True if device.type == "cuda" else None,
    )


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _cpu_copy(tree):
    """A deep copy of a nested state dict with every tensor copied to the CPU."""
    if isinstance(tree, Tensor):
        return tree.detach().to("cpu", copy=True)
    if isinstance(tree, dict):
        return {key: _cpu_copy(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_cpu_copy(value) for value in tree)
    return tree


@dataclass
class _Snapshot:
    """CPU copies of the weights and optimizer state at the last snapshot step."""

    model: dict
    optimizer: dict

    @classmethod
    def take(cls, model: Transformer, optimizer: torch.optim.Optimizer) -> _Snapshot:
        return cls(_cpu_copy(model.state_dict()), _cpu_copy(optimizer.state_dict()))

    def restore(self, model: Transformer, optimizer: torch.optim.Optimizer) -> None:
        model.load_state_dict(self.model)
        # Optimizer.load_state_dict keeps CPU tensors as given and later steps update them in
        # place, so it gets a copy: the snapshot must survive for the next rollback.
        optimizer.load_state_dict(_cpu_copy(self.optimizer))


@torch.inference_mode()
def _heldout_losses(
    model: Transformer, batches: dict[str, tuple[Tensor, Tensor]], device: torch.device
) -> dict[str, float]:
    """Mean next-token loss on each dataset's fixed held-out batch."""
    was_training = model.training
    model.eval()
    try:
        losses = []
        for x, y in batches.values():
            with _autocast(device):
                logits = model(x)
            losses.append(F.cross_entropy(logits.float().flatten(0, 1), y.flatten()))
    finally:
        model.train(was_training)
    values = torch.stack(losses).tolist() if losses else []
    return dict(zip(batches, values, strict=True))


class _Run:
    """One run's training loop and the state it carries from step to step."""

    def __init__(
        self,
        cfg: TrainRunConfig,
        setup: _Setup,
        *,
        device: torch.device,
        out_dir: Path,
        on_event: Callable[[TrainEvent], None] | None,
        hooks: TrainHooks,
        checkpoint_every: int,
        started: float,
    ) -> None:
        self.cfg = cfg
        self.setup = setup
        self.device = device
        self.out_dir = out_dir
        self.on_event = on_event
        self.hooks = hooks
        self.checkpoint_every = checkpoint_every
        self.started = started

        self.sampler = setup.sampler
        self.model = setup.model.to(device).train()
        self.params = list(self.model.parameters())
        self.optimizer = _make_optimizer(self.model, device)
        self.lm = TorchLM(self.model, setup.tok)
        self.style = learning_style(cfg.boldness)
        self.steps = cfg.steps
        self.tokens_per_step = cfg.batch_size * cfg.shape.ctx_len
        self.decay_start = _decay_start(self.steps)
        self.detector_warmup = max(1, int(self.style.warmup_frac * self.steps))
        self.eval_every = max(MIN_EVAL_EVERY, self.steps // 20)
        self.probes = PROBE_PROMPTS + tuple(("chat", prompt) for prompt in cfg.probe_prompts)
        self.heldout_batches = {
            name: (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            for name, (x, y) in setup.heldout.items()
        }

        # Loop state; a resumed run loads all of it from the resume state.
        self.step = 0  # steps done
        self.lr_scale = 1.0
        self.spikes = 0
        self.recent_losses: deque[float] = deque(maxlen=FINAL_LOSS_WINDOW)
        self.heldout: dict[str, float] = {}
        self.detector = SpikeDetector(self.detector_warmup)
        self.snapshot: _Snapshot | None = None
        self.snapshot_is_current = False  # True while the weights equal the snapshot's
        self.lineage_id = setup.parent.lineage_id if setup.parent else uuid.uuid4().hex
        self.prior_seconds = 0.0  # wall time of earlier sessions of a resumed run
        self._telemetry = None

    # -- driver ---------------------------------------------------------------------------------

    def execute(self, saved: tuple[dict, dict] | None) -> TrainResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if saved is None:
            _remove_resume_states(self.out_dir)  # stale, from another run
            self.snapshot = _Snapshot.take(self.model, self.optimizer)
            self.snapshot_is_current = True
            mode = "w"
        else:
            self._load_resume(*saved)
            mode = "a"
        with open(self.out_dir / TELEMETRY_NAME, mode, encoding="utf-8", newline="\n") as f:
            self._telemetry = f
            return self._loop()

    def _loop(self) -> TrainResult:
        steps = self.steps
        status: RunStatus = "completed"
        while self.step < steps:
            i = self.step
            if self.setup.finishing_mixture is not None and i == self.decay_start:
                self.sampler.set_weights(self.setup.finishing_mixture)
                # A new data mix has a new loss level; relearn it rather than call it a spike.
                self.detector = SpikeDetector(i + self.detector_warmup)
            step = i + 1
            lr = wsd_lr(i, steps, self.style) * self.lr_scale
            loss, spike = self._train_step(step, lr)
            self.step = step
            if step == 1 or step % PROGRESS_EVERY == 0 or step == steps:
                self._emit(Progress(step, steps, self.tokens, loss, lr))
            if spike:
                self.spikes += 1
                self.snapshot.restore(self.model, self.optimizer)
                self.snapshot_is_current = True
                if self.spikes >= MAX_SPIKES:
                    self._emit(Instability(step, "stopped", self.lr_scale))
                    status = "unstable_stopped"
                    break
                self.lr_scale *= 0.5
                self._emit(Instability(step, "rollback", self.lr_scale))
            else:
                self.recent_losses.append(loss)
                self.snapshot_is_current = False  # the update moved the weights on
            if step % self.eval_every == 0 or step == steps:
                self._evaluate(step)
            interrupted = step == self.hooks.stop_after_steps and step < steps
            if interrupted or (step % self.checkpoint_every == 0 and step < steps):
                self._save_resume()
            if interrupted:
                return self._result("interrupted", self._meta("interrupted"))
        if status == "unstable_stopped":
            self._evaluate(self.step)  # the kept (last good) weights
        return self._finish(status)

    def _train_step(self, step: int, lr: float) -> tuple[float, bool]:
        """One batch: forward, backward and, unless the loss or gradient is a spike, the update."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        x, y = self.sampler.next_batch(self.cfg.batch_size)
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        with _autocast(self.device):
            logits = self.model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        loss.backward()
        if self.hooks.after_backward is not None:
            self.hooks.after_backward(step, self.model)
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.style.clip_norm)
        # The one host sync per step: spike detection needs both values.
        value, grad_norm = torch.stack([loss.detach().float(), norm.float()]).tolist()
        if self.hooks.loss_override is not None:
            value = self.hooks.loss_override(step, value)
        # A non-finite gradient (finite loss or not) would make the update NaN: it is a spike,
        # and it stays out of the detector's statistics.
        spike = not math.isfinite(grad_norm) or self.detector.update(step, value)
        if not spike:
            if step > 1 and step % SNAPSHOT_EVERY == 1:
                # The weights after the last multiple of 25 steps just gave a good loss and
                # gradient. Only weights validated like this become a rollback target.
                self.snapshot = _Snapshot.take(self.model, self.optimizer)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return value, spike

    # -- telemetry ------------------------------------------------------------------------------

    def _emit(self, event: TrainEvent) -> None:
        self._telemetry.write(json.dumps(event_to_dict(event), allow_nan=False) + "\n")
        self._telemetry.flush()
        if self.on_event is not None:
            self.on_event(event)

    def _evaluate(self, step: int) -> None:
        """Held-out loss per dataset, then one sample per probe prompt."""
        self.heldout = _heldout_losses(self.model, self.heldout_batches, self.device)
        self._emit(HeldoutEval(step, dict(self.heldout)))
        for kind, prompt in self.probes:
            options = {
                "max_new_tokens": SAMPLE_TOKENS,
                "temperature": SAMPLE_TEMPERATURE,
                "seed": step,
            }
            if kind == "doc":
                text = self.lm.complete(prompt, **options)
            else:
                text = self.lm.chat_reply([("user", prompt)], **options)
            self._emit(Sample(step, prompt, text))

    # -- resume state ---------------------------------------------------------------------------

    def _save_resume(self) -> None:
        """Write the resume state to a fresh directory, then swap it in for the old one.

        The old state is moved aside before the new one takes its name, and deleted only
        afterwards, so a complete state exists at every moment. If Windows refuses a rename (a
        virus scanner or indexer holding a file), training goes on: :func:`_read_resume` takes
        the newest complete state under any of the names, and the next save tries again.
        """
        final = self.out_dir / RESUME_DIR
        tmp = self.out_dir / f"{RESUME_DIR}{_TMP_TAG}{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        snapshot = None
        if not self.snapshot_is_current:
            snapshot = {"model": self.snapshot.model, "optimizer": self.snapshot.optimizer}
        tensors = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "snapshot": snapshot,  # None: the snapshot equals the saved weights
        }
        torch.save(tensors, tmp / _RESUME_TENSORS)
        self._telemetry.flush()
        info = {
            "config": json.loads(self.cfg.to_json()),
            "step": self.step,
            "lr_scale": self.lr_scale,
            "spikes": self.spikes,
            "recent_losses": list(self.recent_losses),
            "sampler": self.sampler.rng_state(),
            "detector": {"warmup_steps": self.detector.warmup_steps, **self.detector.state()},
            "lineage_id": self.lineage_id,
            "wall_seconds": self._wall_seconds(),
            "telemetry_bytes": (self.out_dir / TELEMETRY_NAME).stat().st_size,
        }
        # Written last: a state directory with a readable state.json is complete.
        (tmp / _RESUME_INFO).write_text(json.dumps(info, allow_nan=False), encoding="utf-8")
        with contextlib.suppress(OSError):
            if final.exists():
                final.rename(self.out_dir / f"{RESUME_DIR}{_OLD_TAG}{uuid.uuid4().hex[:12]}")
            tmp.rename(final)
        _remove_resume_states(self.out_dir, keep=(final, tmp))

    def _load_resume(self, info: dict, tensors: dict) -> None:
        self.model.load_state_dict(tensors["model"])
        self.optimizer.load_state_dict(tensors["optimizer"])
        saved_snapshot = tensors["snapshot"]
        if saved_snapshot is None:
            self.snapshot = _Snapshot.take(self.model, self.optimizer)
        else:
            self.snapshot = _Snapshot(saved_snapshot["model"], saved_snapshot["optimizer"])
        self.snapshot_is_current = saved_snapshot is None
        self.sampler.set_rng_state(info["sampler"])
        self.step = info["step"]
        if self.setup.finishing_mixture is not None and self.step >= self.decay_start:
            self.sampler.set_weights(self.setup.finishing_mixture)  # the rng state has no weights
        self.lr_scale = info["lr_scale"]
        self.spikes = info["spikes"]
        self.recent_losses.extend(info["recent_losses"])
        self.detector = SpikeDetector(info["detector"]["warmup_steps"])
        self.detector.load_state(info["detector"])
        self.lineage_id = info["lineage_id"]
        self.prior_seconds = info["wall_seconds"]
        # Events after the saved step are emitted again by this session.
        os.truncate(self.out_dir / TELEMETRY_NAME, info["telemetry_bytes"])

    # -- results --------------------------------------------------------------------------------

    @property
    def tokens(self) -> int:
        return self.step * self.tokens_per_step

    def _wall_seconds(self) -> float:
        return self.prior_seconds + (time.perf_counter() - self.started)

    def _final_loss(self) -> float:
        """Mean loss of the last applied steps; the held-out loss if no step was ever applied."""
        if self.recent_losses:
            return sum(self.recent_losses) / len(self.recent_losses)
        finite = [v for v in self.heldout.values() if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else 0.0

    def _summary(self, status: RunStatus) -> dict:
        """This run's entry in the lineage's run history."""
        return {
            "run_id": self.cfg.run_id,
            "steps": self.step,
            "tokens": self.tokens,
            "mixture": self.setup.stable_mixture,
            "prep": self.cfg.prep.to_dict(),
            "shape": self.cfg.shape.to_dict(),
            "status": status,
        }

    def _meta(self, status: RunStatus) -> CheckpointMeta:
        parent = self.setup.parent
        return CheckpointMeta(
            tokenizer=TOKENIZER_VERSION,
            shape=self.cfg.shape,
            lineage_id=self.lineage_id,
            version_id=self.cfg.run_id,
            parent_version_id=parent.version_id if parent else None,
            tokens_trained_total=(parent.tokens_trained_total if parent else 0) + self.tokens,
            last_mixture=dict(self.setup.stable_mixture),
            runs=[*(parent.runs if parent else []), self._summary(status)],
        )

    def _result(self, status: RunStatus, meta: CheckpointMeta) -> TrainResult:
        return TrainResult(
            run_id=self.cfg.run_id,
            status=status,
            steps=self.step,
            tokens=self.tokens,
            final_loss=self._final_loss(),
            heldout_losses=dict(self.heldout),
            tokens_per_dataset=self.sampler.stats(),
            out_dir=self.out_dir,
            meta=meta,
            wall_seconds=self._wall_seconds(),
        )

    def _finish(self, status: RunStatus) -> TrainResult:
        """Save the new model version and the result, drop the resume state, then emit Done."""
        meta = self._meta(status)
        save_checkpoint(self.model, meta, self.out_dir)
        result = self._result(status, meta)
        (self.out_dir / RESULT_NAME).write_text(_result_json(result), encoding="utf-8")
        _remove_resume_states(self.out_dir)
        summary = {
            **self._summary(status),
            "final_loss": result.final_loss,
            "heldout_losses": result.heldout_losses,
            "tokens_per_dataset": result.tokens_per_dataset,
        }
        self._emit(Done(status, summary))
        return result


def _result_json(result: TrainResult) -> str:
    """``result.json``: the result as strict JSON (``out_dir`` is where the file lives)."""
    d = {
        "run_id": result.run_id,
        "status": result.status,
        "steps": result.steps,
        "tokens": result.tokens,
        "final_loss": result.final_loss,
        "heldout_losses": result.heldout_losses,
        "tokens_per_dataset": result.tokens_per_dataset,
        "meta": json.loads(result.meta.to_json()),
        "wall_seconds": result.wall_seconds,
    }
    return json.dumps(json_safe(d), allow_nan=False, indent=2)
