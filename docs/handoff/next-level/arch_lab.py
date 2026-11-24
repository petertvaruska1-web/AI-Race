"""Lab: does a second brain design train better than today's on the RTX 3050? (plan B, spec 5.5)

The product's brain (``airace_ml/model/transformer.py``) is a decoder-only transformer with
pre-norm blocks, rotary positions (RoPE), a SwiGLU feed-forward and an embedding tied to the
output head. This lab tries five changes to it, each switched by a flag of :class:`ArchOptions`:

- ``zero_out``: the output projections of every block (``attn.wo`` and ``mlp.w_down``) start at
  zero, so every block starts as the identity and learns its way in.
- ``qk_norm``: the queries and keys are normalised per head before the rotation: a
  root-mean-square norm over the head's 32 numbers with a learned scale of 32 numbers (one for the
  queries and one for the keys of each block, shared by its heads, starting at 1).
- ``emb_norm``: a root-mean-square norm with a learned scale right after the embedding, so the
  residual stream starts at unit size at every position.
- ``relu2``: a two-matrix feed-forward ``w_down(relu(w_up(x)) ** 2)`` in place of the
  three-matrix SwiGLU, with 1.5 times SwiGLU's hidden width so that it holds as many weights.
- ``rope_base``: the base of the rotary code. At 500.0 instead of 10,000 its slowest rotation
  takes about 2,100 tokens, four times the longest span a brain reads (512).

The three new norms reuse the product's ``RMSNorm`` with an eps of 1e-8 (:data:`NORM_EPS`, which
says why the product's 1e-5 is not used for them).

:class:`LabTransformer` is a subclass of the product's ``Transformer``. With every flag off it is
the product's model bit for bit: the same parameter names and shapes, the same starting values from
the same seed, and the same logits on every path (a whole sequence, explicit positions with left
padding, and the KV cache). The changes that add no weight matrix (``zero_out``, ``qk_norm``,
``emb_norm`` and ``rope_base``) draw the same random numbers as the base model, so on one seed every
parameter they share with it starts at the base's value and a variant differs from the base only
in what its flag says.

Nothing under ``ml/src`` is changed. The lab builds its own RoPE tables with its own base (the
product's ``ROPE_BASE`` is never patched) and reuses the product's ``RMSNorm``, ``SwiGLU``,
``_apply_rope``, ``_attention_mask``, ``Block.forward`` and KV cache as they are.

A lab run's folder holds the trainer's usual checkpoint (``model.safetensors`` and ``meta.json``)
and, beside it, ``lab_arch.json`` with the options the model was built with.
:func:`load_lab_model` reads such a folder and refuses one without that file. The product's
``load_checkpoint`` fails loudly on a lab folder whose weights do not fit the product's model (any
design with ``qk_norm``, ``emb_norm`` or ``relu2``). A folder of ``zero_out`` alone holds a model
the product reads correctly (that flag only changes the start), but one of ``rope_base`` alone
holds weights that fit the product's model and a different function: only ``lab_arch.json`` tells
it apart, so lab models are always read with :func:`load_lab_model`.

The screen at equal time (plan B, Task 2)
-----------------------------------------

Run it from ``ml/`` with nothing else using the card (it uses the GPU and keeps the PC awake):

    AIRACE_DATA=../data2 python -m uv run python ../docs/handoff/next-level/arch_lab.py \\
        --out runs/arch-lab [--sizes tiny small medium] [--seeds 1 2] \\
        [--variants base zero qk emb relu2 rope all] [--bench base all] [--yardstick-root DIR]

It trains every variant of :data:`VARIANTS` asked for, and always ``base`` (today's design), at
the scorecard's sizes and budgets (:data:`SIZES`: Tiny and Small 12,582,912 tokens, Medium
25,165,824), on the scorecard's even mix, each config built by the scorecard's own
``_Card.config``, each seed of ``--seeds`` with the same seed for every variant (so a variant's
weights start as the base's, see above, and it reads the same batches):

1. Timing, for every size before anything is trained: 100 training steps of each variant through
   the real trainer (throw-away runs under ``--out/timing``), the baseline before and after the
   others. A run's speed is the tokens per second between the trainer's own progress events after
   its first 20 steps, leaving out every interval in which the trainer evaluated (held-out loss
   and samples: every 25 steps here, every 76 or more in a real run) or rolled back, so that it is
   the speed of the training steps, which is what a design changes. If the baseline's two speeds
   differ by more than 5%, something else used the card and the screen stops. Before the first
   timing the baseline trains once unmeasured, so that a cold card does not slow the first one.
   The speeds are kept in ``--out/timing/<size>.json`` and reused while the data, the code and
   the card are the same and every variant asked for was timed, so a screen started again trains
   to the same budgets and reuses its runs.
2. The equal-time budget of a variant (:func:`equal_time_budget`): the baseline's tokens times
   the variant's speed over the baseline's (the mean of its two), in whole steps, and never below
   50% or above 150% of the baseline's tokens (a warning says when it is held there).
3. Runs: ``--out/runs/<size>-<variant>-s<seed>``, each trained by the real trainer inside
   :func:`lab_design` (the trainer builds the lab's model; the trainer is checked to be as it
   ships after every run), with ``lab_arch.json`` beside its checkpoint and ``lab_run.json``,
   whose key names the run's options, size, seed, token budget, whole config, data identity,
   training-code digest, the digest of this file's model code (:func:`model_digest`) and the
   device type. A run is reused only while its key matches.
4. Measures, every one on the model :func:`load_lab_model` reads (never the product's
   ``load_checkpoint``, which would read a ``rope`` folder silently as today's design): the
   trainer's held-out mean (the data root's own held-out text), the scorecard's
   ``yardstick_loss`` on the yardstick root's held-out conversations (``--yardstick-root``,
   default ``<repo>/data``, text 1), the training seconds and the budget; and for the variants
   named by ``--bench``, bench-v1 overall exactly as the scorecard takes it (the yardstick root's
   reference judge and novelty index, the scorecard's seed). Measurements are cached under
   ``--out/measured`` by the run's token.
5. A table per size (each variant's mean over the seeds and its difference to ``base``), then the
   gate lines (:func:`gate_answer`); everything goes to ``--out/results.json``.

How long on the RTX 3050: the 42 runs of the full screen take about an hour and a half (each
variant trains about as long as the baseline: Tiny about 50 s, Small 90 s, Medium 4 minutes), the
timing 5 to 10 minutes (25 short runs), the yardstick about 3, and ``--bench base all`` (12
models) about 25: about two hours in all.

Hidden options for checks of the script itself: ``--tiny-root DIR`` screens on a tiny data root
(the trainer tests' fixture) at tiny shapes, budgets and timing runs (:data:`TINY_SCALE`), where
the baseline's two speeds may differ by up to a factor of two (a tiny model's speed on a busy CPU
is not steady over 20 steps; the 5% rule itself is tested with given speeds), and ``--device``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_model
from torch import Tensor, nn

from airace_ml.awake import stay_awake
from airace_ml.console import use_utf8_output
from airace_ml.data.identity import data_identity
from airace_ml.evals.suite import SUITE_VERSION, build_suite, run_benchmarks
from airace_ml.experiments import scorecard as sc
from airace_ml.experiments.gate import GateSetupError, _load_judge, _load_novelty
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
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME, CheckpointMeta, _vocab_size
from airace_ml.model.shape import HEAD_DIM, ModelShape
from airace_ml.model.transformer import (
    _ROPE_CACHE_LENGTHS,
    ROPE_BASE,
    Attention,
    Block,
    KVCache,
    RMSNorm,
    SwiGLU,
    Transformer,
    _apply_rope,
    _attention_mask,
    _in_autocast_dtype,
)
from airace_ml.paths import data_root as default_data_root
from airace_ml.paths import tokenizer_path, yardstick_root
from airace_ml.tokenizer import Tok
from airace_ml.train import trainer
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import Done, Instability, Progress, TrainEvent

LAB_ARCH_NAME = "lab_arch.json"
LAB_ARCH_FORMAT = 1
# The slowest rotation of the code is 2 pi * base ** ((HEAD_DIM - 2) / HEAD_DIM) tokens long:
# about 2,100 at 500, four times the longest span (512), against about 35,000 at 10,000.
SHORT_ROPE_BASE = 500.0
RELU2_STEP = 32
# The eps of the three new norms. They act on small things whose size the design means to remove:
# the embedding, whose mean square starts at INIT_STD ** 2 = 4e-4, and the queries and keys, whose
# mean square per head starts at about 0.02 to 0.05. The product's eps of 1e-5 would be 2.5% of the
# embedding's (every position about 1.2% short of unit size) and would leave attention depending on
# the size of the queries and keys (a Tiny model's logits move by about 1e-4 when they triple,
# against 1e-6 at 1e-8). At 1e-8 each is a norm: unit size, and attention independent of that
# size, to float32 rounding. The product's own norms keep 1e-5.
NORM_EPS = 1e-8


class LabError(Exception):
    """A lab folder, or a set of options, that the lab cannot use."""


@dataclass(frozen=True)
class ArchOptions:
    """Which of the five changes a lab brain has; every flag off is the product's design."""

    zero_out: bool = False
    qk_norm: bool = False
    emb_norm: bool = False
    relu2: bool = False
    rope_base: float = ROPE_BASE

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "rope_base":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 1.0
                ):
                    raise LabError(f"rope_base must be a number above 1, got {value!r}")
                object.__setattr__(self, "rope_base", float(value))
            elif not isinstance(value, bool):
                raise LabError(f"{f.name} must be True or False, got {value!r}")

    @classmethod
    def all(cls) -> ArchOptions:
        """Every change on, with the short RoPE base."""
        return cls(
            zero_out=True, qk_norm=True, emb_norm=True, relu2=True, rope_base=SHORT_ROPE_BASE
        )

    @property
    def name(self) -> str:
        """The variant's name in :data:`VARIANTS`; any other combination is named by its changes
        joined with "+" (for example "zero+qk")."""
        for name, variant in VARIANTS.items():
            if variant == self:
                return name
        flags = (
            ("zero", self.zero_out),
            ("qk", self.qk_norm),
            ("emb", self.emb_norm),
            ("relu2", self.relu2),
        )
        parts = [part for part, on in flags if on]
        if self.rope_base != ROPE_BASE:
            parts.append("rope" if self.rope_base == SHORT_ROPE_BASE else f"rope{self.rope_base:g}")
        return "+".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ArchOptions:
        """Options from :meth:`to_dict`; a record that does not name exactly the five options is
        refused (it was written by another version of the lab)."""
        names = {f.name for f in fields(cls)}
        if not isinstance(d, dict) or set(d) != names:
            raise LabError(f"lab options must name exactly {sorted(names)}, got {d!r}")
        return cls(**d)


VARIANTS: dict[str, ArchOptions] = {
    "base": ArchOptions(),
    "zero": ArchOptions(zero_out=True),
    "qk": ArchOptions(qk_norm=True),
    "emb": ArchOptions(emb_norm=True),
    "relu2": ArchOptions(relu2=True),
    "rope": ArchOptions(rope_base=SHORT_ROPE_BASE),
    "all": ArchOptions.all(),
}


def relu2_hidden(ffn_hidden: int) -> int:
    """Hidden width of the squared-ReLU feed-forward that holds as many weights as a SwiGLU of
    hidden width ``ffn_hidden``: two matrices 1.5 times as wide hold what three hold. It is
    rounded to a multiple of 32, which changes nothing for the product's shapes (their
    ``ffn_hidden`` is a multiple of 64)."""
    return RELU2_STEP * ((3 * ffn_hidden + RELU2_STEP) // (2 * RELU2_STEP))


def _lab_rope_tables(positions: Tensor, base: float) -> tuple[Tensor, Tensor]:
    """The product's ``_rope_tables`` with ``base`` in place of ``ROPE_BASE``: cos/sin tables
    ``[B, 1, T, HEAD_DIM]`` for integer positions ``[B, T]``. The arithmetic is the product's,
    so at the product's base the tables are the very same."""
    freqs = base ** (
        -torch.arange(0, HEAD_DIM, 2, device=positions.device, dtype=torch.float32) / HEAD_DIM
    )
    angles = positions.float().unsqueeze(-1) * freqs
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


class LabAttention(Attention):
    """The product's attention; with ``qk_norm`` the queries and keys of every head are
    normalised (with a learned scale) before the rotation, and the keys are cached normalised and
    rotated, exactly as a whole-sequence pass computes them."""

    def __init__(self, d_model: int, qk_norm: bool):
        super().__init__(d_model)
        self.q_norm = RMSNorm(HEAD_DIM, eps=NORM_EPS) if qk_norm else None
        self.k_norm = RMSNorm(HEAD_DIM, eps=NORM_EPS) if qk_norm else None

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        if self.q_norm is None:
            return super().forward(x, cos, sin, attn_mask, is_causal, cache, layer)
        b, t, d = x.shape
        q, k, v = (
            w(x).view(b, t, self.n_head, HEAD_DIM).transpose(1, 2)
            for w in (self.wq, self.wk, self.wv)
        )
        # Under autocast the norms may answer in float32; q and k go on in the autocast dtype, as
        # v does and as the product's q and k do, so attention runs the product's kernel.
        q = _in_autocast_dtype(self.q_norm(q))
        k = _in_autocast_dtype(self.k_norm(k))
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if cache is not None:
            if layer < len(cache.k):
                k = torch.cat([cache.k[layer], k], dim=2)
                v = torch.cat([cache.v[layer], v], dim=2)
                cache.k[layer], cache.v[layer] = k, v
            else:
                cache.k.append(k)
                cache.v.append(v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return self.wo(out.transpose(1, 2).reshape(b, t, d))


class ReLU2(nn.Module):
    """Two-matrix feed-forward with a squared ReLU: ``w_down(relu(w_up(x)) ** 2)``."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.relu(self.w_up(x)).square())


class LabBlock(Block):
    """The product's block, built from the lab's parts; its ``forward`` is the product's."""

    def __init__(self, shape: ModelShape, options: ArchOptions):
        # Not Block.__init__, which would build the product's parts (and draw random numbers for
        # them). The parts are made in the product's order, so the base draws the same numbers.
        nn.Module.__init__(self)
        self.attn_norm = RMSNorm(shape.d_model)
        self.attn = LabAttention(shape.d_model, options.qk_norm)
        self.mlp_norm = RMSNorm(shape.d_model)
        if options.relu2:
            self.mlp = ReLU2(shape.d_model, relu2_hidden(shape.ffn_hidden))
        else:
            self.mlp = SwiGLU(shape.d_model, shape.ffn_hidden)


class LabTransformer(Transformer):
    """The product's ``Transformer`` with the changes ``options`` switches on.

    Parameter names are the product's wherever a part exists in both. The new ones are
    ``emb_norm.weight`` (``[d_model]``), ``blocks[i].attn.q_norm.weight`` and
    ``blocks[i].attn.k_norm.weight`` (``[HEAD_DIM]``), all starting at 1; with ``relu2`` the
    feed-forward is ``w_up`` and ``w_down`` of hidden width :func:`relu2_hidden`, and there is
    no ``w_gate``.
    """

    def __init__(self, shape: ModelShape, vocab_size: int, options: ArchOptions):
        if not isinstance(options, ArchOptions):
            raise LabError(f"options must be ArchOptions, got {options!r}")
        # Transformer.__init__, part for part and in the same order, with the lab's parts: the
        # base model draws the very same random numbers as the product's.
        nn.Module.__init__(self)
        shape.validate(vocab_size)
        self.shape = shape
        self.options = options
        self.tok_emb = nn.Embedding(vocab_size, shape.d_model)
        self.emb_norm = RMSNorm(shape.d_model, eps=NORM_EPS) if options.emb_norm else None
        self.blocks = nn.ModuleList(LabBlock(shape, options) for _ in range(shape.n_layer))
        self.norm_f = RMSNorm(shape.d_model)
        self.lm_head = nn.Linear(shape.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        # RoPE tables for positions 0..t-1, by (t, device): not parameters, never saved.
        self._rope_cache: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        """The product's start; with ``zero_out`` the output projections are then set to zero
        (after their random start is drawn, so every other weight starts as the base's)."""
        super()._init_weights()
        if self.options.zero_out:
            for block in self.blocks:
                nn.init.zeros_(block.attn.wo.weight)
                nn.init.zeros_(block.mlp.w_down.weight)

    def _default_rope(self, past: int, t: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """The product's ``_default_rope`` with the lab's base: tables ``[1, 1, t, HEAD_DIM]`` for
        positions ``past .. past + t - 1``; those of a whole sequence are built once per length."""
        base = self.options.rope_base
        if past:
            return _lab_rope_tables((past + torch.arange(t, device=device))[None], base)
        key = (t, device)
        tables = self._rope_cache.get(key)
        if tables is None:
            if len(self._rope_cache) >= _ROPE_CACHE_LENGTHS:
                self._rope_cache.clear()
            # Built outside inference mode even when the caller is in it: a table made there
            # could never be used by a later training pass.
            with torch.inference_mode(False):
                tables = _lab_rope_tables(torch.arange(t, device=device)[None], base)
            self._rope_cache[key] = tables
        return tables

    def forward(
        self,
        idx: Tensor,
        *,
        positions: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        """The product's ``forward``, with the lab's RoPE base on every path (whole sequences,
        explicit positions and the cache) and, with ``emb_norm``, the norm after the embedding."""
        b, t = idx.shape
        past = kv_cache.length if kv_cache is not None else 0
        if positions is not None and positions.shape != idx.shape:
            raise ValueError(
                f"positions {tuple(positions.shape)} must match idx {tuple(idx.shape)}"
            )
        if key_padding_mask is not None and key_padding_mask.shape != (b, past + t):
            raise ValueError(
                f"key_padding_mask {tuple(key_padding_mask.shape)} must be (B, T_total) = "
                f"{(b, past + t)}"
            )
        if positions is None:
            cos, sin = self._default_rope(past, t, idx.device)
        else:
            cos, sin = _lab_rope_tables(positions, self.options.rope_base)
        cos, sin = _in_autocast_dtype(cos), _in_autocast_dtype(sin)
        attn_mask, is_causal = _attention_mask(t, past, key_padding_mask, idx.device)
        x = self.tok_emb(idx)
        if self.emb_norm is not None:
            x = self.emb_norm(x)
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, is_causal, kv_cache, layer)
        if kv_cache is not None:
            kv_cache.length = past + t
        return self.lm_head(self.norm_f(x))


def save_lab_arch(out_dir: Path, options: ArchOptions) -> None:
    """Write ``lab_arch.json`` (the options a lab model was built with) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"format": LAB_ARCH_FORMAT, "name": options.name, "options": options.to_dict()}
    (out_dir / LAB_ARCH_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def read_lab_arch(out_dir: Path) -> ArchOptions:
    """The options in ``out_dir``'s ``lab_arch.json``; a folder without one is not a lab run's."""
    path = Path(out_dir) / LAB_ARCH_NAME
    if not path.is_file():
        raise LabError(f"{out_dir} is not a lab run's folder: it has no {LAB_ARCH_NAME}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != LAB_ARCH_FORMAT:
        raise LabError(f"{path} is not a {LAB_ARCH_NAME} of format {LAB_ARCH_FORMAT}")
    return ArchOptions.from_dict(record.get("options"))


def load_lab_model(out_dir: Path, device: torch.device | str = "cpu") -> LabTransformer:
    """The lab model saved in ``out_dir``: its design from ``lab_arch.json``, its shape from the
    run's ``meta.json`` and its weights. Weights that do not fit the design fail loudly; nothing
    is loaded in part."""
    out_dir = Path(out_dir)
    options = read_lab_arch(out_dir)
    meta = CheckpointMeta.from_json((out_dir / META_NAME).read_text(encoding="utf-8"))
    weights = out_dir / WEIGHTS_NAME
    model = LabTransformer(meta.shape, _vocab_size(weights), options)
    load_model(model, str(weights))  # strict: a missing, extra or misshapen weight raises
    return model.to(device)


# == the screen at equal time (plan B, Task 2) ==================================================

LAB_VERSION = "arch-lab-1"
# The scorecard's sizes and budgets (spec 5.5): the shape and the baseline's tokens of each.
SIZES: dict[str, tuple[ModelShape, int]] = {
    "tiny": (sc.TINY, sc.STEP_TOKENS),
    "small": (sc.SMALL, sc.STEP_TOKENS),
    "medium": (sc.MEDIUM, 2 * sc.STEP_TOKENS),
}
TIMING_STEPS = 100  # training steps of a timing run
TIMING_SKIP = 20  # its first steps, not timed (the card and the allocator settle)
TIMING_SEED = 1  # the seed of every timing run, whatever --seeds says
STEADY_TOLERANCE = 0.05  # the most the baseline's two speeds may differ by
LOW_SHARE, HIGH_SHARE = 0.5, 1.5  # the bounds of a variant's budget, in the baseline's tokens
# The gate (spec 5.5, fixed before measuring): better than base by at least this much on every
# size, in held-out loss (lower is better) or in bench-v1 points (higher is better).
GATE_LOSS = 0.04
GATE_BENCH = 3.0
GATE_EPS = 1e-9  # a difference of exactly the margin passes, whatever the float rounding
GATE_LINE = (
    "{who}: better by ≥ {loss_margin:g} held-out loss on every size: {loss}; "
    "by ≥ {bench_margin:g} bench points on every size: {bench}"
)
MEASURES = ("heldout", "yardstick_loss", "seconds", "tokens", "bench_overall")
RECORD_FILE = "lab_run.json"
RESULTS_FILE = "results.json"
RUNS_DIR = "runs"
TIMING_DIR = "timing"
MEASURED_DIR = "measured"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ScreenScale:
    """How big the screen is: the scorecard's scale of its configs (``card``), and how long a
    timing run is, how many of its first steps are not timed and how far apart the baseline's two
    speeds may be."""

    card: sc.CardScale
    timing_steps: int
    timing_skip: int
    steady_tolerance: float


FULL_SCALE = ScreenScale(sc.FULL, TIMING_STEPS, TIMING_SKIP, STEADY_TOLERANCE)
# ``--tiny-root``: the trainer tests' shapes, runs of 12 steps of 1,024 tokens, timing runs of 40
# steps timed after the first 10, and the baseline's two speeds within a factor of two.
TINY_SCALE = ScreenScale(
    sc.CardScale(
        "arch-lab-tiny",
        shapes={
            sc.TINY: ModelShape(2, 64, 64),
            sc.SMALL: ModelShape(3, 96, 64),
            sc.MEDIUM: ModelShape(4, 128, 64),
        },
        tokens=1024 * 12,
        batch_tokens=1024,
    ),
    timing_steps=40,
    timing_skip=10,
    steady_tolerance=1.0,
)


def _require(condition: bool, message: str) -> None:
    """Raise :class:`LabError` with ``message`` unless ``condition`` holds (kept under ``-O``)."""
    if not condition:
        raise LabError(message)


def model_digest() -> str:
    """A digest of this file's model code (every class and function that builds, saves or reads a
    lab model, and their constants), so that a run or a timing of another version of the model is
    never reused. The screen's own code is left out: changing a table does not train again."""
    parts = (
        ArchOptions,
        relu2_hidden,
        _lab_rope_tables,
        LabAttention,
        ReLU2,
        LabBlock,
        LabTransformer,
        save_lab_arch,
        read_lab_arch,
        load_lab_model,
    )
    digest = hashlib.sha256()
    for part in parts:
        digest.update(inspect.getsource(part).encode("utf-8"))
    constants = (NORM_EPS, SHORT_ROPE_BASE, RELU2_STEP, LAB_ARCH_FORMAT, list(VARIANTS.items()))
    digest.update(repr(constants).encode("utf-8"))
    return digest.hexdigest()


# -- the swap of the trainer's model class --------------------------------------------------------


def assert_restored() -> None:
    """The trainer as it ships: it builds the product's ``Transformer``."""
    _require(
        trainer.Transformer is Transformer,
        "the trainer still builds the lab's model: a design swap was left in place",
    )


@contextlib.contextmanager
def lab_design(options: ArchOptions) -> Iterator[list[LabTransformer]]:
    """Inside the block, the trainer builds ``LabTransformer(shape, vocab, options)`` where it
    builds the product's ``Transformer`` (the name ``trainer._start_model`` calls), and the block
    gets the list of the models it built. The trainer is restored when the block ends, however it
    ends, and checked. A swap inside a swap is refused before anything is changed."""
    _require(isinstance(options, ArchOptions), f"options must be ArchOptions, got {options!r}")
    assert_restored()
    built: list[LabTransformer] = []

    def build(shape: ModelShape, vocab_size: int) -> LabTransformer:
        model = LabTransformer(shape, vocab_size, options)
        built.append(model)
        return model

    trainer.Transformer = build
    try:
        yield built
    finally:
        trainer.Transformer = Transformer
    assert_restored()


def _check_built(built: list[LabTransformer], options: ArchOptions, what: str) -> None:
    _require(
        len(built) == 1 and type(built[0]) is LabTransformer and built[0].options == options,
        f"{what}: the trainer built {len(built)} lab models, not one of the design {options.name}",
    )


# -- timing and the equal-time budget -------------------------------------------------------------


def progress_speed(events: Sequence[tuple[float, TrainEvent]], skip: int) -> float:
    """Tokens per second of a run from its events, each with the moment it arrived: the tokens
    and seconds between consecutive progress events from step ``skip`` on, leaving out every
    interval in which another event came (an evaluation's held-out loss and samples, or a
    rollback), which is time not spent on training steps."""
    tokens, seconds = 0, 0.0
    last: tuple[float, Progress] | None = None
    for at, event in events:
        if isinstance(event, Progress):
            if last is not None:
                tokens += event.tokens - last[1].tokens
                seconds += at - last[0]
            last = (at, event) if event.step >= skip else None
        elif not isinstance(event, Done):
            last = None
    _require(
        tokens > 0 and seconds > 0,
        f"no training steps after step {skip} could be timed (no two progress events in a row "
        f"without an evaluation between them)",
    )
    return tokens / seconds


def steady_speed(first: float, second: float, tolerance: float = STEADY_TOLERANCE) -> float:
    """The baseline's speed, the mean of its two timings; :class:`LabError` if they differ by
    more than ``tolerance`` (a share of the slower one), which means that something else used the
    card while the screen timed it."""
    _require(first > 0 and second > 0, f"the baseline's speeds {first}, {second} are not speeds")
    slow, fast = min(first, second), max(first, second)
    if fast > slow * (1.0 + tolerance):
        raise LabError(
            f"the baseline trained at {first:,.0f} and then at {second:,.0f} tokens per second, "
            f"{fast / slow - 1.0:.1%} apart (at most {tolerance:.0%} is steady): something else "
            f"used the card during the timing. Close everything else that uses it and start the "
            f"screen again."
        )
    return (first + second) / 2


@dataclass(frozen=True)
class Budget:
    """A variant's equal-time budget: ``steps`` of the run's batch (``tokens`` in all), its speed
    as a share of the baseline's, and whether the budget was held at a bound."""

    steps: int
    tokens: int
    share: float
    clamped: bool


def equal_time_budget(
    base_tokens: int, batch_tokens: int, base_speed: float, variant_speed: float
) -> Budget:
    """The tokens a variant trains in the time the baseline trains ``base_tokens``:
    ``round(base_tokens * variant_speed / base_speed)`` in whole steps of ``batch_tokens``, held
    within :data:`LOW_SHARE` and :data:`HIGH_SHARE` of the baseline's steps."""
    speeds = (base_speed, variant_speed)
    _require(
        all(isinstance(s, int | float) and math.isfinite(s) and s > 0 for s in speeds),
        f"speeds must be positive numbers, got {base_speed!r} and {variant_speed!r}",
    )
    share = variant_speed / base_speed
    wanted = round(base_tokens * share / batch_tokens)
    low = math.ceil(LOW_SHARE * base_tokens / batch_tokens)
    high = math.floor(HIGH_SHARE * base_tokens / batch_tokens)
    steps = min(max(wanted, low), high)
    return Budget(steps, steps * batch_tokens, share, steps != wanted)


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else device.type


# -- the runs ------------------------------------------------------------------------------------


@dataclass
class LabRun:
    """A finished run of the screen. ``token`` names this exact training: a run trained again gets
    a new one, so its cached measurements are not reused."""

    name: str
    size: str
    variant: str
    seed: int
    dir: Path
    cfg: TrainRunConfig
    options: ArchOptions
    token: str
    status: str
    steps: int
    seconds: float
    heldout_losses: dict[str, float]
    reused: bool

    @property
    def heldout(self) -> float | None:
        return _heldout_mean(self.heldout_losses)


class Screen:
    """Times, trains and measures the screen's runs under ``out`` (see the module docstring)."""

    def __init__(
        self,
        root: Path,
        yardstick_root: Path,
        out: Path,
        device: torch.device,
        scale: ScreenScale,
        bench: Sequence[str],
    ) -> None:
        self.root, self.yardstick_root, self.out = Path(root), Path(yardstick_root), Path(out)
        self.device, self.scale, self.bench = device, scale, list(bench)
        self.timing_steps = scale.timing_steps
        self.timing_skip = scale.timing_skip
        self.steady_tolerance = scale.steady_tolerance
        self.identity = self._identity(self.root, "the data root")
        same = self.root.resolve() == self.yardstick_root.resolve()
        self.yardstick = (
            self.identity if same else self._identity(self.yardstick_root, "the yardstick root")
        )
        _require(
            same
            or tokenizer_path(self.root).read_bytes()
            == tokenizer_path(self.yardstick_root).read_bytes(),
            f"the yardstick root {self.yardstick_root} holds another tokenizer than the data root "
            f"{self.root}; both must use the same one",
        )
        self.code = source_digest(TRAINING_SOURCES)
        self.model_code = model_digest()
        self.tok = Tok.load(tokenizer_path(self.root))
        self._instruments = None
        self._warm = False

    @staticmethod
    def _identity(root: Path, what: str) -> str:
        try:
            return data_identity(root)
        except (OSError, ValueError) as e:
            raise LabError(f"{what} {root} holds no complete training data ({e})") from e

    def config(self, size: str, seed: int, name: str, tokens: int | None = None) -> TrainRunConfig:
        """The config of a run of ``size``: the scorecard's own ``_Card.config`` of the size's
        shape, the baseline's tokens and the even mix at this screen's scale, with ``tokens`` in
        place of the token budget when it is given."""
        shape, base_tokens = SIZES[size]
        step = sc.Step(name, shape, base_tokens, sc.BALANCED_MIX, seed=seed)
        cfg = sc._Card.config(SimpleNamespace(scale=self.scale.card), step)
        return cfg if tokens is None else replace(cfg, token_budget=tokens)

    def timing_config(self, size: str, tag: str) -> TrainRunConfig:
        cfg = self.config(size, TIMING_SEED, f"timing-{size}-{tag}")
        return replace(cfg, token_budget=self.timing_steps * cfg.batch_tokens)

    # -- timing ----------------------------------------------------------------------------------

    def speed(self, size: str, variant: str, tag: str) -> float:
        """Tokens per second of a throw-away timing run of ``variant`` at ``size``
        (:func:`progress_speed` after the scale's first steps)."""
        options = VARIANTS[variant]
        cfg = self.timing_config(size, tag)
        folder = self.out / TIMING_DIR / cfg.run_id
        shutil.rmtree(folder, ignore_errors=True)
        events: list[tuple[float, TrainEvent]] = []

        def on_event(event: TrainEvent) -> None:
            events.append((time.perf_counter(), event))

        with lab_design(options) as built:
            trainer.train_run(
                cfg, out_dir=folder, data_root=self.root, device=self.device, on_event=on_event
            )
        assert_restored()
        _check_built(built, options, cfg.run_id)
        shutil.rmtree(folder, ignore_errors=True)
        value = progress_speed(events, self.timing_skip)
        print(f"  timing {size} {tag}: {value:,.0f} tokens per second", flush=True)
        return value

    def time_size(self, size: str, variants: Sequence[str]) -> dict:
        """The speeds of ``base`` and ``variants`` at ``size`` and their equal-time budgets; the
        speeds are timed (the baseline before and after the others) unless a timing of every one
        of them under the same key is kept, and kept only when the baseline was steady."""
        wanted = ["base", *[v for v in variants if v != "base"]]
        config = json.loads(self.timing_config(size, "probe").to_json())
        config.pop("run_id")
        # The variants' designs are part of the model code's digest.
        key = _plain(
            {
                "lab": LAB_VERSION,
                "model_code": self.model_code,
                "size": size,
                "config": config,
                "skip": self.timing_skip,
                "data": self.identity,
                "code": self.code,
                "device": self.device.type,
                "device_name": _device_name(self.device),
            }
        )
        path = self.out / TIMING_DIR / f"{size}.json"
        record = _read_json(path)
        kept = record.get("value") if isinstance(record, dict) else None
        reused = (
            isinstance(record, dict)
            and record.get("key") == key
            and isinstance(kept, dict)
            and isinstance(kept.get("speeds"), dict)
            and all(v in kept["speeds"] for v in wanted)
        )
        if reused:
            print(f"timing of {size}: measured earlier, reusing it", flush=True)
            value = kept
        else:
            print(f"timing of {size} ({self.timing_steps} steps a run):", flush=True)
            if not self._warm:
                self._warm = True
                self.speed(size, "base", "warm-up")  # not measured: the card warms up
            first = self.speed(size, "base", "base")
            speeds = {v: self.speed(size, v, v) for v in wanted[1:]}
            second = self.speed(size, "base", "base-again")
            base = steady_speed(first, second, self.steady_tolerance)
            value = {"base": [first, second], "speeds": {"base": base, **speeds}}
            _write_json(path, {"key": key, "value": value})
        base_cfg = self.config(size, TIMING_SEED, "budget")
        base_speed = value["speeds"]["base"]
        budgets = {}
        for v in wanted:
            budget = equal_time_budget(
                base_cfg.token_budget, base_cfg.batch_tokens, base_speed, value["speeds"][v]
            )
            if budget.clamped:
                print(
                    f"  warning: {size} {v} trains at {budget.share:.0%} of the baseline's speed; "
                    f"its budget is held at {budget.steps} steps ({budget.tokens:,} tokens), "
                    f"{LOW_SHARE:.0%} to {HIGH_SHARE:.0%} of the baseline's, so it is not at "
                    f"equal time",
                    flush=True,
                )
            budgets[v] = asdict(budget)
        return {
            "base": list(value["base"]),
            "speeds": {v: value["speeds"][v] for v in wanted},
            "budgets": budgets,
            "reused": reused,
        }

    # -- training --------------------------------------------------------------------------------

    def run_key(self, name: str, size: str, variant: str, seed: int, cfg: TrainRunConfig) -> dict:
        """What makes a run the same run: its design, size, seed and budget, its whole config, the
        data, the training code, the lab model's code and the device type."""
        return _plain(
            {
                "lab": LAB_VERSION,
                "name": name,
                "size": size,
                "variant": variant,
                "options": VARIANTS[variant].to_dict(),
                "seed": seed,
                "token_budget": cfg.token_budget,
                "config": json.loads(cfg.to_json()),
                "data": self.identity,
                "code": self.code,
                "model_code": self.model_code,
                "device": self.device.type,
            }
        )

    def train(self, size: str, variant: str, seed: int, tokens: int) -> LabRun:
        """The finished run of ``variant`` at ``size`` and ``seed`` with a budget of ``tokens``;
        reused only when its record's key matches and its folder holds the checkpoint and the
        ``lab_arch.json`` of this design."""
        name = f"{size}-{variant}-s{seed}"
        options = VARIANTS[variant]
        cfg = self.config(size, seed, name, tokens)
        out = self.out / RUNS_DIR / name
        key = self.run_key(name, size, variant, seed, cfg)
        record = _read_json(out / RECORD_FILE)
        reusable = (
            isinstance(record, dict)
            and record.get("key") == key
            and record.get("status") in ("completed", "unstable_stopped")
            and (out / WEIGHTS_NAME).is_file()
            and (out / META_NAME).is_file()
            and self._design_of(out) == options
        )
        if reusable:
            print(f"  {name}: finished earlier, reusing it", flush=True)
        else:
            record = self._train_fresh(name, cfg, out, key, options)
        run = LabRun(
            name=name,
            size=size,
            variant=variant,
            seed=seed,
            dir=out,
            cfg=cfg,
            options=options,
            token=record["token"],
            status=record["status"],
            steps=record["result"]["steps"],
            seconds=record["result"]["wall_seconds"],
            heldout_losses=record["result"]["heldout_losses"],
            reused=reusable,
        )
        self._check_run(run)
        return run

    @staticmethod
    def _design_of(out: Path) -> ArchOptions | None:
        try:
            return read_lab_arch(out)
        except (LabError, OSError, ValueError, TypeError):
            return None

    def _train_fresh(
        self, name: str, cfg: TrainRunConfig, out: Path, key: dict, options: ArchOptions
    ) -> dict:
        _write_json(out / RECORD_FILE, {"key": key, "status": "started"})
        save_lab_arch(out, options)  # the folder is a lab run's from the start
        print(
            f"  {name}: training ({cfg.steps} steps, {cfg.token_budget:,} tokens, "
            f"on {self.device.type})",
            flush=True,
        )
        marks = [cfg.steps * q // 4 for q in (1, 2, 3)]

        def on_event(event: TrainEvent) -> None:
            if isinstance(event, Instability):
                print(f"  {name}: unstable at step {event.step} ({event.action})", flush=True)
            elif isinstance(event, Progress) and marks and event.step >= marks[0]:
                while marks and event.step >= marks[0]:
                    marks.pop(0)
                print(f"  {name}: step {event.step}/{event.total_steps}", flush=True)

        with lab_design(options) as built:
            result = trainer.train_run(
                cfg, out_dir=out, data_root=self.root, device=self.device, on_event=on_event
            )
        assert_restored()
        _check_built(built, options, name)
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
        _write_json(out / RECORD_FILE, record)
        mean = _heldout_mean(result.heldout_losses)
        print(
            f"  {name}: {result.status} in {result.wall_seconds:.1f} s, held-out loss "
            f"{'n/a' if mean is None else f'{mean:.3f}'}",
            flush=True,
        )
        return _plain(record)

    @staticmethod
    def _check_run(run: LabRun) -> None:
        """The folder holds this run's checkpoint, a first generation at its shape, of its
        design, trained to its end unless it stopped as unstable."""
        meta = CheckpointMeta.from_json((run.dir / META_NAME).read_text(encoding="utf-8"))
        _require(
            meta.version_id == run.name
            and meta.shape == run.cfg.shape
            and meta.parent_version_id is None
            and len(meta.runs) == 1,
            f"{run.name}: the checkpoint is {meta.version_id} at {meta.shape}, not this run's",
        )
        _require(
            read_lab_arch(run.dir) == run.options,
            f"{run.name}: {LAB_ARCH_NAME} does not name the design {run.options.name}",
        )
        _require(
            run.status == "unstable_stopped" or run.steps == run.cfg.steps,
            f"{run.name}: ran {run.steps} of {run.cfg.steps} steps",
        )

    # -- measures --------------------------------------------------------------------------------

    def instruments(self):
        """The yardstick root's reference judge and novelty index (bench-v1's creativity)."""
        if self._instruments is None:
            try:
                self._instruments = (
                    _load_judge(self.yardstick_root, self.device, self.yardstick),
                    _load_novelty(self.yardstick_root, self.yardstick),
                )
            except GateSetupError as e:
                raise LabError(str(e)) from e
        return self._instruments

    def measure(self, run: LabRun) -> dict[str, float | None]:
        """The held-out mean, the yardstick loss, the training seconds and, for a variant named by
        ``--bench``, bench-v1 overall of ``run``, on the model :func:`load_lab_model` reads; the
        last two are cached under the run's token."""
        key = {
            "run": run.token,
            "code": source_digest(sc.CARD_SOURCES),
            "model_code": self.model_code,
            "yardstick": self.yardstick,
            "rows": sc.YARDSTICK_SEQS,
        }
        folder = self.out / MEASURED_DIR
        model: LabTransformer | None = None

        def model_() -> LabTransformer:
            nonlocal model
            if model is None:
                model = load_lab_model(run.dir, self.device)
                _require(
                    model.options == run.options and model.shape == run.cfg.shape,
                    f"{run.name}: the folder holds a {model.options.name} model at "
                    f"{model.shape}, not the run's",
                )
            return model

        value = _cached(
            folder / f"{run.name}.json",
            key,
            lambda: {"yardstick_loss": sc.yardstick_loss(model_(), self.yardstick_root)},
        )
        measures = {
            "heldout": run.heldout,
            "yardstick_loss": value["yardstick_loss"],
            "seconds": run.seconds,
        }
        if run.variant in self.bench:
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


# -- the screen, its tables and the gate ----------------------------------------------------------


def run_screen(
    screen: Screen, sizes: Sequence[str], seeds: Sequence[int], variants: Sequence[str]
) -> dict:
    """Time every size, then train every run, then measure them; the results as a dict."""
    if screen.bench:
        screen.instruments()  # a missing judge stops the screen now, not after the training
    timing = {size: screen.time_size(size, variants) for size in sizes}
    runs: list[LabRun] = []
    for size in sizes:
        print(f"{size}:", flush=True)
        for seed in seeds:
            for variant in variants:
                tokens = timing[size]["budgets"][variant]["tokens"]
                runs.append(screen.train(size, variant, seed, tokens))
    assert_restored()
    print("measuring:", flush=True)
    models: dict[str, dict[str, dict[str, dict]]] = {
        size: {variant: {} for variant in variants} for size in sizes
    }
    for run in runs:
        models[run.size][run.variant][str(run.seed)] = {
            "run": run.name,
            "status": run.status,
            "steps": run.steps,
            "tokens": run.cfg.token_budget,
            **screen.measure(run),
        }
    summary = summarise(models, seeds)
    gate = {v: gate_answer(summary, v) for v in _gate_order(variants)}
    return {
        "lab": LAB_VERSION,
        "data_root": str(screen.root),
        "yardstick_root": str(screen.yardstick_root),
        "device": screen.device.type,
        "device_name": _device_name(screen.device),
        "scale": screen.scale.card.name,
        "sizes": list(sizes),
        "seeds": list(seeds),
        "variants": list(variants),
        "bench": list(screen.bench),
        "options": {v: VARIANTS[v].to_dict() for v in variants},
        "timing_steps": screen.timing_steps,
        "timing_skip": screen.timing_skip,
        "steady_tolerance": screen.steady_tolerance,
        "timing": timing,
        "models": models,
        "summary": summary,
        "gate": gate,
    }


def _mean(values: Sequence[object]) -> float | None:
    """The mean of ``values`` if every one is a finite number, else ``None``."""
    finite = [
        float(v)
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    ]
    return math.fsum(finite) / len(finite) if finite and len(finite) == len(values) else None


def summarise(models: dict, seeds: Sequence[int]) -> dict:
    """Per size and variant, for each measure taken: the values per seed, their mean and the
    mean's difference to ``base``'s."""
    out: dict = {}
    for size, per_variant in models.items():
        rows: dict = {}
        for variant, per_seed in per_variant.items():
            row = {}
            for measure in MEASURES:
                values = [per_seed[str(s)].get(measure) for s in seeds]
                if all(v is None for v in values) and measure == "bench_overall":
                    continue  # not benched
                row[measure] = {"per_seed": values, "mean": _mean(values), "minus_base": None}
            rows[variant] = row
        base = rows.get("base", {})
        for row in rows.values():
            for measure, cell in row.items():
                theirs = base.get(measure, {}).get("mean")
                if cell["mean"] is not None and theirs is not None:
                    cell["minus_base"] = cell["mean"] - theirs
        out[size] = rows
    return out


def _gate_order(variants: Sequence[str]) -> list[str]:
    """``all`` first, then each single change, in :data:`VARIANTS` order."""
    order = ["all", *[v for v in VARIANTS if v not in ("base", "all")]]
    return [v for v in order if v in variants]


def _yes(answer: bool | None) -> str:
    return "n/a" if answer is None else ("yes" if answer else "no")


def gate_answer(summary: dict, variant: str) -> dict:
    """The gate for ``variant`` against ``base`` on every size of :data:`SIZES` (the means over
    the seeds): better by at least :data:`GATE_LOSS` held-out loss on every size, and by at least
    :data:`GATE_BENCH` bench-v1 points on every size. ``None`` where a size or a measure was not
    screened, and the line the screen prints."""

    def every_size(measure: str, passes) -> bool | None:
        diffs = []
        for size in SIZES:
            cell = summary.get(size, {}).get(variant, {}).get(measure)
            if cell is None or cell["minus_base"] is None:
                return None
            diffs.append(cell["minus_base"])
        return all(passes(d) for d in diffs)

    heldout = every_size("heldout", lambda d: -d >= GATE_LOSS - GATE_EPS)
    bench = every_size("bench_overall", lambda d: d >= GATE_BENCH - GATE_EPS)
    line = GATE_LINE.format(
        who="all together" if variant == "all" else variant,
        loss_margin=GATE_LOSS,
        loss=_yes(heldout),
        bench_margin=GATE_BENCH,
        bench=_yes(bench),
    )
    return {"heldout": heldout, "bench": bench, "line": line}


def _fmt(value: float | None, measure: str, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    if measure == "tokens":
        return f"{value:{sign},.0f}"
    if measure in ("seconds", "bench_overall"):
        return f"{value:{sign}.1f}"
    return f"{value:{sign}.4f}"


def print_tables(results: dict) -> None:
    print(
        f"\ntiming (tokens per second of the training steps after the first "
        f"{results['timing_skip']} of {results['timing_steps']}; the baseline before and after "
        f"the others; share of the baseline's speed; the equal-time budget):"
    )
    for size, timing in results["timing"].items():
        first, second = timing["base"]
        apart = max(first, second) / min(first, second) - 1.0
        reused = ", measured earlier" if timing["reused"] else ""
        print(f"  {size}: baseline {first:,.0f} then {second:,.0f} ({apart:.1%} apart{reused})")
        for variant, speed in timing["speeds"].items():
            b = timing["budgets"][variant]
            held = "  (held at a bound: not equal time)" if b["clamped"] else ""
            print(
                f"    {variant:<6} {speed:>12,.0f}  {b['share']:>7.1%}  {b['steps']:>6,} steps  "
                f"{b['tokens']:>12,} tokens{held}"
            )
    seeds = " ".join(str(s) for s in results["seeds"])
    for size, rows in results["summary"].items():
        print(
            f"\n{size}: mean of seeds {seeds} (in brackets the difference to base; lower loss is "
            f"better)"
        )
        for variant, row in rows.items():
            cells = []
            for measure in MEASURES:
                if measure not in row:
                    continue
                cell = row[measure]
                text = f"{measure} {_fmt(cell['mean'], measure)}"
                if variant != "base":
                    text += f" ({_fmt(cell['minus_base'], measure, signed=True)})"
                cells.append(text)
            print(f"  {variant:<6} " + "  ".join(cells))
    if results["gate"]:
        print("\ngate (spec 5.5; the mean of the seeds against base on Tiny, Small and Medium):")
        for answer in results["gate"].values():
            print(answer["line"])


def _outside_data(out: Path) -> bool:
    resolved = out.resolve()
    for name in ("data", "data2"):
        if resolved.is_relative_to((REPO_ROOT / name).resolve()):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
        help="designs to screen (base is always screened)",
    )
    parser.add_argument(
        "--bench", nargs="+", default=[], choices=list(VARIANTS), help="designs to bench"
    )
    parser.add_argument(
        "--yardstick-root",
        type=Path,
        help="data root whose held-out conversations, judge and novelty index measure every "
        "model (default: <repo>/data, text 1)",
    )
    parser.add_argument("--tiny-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("sizes", "seeds", "variants", "bench"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name} must not repeat")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be 0 or more")
    variants = [v for v in VARIANTS if v == "base" or v in args.variants]
    sizes = [s for s in SIZES if s in args.sizes]
    unscreened = [v for v in args.bench if v not in variants]
    if unscreened:
        parser.error(f"--bench names designs that are not screened: {' '.join(unscreened)}")
    if not _outside_data(args.out):
        parser.error("--out must not be inside data/ or data2/")
    tiny = args.tiny_root is not None
    root = Path(args.tiny_root) if tiny else default_data_root()
    yardstick = args.yardstick_root or (root if tiny else yardstick_root(REPO_ROOT / "data"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_utf8_output()
    try:
        screen = Screen(
            root, yardstick, args.out, device, TINY_SCALE if tiny else FULL_SCALE, args.bench
        )
        print(
            f"arch lab: sizes {' '.join(sizes)}; seeds {' '.join(map(str, args.seeds))}; "
            f"variants {' '.join(variants)}; bench {' '.join(args.bench) or 'none'}; "
            f"{_device_name(device)}; data {root}; yardstick {yardstick}; into {args.out}",
            flush=True,
        )
        with stay_awake():
            results = run_screen(screen, sizes, list(args.seeds), variants)
    except LabError as e:
        print(f"arch lab stopped: {e}", file=sys.stderr, flush=True)
        return 1
    assert_restored()
    _write_json(args.out / RESULTS_FILE, results)
    print_tables(results)
    print(f"\nwritten to {args.out / RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
