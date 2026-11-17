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

This file holds the model part (plan B, Task 1); the screen at equal time follows (Task 2).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_model
from torch import Tensor, nn

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
