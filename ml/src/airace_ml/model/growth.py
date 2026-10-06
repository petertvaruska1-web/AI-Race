"""Exactly function-preserving growth (spec §4.3): a bigger brain that keeps what it knew.

``grow`` builds a fresh ``Transformer`` at the target shape and copies the old weights into it.
Modules are never swapped in place, because ``Attention`` caches ``n_head`` at construction.

Width ``d -> d'``: the residual stream is zero-padded. Every weight that *writes* a new residual
dimension, a new attention head's output or a new MLP unit's output is zero, so the new parts
contribute exactly nothing; weights that only *read* new dimensions keep their random init because
they read zeros. Each RMSNorm sees a mean square over ``d'`` entries of which only ``d`` are
non-zero, i.e. ``d/d'`` times the old one, so its gain is scaled by ``sqrt(d/d')`` and its ``eps``
by ``d/d'``, which makes the normalised old dimensions identical to before.

Depth: fresh blocks have zero ``wo`` and ``w_down``, so each is the identity on the residual stream.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Block, RMSNorm, Transformer


class GrowthError(ValueError):
    """The requested growth is not a function-preserving enlargement of the model."""


def insertion_layout(n_old: int, n_new: int) -> list[int | None]:
    """The grown block list: an ``int`` is a copied old block index, ``None`` a fresh block.

    Fresh block ``j`` (1-based, of ``k = n_new - n_old``) goes directly after the first
    ``floor(j * n_old / k + 0.5)`` old blocks, so the fresh blocks are spread evenly through the
    stack. Blocks that land on the same spot keep their input order.
    """
    if n_new < n_old:
        raise GrowthError(f"cannot shrink from {n_old} to {n_new} layers")
    k = n_new - n_old
    fresh_after = [0] * (n_old + 1)  # fresh_after[i]: fresh blocks right after i old blocks
    for j in range(1, k + 1):
        fresh_after[(2 * j * n_old + k) // (2 * k)] += 1  # exact floor(j * n_old / k + 0.5)
    layout: list[int | None] = []
    for i in range(n_old + 1):
        layout.extend([None] * fresh_after[i])
        if i < n_old:
            layout.append(i)
    return layout


def _copy_into(dst: Tensor, src: Tensor) -> None:
    """Copy ``src`` into the leading slice of ``dst`` (``dst`` is at least as large everywhere)."""
    dst[tuple(slice(0, n) for n in src.shape)] = src


def _grow_norm(dst: RMSNorm, src: RMSNorm, d_old: int, d_new: int) -> None:
    dst.weight.fill_(1.0)
    _copy_into(dst.weight, src.weight * math.sqrt(d_old / d_new))
    dst.eps = (src.eps.double() * (d_old / d_new)).to(src.eps.dtype)


def _grow_block(dst: Block, src: Block, d_old: int, d_new: int) -> None:
    """Copy one old block into a fresh one at the new width, zeroing every new writer."""
    _grow_norm(dst.attn_norm, src.attn_norm, d_old, d_new)
    _grow_norm(dst.mlp_norm, src.mlp_norm, d_old, d_new)
    for name in ("wq", "wk", "wv"):  # new heads / new input columns keep their random init
        _copy_into(getattr(dst.attn, name).weight, getattr(src.attn, name).weight)
    for lin_dst, lin_src in ((dst.attn.wo, src.attn.wo), (dst.mlp.w_down, src.mlp.w_down)):
        lin_dst.weight.zero_()  # writers: new output dims and new heads/units contribute nothing
        _copy_into(lin_dst.weight, lin_src.weight)
    for name in ("w_gate", "w_up"):
        _copy_into(getattr(dst.mlp, name).weight, getattr(src.mlp, name).weight)


def grow(model: Transformer, target: ModelShape, *, seed: int) -> Transformer:
    """A new model at ``target`` that computes exactly the same function as ``model``.

    ``seed`` drives the random init of the new parameters (the global RNG is left untouched). The
    result is on ``model``'s device and dtype and keeps its train/eval mode.
    """
    old = model.shape
    if target.n_layer < old.n_layer or target.d_model < old.d_model:
        raise GrowthError(
            f"model can only grow: ({old.n_layer} layers, d_model {old.d_model}) cannot become "
            f"({target.n_layer} layers, d_model {target.d_model})"
        )
    vocab_size = model.tok_emb.num_embeddings
    ref = model.tok_emb.weight
    # Build on the CPU so the new parameters depend on the seed alone, whatever the device.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new = Transformer(target, vocab_size)
    new.to(device=ref.device, dtype=ref.dtype)

    d_old, d_new = old.d_model, target.d_model
    with torch.no_grad():
        new.tok_emb.weight.zero_()  # new embedding columns; the head is tied, so it follows
        _copy_into(new.tok_emb.weight, model.tok_emb.weight)
        for dst, src_idx in zip(
            new.blocks, insertion_layout(old.n_layer, target.n_layer), strict=True
        ):
            if src_idx is None:
                dst.attn.wo.weight.zero_()
                dst.mlp.w_down.weight.zero_()
            else:
                _grow_block(dst, model.blocks[src_idx], d_old, d_new)
        _grow_norm(new.norm_f, model.norm_f, d_old, d_new)
    return new.train(model.training)
