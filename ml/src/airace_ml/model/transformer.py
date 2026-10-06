"""Decoder-only transformer: pre-norm blocks, RoPE, SwiGLU, tied embeddings, KV cache.

Module and attribute names here are a contract with growth (Task 4), which copies and zeroes
slices of exactly these modules: ``tok_emb``, ``blocks[i].attn_norm``, ``.attn.wq/.wk/.wv/.wo``,
``.mlp_norm``, ``.mlp.w_gate/.w_up/.w_down``, ``norm_f`` and ``lm_head``. All linears are
bias-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from airace_ml.model.shape import HEAD_DIM, ModelShape
from airace_ml.tokenizer import VOCAB_SIZE

ROPE_BASE = 10000.0
INIT_STD = 0.02


class RMSNorm(nn.Module):
    """Root-mean-square norm. ``eps`` is a saved buffer so growth can rescale it per module."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps.float())
        return normed.to(x.dtype) * self.weight


@dataclass
class KVCache:
    """Per-layer keys and values, each ``[B, H, T, HEAD_DIM]``, plus the cached length."""

    k: list[Tensor] = field(default_factory=list)
    v: list[Tensor] = field(default_factory=list)
    length: int = 0


def _rope_tables(positions: Tensor) -> tuple[Tensor, Tensor]:
    """cos/sin tables ``[B, 1, T, HEAD_DIM]`` for explicit integer positions ``[B, T]``."""
    freqs = ROPE_BASE ** (
        -torch.arange(0, HEAD_DIM, 2, device=positions.device, dtype=torch.float32) / HEAD_DIM
    )
    angles = positions.float().unsqueeze(-1) * freqs
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.n_head = d_model // HEAD_DIM
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

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
        b, t, d = x.shape
        q, k, v = (
            w(x).view(b, t, self.n_head, HEAD_DIM).transpose(1, 2)
            for w in (self.wq, self.wk, self.wv)
        )
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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, shape: ModelShape):
        super().__init__()
        self.attn_norm = RMSNorm(shape.d_model)
        self.attn = Attention(shape.d_model)
        self.mlp_norm = RMSNorm(shape.d_model)
        self.mlp = SwiGLU(shape.d_model, shape.ffn_hidden)

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
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, is_causal, cache, layer)
        return x + self.mlp(self.mlp_norm(x))


def _attention_mask(
    t: int, past: int, key_padding_mask: Tensor | None, device: torch.device
) -> tuple[Tensor | None, bool]:
    """Return ``(attn_mask, is_causal)`` for ``t`` new queries after ``past`` cached keys.

    A boolean mask is True where attention is allowed. Padded queries (left-pad slots) would
    otherwise have no valid key at all, and softmax over nothing is NaN, so every query may
    always attend to itself. Real queries are unaffected (their own key is a real key); padded
    rows end up finite and are never read by real positions, because padded keys stay masked.
    """
    if key_padding_mask is None:
        if past == 0:
            return None, True
        if t == 1:
            return None, False  # a single new query may see every cached key
    q_pos = (past + torch.arange(t, device=device)).unsqueeze(1)  # [T, 1]
    k_pos = torch.arange(past + t, device=device).unsqueeze(0)  # [1, T_total]
    allowed = k_pos <= q_pos  # [T, T_total]
    if key_padding_mask is None:
        return allowed[None, None], False
    allowed = (allowed.unsqueeze(0) & key_padding_mask.unsqueeze(1)) | (k_pos == q_pos).unsqueeze(0)
    return allowed.unsqueeze(1), False  # [B, 1, T, T_total]


class Transformer(nn.Module):
    def __init__(self, shape: ModelShape, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        shape.validate(vocab_size)
        self.shape = shape
        self.tok_emb = nn.Embedding(vocab_size, shape.d_model)
        self.blocks = nn.ModuleList(Block(shape) for _ in range(shape.n_layer))
        self.norm_f = RMSNorm(shape.d_model)
        self.lm_head = nn.Linear(shape.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        residual_std = INIT_STD / math.sqrt(2 * self.shape.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=residual_std)

    def new_kv_cache(self) -> KVCache:
        return KVCache()

    def num_params(self) -> int:
        """Unique trainable parameters (the tied head is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: Tensor,
        *,
        positions: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        """Logits ``[B, T, V]`` for token ids ``idx`` ``[B, T]``.

        ``positions`` ``[B, T]`` are the RoPE positions (default: continue after the cache).
        ``key_padding_mask`` ``[B, T_total]`` is True for real tokens and covers the cached keys
        followed by the new tokens. When ``kv_cache`` is given it is extended in place.
        """
        b, t = idx.shape
        past = kv_cache.length if kv_cache is not None else 0
        if positions is None:
            positions = (past + torch.arange(t, device=idx.device)).expand(b, t)
        elif positions.shape != idx.shape:
            raise ValueError(
                f"positions {tuple(positions.shape)} must match idx {tuple(idx.shape)}"
            )
        if key_padding_mask is not None and key_padding_mask.shape != (b, past + t):
            raise ValueError(
                f"key_padding_mask {tuple(key_padding_mask.shape)} must be (B, T_total) = "
                f"{(b, past + t)}"
            )
        cos, sin = _rope_tables(positions)
        attn_mask, is_causal = _attention_mask(t, past, key_padding_mask, idx.device)
        x = self.tok_emb(idx)
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, is_causal, kv_cache, layer)
        if kv_cache is not None:
            kv_cache.length = past + t
        return self.lm_head(self.norm_f(x))
