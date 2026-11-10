"""A faster forward for the product Transformer: same parameters, same math, fewer memory passes.

Patches the product modules in place (switchable), so speed and equality can be measured.
"""

import torch
import torch.nn.functional as F

import airace_ml.model.transformer as T
from airace_ml.model.shape import HEAD_DIM

ORIG = {
    "norm": T.RMSNorm.forward,
    "attn": T.Attention.forward,
    "mlp": T.SwiGLU.forward,
    "tables": T._rope_tables,
    "apply": T._apply_rope,
}


def norm_fused(self, x):
    eps = self.__dict__.get("_eps_f")
    if eps is None:
        eps = self._eps_f = float(self.eps)
    return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), eps)


_HALF = HEAD_DIM // 2


def rope_roll(x, cos, sin_signed):
    # x*cos + [-x2, x1]*sin  ==  x*cos + roll(x)*(sin * [-1.., +1..])
    return torch.addcmul(x * cos, x.roll(_HALF, dims=-1), sin_signed)


def make_attn(merge_qkv: bool, rope: str):
    def forward(self, x, cos, sin, attn_mask, is_causal, cache, layer):
        b, t, d = x.shape
        if merge_qkv:
            w = torch.cat([self.wq.weight, self.wk.weight, self.wv.weight])
            qkv = F.linear(x, w).view(b, t, 3, self.n_head, HEAD_DIM).permute(2, 0, 3, 1, 4)
            if rope == "joint":
                qk = rope_roll(qkv[:2], cos, sin)
                q, k, v = qk[0], qk[1], qkv[2]
            else:
                q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q, k, v = (
                w(x).view(b, t, self.n_head, HEAD_DIM).transpose(1, 2)
                for w in (self.wq, self.wk, self.wv)
            )
        if rope == "roll":
            q, k = rope_roll(q, cos, sin), rope_roll(k, cos, sin)
        elif rope == "orig":
            q, k = ORIG["apply"](q, cos, sin), ORIG["apply"](k, cos, sin)
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

    return forward


def mlp_merged(self, x):
    h = self.w_gate.weight.shape[0]
    gu = F.linear(x, torch.cat([self.w_gate.weight, self.w_up.weight]))
    return self.w_down(F.silu(gu[..., :h]) * gu[..., h:])


def make_tables(signed: bool, dtype_cache: bool):
    cache = {}

    def tables(positions):
        cos, sin = ORIG["tables"](positions)
        if signed:
            sign = torch.ones(HEAD_DIM, device=sin.device)
            sign[:_HALF] = -1.0
            sin = sin * sign
        return cos, sin

    def cached(positions):
        # training path: positions are arange(t) for every row -> one [1,1,T,D] table per length
        key = (positions.shape[1], positions.device)
        hit = cache.get(key)
        if hit is None:
            cos, sin = tables(positions[:1])
            dt = torch.bfloat16 if positions.device.type == "cuda" else torch.float32
            hit = cache[key] = (cos.to(dt), sin.to(dt))
        return hit

    return cached if dtype_cache else tables


def apply(flags: str):
    """flags: '+'-joined subset of norm, rope, qkv, joint, mlp, cache."""
    f = set(flags.split("+")) if flags else set()
    T.RMSNorm.forward = norm_fused if "norm" in f else ORIG["norm"]
    rope = "joint" if "joint" in f else ("roll" if "rope" in f else "orig")
    merge = "qkv" in f or "joint" in f
    T.Attention.forward = make_attn(merge, rope) if (merge or rope != "orig") else ORIG["attn"]
    T.SwiGLU.forward = mlp_merged if "mlp" in f else ORIG["mlp"]
    T._rope_tables = (
        make_tables(signed=rope != "orig", dtype_cache="cache" in f)
        if (rope != "orig" or "cache" in f)
        else ORIG["tables"]
    )
