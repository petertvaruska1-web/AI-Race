"""Ablation timings of one training step: what each part of the model step costs on this card."""

import sys
import time

import torch
import torch.nn.functional as F

import airace_ml.model.transformer as T
from airace_ml.model.shape import ModelShape

dev = torch.device("cuda")
V = 4096
shape = ModelShape(*[int(a) for a in sys.argv[1:4]])
variants = sys.argv[4:] or ["base"]
rows = 8192 // shape.ctx_len
data = torch.randint(0, V, (8, rows, shape.ctx_len + 1), device=dev)

_orig_norm_forward = T.RMSNorm.forward
_orig_apply_rope = T._apply_rope
_orig_rope_tables = T._rope_tables


def norm_native(self, x):
    # one reduction in fp32 (accurate), the scaling in the input dtype
    scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
    return x * scale.to(x.dtype) * self.weight


def norm_native2(self, x):
    # everything in the input dtype
    scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps.to(x.dtype))
    return x * scale * self.weight


def norm_vecnorm(self, x):
    # vector_norm accumulates in fp32 without a full-size fp32 copy (if the kernel allows)
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
    scale = torch.rsqrt(n.square() / x.shape[-1] + self.eps)
    return x * (scale.to(x.dtype) * self.weight)


def norm_F(self, x):
    return F.rms_norm(x, (x.shape[-1],), self.weight, 1e-5)


def run(variant, seconds=3.0):
    T.RMSNorm.forward = _orig_norm_forward
    T._apply_rope = _orig_apply_rope
    T._rope_tables = _orig_rope_tables
    parts = set(variant.split("+"))
    if "norm_native" in parts:
        T.RMSNorm.forward = norm_native
    if "norm_native2" in parts:
        T.RMSNorm.forward = norm_native2
    if "norm_vecnorm" in parts:
        T.RMSNorm.forward = norm_vecnorm
    if "norm_F" in parts:
        T.RMSNorm.forward = norm_F
    if "no_norm" in parts:
        T.RMSNorm.forward = lambda self, x: x * self.weight
    if "no_rope" in parts:
        T._apply_rope = lambda x, cos, sin: x
    if "rope_cache" in parts:
        cache = {}

        def tables(positions):
            key = tuple(positions.shape)
            if key not in cache:
                cos, sin = _orig_rope_tables(positions)
                cache[key] = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
            return cache[key]

        T._rope_tables = tables
    torch.manual_seed(0)
    model = T.Transformer(shape, V).to(dev).train()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    fwd = model
    if "compile" in parts:
        fwd = torch.compile(model)

    def step(i):
        ids = data[i % 8]
        x, y = ids[:, :-1], ids[:, 1:]
        if "loss_ac" in parts:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        elif "no_loss" in parts:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x)
            loss = logits.float().mean()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        loss.backward()
        if "no_clip" not in parts:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        if "no_opt" not in parts:
            opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for i in range(4):
        step(i)
    torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    while True:
        step(n).item()
        n += 1
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    return n * rows * shape.ctx_len / el, el / n * 1000


print(f"shape {shape}  params {shape.param_count()/1e6:.2f}M")
for v in variants:
    try:
        tps, ms = run(v)
        print(f"{v:36s} {tps:10.0f} tok/s {ms:8.2f} ms/step", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{v:36s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
