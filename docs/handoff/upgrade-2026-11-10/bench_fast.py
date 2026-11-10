"""Speed and equality of the faster forward variants."""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import fastmodel  # noqa: E402

import airace_ml.model.transformer as T  # noqa: E402
from airace_ml.model.shape import ModelShape  # noqa: E402

dev = torch.device("cuda")
V = 4096
shape = ModelShape(*[int(a) for a in sys.argv[1:4]])
variants = sys.argv[4:]
rows = 8192 // shape.ctx_len
data = torch.randint(0, V, (8, rows, shape.ctx_len + 1), device=dev)


def equality(flags):
    """Max abs logit difference vs the original forward, fp32 on the CPU, and grads."""
    torch.manual_seed(0)
    fastmodel.apply("")
    model = T.Transformer(ModelShape(shape.n_layer, shape.d_model, 64), V)
    with torch.no_grad():
        for p in model.parameters():  # non-trivial gains
            if p.dim() == 1:
                p.mul_(torch.rand_like(p) + 0.5)
    ids = torch.randint(0, V, (3, 48))
    ref = model(ids)
    ref.square().mean().backward()
    g_ref = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    fastmodel.apply(flags.replace("+cache", "").replace("cache", ""))
    out = model(ids)
    out.square().mean().backward()
    g = [p.grad.clone() for p in model.parameters()]
    gdiff = max((a - b).abs().max().item() for a, b in zip(g, g_ref, strict=True))
    return (out - ref).abs().max().item(), gdiff


def run(flags, loss_mode, seconds=3.0):
    fastmodel.apply(flags)
    torch.manual_seed(0)
    model = T.Transformer(shape, V).to(dev).train()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    n_tok = rows * shape.ctx_len

    def step(i):
        ids = data[i % 8]
        x, y = ids[:, :-1], ids[:, 1:]
        if loss_mode == "ac":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum") / n_tok
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum") / n_tok
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
    return n * n_tok / el, el / n * 1000


print(f"shape {shape}  params {shape.param_count()/1e6:.2f}M")
base = None
for v in variants:
    flags, _, loss_mode = v.partition("/")
    flags = "" if flags == "base" else flags
    try:
        dl, dg = equality(flags)
        tps, ms = run(flags, loss_mode)
        base = base or tps
        print(f"{v:34s} {tps:10.0f} tok/s {ms:7.2f} ms  x{tps/base:4.2f}   cpu logit diff {dl:.1e} grad diff {dg:.1e}", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"{v:34s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
