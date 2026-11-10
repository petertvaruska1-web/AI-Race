"""Raw model throughput on this card: forward + backward + AdamW step on random tokens.

Finds the hardware ceiling per shape and batch, to compare with what the trainer achieves.
"""

import sys
import time

import torch
import torch.nn.functional as F

from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer

dev = torch.device("cuda")
V = 4096


def bench(shape, batch_tokens, seconds=3.0, dtype=torch.bfloat16, micro=None):
    torch.manual_seed(0)
    model = Transformer(shape, V).to(dev).train()
    params = [p for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    rows = max(1, batch_tokens // shape.ctx_len)
    data = torch.randint(0, V, (8, rows, shape.ctx_len + 1), device=dev)
    micro_rows = rows if micro is None else max(1, micro // shape.ctx_len)
    torch.cuda.reset_peak_memory_stats()

    def step(i):
        ids = data[i % 8]
        x, y = ids[:, :-1], ids[:, 1:]
        total = None
        for s in range(0, rows, micro_rows):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                logits = model(x[s : s + micro_rows])
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), y[s : s + micro_rows].flatten(), reduction="sum"
            ) / (rows * shape.ctx_len)
            loss.backward()
            total = loss.detach() if total is None else total + loss.detach()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return total

    for i in range(3):
        step(i)
    torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    while True:
        step(n).item()  # one sync per step, like the trainer
        n += 1
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2**20
    del model, opt, data
    torch.cuda.empty_cache()
    return n * rows * shape.ctx_len / el, mem, el / n * 1000


shapes = [
    ("tiny", ModelShape(4, 128, 256)),
    ("small", ModelShape(6, 192, 256)),
    ("medium", ModelShape(8, 256, 256)),
    ("medium512", ModelShape(8, 256, 512)),
    ("large", ModelShape(10, 384, 512)),
    ("xl", ModelShape(12, 512, 512)),
]
which = sys.argv[1:] or [n for n, _ in shapes]
print(f"{'shape':10s} {'params':>8s} {'batch':>7s} {'micro':>7s} {'tok/s':>10s} {'ms/step':>8s} {'MiB':>7s}")
for name, shape in shapes:
    if name not in which:
        continue
    for bt in (8192, 16384, 32768, 65536):
        for micro in (8192, None):
            if micro == 8192 and bt == 8192:
                continue
            try:
                tps, mem, ms = bench(shape, bt, micro=micro)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{name:10s} {shape.param_count()/1e6:7.2f}M {bt:7d} {str(micro):>7s}        OOM")
                continue
            print(
                f"{name:10s} {shape.param_count()/1e6:7.2f}M {bt:7d} {str(micro):>7s} {tps:10.0f} {ms:8.1f} {mem:7.0f}",
                flush=True,
            )
