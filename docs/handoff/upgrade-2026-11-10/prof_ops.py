"""Where the GPU time of one training step goes, by operation."""

import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer

dev = torch.device("cuda")
V = 4096
shape = ModelShape(*[int(a) for a in sys.argv[1:4]]) if len(sys.argv) > 3 else ModelShape(4, 128, 256)
torch.manual_seed(0)
model = Transformer(shape, V).to(dev).train()
params = list(model.parameters())
opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
rows = 8192 // shape.ctx_len
data = torch.randint(0, V, (rows, shape.ctx_len + 1), device=dev)


def step():
    x, y = data[:, :-1], data[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum") / (
        rows * shape.ctx_len
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


for _ in range(5):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step().item()
    torch.cuda.synchronize()
print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=32, max_name_column_width=46
    )
)
