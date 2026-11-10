"""Where a real training run's wall time goes (the product trainer, untouched, with timers)."""

import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

import airace_ml.train.trainer as tr
from airace_ml.data.sampler import MixtureSampler
from airace_ml.model.shape import ModelShape
from airace_ml.train.config import TrainRunConfig

n_layer, d_model, ctx, tokens = (int(a) for a in sys.argv[1:5])
out = Path(sys.argv[5])
shutil.rmtree(out, ignore_errors=True)

T = defaultdict(float)
N = defaultdict(int)


def timed(name, fn, sync=False):
    def wrapper(*a, **k):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if sync:
                torch.cuda.synchronize()
            T[name] += time.perf_counter() - t0
            N[name] += 1

    return wrapper


MixtureSampler.next_batch = timed("sampler.next_batch", MixtureSampler.next_batch)
tr._Run._forward_backward = timed("forward_backward", tr._Run._forward_backward, sync=True)
tr._Run._train_step = timed("train_step(total)", tr._Run._train_step, sync=True)
tr._Snapshot.take = classmethod(timed("snapshot.take", tr._Snapshot.take.__func__, sync=True))
tr._Run._evaluate = timed("evaluate(heldout+samples)", tr._Run._evaluate, sync=True)
tr._heldout_losses = timed("  heldout_losses", tr._heldout_losses, sync=True)
tr._Run._save_resume = timed("save_resume", tr._Run._save_resume, sync=True)
tr._Run._emit = timed("emit(telemetry)", tr._Run._emit)
tr._prepare = timed("prepare(setup)", tr._prepare)
tr._Run._finish = timed("finish(save)", tr._Run._finish, sync=True)

mix = {k: 1.0 for k in ("web", "books", "educational", "conversations", "code", "reasoning", "facts", "creative")}
cfg = TrainRunConfig(
    run_id="prof", seed=1, shape=ModelShape(n_layer, d_model, ctx), token_budget=tokens, mixture=mix
)
t0 = time.perf_counter()
res = tr.train_run(cfg, out_dir=out, device=torch.device("cuda"))
wall = time.perf_counter() - t0
print(f"shape {n_layer}x{d_model}@{ctx} tokens {res.tokens:,} steps {res.steps} wall {wall:.1f}s "
      f"-> {res.tokens / wall:,.0f} tok/s (loop only: {res.tokens / (wall - T['prepare(setup)']):,.0f})")
print(f"final_loss {res.final_loss:.4f} heldout { {k: round(v, 3) for k, v in res.heldout_losses.items()} }")
for name in sorted(T, key=lambda k: -T[k]):
    print(f"  {name:28s} {T[name]:8.2f}s {100 * T[name] / wall:5.1f}%  n={N[name]:6d}  {1000 * T[name] / max(1, N[name]):8.2f} ms each")
shutil.rmtree(out, ignore_errors=True)
