"""Train one AI with the product trainer exactly as the game does (the default even mix, standard
cleaning, the middle of the learning-style slider) and keep it, with a timing breakdown.

    python train_product.py NAME N_LAYER D_MODEL CTX TOKENS [--seed 1] [--boldness 0.5]
        [--parent DIR] [--mix web=1,books=1,...]

The AI is saved under ml/runs/upgrade/NAME (run from ml/).
"""

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch

import airace_ml.train.trainer as tr
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.data.sampler import MixtureSampler
from airace_ml.model.shape import ModelShape
from airace_ml.train.config import TrainRunConfig

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("n_layer", type=int)
p.add_argument("d_model", type=int)
p.add_argument("ctx", type=int)
p.add_argument("tokens", type=int)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--boldness", type=float, default=0.5)
p.add_argument("--parent", default=None)
p.add_argument("--replay", type=float, default=0.0)
p.add_argument("--mix", default="even")
p.add_argument("--cleaning", default="standard")
p.add_argument("--lr-scale", type=float, default=1.0)  # scales both ends of the learning-style slider
p.add_argument("--warmup", type=float, default=None)  # the warm-up share of the run
p.add_argument("--ns-steps", type=int, default=None)  # Newton-Schulz iterations
p.add_argument("--batch-tokens", type=int, default=None)  # tokens a step (the product uses 8,192)
a = p.parse_args()

import airace_ml.train.config as config_module  # noqa: E402

config_module.CAREFUL_LR *= a.lr_scale
config_module.BOLD_LR *= a.lr_scale
if a.ns_steps is not None:
    from airace_ml.train.optim import MuonAdamW

    MuonAdamW.__init__.__kwdefaults__["ns_steps"] = a.ns_steps

out = Path("runs/upgrade") / a.name
shutil.rmtree(out, ignore_errors=True)
T, N = defaultdict(float), defaultdict(int)


def timed(name, fn, sync=False):
    def wrapper(*args, **kwargs):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if sync:
                torch.cuda.synchronize()
            T[name] += time.perf_counter() - t0
            N[name] += 1

    return wrapper


MixtureSampler.next_batch = timed("sampler", MixtureSampler.next_batch)
tr._Run._forward_backward = timed("forward+backward", tr._Run._forward_backward, sync=True)
tr._Run._train_step = timed("step (total)", tr._Run._train_step, sync=True)
tr._Snapshot.take = classmethod(timed("snapshot", tr._Snapshot.take.__func__, sync=True))
tr._Run._evaluate = timed("held-out + samples", tr._Run._evaluate, sync=True)
tr._Run._save_resume = timed("resume save", tr._Run._save_resume, sync=True)
SAVED = {"bytes": 0}
_save = tr._Run._save_resume


def _measured_save(self, *args, **kwargs):
    result = _save(self, *args, **kwargs)
    size = sum(f.stat().st_size for f in (out / tr.RESUME_DIR).rglob("*") if f.is_file())
    SAVED["bytes"] = max(SAVED["bytes"], size)
    return result


tr._Run._save_resume = _measured_save
tr._prepare = timed("setup", tr._prepare)

if a.mix == "even":
    mix = {k: 1.0 for k in DATASET_IDS}
else:
    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
from airace_ml.data.prep import PrepConfig  # noqa: E402

cfg = TrainRunConfig(
    run_id=a.name,
    seed=a.seed,
    shape=ModelShape(a.n_layer, a.d_model, a.ctx),
    token_budget=a.tokens,
    mixture=mix,
    boldness=a.boldness,
    prep=PrepConfig(cleaning=a.cleaning),
    parent_dir=a.parent,
    replay=a.replay,
    **({"batch_tokens": a.batch_tokens} if a.batch_tokens else {}),
)
t0 = time.perf_counter()
res = tr.train_run(cfg, out_dir=out, device=torch.device("cuda"), checkpoint_seconds=30.0)
wall = time.perf_counter() - t0
line = {
    "name": a.name,
    "wall_s": round(wall, 1),
    "tok_per_s": round(res.tokens / wall),
    "status": res.status,
    "final_loss": round(res.final_loss, 4),
    "heldout": {k: round(v, 3) for k, v in res.heldout_losses.items()},
    "parts_s": {k: round(v, 2) for k, v in sorted(T.items(), key=lambda kv: -kv[1])},
    "peak_vram_mb": round(torch.cuda.max_memory_reserved() / 2**20),
    "resume_mb": round(SAVED["bytes"] / 2**20, 1),
}
print(json.dumps(line))
with open("runs/upgrade/product_runs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(line) + "\n")
