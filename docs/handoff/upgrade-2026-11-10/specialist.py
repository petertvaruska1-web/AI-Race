"""Why a specialist's own skill moved between engines: the gate's G3 recipe (Small brain 6x192,
8,388,608 tokens, 60% of one dataset) trained with other casual-typing shares and learning
styles, then scored on chosen benchmark categories.

    python specialist.py TARGET [--seeds 1,2] [--casual 0.3,0.0] [--boldness 0.5,0.25]
        [--categories knowledge] [--tokens 8388608]

Run from ml/. Appends to runs/upgrade/specialist.jsonl.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

import airace_ml.train.trainer as tr
from airace_ml.evals.suite import run_benchmarks
from airace_ml.experiments.configs import mix_heavy
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.paths import data_root, tokenizer_path
from airace_ml.tokenizer import Tok
from airace_ml.train.config import TrainRunConfig

p = argparse.ArgumentParser()
p.add_argument("target")
p.add_argument("--seeds", default="1,2")
p.add_argument("--casual", default="0.3,0.0")
p.add_argument("--boldness", default="0.5")
p.add_argument("--categories", default="knowledge")
p.add_argument("--tokens", type=int, default=8_388_608)
p.add_argument("--share", type=float, default=0.6)
p.add_argument("--other-share", type=float, default=None)  # embedding and gain rate / matrix rate
p.add_argument("--embed-wd", type=float, default=None)  # the embedding's weight decay
p.add_argument("--decay-frac", type=float, default=None)
p.add_argument("--min-lr", type=float, default=None)
p.add_argument("--tag", default="")
a = p.parse_args()

import airace_ml.train.config as config_module  # noqa: E402
import airace_ml.train.schedule as schedule_module  # noqa: E402
from airace_ml.train.optim import MuonAdamW  # noqa: E402

if a.other_share is not None:
    config_module.OTHER_LR_SHARE = a.other_share
if a.embed_wd is not None:
    MuonAdamW.__init__.__kwdefaults__["weight_decay"] = a.embed_wd
if a.decay_frac is not None:
    schedule_module.DECAY_FRAC = a.decay_frac
if a.min_lr is not None:
    schedule_module.MIN_LR_RATIO = a.min_lr
dev = torch.device("cuda")
tok = Tok.load(tokenizer_path(data_root()))
cats = a.categories.split(",")
for casual in map(float, a.casual.split(",")):
    for bold in map(float, a.boldness.split(",")):
        for seed in map(int, a.seeds.split(",")):
            name = f"spec-{a.target}{a.tag}-c{casual}-b{bold}-s{seed}"
            out = Path("runs/upgrade") / name
            shutil.rmtree(out, ignore_errors=True)
            tr.CASUAL_SHARE = casual
            cfg = TrainRunConfig(
                run_id=f"g3-{a.target}-s{seed}",
                seed=seed,
                shape=ModelShape(6, 192, 256),
                token_budget=a.tokens,
                mixture=mix_heavy(a.target, a.share),
                boldness=bold,
            )
            res = tr.train_run(cfg, out_dir=out, device=dev)
            model, _ = load_checkpoint(out, device=dev)
            report = run_benchmarks(TorchLM(model, tok, dev), tok, categories=cats)
            line = {
                "name": name,
                "casual": casual,
                "boldness": bold,
                "seed": seed,
                "heldout_target": round(res.heldout_losses[a.target], 3),
                "scores": {c: round(s.score, 1) for c, s in report.scores.items()},
            }
            print(json.dumps(line), flush=True)
            with open("runs/upgrade/specialist.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
