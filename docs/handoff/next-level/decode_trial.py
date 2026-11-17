"""Task 18's trial: do the decoding rules that block repeats or unlikely words help small talk?

    python ../docs/handoff/next-level/decode_trial.py --card runs/scorecard-e3-text2 \
        --out runs/decode-trial.json NAME=CKPT_DIR [NAME=CKPT_DIR ...]

Run from ml/ (GPU, a few minutes a model). The judge, the fit threshold and the on-topic rule come
from the yardstick root (``--yardstick-root``, default ``../data``) and the calibration cached in
``--card`` (a finished scorecard's folder), so the verdicts are the scorecard's own. Each model
answers the scorecard's 20 small-talk lines five times under each setting:

* ``served``: what the game serves today (temperature 0.6, top-p 0.9);
* ``block4``: the same, and a token that would repeat a run of four tokens of the reply is blocked;
* ``minp05`` / ``minp10``: min-p 0.05 / 0.1 in place of top-p (top-p 1.0).

It prints, per model and setting, the rates the scorecard reports for small talk (well-formed,
fits the question, on topic, both, cut off, loops, life claims) and stores every reply, in the
shape ``blind_read.py sheet`` reads (``LABEL=FILE.json:MODEL/SETTING``).
"""

import argparse
import json
from pathlib import Path

import torch

from airace_ml.experiments import chatprobes, scorecard
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import tokenizer_path
from airace_ml.tokenizer import Tok

SETTINGS = {
    "served": {"top_p": 0.9, "block_repeat": 0, "min_p": 0.0},
    "block4": {"top_p": 0.9, "block_repeat": 4, "min_p": 0.0},
    "minp05": {"top_p": 1.0, "block_repeat": 0, "min_p": 0.05},
    "minp10": {"top_p": 1.0, "block_repeat": 0, "min_p": 0.1},
}

p = argparse.ArgumentParser()
p.add_argument("models", nargs="+", metavar="NAME=CKPT_DIR")
p.add_argument("--card", required=True, type=Path)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--yardstick-root", type=Path, default=Path("../data"))
p.add_argument("--samples", type=int, default=5)
a = p.parse_args()

dev = torch.device("cuda")
root = a.yardstick_root
identity = scorecard._identity(root, "the yardstick root")
instruments = scorecard._load_instruments(root, dev, identity, a.card, print)
tok = Tok.load(tokenizer_path(root))
rows, models = [], {}
for spec in a.models:
    name, _, ckpt = spec.partition("=")
    model, _ = load_checkpoint(ckpt, device=dev)
    lm = TorchLM(model, tok, dev)
    for setting, values in SETTINGS.items():
        chatprobes.SERVING_TOP_P = values["top_p"]
        chatprobes.SERVING_BLOCK_REPEAT = values["block_repeat"]
        chatprobes.SERVING_MIN_P = values["min_p"]
        transcript: list[dict] = []
        rates = chatprobes.small_talk(
            lm,
            instruments.well_formed,
            samples=a.samples,
            fits=instruments.fits,
            on_topic=instruments.on_topic,
            transcript=transcript,
        )
        models[f"{name}/{setting}"] = {"rates": rates, "examples": {"small_talk": transcript}}
        rows.append((name, setting, rates))
        print(f"{name:10} {setting:7} " + " ".join(f"{k} {v:.0%}" for k, v in rates.items()), flush=True)
a.out.write_text(
    json.dumps({"decode_trial": 1, "settings": SETTINGS, "models": models}, indent=1),
    encoding="utf-8",
)
print(f"written to {a.out}")
