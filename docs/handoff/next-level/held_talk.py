"""Draw trained AIs' replies to small-talk lines, for the blind reading.

    python ../docs/handoff/next-level/held_talk.py --out FILE.json NAME=CKPT_DIR [NAME=CKPT_DIR ...]
                                                   [--system "You are Zula, an AI made by Pengu."]
                                                   [--lines held|standard]

Run from ml/ (uses the GPU for a minute). Each AI answers every line of
``airace_content.held.HELD_SMALL_TALK`` (``--lines held``, the default) or of the scorecard's
``airace_ml.experiments.chatprobes.SMALL_TALK`` (``--lines standard``) five times (seeds 0 to 4) with the settings the game
serves replies with. The file has the shape of a scorecard's stored small talk, so
``blind_read.py sheet`` reads it: ``LABEL=FILE.json:NAME``.
"""

import argparse
import json
from pathlib import Path

import torch

from airace_content.held import HELD_SMALL_TALK
from airace_ml.experiments.chatprobes import SMALL_TALK
from airace_ml.infer.lm import TorchLM
from airace_ml.infer.serving import (
    SERVING_TEMPERATURE,
    SERVING_TOP_P,
)
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import data_root, tokenizer_path
from airace_ml.tokenizer import Tok

p = argparse.ArgumentParser()
p.add_argument("models", nargs="+", metavar="NAME=CKPT_DIR")
p.add_argument("--out", required=True)
p.add_argument("--samples", type=int, default=5)
p.add_argument("--system", default=None)
p.add_argument("--lines", choices=("held", "standard"), default="held")
a = p.parse_args()

dev = torch.device("cuda")
tok = Tok.load(tokenizer_path(data_root()))
models = {}
for spec in a.models:
    name, _, ckpt = spec.partition("=")
    model, _ = load_checkpoint(ckpt, device=dev)
    lm = TorchLM(model, tok, dev)
    talk = []
    for prompt in HELD_SMALL_TALK if a.lines == "held" else SMALL_TALK:
        for seed in range(a.samples):
            reply = lm.chat_reply(
                [("user", prompt)],
                temperature=SERVING_TEMPERATURE,
                top_p=SERVING_TOP_P,
                seed=seed,
                stop_on_loop=True,
                system=a.system,
            )
            talk.append({"prompt": prompt, "sample": seed, "reply": reply})
    models[name] = {"examples": {"small_talk": talk}}
    print(f"{name}: {len(talk)} replies")
Path(a.out).write_text(
    json.dumps(
        {
            "held_talk": 1,
            "lines": a.lines,
            "system": a.system,
            "models": models,
        },
        indent=1,
    ),
    encoding="utf-8",
)
