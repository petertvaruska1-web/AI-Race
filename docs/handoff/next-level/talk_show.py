"""Show a checkpoint's small-talk replies as the scorecard draws them, with the judge's verdict.

    python talk_show.py CKPT_DIR [CKPT_DIR ...] [--bad-only] [--system "You are Zula, an AI."]

Run from ml/. Uses the data root's judge (the yardstick).
"""

import argparse

import torch

from airace_ml.evals.judge import Judge, is_well_formed
from airace_ml.experiments.chatprobes import SMALL_TALK
from airace_ml.infer.lm import TorchLM
from airace_ml.infer.serving import SERVING_TEMPERATURE, SERVING_TOP_P
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import data_root, judge_dir, tokenizer_path
from airace_ml.tokenizer import Tok

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="+")
p.add_argument("--bad-only", action="store_true")
p.add_argument("--samples", type=int, default=3)
p.add_argument("--system", default=None)
a = p.parse_args()

dev = torch.device("cuda")
root = data_root()
tok = Tok.load(tokenizer_path(root))
judge = Judge.load(judge_dir(root), dev)
for ckpt in a.ckpts:
    model, _ = load_checkpoint(ckpt, device=dev)
    lm = TorchLM(model, tok, dev)
    good = total = 0
    lines = []
    for prompt in SMALL_TALK:
        for seed in range(a.samples):
            reply = lm.chat_reply(
                [("user", prompt)],
                temperature=SERVING_TEMPERATURE,
                top_p=SERVING_TOP_P,
                seed=seed,
                stop_on_loop=True,
                system=a.system,
            )
            ok = is_well_formed(reply, judge)
            good += ok
            total += 1
            if not (a.bad_only and ok):
                lines.append(f"  {'ok ' if ok else 'BAD'} {prompt!r} -> {reply!r}")
    print(f"== {ckpt}: {good}/{total} well-formed")
    print("\n".join(lines))
