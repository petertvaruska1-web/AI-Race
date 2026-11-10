"""How well a checkpoint talks: the gate's legibility measure, spelling, and sample replies.

    python probe_quality.py CKPT_DIR [CKPT_DIR ...] [--temps 0.8,0.6] [--top_ps 0.95] [--show 6]
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import fastmodel  # noqa: E402

from airace_ml.evals.judge import Judge, _has_reply_words  # noqa: E402
from airace_ml.experiments.gate import garble_rate  # noqa: E402
from airace_ml.infer.lm import TorchLM  # noqa: E402
from airace_ml.model.checkpoint import load_checkpoint  # noqa: E402
from airace_ml.paths import corpus_dir, data_root, judge_dir, tokenizer_path  # noqa: E402
from airace_ml.personality.fingerprint import load_probes  # noqa: E402
from airace_ml.skills.checkers import words  # noqa: E402
from airace_ml.tokenizer import Tok  # noqa: E402

EVERYDAY = [
    "hello",
    "Hello, how are you?",
    "what is your name?",
    "is 3 bigger than 2?",
    "What is the capital of France?",
    "Tell me a short story about a dog.",
    "What color is the sky?",
    "Can you help me with my homework?",
    "What do you like to eat?",
    "Say something nice.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--temps", default="0.8")
    p.add_argument("--top_ps", default="0.95")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--tokens", type=int, default=64)
    a = p.parse_args()
    fastmodel.apply("norm")
    root = data_root()
    dev = torch.device("cuda")
    tok = Tok.load(tokenizer_path(root))
    judge = Judge.load(judge_dir(root), dev)
    known = set((corpus_dir(root) / "known_vocab.txt").read_text(encoding="utf-8").split())
    probes = load_probes()
    chosen = [q for q in probes if q.kind == "open"][:10] + [q for q in probes if q.kind == "help"][:10]
    bar = judge.calibration.conv_reply_p90
    for ck in a.ckpts:
        model, meta = load_checkpoint(Path(ck), dev)
        lm = TorchLM(model, tok, dev)
        for temp in [float(t) for t in a.temps.split(",")]:
            for top_p in [float(t) for t in a.top_ps.split(",")]:
                asked = [q.text for _ in range(a.samples) for q in chosen]
                replies = [
                    lm.chat_reply([("user", q)], max_new_tokens=a.tokens, temperature=temp, top_p=top_p, seed=i)
                    for i, q in enumerate(asked)
                ]
                nll = judge.nll_per_token([tok.encode(r.strip()) for r in replies])
                ok = [_has_reply_words(r) and n <= bar for r, n in zip(replies, nll, strict=True)]
                wordy = [_has_reply_words(r) for r in replies]
                n_words = statistics.mean(len(words(r)) for r in replies)
                fin = [n for n in nll if n != float("inf")]
                print(
                    f"{Path(ck).name:28s} T={temp:<4} p={top_p:<4} well-formed {sum(ok):3d}/{len(ok)} "
                    f"({100*sum(ok)/len(ok):4.0f}%)  word-ok {100*sum(wordy)/len(wordy):4.0f}%  "
                    f"judge nll median {statistics.median(fin):.2f} (bar {bar:.2f})  "
                    f"garble {100*garble_rate(replies, known):.1f}%  words/reply {n_words:.1f}",
                    flush=True,
                )
                if a.show:
                    for q in EVERYDAY[: a.show]:
                        r = lm.chat_reply([("user", q)], max_new_tokens=48, temperature=temp, top_p=top_p, seed=7)
                        print(f"      > {q}\n        {r!r}")
        del model, lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
