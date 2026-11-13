"""How a checkpoint copes with casually typed prompts: facts asked in training phrasing, typed
four ways (greedy, right when the reply holds the answer), and small talk judged for fluency.

    python probe_casual.py CKPT_DIR [CKPT_DIR ...] [--show]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fastmodel  # noqa: E402

from airace_ml.experiments.chatprobes import SMALL_TALK, STYLES  # noqa: E402
from airace_ml.evals.judge import Judge, _has_reply_words  # noqa: E402
from airace_ml.infer.lm import TorchLM  # noqa: E402
from airace_ml.model.checkpoint import load_checkpoint  # noqa: E402
from airace_ml.paths import data_root, judge_dir, tokenizer_path  # noqa: E402
from airace_ml.skills.kb import load_kb  # noqa: E402
from airace_ml.tokenizer import Tok, encode_chat  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--facts", type=int, default=120)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    fastmodel.apply("norm")
    root = data_root()
    dev = torch.device("cuda")
    tok = Tok.load(tokenizer_path(root))
    judge = Judge.load(judge_dir(root), dev)
    bar = judge.calibration.conv_reply_p90
    kb = load_kb()
    rels = kb.relations if hasattr(kb, "relations") else kb._relations
    rng = np.random.default_rng(0)
    facts = []
    for name, rel in rels.items():
        fs = kb.facts_for(name)
        for k in rng.permutation(len(fs))[: max(1, a.facts // len(rels))]:
            facts.append((rel.chat_templates[0][0].replace("{s}", fs[k].subject), [x.lower() for x in kb.accepted_answers(fs[k])]))
    for ck in a.ckpts:
        model, _ = load_checkpoint(Path(ck), dev)
        lm = TorchLM(model, tok, dev)
        line = f"{Path(ck).name:26s}"
        for style, fn in STYLES.items():
            prompts = [encode_chat(tok, [("user", fn(q))], add_generation_prompt=True) for q, _ in facts]
            gens = lm.generate(prompts, max_new_tokens=24, temperature=0.0, top_p=1.0, seed=0)
            right = sum(any(ans in tok.decode(g.tokens).lower() for ans in answers) for g, (_, answers) in zip(gens, facts, strict=True))
            line += f" | facts {style}: {100 * right / len(facts):3.0f}%"
        print(line, flush=True)
        line = f"{'':26s}"
        for style, fn in STYLES.items():
            replies = [lm.chat_reply([("user", fn(q))], max_new_tokens=48, temperature=0.5, top_p=0.95, seed=i + 100 * s) for s in range(2) for i, q in enumerate(SMALL_TALK)]
            nll = judge.nll_per_token([tok.encode(r.strip()) for r in replies])
            ok = sum(_has_reply_words(r) and n <= bar for r, n in zip(replies, nll, strict=True))
            line += f" | talk {style}: {100 * ok / len(replies):3.0f}%"
            if a.show and style in ("clean", "lower_nopunct"):
                for q, r in list(zip([fn(q) for q in SMALL_TALK], replies))[:8]:
                    print(f"      [{style}] {q!r} -> {r!r}")
        print(line, flush=True)
        del model, lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
