"""Gate iteration evidence for G7(a): how much the gate's garble rate of one model moves with the
sampling, against how much it differs between seeds. Measures each G7 web model on the gate's clean
prompts with 3 sampling rounds (seeds offset by 1000 a round). Run from ml/:
  python -m uv run python ../docs/handoff/calibration/g7_sampling_probe.py
"""

import sys
from pathlib import Path
from statistics import fmean

import torch

from airace_content.manifest import read_vocab
import airace_ml.experiments.gate as G
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok, encode_doc

ROOT = Path(__file__).resolve().parents[3] / "data"
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "ml" / "runs" / "gate-1" / "runs"
dev = torch.device("cuda")
tok = Tok.load(tokenizer_path(ROOT))
known = read_vocab(corpus_dir(ROOT) / "known_vocab.txt")
prefixes = G._web_prefixes(ROOT, tok, 200)
rates = {}
for arm in ("light", "thorough"):
    for s in (1, 2, 3):
        model, _ = load_checkpoint(RUNS / f"g7-web-{arm}-s{s}", dev)
        lm = TorchLM(model, tok, dev)
        rounds = []
        prompts = [encode_doc(tok, p) for p in prefixes]
        for r in range(3):  # batched (one call a round): samples depend on the batch, rates do not
            gens = lm.generate(prompts, max_new_tokens=G.GARBLE_NEW_TOKENS,
                               temperature=G.SAMPLE_TEMPERATURE, top_p=G.SAMPLE_TOP_P, seed=1000 * r)
            rounds.append(G.garble_rate([tok.decode(g.tokens) for g in gens], known))
        rates[(arm, s)] = rounds
        print(f"RESULT {arm} s{s}: rounds {[round(x * 100, 2) for x in rounds]} mean {fmean(rounds) * 100:.2f}%", flush=True)
falls = [fmean(rates[("light", s)]) - fmean(rates[("thorough", s)]) for s in (1, 2, 3)]
print(f"RESULT falls over 600 completions: {[round(f * 100, 2) for f in falls]}", flush=True)
