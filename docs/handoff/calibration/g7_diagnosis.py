"""Gate iteration evidence for G7(a): where the web text's garble-like words come from.

Per noise kind of the web dataset: how many documents, their quality, how many each cleaning
level keeps, and the garble-like word rate (the gate's own measure) of a sample of them.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/g7_diagnosis.py
"""

import sys
from pathlib import Path

import numpy as np

from airace_content.manifest import read_vocab
from airace_ml.data.corpus import NOISE_KINDS, Corpus
from airace_ml.data.prep import CLEANING_THRESHOLDS
from airace_ml.experiments.gate import garble_rate
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3] / "data"
tok = Tok.load(tokenizer_path(ROOT))
known = read_vocab(corpus_dir(ROOT) / "known_vocab.txt")
c = Corpus.open(corpus_dir(ROOT) / "web")
q, kinds = c.tags.quality, c.tags.noise_kind
rng = np.random.default_rng(0)
print(f"{'kind':12} {'docs':>6} {'q mean':>7} " + " ".join(f"{k:>9}" for k in CLEANING_THRESHOLDS)
      + "   garble-like words")
total_words = {}
for k, name in enumerate(NOISE_KINDS):
    idx = np.flatnonzero(kinds == k)
    if not len(idx):
        continue
    keeps = " ".join(f"{(q[idx] >= t).mean():9.2f}" for t in CLEANING_THRESHOLDS.values())
    sample = rng.choice(idx, size=min(300, len(idx)), replace=False)
    texts = [tok.decode(c.doc(int(i))) for i in sample]
    print(f"{name:12} {len(idx):6} {q[idx].mean():7.2f} {keeps}   {garble_rate(texts, known):.3%}")
