"""Gate iteration evidence for G7(a): what the cleaning levels keep under candidate known-word ranges
of ``textproc.quality_score`` (``_KNOWN_SHARE_RANGE``), on decoded documents of the built corpus.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/quality_range_sim.py
"""

import sys
from pathlib import Path

import numpy as np

import airace_content.textproc as textproc
from airace_content.manifest import read_vocab
from airace_ml.data.corpus import NOISE_KINDS, Corpus
from airace_ml.data.prep import CLEANING_THRESHOLDS
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3] / "data"
RANGES = [(0.5, 0.9), (0.7, 0.95), (0.75, 0.95), (0.8, 0.97)]
tok = Tok.load(tokenizer_path(ROOT))
known = read_vocab(corpus_dir(ROOT) / "known_vocab.txt")
rng = np.random.default_rng(0)


def keeps(texts: list[str]) -> str:
    out = []
    for r in RANGES:
        textproc._KNOWN_SHARE_RANGE = r
        q = np.array([textproc.quality_score(t, known) for t in texts])
        out.append("/".join(f"{(q >= t).mean():.2f}" for t in CLEANING_THRESHOLDS.values()))
    return "  ".join(f"{o:>14}" for o in out)


print(f"{'':24}" + "  ".join(f"{str(r):>14}" for r in RANGES) + "   (light/standard/thorough keep)")
web = Corpus.open(corpus_dir(ROOT) / "web")
for k in range(len(NOISE_KINDS)):
    idx = np.flatnonzero(web.tags.noise_kind == k)
    if len(idx):
        texts = [tok.decode(web.doc(int(i))) for i in rng.choice(idx, min(400, len(idx)), replace=False)]
        print(f"{'web ' + NOISE_KINDS[k]:24}" + keeps(texts))
for ds in ("books", "educational", "creative", "conversations", "facts"):
    c = Corpus.open(corpus_dir(ROOT) / ds)
    idx = np.flatnonzero(c.tags.noise_kind == 0)
    texts = [tok.decode(c.doc(int(i))) for i in rng.choice(idx, 400, replace=False)]
    print(f"{ds + ' clean':24}" + keeps(texts))
