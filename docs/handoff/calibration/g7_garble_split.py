"""Gate iteration evidence for G7(a): the gate's garble-like-word rate of the light and thorough web
models, split into mojibake words and one-edit typo words, and by whether the held-out prompt was a
clean document or a noisy one.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/g7_garble_split.py [RUNS_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import torch

from airace_content.manifest import read_vocab
from airace_content.textproc import MOJIBAKE_CHARS
from airace_ml.data.corpus import Corpus
from airace_ml.data.prep import heldout_docs
from airace_ml.evals.creativity import words
from airace_ml.experiments.gate import (
    _GARBLE_WORD,
    GARBLE_MIN_TYPO_LENGTH,
    GARBLE_NEW_TOKENS,
    GARBLE_PREFIX_TOKENS,
    MEASURE_SEED,
    SAMPLE_TEMPERATURE,
    SAMPLE_TOP_P,
    _one_edit_from_known,
    trim_to_word,
)
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

ROOT = Path(__file__).resolve().parents[3] / "data"
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "ml/runs/gate-1/runs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tok.load(tokenizer_path(ROOT))
known_vocab = read_vocab(corpus_dir(ROOT) / "known_vocab.txt")
known = {w for entry in known_vocab for w in words(entry)}
alphabet = "".join(sorted({c for w in known for c in w}))

web = Corpus.open(corpus_dir(ROOT) / "web")
order = np.random.default_rng(MEASURE_SEED).permutation(heldout_docs(web))[:200]
prefixes, clean = [], []
for i in order:
    doc = web.doc(int(i))
    start = 1 if len(doc) and doc[0] == tok.bos_id else 0
    prefixes.append(trim_to_word(tok.decode(doc[start : start + GARBLE_PREFIX_TOKENS].tolist())))
    clean.append(web.tags.noise_kind[int(i)] == 0)
clean = np.array(clean)


def split(texts):
    moji = typo = total = 0
    for text in texts:
        for raw in _GARBLE_WORD.findall(text):
            total += 1
            if any(c in MOJIBAKE_CHARS for c in raw):
                moji += 1
            else:
                w = raw.lower()
                typo += w not in known and len(w) >= GARBLE_MIN_TYPO_LENGTH and _one_edit_from_known(w, known, alphabet)
    return moji / max(total, 1), typo / max(total, 1)


print(f"prompts: {clean.sum()} clean, {(~clean).sum()} noisy")
for arm in ("light", "thorough"):
    for s in (1, 2, 3):
        folder = RUNS / f"g7-web-{arm}-s{s}"
        if not folder.exists():
            continue
        model, _ = load_checkpoint(folder, device)
        lm = TorchLM(model, tok, device)
        outs = [lm.complete(p, max_new_tokens=GARBLE_NEW_TOKENS, temperature=SAMPLE_TEMPERATURE,
                            top_p=SAMPLE_TOP_P, seed=i) for i, p in enumerate(prefixes)]
        m_all, t_all = split(outs)
        m_c, t_c = split([o for o, c in zip(outs, clean) if c])
        m_n, t_n = split([o for o, c in zip(outs, clean) if not c])
        print(f"{arm:8} s{s}: all mojibake {m_all:.2%} typo {t_all:.2%} | clean prompts mojibake {m_c:.2%} "
              f"typo {t_c:.2%} | noisy prompts mojibake {m_n:.2%} typo {t_n:.2%}")
