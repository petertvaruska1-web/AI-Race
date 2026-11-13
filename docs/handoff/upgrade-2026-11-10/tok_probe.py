"""How much text a token holds with tok-v1, and what a larger vocabulary would hold.

Decodes a sample of every dataset, trains byte-level BPE vocabularies of several sizes on half of
it (as tok-v1 was trained, plus an option that adds casually typed copies of chat turns) and
reports bytes per token on the other half, and how casually typed greetings split.

    python tok_probe.py [--docs 3000] [--sizes 4096,8192,16384]

Run from ml/.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.paths import data_root, tokenizer_path
from airace_ml.tokenizer import Tok, train_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--docs", type=int, default=3000)
p.add_argument("--sizes", default="4096,8192,16384")
a = p.parse_args()

root = data_root()
tok = Tok.load(tokenizer_path(root))
rng = np.random.default_rng(0)
train: list[str] = []
test: dict[str, list[str]] = {}
for name in DATASET_IDS:
    corpus = Corpus.open(root / "corpus" / "v1" / name)
    n = len(corpus.offsets) - 1 if hasattr(corpus, "offsets") else corpus.n_docs
    picks = rng.choice(n, size=min(a.docs, n), replace=False)
    texts = [tok.decode(corpus.doc(int(i)).tolist()) for i in picks]
    half = len(texts) // 2
    train += texts[:half]
    test[name] = texts[half:]

WORDS = ["hello", "Hello", "hi there", "what is your name", "thanks", "how are you", "tell me a story"]


def report(label: str, t: Tok) -> None:
    row = []
    total_b = total_t = 0
    for name, texts in test.items():
        b = sum(len(x.encode("utf-8")) for x in texts)
        n = sum(len(t.encode(x)) for x in texts)
        total_b += b
        total_t += n
        row.append(f"{name[:5]} {b / n:.2f}")
    splits = " | ".join(f"{w}={len(t.encode(w))}" for w in WORDS)
    print(f"{label:22s} bytes/token {total_b / total_t:.3f}  [{' '.join(row)}]\n{'':22s} {splits}")


report("tok-v1 (4096)", tok)
casual = [x.lower() for x in train[::7]] + [x.lower().rstrip(".?!") for x in train[3::11]]
with tempfile.TemporaryDirectory() as tmp:
    for size in map(int, a.sizes.split(",")):
        report(f"retrained {size}", train_tokenizer(train, Path(tmp) / f"a{size}.json", size))
        report(
            f"retrained {size} +casual",
            train_tokenizer(train + casual, Path(tmp) / f"b{size}.json", size),
        )
