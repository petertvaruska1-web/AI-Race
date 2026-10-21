"""Task 15 step 3: sanity checks of the full corpus build.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/corpus_checks.py

Per dataset: 3 random documents, the quality histogram and the share each cleaning level keeps,
mojibake outside the deliberately garbled documents, and (for educational and books) the reading
level; for books, how many passages end at a sentence end. Writes the samples to
corpus_samples.txt next to this script and prints the numbers.
"""

import json
import sys
import re
from pathlib import Path

import numpy as np

from airace_content.manifest import read_vocab
from airace_content.textproc import MOJIBAKE_CHARS, simplicity
from airace_ml.data.corpus import DATASET_IDS, NOISE_KINDS, Corpus
from airace_ml.data.prep import CLEANING_THRESHOLDS
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3] / "data"
HERE = Path(__file__).resolve().parent
GARBLED = NOISE_KINDS.index("garbled")
rng = np.random.default_rng(0)
tok = Tok.load(tokenizer_path(ROOT))
vocab = read_vocab(corpus_dir(ROOT) / "simple_vocab.txt")
samples: list[str] = []
report: dict = {}
for ds in DATASET_IDS:
    c = Corpus.open(corpus_dir(ROOT) / ds)
    tags = c.tags
    n = c.n_docs
    hist, _ = np.histogram(tags.quality, bins=10, range=(0, 1))
    keeps = {lvl: round(float((tags.quality >= t).mean()), 3) for lvl, t in CLEANING_THRESHOLDS.items()}
    idx = rng.choice(n, size=min(n, 1500), replace=False)
    texts = {int(i): tok.decode(c.doc(int(i))) for i in idx}
    moji = [i for i, t in texts.items() if tags.noise_kind[i] != GARBLED and any(ch in MOJIBAKE_CHARS for ch in t)]
    entry = {
        "docs": n,
        "quality_hist_0_to_1": hist.tolist(),
        "cleaning_keeps": keeps,
        "mojibake_outside_garbled": f"{len(moji)}/{len(texts)}",
        "mojibake_examples": [texts[i][max(0, j - 40): j + 40] for i in moji[:3] for j in [next(k for k, ch in enumerate(texts[i]) if ch in MOJIBAKE_CHARS)]],
    }
    if ds in ("educational", "books", "web"):
        s = np.array([simplicity(t, vocab) for t in texts.values()])
        entry["simplicity_q10_q50_q90"] = [round(float(np.quantile(s, q)), 3) for q in (0.1, 0.5, 0.9)]
        words = np.array([len(t.split()) for t in texts.values()])
        entry["words_q10_q50_q90"] = [int(np.quantile(words, q)) for q in (0.1, 0.5, 0.9)]
    if ds == "books":
        ends = [bool(re.search(r"[.!?\"'”’)]\s*$", t)) for t in texts.values()]
        entry["passages_ending_at_sentence_end"] = round(sum(ends) / len(ends), 3)
        entry["passages_with_paragraph_breaks"] = round(sum("\n\n" in t or "\n" in t for t in texts.values()) / len(texts), 3)
    report[ds] = entry
    for i in rng.choice(n, size=3, replace=False):
        i = int(i)
        samples.append(f"===== {ds} #{i} noise={NOISE_KINDS[tags.noise_kind[i]]} quality={tags.quality[i]:.2f}\n{tok.decode(c.doc(i), skip_special=False)[:1200]}\n")
(HERE / "corpus_samples.txt").write_text("\n".join(samples), encoding="utf-8")
(HERE / "corpus_checks.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=1, ensure_ascii=False))
