"""Strict mojibake count per dataset, over every document: sequences that only misdecoded UTF-8
leaves (``Ã``, ``Â``, ``â€``, U+FFFD, and the ``"¦`` remnant of a misdecoded ellipsis), outside the
deliberately garbled documents.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/mojibake_strict.py
"""

import re
import sys
from pathlib import Path

from airace_ml.data.corpus import DATASET_IDS, NOISE_KINDS, Corpus
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3] / "data"
GARBLED = NOISE_KINDS.index("garbled")
STRICT = re.compile("Ã|Â|â€|�|[\"”]¦")
tok = Tok.load(tokenizer_path(ROOT))
for ds in sys.argv[1:] or DATASET_IDS:
    c = Corpus.open(corpus_dir(ROOT) / ds)
    hits, garbled_hits, examples, sources = 0, 0, [], {}
    for i in range(c.n_docs):
        text = tok.decode(c.doc(i))
        m = STRICT.search(text)
        if not m:
            continue
        if c.tags.noise_kind[i] == GARBLED:
            garbled_hits += 1
            continue
        hits += 1
        if len(examples) < 4:
            examples.append(repr(text[max(0, m.start() - 50) : m.end() + 30]))
    print(f"{ds}: {hits}/{c.n_docs} outside garbled; garbled docs with marks: {garbled_hits}")
    for e in examples:
        print("   ", e)
