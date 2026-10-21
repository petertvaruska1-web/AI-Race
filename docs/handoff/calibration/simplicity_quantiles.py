"""F15 calibration: per-source simplicity quantiles on real hub rows with the real simple_vocab.

Run from ml/:  python -m uv run --extra content python ../docs/handoff/calibration/simplicity_quantiles.py

It builds the full-scale simple vocabulary (the 6,000 most frequent words of 200,000 TinyStories
rows, exactly as ``airace-content build --scale full`` does), then reads the first documents of
each simplicity-filtered source through the same adapter and preparation the build uses, and
prints the simplicity quantiles and the keep-rate of candidate thresholds.
"""

import json
import sys

import numpy as np

from airace_content.assemble import RECIPES, content_text, prepare_content
from airace_content.build import _simple_vocab
from airace_content.sources import SOURCES, hf_fetch, row_to_content
from airace_content.textproc import simplicity

DOCS = 2000
SOURCES_TO_MEASURE = ("fineweb", "fineweb_edu", "gutenberg", "cosmo_khan", "tinystories")
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def main() -> None:
    vocab = _simple_vocab("full", hf_fetch, lambda m: print(m, file=sys.stderr, flush=True))
    current = {
        c.name: c.simplicity_min for r in RECIPES.values() for c in r.components if c.kind == "source"
    }
    out = {}
    for sid in SOURCES_TO_MEASURE:
        scores: list[float] = []
        for row in hf_fetch(SOURCES[sid], 200_000):
            for piece in row_to_content(sid, row):
                content = prepare_content(piece)
                if content is not None:
                    scores.append(simplicity(content_text(content), vocab))
            if len(scores) >= DOCS:
                break
        a = np.array(scores[:DOCS])
        q = {f"q{int(p * 100):02d}": round(float(np.quantile(a, p)), 3) for p in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)}
        keep = {f"{t:.2f}": round(float((a >= t).mean()), 3) for t in THRESHOLDS}
        out[sid] = {"docs": len(a), "current_min": current.get(sid), "quantiles": q, "keep_rate": keep}
        print(sid, json.dumps(out[sid]), flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
