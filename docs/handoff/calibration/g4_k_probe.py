"""Gate iteration evidence for G4: the seed, noise and mix distances at k replies per probe.

Replays the gate's G4 measurement on the gate's own models (runs/gate-1/runs) at each k given,
with the same measuring seeds and one population per k. Run from ml/:
  python -m uv run python ../docs/handoff/calibration/g4_k_probe.py 3 8
"""

import sys
from itertools import combinations
from pathlib import Path
from statistics import fmean

import torch

from airace_ml.experiments.gate import MEASURE_SEED, NOISE_MEASURE_SEED, population_distance
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import tokenizer_path
from airace_ml.personality.fingerprint import measure_fingerprint
from airace_ml.tokenizer import Tok

ROOT = Path(__file__).resolve().parents[3] / "data"
RUNS = Path(__file__).resolve().parents[3] / "ml" / "runs" / "gate-1" / "runs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tok.load(tokenizer_path(ROOT))
mixes = ["creative", "code", "facts", "conversations"]
g3 = [f"g3-{m}-s{s}" for m in mixes for s in (1, 2, 3)]
g4 = [f"g4-balanced-s{s}" for s in (1, 2, 3)]
for k in map(int, sys.argv[1:]):
    fps, again = {}, {}
    for name in g3 + g4:
        model, _ = load_checkpoint(RUNS / name, device)
        lm = TorchLM(model, tok, device)
        fps[name] = measure_fingerprint(lm, tok, k=k, seed=MEASURE_SEED)
        if name in g4:
            again[name] = measure_fingerprint(lm, tok, k=k, seed=NOISE_MEASURE_SEED)
    pop = [*fps.values(), *again.values()]
    noise = [population_distance(fps[n], again[n], pop) for n in g4]
    seed = [population_distance(fps[a], fps[b], pop) for a, b in combinations(g4, 2)]
    mix = [population_distance(fps[f"g3-{a}-s1"], fps[f"g3-{b}-s1"], pop) for a, b in combinations(mixes, 2)]
    print(f"RESULT k={k}: noise {fmean(noise):.2f} {[round(x, 2) for x in noise]} | seeds {fmean(seed):.2f} "
          f"{[round(x, 2) for x in seed]} | mixes {fmean(mix):.2f}", flush=True)
