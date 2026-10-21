"""Gate iteration evidence for G3: a model's creativity score and stories, as the benchmark makes them.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/creativity_probe.py RUN_DIR ...
"""

import sys
from pathlib import Path

import torch

from airace_ml.evals.creativity import score_creativity
from airace_ml.evals.judge import Judge
from airace_ml.evals.novelty import NoveltyIndex
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import judge_dir, novelty_path, tokenizer_path
from airace_ml.tokenizer import Tok

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3] / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tok.load(tokenizer_path(ROOT))
judge = Judge.load(judge_dir(ROOT), device)
novelty = NoveltyIndex.load(novelty_path(ROOT))
print(f"judge calibration: {judge.calibration}")
for folder in sys.argv[1:]:
    model, _ = load_checkpoint(Path(folder), device)
    lm = TorchLM(model, tok, device)
    score, items = score_creativity(lm, tok, judge, novelty)
    losses = judge.nll_per_token([judge.tok.encode(i.output) for i in items])
    print(f"{folder}: creativity {score.score:.1f} (raw {score.raw:.3f}); story judge losses "
          f"{sorted(round(x, 2) for x in losses)}")
    for item in items[:4]:
        print(f"   [{item.score:.2f}] {item.output[:220]!r}")
