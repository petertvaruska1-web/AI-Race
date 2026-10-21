"""Gate iteration evidence for G7(b): the false-fact rate of trained models, as the gate measures it.

Run from ml/:  python -m uv run python ../docs/handoff/calibration/g7_false_fact_probe.py RUN_DIR ...
"""

import sys
from pathlib import Path

import torch

from airace_ml.experiments.gate import false_fact_rate
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.skills.facts import FalseFactPlan
from airace_ml.skills.kb import load_kb
from airace_ml.tokenizer import Tok

ROOT = Path(__file__).resolve().parents[3] / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tok.load(tokenizer_path(ROOT))
kb = load_kb()
plan = FalseFactPlan.from_json((corpus_dir(ROOT) / "false_facts.json").read_text(encoding="utf-8"))
for folder in sys.argv[1:]:
    model, _ = load_checkpoint(Path(folder), device)
    print(f"{folder}: false facts preferred {false_fact_rate(TorchLM(model, tok, device), tok, kb, plan):.1%}")
