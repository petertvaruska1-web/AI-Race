"""Gate iteration evidence for G2: replays the legibility probe on several models and splits each
failed reply into "too few words" (the word checks) and "judge loss above the bar".

Run from ml/:  python -m uv run python ../docs/handoff/calibration/g2_diagnosis.py [run ...]
Each run is a folder under runs/gate-1/runs, or a path; "judge" means the reference judge.
"""

import sys
from pathlib import Path

import torch

from airace_ml.evals.judge import Judge, _has_reply_words
from airace_ml.experiments.gate import LEGIBILITY_REPLY_TOKENS, SAMPLE_TEMPERATURE, SAMPLE_TOP_P
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import judge_dir, tokenizer_path
from airace_ml.personality.fingerprint import load_probes
from airace_ml.tokenizer import Tok

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3] / "data"
RUNS = Path(__file__).resolve().parents[3] / "ml" / "runs" / "gate-1" / "runs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tok.load(tokenizer_path(ROOT))
judge = Judge.load(judge_dir(ROOT), device)
bar = judge.calibration.conv_reply_p90
probes = load_probes()
chosen = [p for p in probes if p.kind == "open"][:10] + [p for p in probes if p.kind == "help"][:10]
verbose = "-v" in sys.argv
rounds = 3 if "-3" in sys.argv else 1  # -3: three sampling rounds (60 replies), a steadier rate
for name in [a for a in sys.argv[1:] if a not in ("-v", "-3")] or ["starter", "g4-balanced-s1", "judge"]:
    folder = judge_dir(ROOT) if name == "judge" else (RUNS / name if (RUNS / name).exists() else Path(name))
    model, _ = load_checkpoint(folder, device)
    lm = TorchLM(model, tok, device)
    asked = [p for _ in range(rounds) for p in chosen]
    replies = [
        lm.chat_reply([("user", p.text)], max_new_tokens=LEGIBILITY_REPLY_TOKENS,
                      temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, seed=i)
        for i, p in enumerate(asked)
    ]
    nlls = judge.nll_per_token([judge.tok.encode(r.strip()) for r in replies])
    wordy = [_has_reply_words(r) for r in replies]
    fluent = [n <= bar for n in nlls]
    ok = sum(w and f for w, f in zip(wordy, fluent))
    n = len(replies)
    print(f"{name}: well-formed {ok}/{n} ({ok / n:.0%}); pass word checks {sum(wordy)}/{n}; judge loss "
          f"under {bar:.2f}: {sum(fluent)}/{n}; median judge loss {sorted(nlls)[n // 2]:.2f}")
    if verbose:
        for p, r, n, w in zip(asked, replies, nlls, wordy):
            print(f"   [{n:.2f} {'w' if w else '-'}] {p.text} -> {r!r}")
    del model, lm
