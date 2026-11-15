"""Does drawing documents without replacement change a first AI? A paired comparison over seeds.

    python ../docs/handoff/next-level/replacement_lab.py --seeds 1 2 3 4 5 6

Run from ml/ (GPU; about 5 minutes a model). For each seed it trains the scorecard's first AI
(Tiny, 12.6M tokens, even mix) twice: with the training pools drawing with replacement (as
engine 2 did) and without (engine 3), and measures both with the whole scorecard. The two arms
train into ``runs/lab-replacement-on`` and ``runs/lab-replacement-off``; finished runs are reused.
Prints each measure per seed and arm, and the mean difference with its standard error.
"""

import argparse
import json
import math
from pathlib import Path

import airace_ml.train.trainer as trainer
from airace_ml.awake import stay_awake
from airace_ml.experiments import scorecard as sc
from airace_ml.paths import data_root

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
a = p.parse_args()

sc.RECIPES["first"] = [
    sc.Step(f"first-s{seed}", sc.TINY, sc.STEP_TOKENS, sc.BALANCED_MIX, seed=seed) for seed in a.seeds
]
real_load_pools = trainer.load_pools


def with_replacement(*args, **kwargs):
    pools = real_load_pools(*args, **kwargs)
    for pool in pools.values():
        pool.replacement = True
    return pools


cards = {}
with stay_awake():
    for arm, loader in (("on", with_replacement), ("off", real_load_pools)):
        trainer.load_pools = loader
        out = Path(f"runs/lab-replacement-{arm}")
        cards[arm] = sc.run_scorecard(
            data_root(),
            out,
            yardstick_root=data_root(),
            recipes=["first"],
            device="cuda",
            on_status=lambda line, arm=arm: print(f"[{arm}] {line}", flush=True),
        )
        (out / "lab_card.json").write_text(json.dumps(cards[arm], indent=1), encoding="utf-8")

MEASURES = (
    "heldout", "yardstick_loss", "bench_overall", "well_formed", "fits", "on_topic", "good_reply",
    "casual", "followups",
)  # fmt: skip
print("\nmeasure: per seed with replacement | without | mean difference (without - with) +- s.e.")
for m in MEASURES:
    on = [cards["on"]["models"][f"first-s{s}"]["measures"][m] for s in a.seeds]
    off = [cards["off"]["models"][f"first-s{s}"]["measures"][m] for s in a.seeds]
    d = [y - x for x, y in zip(on, off, strict=True)]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((v - mean) ** 2 for v in d) / (len(d) - 1) / len(d)) if len(d) > 1 else 0.0
    fmt = (lambda v: f"{v:.3f}") if m in ("heldout", "yardstick_loss") else (lambda v: f"{v:.2f}")
    print(f"{m:15} {' '.join(map(fmt, on))} | {' '.join(map(fmt, off))} | {mean:+.3f} +- {se:.3f}")
