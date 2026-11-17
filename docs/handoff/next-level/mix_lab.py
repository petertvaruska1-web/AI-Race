"""Should a first AI's default reading lean toward conversations? (Task 15, the default-mix trial.)

    AIRACE_DATA=../data2 python ../docs/handoff/next-level/mix_lab.py --conversations 0.3 \
        --out runs/mix-lab-c30 [--seeds 1 2]

Run from ml/ (GPU, about 15 minutes). It runs the whole scorecard's ``first`` recipe (Tiny brain,
12.6M tokens, seeds 1 and 2) on the data root's text with one change: the conversations dataset
gets the share ``--conversations`` and the other seven datasets share the rest equally (the game's
default plan today gives each of the eight datasets the same share, 12.5%). The yardstick root is
``../data`` (text 1), as for every card of Task 15. Compare the card with the text-2 card's
``first`` models, and draw replies for the blind reading with ``held_talk.py``.
"""

import argparse
from pathlib import Path

from airace_ml.awake import stay_awake
from airace_ml.experiments import scorecard as sc
from airace_ml.experiments.scorecard_report import write_reports
from airace_ml.paths import data_root

p = argparse.ArgumentParser()
p.add_argument("--conversations", type=float, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
p.add_argument("--yardstick-root", type=Path, default=Path("../data"))
a = p.parse_args()

if not 0.0 < a.conversations < 1.0:
    raise SystemExit("--conversations must be a share between 0 and 1")
others = [ds for ds in sc.BALANCED_MIX if ds != "conversations"]
if set(sc.BALANCED_MIX) != {*others, "conversations"} or len(others) != 7:
    raise SystemExit(f"unexpected scorecard mix {dict(sc.BALANCED_MIX)}")
mix = {ds: (1.0 - a.conversations) / len(others) for ds in others}
mix["conversations"] = a.conversations
print("mix: " + ", ".join(f"{ds} {share:.3f}" for ds, share in mix.items()), flush=True)
sc.RECIPES["first"] = [
    sc.Step(f"first-s{seed}", sc.TINY, sc.STEP_TOKENS, mix, seed=seed) for seed in a.seeds
]
with stay_awake():
    card = sc.run_scorecard(
        data_root(),
        a.out,
        yardstick_root=a.yardstick_root,
        recipes=["first"],
        device="cuda",
        on_status=lambda line: print(line, flush=True),
    )
write_reports(card, a.out)
print(f"written to {a.out}")
