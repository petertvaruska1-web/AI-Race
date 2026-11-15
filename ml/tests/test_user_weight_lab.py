import importlib.util
from pathlib import Path

import pytest
import torch

from airace_ml.experiments import scorecard as sc
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer

LAB = Path(__file__).resolve().parents[2] / "docs/handoff/next-level/user_weight_lab.py"


@pytest.fixture(scope="module")
def lab():
    spec = importlib.util.spec_from_file_location("user_weight_lab", LAB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ai_turn_mask_covers_the_ai_text_and_its_end(lab, tiny_tok):
    t = tiny_tok
    a, b = 10, 11
    row = [t.bos_id, t.user_id, a, t.end_id, t.ai_id, b, a, t.end_id, t.user_id, a]
    full = torch.tensor([row])
    mask = lab.ai_turn_mask(full[:, :-1], full[:, 1:], t)
    # Targets: user a end ai b a end user a. The AI's turn is b, a and its end.
    assert mask[0].tolist() == [False, False, False, False, True, True, True, False, False]
    assert lab.user_turn_mask(full[:, :-1], full[:, 1:], t)[0].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_the_loss_over_all_targets_is_the_scorecards_yardstick_loss(lab, tiny_data_root, tiny_tok):
    torch.manual_seed(0)
    model = Transformer(ModelShape(2, 64, 64), 512)
    figures = lab.turn_losses(model, tiny_tok, tiny_data_root)
    assert figures["all_loss"] == pytest.approx(sc.yardstick_loss(model, tiny_data_root), abs=1e-3)
    assert figures["all_targets"] == sc.YARDSTICK_SEQS * 64
    assert figures["ai_targets"] > 0
    assert figures["user_targets"] > 0
    assert figures["ai_targets"] + figures["user_targets"] <= figures["all_targets"]
