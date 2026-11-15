"""The growth lab script (docs/handoff/next-level/growth_lab.py) on the tiny fixture, on the CPU:
its self-checks pass, every arm trains and is measured, the wrapper is removed afterwards, a
second call reuses every run, and the masks mark exactly the copied elements."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from airace_ml.model import growth
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.train import trainer
from airace_ml.train.optim import MuonAdamW

LAB_PATH = Path(__file__).resolve().parents[2] / "docs" / "handoff" / "next-level" / "growth_lab.py"


@pytest.fixture(scope="module")
def lab():
    spec = importlib.util.spec_from_file_location("growth_lab", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["growth_lab"] = module  # dataclasses look their module up here
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("growth_lab", None)


def test_masks_mark_the_copied_elements(lab):
    parent = Transformer(ModelShape(2, 64, 64), 512)
    grown = growth.grow(parent, ModelShape(3, 96, 64), seed=5)
    old = lab.copied_elements(parent, grown)
    assert old["tok_emb.weight"][:, :64].all() and not old["tok_emb.weight"][:, 64:].any()
    layout = growth.insertion_layout(2, 3)
    fresh = layout.index(None)
    assert not any(m.any() for n, m in old.items() if n.startswith(f"blocks.{fresh}."))
    copied = layout.index(0)
    assert old[f"blocks.{copied}.attn.wq.weight"][:64, :64].all()
    assert old[f"blocks.{copied}.attn.wq.weight"].sum() == 64 * 64
    assert lab.element_counts(old)["old"] == parent.num_params()
    with torch.no_grad():  # a mask on the wrong elements is caught by the value check
        grown.blocks[copied].attn.wq.weight[0, 0] += 1.0
    with pytest.raises(lab.LabError):
        lab.copied_elements(parent, grown)


def test_the_lab_runs_on_the_tiny_fixture(lab, tiny_data_root, tmp_path, capsys):
    out = tmp_path / "lab"
    argv = ["--out", str(out), "--seeds", "1", "--tiny-root", str(tiny_data_root)]
    argv += ["--device", "cpu"]
    assert lab.main(argv) == 0
    lab.assert_unpatched()
    assert trainer.grow is growth.grow and not lab._has_lab_wrapper(MuonAdamW.step)
    first = capsys.readouterr().out
    assert "self-check 2:" in first and "self-check 3:" in first
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    models = results["models"]
    assert set(models) == {"A", "B", "C", "D", "E"}
    assert models["B"]["1"]["transfer_steps"] == [1, 2, 3, 4]  # 12 steps, the first third
    assert models["C"]["1"]["transfer_steps"] == [1, 2, 3, 4]
    assert "transfer_steps" not in models["A"]["1"] and models["A"]["1"]["curve"]
    counts = results["self_checks"]["elements"]
    assert counts["old"] > 0 and counts["new"] > 0
    assert results["summary"]["heldout"]["A"]["minus_A"] == 0.0
    # B and C trained differently from A; A, B and C share the parent.
    assert models["B"]["1"]["heldout"] != models["A"]["1"]["heldout"]
    assert lab.main(argv) == 0
    again = capsys.readouterr().out
    assert ": training (" not in again and "reusing it" in again
