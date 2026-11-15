"""The lessons lab builds the mixture the game server builds (config.ts, even mix, one source)."""

import importlib.util
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "docs" / "handoff" / "next-level" / "lessons_lab.py"
_spec = importlib.util.spec_from_file_location("lessons_lab", _PATH)
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)


@pytest.mark.parametrize(
    ("share", "each"),
    # Worked by hand: (1 - share) * 12 / 96 for each of the 8 datasets.
    [(0.05, 0.11875), (0.1, 0.1125), (0.2, 0.1)],
)
def test_mixture_matches_the_server_rule(share, each):
    mixture = lab.server_mixture(share)
    assert set(mixture) == set(lab.DATASETS) | {"coaching"}
    for ds in lab.DATASETS:
        assert math.isclose(mixture[ds], each)
    assert math.isclose(mixture["coaching"], share)
    assert math.isclose(sum(mixture.values()), 1.0)


def test_config_fields_equal_the_servers():
    cfg = lab.server_config(
        0.05,
        run_id="x",
        seed=7,
        shape=lab.TINY_SHAPE,
        tokens=lab.SHORT_TOKENS,
        batch_tokens=lab.BATCH_TOKENS,
        parent_dir="p",
        lessons=[("q", "a")],
    )
    cfg.validate()
    assert cfg.steps == 768 and cfg.batch_tokens == 8192 and cfg.replay == 0.0
    assert cfg.boldness == 0.5 and cfg.probe_prompts == [] and cfg.notebook == []
    assert (cfg.prep.cleaning, cfg.prep.dedup, cfg.prep.fact_check, cfg.prep.variety) == (
        "standard",
        False,
        False,
        "natural",
    )


def test_dry_run_end_to_end_on_the_tiny_data(tiny_data_root, tmp_path, monkeypatch, capsys):
    from airace_ml.model.shape import ModelShape
    from airace_ml.train.config import TrainRunConfig
    from airace_ml.train.trainer import train_run

    monkeypatch.setenv("AIRACE_DEVICE", "cpu")
    shape = ModelShape(2, 64, 64)
    parent = tmp_path / "parent"
    train_run(
        TrainRunConfig(
            run_id="parent",
            seed=1,
            shape=shape,
            token_budget=1024 * 8,
            mixture={"creative": 1.0, "conversations": 1.0},
            batch_tokens=1024,
        ),
        out_dir=parent,
        data_root=tiny_data_root,
        device=lab.torch.device("cpu"),
    )
    argv = [
        "--parent", str(parent), "--out", str(tmp_path / "lab"), "--shares", "0.1",
        "--data-root", str(tiny_data_root), "--device", "cpu",
        "--tokens", str(1024 * 16), "--batch-tokens", "1024", "--shape", "2", "64", "64",
    ]  # fmt: skip
    assert lab.main(argv) == 0
    results = lab.json.loads((tmp_path / "lab" / "results.json").read_text(encoding="utf-8"))
    row = results["shares"]["0.1"]
    assert row["written"]["total"] == 10 and row["casual"]["total"] == 10
    assert row["passes"]["source"] == "run" and row["passes"]["passes"] > 0
    assert lab.main(argv) == 0  # a re-run reuses the finished run
    assert "finished earlier" in capsys.readouterr().out
