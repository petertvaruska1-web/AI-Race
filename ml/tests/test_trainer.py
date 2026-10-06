import pytest
import torch

from airace_ml.data.sampler import MixtureError
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.growth import GrowthError
from airace_ml.model.shape import ModelShape
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import Done, HeldoutEval, Instability, Progress, Sample
from airace_ml.train.trainer import TrainHooks, train_run


@pytest.fixture(autouse=True)
def _cpu_by_default(monkeypatch):
    """`device=None` means `pick_device()`; keep these tests on the CPU even on a CUDA machine."""
    monkeypatch.setenv("AIRACE_DEVICE", "cpu")


def cfg(**k):
    base = {
        "run_id": "r1",
        "seed": 1,
        "shape": ModelShape(2, 64, 64),
        "token_budget": 1024 * 60,
        "mixture": {"creative": 1.0, "conversations": 1.0},
        "batch_tokens": 1024,
    }
    base.update(k)
    return TrainRunConfig(**base)


def test_loss_decreases_and_events(tiny_data_root, tmp_path):
    ev = []
    r = train_run(cfg(), out_dir=tmp_path / "r1", data_root=tiny_data_root, on_event=ev.append)
    losses = [e.loss for e in ev if isinstance(e, Progress)]
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3 - 1.0
    assert isinstance(ev[-1], Done) and ev[-1].status == "completed"
    assert any(isinstance(e, Sample) for e in ev)
    assert set(next(e for e in ev if isinstance(e, HeldoutEval)).losses) == {
        "web",
        "books",
        "educational",
        "conversations",
        "code",
        "reasoning",
        "facts",
        "creative",
    }
    lines = (tmp_path / "r1" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ev)
    assert (
        r.meta.version_id == "r1"
        and r.meta.parent_version_id is None
        and r.meta.tokens_trained_total == 60 * 1024
    )


def test_continue_and_grow(tiny_data_root, tmp_path):
    a = train_run(cfg(token_budget=1024 * 10), out_dir=tmp_path / "a", data_root=tiny_data_root)
    b = train_run(
        cfg(
            run_id="b",
            token_budget=1024 * 10,
            parent_dir=str(tmp_path / "a"),
            shape=ModelShape(3, 96, 64),
        ),
        out_dir=tmp_path / "b",
        data_root=tiny_data_root,
    )
    assert b.meta.lineage_id == a.meta.lineage_id and b.meta.parent_version_id == "r1"
    assert b.meta.shape == ModelShape(3, 96, 64) and len(b.meta.runs) == 2
    with pytest.raises(GrowthError):
        train_run(
            cfg(run_id="c", parent_dir=str(tmp_path / "b"), shape=ModelShape(2, 64, 64)),
            out_dir=tmp_path / "c",
            data_root=tiny_data_root,
        )
    assert not (tmp_path / "c").exists()


def test_resume_equivalence(tiny_data_root, tmp_path):
    full = train_run(
        cfg(token_budget=1024 * 40), out_dir=tmp_path / "full", data_root=tiny_data_root
    )
    part = train_run(
        cfg(token_budget=1024 * 40),
        out_dir=tmp_path / "p",
        data_root=tiny_data_root,
        checkpoint_every=10,
        _hooks=TrainHooks(stop_after_steps=20),
    )
    assert part.status == "interrupted"
    res = train_run(
        cfg(token_budget=1024 * 40), out_dir=tmp_path / "p", data_root=tiny_data_root, resume=True
    )
    m1, _ = load_checkpoint(full.out_dir)
    m2, _ = load_checkpoint(res.out_dir)
    for p, q in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)


def test_spike_rollback_and_stop(tiny_data_root, tmp_path):
    ev = []
    r = train_run(
        cfg(),
        out_dir=tmp_path / "s",
        data_root=tiny_data_root,
        on_event=ev.append,
        _hooks=TrainHooks(loss_override=lambda s, loss: float("nan") if s == 30 else loss),
    )
    assert r.status == "completed" and any(
        isinstance(e, Instability) and e.action == "rollback" for e in ev
    )
    r2 = train_run(
        cfg(run_id="s2"),
        out_dir=tmp_path / "s2",
        data_root=tiny_data_root,
        _hooks=TrainHooks(
            loss_override=lambda s, loss: float("nan") if s in (20, 30, 40) else loss
        ),
    )
    assert r2.status == "unstable_stopped"


def test_invalid_mixture_fails_before_training(tiny_data_root, tmp_path):  # Review Focus 3
    with pytest.raises(MixtureError):
        train_run(
            cfg(mixture={"notebook": 1.0}, notebook=["   ", ""]),
            out_dir=tmp_path / "x",
            data_root=tiny_data_root,
        )
    assert not (tmp_path / "x").exists()


def test_finishing_mixture(tiny_data_root, tmp_path):
    r = train_run(
        cfg(mixture={"creative": 1.0}, finishing_mixture={"code": 1.0}, token_budget=1024 * 50),
        out_dir=tmp_path / "f",
        data_root=tiny_data_root,
    )
    share = r.tokens_per_dataset.get("code", 0) / sum(r.tokens_per_dataset.values())
    assert 0.15 <= share <= 0.25


def _telemetry(out_dir):
    return (out_dir / "telemetry.jsonl").read_text(encoding="utf-8")


def _nan_at(*steps):
    return lambda s, loss: float("nan") if s in steps else loss


def test_resume_continues_where_it_stopped(tiny_data_root, tmp_path):
    c = cfg(token_budget=1024 * 40)
    train_run(c, out_dir=tmp_path / "full", data_root=tiny_data_root)
    train_run(
        c,
        out_dir=tmp_path / "p",
        data_root=tiny_data_root,
        checkpoint_every=10,
        _hooks=TrainHooks(stop_after_steps=20),
    )
    assert (tmp_path / "p" / "resume").is_dir() and not (
        tmp_path / "p" / "model.safetensors"
    ).exists()
    with pytest.raises(ValueError, match="different run config"):
        train_run(
            cfg(token_budget=1024 * 40, seed=2),
            out_dir=tmp_path / "p",
            resume=True,
            data_root=tiny_data_root,
        )
    ev = []
    r = train_run(
        c, out_dir=tmp_path / "p", data_root=tiny_data_root, resume=True, on_event=ev.append
    )
    assert r.status == "completed" and r.steps == 40
    assert next(e for e in ev if isinstance(e, Progress)).step == 30  # picks up after step 20
    assert _telemetry(tmp_path / "p") == _telemetry(tmp_path / "full")  # same events, same order
    assert not (tmp_path / "p" / "resume").exists()
    with pytest.raises(FileNotFoundError):
        train_run(c, out_dir=tmp_path / "p", data_root=tiny_data_root, resume=True)


def test_resume_keeps_the_rollback_snapshot(tiny_data_root, tmp_path):
    # Interrupted at 30, the last snapshot (step 25) is older than the saved weights; the spike
    # at 35 must still roll back to it.
    c = cfg(token_budget=1024 * 40)
    full = train_run(
        c,
        out_dir=tmp_path / "full",
        data_root=tiny_data_root,
        _hooks=TrainHooks(loss_override=_nan_at(35)),
    )
    train_run(
        c,
        out_dir=tmp_path / "p",
        data_root=tiny_data_root,
        _hooks=TrainHooks(loss_override=_nan_at(35), stop_after_steps=30),
    )
    res = train_run(
        c,
        out_dir=tmp_path / "p",
        data_root=tiny_data_root,
        resume=True,
        _hooks=TrainHooks(loss_override=_nan_at(35)),
    )
    assert '"action": "rollback"' in _telemetry(tmp_path / "p")
    assert _telemetry(tmp_path / "p") == _telemetry(tmp_path / "full")
    m1, _ = load_checkpoint(full.out_dir)
    m2, _ = load_checkpoint(res.out_dir)
    for p, q in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)


def test_rollback_halves_lr_and_stop_keeps_last_good_weights(tiny_data_root, tmp_path):
    ev = []
    r = train_run(
        cfg(),
        out_dir=tmp_path / "s",
        data_root=tiny_data_root,
        on_event=ev.append,
        _hooks=TrainHooks(loss_override=_nan_at(20, 30, 40)),
    )
    instabilities = [(e.step, e.action, e.lr_scale) for e in ev if isinstance(e, Instability)]
    assert instabilities == [(20, "rollback", 0.5), (30, "rollback", 0.25), (40, "stopped", 0.25)]
    evals = {e.step: e.losses for e in ev if isinstance(e, HeldoutEval)}
    # Eval at 25 runs on the step-25 snapshot weights; the stop at 40 restores and keeps them.
    assert sorted(evals) == [25, 40] and evals[40] == evals[25]
    assert r.steps == 40 and isinstance(ev[-1], Done) and ev[-1].status == "unstable_stopped"
    _, meta = load_checkpoint(tmp_path / "s")
    assert meta.runs[-1]["status"] == "unstable_stopped"


def test_finishing_mix_loss_level_is_not_instability(tiny_data_root, tmp_path):
    # Harder finishing data sits at a much higher loss than the stable phase. The detector
    # relearns the level when the mix switches (step 41) instead of calling every step a spike.
    r = train_run(
        cfg(mixture={"creative": 1.0}, finishing_mixture={"code": 1.0}, token_budget=1024 * 50),
        out_dir=tmp_path / "f",
        data_root=tiny_data_root,
        _hooks=TrainHooks(loss_override=lambda s, loss: loss + 10.0 if s > 40 else loss),
    )
    assert r.status == "completed"


def test_replay_notebook_and_coaching(tiny_data_root, tmp_path):
    train_run(
        cfg(token_budget=1024 * 10, mixture={"creative": 1.0}),
        out_dir=tmp_path / "a",
        data_root=tiny_data_root,
    )
    r = train_run(
        cfg(
            run_id="b",
            token_budget=1024 * 20,
            parent_dir=str(tmp_path / "a"),
            replay=0.5,
            mixture={"notebook": 1.0, "coaching": 1.0},
            notebook=["  The sky is blue.  ", "   "],
            coaching=[("What colour is the sky?", "Blue."), ("  ", "dropped: no prompt")],
        ),
        out_dir=tmp_path / "b",
        data_root=tiny_data_root,
    )
    assert r.meta.last_mixture == pytest.approx(
        {"notebook": 0.25, "coaching": 0.25, "creative": 0.5}
    )
    assert all(r.tokens_per_dataset[name] > 0 for name in ("notebook", "coaching", "creative"))
    assert r.meta.tokens_trained_total == 30 * 1024
    with pytest.raises(MixtureError, match="coaching"):
        train_run(
            cfg(run_id="c", mixture={"coaching": 1.0}, coaching=[(" ", "x"), ("y", "")]),
            out_dir=tmp_path / "c",
            data_root=tiny_data_root,
        )
    assert not (tmp_path / "c").exists()


def test_attention_span_cannot_shrink(tiny_data_root, tmp_path):
    train_run(
        cfg(token_budget=1024 * 5, shape=ModelShape(2, 64, 128)),
        out_dir=tmp_path / "a",
        data_root=tiny_data_root,
    )
    with pytest.raises(GrowthError):
        train_run(
            cfg(run_id="b", parent_dir=str(tmp_path / "a")),
            out_dir=tmp_path / "b",
            data_root=tiny_data_root,
        )
    assert not (tmp_path / "b").exists()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_starter_speed_gpu(tiny_data_root, tmp_path):
    r = train_run(
        cfg(shape=ModelShape(4, 128, 256), token_budget=1_048_576, batch_tokens=16384),
        out_dir=tmp_path / "g",
        data_root=tiny_data_root,
        device=torch.device("cuda"),
    )
    assert r.status == "completed"
    assert r.tokens / r.wall_seconds > 30_000
