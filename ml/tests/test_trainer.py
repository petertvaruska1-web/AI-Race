import dataclasses
import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from airace_ml.data.sampler import MixtureError
from airace_ml.evals.judge import judge_train_config
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.growth import GrowthError
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import tokenizer_path
from airace_ml.tokenizer import Tok
from airace_ml.train import trainer as trainer_module
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import Done, HeldoutEval, Instability, Progress, Sample
from airace_ml.train.trainer import (
    CHECKPOINT_EVERY,
    TrainHooks,
    can_resume,
    has_resume_state,
    train_run,
)


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


def _notebook_parent(root, tmp_path, **over):
    """Run "a": a quarter of its tokens go to the player's notebook, the rest to creative."""
    return train_run(
        cfg(
            run_id="a",
            token_budget=1024 * 4,
            mixture={"creative": 3.0, "notebook": 1.0},
            notebook=["The sky is blue."],
            **over,
        ),
        out_dir=tmp_path / "a",
        data_root=root,
    )


def test_replay_covers_base_datasets_only_down_a_lineage(tiny_data_root, tmp_path):
    a = _notebook_parent(tiny_data_root, tmp_path)
    assert a.meta.last_mixture == pytest.approx({"creative": 0.75, "notebook": 0.25})
    # The notebook text is not kept (R2), so "review what it knows" cannot replay it: a child that
    # sends no notebook reviews the parent's base datasets, and the share it asked for stays whole.
    b = train_run(
        cfg(
            run_id="b",
            token_budget=1024 * 4,
            mixture={"code": 1.0},
            replay=0.3,
            parent_dir=str(tmp_path / "a"),
        ),
        out_dir=tmp_path / "b",
        data_root=tiny_data_root,
    )
    assert b.status == "completed"
    assert b.meta.last_mixture == pytest.approx({"code": 0.7, "creative": 0.3})
    assert "notebook" not in b.tokens_per_dataset or b.tokens_per_dataset["notebook"] == 0
    assert b.meta.runs[-1]["replay_dropped"] == ["notebook"]
    assert b.meta.runs[0]["replay_dropped"] == []  # the first run replayed nothing
    # ... and the child's own child, which used to inherit the notebook share and fail with it.
    c = train_run(
        cfg(
            run_id="c",
            token_budget=1024 * 4,
            mixture={"web": 1.0},
            replay=0.3,
            parent_dir=str(tmp_path / "b"),
        ),
        out_dir=tmp_path / "c",
        data_root=tiny_data_root,
    )
    assert c.meta.last_mixture == pytest.approx({"web": 0.7, "code": 0.21, "creative": 0.09})
    assert c.meta.runs[-1]["replay_dropped"] == []
    assert [run["run_id"] for run in c.meta.runs] == ["a", "b", "c"]


def test_a_child_with_a_notebook_gets_only_its_own_share_for_it(tiny_data_root, tmp_path):
    _notebook_parent(tiny_data_root, tmp_path)
    r = train_run(
        cfg(
            run_id="b",
            token_budget=1024 * 4,
            mixture={"code": 3.0, "notebook": 1.0},
            notebook=["A different note."],
            replay=0.2,
            parent_dir=str(tmp_path / "a"),
        ),
        out_dir=tmp_path / "b",
        data_root=tiny_data_root,
    )
    # 0.8 of the run is the player's mix (a quarter of it the new notebook); the review share goes
    # to the parent's base datasets, not on top of the notebook.
    assert r.meta.last_mixture == pytest.approx({"code": 0.6, "notebook": 0.2, "creative": 0.2})
    assert r.meta.runs[-1]["replay_dropped"] == ["notebook"]


def test_replay_of_a_parent_with_nothing_but_custom_data_is_dropped(tiny_data_root, tmp_path):
    train_run(
        cfg(
            run_id="a",
            token_budget=1024 * 4,
            mixture={"notebook": 1.0, "coaching": 1.0},
            notebook=["The sky is blue."],
            coaching=[("What colour is the sky?", "Blue.")],
        ),
        out_dir=tmp_path / "a",
        data_root=tiny_data_root,
    )
    r = train_run(
        cfg(
            run_id="b",
            token_budget=1024 * 4,
            mixture={"code": 1.0, "web": 3.0},
            replay=0.5,
            parent_dir=str(tmp_path / "a"),
        ),
        out_dir=tmp_path / "b",
        data_root=tiny_data_root,
    )
    assert r.status == "completed"
    assert r.meta.last_mixture == pytest.approx({"code": 0.25, "web": 0.75})  # all of it, whole
    assert r.meta.runs[-1]["replay_dropped"] == ["coaching", "notebook"]
    assert sum(r.meta.last_mixture.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "over",
    [
        {"notebook": ["a \ud800 b"], "mixture": {"notebook": 1.0}},
        {"coaching": [("prompt", "reply \ude00")], "mixture": {"coaching": 1.0}},
        {"probe_prompts": ["hi \ud83d"]},  # used to die at the first eval, 25 steps in
    ],
)
def test_a_lone_surrogate_in_the_players_text_fails_before_training(tiny_data_root, tmp_path, over):
    with pytest.raises(ValueError, match="lone surrogate"):
        train_run(cfg(**over), out_dir=tmp_path / "x", data_root=tiny_data_root)
    assert not (tmp_path / "x").exists()


def test_attention_span_changes_freely_within_a_lineage(tiny_data_root, tmp_path):
    # Spec 4.3: context length is not a parameter shape, so it may go down as well as up.
    def run(run_id, shape, parent=None):
        return train_run(
            cfg(run_id=run_id, token_budget=1024 * 5, shape=shape, parent_dir=parent),
            out_dir=tmp_path / run_id,
            data_root=tiny_data_root,
        )

    a = run("a", ModelShape(2, 96, 128))
    b = run("b", ModelShape(2, 96, 64), parent=str(tmp_path / "a"))  # 128 -> 64
    c = run("c", ModelShape(2, 96, 128), parent=str(tmp_path / "b"))  # 64 -> 128
    assert b.status == c.status == "completed"
    assert a.meta.lineage_id == b.meta.lineage_id == c.meta.lineage_id
    assert (b.meta.shape.ctx_len, c.meta.shape.ctx_len) == (64, 128)
    assert c.meta.parent_version_id == "b" and len(c.meta.runs) == 3
    for run_id, shape in (
        ("fewer_layers", ModelShape(1, 96, 128)),
        ("narrower", ModelShape(2, 64, 128)),
    ):
        with pytest.raises(GrowthError):
            run(run_id, shape, parent=str(tmp_path / "c"))
        assert not (tmp_path / run_id).exists()


@pytest.mark.parametrize(
    "over",
    [
        {"mixture": {"creative": 0.0}},  # all-zero weights
        {"mixture": {"creative": -1.0, "code": 2.0}},  # a negative weight
        {"mixture": {"bogus": 1.0}},  # an unknown dataset
        {"mixture": {"creative": 1.0, "notebook": 1.0}, "notebook": [" ", "\n"]},  # blank pool
        {"finishing_mixture": {"notebook": 1.0}, "notebook": ["\t"]},  # unsampleable finish
        {"finishing_mixture": {"bogus": 1.0}},
    ],
)
def test_unsampleable_mixtures_fail_before_training(tiny_data_root, tmp_path, over):  # Focus 3
    with pytest.raises(MixtureError):
        train_run(cfg(**over), out_dir=tmp_path / "x", data_root=tiny_data_root)
    assert not (tmp_path / "x").exists()


def _inf_gradient(step, model):
    if step == 25:
        model.tok_emb.weight.grad.fill_(float("inf"))


def _broken_by_the_update(step, model):
    # A finite gradient whose update still wrecks the model (finite weights, huge loss), as a
    # too-bold step can.
    if step == 25:
        with torch.no_grad():
            model.norm_f.weight.fill_(1e4)


@pytest.mark.parametrize(
    ("hook", "spike_step"),
    [
        (_inf_gradient, 25),  # caught by the gradient norm before the update is applied
        (_broken_by_the_update, 26),  # caught by the next loss; never became a snapshot
    ],
)
def test_bad_updates_never_become_the_kept_weights(tiny_data_root, tmp_path, hook, spike_step):
    # Step 25 ends a snapshot interval: whatever it does must not become a rollback target.
    ev = []
    r = train_run(
        cfg(),
        out_dir=tmp_path / "g",
        data_root=tiny_data_root,
        on_event=ev.append,
        _hooks=TrainHooks(after_backward=hook),
    )
    instabilities = [(e.step, e.action) for e in ev if isinstance(e, Instability)]
    assert instabilities == [(spike_step, "rollback")]
    assert r.status == "completed"
    m, _ = load_checkpoint(r.out_dir)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert all(math.isfinite(v) for v in r.heldout_losses.values())


def test_resume_recovers_from_an_interrupted_swap(tiny_data_root, tmp_path):
    c = cfg(token_budget=1024 * 40)
    full = train_run(c, out_dir=tmp_path / "full", data_root=tiny_data_root)
    out = tmp_path / "p"
    train_run(c, out_dir=out, data_root=tiny_data_root, _hooks=TrainHooks(stop_after_steps=20))
    # A crash mid-swap: the complete state never got its final name, and a later save was
    # cut short before writing its state.json.
    (out / "resume").rename(out / "resume.tmp.crashed")
    (out / "resume.tmp.partial").mkdir()
    (out / "resume.tmp.partial" / "state.pt").write_bytes(b"half a file")
    res = train_run(c, out_dir=out, data_root=tiny_data_root, resume=True)
    assert res.status == "completed" and res.steps == 40
    m1, _ = load_checkpoint(full.out_dir)
    m2, _ = load_checkpoint(res.out_dir)
    for p, q in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)
    assert _telemetry(out) == _telemetry(tmp_path / "full")
    assert not [p.name for p in out.iterdir() if p.name.startswith("resume")]


def _uninterrupted(root, tmp_path_factory, name, config, hooks=None):
    """Train ``config`` straight through once, for the resume tests of a module to compare to."""
    out = tmp_path_factory.mktemp(name) / "full"
    with pytest.MonkeyPatch.context() as m:
        m.setenv("AIRACE_DEVICE", "cpu")
        train_run(config, out_dir=out, data_root=root, _hooks=hooks)
    return out


def _assert_same_run(resumed, uninterrupted):
    """Same events in the same order, and the same weights, as the run that was never stopped."""
    assert _telemetry(resumed) == _telemetry(uninterrupted)
    m1, _ = load_checkpoint(uninterrupted)
    m2, _ = load_checkpoint(resumed)
    for p, q in zip(m1.parameters(), m2.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)


# 30 steps: the last 6 (steps 25 to 30) are the decay phase, which trains on the finishing mix.
_FINISHING = cfg(mixture={"creative": 1.0}, finishing_mixture={"code": 1.0}, token_budget=1024 * 30)


@pytest.fixture(scope="module")
def plain_run(tiny_data_root, tmp_path_factory):
    return _uninterrupted(tiny_data_root, tmp_path_factory, "plain", cfg(token_budget=1024 * 24))


@pytest.fixture(scope="module")
def finishing_run(tiny_data_root, tmp_path_factory):
    return _uninterrupted(tiny_data_root, tmp_path_factory, "finishing", _FINISHING)


@pytest.fixture(scope="module")
def rolled_back_run(tiny_data_root, tmp_path_factory):
    return _uninterrupted(
        tiny_data_root,
        tmp_path_factory,
        "rolled_back",
        cfg(token_budget=1024 * 40),
        TrainHooks(loss_override=_nan_at(30, 35)),
    )


def test_resume_inside_the_finishing_phase_trains_the_rest_on_the_finishing_mix(
    tiny_data_root, tmp_path, finishing_run
):
    out = tmp_path / "p"
    part = train_run(
        _FINISHING, out_dir=out, data_root=tiny_data_root, _hooks=TrainHooks(stop_after_steps=27)
    )
    assert part.status == "interrupted" and part.steps == 27
    res = train_run(_FINISHING, out_dir=out, data_root=tiny_data_root, resume=True)
    assert res.status == "completed" and res.steps == 30
    _assert_same_run(out, finishing_run)


def test_resume_after_a_rollback_at_the_stop_step_keeps_rolling_back(
    tiny_data_root, tmp_path, rolled_back_run
):
    # Step 30 is a spike and also a resume point: the saved weights are the snapshot's, so the
    # state keeps no copy of it and the resumed run rebuilds the snapshot from the weights. The
    # spike at step 35 must roll back to the same weights the uninterrupted run rolls back to.
    c = cfg(token_budget=1024 * 40)
    out = tmp_path / "p"
    part = train_run(
        c,
        out_dir=out,
        data_root=tiny_data_root,
        checkpoint_every=10,
        _hooks=TrainHooks(loss_override=_nan_at(30, 35), stop_after_steps=30),
    )
    assert part.status == "interrupted" and '"action": "rollback"' in _telemetry(out)
    train_run(
        c,
        out_dir=out,
        data_root=tiny_data_root,
        resume=True,
        _hooks=TrainHooks(loss_override=_nan_at(30, 35)),
    )
    assert _telemetry(out).count('"action": "rollback"') == 2
    _assert_same_run(out, rolled_back_run)


@pytest.mark.parametrize(
    "refused",
    [
        "every rename of a resume folder",
        "moving the previous state aside",  # the new state then stays under its temporary name
    ],
)
def test_training_goes_on_when_windows_refuses_to_rename_the_resume_folder(
    tiny_data_root, tmp_path, monkeypatch, plain_run, refused
):
    # A virus scanner or indexer holding the folder makes Windows refuse a rename (PermissionError).
    # Training must go on, and a resume must find the newest complete state wherever it ended up.
    real_rename = Path.rename

    def rename(self, target):
        if self.name == "resume" or (
            refused.startswith("every") and self.name.startswith("resume.tmp.")
        ):
            raise PermissionError(13, "refused, the folder is in use", str(self))
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    c = cfg(token_budget=1024 * 24)
    out = tmp_path / "p"
    part = train_run(
        c,
        out_dir=out,
        data_root=tiny_data_root,
        checkpoint_every=6,
        _hooks=TrainHooks(stop_after_steps=12),
    )
    assert part.status == "interrupted"
    kept_tmp = [p.name for p in out.iterdir() if p.name.startswith("resume.tmp.")]
    assert kept_tmp  # the swap was refused, so a state stayed under its temporary name
    assert can_resume(out, c)
    ev = []
    res = train_run(c, out_dir=out, data_root=tiny_data_root, resume=True, on_event=ev.append)
    assert res.status == "completed" and res.steps == 24
    assert next(e for e in ev if isinstance(e, Progress)).step == 20  # from step 12, not 6
    _assert_same_run(out, plain_run)
    assert not [p.name for p in out.iterdir() if p.name.startswith("resume")]


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


def test_can_resume_only_a_complete_state_of_the_same_config(tiny_data_root, tmp_path):
    c = cfg(token_budget=1024 * 40)
    out = tmp_path / "p"
    assert not can_resume(out, c)  # no directory yet
    train_run(c, out_dir=out, data_root=tiny_data_root, _hooks=TrainHooks(stop_after_steps=20))
    assert can_resume(out, c)
    assert not can_resume(out, cfg(token_budget=1024 * 40, seed=2))  # another run's state
    assert not can_resume(out, cfg(token_budget=1024 * 50))
    (out / "resume").rename(out / "resume.tmp.crashed")  # still complete under another name
    assert can_resume(out, c)
    (out / "resume.tmp.crashed" / "state.json").unlink()  # incomplete: no state.json
    assert not can_resume(out, c)
    (out / "resume.tmp.crashed" / "state.json").write_text("{not json", encoding="utf-8")
    assert not can_resume(out, c)


def test_has_resume_state_is_true_for_any_complete_state_whatever_its_config(
    tiny_data_root, tmp_path
):
    c = cfg(token_budget=1024 * 12)
    out = tmp_path / "p"
    assert not has_resume_state(out)  # no folder yet
    out.mkdir()
    assert not has_resume_state(out)  # an empty folder
    train_run(c, out_dir=out, data_root=tiny_data_root, _hooks=TrainHooks(stop_after_steps=6))
    assert has_resume_state(out) and has_resume_state(str(out))
    assert can_resume(out, c) and not can_resume(out, cfg(token_budget=1024 * 12, seed=2))
    (out / "resume").rename(out / "resume.tmp.crashed")  # still complete under another name
    assert has_resume_state(out)
    (out / "resume.tmp.crashed" / "state.json").write_text("{not json", encoding="utf-8")
    assert not has_resume_state(out)  # incomplete states are not progress
    (out / "resume.tmp.crashed" / "state.json").unlink()
    assert not has_resume_state(out)


def test_has_resume_state_is_false_once_the_run_has_finished(tiny_data_root, tmp_path):
    c = cfg(token_budget=1024 * 12)
    out = tmp_path / "p"
    train_run(c, out_dir=out, data_root=tiny_data_root, _hooks=TrainHooks(stop_after_steps=6))
    assert has_resume_state(out)
    train_run(c, out_dir=out, data_root=tiny_data_root, resume=True)
    assert not has_resume_state(out)


def test_the_default_checkpoint_interval_is_the_exported_constant():
    default = inspect.signature(train_run).parameters["checkpoint_every"].default
    assert default == CHECKPOINT_EVERY == 200


# -- micro-batches: one optimizer step over a batch that does not fit in memory at once -----------


class _Probe:
    """What a short run did: per step the logged loss, the gradients just before clipping and the
    rows of each forward pass of the step, plus the weights it ended with."""

    def __init__(self):
        self.loss: dict[int, float] = {}
        self.grads: dict[int, list[torch.Tensor]] = {}
        self.forwards: dict[int, list[int]] = {}
        self.weights: list[torch.Tensor] = []
        self._calls: list[int] = []

    def hooks(self) -> TrainHooks:
        def loss_override(step, loss):
            self.loss[step] = loss
            self.forwards[step] = self._calls[:]
            self._calls.clear()
            return loss

        def after_backward(step, model):
            self.grads[step] = [p.grad.detach().clone() for p in model.parameters()]

        return TrainHooks(loss_override=loss_override, after_backward=after_backward)


def _probe_run(root, out_dir, monkeypatch, *, micro_tokens=None, plain_step=False, **over):
    """Train ``cfg(**over)`` for 3 steps, with the micro-batch cap set to ``micro_tokens`` (the
    trainer's own when None) and, with ``plain_step``, the step the trainer had before there were
    micro-batches: the whole batch in one forward, a mean cross-entropy, one backward."""
    probe = _Probe()
    with monkeypatch.context() as m:
        if micro_tokens is not None:
            m.setattr(trainer_module, "MICRO_BATCH_TOKENS", micro_tokens)
        if plain_step:

            def one_pass(self, x, y):
                with trainer_module._autocast(self.device):
                    logits = self.model(x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                loss.backward()
                return loss.detach()

            m.setattr(trainer_module._Run, "_forward_backward", one_pass)
        m.setattr(trainer_module._Run, "_evaluate", lambda self, step: None)  # not under test
        original = Transformer.forward

        def spy(self, x, *args, **kwargs):
            probe._calls.append(x.shape[0])
            return original(self, x, *args, **kwargs)

        m.setattr(Transformer, "forward", spy)
        train_run(
            cfg(token_budget=1024 * 3, **over),
            out_dir=out_dir,
            data_root=root,
            _hooks=probe.hooks(),
        )
    model, _ = load_checkpoint(out_dir)
    probe.weights = [p.detach().clone() for p in model.parameters()]
    return probe


@pytest.fixture(scope="module")
def whole_batch_run(tiny_data_root, tmp_path_factory):
    """The reference: 3 steps of 16 rows x 64 tokens, which fit in one micro-batch."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("AIRACE_DEVICE", "cpu")
        return _probe_run(tiny_data_root, tmp_path_factory.mktemp("whole") / "w", m)


@pytest.mark.parametrize(
    ("micro_tokens", "rows_per_forward"),
    [
        (512, [8, 8]),  # 8 rows of 64 tokens fit in a micro-batch
        (192, [3, 3, 3, 3, 3, 1]),  # the last micro-batch is the short one
        (10, [1] * 16),  # a row longer than the cap still goes through, alone
    ],
)
def test_micro_batches_give_the_same_step_as_the_whole_batch(
    tiny_data_root, tmp_path, monkeypatch, whole_batch_run, micro_tokens, rows_per_forward
):
    ref = whole_batch_run
    run = _probe_run(tiny_data_root, tmp_path / "m", monkeypatch, micro_tokens=micro_tokens)
    assert run.forwards[1] == run.forwards[2] == run.forwards[3] == rows_per_forward
    # The same draw from the sampler, so the same loss and the same gradient up to the order the
    # rows are summed in (the first step starts from identical weights).
    for step in (1, 2, 3):
        assert run.loss[step] == pytest.approx(ref.loss[step], rel=1e-4)
    for g, expected in zip(run.grads[1], ref.grads[1], strict=True):
        torch.testing.assert_close(g, expected, rtol=1e-3, atol=1e-6)
    # And it learned the same thing.
    for w, expected in zip(run.weights, ref.weights, strict=True):
        torch.testing.assert_close(w, expected, rtol=0, atol=2e-3)


def test_a_batch_in_one_micro_batch_is_bit_identical_to_the_plain_step(
    tiny_data_root, tmp_path, monkeypatch, whole_batch_run
):
    plain = _probe_run(tiny_data_root, tmp_path / "plain", monkeypatch, plain_step=True)
    assert plain.forwards[1] == whole_batch_run.forwards[1] == [16]  # 16 x 64 tokens: one pass
    assert plain.loss == whole_batch_run.loss
    for step in (1, 2, 3):
        for g, expected in zip(whole_batch_run.grads[step], plain.grads[step], strict=True):
            assert torch.equal(g, expected)
    for w, expected in zip(whole_batch_run.weights, plain.weights, strict=True):
        assert torch.equal(w, expected)


def test_micro_batches_do_not_change_what_a_resumed_run_draws(
    tiny_data_root, tmp_path, monkeypatch
):
    # One sampler draw per step whatever the micro-batch size, so an interrupted run resumes onto
    # the same stream and ends with the weights of an uninterrupted one.
    monkeypatch.setattr(trainer_module, "MICRO_BATCH_TOKENS", 192)
    c = cfg(token_budget=1024 * 16)
    full = train_run(c, out_dir=tmp_path / "full", data_root=tiny_data_root)
    train_run(
        c,
        out_dir=tmp_path / "p",
        data_root=tiny_data_root,
        checkpoint_every=4,
        _hooks=TrainHooks(stop_after_steps=8),
    )
    res = train_run(c, out_dir=tmp_path / "p", data_root=tiny_data_root, resume=True)
    assert _telemetry(tmp_path / "p") == _telemetry(tmp_path / "full")
    m1, _ = load_checkpoint(full.out_dir)
    m2, _ = load_checkpoint(res.out_dir)
    for p, q in zip(m1.parameters(), m2.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_judge_recipe_step_fits_the_8gb_reference_card(gate_data_root, tmp_path):
    # 128 rows x 256 tokens per step, 8 layers x 384 wide, on the real 4096-entry vocabulary (a
    # real tiny build, not the 512-entry test tokenizer): in one piece a step needs about 7 GiB on
    # a card that also drives the desktop. In micro-batches of 8192 tokens it needs about 2 GiB, so
    # a 4 GiB bound tells the two apart.
    assert Tok.load(tokenizer_path(gate_data_root)).vocab_size == 4096
    judge = judge_train_config()
    c = dataclasses.replace(judge, run_id="judge-memory", token_budget=judge.batch_tokens)
    assert c.steps == 1 and c.batch_size * c.shape.ctx_len == 32768
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    r = train_run(c, out_dir=tmp_path / "j", data_root=gate_data_root, device=torch.device("cuda"))
    assert r.status == "completed" and r.steps == 1 and r.tokens == 32768
    assert torch.cuda.max_memory_allocated() < 4 * 1024**3
