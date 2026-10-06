import json
import math

import pytest

from airace_ml.data.prep import PrepConfig
from airace_ml.model.shape import ModelShape
from airace_ml.train.config import TrainRunConfig, effective_mixture, learning_style
from airace_ml.train.events import (
    Done,
    HeldoutEval,
    Instability,
    Progress,
    Sample,
    event_to_dict,
)
from airace_ml.train.schedule import wsd_lr
from airace_ml.train.stability import SpikeDetector


def test_learning_style_endpoints_and_monotonic():
    lo, hi = learning_style(0.0), learning_style(1.0)
    assert math.isclose(lo.peak_lr, 6e-4) and math.isclose(hi.peak_lr, 1.2e-2)
    assert (lo.clip_norm, hi.clip_norm) == (0.5, 2.5) and math.isclose(lo.warmup_frac, 0.06)
    lrs = [learning_style(i / 10).peak_lr for i in range(11)]
    assert lrs == sorted(lrs)
    with pytest.raises(ValueError):
        learning_style(1.5)


def test_wsd_shape():
    st = learning_style(0.5)
    T = 1000
    w = int(st.warmup_frac * T)
    assert 0 < wsd_lr(0, T, st) < st.peak_lr
    assert math.isclose(wsd_lr(w, T, st), st.peak_lr) and math.isclose(
        wsd_lr(500, T, st), st.peak_lr
    )
    assert math.isclose(wsd_lr(T - 1, T, st), 0.1 * st.peak_lr, rel_tol=1e-3)
    decay = [wsd_lr(s, T, st) for s in range(800, T)]
    assert decay == sorted(decay, reverse=True)


def test_effective_mixture():
    assert effective_mixture({"code": 2.0}, 0.3, {"creative": 1.0}) == pytest.approx(
        {"code": 0.7, "creative": 0.3}
    )
    assert effective_mixture({"a": 1, "b": 3}, 0.0, None) == pytest.approx({"a": 0.25, "b": 0.75})
    with pytest.raises(ValueError):
        effective_mixture({"a": 1}, 0.2, None)


def test_config_roundtrip_and_validation():
    c = TrainRunConfig(
        run_id="r1", seed=1, shape=ModelShape(2, 64, 64), token_budget=32768, mixture={"web": 1.0}
    )
    assert TrainRunConfig.from_json(c.to_json()) == c and c.steps == 2 and c.batch_size == 256
    for bad in [
        {"token_budget": 10},
        {"mixture": {"nope": 1.0}},
        {"replay": 0.95},
        {"purchases": {"web": 0.0}},
    ]:
        with pytest.raises(ValueError):
            TrainRunConfig(**{**c.__dict__, **bad}).validate()


def test_spike_detector():
    d = SpikeDetector(warmup_steps=10)
    assert not any(d.update(i, 3.0 - i * 0.001) for i in range(100))
    assert d.update(101, 30.0) and d.update(102, float("nan"))
    assert SpikeDetector(warmup_steps=10).update(1, float("inf"))


def test_event_to_dict():
    assert event_to_dict(Progress(1, 10, 100, 2.5, 1e-3))["type"] == "progress"


# --- beyond the brief: edge cases and the contracts later tasks rely on ---


def _cfg(**over):
    base = {
        "run_id": "r1",
        "seed": 1,
        "shape": ModelShape(2, 64, 64),
        "token_budget": 32768,
        "mixture": {"web": 1.0},
    }
    return TrainRunConfig(**{**base, **over})


def test_wsd_tiny_runs_and_floor():
    st = learning_style(0.0)
    for total in (1, 2, 3, 5):
        lrs = [wsd_lr(s, total, st) for s in range(total)]
        assert all(0 < lr <= st.peak_lr for lr in lrs)
    assert math.isclose(wsd_lr(1, 2, st), 0.1 * st.peak_lr)
    assert math.isclose(wsd_lr(50, 10, st), 0.1 * st.peak_lr)  # past the end holds the floor
    with pytest.raises(ValueError):
        wsd_lr(0, 0, st)
    with pytest.raises(ValueError):
        wsd_lr(-1, 10, st)


def test_wsd_decay_starts_at_peak_and_warmup_is_linear():
    st = learning_style(0.5)
    T = 1000
    assert math.isclose(wsd_lr(T - 200, T, st), st.peak_lr)
    assert wsd_lr(T - 199, T, st) < st.peak_lr
    w = int(st.warmup_frac * T)
    assert math.isclose(wsd_lr(w // 2 - 1, T, st), st.peak_lr * (w // 2) / w)


def test_learning_style_rejects_nan_and_negative():
    with pytest.raises(ValueError):
        learning_style(float("nan"))
    with pytest.raises(ValueError):
        learning_style(-0.1)


def test_effective_mixture_full_replay_and_zero_weights():
    assert effective_mixture({"web": 1.0}, 1.0, {"code": 1.0}) == pytest.approx({"code": 1.0})
    mixed = effective_mixture({"web": 1.0, "code": 0.0}, 0.0, None)
    assert mixed["web"] == pytest.approx(1.0) and mixed["code"] == 0.0
    blend = effective_mixture({"a": 1, "b": 2}, 0.4, {"b": 1, "c": 1})
    assert sum(blend.values()) == pytest.approx(1.0)
    assert blend == pytest.approx({"a": 0.6 / 3, "b": 0.6 * 2 / 3 + 0.2, "c": 0.2})
    for bad in ({}, {"a": 0.0}, {"a": -1.0, "b": 2.0}, {"a": float("nan")}):
        with pytest.raises(ValueError):
            effective_mixture(bad, 0.0, None)
    with pytest.raises(ValueError):
        effective_mixture({"a": 1.0}, 0.5, {"b": 0.0})


@pytest.mark.parametrize(
    ("over", "field_name"),
    [
        ({"token_budget": 10}, "token_budget"),
        ({"batch_tokens": 0}, "batch_tokens"),
        ({"mixture": {"nope": 1.0}}, "mixture"),
        ({"mixture": {"web": 0.0}}, "mixture"),
        ({"mixture": {"web": -1.0}}, "mixture"),
        ({"finishing_mixture": {"nope": 1.0}}, "finishing_mixture"),
        ({"replay": 0.95}, "replay"),
        ({"replay": -0.1}, "replay"),
        ({"purchases": {"web": 0.0}}, "purchases"),
        ({"purchases": {"web": 1.5}}, "purchases"),
        ({"purchases": {"nope": 0.5}}, "purchases"),
        ({"boldness": 1.5}, "boldness"),
        ({"boldness": float("nan")}, "boldness"),
        ({"notebook": [3]}, "notebook"),
        ({"coaching": [("only one",)]}, "coaching"),
        ({"shape": ModelShape(2, 65, 64)}, "d_model"),
    ],
)
def test_validate_names_the_offending_field(over, field_name):
    with pytest.raises(ValueError, match=field_name):
        _cfg(**over).validate()


def test_validate_accepts_a_full_config():
    _cfg(
        finishing_mixture={"creative": 1.0, "notebook": 2.0},
        replay=0.9,
        purchases={"web": 0.5, "code": 1.0},
        boldness=1.0,
        notebook=["a note"],
        coaching=[("q", "a")],
        parent_dir="runs/parent",
    ).validate()


def test_config_json_roundtrip_restores_tuples_and_nested_objects():
    c = _cfg(
        finishing_mixture={"creative": 1.0},
        replay=0.25,
        prep=PrepConfig(cleaning="thorough", dedup=True, fact_check=True, variety="balanced"),
        purchases={"web": 0.5},
        boldness=0.8,
        notebook=["x", "y"],
        coaching=[("hi", "hello"), ("2+2", "4")],
        probe_prompts=["Once upon a time"],
        parent_dir="runs/parent",
    )
    back = TrainRunConfig.from_json(c.to_json())
    assert back == c
    assert all(isinstance(pair, tuple) for pair in back.coaching)
    assert isinstance(back.shape, ModelShape) and isinstance(back.prep, PrepConfig)


def test_config_defaults_are_not_shared_between_instances():
    a, b = _cfg(), _cfg()
    a.notebook.append("x")
    a.purchases["web"] = 0.5
    assert b.notebook == [] and b.purchases == {}


def test_config_from_json_rejects_unknown_and_missing_fields():
    good = _cfg().to_json()
    with pytest.raises(ValueError, match="unknown"):
        TrainRunConfig.from_json(good[:-1] + ', "bogus": 1}')
    with pytest.raises(ValueError, match="missing"):
        TrainRunConfig.from_json('{"run_id": "r1"}')
    with pytest.raises(ValueError):
        TrainRunConfig.from_json("[1, 2]")


def test_config_steps_and_batch_size_floor_at_one():
    c = _cfg(token_budget=100, batch_tokens=16384, shape=ModelShape(1, 64, 1024))
    assert c.steps == 1 and c.batch_size == 16
    assert _cfg(batch_tokens=100, shape=ModelShape(1, 64, 1024)).batch_size == 1


def test_spike_detector_ignores_spikes_in_warmup_and_does_not_fold_them():
    d = SpikeDetector(warmup_steps=50)
    assert not d.update(0, 3.0)
    assert not d.update(1, 30.0)  # a jump during warmup is not a spike
    d = SpikeDetector(warmup_steps=5)
    for i in range(100):
        d.update(i, 3.0)
    before = d.state()
    assert d.update(100, 30.0) and d.update(101, float("-inf"))
    assert d.state() == before  # spikes leave the statistics untouched
    assert not d.update(102, 3.0)  # and the next normal loss is still normal


def test_spike_detector_needs_both_conditions():
    d = SpikeDetector(warmup_steps=5)
    for i in range(200):
        d.update(i, 3.0 + (0.5 if i % 2 else -0.5))  # noisy: std is large
    assert not d.update(200, 4.0)  # above the mean but inside the noise
    flat = SpikeDetector(warmup_steps=5)
    for i in range(200):
        flat.update(i, 3.0)  # zero variance: only the ratio guards
    assert not flat.update(200, 3.5)  # under 1.3x the mean
    assert flat.update(201, 4.0)  # over 1.3x the mean and over mean + z * 0


def test_spike_detector_state_roundtrip_resumes_exactly():
    losses = [3.0 - i * 0.002 + (0.05 if i % 7 == 0 else 0.0) for i in range(300)]
    full = SpikeDetector(warmup_steps=20)
    full_flags = [full.update(i, x) for i, x in enumerate(losses)]

    first = SpikeDetector(warmup_steps=20)
    for i, x in enumerate(losses[:150]):
        first.update(i, x)
    resumed = SpikeDetector(warmup_steps=20)
    resumed.load_state(json.loads(json.dumps(first.state())))
    assert resumed.state() == first.state()
    tail = [resumed.update(i, x) for i, x in enumerate(losses[150:], start=150)]
    assert tail == full_flags[150:] and resumed.state() == full.state()


def test_event_to_dict_every_event_is_json_safe():
    events = [
        Progress(1, 10, 100, 2.5, 1e-3),
        HeldoutEval(5, {"web": 3.1, "code": 4.2}),
        Sample(5, "Once", "Once upon"),
        Instability(7, "rollback", 0.5),
        Done("completed", {"steps": 10, "final_loss": 2.1}),
    ]
    types = [event_to_dict(e)["type"] for e in events]
    assert types == ["progress", "heldouteval", "sample", "instability", "done"]
    for e in events:
        d = event_to_dict(e)
        assert json.loads(json.dumps(d)) == d
    assert event_to_dict(events[1])["losses"] == {"web": 3.1, "code": 4.2}
    assert event_to_dict(events[4])["summary"] == {"steps": 10, "final_loss": 2.1}
    assert event_to_dict(events[3])["action"] == "rollback"
