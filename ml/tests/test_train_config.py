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


_NAN = float("nan")


@pytest.mark.parametrize(
    ("over", "field_name"),
    [
        # seed: an int >= 0 (bool is not an int here)
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"seed": "1"}, "seed"),
        ({"seed": None}, "seed"),
        # run_id ends up in file paths
        ({"run_id": ""}, "run_id"),
        ({"run_id": "../../x"}, "run_id"),
        ({"run_id": "a/b"}, "run_id"),
        ({"run_id": "a\\b"}, "run_id"),
        ({"run_id": "a b"}, "run_id"),
        ({"run_id": ".hidden"}, "run_id"),
        ({"run_id": "-x"}, "run_id"),
        ({"run_id": "x" * 65}, "run_id"),
        ({"run_id": "ok\n"}, "run_id"),
        ({"run_id": 7}, "run_id"),
        ({"run_id": None}, "run_id"),
        # int fields must be ints
        ({"token_budget": 32768.5}, "token_budget"),
        ({"token_budget": 32768.0}, "token_budget"),
        ({"token_budget": "32768"}, "token_budget"),
        ({"token_budget": True}, "token_budget"),
        ({"token_budget": None}, "token_budget"),
        ({"batch_tokens": 16384.5}, "batch_tokens"),
        ({"batch_tokens": "16384"}, "batch_tokens"),
        ({"batch_tokens": True}, "batch_tokens"),
        ({"batch_tokens": None}, "batch_tokens"),
        # float fields must be real numbers
        ({"replay": "0.1"}, "replay"),
        ({"replay": True}, "replay"),
        ({"replay": None}, "replay"),
        ({"replay": _NAN}, "replay"),
        ({"boldness": "0.5"}, "boldness"),
        ({"boldness": True}, "boldness"),
        ({"boldness": None}, "boldness"),
        ({"purchases": {"web": "0.5"}}, "purchases"),
        ({"purchases": {"web": True}}, "purchases"),
        ({"purchases": {"web": None}}, "purchases"),
        ({"purchases": {"web": _NAN}}, "purchases"),
        ({"mixture": {"web": "1"}}, "mixture"),
        ({"mixture": {"web": True}}, "mixture"),
        ({"mixture": {"web": None}}, "mixture"),
        ({"mixture": {"web": _NAN}}, "mixture"),
        ({"mixture": {"web": float("inf")}}, "mixture"),
        ({"finishing_mixture": {"web": "1"}}, "finishing_mixture"),
        # containers
        ({"mixture": None}, "mixture"),
        ({"mixture": [("web", 1.0)]}, "mixture"),
        ({"finishing_mixture": [("web", 1.0)]}, "finishing_mixture"),
        ({"purchases": None}, "purchases"),
        ({"purchases": [("web", 0.5)]}, "purchases"),
        ({"notebook": None}, "notebook"),
        ({"notebook": "abc"}, "notebook"),
        ({"coaching": None}, "coaching"),
        ({"coaching": {"a": "b"}}, "coaching"),
        ({"coaching": [{"prompt": "hi", "reply": "yo"}]}, "coaching"),
        ({"coaching": ["ab"]}, "coaching"),
        ({"coaching": [("a", 1)]}, "coaching"),
        ({"coaching": [("a", "b", "c")]}, "coaching"),
        ({"probe_prompts": None}, "probe_prompts"),
        ({"probe_prompts": [1]}, "probe_prompts"),
        ({"parent_dir": 5}, "parent_dir"),
        ({"parent_dir": ""}, "parent_dir"),
        # replay needs somewhere to replay from
        ({"replay": 0.3}, "parent_dir"),
        ({"replay": 0.3, "parent_dir": None}, "parent_dir"),
        # nested objects
        ({"shape": None}, "shape"),
        ({"shape": {"n_layer": 2, "d_model": 64, "ctx_len": 64}}, "shape"),
        ({"shape": ModelShape("2", 64, 64)}, "n_layer"),
        ({"shape": ModelShape(2, 64, 64.0)}, "ctx_len"),
        ({"prep": None}, "prep"),
        ({"prep": PrepConfig(dedup="yes")}, "dedup"),
        ({"prep": PrepConfig(fact_check=1)}, "fact_check"),
    ],
)
def test_validate_rejects_malformed_values_with_a_value_error(over, field_name):
    with pytest.raises(ValueError, match=field_name):
        _cfg(**over).validate()


def test_validate_accepts_replay_with_a_parent_and_int_valued_numbers():
    _cfg(replay=0.3, parent_dir="runs/parent").validate()
    _cfg(replay=0, boldness=1, purchases={"web": 1}, mixture={"web": 1, "code": 2}).validate()
    _cfg(run_id="Run_1.v2-b", seed=0).validate()
    _cfg(run_id="x" * 64).validate()


def _doc(**over):
    """A valid config document as a dict, with fields overridden or (value ``...``) removed."""
    doc = json.loads(_cfg().to_json())
    for key, value in over.items():
        if value is ...:
            del doc[key]
        else:
            doc[key] = value
    return doc


@pytest.mark.parametrize(
    ("over", "field_name"),
    [
        # coaching pairs must be lists of exactly two strings, never coerced
        ({"coaching": [{"prompt": "hi", "reply": "yo"}]}, "coaching"),
        ({"coaching": ["ab"]}, "coaching"),
        ({"coaching": [["a"]]}, "coaching"),
        ({"coaching": [["a", "b", "c"]]}, "coaching"),
        ({"coaching": [["a", 1]]}, "coaching"),
        ({"coaching": [None]}, "coaching"),
        ({"coaching": None}, "coaching"),
        ({"coaching": "ab"}, "coaching"),
        ({"coaching": {"a": "b"}}, "coaching"),
        # other containers
        ({"notebook": None}, "notebook"),
        ({"notebook": "abc"}, "notebook"),
        ({"probe_prompts": None}, "probe_prompts"),
        ({"mixture": None}, "mixture"),
        ({"mixture": [["web", 1.0]]}, "mixture"),
        ({"purchases": None}, "purchases"),
        ({"purchases": ["web"]}, "purchases"),
        ({"finishing_mixture": ["web"]}, "finishing_mixture"),
        # nested objects: shape
        ({"shape": None}, "shape"),
        ({"shape": [2, 64, 64]}, "shape"),
        ({"shape": "2x64x64"}, "shape"),
        ({"shape": {"n_layer": 2, "d_model": 64}}, "ctx_len"),
        ({"shape": {}}, "shape"),
        ({"shape": {"n_layer": 2, "d_model": 64, "ctx_len": 64, "extra": 1}}, "shape"),
        # nested objects: prep
        ({"prep": None}, "prep"),
        ({"prep": ["standard"]}, "prep"),
        ({"prep": {"cleaning": "standard", "bogus": True}}, "prep"),
        ({"prep": {"cleaning": "extreme"}}, "prep"),
        ({"prep": {"cleaning": ["standard"]}}, "prep"),
        ({"prep": {"variety": "wild"}}, "prep"),
    ],
)
def test_from_json_rejects_malformed_documents_with_a_value_error(over, field_name):
    with pytest.raises(ValueError, match=field_name):
        TrainRunConfig.from_json(json.dumps(_doc(**over)))


def test_from_json_accepts_nulls_only_where_allowed_and_defaults_when_absent():
    c = TrainRunConfig.from_json(json.dumps(_doc(finishing_mixture=None, parent_dir=None)))
    assert c.finishing_mixture is None and c.parent_dir is None
    c = TrainRunConfig.from_json(
        json.dumps(
            _doc(coaching=..., notebook=..., prep=..., purchases=..., probe_prompts=..., replay=...)
        )
    )
    assert c.coaching == [] and c.prep == PrepConfig() and c.replay == 0.0
    c.validate()


def test_from_json_then_validate_catches_what_the_structure_check_lets_through():
    # Wrong types inside scalar fields are validate()'s job; both must end in a ValueError.
    for over, name in [
        ({"token_budget": "32768"}, "token_budget"),
        ({"token_budget": 32768.5}, "token_budget"),
        ({"seed": -3}, "seed"),
        ({"run_id": "../../x"}, "run_id"),
        ({"replay": 0.3}, "parent_dir"),
        ({"prep": {"dedup": "no"}}, "dedup"),
        ({"shape": {"n_layer": "2", "d_model": 64, "ctx_len": 64}}, "n_layer"),
        ({"mixture": {"web": "1"}}, "mixture"),
    ]:
        c = TrainRunConfig.from_json(json.dumps(_doc(**over)))
        with pytest.raises(ValueError, match=name):
            c.validate()


def test_from_json_malformed_json_is_a_value_error():
    with pytest.raises(ValueError):
        TrainRunConfig.from_json("{not json")
    with pytest.raises(ValueError):
        TrainRunConfig.from_json("")


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
    losses[200] = 30.0  # a real spike after the resume point
    losses[250] = float("nan")
    full = SpikeDetector(warmup_steps=20)
    full_flags = [full.update(i, x) for i, x in enumerate(losses)]
    assert [i for i, flagged in enumerate(full_flags) if flagged] == [200, 250]

    first = SpikeDetector(warmup_steps=20)
    for i, x in enumerate(losses[:150]):
        first.update(i, x)
    resumed = SpikeDetector(warmup_steps=20)
    resumed.load_state(json.loads(json.dumps(first.state())))
    assert resumed.state() == first.state()
    tail = [resumed.update(i, x) for i, x in enumerate(losses[150:], start=150)]
    assert tail == full_flags[150:] and tail[200 - 150] and tail[250 - 150]
    assert resumed.state() == full.state()


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


def test_event_to_dict_maps_non_finite_floats_to_null_so_the_json_is_strict():
    inf, nan = float("inf"), float("nan")
    events = [
        Progress(1, 10, 100, nan, 1e-3),
        Progress(1, 10, 100, 2.5, inf),
        HeldoutEval(5, {"web": nan, "code": -inf, "books": 3.0}),
        Done("unstable_stopped", {"final_loss": inf, "history": [1.0, nan, {"deep": -inf}]}),
        Instability(7, "stopped", nan),
    ]
    for e in events:
        text = json.dumps(event_to_dict(e), allow_nan=False)  # raises on NaN / Infinity
        assert "NaN" not in text and "Infinity" not in text
    assert event_to_dict(events[0])["loss"] is None and event_to_dict(events[0])["lr"] == 1e-3
    assert event_to_dict(events[1])["lr"] is None and event_to_dict(events[1])["loss"] == 2.5
    assert event_to_dict(events[2])["losses"] == {"web": None, "code": None, "books": 3.0}
    assert event_to_dict(events[3])["summary"] == {
        "final_loss": None,
        "history": [1.0, None, {"deep": None}],
    }
    assert event_to_dict(events[4])["lr_scale"] is None
    assert event_to_dict(events[3])["type"] == "done"


def test_event_to_dict_does_not_mutate_the_event():
    e = Done("completed", {"final_loss": float("inf"), "steps": 3})
    event_to_dict(e)
    assert e.summary["final_loss"] == float("inf")
