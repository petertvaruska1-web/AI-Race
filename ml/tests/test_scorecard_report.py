import json

from airace_ml.experiments import scorecard as sc
from airace_ml.experiments import scorecard_report as rep
from airace_ml.experiments.runs import PACKAGE_ROOT, TRAINING_SOURCES, UNDIGESTED
from tests.test_gate import _covered, _imported_modules

MEASURED_WITH = {
    "yardstick": "yardstick-v1",
    "instruments": {"label": "reference judge", "judge": {"conv_reply_p90": 4.2}, "novelty": 900},
    "yardstick_rows": 128,
    "scale": {"name": "full", "followups": 60},
}


def _measures(**changes) -> dict:
    base = {key: 0.5 for key in sc.MEASURES}
    base.update(bench_overall=40.0, heldout=2.5, heldout_conversations=2.4, yardstick_loss=2.6)
    return {**base, **changes}


def _model(status="completed", span=256, **changes) -> dict:
    return {
        "recipe": "first",
        "parent": None,
        "status": status,
        "yardstick_span": span,
        "measures": _measures(**changes),
    }


def _card(models=None, data="text-v1", engine=2, measured_with=None, **changes) -> dict:
    return {
        "data_root": "data",
        "yardstick_root": "data",
        "data_identity": data,
        "engine": engine,
        "date": "2026-11-13",
        "scale": "full",
        "measured_with": measured_with if measured_with is not None else dict(MEASURED_WITH),
        "models": models if models is not None else {"first-s1": _model(**changes)},
        "timing": None,
    }


# -- what is digested ---------------------------------------------------------------------------

# Modules scorecard.py imports that no card digest holds, and why that is safe: each shapes a
# run's training key (and so its token) or the caches themselves, never a measured value.
ORCHESTRATION = {
    "experiments/__init__.py": "the package marker",
    "experiments/configs.py": "the recipes' mixes, part of every run's training key",
    "experiments/gate.py": "the setup checks and the continuation seed, part of the run key",
    "experiments/runs.py": "the run and measurement caches themselves",
}


def test_the_report_is_not_digested_and_everything_the_measuring_uses_is():
    assert not _covered("experiments/scorecard_report.py", sc.CARD_SOURCES)
    assert _covered("experiments/scorecard.py", sc.CARD_SOURCES)
    assert _covered("experiments/chatprobes.py", sc.CARD_SOURCES)
    imported = {
        p.relative_to(PACKAGE_ROOT).as_posix()
        for p in _imported_modules(PACKAGE_ROOT / "experiments" / "scorecard.py")
    }
    assert "experiments/scorecard_report.py" not in imported
    assert "experiments/chatprobes.py" in imported and "train/trainer.py" in imported
    for module in imported - set(ORCHESTRATION):
        digested = _covered(module, (*TRAINING_SOURCES, *sc.CARD_SOURCES))
        assert digested or module in UNDIGESTED, module


# -- compare --------------------------------------------------------------------------------------


def test_compare_reports_a_six_point_drop_and_ignores_a_two_point_one():
    baseline = _card(well_formed=0.80)
    assert rep.compare(_card(well_formed=0.78), baseline) == []
    (said,) = rep.compare(_card(well_formed=0.74), baseline)
    assert "first-s1" in said and "well-formed" in said and "74%" in said and "80%" in said
    assert rep.compare(_card(well_formed=0.95), baseline) == []  # better is never reported


def test_compare_knows_that_a_lower_loss_is_better():
    baseline = _card(yardstick_loss=2.600, heldout=2.500)
    assert rep.compare(_card(yardstick_loss=2.400, heldout=2.300), baseline) == []
    assert rep.compare(_card(yardstick_loss=2.620, heldout=2.515), baseline) == []
    said = rep.compare(_card(yardstick_loss=2.640, heldout=2.530), baseline)
    assert len(said) == 2 and "2.640" in said[1] and "2.600" in said[1]
    assert set(rep.TOLERANCE) == {
        "well_formed", "casual", "followups", "bench_overall", "yardstick_loss", "heldout",
    }  # fmt: skip
    (bench,) = rep.compare(_card(bench_overall=37.5), _card(bench_overall=40.0))
    assert "37.5" in bench and "40.0" in bench


def test_compare_reports_a_measure_that_has_no_value_any_more():
    for lost in (None, float("nan"), float("inf")):
        (said,) = rep.compare(_card(well_formed=lost), _card())
        assert "first-s1" in said and "well-formed" in said and "no value" in said
    card = _card()
    del card["models"]["first-s1"]["measures"]["person_life"]  # one with no tolerance
    (said,) = rep.compare(card, _card())
    assert "claims a human life" in said and "no value" in said
    baseline = _card()  # a measure the baseline lacks is not compared
    del baseline["models"]["first-s1"]["measures"]["casual"]
    assert rep.compare(_card(casual=0.0), baseline) == []


def test_compare_reports_a_model_that_stopped_early_this_time():
    (said,) = rep.compare(_card(status="unstable_stopped"), _card())
    assert "first-s1" in said and "stopped early" in said
    assert rep.compare(_card(status="unstable_stopped"), _card(status="unstable_stopped")) == []


def test_compare_skips_own_text_losses_when_the_training_text_differs():
    baseline = _card(heldout=2.0, heldout_conversations=2.0, yardstick_loss=2.6)
    worse = _card(data="text-v2", heldout=3.0, heldout_conversations=None, yardstick_loss=2.7)
    (said,) = rep.compare(worse, baseline)  # only the yardstick loss, which still compares
    assert "yardstick" in said
    (note,) = [n for n in rep.notes(worse, baseline) if "held-out" in n]
    assert "training text" in note
    assert not [n for n in rep.notes(_card(), baseline) if "held-out" in n]


def test_compare_names_models_the_baseline_does_not_have():
    card = _card(models={"first-s1": _model(), "small": _model(well_formed=0.0)})
    assert rep.compare(card, _card()) == []
    (note,) = rep.notes(card, _card())
    assert "small" in note and "first-s1" not in note


def test_different_engines_compare():
    assert rep.unlike(_card(engine=3), _card(engine=2)) == []
    assert rep.compare(_card(engine=3, well_formed=0.4), _card(engine=2)) != []


def test_a_baseline_measured_differently_is_unlike():
    def changed(field, value):
        return _card(measured_with={**MEASURED_WITH, field: value})

    cases = {
        "yardstick data": changed("yardstick", "yardstick-v2"),
        "judge": changed("instruments", {**MEASURED_WITH["instruments"], "judge": {"x": 1.0}}),
        "novelty index": changed("instruments", {**MEASURED_WITH["instruments"], "novelty": 1}),
        "yardstick rows": changed("yardstick_rows", 8),
        "scale": changed("scale", {"name": "test"}),
    }
    for words, baseline in cases.items():
        (said,) = rep.unlike(_card(), baseline)
        assert words in said, words
    (said,) = rep.unlike(_card(), _card(span=64))
    assert "first-s1" in said and "256" in said and "64" in said
    old = _card()
    del old["measured_with"]
    (said,) = rep.unlike(_card(), old)
    assert "before" in said and "again" in said
    assert rep.unlike(_card(), _card()) == []


# -- reports --------------------------------------------------------------------------------------


def test_the_reports_have_one_row_per_model_and_one_column_per_measure(tmp_path):
    card = _card(models={"first-s1": _model(well_formed=0.8), "large": _model(status="x")})
    card["timing"] = {"model": "first-s1", "seconds": 52.25, "device": "cuda", "reused": True}
    rep.write_reports(card, tmp_path)
    assert json.loads((tmp_path / "scorecard.json").read_text("utf-8")) == card
    lines = (tmp_path / "scorecard.md").read_text("utf-8").splitlines()
    header = next(line for line in lines if line.startswith("Data root:"))
    assert card["data_root"] in header and card["date"] in header and "engine 2" in header
    assert any("52.2 s" in line and "reused" in line for line in lines)
    rows = [line for line in lines if line.startswith("| ")]
    assert len(rows) == 2 + len(card["models"])  # the heading, its rule, one row per model
    assert all(row.count("|") == rows[0].count("|") for row in rows)
    assert rows[0].count("|") == 3 + len(sc.MEASURES)  # model, recipe, then the measures
    assert "| Says it is an AI (unnamed) |" in rows[0]
    first = next(row for row in rows if row.startswith("| first-s1 "))
    assert "| 2.600 |" in first and "| 80% |" in first and "| 40.0 |" in first
    assert any(row.startswith("| large (stopped early) ") for row in rows)
