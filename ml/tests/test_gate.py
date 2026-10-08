import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.prep import heldout_docs
from airace_ml.evals.judge import JudgeCalibration
from airace_ml.evals.novelty import NoveltyIndex
from airace_ml.evals.scoring import BenchReport, CategoryScore
from airace_ml.experiments import gate
from airace_ml.experiments import runs as gate_runs
from airace_ml.experiments.configs import (
    BALANCED_MIX,
    EARLY_BUDGET,
    EARLY_SHAPE,
    FULL_SCALE,
    QUICK_SCALE,
    STARTER_BUDGET,
    STARTER_SHAPE,
    TARGET_CATEGORY,
    GateScale,
    mix_heavy,
)
from airace_ml.experiments.gate import (
    NO_CUDA_DETAIL,
    GateCriterion,
    GateOutcome,
    eval_differentiation,
    eval_forgetting,
    eval_growth,
    eval_legibility,
    eval_prep,
    eval_seed_variation,
    eval_speed,
    false_fact_probes,
    false_fact_rate,
    garble_rate,
    max_logit_diff,
    speed_criterion,
)
from airace_ml.experiments.report import TRANSCRIPT_CHARS, write_gate_report
from airace_ml.experiments.runs import GateRun, GateRuns
from airace_ml.infer.lm import Generation
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME
from airace_ml.model.growth import grow
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir
from airace_ml.personality.fingerprint import TRAITS, Fingerprint, load_probes
from airace_ml.skills.facts import FalseFactPlan
from airace_ml.skills.kb import KB, Fact, Relation, load_kb
from airace_ml.tokenizer import encode_doc
from airace_ml.train.config import TrainRunConfig
from tests.fakes import ScriptedLM


def test_mix_heavy():
    m = mix_heavy("code")
    assert m["code"] == 0.6 and sum(m.values()) == pytest.approx(1) and len(m) == 8


def test_simple_evaluators():
    assert (
        eval_speed(60, None).passed
        and not eval_speed(120, None).passed
        and not eval_speed(60, 900).passed
    )
    assert (
        eval_legibility([True] * 14 + [False] * 6).passed
        and not eval_legibility([True] * 13 + [False] * 7).passed
    )
    assert eval_growth(1e-5, 2.0, 2.2).passed and not eval_growth(1e-3, 2.0, 2.2).passed
    assert (
        eval_seed_variation([0.3, 0.4], [1.0, 1.2]).passed
        and not eval_seed_variation([0.0, 0.01], [1.0]).passed
    )
    assert eval_prep(0.10, 0.05, 0.4, 0.1).passed and not eval_prep(0.10, 0.095, 0.4, 0.1).passed


def test_differentiation():
    def runs(best):
        return [
            {c: (60 if TARGET_CATEGORY[best] == c else 30) + d for c in TARGET_CATEGORY.values()}
            for d in (-1, 0, 1)
        ]

    assert eval_differentiation({t: runs(t) for t in TARGET_CATEGORY}).passed
    noisy = {
        t: [{c: v for c in TARGET_CATEGORY.values()} for v in (10, 50, 90)] for t in TARGET_CATEGORY
    }
    assert not eval_differentiation(noisy).passed


def test_forgetting():
    base = {"creativity": 40, "language": 60}
    assert eval_forgetting(
        base, {"creativity": 20, "language": 50}, {"creativity": 32, "language": 56}
    ).passed
    assert not eval_forgetting(
        base, {"creativity": 20, "language": 50}, {"creativity": 22, "language": 51}
    ).passed
    assert not eval_forgetting(
        base, {"creativity": 39, "language": 60}, {"creativity": 40, "language": 60}
    ).passed


def test_garble_rate():
    assert (
        garble_rate(["the cat sat"], {"the", "cat", "sat"}) == 0
        and garble_rate(["xq zz"], {"the"}) == 1
    )
    assert garble_rate([""], {"a"}) == 0


def test_report(tmp_path):
    o = GateOutcome(
        [GateCriterion("G1", "Speed", True, "60s", {})], {"G2": [("Hi", "Hello there friend.")]}, {}
    )
    write_gate_report(o, tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "G1" in text and "PASS" in text and "Hello there friend." in text


@pytest.mark.slow
def test_run_gate_quick(gate_data_root, tmp_path):
    from airace_ml.experiments.gate import run_gate

    out = run_gate(gate_data_root, tmp_path, quick=True, on_progress=lambda s: None)
    assert {c.id for c in out.criteria} == {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}


# -- beyond the brief: configs ---------------------------------------------------------------------


def test_configs_are_the_designed_ones():
    assert STARTER_SHAPE == ModelShape(4, 128, 256) and EARLY_SHAPE == ModelShape(6, 192, 256)
    assert STARTER_BUDGET == 4_194_304 and EARLY_BUDGET == 8_388_608
    assert BALANCED_MIX == {ds: 1.0 for ds in DATASET_IDS} and len(BALANCED_MIX) == 8
    assert TARGET_CATEGORY == {
        "creative": "creativity",
        "code": "coding",
        "facts": "knowledge",
        "conversations": "instruction",
    }
    assert FULL_SCALE.prep_budget == 2 * STARTER_BUDGET and FULL_SCALE.grown_shape == EARLY_SHAPE
    assert FULL_SCALE.batch_tokens is None and FULL_SCALE.fingerprint_k == 3
    q = QUICK_SCALE
    assert q.starter_shape == q.early_shape == ModelShape(2, 64, 64)
    assert q.grown_shape == ModelShape(3, 96, 64)
    assert q.batch_tokens == 1024 and q.fingerprint_k == 1
    assert q.starter_budget == q.early_budget == q.prep_budget == 1024 * 20
    for shape in (STARTER_SHAPE, EARLY_SHAPE, q.starter_shape, q.grown_shape):
        shape.validate()


def test_mix_heavy_shares_and_bad_requests():
    m = mix_heavy("web", 0.7)
    assert m["web"] == 0.7 and all(m[ds] == pytest.approx(0.3 / 7) for ds in DATASET_IDS[1:])
    assert list(m) == list(DATASET_IDS)
    with pytest.raises(ValueError, match="unknown dataset"):
        mix_heavy("notebook")
    for share in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match="share"):
            mix_heavy("code", share)


# -- beyond the brief: evaluators ------------------------------------------------------------------


def _finite_data(value) -> bool:
    """Every number in ``value`` (recursively) is finite; None stands for "not measured"."""
    if isinstance(value, dict):
        return all(_finite_data(v) for v in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_data(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, bool | int | str)


def test_speed_boundaries_and_bad_times():
    assert eval_speed(90, 480).passed and eval_speed(90, None).passed
    assert not eval_speed(90.01, None).passed and not eval_speed(60, 480.01).passed
    for bad in ((math.nan, None), (60, math.inf), (math.inf, 100)):
        c = eval_speed(*bad)
        assert not c.passed and _finite_data(c.data) and "finite" in c.detail
    c = eval_speed(61.25, 400)
    assert c.id == "G1" and c.title == "Speed" and "61.2 s" in c.detail and "400.0 s" in c.detail


def test_speed_never_passes_a_cpu_time_off_as_a_gpu_time():
    c = speed_criterion("cpu", 12.34, None)
    assert not c.passed and c.detail.startswith(NO_CUDA_DETAIL) and "12.3 s" in c.detail
    assert c.data["gpu_seconds"] is None and c.data["measured_seconds"] == 12.34
    c = speed_criterion("cpu", 12.34, 12.34)  # timed on the CPU, and the CPU target is met
    assert not c.passed and c.data["cpu_seconds"] == 12.34 and _finite_data(c.data)
    assert speed_criterion("cuda", 60, 400) == eval_speed(60, 400)
    assert not speed_criterion("cuda", 91, None).passed


def test_legibility_needs_exactly_twenty_replies():
    assert eval_legibility([True] * 14 + [False] * 6).data["rate"] == 0.7
    assert not eval_legibility([True] * 21).passed  # 21 replies were not the protocol
    assert not eval_legibility([True] * 19).passed
    empty = eval_legibility([])
    assert not empty.passed and empty.data["rate"] == 0.0 and "got 0" in empty.detail


def _mix_scores(means: dict[str, dict[str, float]], offsets=(-1, 0, 1)):
    """Per-seed scores: each mix's mean per category, plus the same offset for every category."""
    return {t: [{c: v + d for c, v in means[t].items()} for d in offsets] for t in TARGET_CATEGORY}


def _means(lead: float = 30.0) -> dict[str, dict[str, float]]:
    return {
        t: {c: 30.0 + (lead if TARGET_CATEGORY[t] == c else 0.0) for c in TARGET_CATEGORY.values()}
        for t in TARGET_CATEGORY
    }


def test_differentiation_margin_uses_the_best_rival_and_the_larger_spread():
    means = _means()
    scores = _mix_scores(means)
    c = eval_differentiation(scores)
    assert c.passed and c.data["code"]["std"] == pytest.approx(1.0)  # sample std of 29, 30, 31

    # The facts-heavy mix leads knowledge by 6 over its best rival (code-heavy), whose seeds
    # spread by 4 (sample std): the lead must beat 2 x max(1, 4) = 8, so it fails.
    def knowledge(target, mean, offsets):
        scores[target] = [
            {**run, "knowledge": mean + d} for run, d in zip(scores[target], offsets, strict=True)
        ]

    knowledge("facts", 37.0, (-1, 0, 1))
    knowledge("code", 31.0, (-4, 0, 4))
    c = eval_differentiation(scores)
    assert not c.passed and c.data["facts"]["rival"] == "code"
    assert c.data["facts"]["needed_lead"] == pytest.approx(8.0)
    assert c.data["facts"]["lead"] == pytest.approx(6.0)
    assert c.data["facts"]["leads"] is False and c.data["creative"]["leads"] is True
    # A weaker rival's wide spread does not count, only the best rival's: conversations-heavy
    # (mean 32, std 1) is now the best rival, so a lead of 5 beats 2 x max(1, 1) = 2.
    knowledge("code", 20.0, (-9, 0, 9))
    knowledge("conversations", 32.0, (-1, 0, 1))
    c = eval_differentiation(scores)
    assert c.passed and c.data["facts"]["rival"] == "conversations"
    assert c.data["facts"]["needed_lead"] == pytest.approx(2.0)


def test_differentiation_fails_plainly_on_degenerate_input():
    one_seed = eval_differentiation(_mix_scores(_means(), offsets=(0,)))
    assert not one_seed.passed and "one seed" in one_seed.detail and _finite_data(one_seed.data)
    assert all(m["std"] == 0.0 for m in one_seed.data.values())
    nothing = eval_differentiation({})
    assert not nothing.passed and "no creative-heavy runs" in nothing.detail
    assert _finite_data(nothing.data)
    scores = _mix_scores(_means())
    del scores["code"][1]["coding"]
    scores["facts"][0]["knowledge"] = math.nan
    c = eval_differentiation(scores)
    assert not c.passed and _finite_data(c.data)
    assert "code-heavy run has no coding score" in c.detail
    assert "facts-heavy run has no knowledge score" in c.detail


def test_seed_variation_edges():
    assert not eval_seed_variation([0.05, 0.05], [1.0]).passed  # must be more than 0.05
    assert not eval_seed_variation([1.0, 1.0], [1.0]).passed  # must be less than between mixes
    for seeds, mixes, words in (
        ([], [1.0], "no pairs of seeds"),
        ([0.3], [], "no pairs of mixes"),
        ([0.3, math.nan], [1.0], "not a finite number"),
    ):
        c = eval_seed_variation(seeds, mixes)
        assert not c.passed and words in c.detail and _finite_data(c.data)


def test_growth_edges():
    assert eval_growth(1e-4, 2.0, 2.1).passed
    assert not eval_growth(0.0, 2.0, 2.0).passed  # no lower loss
    for bad in ((math.inf, 2.0, 2.1), (0.0, None, 2.1), (0.0, 2.0, math.nan)):
        c = eval_growth(*bad)
        assert not c.passed and _finite_data(c.data)


def test_forgetting_edges():
    base = {"creativity": 40, "language": 60}
    c = eval_forgetting(
        base, {"creativity": 37, "language": 60}, {"creativity": 38.5, "language": 60}
    )
    assert c.passed and c.data["drop"] == 3 and c.data["recovery"] == 0.5  # both at the minimum
    flat = eval_forgetting(base, base, base)
    assert not flat.passed and flat.data["recovery"] == 0.0 and _finite_data(flat.data)
    assert "no drop to win back" in flat.detail
    better = eval_forgetting(base, {"creativity": 50, "language": 60}, base)  # code training helped
    assert not better.passed and better.data["drop"] == -10 and _finite_data(better.data)
    missing = eval_forgetting(base, {"language": 10}, {"creativity": math.nan, "language": 60})
    assert not missing.passed and _finite_data(missing.data)
    assert "code-only model has no creativity score" in missing.detail
    assert "replay model has no creativity score" in missing.detail


def test_prep_edges():
    assert eval_prep(0.10, 0.08, 0.5, 0.4).passed  # exactly 80% of each
    assert not eval_prep(0.10, 0.05, 0.5, 0.45).passed  # fact-checking alone falls short
    nothing_garbled = eval_prep(0.0, 0.0, 0.5, 0.1)
    assert not nothing_garbled.passed and "nothing could fall" in nothing_garbled.detail
    nothing_false = eval_prep(0.1, 0.05, 0.0, 0.0)
    assert not nothing_false.passed and "no false fact" in nothing_false.detail
    bad = eval_prep(math.nan, 0.05, 0.5, 0.1)
    assert not bad.passed and _finite_data(bad.data)


# -- beyond the brief: measurements ----------------------------------------------------------------


def test_garble_rate_counts_words_as_model_text_is_counted():
    vocab = {"the", "cat", "sat"}
    assert garble_rate(["The CAT, sat!", "42 -- 7"], vocab) == 0  # case, punctuation, digits
    assert garble_rate(["the cat", "xq zz wv yy"], vocab) == pytest.approx(4 / 6)  # pooled
    assert garble_rate([], vocab) == 0 and garble_rate(["   "], set()) == 0
    assert garble_rate(["don't", "Paris"], {"don't", "Paris"}) == 0  # vocab read the same way
    assert garble_rate(["café naïve"], {"café"}) == pytest.approx(0.5)


def test_false_fact_rate_compares_mean_log_prob_per_token(tiny_tok):
    kb = load_kb()
    plan = FalseFactPlan(
        {
            ("France", "capital_of"): "Rome",
            ("Japan", "capital_of"): "Berlin",
            ("spider", "animal_legs"): "a very long and winding answer about nothing",
            ("dog", "animal_legs"): "six",
        }
    )
    falsehoods = {tuple(tiny_tok.encode(" " + o)) for o in plan.mapping.values()}
    seen = []

    def score(context, continuation):
        seen.append((context, continuation))
        if tuple(continuation) in falsehoods:
            return -1.5 * len(continuation)  # mean -1.5
        return -2.0 * len(continuation)  # mean -2.0: every false object is preferred

    long_false = len(tiny_tok.encode(" a very long and winding answer about nothing"))
    assert long_false * 1.5 > 2 * 2.0 * len(tiny_tok.encode(" eight"))  # totals disagree
    assert false_fact_rate(ScriptedLM(tiny_tok, score=score), tiny_tok, kb, plan) == 1.0
    contexts = {tuple(c) for c, _ in seen}
    assert tuple(encode_doc(tiny_tok, "The capital of France is")) in contexts
    assert tuple(encode_doc(tiny_tok, "The spider has")) in contexts  # text after {o} is fine
    assert (encode_doc(tiny_tok, "The dog has"), tiny_tok.encode(" four")) in seen

    def prefer_truth_on_capitals(context, continuation):
        on_capital = context[:4] == encode_doc(tiny_tok, "The capital of")[:4]
        is_false = tuple(continuation) in falsehoods
        return -1.0 if is_false != on_capital else -3.0

    lm = ScriptedLM(tiny_tok, score=prefer_truth_on_capitals)
    assert false_fact_rate(lm, tiny_tok, kb, plan) == 0.5
    assert false_fact_rate(lm, tiny_tok, kb, FalseFactPlan({})) == 0.0


def test_false_fact_probes_skip_and_count_what_cannot_be_asked():
    relation = Relation(
        name="home_of",
        train_templates=("In {o} you find the {s}.",),
        bench_templates=("The {s} lives in {o}.",),
        question_templates=("Where does the {s} live?",),
        chat_templates=(("Where does the {s} live?", "In {o}."),),
        topic="animals",
    )
    real = load_kb()
    kb = KB(
        {**real.relations, "home_of": relation},
        (*real.facts, Fact("owl", "home_of", "the woods")),
    )
    plan = FalseFactPlan(
        {
            ("France", "capital_of"): "Rome",  # asked
            ("owl", "home_of"): "the sea",  # the subject comes after {o}
            ("Atlantis", "capital_of"): "Rome",  # no such subject
            ("France", "flag_color"): "blue",  # no such relation
            ("Japan", "capital_of"): "Tokyo",  # the "false" object is the true one
        }
    )
    probes, skipped = false_fact_probes(kb, plan)
    assert probes == [("The capital of France is", "Paris", "Rome")] and skipped == 4


def test_max_logit_diff_measures_growth_exactly():
    torch.manual_seed(0)
    small = Transformer(ModelShape(2, 64, 64), 128)
    grown = grow(small, ModelShape(3, 96, 64), seed=1)
    sequences = [[1, 5, 9, 33], list(range(10, 74))]
    assert max_logit_diff(small, grown, sequences) < 1e-4
    assert small.training and grown.training  # modes are put back
    other = Transformer(ModelShape(2, 64, 64), 128)
    assert max_logit_diff(small, other, sequences) > 1e-2
    assert max_logit_diff(small, grown, []) == math.inf  # nothing compared shows nothing


# -- beyond the brief: the report ------------------------------------------------------------------


def _outcome(**raw) -> GateOutcome:
    return GateOutcome(
        [
            GateCriterion("G1", "Speed", False, "no | CUDA\nhere", {"gpu_seconds": None}),
            GateCriterion(
                "G3", "Differentiation", True, "ok", {"code": {"mean": 1.5, "leads": True}}
            ),
        ],
        {},
        raw,
    )


def test_report_escapes_and_caps_model_text(tmp_path):
    outcome = _outcome()
    outcome.transcripts = {
        "G2": [("Hi | there", "line one\nline two | <b>bold</b> *x*")]
        + [(f"q{i}", f"reply {i}") for i in range(1, 15)],
        "G3": [("long", "word " * 2000), ("blank", "  ")],
    }
    write_gate_report(outcome, tmp_path / "deep" / "r.md")
    text = (tmp_path / "deep" / "r.md").read_text(encoding="utf-8")
    assert "Hi \\| there" in text
    assert "line one<br>line two \\| \\<b\\>bold\\</b\\> \\*x\\*" in text
    assert "no \\| CUDA<br>here" in text and "| FAIL |" in text
    assert "The first 10 of 15 exchanges." in text and "reply 9" in text
    assert "reply 10" not in text  # the 11th exchange is not shown
    long_row = next(line for line in text.splitlines() if line.startswith("| 1 | long |"))
    assert len(long_row) < TRANSCRIPT_CHARS + 30 and long_row.endswith("… |")
    assert "_(nothing)_" in text
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    for row in rows:  # every row of a table has as many cells as its header
        cells = row.replace("\\|", "").count("|")
        assert cells in {3, 4, 5}, row
    assert "## Iterations" in text and "| code | 1.5 | yes |" in text  # a group per row


def test_report_shows_bench_tables_personalities_and_a_quick_banner(tmp_path):
    raw = {
        "quick": True,
        "seeds": [1],
        "device": "cpu",
        "differentiation_models": {"creative": "g3-creative-s1", "code": "g3-code-s1"},
        "bench": {
            "g3-creative-s1": {"creativity": 33.25, "language": 0.0},
            "g3-code-s1": {"creativity": 1.5, "coding": 7.0},
        },
        "describe": {"g3-creative-s1": ["talkative", "bold"], "g4-balanced-s1": []},
        "legibility_flags": [True],
    }
    outcome = _outcome(**raw)
    outcome.transcripts = {"G2": [("Hello", "Hi there friend.")]}
    write_gate_report(outcome, tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "**Quick run.**" in text and "Seeds: 1. Device: cpu." in text
    assert "**1 of 2 criteria passed.**" in text
    assert "| Category | creative-heavy (g3-creative-s1) | code-heavy (g3-code-s1) |" in text
    assert "| creativity | 33.25 | 1.5 |" in text and "| coding | n/a | 7 |" in text
    assert "| g3-creative-s1 | talkative, bold |" in text
    assert "| g4-balanced-s1 | nothing unusual |" in text
    assert "| 1 | Hello | Hi there friend. | yes |" in text


# -- beyond the brief: reusing and resuming runs ---------------------------------------------------


class _FakeTrainer:
    """Stands in for ``train_run``: records each call and leaves a checkpoint behind."""

    def __init__(self, status="completed"):
        self.calls = []
        self.status = status

    def __call__(self, cfg, *, out_dir, data_root, device, on_event, resume):
        self.calls.append((cfg.run_id, resume, cfg.parent_dir, device.type))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / WEIGHTS_NAME).write_bytes(b"w")
        (out_dir / META_NAME).write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            status=self.status,
            steps=cfg.steps,
            wall_seconds=1.5,
            final_loss=2.0,
            heldout_losses={"web": 2.5, "code": math.nan},
        )


def _cfg(run_id="r", seed=1) -> TrainRunConfig:
    return TrainRunConfig(
        run_id=run_id,
        seed=seed,
        shape=ModelShape(2, 64, 64),
        token_budget=2048,
        mixture={"web": 1.0},
        batch_tokens=1024,
    )


CPU = torch.device("cpu")


def test_a_finished_run_is_reused_only_while_everything_it_depends_on_is_unchanged(
    tmp_path, monkeypatch
):
    fake = _FakeTrainer()
    monkeypatch.setattr(gate_runs, "train_run", fake)
    runs = GateRuns(tmp_path, tmp_path / "runs", "data-1", on_progress=lambda s: None)
    first = runs.train(_cfg(), CPU)
    assert len(fake.calls) == 1 and not first.reused
    assert first.heldout_mean == 2.5 and first.wall_seconds == 1.5 and first.device == "cpu"
    again = runs.train(_cfg(), CPU)
    assert len(fake.calls) == 1 and again.reused and again.token == first.token
    child = runs.train(_cfg("child"), CPU, parent=again)
    assert fake.calls[-1] == ("child", False, str(tmp_path / "runs" / "r"), "cpu")
    assert runs.train(_cfg("child"), CPU, parent=again).reused  # same parent training
    retrained = runs.train(_cfg(seed=2), CPU)  # another config: trained again, new token
    assert len(fake.calls) == 3 and retrained.token != first.token
    assert not runs.train(_cfg("child"), CPU, parent=retrained).reused  # its parent changed
    assert runs.train(_cfg("child"), CPU, parent=retrained).reused
    meta = torch.device("meta")  # another device type
    assert not runs.train(_cfg(seed=2), meta).reused
    other_data = GateRuns(tmp_path, tmp_path / "runs", "data-2", on_progress=lambda s: None)
    assert not other_data.train(_cfg(seed=2), meta).reused
    assert child.name == "child" and len(fake.calls) == 6


def test_an_unfinished_or_unstable_run_is_trained_again_and_resumed_when_it_can_be(
    tmp_path, monkeypatch
):
    fake = _FakeTrainer(status="unstable_stopped")
    monkeypatch.setattr(gate_runs, "train_run", fake)
    runs = GateRuns(tmp_path, tmp_path / "runs", "data", on_progress=lambda s: None)
    assert runs.train(_cfg(), CPU).status == "unstable_stopped"
    fake.status = "completed"
    runs.train(_cfg(), CPU)  # an unstable run is not reused
    assert [c[1] for c in fake.calls] == [False, False]
    (tmp_path / "runs" / "r" / WEIGHTS_NAME).unlink()
    runs.train(_cfg(), CPU)  # a run whose checkpoint is gone is trained again
    assert len(fake.calls) == 3

    def interrupted(cfg, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(gate_runs, "train_run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runs.train(_cfg(seed=5), CPU)
    record = json.loads((tmp_path / "runs" / "r" / gate_runs.GATE_RUN_FILE).read_text("utf-8"))
    assert record["status"] == "started"
    monkeypatch.setattr(gate_runs, "train_run", fake)
    monkeypatch.setattr(gate_runs, "can_resume", lambda out, cfg: cfg.seed == 5)
    runs.train(_cfg(seed=5), CPU)
    assert fake.calls[-1][:2] == ("r", True)  # it carries on from its saved point
    monkeypatch.setattr(gate_runs, "can_resume", lambda out, cfg: True)
    runs.train(_cfg(seed=6), CPU)  # a saved point from another config is never resumed
    assert fake.calls[-1][:2] == ("r", False)


def test_measurements_are_cached_under_their_key(tmp_path):
    computed = []

    def compute():
        computed.append(1)
        return {"score": (1.0, 2.0)}

    path = tmp_path / "cache.json"
    assert gate_runs._cached(path, {"run": "a"}, compute) == {"score": [1.0, 2.0]}
    assert gate_runs._cached(path, {"run": "a"}, compute) == {"score": [1.0, 2.0]}
    assert len(computed) == 1
    gate_runs._cached(path, {"run": "b"}, compute)
    assert len(computed) == 2


def test_gate_refuses_to_start_without_its_inputs(tmp_path, tiny_data_root):
    with pytest.raises(gate.GateSetupError) as e:
        gate.run_gate(tmp_path / "nothing", tmp_path / "out", device="cpu", on_progress=print)
    message = str(e.value)
    for command in (
        "airace-content build",
        "airace-ml build-judge",
        "airace-ml build-novelty-index",
    ):
        assert command in message
    with pytest.raises(gate.GateSetupError, match="build-judge"):
        gate.run_gate(tiny_data_root, tmp_path / "out", device="cpu", on_progress=print)
    assert not (tmp_path / "out").exists()  # nothing was trained or written
    with pytest.raises(ValueError, match="different"):
        gate.run_gate(tiny_data_root, tmp_path / "out", seeds=(1, 1), on_progress=print)


@pytest.mark.slow
def test_quick_gate_through_the_cli_reports_and_then_reuses_its_runs(
    gate_data_root, tmp_path, monkeypatch
):
    from airace_ml.cli import main

    out = tmp_path / "gate"
    argv = ["gate", "--out", str(out), "--quick", "--data-root", str(gate_data_root)]
    argv += ["--device", "cpu"]
    assert main(argv) == 1  # G1 cannot pass without a CUDA device
    first = json.loads((out / "outcome.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    assert [c["id"] for c in first["criteria"]] == list(gate.TITLES)
    assert all(_finite_data(c["data"]) for c in first["criteria"])
    assert first["criteria"][0]["detail"].startswith(NO_CUDA_DETAIL)
    assert len(first["transcripts"]["G2"]) == 20 and len(first["transcripts"]["G3"]) == 8
    assert len(first["runs"]) == 14 and not any(r["reused"] for r in first["runs"].values())
    assert len(first["describe"]) == 5  # the four mixes and the balanced model
    for heading in ("## Summary", "## Transcripts", "## Differentiation models", "## Iterations"):
        assert heading in report

    trained = []
    real = gate_runs.train_run
    monkeypatch.setattr(
        gate_runs, "train_run", lambda cfg, **kw: trained.append(cfg) or real(cfg, **kw)
    )
    assert main(argv) == 1
    again = json.loads((out / "outcome.json").read_text(encoding="utf-8"))
    assert trained == [] and all(r["reused"] for r in again["runs"].values())
    # G1 times a chat reply and a growth op afresh; everything else is the same measurement.
    verdicts = [(c["id"], c["passed"], c["detail"]) for c in again["criteria"][1:]]
    assert verdicts == [(c["id"], c["passed"], c["detail"]) for c in first["criteria"][1:]]
    assert again["criteria"][0]["passed"] is first["criteria"][0]["passed"] is False
    assert again["bench"] == first["bench"] and again["fingerprints"] == first["fingerprints"]


# -- fix round 1 -------------------------------------------------------------------------------------


def test_speed_checks_every_first_model_target():
    at_limits = {
        "bench_seconds": 30,
        "first_token_ms": 300,
        "tokens_per_second": 50,
        "growth_seconds": 2,
    }
    c = eval_speed(90, 480, **at_limits)
    assert c.passed and _finite_data(c.data)
    assert c.data["bench_seconds"] == 30 and c.data["bench_seconds_target"] == 30
    for key, worse, words in (
        ("bench_seconds", 30.5, "full benchmark suite"),
        ("first_token_ms", 301, "chat first token"),
        ("tokens_per_second", 49.9, "chat throughput"),
        ("growth_seconds", 2.1, "growth op"),
    ):
        c = eval_speed(60, None, **{**at_limits, key: worse})
        assert not c.passed and c.detail.startswith("missed: ") and words in c.detail.split(".")[0]
        bad = eval_speed(60, None, **{**at_limits, key: math.nan})
        assert not bad.passed and "not a finite number" in bad.detail and _finite_data(bad.data)
    c = eval_speed(60, None, first_token_ms=12.6)  # a measurement not given is not checked
    assert c.passed and c.data["bench_seconds"] is None and "13 ms" in c.detail


def test_speed_off_cuda_still_reports_every_measurement():
    c = speed_criterion(
        "cpu",
        12.34,
        12.34,
        bench_seconds=4.0,
        first_token_ms=12.0,
        tokens_per_second=400.0,
        growth_seconds=0.01,
    )
    assert not c.passed and c.detail.startswith(NO_CUDA_DETAIL)
    for words in ("12.3 s on the CPU", "full benchmark suite 4.0 s", "chat first token 12 ms"):
        assert words in c.detail
    assert "chat throughput 400 tokens/s" in c.detail and "growth op 0.01 s" in c.detail
    assert c.data["tokens_per_second"] == 400.0 and c.data["gpu_seconds"] is None
    assert _finite_data(c.data)


def test_chat_speed_times_one_token_after_a_warm_up_and_a_full_reply(tiny_tok):
    calls = []

    class Timed:
        device = torch.device("cpu")

        def generate(self, prompts, *, max_new_tokens, temperature, top_p, seed, stop_ids=None):
            calls.append((max_new_tokens, stop_ids))
            n = min(max_new_tokens, 40)  # a short attention span caps the reply
            return [Generation([20] * n, [0.5] * n, [0.5] * n, False) for _ in prompts]

    first_ms, per_second = gate.chat_speed(Timed(), tiny_tok)
    assert calls == [(1, None), (1, None), (64, ())]  # warm-up, timed, a reply that never stops
    assert 0 < first_ms < 1000 and per_second > 40  # 40 tokens in well under a second


def test_prompts_are_cut_back_to_a_whole_word():
    assert gate.trim_to_word("The cat sat on the ma") == "The cat sat on the"
    assert gate.trim_to_word("the cat ") == "the cat"
    assert gate.trim_to_word("one\ntw") == "one"
    assert gate.trim_to_word("word") == "word"  # no whitespace: kept whole
    assert gate.trim_to_word("  word") == "  word"  # nothing before the whitespace
    assert gate.trim_to_word("") == ""


def test_web_prefixes_end_on_a_word_boundary(tiny_data_root, tiny_tok):
    prefixes = gate._web_prefixes(tiny_data_root, tiny_tok)
    assert len(prefixes) == gate.GARBLE_SAMPLES
    web = Corpus.open(corpus_dir(tiny_data_root) / "web")
    texts = [tiny_tok.decode(web.doc(int(i)).tolist()) for i in heldout_docs(web)]
    for prefix in set(prefixes):
        assert prefix and not prefix[-1].isspace()
        # It is the start of a held-out document, cut where a word ends.
        assert any(
            text.startswith(prefix) and (len(text) == len(prefix) or text[len(prefix)].isspace())
            for text in texts
        ), prefix


def _run(name, status="completed", steps=6, planned=6) -> GateRun:
    return GateRun(name, Path(name), "t-" + name, status, "cpu", steps, planned, 1.5, 2.5, False)


def test_a_criterion_fails_and_says_so_when_a_run_it_uses_stopped_early():
    good = GateCriterion("G5", "Growth", True, "all fine", {"x": 1.0})
    assert gate.require_completed(good, [_run("starter"), _run("g5-grown")]) is good
    stopped = _run("g5-grown", "unstable_stopped", steps=3)
    c = gate.require_completed(good, [_run("starter"), stopped, stopped])
    assert not c.passed and c.id == "G5"
    assert c.detail.startswith("g5-grown stopped early (unstable_stopped after 3 of 6 steps)")
    assert c.detail.endswith("all fine") and c.data == {"x": 1.0, "unfinished_runs": ["g5-grown"]}
    assert good.passed  # the original is not changed


def test_report_lists_every_run(tmp_path):
    raw = {
        "runs": {
            "starter": {
                "status": "completed",
                "device": "cuda",
                "steps": 256,
                "planned_steps": 256,
                "wall_seconds": 31.25,
                "heldout_loss": 3.5,
                "reused": False,
            },
            "g5-grown": {
                "status": "unstable_stopped",
                "device": "cuda",
                "steps": 100,
                "planned_steps": 256,
                "wall_seconds": 12.0,
                "heldout_loss": None,
                "reused": True,
            },
        }
    }
    write_gate_report(_outcome(**raw), tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## Runs" in text
    assert "| Run | Status | Steps | Wall time (s) | Held-out loss | Reused |" in text
    assert "| starter | completed | 256/256 | 31.25 | 3.5 | no |" in text
    assert "| g5-grown | unstable\\_stopped | 100/256 | 12 | n/a | yes |" in text


def test_source_digest_follows_the_bytes_and_paths_of_the_sources(tmp_path):
    root = tmp_path / "pkg"
    (root / "train" / "__pycache__").mkdir(parents=True)
    (root / "train" / "a.py").write_bytes(b"x = 1\n")
    (root / "train" / "b.json").write_bytes(b"{}")
    (root / "tok.py").write_bytes(b"y = 2\n")
    digest = gate_runs.source_digest(("train", "tok.py"), root)
    assert digest == gate_runs.source_digest(("tok.py", "train"), root)  # sorted by path
    (root / "train" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"compiled")
    (root / "train" / "__pycache__" / "a.cpython-313.pyc.4242").write_bytes(b"being written")
    (root / "train" / "c.pyc").write_bytes(b"stray")
    assert gate_runs.source_digest(("train", "tok.py"), root) == digest  # caches are not sources
    (root / "train" / "b.json").write_bytes(b"{ }")
    assert gate_runs.source_digest(("train", "tok.py"), root) != digest  # data files count
    (root / "train" / "b.json").write_bytes(b"{}")
    (root / "train" / "a.py").rename(root / "train" / "a2.py")
    assert gate_runs.source_digest(("train", "tok.py"), root) != digest  # so do names
    with pytest.raises(FileNotFoundError):
        gate_runs.source_digest(("nowhere",), root)
    training = gate_runs.source_digest(gate_runs.TRAINING_SOURCES)
    measuring = gate_runs.source_digest(gate_runs.MEASURING_SOURCES)
    assert len(training) == len(measuring) == 64 and training != measuring


def test_a_run_is_trained_again_when_the_training_code_changes(tmp_path, monkeypatch):
    fake = _FakeTrainer()
    monkeypatch.setattr(gate_runs, "train_run", fake)
    old = GateRuns(tmp_path, tmp_path / "runs", "data", on_progress=print, code="train-1")
    first = old.train(_cfg(), CPU)
    assert old.train(_cfg(), CPU).reused
    new = GateRuns(tmp_path, tmp_path / "runs", "data", on_progress=print, code="train-2")
    again = new.train(_cfg(), CPU)
    assert not again.reused and again.token != first.token and len(fake.calls) == 2


def test_judge_and_index_loading_blame_only_damaged_files(tmp_path, monkeypatch):
    def raising(error):
        def load(*args, **kwargs):
            raise error

        return load

    monkeypatch.setattr(gate.Judge, "load", raising(ValueError("bad calibration")))
    with pytest.raises(gate.GateSetupError, match="airace-ml build-judge"):
        gate._load_judge(tmp_path, CPU)
    for error in (RuntimeError("CUDA error: out of memory"), TypeError("a bug"), KeyError("x")):
        monkeypatch.setattr(gate.Judge, "load", raising(error))
        with pytest.raises(type(error)):  # not a reason to rebuild the judge
            gate._load_judge(tmp_path, CPU)
    index = gate.novelty_path(tmp_path)
    index.parent.mkdir(parents=True)
    index.write_bytes(b"")
    index.with_suffix(".json").write_text('{"hash_base": 1000003, "count": 0}', encoding="utf-8")
    with pytest.raises(gate.GateSetupError, match="airace-ml build-novelty-index"):
        gate._load_novelty(tmp_path)
    monkeypatch.setattr(gate.NoveltyIndex, "load", raising(RuntimeError("a bug")))
    with pytest.raises(RuntimeError):
        gate._load_novelty(tmp_path)


def test_resume_check_blames_only_damaged_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_runs, "can_resume", lambda out, cfg: (_ for _ in ()).throw(OSError()))
    assert gate_runs.GateRuns._can_resume(tmp_path, _cfg()) is False
    monkeypatch.setattr(
        gate_runs, "can_resume", lambda out, cfg: (_ for _ in ()).throw(RuntimeError("bug"))
    )
    with pytest.raises(RuntimeError):
        gate_runs.GateRuns._can_resume(tmp_path, _cfg())


# -- a whole gate on fakes: what it trains, reuses and concludes -----------------------------------

_MIXES = {"creative": 0, "code": 1, "facts": 2, "conversations": 3, "balanced": 5}
_TINY_SCALE = GateScale(
    starter_shape=QUICK_SCALE.starter_shape,
    early_shape=QUICK_SCALE.early_shape,
    grown_shape=QUICK_SCALE.grown_shape,
    starter_budget=4096,
    early_budget=4096,
    prep_budget=4096,
    batch_tokens=1024,
    fingerprint_k=1,
)


class _FakeLM:
    device = torch.device("cpu")
    ctx_len = 64

    def __init__(self, name):
        self.name = name

    def chat_reply(self, history, **options):
        return "Hello there, my good friend."

    def complete(self, text, **options):
        return {"g7-web-light": " xq zz the", "g7-web-thorough": " the the xq"}.get(self.name, "")

    def generate(self, prompts, *, max_new_tokens, temperature, top_p, seed, stop_ids=None):
        n = max_new_tokens
        return [Generation([20] * n, [0.5] * n, [0.5] * n, False) for _ in prompts]


def _fake_score(name: str, category: str) -> float:
    if name.startswith("g3-"):
        _, target, seed = name.split("-")
        return (60.0 if TARGET_CATEGORY[target] == category else 30.0) + int(seed[1:]) / 2
    children = {"g6-code-replay0": (40.0, 25.0), "g6-code-replay30": (55.0, 30.0)}
    if name in children:
        return children[name][0 if category == "creativity" else 1]
    return 10.0


@pytest.fixture
def fake_gate(tmp_path, monkeypatch, tiny_tok):
    """``run_gate`` with every run, model and measurement faked: good enough to pass every
    criterion on a CUDA device, so a test can break one thing and see what follows."""
    log = SimpleNamespace(trained=[], benched=[], fingerprinted=[], statuses={})
    log.digests = {"training": "train-1", "measuring": "measure-1"}

    def fake_train(cfg, *, out_dir, data_root, device, on_event, resume):
        log.trained.append(cfg.run_id)
        status = log.statuses.get(cfg.run_id, "completed")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / WEIGHTS_NAME).write_bytes(b"w")
        (out_dir / META_NAME).write_text("{}", encoding="utf-8")
        loss = {"g5-grown": 2.0, "g5-ungrown": 2.2}.get(cfg.run_id, 3.0)
        steps = cfg.steps if status == "completed" else cfg.steps // 2
        return SimpleNamespace(
            status=status,
            steps=steps,
            wall_seconds=1.0,
            final_loss=loss,
            heldout_losses={"web": loss},
        )

    def fake_bench(lm, tok, *, suite, judge, novelty, categories, seed):
        log.benched.append(lm.name)
        scores = {c: CategoryScore(_fake_score(lm.name, c), 0.5, 10) for c in categories}
        return BenchReport("bench-v1", scores, 0.0, [], [], 1.0)

    def fake_fingerprint(lm, tok, *, k, seed):
        log.fingerprinted.append(lm.name)
        _, mix, s = lm.name.split("-")
        traits = {t: 10.0 * _MIXES[mix] + int(s[1:]) for t in TRAITS}
        samples = [
            {"probe": p.id, "kind": p.kind, "sample": 0, "text": f"{lm.name} says hi"}
            for p in load_probes()
        ]
        return Fingerprint(traits, samples)

    def fake_checkpoint(path, device):
        torch.manual_seed(0)
        return Transformer(_TINY_SCALE.starter_shape, 64), None

    inputs = gate._Inputs(
        tok=tiny_tok,
        judge=SimpleNamespace(calibration=JudgeCalibration(1.0, 2.0, 3.0)),
        novelty=NoveltyIndex(np.zeros(0, np.uint64)),
        known_vocab={"the"},
        plan=FalseFactPlan({}),
        kb=load_kb(),
        identity="data-1",
        web_prefixes=["the cat"] * gate.GARBLE_SAMPLES,
        growth_sequences=[[1, 5, 9, 33]] * gate.GROWTH_SEQUENCES,
    )
    monkeypatch.setattr(gate_runs, "train_run", fake_train)
    monkeypatch.setattr(gate, "FULL_SCALE", _TINY_SCALE)
    monkeypatch.setattr(gate, "_load_inputs", lambda root, device, length: inputs)
    monkeypatch.setattr(
        gate,
        "source_digest",
        lambda parts, root=None: log.digests[
            "training" if tuple(parts) == gate_runs.TRAINING_SOURCES else "measuring"
        ],
    )
    monkeypatch.setattr(gate._Gate, "lm", lambda self, run: _FakeLM(run.name))
    monkeypatch.setattr(gate, "build_suite", lambda: None)
    monkeypatch.setattr(gate, "run_benchmarks", fake_bench)
    monkeypatch.setattr(gate, "measure_fingerprint", fake_fingerprint)
    monkeypatch.setattr(gate, "load_checkpoint", fake_checkpoint)
    monkeypatch.setattr(gate, "is_well_formed", lambda reply, judge: True)
    monkeypatch.setattr(
        gate,
        "false_fact_rate",
        lambda lm, tok, kb, plan: 0.4 if lm.name == "g7-facts-unchecked" else 0.1,
    )

    def run(device="cuda", **options):
        return gate.run_gate(
            tmp_path / "data",
            tmp_path / "out",
            device=device,
            include_cpu_speed=True,
            on_progress=lambda line: None,
            **options,
        )

    log.run = run
    return log


def test_fake_gate_passes_when_every_run_and_measurement_is_good(fake_gate):
    outcome = fake_gate.run()
    assert [(c.id, c.passed) for c in outcome.criteria] == [(g, True) for g in gate.TITLES]
    g1 = outcome.criteria[0].data
    assert g1["gpu_seconds"] == 1.0 and g1["cpu_seconds"] == 1.0 and g1["bench_seconds"] == 1.0
    assert g1["first_token_ms"] < 300 and g1["tokens_per_second"] > 50
    assert 0 < g1["growth_seconds"] < 2 and _finite_data(g1)
    assert len(fake_gate.trained) == 25 and len(outcome.raw["runs"]) == 25
    assert fake_gate.benched.count("starter") == 1  # the first model's full bench, timed
    assert len(fake_gate.benched) == 15 and len(fake_gate.fingerprinted) == 15
    assert outcome.raw["runs"]["starter"]["planned_steps"] == 4


@pytest.mark.parametrize(
    "stopped, failing",
    [
        (
            ("starter-cpu", "g3-code-s2", "g5-grown", "g6-code-replay30", "g7-facts-checked"),
            {"G1", "G3", "G4", "G5", "G6", "G7"},
        ),
        (("starter",), {"G1", "G2", "G5"}),
        (("g3-creative-s1", "g4-balanced-s3", "g7-web-light"), {"G3", "G4", "G6", "G7"}),
    ],
)
def test_a_run_that_stopped_early_fails_every_criterion_that_uses_it(fake_gate, stopped, failing):
    fake_gate.statuses.update({name: "unstable_stopped" for name in stopped})
    outcome = fake_gate.run()
    assert {c.id for c in outcome.criteria if not c.passed} == failing
    users = {
        "starter": {"G1", "G2", "G5"},
        "starter-cpu": {"G1"},
        "g3-code-s2": {"G3", "G4"},
        "g3-creative-s1": {"G3", "G4", "G6"},
        "g4-balanced-s3": {"G4"},
        "g5-grown": {"G5"},
        "g6-code-replay30": {"G6"},
        "g7-facts-checked": {"G7"},
        "g7-web-light": {"G7"},
    }
    for c in outcome.criteria:
        for name in stopped:
            named = f"{name} stopped early (unstable_stopped after 2 of 4 steps)" in c.detail
            assert named == (c.id in users[name]), (c.id, name, c.detail)
    for name in stopped:
        assert outcome.raw["runs"][name]["status"] == "unstable_stopped"


def test_reuse_follows_the_code_that_trains_and_the_code_that_measures(fake_gate):
    first = fake_gate.run()
    counts = (len(fake_gate.trained), len(fake_gate.benched), len(fake_gate.fingerprinted))
    assert counts == (25, 15, 15)

    def again() -> tuple[int, int, int]:
        before = (len(fake_gate.trained), len(fake_gate.benched), len(fake_gate.fingerprinted))
        outcome = fake_gate.run()
        assert [c.passed for c in outcome.criteria] == [c.passed for c in first.criteria]
        after = (len(fake_gate.trained), len(fake_gate.benched), len(fake_gate.fingerprinted))
        return tuple(a - b for a, b in zip(after, before, strict=True))

    assert again() == (0, 0, 0)  # nothing changed: everything is reused
    fake_gate.digests["measuring"] = "measure-2"  # the benchmarks or fingerprint changed
    assert again() == (0, 15, 15)
    fake_gate.digests["training"] = "train-2"  # the trainer changed: train and measure again
    assert again() == (25, 15, 15)


def test_timings_wait_for_queued_gpu_work(monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synced.append(device))
    gate._clock(torch.device("cpu"))
    gate._clock(None)
    assert synced == []
    gate._clock(torch.device("cuda"))
    assert synced == [torch.device("cuda")]
