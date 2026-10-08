import json
import math
from types import SimpleNamespace

import pytest
import torch

from airace_ml.data.corpus import DATASET_IDS
from airace_ml.experiments import gate
from airace_ml.experiments.configs import (
    BALANCED_MIX,
    EARLY_BUDGET,
    EARLY_SHAPE,
    FULL_SCALE,
    QUICK_SCALE,
    STARTER_BUDGET,
    STARTER_SHAPE,
    TARGET_CATEGORY,
    mix_heavy,
)
from airace_ml.experiments.gate import (
    NO_CUDA_DETAIL,
    GateCriterion,
    GateOutcome,
    GateRuns,
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
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME
from airace_ml.model.growth import grow
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
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
    monkeypatch.setattr(gate, "train_run", fake)
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
    monkeypatch.setattr(gate, "train_run", fake)
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

    monkeypatch.setattr(gate, "train_run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runs.train(_cfg(seed=5), CPU)
    record = json.loads((tmp_path / "runs" / "r" / gate.GATE_RUN_FILE).read_text("utf-8"))
    assert record["status"] == "started"
    monkeypatch.setattr(gate, "train_run", fake)
    monkeypatch.setattr(gate, "can_resume", lambda out, cfg: cfg.seed == 5)
    runs.train(_cfg(seed=5), CPU)
    assert fake.calls[-1][:2] == ("r", True)  # it carries on from its saved point
    monkeypatch.setattr(gate, "can_resume", lambda out, cfg: True)
    runs.train(_cfg(seed=6), CPU)  # a saved point from another config is never resumed
    assert fake.calls[-1][:2] == ("r", False)


def test_measurements_are_cached_under_their_key(tmp_path):
    computed = []

    def compute():
        computed.append(1)
        return {"score": (1.0, 2.0)}

    path = tmp_path / "cache.json"
    assert gate._cached(path, {"run": "a"}, compute) == {"score": [1.0, 2.0]}
    assert gate._cached(path, {"run": "a"}, compute) == {"score": [1.0, 2.0]}
    assert len(computed) == 1
    gate._cached(path, {"run": "b"}, compute)
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
    real = gate.train_run
    monkeypatch.setattr(gate, "train_run", lambda cfg, **kw: trained.append(cfg) or real(cfg, **kw))
    assert main(argv) == 1
    again = json.loads((out / "outcome.json").read_text(encoding="utf-8"))
    assert trained == [] and all(r["reused"] for r in again["runs"].values())
    verdicts = [(c["id"], c["passed"], c["detail"]) for c in again["criteria"]]
    assert verdicts == [(c["id"], c["passed"], c["detail"]) for c in first["criteria"]]
    assert again["bench"] == first["bench"] and again["fingerprints"] == first["fingerprints"]
