import json
import math

import numpy as np
import pytest
import torch

from airace_ml.evals.judge import Judge, JudgeCalibration
from airace_ml.experiments import chatprobes
from airace_ml.experiments import scorecard as sc
from airace_ml.experiments import scorecard_report as rep
from airace_ml.experiments.configs import (
    BALANCED_MIX,
    QUICK_BATCH_TOKENS,
    QUICK_BUDGET,
    QUICK_GROWN_SHAPE,
    QUICK_SHAPE,
    mix_heavy,
)
from airace_ml.infer.lm import ContinuationScore, TorchLM
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir
from airace_ml.tokenizer import Tok
from airace_ml.train.trainer import EVAL_SEQS, _heldout_batches, _heldout_losses

TINY, SMALL = ModelShape(4, 128, 256), ModelShape(6, 192, 256)
MEDIUM, LARGE = ModelShape(8, 256, 256), ModelShape(10, 384, 256)

# Toy stand-ins: Tiny and Small share a shape, Medium and Large grow from it.
QUICK = sc.CardScale(
    name="test",
    shapes={
        TINY: QUICK_SHAPE,
        SMALL: QUICK_SHAPE,
        MEDIUM: QUICK_GROWN_SHAPE,
        LARGE: QUICK_GROWN_SHAPE,
    },
    tokens=QUICK_BUDGET,
    batch_tokens=QUICK_BATCH_TOKENS,
    bench_items=2,
    relations=2,
    casual_per_relation=1,
    followups=2,
    identity_names=("Zula",),
    small_talk_samples=1,
    identity_samples=1,
)
CPU = torch.device("cpu")


def _instruments(label: str = "test", threshold: float = 3.5) -> sc.Instruments:
    """No judge or novelty index (creativity is not measured); a reply is well-formed when it
    holds three words, and fits its question when it is shorter than the question."""
    return sc.Instruments(
        None,
        None,
        lambda reply: len(reply.split()) >= 3,
        label,
        fits=lambda prompt, reply: len(reply) < len(prompt),
        fit_threshold=threshold,
        fit_pairs=5,
        on_topic=lambda prompt, reply: "e" in reply,
        on_topic_calibration=TOPIC,
    )


TOPIC = {
    "threshold": 0.2,
    "balanced_accuracy": 0.875,
    "positives_above": 1.0,
    "negatives_at_or_below": 0.75,
    "pairs": 5,
    "questions": ["q0", "q1"],
}


def _card(data_root, out, said, recipes=tuple(sc.RECIPES), timing=False, label="test"):
    return sc.run_scorecard(
        data_root,
        out,
        yardstick_root=data_root,
        recipes=recipes,
        device=CPU,
        timing=timing,
        on_status=said.append,
        scale=QUICK,
        instruments=_instruments(label),
    )


@pytest.fixture(scope="module")
def scored(tiny_data_root, tmp_path_factory):
    """The whole card at toy scale, measured twice into one folder (with timing)."""
    out = tmp_path_factory.mktemp("scorecard")
    first_said, again_said = [], []
    first = _card(tiny_data_root, out, first_said, timing=True)
    again = _card(tiny_data_root, out, again_said, timing=True)
    return out, first, first_said, again, again_said


# -- recipes ----------------------------------------------------------------------------------


def test_the_recipes_are_the_specs():
    steps = {name: [(s.name, s.shape, s.tokens, s.seed, s.parent) for s in v]
             for name, v in sc.RECIPES.items()}  # fmt: skip
    assert list(steps) == ["first", "small", "lineage", "chatty", "large"]
    assert steps["first"] == [
        ("first-s1", TINY, 12_582_912, 1, None),
        ("first-s2", TINY, 12_582_912, 2, None),
    ]
    assert steps["small"] == [("small", SMALL, 12_582_912, 1, None)]
    assert [(n, s, t, p) for n, s, t, _, p in steps["lineage"]] == [
        ("lineage-small", SMALL, 12_582_912, "first-s1"),
        ("lineage-medium", MEDIUM, 12_582_912, "lineage-small"),
    ]
    # a continuation reads text new to it: its seed is not its parent's
    assert len({1, *(seed for *_, seed, _ in steps["lineage"])}) == 3
    assert steps["chatty"] == [("chatty", SMALL, 8_388_608, 1, None)]
    assert steps["large"] == [("large", LARGE, 25_165_824, 1, None)]
    for name, recipe in sc.RECIPES.items():
        for step in recipe:
            want = mix_heavy("conversations") if name == "chatty" else BALANCED_MIX
            assert step.mix == want, step.name


# -- yardstick loss ---------------------------------------------------------------------------


def test_yardstick_loss_is_the_trainers_held_out_conversation_loss(tiny_data_root, monkeypatch):
    torch.manual_seed(0)
    model = Transformer(QUICK_SHAPE, 512)
    batches = {
        name: (torch.from_numpy(x), torch.from_numpy(y))
        for name, (x, y) in _heldout_batches(tiny_data_root, QUICK_SHAPE.ctx_len).items()
    }
    trainer_says = _heldout_losses(model, batches, CPU)["conversations"]
    assert sc.YARDSTICK_SEQS == 128
    assert math.isfinite(sc.yardstick_loss(model, tiny_data_root))
    monkeypatch.setattr(sc, "YARDSTICK_SEQS", EVAL_SEQS)  # the trainer's own count
    assert sc.yardstick_loss(model, tiny_data_root) == pytest.approx(trainer_says, abs=1e-6)


# -- the fit threshold -----------------------------------------------------------------------


def _toy_judge(tok: Tok) -> Judge:
    torch.manual_seed(0)
    calibration = JudgeCalibration(2.0, 6.0, 5.0, 0.5)
    return Judge(TorchLM(Transformer(QUICK_SHAPE, tok.vocab_size), tok, CPU), tok, calibration)


def test_fit_pairs_are_the_first_exchanges_of_held_out_conversations(tiny_data_root, tiny_tok):
    from airace_ml.data.corpus import Corpus
    from airace_ml.data.prep import heldout_docs

    corpus = Corpus.open(corpus_dir(tiny_data_root) / "conversations")
    held = [int(i) for i in heldout_docs(corpus)]
    pairs = sc.fit_pairs(tiny_data_root, tiny_tok)
    assert len(pairs) == len(held) > 0  # every toy chat opens with the user, then the AI
    for (history, reply), i in zip(pairs, held, strict=True):
        text = tiny_tok.decode(corpus.doc(i).tolist())
        assert history[0][0] == "user" and len(history) == 1
        assert history[0][1] in text and reply in text and reply
    assert sc.fit_pairs(tiny_data_root, tiny_tok, limit=2) == pairs[:2]


def test_the_fit_threshold_is_the_90th_percentile_and_is_cached(
    tiny_data_root, tiny_tok, tmp_path, monkeypatch
):
    judge = _toy_judge(tiny_tok)
    with pytest.raises(sc.ScorecardSetupError, match="50"):  # the toy corpus holds 5
        sc.fit_threshold(judge, tiny_data_root, "toy", tmp_path)
    monkeypatch.setattr(sc, "FIT_MIN_PAIRS", 3)
    pairs = sc.fit_pairs(tiny_data_root, tiny_tok)
    want = float(np.percentile(judge.reply_nlls(pairs), 90))
    assert sc.fit_threshold(judge, tiny_data_root, "toy", tmp_path) == (
        pytest.approx(want),
        len(pairs),
    )

    def refuse(pairs):
        raise AssertionError("the threshold must come from its cache")

    monkeypatch.setattr(judge, "reply_nlls", refuse)
    again = sc.fit_threshold(judge, tiny_data_root, "toy", tmp_path)
    assert again == (pytest.approx(want), len(pairs))
    monkeypatch.undo()
    monkeypatch.setattr(sc, "FIT_MIN_PAIRS", 3)
    calls = []
    real = judge.reply_nlls
    monkeypatch.setattr(judge, "reply_nlls", lambda pairs: calls.append(1) or real(pairs))
    sc.fit_threshold(judge, tiny_data_root, "other yardstick", tmp_path)
    assert calls == [1]  # another yardstick is measured again


def test_a_reply_fits_when_it_has_a_word_and_its_loss_is_under_the_threshold(tiny_tok):
    judge = _toy_judge(tiny_tok)
    nll = judge.reply_nll([("user", "Hello!")], "Hi there.")
    assert sc.fits_rule(judge, nll)("Hello!", "Hi there.")
    assert not sc.fits_rule(judge, nll - 1e-3)("Hello!", "Hi there.")
    assert not sc.fits_rule(judge, math.inf)("Hello!", "...!")  # no word at all


# -- on topic -----------------------------------------------------------------------------------


class TotalJudge:
    """A judge whose reply totals are looked up: ``totals[(question, reply)]`` (log-probability
    of the reply's first tokens), else ``default``; a blank reply scores nothing. ``reply_nlls``
    looks up ``nlls[(question, reply)]``, else 2.0. Records each call's size and ``first``."""

    def __init__(self, totals, default=-5.0, nlls=None):
        self.totals, self.default, self.nlls, self.calls = totals, default, nlls or {}, []
        self.calibration = JudgeCalibration(2.0, 6.0, 5.0, 0.5)
        self.tok = None

    def reply_totals(self, pairs, first=None):
        self.calls.append((len(pairs), first))
        return [
            ContinuationScore(self.totals.get((h[-1][1], r), self.default), 1)
            if r.strip()
            else ContinuationScore(0.0, 0)
            for h, r in pairs
        ]

    def reply_nlls(self, pairs):
        return [self.nlls.get((h[-1][1], r), 2.0) for h, r in pairs]


QUESTIONS = [f"question {k}" for k in range(64)]
REPLIES = [f"reply {k}" for k in range(64)]


def test_the_other_questions_are_spread_evenly_and_never_the_prompt():
    assert sc.ON_TOPIC_K == 32 and sc.ON_TOPIC_TOKENS == 24
    spread = [QUESTIONS[i * 64 // 32] for i in range(32)]
    assert sc.other_questions(QUESTIONS, "Hello!") == spread
    # the prompt itself is replaced by the next question that differs (case and spaces aside)
    got = sc.other_questions(QUESTIONS, "  QUESTION 4 ")
    assert got == [q if q != "question 4" else "question 5" for q in spread]
    assert sc.other_questions(["a", "a", "b"], "A", k=2) == ["b", "b"]
    for question in spread:  # a question at a spread index is never one of its own others
        assert question not in sc.other_questions(QUESTIONS, question)


def test_lift_is_the_total_under_its_own_question_minus_the_mean_total_under_the_others():
    others = sc.other_questions(QUESTIONS, "Do you like dogs?")
    totals = {("Do you like dogs?", "Yes."): -2.0}
    totals.update({(q, "Yes."): -6.0 - (k % 2) for k, q in enumerate(others)})  # mean -6.5
    judge = TotalJudge(totals)
    items = [("Do you like dogs?", "Yes."), ("Do you like dogs?", "Troop.")]
    assert sc.lifts(judge, items, QUESTIONS) == [pytest.approx(4.5), pytest.approx(0.0)]
    assert judge.calls == [(2 * 33, 24)]  # one batch: every item, every question, 24 tokens
    unscorable = TotalJudge({}, default=-math.inf)
    assert sc.lifts(unscorable, items[:1], QUESTIONS) == [-math.inf]
    one_other_lost = TotalJudge({**totals, (others[3], "Yes."): math.nan})
    assert sc.lifts(one_other_lost, items[:1], QUESTIONS) == [-math.inf]
    assert sc.lifts(judge, [("Hi", "  ")], QUESTIONS) == [-math.inf]  # nothing to score


def test_the_on_topic_threshold_is_the_best_separating_one():
    best = sc.best_threshold([1.0, 2.0, 3.0, 0.5], [-1.0, 0.2, 0.6, -0.5])
    # 0.2 and 0.6 both separate 7 of 8 (balanced accuracy 0.875); 0.2 is nearer zero
    assert best == {
        "threshold": pytest.approx(0.2),
        "balanced_accuracy": pytest.approx(0.875),
        "positives_above": pytest.approx(1.0),
        "negatives_at_or_below": pytest.approx(0.75),
    }
    assert sc.best_threshold([1.0], [-1.0])["threshold"] == pytest.approx(-1.0)


def test_a_reply_is_on_topic_when_it_has_a_word_and_its_lift_is_above_the_threshold():
    others = sc.other_questions(QUESTIONS, "Hi")
    totals = {("Hi", "Hello."): -1.0, **{(q, "Hello."): -2.0 for q in others}}
    judge = TotalJudge(totals)
    assert sc.on_topic_rule(judge, QUESTIONS, 0.99)("Hi", "Hello.")  # lift 1.0
    assert not sc.on_topic_rule(judge, QUESTIONS, 1.0)("Hi", "Hello.")
    assert not sc.on_topic_rule(judge, QUESTIONS, -9.0)("Hi", "...")


def _real_totals(replies):
    """-1 for each real pair (its own question), -3 for any other question."""
    return {(q, r): -1.0 for q, r in zip(QUESTIONS, replies, strict=True)}


def _calibrate(monkeypatch, tmp_path, judge, replies, fit=2.5, yardstick="toy", said=None):
    pairs = [([("user", q)], r) for q, r in zip(QUESTIONS, replies, strict=True)]
    monkeypatch.setattr(sc, "fit_pairs", lambda root, tok: pairs)
    record = said if said is not None else []
    return sc.on_topic_calibration(judge, tmp_path, yardstick, tmp_path, record.append, fit)


def test_the_on_topic_calibration_states_how_well_it_separates_and_is_cached(tmp_path, monkeypatch):
    # With 64 pairs the other questions are the even ones (the prompt replaced by the next).
    # Real pairs: lift -1 - (-3) = 2. Swapped (q_i, r_{i+32}): q_{i+32} is among the others for
    # an even i (lift -3 - (-1 - 31 * 3) / 32 = -0.0625), not for an odd one (lift 0). So 0.0
    # separates them all. Hard (q_i, r_{i+1}): -0.0625 each, except (q_10, r_11), which the
    # judge finds as likely as a real pair: -1 - (-1 - 31 * 3) / 32 = 1.9375, above.
    totals = {**_real_totals(REPLIES), ("question 10", "reply 11"): -1.0}
    nlls = {("question 3", "reply 3"): 3.0}  # one real reply does not fit (2.0 elsewhere)
    judge = TotalJudge(totals, default=-3.0, nlls=nlls)
    said = []
    got = _calibrate(monkeypatch, tmp_path, judge, REPLIES, fit=2.5, said=said)
    assert got["threshold"] == pytest.approx(0.0)
    assert got["balanced_accuracy"] == 1.0
    assert got["positives_above"] == 1.0 and got["negatives_at_or_below"] == 1.0
    assert got["hard_negatives_at_or_below"] == pytest.approx(63 / 64)
    assert got["real_fits_and_on_topic"] == pytest.approx(63 / 64)
    assert got["pairs"] == 64 and got["questions"] == sc.other_questions(QUESTIONS, "")
    (line,) = said
    assert "balanced accuracy 100%" in line and "hard" in line and "98%" in line
    calls = len(judge.calls)
    again = _calibrate(monkeypatch, tmp_path, judge, REPLIES, fit=2.5)
    assert again == got and len(judge.calls) == calls  # from its cache
    other_fit = _calibrate(monkeypatch, tmp_path, judge, REPLIES, fit=1.5)
    assert len(judge.calls) > calls and other_fit["real_fits_and_on_topic"] == 0.0
    _calibrate(monkeypatch, tmp_path, judge, REPLIES, fit=1.5, yardstick="another")
    assert len(judge.calls) > calls + 1


def test_the_calibration_skips_a_swapped_reply_equal_to_the_real_one(tmp_path, monkeypatch):
    # r_0 = r_32 and r_5 = r_6: under its own question each would score like a real reply, so
    # (q_0, r_32), (q_32, r_0) and the hard (q_5, r_6) are left out, and every other swapped or
    # hard reply stays at or below 0.0.
    replies = list(REPLIES)
    replies[0] = replies[32] = "same"
    replies[5] = replies[6] = "dup"
    judge = TotalJudge(_real_totals(replies), default=-3.0)
    got = _calibrate(monkeypatch, tmp_path, judge, replies)
    assert got["threshold"] == pytest.approx(0.0)
    assert got["balanced_accuracy"] == 1.0 and got["negatives_at_or_below"] == 1.0
    assert got["hard_negatives_at_or_below"] == 1.0


# -- the card ---------------------------------------------------------------------------------


def test_the_card_has_every_recipe_model_and_measure(scored, tiny_data_root):
    _, card, said, _, _ = scored
    assert card["recipes"] == {name: [s.name for s in v] for name, v in sc.RECIPES.items()}
    assert list(card["models"]) == [
        "first-s1", "first-s2", "small", "lineage-small", "lineage-medium", "chatty", "large",
    ]  # fmt: skip
    for name, model in card["models"].items():
        measures = model["measures"]
        assert set(measures) == set(sc.MEASURES), name
        for key in sc.MEASURES:
            assert isinstance(measures[key], float) and math.isfinite(measures[key]), (name, key)
        for key in sc.RATES:
            assert 0.0 <= measures[key] <= 1.0
        assert model["status"] == "completed" and model["reused"] is False
        assert set(model["bench"]) >= {"language", "knowledge"}
        assert "creativity" not in model["bench"]  # no judge was given
    assert card["models"]["lineage-medium"]["shape"] == [3, 96, 64]  # the toy stand-in
    assert card["models"]["lineage-medium"]["parent"] == "lineage-small"
    assert all(model["yardstick_span"] == 64 for model in card["models"].values())
    assert card["measured_with"] == {
        "yardstick": card["yardstick_identity"],
        "instruments": {
            "label": "test",
            "judge": None,
            "novelty": None,
            "fit_threshold": 3.5,
            "fit_pairs": 5,
            "on_topic": TOPIC,
        },
        "yardstick_rows": sc.YARDSTICK_SEQS,
        "scale": {"name": "test", **QUICK.probes()},
    }
    talk = card["models"]["small"]["examples"]["small_talk"]
    assert len(talk) == len(chatprobes.SMALL_TALK) * QUICK.small_talk_samples
    keys = ("prompt", "sample", "reply", "well_formed", "fits", "on_topic")
    assert {tuple(t) for t in talk} == {keys}
    good = sum(t["fits"] and t["on_topic"] for t in talk) / len(talk)
    assert card["models"]["small"]["measures"]["good_reply"] == pytest.approx(good)
    on_topic = sum(t["on_topic"] for t in talk) / len(talk)
    assert card["models"]["small"]["measures"]["on_topic"] == pytest.approx(on_topic)
    fit_share = sum(t["fits"] for t in talk) / len(talk)
    assert card["models"]["small"]["measures"]["fits"] == pytest.approx(fit_share)
    assert card["engine"] == sc.ENGINE_VERSION and card["scale"] == "test"
    assert card["data_root"] == card["yardstick_root"] == str(tiny_data_root)
    assert any("training" in line for line in said)


def test_a_second_call_reuses_every_run_and_measurement(scored):
    out, card, _, again, said = scored
    assert not [line for line in said if "training" in line or "measuring" in line]
    for name in card["models"]:
        assert any(line.startswith(f"  {name}:") and "reusing" in line for line in said), name
        assert again["models"][name]["measures"] == card["models"][name]["measures"]
        assert again["models"][name]["reused"] is True
    cached = json.loads((out / "measured" / "first-s1.json").read_text("utf-8"))
    assert cached["key"]["yardstick_span"] == 64 and cached["key"]["yardstick_rows"] == 128
    # the parent of the lineage is trained once, under its own name
    assert sorted(p.name for p in (out / "runs").iterdir()) == sorted(card["models"])


def test_timing_is_the_first_runs_own_and_says_when_it_was_reused(scored):
    _, card, _, again, _ = scored
    assert card["timing"]["model"] == "first-s1" and card["timing"]["reused"] is False
    assert card["timing"]["seconds"] > 0
    assert again["timing"] == {**card["timing"], "reused": True}


def test_a_changed_yardstick_is_measured_again_on_the_same_runs(scored, tiny_data_root):
    out, *_ = scored
    said = []
    _card(tiny_data_root, out, said, recipes=["small"], label="another judge")
    assert any(line.startswith("  small:") and "reusing" in line for line in said)  # the run
    assert "  small: measuring" in said


def test_no_timing_unless_asked(scored, tiny_data_root):
    out, *_ = scored
    assert _card(tiny_data_root, out, [], recipes=["small"])["timing"] is None


def test_a_missing_yardstick_is_an_error_before_any_training(tiny_data_root, tmp_path):
    out = tmp_path / "card"
    with pytest.raises(sc.ScorecardSetupError, match="build-judge"):
        sc.run_scorecard(
            tiny_data_root,
            out,
            yardstick_root=tiny_data_root,  # it holds no judge and no novelty index
            recipes=["first"],
            device=CPU,
            on_status=lambda line: None,
            scale=QUICK,
        )
    with pytest.raises(sc.ScorecardSetupError, match="yardstick"):
        sc.run_scorecard(
            tiny_data_root,
            out,
            yardstick_root=tmp_path / "nowhere",
            recipes=["first"],
            device=CPU,
            on_status=lambda line: None,
            scale=QUICK,
            instruments=_instruments(),
        )
    assert not out.exists()


def test_an_unknown_recipe_is_an_error_before_any_training(tiny_data_root, tmp_path):
    with pytest.raises(sc.ScorecardSetupError, match="medium"):
        _card(tiny_data_root, tmp_path / "card", [], recipes=["first", "medium"])
    assert not (tmp_path / "card").exists()


def test_the_card_writes_its_reports(scored, tmp_path):
    _, card, _, _, _ = scored
    rep.write_reports(card, tmp_path)
    assert json.loads((tmp_path / "scorecard.json").read_text("utf-8")) == card
    assert "| first-s1 |" in (tmp_path / "scorecard.md").read_text("utf-8")


def test_no_held_out_conversations_is_a_setup_error_naming_the_root(tiny_data_root, monkeypatch):
    monkeypatch.setattr(sc, "heldout_docs", lambda corpus: np.zeros(0, np.int64))
    with pytest.raises(sc.ScorecardSetupError, match="held-out conversations") as caught:
        sc.yardstick_loss(Transformer(QUICK_SHAPE, 512), tiny_data_root)
    assert str(tiny_data_root) in str(caught.value)


def test_training_data_that_cannot_be_read_is_a_setup_error(tiny_data_root, tmp_path, monkeypatch):
    def refuse(self, cfg, device, parent=None):
        raise ValueError("the corpus offsets are cut short")

    monkeypatch.setattr(sc.GateRuns, "train", refuse)
    with pytest.raises(sc.ScorecardSetupError, match="cut short") as caught:
        _card(tiny_data_root, tmp_path / "card", [], recipes=["small"])
    assert str(tiny_data_root) in str(caught.value)


def test_the_full_scale_asks_enough_to_tell_a_change_from_noise():
    assert (sc.FULL.small_talk_samples, sc.FULL.followups, sc.FULL.identity_samples) == (5, 120, 3)
    assert sc.FULL.probes()["identity_samples"] == 3
