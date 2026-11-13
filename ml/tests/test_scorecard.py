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
from airace_ml.infer.lm import TorchLM
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
    )


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
        },
        "yardstick_rows": sc.YARDSTICK_SEQS,
        "scale": {"name": "test", **QUICK.probes()},
    }
    talk = card["models"]["small"]["examples"]["small_talk"]
    assert len(talk) == len(chatprobes.SMALL_TALK) * QUICK.small_talk_samples
    assert {tuple(t) for t in talk} == {("prompt", "sample", "reply", "well_formed", "fits")}
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
