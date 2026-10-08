import copy
import json
import math
import sys
import zlib
from collections import Counter
from types import ModuleType

import pytest

from airace_ml.evals.scoring import (
    MIN_TAG_ITEMS,
    BenchReport,
    CategoryScore,
    ItemResult,
    exact_match,
    extract_answer,
    mc_choice,
    normalize,
    normalize_answer,
    pair_correct,
)
from airace_ml.evals.suite import CATEGORIES, Suite, build_suite, run_benchmarks
from airace_ml.infer.lm import ContinuationScore as CS
from airace_ml.infer.lm import Generation
from airace_ml.paths import tokenizer_path
from airace_ml.skills.checkers import CHECKERS
from airace_ml.skills.code import code_bench_items
from airace_ml.skills.facts import consistency_groups, knowledge_bench_items
from airace_ml.skills.grammar import grammar_pairs
from airace_ml.skills.instructions import instruction_bench_items
from airace_ml.skills.kb import load_kb
from airace_ml.skills.patterns import pattern_bench_items
from airace_ml.skills.reasoning import reasoning_bench_items
from airace_ml.skills.types import CheckItem, ExactItem, MCItem, PairItem, skill_rng
from airace_ml.tokenizer import Tok, encode_chat, encode_doc
from tests.fakes import ScriptedLM, answer_key_lm


# The brief's tests, verbatim.
# fmt: off
def test_normalize():
    assert normalize(0.5, 0.5) == 0 and normalize(1.0, 0.25) == 100 and normalize(0.1, 0.25) == 0
    assert normalize(0.625, 0.25) == pytest.approx(50)

def test_suite_deterministic_and_sized():
    a, b = build_suite(), build_suite()
    sizes = {k: len(v) for k, v in a.items.items()}
    assert sizes == {"language": 200, "reasoning": 200, "pattern": 150, "knowledge": 200, "coding": 150,
                     "consistency": 120, "instruction": 120}
    assert [i.id for i in a.items["coding"]] == [i.id for i in b.items["coding"]]

def test_answer_key_scores_100(tiny_tok):
    suite = build_suite(); rep = run_benchmarks(answer_key_lm(tiny_tok, suite), tiny_tok, suite=suite)
    for c in CATEGORIES:
        if c not in ("creativity", "consistency"):
            assert rep.scores[c].score == pytest.approx(100), c
    assert rep.missing == ["creativity"]

def test_constant_model_near_zero(tiny_tok):
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, max_items_per_category=40)
    assert all(s.score <= 25 for s in rep.scores.values())  # ties resolve to the first option → chance level

@pytest.mark.parametrize("reply", [lambda p: "", lambda p: "<|end|><|bos|>", lambda p: "the the the " * 30])
def test_degenerate_outputs(tiny_tok, reply):  # Review Focus 4
    rep = run_benchmarks(ScriptedLM(tiny_tok, reply=reply), tiny_tok, max_items_per_category=20)
    assert all(math.isfinite(s.score) for s in rep.scores.values()) and math.isfinite(rep.overall)

def test_report_serializable_and_tags(tiny_tok):
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, max_items_per_category=10)
    json.dumps(rep.to_dict()); assert all(0 <= v[0] <= 100 for v in rep.tag_breakdown().values())

def test_real_tiny_model_end_to_end(tiny_lm, tiny_tok):
    rep = run_benchmarks(tiny_lm, tiny_tok, max_items_per_category=5)
    assert rep.seconds > 0 and math.isfinite(rep.overall)
# fmt: on


# -- Beyond the brief: the Task 16 rulings, pinned down -------------------------------------------

SCORED_CATEGORIES = [c for c in CATEGORIES if c != "creativity"]


def test_normalize_never_divides_by_zero_or_returns_nan():
    assert normalize(1.0, 1.0) == 0 and normalize(0.5, 2.0) == 0
    assert normalize(math.nan, 0.25) == 0 and normalize(0.5, math.nan) == 0
    assert normalize(2.0, 0.0) == 100 and normalize(-1.0, 0.0) == 0


@pytest.mark.parametrize(
    ("reply", "answer"),
    [
        ("Paris.", "paris"),
        (" PARIS ", "Paris"),
        ('"Paris".', "Paris"),
        ("“Paris!”", "paris"),
        ("‘Paris’", "Paris"),
        ("'Paris'?", "Paris"),
        ("New   York", "new york"),
        ("St. Louis", "st. louis"),
        ("Paris", "Paris."),
        ("-3", "-3"),
        ("[1, 2]", "[1, 2]"),
    ],
)
def test_exact_match_accepts_harmless_variants(reply, answer):
    assert exact_match(reply, [answer], "first_line")


@pytest.mark.parametrize(
    ("reply", "answer"),
    [
        ("-3", "3"),
        ("3", "-3"),
        ("[1, 2]", "1, 2"),
        ("1, 2", "[1, 2]"),
        ("(Paris)", "Paris"),
        ("Paris-", "Paris"),
        ("Pariss", "Paris"),
        ("", "."),  # an answer that normalizes to nothing is never matched by saying nothing
        ("...", "!"),
        ("", ""),
    ],
)
def test_exact_match_keeps_content_punctuation(reply, answer):
    assert not exact_match(reply, [answer], "first_line")


def test_exact_match_any_accepted_answer():
    assert exact_match(" 5th\n", ["fifth", "5th"], "first_line")
    assert normalize_answer(" “ New \t York .” ") == "new york"


def test_extraction_first_item_vs_first_line():
    reply = " 10, 12, 14\n16"
    assert extract_answer(reply, "first_item") == " 10"
    assert extract_answer(reply, "first_line") == " 10, 12, 14"
    assert extract_answer("10\n12, 14", "first_item") == "10"
    assert exact_match(reply, ["10"], "first_item") and not exact_match(reply, ["10"], "first_line")
    assert exact_match("[1, 2]\nmore", ["[1, 2]"], "first_line")
    assert not exact_match("[1, 2]", ["[1, 2]"], "first_item")
    assert not exact_match("\n10", ["10"], "first_line")  # nothing before the first new line
    with pytest.raises(ValueError, match="extract"):
        extract_answer("10", "whole")


def test_mc_choice_by_mean_logprob_per_token_first_on_ties():
    assert mc_choice([CS(-2.0, 1), CS(-3.0, 3)]) == 1  # mean -1 beats -2 though the sum is lower
    assert mc_choice([CS(-1.0, 1), CS(-2.0, 2), CS(-3.0, 3)]) == 0  # all tie at -1 per token
    assert mc_choice([CS(-5.0, 1), CS(-1.0, 1), CS(-1.0, 1)]) == 1
    assert mc_choice([CS(0.0, 0), CS(-9.0, 1)]) == 1  # an empty continuation never wins
    assert mc_choice([CS(math.nan, 2), CS(-9.0, 1)]) == 1  # nor does a NaN
    assert mc_choice([CS(math.nan, 1), CS(math.nan, 1)]) == 0


def test_pair_compares_total_logprob_strictly():
    assert pair_correct(CS(-3.0, 3), CS(-4.0, 1))
    assert not pair_correct(CS(-5.0, 5), CS(-3.0, 1))  # better per token, worse in total
    assert not pair_correct(CS(-2.0, 2), CS(-2.0, 2))  # a tie is not a win
    assert not pair_correct(CS(math.nan, 2), CS(-2.0, 2))


def test_mc_item_through_the_runner(tiny_tok):
    short, long = " cat", " elephant seal"
    assert len(tiny_tok.encode(long)) > 2 * len(tiny_tok.encode(short))  # so sum and mean disagree
    means = {short: -2.0, long: -1.0}
    lm = ScriptedLM(tiny_tok, score=lambda ctx, cont: means[tiny_tok.decode(cont)] * len(cont))
    item = MCItem(
        "r-0",
        "reasoning",
        "Question: Which is bigger?\nAnswer:",
        ["cat", "elephant seal"],
        1,
        ("fam:x",),
    )
    rep = run_benchmarks(
        lm, tiny_tok, suite=Suite("t", {"reasoning": [item]}), categories=["reasoning"]
    )
    assert rep.scores["reasoning"] == CategoryScore(100.0, 1.0, 1)
    assert rep.items == [ItemResult("r-0", "reasoning", 1.0, ("fam:x",), "elephant seal")]


def test_pair_item_scored_after_bos(tiny_tok):
    seen = []

    def score(ctx, cont):
        seen.append((ctx, cont))
        return {"They see.": -1.0, "They sees.": -2.0}[tiny_tok.decode(cont)]

    items = [PairItem("l-0", "language", "They see.", "They sees.", ("fam:agreement",))]
    suite = Suite("t", {"language": items})
    rep = run_benchmarks(
        ScriptedLM(tiny_tok, score=score), tiny_tok, suite=suite, categories=["language"]
    )
    bos = [tiny_tok.bos_id]
    assert seen == [(bos, tiny_tok.encode("They see.")), (bos, tiny_tok.encode("They sees."))]
    assert rep.scores["language"].score == 100 and rep.items[0].output is None


def test_chat_options_have_no_leading_space(tiny_tok):
    question = "Is the sky green?"
    items = [
        MCItem("c-0", "reasoning", question, ["yes", "no"], 1, (), chat=True),
        MCItem("p-0", "reasoning", question, ["yes", "no"], 1, ()),
    ]
    right = {  # only these exact encodings score well; anything else ties and picks "yes"
        (tuple(encode_chat(tiny_tok, [("user", question)], True)), tuple(tiny_tok.encode("no"))),
        (tuple(encode_doc(tiny_tok, question)), tuple(tiny_tok.encode(" no"))),
    }
    lm = ScriptedLM(
        tiny_tok, score=lambda ctx, cont: 0.0 if (tuple(ctx), tuple(cont)) in right else -10.0
    )
    rep = run_benchmarks(
        lm, tiny_tok, suite=Suite("t", {"reasoning": items}), categories=["reasoning"]
    )
    assert [(r.score, r.output) for r in rep.items] == [(1.0, "no"), (1.0, "no")]


class RecordingLM(ScriptedLM):
    """Records every model call: ("score", n) or ("generate", n, keyword arguments)."""

    def __init__(self, tok, **kwargs):
        super().__init__(tok, **kwargs)
        self.calls = []
        self.prompts = []

    def score_continuations(self, contexts, continuations):
        self.calls.append(("score", len(contexts)))
        return super().score_continuations(contexts, continuations)

    def generate(self, prompts, **kwargs):
        self.calls.append(("generate", len(prompts), kwargs))
        self.prompts.append(prompts)
        return super().generate(prompts, **kwargs)


def test_generation_prompts_are_greedy_seeded_and_use_default_stops(tiny_tok):
    prompt = "Question: What is the capital of France?\nAnswer:"
    items = [
        ExactItem("e-0", "knowledge", prompt, ["Paris"], (), chat=True),
        ExactItem("e-1", "knowledge", prompt, ["Paris"], ()),
    ]
    lm = RecordingLM(tiny_tok, reply=lambda ids: " Paris.\nQuestion:")
    rep = run_benchmarks(
        lm, tiny_tok, suite=Suite("t", {"knowledge": items}), categories=["knowledge"], seed=7
    )
    assert lm.prompts == [
        [encode_chat(tiny_tok, [("user", prompt)], True), encode_doc(tiny_tok, prompt)]
    ]
    ((_, _, kwargs),) = lm.calls
    assert kwargs["max_new_tokens"] == 8 and kwargs["temperature"] == 0 and kwargs["seed"] == 7
    assert kwargs.get("stop_ids") is None  # (end, bos)
    assert [(r.score, r.output) for r in rep.items] == [(1.0, " Paris.\nQuestion:")] * 2


def test_one_batched_call_per_category(tiny_tok):
    suite = build_suite()
    for category in SCORED_CATEGORIES:
        lm = RecordingLM(tiny_tok)
        run_benchmarks(lm, tiny_tok, suite=suite, categories=[category])
        items = suite.items[category]
        n_scored = sum(
            2 if isinstance(i, PairItem) else len(i.options)
            for i in items
            if isinstance(i, (PairItem, MCItem))
        )
        lengths = Counter(i.max_new_tokens for i in items if isinstance(i, (ExactItem, CheckItem)))
        expected = ([("score", n_scored)] if n_scored else []) + [
            ("generate", n, m) for m, n in lengths.items()
        ]
        got = [
            call if call[0] == "score" else call[:2] + (call[2]["max_new_tokens"],)
            for call in lm.calls
        ]
        assert got == expected, category
    assert len(lengths) == 1  # instruction: one reply length; coding has two (outputs, functions)


def test_checkers_get_the_full_reply_and_untouched_check_args(tiny_tok, monkeypatch):
    suite = build_suite()
    checked = [
        i for c in ("coding", "instruction") for i in suite.items[c] if isinstance(i, CheckItem)
    ]
    before = [copy.deepcopy(i.check_args) for i in checked]
    seen = []
    for name in {i.check for i in checked}:

        def spy(reply, args, real=CHECKERS[name]):
            seen.append((reply, args))
            return real(reply, args)

        monkeypatch.setitem(CHECKERS, name, spy)
    rep = run_benchmarks(
        answer_key_lm(tiny_tok, suite), tiny_tok, suite=suite, categories=["coding", "instruction"]
    )
    assert len(seen) == len(checked) == 170
    for item, original, (reply, args) in zip(checked, before, seen, strict=True):
        assert reply == item.reference and args is item.check_args and args == original
    assert {"instruction", "not", "min_words"} <= {key for _, args in seen for key in args}
    assert all(s.score == 100 for s in rep.scores.values())


def _group(gid, option_lists, answer, tag_lists):
    return [
        MCItem(
            f"{gid}-{k}",
            "consistency",
            f"Question: {gid}, wording {k}?\nAnswer:",
            opts,
            opts.index(answer),
            tags,
            group=gid,
        )
        for k, (opts, tags) in enumerate(zip(option_lists, tag_lists, strict=True))
    ]


def test_consistency_group_is_right_only_when_every_paraphrase_is(tiny_tok):
    a = _group(
        "a",
        [
            ["Paris", "Rome", "Oslo", "Bern"],
            ["Rome", "Paris", "Bern", "Oslo"],
            ["Oslo", "Bern", "Rome", "Paris"],
        ],
        "Paris",
        [("rel:capital_of",)] * 3,
    )
    b = _group(
        "b", [["yes", "no"], ["no", "yes"], ["yes", "no"]], "yes", [("x",), ("x", "y"), ("z",)]
    )
    c = _group(
        "c",
        [["4", "2", "6", "8"], ["8", "6", "2", "4"], ["2", "4", "8", "6"]],
        "4",
        [("legs",)] * 3,
    )
    d = _group("d", [["no", "yes"]] * 3, "no", [("y",)] * 3)
    chosen = {  # what the model answers to each paraphrase of a group
        "a": ["Paris", "Paris", "Paris"],  # every paraphrase right: 1
        "b": ["yes", "yes", "no"],  # right on 2 of 3: 0
        "c": ["6", "6", "6"],  # the same wrong answer every time: 0
        "d": ["no", "no", "no"],  # every paraphrase right: 1
    }
    members = a + b + c + d
    by_prompt = {
        tuple(encode_doc(tiny_tok, i.prompt)): chosen[i.group][k % 3] for k, i in enumerate(members)
    }

    def score(ctx, cont):
        return 0.0 if tiny_tok.decode(cont) == " " + by_prompt[tuple(ctx)] else -10.0

    suite = Suite("t", {"consistency": members})
    rep = run_benchmarks(
        ScriptedLM(tiny_tok, score=score), tiny_tok, suite=suite, categories=["consistency"]
    )
    s = rep.scores["consistency"]
    chance = (1 / 4 + 1 / 2 + 1 / 4 + 1 / 2) / 4  # mean over groups of 1/len(options)
    assert s.n == 4 and s.raw == 0.5
    assert s.score == pytest.approx(normalize(0.5, chance)) and s.score == pytest.approx(20)
    assert [(r.item_id, r.category, r.score, r.tags, r.output) for r in rep.items] == [
        ("a", "consistency", 1.0, ("rel:capital_of",), None),
        ("b", "consistency", 0.0, ("x", "y", "z"), None),
        ("c", "consistency", 0.0, ("legs",), None),
        ("d", "consistency", 1.0, ("y",), None),
    ]
    # a constant model takes the first option: right only where every paraphrase lists it first
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, suite=suite, categories=["consistency"])
    assert [r.score for r in rep.items] == [0.0, 0.0, 0.0, 1.0]


def test_consistency_rejects_ungrouped_items(tiny_tok):
    loose = MCItem("k-0", "consistency", "Question?", ["a", "b"], 0, ())
    with pytest.raises(ValueError, match="group"):
        run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, suite=Suite("t", {"consistency": [loose]}))


def test_limits_take_the_first_items_and_whole_consistency_groups(tiny_tok):
    suite = build_suite()
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, suite=suite, max_items_per_category=7)
    for category in SCORED_CATEGORIES:
        if category != "consistency":
            assert [r.item_id for r in rep.items if r.category == category] == [
                i.id for i in suite.items[category][:7]
            ]
            assert rep.scores[category].n == 7
    groups = [r.item_id for r in rep.items if r.category == "consistency"]
    assert groups == ["consistency-000", "consistency-001"]  # 7 items: two whole groups of 3
    for limit, n_groups in [
        (1, 1),
        (2, 1),
        (3, 1),
        (5, 1),
        (6, 2),
        (40, 13),
        (119, 39),
        (None, 40),
    ]:
        rep = run_benchmarks(
            ScriptedLM(tiny_tok),
            tiny_tok,
            suite=suite,
            categories=["consistency"],
            max_items_per_category=limit,
        )
        assert rep.scores["consistency"].n == n_groups, limit


def test_empty_categories_are_missing_not_nan(tiny_tok):
    rep = run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, max_items_per_category=0)
    assert (
        rep.scores == {}
        and rep.items == []
        and rep.overall == 0.0
        and rep.missing == list(CATEGORIES)
    )
    suite = Suite("t", {"language": []})
    rep = run_benchmarks(
        ScriptedLM(tiny_tok), tiny_tok, suite=suite, categories=["language", "pattern"]
    )
    assert rep.scores == {} and rep.missing == ["language", "pattern"] and rep.overall == 0.0
    json.dumps(rep.to_dict(), allow_nan=False)


def test_rejects_unknown_categories_and_negative_limits(tiny_tok):
    lm = ScriptedLM(tiny_tok)
    with pytest.raises(ValueError, match="unknown"):
        run_benchmarks(lm, tiny_tok, categories=["speed"])
    with pytest.raises(ValueError, match="unknown"):
        run_benchmarks(lm, tiny_tok, categories="language")  # a string is not a list of categories
    with pytest.raises(ValueError, match="max_items_per_category"):
        run_benchmarks(lm, tiny_tok, max_items_per_category=-1)


def test_overall_is_the_mean_of_measured_categories(tiny_tok):
    suite = build_suite()
    lm = answer_key_lm(tiny_tok, suite)
    for item in suite.items["pattern"]:
        item.answers = ["never said"]
    rep = run_benchmarks(
        lm, tiny_tok, suite=suite, categories=["language", "pattern", "creativity"]
    )
    assert rep.scores["language"].score == 100 and rep.scores["pattern"].score == 0
    assert (
        rep.overall == 50
        and rep.missing == ["creativity"]
        and list(rep.scores) == ["language", "pattern"]
    )


class SpecialTokensLM(ScriptedLM):
    """Replies with nothing but control tokens (the model can emit them; decoding drops them)."""

    def generate(self, prompts, **kwargs):
        ids = [3, 4, 5, 0, 7, 3]  # user, ai, sep, pad, r0, user
        return [Generation(list(ids), [0.5] * len(ids), [0.5] * len(ids), False) for _ in prompts]


@pytest.mark.parametrize(
    "lm_factory",
    [
        lambda tok: ScriptedLM(tok, reply=lambda p: ""),
        lambda tok: ScriptedLM(tok, reply=lambda p: "<|end|><|bos|>"),
        lambda tok: ScriptedLM(tok, reply=lambda p: "the the the " * 30),
        lambda tok: ScriptedLM(tok, reply=lambda p: "\n\n\n"),
        lambda tok: SpecialTokensLM(tok),
    ],
)
def test_degenerate_replies_score_zero_on_the_whole_suite(tiny_tok, lm_factory):  # Review Focus 4
    rep = run_benchmarks(
        lm_factory(tiny_tok), tiny_tok, categories=["pattern", "coding", "instruction"]
    )
    assert all(s.score == 0 and s.raw == 0 for s in rep.scores.values())
    json.dumps(rep.to_dict(), allow_nan=False)


@pytest.mark.parametrize("value", [math.nan, -math.inf, math.inf])
def test_non_finite_logprobs_give_finite_scores(tiny_tok, value):  # a diverged model
    rep = run_benchmarks(
        ScriptedLM(tiny_tok, score=lambda ctx, cont: value), tiny_tok, max_items_per_category=20
    )
    assert all(math.isfinite(s.score) and math.isfinite(s.raw) for s in rep.scores.values())
    assert rep.scores["language"].score == 0  # equal texts never beat each other
    json.dumps(rep.to_dict(), allow_nan=False)


def test_to_dict_round_trips_through_json(tiny_tok):
    text = 'Paris \U0001f642 é 中 "q" \\ <|end|>'
    rep = run_benchmarks(
        ScriptedLM(tiny_tok, reply=lambda p: text), tiny_tok, max_items_per_category=10
    )
    d = rep.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d  # plain types only: no tuples, no NaN
    assert (
        set(d) == {"suite", "scores", "overall", "missing", "items", "seconds"}
        and d["suite"] == "bench-v1"
    )
    assert d["scores"]["language"] == {"score": 0.0, "raw": 0.0, "n": 10}
    assert d["missing"] == ["creativity"] and len(d["items"]) == len(rep.items)
    assert {i["output"] for i in d["items"] if i["category"] == "pattern"} == {text}


def test_tag_breakdown_needs_min_items_within_a_category():
    tags = ("topic:animals", "rel:x")
    items = (
        [ItemResult(f"k{i}", "knowledge", float(i % 2), tags) for i in range(MIN_TAG_ITEMS)]
        + [ItemResult(f"c{i}", "consistency", 1.0, tags) for i in range(MIN_TAG_ITEMS - 1)]
        + [ItemResult(f"p{i}", "pattern", 0.0, ("fam:double", "fam:double")) for i in range(5)]
        + [ItemResult(f"o{i}", "coding", 1.0, ("fam:double",)) for i in range(5)]
    )
    rep = BenchReport("bench-v1", {}, 0.0, [], items, 1.0)
    assert MIN_TAG_ITEMS == 10
    assert rep.tag_breakdown() == {
        "knowledge/topic:animals": (50.0, 10),
        "knowledge/rel:x": (50.0, 10),
    }


def test_small_tags_never_reach_the_breakdown(tiny_tok):
    suite = build_suite()
    breakdown = run_benchmarks(
        answer_key_lm(tiny_tok, suite), tiny_tok, suite=suite
    ).tag_breakdown()
    assert not any(key.endswith("/fam:double") for key in breakdown)  # 7 pattern + 5 coding items
    assert breakdown["coding/fmt:function"] == (100.0, 50)
    assert breakdown["consistency/topic:animals"] == (100.0, 11)  # 11 groups
    assert all(count >= MIN_TAG_ITEMS and score == 100 for score, count in breakdown.values())


def test_build_suite_hands_out_copies():
    a = build_suite()
    original = copy.deepcopy(a.items["instruction"][0].check_args)
    a.items["coding"].clear()
    a.items["language"][0].good = "changed"
    a.items["instruction"][0].check_args["n"] = 99
    b = build_suite()
    assert b.version == "bench-v1" and len(b.items["coding"]) == 150
    assert (
        b.items["language"][0].good != "changed"
        and b.items["instruction"][0].check_args == original
    )


def test_suite_items_come_from_the_bench_streams():
    suite, kb = build_suite(), load_kb()
    assert suite.items == {
        "language": grammar_pairs(skill_rng("language", "bench"), n=200),
        "reasoning": reasoning_bench_items(skill_rng("reasoning", "bench"), n=200),
        "pattern": pattern_bench_items(skill_rng("pattern", "bench"), n=150),
        "knowledge": knowledge_bench_items(kb, skill_rng("knowledge", "bench")),
        "coding": code_bench_items(skill_rng("coding", "bench"), n_output=100, n_func=50),
        "consistency": consistency_groups(kb, skill_rng("consistency", "bench"), n_groups=40),
        "instruction": instruction_bench_items(kb, skill_rng("instruction", "bench"), n=120),
    }


def test_suite_items_are_scorable():
    suite = build_suite()
    ids = [i.id for items in suite.items.values() for i in items]
    assert len(set(ids)) == len(ids)
    for category, items in suite.items.items():
        for item in items:
            assert item.category == category
            if isinstance(item, MCItem):
                assert len(set(item.options)) == len(item.options)
                assert 0 <= item.answer_index < len(item.options)
            if isinstance(item, ExactItem):  # every accepted answer survives its own extraction
                for answer in item.answers:
                    assert (
                        normalize_answer(extract_answer(answer, item.extract))
                        == normalize_answer(answer)
                        != ""
                    )
    groups = {}
    for item in suite.items["consistency"]:
        groups.setdefault(item.group, []).append(sorted(item.options))
    assert len(groups) == 40 and all(len(g) == 3 and g[0] == g[1] == g[2] for g in groups.values())


def test_creativity_hook(tiny_tok, monkeypatch):
    calls = []

    def score_creativity(lm, tok, judge, novelty, *, seed=0):
        calls.append((lm, tok, judge, novelty, seed))
        return CategoryScore(40.0, 0.4, 24), [
            ItemResult("story-00", "creativity", 0.4, ("fmt:story",))
        ]

    module = ModuleType("airace_ml.evals.creativity")
    module.score_creativity = score_creativity
    monkeypatch.setitem(sys.modules, "airace_ml.evals.creativity", module)
    lm = ScriptedLM(tiny_tok)
    rep = run_benchmarks(
        lm,
        tiny_tok,
        categories=["language", "creativity"],
        judge="J",
        novelty="N",
        seed=5,
        max_items_per_category=4,
    )
    assert calls == [(lm, tiny_tok, "J", "N", 5)] and rep.missing == []
    assert rep.scores["creativity"] == CategoryScore(40.0, 0.4, 24)
    assert rep.overall == pytest.approx(20)  # language scores 0 for the constant model
    assert [r.item_id for r in rep.items if r.category == "creativity"] == ["story-00"]
    for judge, novelty in [("J", None), (None, "N")]:
        rep = run_benchmarks(lm, tiny_tok, categories=["creativity"], judge=judge, novelty=novelty)
        assert rep.missing == ["creativity"] and rep.scores == {} and rep.overall == 0.0
    assert len(calls) == 1


def test_real_tiny_model_is_reproducible(tiny_lm, tiny_tok):
    a = run_benchmarks(tiny_lm, tiny_tok, max_items_per_category=6, seed=3)
    b = run_benchmarks(tiny_lm, tiny_tok, max_items_per_category=6, seed=3)
    assert a.scores == b.scores and a.missing == b.missing == ["creativity"]
    assert [(r.item_id, r.score, r.output) for r in a.items] == [
        (r.item_id, r.score, r.output) for r in b.items
    ]


def test_rejects_short_score_lists_and_unknown_items(tiny_tok):
    class ShortLM(ScriptedLM):
        def score_continuations(self, contexts, continuations):
            return super().score_continuations(contexts, continuations)[:-1]

    with pytest.raises(ValueError, match="scores"):
        run_benchmarks(ShortLM(tiny_tok), tiny_tok, categories=["language"])
    with pytest.raises(TypeError, match="benchmark item"):
        run_benchmarks(ScriptedLM(tiny_tok), tiny_tok, suite=Suite("t", {"language": ["text"]}))


# -- Pre-review rulings: consistency as robust knowledge (A, revised) and reply budgets (C) --------


def _text_prior(text: str) -> float:
    """A fixed log-prob per token for each option text (a different one for nearly every text)."""
    return -1.0 - zlib.crc32(text.encode("utf-8")) % 997 / 100


def test_consistency_gives_a_question_blind_option_prior_chance_level(tiny_tok):
    def score(ctx, cont):  # what the model says depends on the option, never on the context
        return _text_prior(tiny_tok.decode(cont)) * len(cont)

    lm, suite = ScriptedLM(tiny_tok, score=score), build_suite()
    choices: dict[str, set[str]] = {}
    for item in suite.items["consistency"]:
        conts = [tiny_tok.encode(" " + option) for option in item.options]
        scores = lm.score_continuations([encode_doc(tiny_tok, item.prompt)] * len(conts), conts)
        choices.setdefault(item.group, set()).add(item.options[mc_choice(scores)])
    assert all(len(texts) == 1 for texts in choices.values())  # perfectly steady...
    rep = run_benchmarks(lm, tiny_tok, suite=suite, categories=["consistency"])
    assert rep.scores["consistency"].score <= 25  # ...yet right only where its favourite is


@pytest.mark.parametrize(
    ("says", "expected"),
    [("the right answer", 100.0), ("right but once wrong", 0.0), ("the same wrong answer", 0.0)],
)
def test_consistency_needs_every_paraphrase_right(tiny_tok, says, expected):
    suite = build_suite()
    target = {}  # paraphrase context -> the option the question makes the model say
    for item in suite.items["consistency"]:
        right = item.options[item.answer_index]
        wrong = min(option for option in item.options if option != right)  # same in every wording
        if says == "the right answer" or (says == "right but once wrong" and item.id[-2:] != "-2"):
            target[tuple(encode_doc(tiny_tok, item.prompt))] = right
        else:
            target[tuple(encode_doc(tiny_tok, item.prompt))] = wrong

    def score(ctx, cont):  # a strong text prior everywhere, outweighed by what the question says
        text = tiny_tok.decode(cont)
        lift = 20.0 if text == " " + target[tuple(ctx)] else 0.0
        return (_text_prior(text) + lift) * len(cont)

    rep = run_benchmarks(
        ScriptedLM(tiny_tok, score=score), tiny_tok, suite=suite, categories=["consistency"]
    )
    assert rep.scores["consistency"] == CategoryScore(expected, expected / 100, 40)


def test_an_untrained_model_is_not_consistent(tiny_lm, tiny_tok):
    # Agreement-only rules gave untrained models 62-91; this rule gives them about 0 (a group is
    # right by luck about 1 time in 4, which is chance), so 50 leaves a wide margin.
    rep = run_benchmarks(tiny_lm, tiny_tok, categories=["consistency"])
    assert rep.scores["consistency"].n == 40 and rep.scores["consistency"].score <= 50


def _reply_budget_overruns(tok: Tok) -> list[tuple[str, str, int, int]]:
    """(item id, text, tokens, max_new_tokens) for every answer or reference that cannot fit."""
    overruns = []
    for items in build_suite().items.values():
        for item in items:
            if isinstance(item, ExactItem):  # a plain reply usually starts with a space
                texts = [t for a in item.answers for t in ([a] if item.chat else [a, " " + a])]
            elif isinstance(item, CheckItem):
                texts = [item.reference]
            else:
                continue
            for text in texts:
                if (n := len(tok.encode(text))) > item.max_new_tokens:
                    overruns.append((item.id, text, n, item.max_new_tokens))
    return overruns


def test_reply_budget_check_finds_overruns(tiny_tok):  # the check itself, on the test tokenizer
    overruns = {item_id for item_id, *_ in _reply_budget_overruns(tiny_tok)}
    assert {"knowledge-0168", "instruction-0004", "instruction-0053"} <= overruns


@pytest.mark.skipif(not tokenizer_path().exists(), reason="needs the real tok-v1 tokenizer")
def test_every_answer_fits_its_reply_budget_with_the_real_tokenizer():
    assert _reply_budget_overruns(Tok.load(tokenizer_path())) == []
