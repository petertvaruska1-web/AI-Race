from airace_ml.discoveries import eval_signals, find_discoveries, tag_words
from airace_ml.evals.creativity import STORY_PROMPTS
from airace_ml.evals.scoring import BenchReport, CategoryScore, ItemResult
from airace_ml.evals.suite import build_suite
from airace_ml.personality.fingerprint import TRAITS, Fingerprint, load_probes
from airace_ml.skills.types import CheckItem, ExactItem, MCItem

SUITE = build_suite()
THRESHOLDS = {"minPoints": 25, "minZ": 2, "minPeers": 3, "perEval": 3}


def _animals(n: int = 10) -> list[MCItem]:
    return [i for i in SUITE.items["knowledge"] if "topic:animals" in i.tags][:n]


def _report(results: list[ItemResult]) -> BenchReport:
    return BenchReport("bench-v1", {"knowledge": CategoryScore(10, 0.1, 10)}, 10, [], results, 1.0)


def _results(items, scores, outputs=None) -> list[ItemResult]:
    outputs = outputs or [f"reply {k}" for k in range(len(items))]
    return [
        ItemResult(i.id, i.category, s, tuple(i.tags), o)
        for i, s, o in zip(items, scores, outputs, strict=True)
    ]


def _fp(**traits: float) -> Fingerprint:
    probe = load_probes()[0]
    sample = {"probe": probe.id, "kind": probe.kind, "sample": 0, "text": "Hello there, friend."}
    return Fingerprint({t: traits.get(t, 0.5) for t in TRAITS}, [sample])


def _prev(value: float, label: str = "Spark 1", **traits: float) -> dict:
    return {
        "label": label,
        "breakdown": {"knowledge/topic:animals": [value, 10]},
        "traits": {t: traits.get(t, 0.5) for t in TRAITS},
    }


# -- signals ---------------------------------------------------------------------------------


class FakeJudge:
    """Passes only replies that say "friend"."""


def test_signals_count_free_answer_facts_and_passing_programs():
    exact = [i for i in SUITE.items["knowledge"] if isinstance(i, ExactItem)][:3]
    mc = [i for i in SUITE.items["knowledge"] if isinstance(i, MCItem)][:2]
    checked = [i for i in SUITE.items["coding"] if isinstance(i, CheckItem)][:2]
    output = [i for i in SUITE.items["coding"] if isinstance(i, ExactItem)][:2]
    report = _report(
        _results(exact, [1, 1, 0])
        + _results(mc, [1, 1])  # multiple choice can be right by luck: not counted
        + _results(checked, [1, 0])
        + _results(output, [1, 1])  # predicting output is not a working program
    )
    signals = eval_signals(report, _fp(), judge=None)
    assert signals == {"legibleReplies": None, "correctFacts": 2, "passingPrograms": 1}


def test_legible_replies_count_well_formed_samples_under_the_judge(monkeypatch):
    import airace_ml.discoveries as d

    monkeypatch.setattr(d, "is_well_formed", lambda reply, judge: "friend" in reply)
    fp = _fp()
    fp.samples.append({**fp.samples[0], "sample": 1, "text": "no"})
    assert eval_signals(_report([]), fp, judge=FakeJudge())["legibleReplies"] == 1


# -- discoveries -----------------------------------------------------------------------------


def test_thirty_points_above_the_previous_version_is_better_and_twenty_is_nothing():
    items = _animals()
    report = _report(_results(items, [1] * 8 + [0] * 2))  # 80
    found = find_discoveries(report, _fp(), _prev(50), [], THRESHOLDS)
    assert [(f["kind"], f["value"], f["compared"]) for f in found] == [("better", 80.0, 50.0)]
    assert found[0]["title"] == "Much better at facts about animals than Spark 1"
    assert found[0]["key"] == "knowledge/topic:animals"
    assert found[0]["category"] == "knowledge" and found[0]["tag"] == "topic:animals"
    assert find_discoveries(report, _fp(), _prev(60), [], THRESHOLDS) == []
    halves = _report(_results(items, [1] * 5 + [0] * 5))  # 50
    worse = find_discoveries(halves, _fp(), _prev(100), [], THRESHOLDS)
    assert [f["kind"] for f in worse] == ["worse"]
    assert worse[0]["title"] == "Worse at facts about animals than Spark 1"


def test_evidence_is_quoted_byte_for_byte():
    items = _animals()
    hostile = '<script>alert("x")</script>  odd   spacing\n'
    outputs = [hostile, *[f"r{k}" for k in range(9)]]
    report = _report(_results(items, [1] + [0] * 9, outputs))
    better = find_discoveries(report, _fp(), _prev(-30), [], THRESHOLDS)[0]
    assert better["evidence"] == {"prompt": items[0].prompt, "reply": hostile}
    worse = find_discoveries(report, _fp(), _prev(40), [], THRESHOLDS)[0]
    assert worse["evidence"] == {"prompt": items[1].prompt, "reply": "r0"}


def test_a_strength_needs_enough_peers():
    report = _report(_results(_animals(), [1] * 10))
    peer = {"breakdown": {"knowledge/topic:animals": [20, 10]}, "traits": {}}
    assert find_discoveries(report, _fp(), None, [peer, peer], THRESHOLDS) == []
    found = find_discoveries(report, _fp(), None, [peer, peer, peer], THRESHOLDS)
    assert [(f["kind"], f["compared"]) for f in found] == [("strength", 20.0)]
    assert found[0]["title"] == "Surprisingly good at facts about animals"
    weak = find_discoveries(
        _report(_results(_animals(), [0] * 10)),
        _fp(),
        None,
        [{"breakdown": {"knowledge/topic:animals": [90, 10]}, "traits": {}}] * 3,
        THRESHOLDS,
    )
    assert [f["title"] for f in weak] == ["Struggles with facts about animals"]


def test_a_trait_that_jumps_three_peer_spreads_is_a_change_titled_with_its_word():
    peers = [{"breakdown": {}, "traits": {"verbosity": v}} for v in (0.4, 0.5, 0.6)]
    found = find_discoveries(
        _report([]), _fp(verbosity=0.8), _prev(0, verbosity=0.55), peers, THRESHOLDS
    )
    assert [(f["kind"], f["about"], f["category"], f["tag"]) for f in found] == [
        ("changed", "talkative", "personality", "verbosity")
    ]
    assert found[0]["title"] == "More talkative than Spark 1"
    probe = load_probes()[0]
    assert found[0]["evidence"] == {"prompt": probe.text, "reply": "Hello there, friend."}
    terse = find_discoveries(
        _report([]), _fp(verbosity=0.2), _prev(0, verbosity=0.55), peers, THRESHOLDS
    )
    assert [f["about"] for f in terse] == ["terse"]
    # A small move is nothing, and so is any move without enough peers to know the scale.
    assert find_discoveries(_report([]), _fp(verbosity=0.6), _prev(0), peers, THRESHOLDS) == []
    assert find_discoveries(_report([]), _fp(verbosity=0.9), _prev(0), peers[:2], THRESHOLDS) == []


def test_only_the_largest_findings_are_kept_and_each_tag_once():
    tags = ["topic:animals", "topic:places", "topic:school", "topic:science"]
    results, breakdown = [], {}
    for k, tag in enumerate(tags):
        items = [i for i in SUITE.items["knowledge"] if tag in i.tags][:10]
        results += _results(items, [1] * 10)
        breakdown[f"knowledge/{tag}"] = [k * 10, 10]  # 0, 10, 20, 30 before
    previous = {"label": "Spark 1", "breakdown": breakdown, "traits": {}}
    peers = [{"breakdown": {f"knowledge/{t}": [0, 10] for t in tags}, "traits": {}}] * 3
    found = find_discoveries(_report(results), _fp(), previous, peers, THRESHOLDS)
    assert [f["key"] for f in found] == [
        "knowledge/topic:animals",
        "knowledge/topic:places",
        "knowledge/topic:school",
    ]
    # Animals is +100 on both comparisons: the previous-version finding wins the tie.
    assert found[0]["kind"] == "better"
    assert found[1]["kind"] == "strength"  # +100 on peers beats +90 on the previous version


def test_stories_and_consistency_groups_quote_their_own_prompts():
    stories = [
        ItemResult(f"story-{k:02d}", "creativity", 0.9, ("fmt:story",), f"Once {k}")
        for k in range(10)
    ]
    found = find_discoveries(
        _report(stories),
        _fp(),
        {"label": "v1", "breakdown": {"creativity/fmt:story": [10, 10]}, "traits": {}},
        [],
        THRESHOLDS,
    )
    assert found[0]["about"] == "stories"
    assert found[0]["evidence"] == {"prompt": STORY_PROMPTS[0], "reply": "Once 0"}
    group = SUITE.items["consistency"][:3]
    groups = [
        ItemResult(f"consistency-{k:03d}", "consistency", 0.0, ("topic:animals",), "a | b | c")
        for k in range(10)
    ]
    weak = find_discoveries(
        _report(groups),
        _fp(),
        {"label": "v1", "breakdown": {"consistency/topic:animals": [90, 10]}, "traits": {}},
        [],
        THRESHOLDS,
    )
    assert weak[0]["evidence"] == {"prompt": group[0].prompt, "reply": "a | b | c"}


def test_tag_words():
    assert tag_words("knowledge/topic:space_travel") == "facts about space travel"
    assert tag_words("creativity/fmt:story") == "stories"
    assert tag_words("knowledge/rel:capital_of") == "capital cities"
    assert tag_words("reasoning/fam:count") == "counting"
    assert tag_words("coding/fam:mystery") == "coding (mystery)"
    assert tag_words("instruction/topic:animals") == "instructions about animals"


def test_nothing_is_found_without_anything_to_compare():
    report = _report(_results(_animals(), [1] * 10))
    assert find_discoveries(report, _fp(), None, [], THRESHOLDS) == []
    assert find_discoveries(_report([]), _fp(), _prev(0), [], THRESHOLDS) == []
