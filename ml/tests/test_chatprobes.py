import itertools
import re

from airace_ml.experiments import chatprobes as cp
from airace_ml.infer.lm import Generation
from airace_ml.infer.serving import SERVING_TEMPERATURE, SERVING_TOP_P
from airace_ml.persona import identity_line
from airace_ml.skills.kb import load_kb
from airace_ml.skills.probe_split import PROBE_SUBJECT_SHARE, is_probe_subject

KB = load_kb()


class FakeLM:
    """Replies looked up by the text of the last user turn; records every call."""

    def __init__(self, replies=None, default="", gen=None, fn=None):
        self.fn = fn
        self.replies = replies or {}
        self.default = default
        self.gen = gen
        self.calls = []

    def _reply(self, history, kwargs):
        self.calls.append((list(history), kwargs))
        if self.fn is not None:
            return self.fn(history)
        key = history[-1][1]
        r = self.replies.get(key, self.default)
        return r(kwargs.get("system")) if callable(r) else r

    def chat_reply(self, history, **kwargs):
        return self._reply(history, kwargs)

    def chat_generation(self, history, **kwargs):
        text = self._reply(history, kwargs)
        gen = self.gen if self.gen is not None else Generation([1] * 3, [], [], True)
        return text, gen


def _first_item(relation="capital"):
    rel = KB.relations[relation]
    return rel, KB.facts_for(relation)


def _casual_relation_names():
    return list(KB.relations)


# -- probe split ------------------------------------------------------------------------------


def test_probe_subject_share_and_determinism():
    pairs = [(f.relation, f.subject) for f in KB.facts]
    hits = [is_probe_subject(r, s) for r, s in pairs]
    assert 0.05 <= sum(hits) / len(hits) <= 0.15
    assert hits == [is_probe_subject(r, s) for r, s in pairs]
    assert PROBE_SUBJECT_SHARE == 0.1


# -- ProbeScore -------------------------------------------------------------------------------


def test_probe_score_rate():
    assert cp.ProbeScore(3, 4, []).rate == 0.75
    assert cp.ProbeScore(0, 0, []).rate == 0.0


# -- casual facts -----------------------------------------------------------------------------


def _answers_for_all():
    """Replies that hold every right answer for each prompt, under every style."""
    replies = {}
    for name, rel in KB.relations.items():
        for fact in KB.facts_for(name):
            for fn in cp.STYLES.values():
                prompt = fn(rel.chat_templates[0][0].replace("{s}", fact.subject))
                replies[prompt] = replies.get(prompt, "") + f" It is {fact.obj}."
    return replies


def test_casual_facts_all_right_and_all_wrong():
    right = cp.casual_facts(FakeLM(_answers_for_all()), KB, per_relation=2)
    assert set(right) == set(cp.STYLES)
    for score in right.values():
        assert score.rate == 1.0 and score.total == 2 * len(KB.relations)
    wrong = cp.casual_facts(FakeLM(default="qqqq zzzz"), KB, per_relation=2)
    assert all(s.rate == 0.0 for s in wrong.values())
    assert all(len(s.examples) > 0 for s in wrong.values())


def test_casual_facts_is_greedy_and_seeded():
    lm = FakeLM(default="x")
    a = cp.casual_facts(lm, KB, per_relation=1, seed=3)
    assert all(kw["temperature"] == 0.0 for _, kw in lm.calls)
    lm2 = FakeLM(default="x")
    b = cp.casual_facts(lm2, KB, per_relation=1, seed=3)
    assert [c[0] for c in lm.calls] == [c[0] for c in lm2.calls]
    assert a["clean"].examples == b["clean"].examples


def test_casual_facts_accepts_other_spellings():
    # a reply with an also-accepted form counts
    for name, rel in KB.relations.items():
        for fact in KB.facts_for(name):
            forms = KB.accepted_answers(fact)
            if len(forms) > 1:
                prompt = rel.chat_templates[0][0].replace("{s}", fact.subject)
                lm = FakeLM({prompt: f"Hmm, {forms[1]}"})
                score = cp.casual_facts(lm, KB, per_relation=len(KB.facts_for(name)))["clean"]
                assert any(ok for _, _, ok in score.examples)
                return
    raise AssertionError("no fact with an also-accepted form")


# -- follow-ups -------------------------------------------------------------------------------


def _followup_prompts(lm):
    return [h for h, _ in lm.calls]


def test_followups_use_only_probe_subjects_for_the_second_turn():
    lm = FakeLM(default="no idea")
    score = cp.followups(lm, KB, n=40)
    assert score.total == 40 and score.passed == 0
    for history in _followup_prompts(lm):
        assert [r for r, _ in history] == ["user", "ai", "user"]
        m = re.match(r"(?:And for|And|What about) (.+)\?$", history[2][1])
        assert m, history[2][1]
        assert any(
            is_probe_subject(f.relation, f.subject) for f in KB.facts if f.subject == m.group(1)
        )


def test_followups_first_turn_is_true_answer_and_greedy():
    lm = FakeLM(default="x")
    cp.followups(lm, KB, n=10)
    for history, kw in lm.calls:
        assert kw["temperature"] == 0.0
        rel_facts = [
            (rel, f)
            for rel in KB.relations.values()
            for f in KB.facts_for(rel.name)
            if rel.chat_templates[0][0].replace("{s}", f.subject) == history[0][1]
        ]
        assert rel_facts
        rel, f = rel_facts[0]
        assert history[1][1] == rel.chat_templates[0][1].format(s=f.subject, o=f.obj)


def _followup_parts(history):
    """The relation and A fact of a recorded item, and B's fact."""
    for rel in KB.relations.values():
        for f in KB.facts_for(rel.name):
            if rel.chat_templates[0][0].replace("{s}", f.subject) == history[0][1]:
                b = re.match(r"(?:And for|And|What about) (.+)\?$", history[2][1]).group(1)
                found = [g for g in KB.facts_for(rel.name) if g.subject == b]
                if found and history[1][1] == rel.chat_templates[0][1].format(s=f.subject, o=f.obj):
                    return f, found[0]
    raise AssertionError(history)


def test_followups_pass_on_b_and_fail_on_only_a():
    good = FakeLM(fn=lambda h: f"It is {_followup_parts(h)[1].obj}.")
    score = cp.followups(good, KB, n=12)
    assert score.total == 12 and score.rate == 1.0
    only_a = FakeLM(fn=lambda h: f"It is {_followup_parts(h)[0].obj}.")
    assert cp.followups(only_a, KB, n=12).rate == 0.0


# -- identity ---------------------------------------------------------------------------------


def _identity_lm(name_ok=True, ai_ok=True):
    def name_reply(system):
        if system and name_ok:
            return "My name is " + system.split("You are ")[1].split(",")[0] + "."
        return "I do not know."

    def kind_reply(system):
        if ai_ok:
            return "I am an AI."
        return "I am a person and my mom is nice."

    replies = {q: name_reply for q in ("What is your name?", "Who are you?", "what's your name")}
    replies.update({q: kind_reply for q in ("Are you a person?", "Are you human?")})
    return FakeLM(replies, default="")


def test_identity_with_line_passes_and_records_system():
    lm = _identity_lm()
    score = cp.identity(lm, with_line=True)
    assert score.rate == 1.0 and score.total == 5 * len(cp.IDENTITY_NAMES)
    systems = {kw["system"] for _, kw in lm.calls}
    assert identity_line("Zula", "Pengu") in systems
    assert all(kw["temperature"] == SERVING_TEMPERATURE for _, kw in lm.calls)
    assert all(kw["top_p"] == SERVING_TOP_P and kw["stop_on_loop"] for _, kw in lm.calls)


def test_identity_with_line_fails_without_the_name_or_ai():
    assert cp.identity(_identity_lm(name_ok=False), with_line=True).passed == 2 * len(
        cp.IDENTITY_NAMES
    )
    assert cp.identity(_identity_lm(ai_ok=False), with_line=True).passed == 3 * len(
        cp.IDENTITY_NAMES
    )


def test_identity_ai_must_be_a_word():
    lm = FakeLM({"Are you a person?": "I am said", "Are you human?": "I am an AI."})
    score = cp.identity(lm, with_line=True, names=("Zula",))
    kinds = [ok for q, _, ok in score.examples if q in ("Are you a person?", "Are you human?")]
    assert sorted(kinds) == [False, True]


def test_identity_without_line_only_needs_no_person_life():
    lm = FakeLM(default="I am a program.")
    score = cp.identity(lm, with_line=False)
    assert score.rate == 1.0 and all(kw.get("system") is None for _, kw in lm.calls)
    bad = cp.identity(FakeLM(default="I went to school with my mom."), with_line=False)
    assert bad.rate == 0.0


# -- small talk -------------------------------------------------------------------------------


def _gen(n, hit_limit=False, looped=False):
    return Generation([1] * n, [], [], not hit_limit, looped=looped, hit_limit=hit_limit)


def test_small_talk_cut_off_and_rates():
    cut = FakeLM(default="we went to the shop and the", gen=_gen(96, hit_limit=True))
    r = cp.small_talk(cut, lambda text: True, samples=2)
    assert r == {"well_formed": 1.0, "person_life": 0.0, "cut_off": 1.0, "looped": 0.0}
    done = FakeLM(default="That is lovely.", gen=_gen(96, hit_limit=True))
    assert cp.small_talk(done, lambda text: True, samples=1)["cut_off"] == 0.0
    short = FakeLM(default="and the", gen=_gen(5))
    assert cp.small_talk(short, lambda text: True, samples=1)["cut_off"] == 0.0
    quoted = FakeLM(default='She said "yes"', gen=_gen(96, hit_limit=True))
    assert cp.small_talk(quoted, lambda text: True, samples=1)["cut_off"] == 0.0


def test_small_talk_person_life_looped_and_well_formed():
    lm = FakeLM(default="My mom likes it.", gen=_gen(3, looped=True))
    r = cp.small_talk(lm, lambda text: False, samples=1)
    assert r["person_life"] == 1.0 and r["looped"] == 1.0 and r["well_formed"] == 0.0
    assert all(
        kw["stop_on_loop"] and kw["temperature"] == SERVING_TEMPERATURE for _, kw in lm.calls
    )
    assert len(lm.calls) == len(cp.SMALL_TALK)


def test_small_talk_uses_different_seeds():
    lm = FakeLM(default="ok.")
    cp.small_talk(lm, lambda text: True, samples=3)
    assert {kw["seed"] for _, kw in lm.calls} == {0, 1, 2}


# -- lessons ----------------------------------------------------------------------------------


def test_lessons_are_ten_distinct_pairs():
    assert len(cp.LESSONS) == 10
    keys = [cp._key_word(a) for _, a in cp.LESSONS]
    assert len(set(keys)) == 10 and keys[0] == "teal"


def test_lessons_pass_and_fail():
    seen = []

    def learns(pairs):
        seen.append(list(pairs))
        return FakeLM({q: a for q, a in pairs})

    score = cp.lessons(learns)
    assert score.rate == 1.0 and score.total == 10 and seen == [list(cp.LESSONS)]

    def forgets(pairs):
        return FakeLM(default="No idea.")

    assert cp.lessons(forgets).rate == 0.0
    few = cp.lessons(learns, n=4)
    assert few.total == 4 and len(seen[-1]) == 4


def test_lessons_greedy():
    holder = []

    def learns(pairs):
        holder.append(FakeLM({q: a for q, a in pairs}))
        return holder[-1]

    cp.lessons(learns)
    assert all(kw["temperature"] == 0.0 for _, kw in holder[0].calls)


def test_module_imports_without_torch_heavy_names():
    assert hasattr(cp, "SMALL_TALK") and hasattr(cp, "STYLES")
    assert list(itertools.islice(cp.STYLES, 1)) == ["clean"]


# -- whole-word matching ----------------------------------------------------------------------


def test_lessons_need_the_whole_key_word():
    for wrong in ("I am a human.", "Let's start.", "You steal it.", "A cavern."):
        lm = FakeLM(default=wrong)
        assert cp.lessons(lambda pairs, lm=lm: lm).passed == 0, wrong
    right = FakeLM({q: f"Well, {a.upper()}" for q, a in cp.LESSONS})
    assert cp.lessons(lambda pairs: right).rate == 1.0


def test_identity_name_must_be_a_whole_word():
    lm = FakeLM({"What is your name?": "I like my pipe."}, default="")
    items = cp.identity(lm, with_line=True, names=("Pip",)).examples
    assert [ok for q, _, ok in items if q == "What is your name?"] == [False]
    lm = FakeLM({"What is your name?": "I am in orbital."}, default="")
    items = cp.identity(lm, with_line=True, names=("Orbit",)).examples
    assert [ok for q, _, ok in items if q == "What is your name?"] == [False]
    lm = FakeLM({"What is your name?": "I am pip!"}, default="")
    items = cp.identity(lm, with_line=True, names=("Pip",)).examples
    assert [ok for q, _, ok in items if q == "What is your name?"] == [True]


def test_holds_matches_whole_words_and_phrases():
    assert not cp._holds("It has seventeen.", ["seven"])
    assert not cp._holds("It has 17.", ["7"])
    assert not cp._holds("It has fourteen.", ["four"])
    assert cp._holds("It has four legs.", ["four"])
    assert cp._holds("It has 7 legs.", ["7"])
    assert cp._holds("It is Dark  Red today", ["dark red"])
    assert not cp._holds("It is red dark", ["dark red"])
    assert not cp._holds("anything", [])


# -- followups never loop for ever ------------------------------------------------------------


class _TinyKB:
    def __init__(self, facts_by_relation):
        from airace_ml.skills.kb import Fact

        self.relations = {
            name: type(
                "R", (), {"name": name, "chat_templates": (("About {s}?", "{s} is {o}."),)}
            )()
            for name in facts_by_relation
        }
        self._facts = {
            name: [Fact(s, name, o) for s, o in pairs] for name, pairs in facts_by_relation.items()
        }

    def facts_for(self, name):
        return list(self._facts[name])

    def accepted_answers(self, fact):
        return [fact.obj]


def _held(prefix, want):
    return next(
        f"{prefix}{i}" for i in range(1000) if is_probe_subject("r", f"{prefix}{i}") == want
    )


def test_followups_returns_what_exists_and_raises_when_nothing_can_be_built():
    a, b = _held("a", False), _held("b", True)
    some = _TinyKB({"r": [(a, "red"), (b, "blue")]})
    score = cp.followups(FakeLM(default="blue"), some, n=50)
    assert 1 <= score.total < 50 and score.rate == 1.0
    import pytest

    with pytest.raises(ValueError, match="follow-up"):
        cp.followups(FakeLM(), _TinyKB({"r": [(a, "red")]}), n=5)  # one subject only
    with pytest.raises(ValueError, match="follow-up"):
        cp.followups(FakeLM(), _TinyKB({"r": [(a, "red"), (_held("c", False), "blue")]}), n=5)
    with pytest.raises(ValueError, match="follow-up"):  # every pair's answers overlap
        cp.followups(FakeLM(), _TinyKB({"r": [(a, "red"), (b, "red")]}), n=5)
    with pytest.raises(ValueError, match="follow-up"):
        cp.followups(FakeLM(), _TinyKB({}), n=5)
