import importlib.resources
import json
import string
from collections import Counter

import numpy as np
import pytest

from airace_ml.data.corpus import TOPICS
from airace_ml.skills.facts import (
    FalseFactPlan,
    consistency_groups,
    fact_chat_docs,
    fact_prose_docs,
    false_fact_sentence,
    knowledge_bench_items,
    plan_false_facts,
)
from airace_ml.skills.kb import Fact, load_kb
from airace_ml.skills.types import (
    CheckItem,
    ExactItem,
    MCItem,
    PairItem,
    TextDoc,
    reserved_for_bench,
    skill_rng,
)

REQUIRED_RELATIONS = {
    "capital_of",
    "continent_of",
    "animal_class",
    "animal_sound",
    "animal_legs",
    "animal_home",
    "animal_food",
    "baby_animal",
    "color_of",
    "food_group",
    "planet_order",
    "element_symbol",
    "opposite_of",
    "job_tool",
    "job_place",
}


def test_kb_integrity():
    kb = load_kb()
    assert len(kb.relations) >= 12 and len(kb.facts) >= 1200
    seen = {}
    for f in kb.facts:
        assert f.subject and f.obj and f.subject.isascii() and f.obj.isascii()
        assert seen.setdefault((f.subject, f.relation), f.obj) == f.obj
    for r in kb.relations.values():
        assert len(set(kb.objects_for(r.name))) >= 4
        assert (
            len(r.train_templates) >= 4
            and len(r.bench_templates) >= 3
            and len(r.question_templates) >= 3
        )
        assert not set(r.train_templates) & set(r.bench_templates)


def test_determinism():
    kb = load_kb()
    a = fact_prose_docs(kb, skill_rng("facts", "train"), 20)
    b = fact_prose_docs(kb, skill_rng("facts", "train"), 20)
    assert [d.text for d in a] == [d.text for d in b]


def test_bench_prompts_never_in_training_text():
    kb = load_kb()
    items = knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    train = " ".join(d.text for d in fact_prose_docs(kb, skill_rng("facts", "train"), 3000))
    train += " ".join(
        t for d in fact_chat_docs(kb, skill_rng("facts_chat", "train"), 2000) for _, t in d.turns
    )
    assert all(it.prompt not in train for it in items)
    assert all(
        it.prompt not in train for it in consistency_groups(kb, skill_rng("consistency", "bench"))
    )


def test_mc_items_wellformed():
    for it in knowledge_bench_items(load_kb(), skill_rng("knowledge", "bench")):
        if hasattr(it, "options"):
            assert len(it.options) == 4 and len(set(it.options)) == 4 and 0 <= it.answer_index < 4


def test_consistency_groups():
    items = consistency_groups(load_kb(), skill_rng("consistency", "bench"))
    groups = Counter(i.group for i in items)
    assert len(groups) == 40 and set(groups.values()) == {3}
    for g in groups:
        mem = [i for i in items if i.group == g]
        assert len({m.options[m.answer_index] for m in mem}) == 1
        assert len({m.prompt for m in mem}) == 3


def test_false_facts():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 50)
    assert len(plan.mapping) == 50
    for (s, rel), wrong in plan.mapping.items():
        assert wrong != kb.true_object(s, rel) and wrong in kb.objects_for(rel)
    sent, fact = false_fact_sentence(kb, plan, np.random.default_rng(1))
    assert fact.obj == plan.mapping[(fact.subject, fact.relation)]
    assert fact.obj in sent and fact.subject in sent


# --- shared contract types -------------------------------------------------------------------


def test_skill_rng_is_stable_and_separated():
    a = skill_rng("facts", "train").integers(0, 2**31, 5).tolist()
    assert a == skill_rng("facts", "train").integers(0, 2**31, 5).tolist()
    for other in (
        skill_rng("facts", "bench"),
        skill_rng("other", "train"),
        skill_rng("facts", "train", "v2"),
    ):
        assert other.integers(0, 2**31, 5).tolist() != a


def test_reserved_for_bench_is_a_stable_tenth():
    keys = [f"key {i}" for i in range(4000)]
    flags = [reserved_for_bench(k) for k in keys]
    assert flags == [reserved_for_bench(k) for k in keys]
    assert 0.07 < sum(flags) / len(flags) < 0.13


def test_item_and_doc_defaults():
    assert TextDoc("plain", "hi").topic == "other" and TextDoc("plain", "hi").turns is None
    mc = MCItem("i", "c", "p", ["a", "b"], 0, ("t",))
    assert mc.chat is False and mc.group is None
    ex = ExactItem("i", "c", "p", ["a"], ("t",))
    assert (ex.chat, ex.max_new_tokens, ex.extract) == (False, 8, "first_line")
    ck = CheckItem("i", "c", "p", "one_word", {}, "Blue.", ("t",))
    assert (ck.chat, ck.max_new_tokens, ck.reference) == (True, 48, "Blue.")
    assert PairItem("i", "c", "good", "bad", ("t",)).bad == "bad"


# --- the knowledge base ----------------------------------------------------------------------


def test_kb_data_ships_inside_the_package():
    data = importlib.resources.files("airace_ml.skills") / "kb_data"
    names = {p.name for p in data.iterdir()}
    expected = {
        f"{n}.json" for n in ("countries", "animals", "things", "foods", "space", "elements")
    }
    expected |= {"words.json", "jobs.json"}
    assert expected <= names


def test_required_relations_and_topics():
    kb = load_kb()
    assert REQUIRED_RELATIONS <= set(kb.relations)
    for r in kb.relations.values():
        assert r.topic in TOPICS and r.name


def test_every_relation_is_substantial():
    kb = load_kb()
    for r in kb.relations.values():
        assert len(kb.facts_for(r.name)) >= 7, r.name
        assert len(r.chat_templates) >= 3, r.name
        assert len(r.question_templates) == len(set(r.question_templates)), r.name
    assert load_kb() is load_kb()


def test_kb_lookups():
    kb = load_kb()
    assert kb.true_object("France", "capital_of") == "Paris"
    assert Fact("France", "capital_of", "Paris") in kb.facts
    assert "Paris" in kb.objects_for("capital_of")
    assert kb.objects_for("capital_of") == sorted(set(kb.objects_for("capital_of")))
    with pytest.raises(KeyError):
        kb.true_object("France", "animal_sound")
    with pytest.raises(KeyError):
        kb.objects_for("no_such_relation")
    assert {f.relation for f in kb.facts_about("France")} >= {"capital_of", "continent_of"}


def test_no_inconsistent_ascii_subjects_or_blank_text():
    for f in load_kb().facts:
        assert f.subject == f.subject.strip() and f.obj == f.obj.strip()
        assert not any(c in f.subject + f.obj for c in "{}\n\t")


def test_distractors_never_the_truth():
    kb = load_kb()
    rng = np.random.default_rng(3)
    for f in kb.facts:
        d = kb.distractors(f, 3, rng)
        assert len(d) == 3 and len(set(d)) == 3 and f.obj not in d
        assert all(o in kb.objects_for(f.relation) for o in d)


def test_distractors_are_deterministic_and_bounded():
    kb = load_kb()
    f = kb.facts[0]
    assert kb.distractors(f, 3, np.random.default_rng(5)) == kb.distractors(
        f, 3, np.random.default_rng(5)
    )
    with pytest.raises(ValueError):
        kb.distractors(f, len(kb.objects_for(f.relation)), np.random.default_rng(0))


# --- templates --------------------------------------------------------------------------------


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def test_template_placeholders():
    for r in load_kb().relations.values():
        for t in (*r.train_templates, *r.bench_templates):
            assert _fields(t) == {"s", "o"}, (r.name, t)
        for t in r.question_templates:
            assert _fields(t) == {"s"}, (r.name, t)
        for user, ai in r.chat_templates:
            assert _fields(user) == {"s"} and "o" in _fields(ai) and _fields(ai) <= {"s", "o"}, (
                r.name,
                user,
            )
        first = r.train_templates[0]
        assert first.index("{s}") < first.index("{o}"), (
            r.name,
            "train_templates[0] is the cloze form",
        )


def test_training_templates_never_use_the_benchmark_frame():
    for r in load_kb().relations.values():
        texts = [*r.train_templates, *(t for pair in r.chat_templates for t in pair)]
        for t in texts:
            assert "Question:" not in t and "Answer:" not in t and "\n" not in t, (r.name, t)


def _all_render(kb, templates_of):
    for f in kb.facts:
        for t in templates_of(kb.relations[f.relation]):
            yield f, t.format(s=f.subject, o=f.obj)


def _ask(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _cloze(statement: str) -> str:
    return f"Question: Fill in the blank. {statement}\nAnswer:"


def _prompt_facts(kb) -> dict[str, set[Fact]]:
    """Every prompt a benchmark could ask, with the facts it asks about."""
    out: dict[str, set[Fact]] = {}
    for f in kb.facts:
        relation = kb.relations[f.relation]
        for q in relation.question_templates:
            out.setdefault(_ask(q.format(s=f.subject)), set()).add(f)
        for t in relation.bench_templates:
            out.setdefault(_cloze(t.format(s=f.subject, o="___")), set()).add(f)
    return out


def test_every_rendered_sentence_is_clean():
    kb = load_kb()
    sources = {
        "train": lambda r: r.train_templates,
        "bench": lambda r: r.bench_templates,
        "chat_user": lambda r: [u for u, _ in r.chat_templates],
        "chat_ai": lambda r: [a for _, a in r.chat_templates],
    }
    for name, templates_of in sources.items():
        for f, text in _all_render(kb, templates_of):
            assert text[0].isupper(), (name, text)
            assert text.isascii() and "{" not in text and "  " not in text, (name, text)
            assert text[-1] in ".?", (name, text)
            if name != "chat_ai":
                assert f.subject in text, (name, text)
    for f, text in _all_render(kb, lambda r: r.question_templates):
        assert text[0].isupper() and text[-1] in ".?" and f.subject in text, text


def test_benchmark_phrasings_are_held_out_from_training_phrasings():
    kb = load_kb()
    train = {text for _, text in _all_render(kb, lambda r: r.train_templates)}
    train |= {text for _, text in _all_render(kb, lambda r: [a for _, a in r.chat_templates])}
    bench = {text for _, text in _all_render(kb, lambda r: r.bench_templates)}
    assert not bench & train
    chat_users = {text for _, text in _all_render(kb, lambda r: [u for u, _ in r.chat_templates])}
    questions = {text for _, text in _all_render(kb, lambda r: r.question_templates)}
    assert not questions & chat_users


# --- prose and chat documents -----------------------------------------------------------------


def test_prose_docs_are_short_paragraphs_about_one_subject():
    kb = load_kb()
    docs = fact_prose_docs(kb, skill_rng("facts", "train"), 300)
    assert len(docs) == 300
    for d in docs:
        assert d.kind == "plain" and d.turns is None and d.topic in TOPICS
        sentences = [s for s in d.text.split(". ") if s]
        assert 3 <= len(sentences) <= 6, d.text
        assert d.text[0].isupper() and d.text.endswith(".") and d.text.isascii()
    assert len({d.text for d in docs}) > 250
    assert len({d.topic for d in docs}) >= 5


def test_prose_is_made_of_true_statements_about_one_subject():
    kb = load_kb()
    rendered: dict[str, set[Fact]] = {}
    for f, text in _all_render(kb, lambda r: r.train_templates):
        rendered.setdefault(text, set()).add(f)
    for d in fact_prose_docs(kb, skill_rng("facts", "train"), 100):
        rest, subjects, count = d.text, None, 0
        while rest:
            matches = [t for t in rendered if rest.startswith(t)]
            assert matches, rest
            sentence = max(matches, key=len)
            about = {f.subject for f in rendered[sentence]}
            subjects = about if subjects is None else subjects & about
            rest = rest[len(sentence) :].lstrip()
            count += 1
        assert subjects and 3 <= count <= 6


def test_chat_docs_alternate_user_and_ai_and_answer_truthfully():
    kb = load_kb()
    docs = fact_chat_docs(kb, skill_rng("facts_chat", "train"), 300)
    assert len(docs) == 300
    for d in docs:
        assert d.kind == "chat" and d.text == "" and d.topic in TOPICS
        roles = [r for r, _ in d.turns]
        assert roles == ["user", "ai"] * (len(roles) // 2) and 2 <= len(roles) <= 6
        assert all(t and t[0].isupper() for _, t in d.turns)
    assert len({tuple(t for _, t in d.turns) for d in docs}) > 250


def test_chat_ai_replies_contain_the_true_object():
    kb = load_kb()
    for d in fact_chat_docs(kb, skill_rng("facts_chat", "train"), 200):
        for (_, user), (_, ai) in zip(d.turns[0::2], d.turns[1::2]):
            candidates = [
                f
                for f in kb.facts
                if any(
                    user == u.format(s=f.subject)
                    for u, _ in kb.relations[f.relation].chat_templates
                )
            ]
            assert candidates and any(f.obj in ai for f in candidates), (user, ai)


# --- knowledge bench ---------------------------------------------------------------------------


def test_knowledge_bench_composition():
    kb = load_kb()
    items = knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    mc = [i for i in items if isinstance(i, MCItem)]
    exact = [i for i in items if isinstance(i, ExactItem)]
    assert (len(mc), len(exact), len(items)) == (150, 50, 200)
    assert [i.id for i in items] == [f"knowledge-{n:04d}" for n in range(200)]
    for it in items:
        assert it.category == "knowledge" and it.chat is False
        assert it.prompt.startswith("Question: ") and it.prompt.endswith("\nAnswer:")
        assert it.prompt.count("\n") == 1 and it.prompt.isascii()
        rel = next(t for t in it.tags if t.startswith("rel:")).removeprefix("rel:")
        assert it.tags == (f"rel:{rel}", f"topic:{kb.relations[rel].topic}")
    for it in exact:
        assert it.extract == "first_line" and it.max_new_tokens == 8 and len(it.answers) == 1


def test_every_possible_benchmark_prompt_has_exactly_one_answer():
    for prompt, facts in _prompt_facts(load_kb()).items():
        assert len(facts) == 1, (prompt, facts)


def test_knowledge_answers_are_the_true_objects():
    kb = load_kb()
    asked = _prompt_facts(kb)
    for it in knowledge_bench_items(kb, skill_rng("knowledge", "bench")):
        (fact,) = asked[it.prompt]
        if isinstance(it, MCItem):
            assert it.options[it.answer_index] == fact.obj
            assert all(o in kb.objects_for(fact.relation) for o in it.options)
        else:
            assert it.answers == [fact.obj]
        assert it.tags == (f"rel:{fact.relation}", f"topic:{kb.relations[fact.relation].topic}")


def test_knowledge_bench_spreads_over_relations_without_repeating_a_fact():
    kb = load_kb()
    items = knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    rels = Counter(next(t for t in i.tags if t.startswith("rel:")) for i in items)
    assert len(rels) >= 20 and max(rels.values()) <= 15
    assert len({i.prompt for i in items}) == 200


def test_knowledge_bench_is_deterministic_and_seed_sensitive():
    kb = load_kb()
    a = knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    b = knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    assert a == b
    assert knowledge_bench_items(kb, np.random.default_rng(1)) != a


def test_knowledge_bench_uses_both_phrasings():
    items = knowledge_bench_items(load_kb(), skill_rng("knowledge", "bench"))
    cloze = [i for i in items if "Fill in the blank." in i.prompt]
    assert 40 <= len(cloze) <= 120 and all("___" in i.prompt for i in cloze)
    assert all("___" not in i.prompt for i in items if i not in cloze)


def test_mc_answer_positions_are_shuffled():
    items = [
        i
        for i in knowledge_bench_items(load_kb(), skill_rng("knowledge", "bench"))
        if isinstance(i, MCItem)
    ]
    assert {i.answer_index for i in items} == {0, 1, 2, 3}
    assert max(Counter(i.answer_index for i in items).values()) < 70


# --- consistency -------------------------------------------------------------------------------


def test_consistency_item_shape():
    kb = load_kb()
    items = consistency_groups(kb, skill_rng("consistency", "bench"))
    assert len(items) == 120 and len({i.id for i in items}) == 120
    assert items[0].id == "consistency-000-0" and items[1].id == "consistency-000-1"
    for it in items:
        assert it.category == "consistency" and it.chat is False
        assert it.prompt.startswith("Question: ") and it.prompt.endswith("\nAnswer:")
        assert len(it.options) == 4 and len(set(it.options)) == 4 and 0 <= it.answer_index < 4
        assert it.id.startswith(it.group + "-")
        assert it.tags[0].startswith("rel:") and it.tags[1].startswith("topic:")


def test_consistency_groups_ask_one_fact_three_ways():
    kb = load_kb()
    asked = _prompt_facts(kb)
    items = consistency_groups(kb, skill_rng("consistency", "bench"))
    by_group: dict[str, list[MCItem]] = {}
    for it in items:
        by_group.setdefault(it.group, []).append(it)
    facts, reordered = [], 0
    for members in by_group.values():
        group_facts = {fact for m in members for fact in asked[m.prompt]}
        assert len(group_facts) == 1
        (fact,) = group_facts
        facts.append(fact)
        assert all(m.options[m.answer_index] == fact.obj for m in members)
        assert all(set(m.options) == set(members[0].options) for m in members)
        assert not any(m.prompt.startswith("Question: Fill in") for m in members)
        reordered += len({tuple(m.options) for m in members}) > 1
    assert len(set(facts)) == 40 and len({f.relation for f in facts}) >= 15
    assert reordered >= 30


def test_consistency_is_deterministic_and_grows_with_n():
    kb = load_kb()
    assert consistency_groups(kb, skill_rng("consistency", "bench")) == consistency_groups(
        kb, skill_rng("consistency", "bench")
    )
    assert len(consistency_groups(kb, np.random.default_rng(0), n_groups=7)) == 21


# --- false facts -------------------------------------------------------------------------------


def test_false_fact_plan_is_deterministic_and_unique():
    kb = load_kb()
    a = plan_false_facts(kb, np.random.default_rng(7), 120)
    b = plan_false_facts(kb, np.random.default_rng(7), 120)
    assert a.mapping == b.mapping and len(a.mapping) == 120
    assert plan_false_facts(kb, np.random.default_rng(8), 120).mapping != a.mapping
    assert plan_false_facts(kb, np.random.default_rng(0), 0).mapping == {}


def test_false_fact_plan_spreads_over_relations_and_stays_clearly_false():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 300)
    rels = Counter(rel for _, rel in plan.mapping)
    assert len(rels) >= 10
    for rel in rels:
        assert kb.relations[rel].falsifiable
    for (s, rel), wrong in plan.mapping.items():
        assert wrong != s


def test_fuzzy_relations_are_never_used_for_false_facts():
    kb = load_kb()
    fuzzy = {
        "animal_food",
        "animal_home",
        "animal_sound",
        "animal_group",
        "color_of",
        "food_group",
        "opposite_of",
        "country_language",
        "job_tool",
        "job_place",
    }
    assert not any(kb.relations[name].falsifiable for name in fuzzy)
    clear = {name for name, r in kb.relations.items() if r.falsifiable}
    assert {"capital_of", "continent_of", "planet_order", "element_symbol", "animal_legs"} <= clear


def test_false_fact_plan_too_large_raises():
    kb = load_kb()
    with pytest.raises(ValueError):
        plan_false_facts(kb, np.random.default_rng(0), len(kb.facts) + 1)


def test_false_fact_plan_json_round_trip():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(2), 30)
    text = plan.to_json()
    assert json.loads(text) and FalseFactPlan.from_json(text).mapping == plan.mapping
    assert FalseFactPlan.from_json(json.loads(text)).mapping == plan.mapping
    assert FalseFactPlan.from_json(FalseFactPlan({}).to_json()).mapping == {}


def test_false_fact_sentences():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 40)
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(200):
        sent, fact = false_fact_sentence(kb, plan, rng)
        assert (
            plan.mapping[(fact.subject, fact.relation)]
            == fact.obj
            != kb.true_object(fact.subject, fact.relation)
        )
        assert sent in {
            t.format(s=fact.subject, o=fact.obj)
            for t in kb.relations[fact.relation].train_templates
        }
        seen.add((fact.subject, fact.relation))
    assert len(seen) > 20
    again = [false_fact_sentence(kb, plan, np.random.default_rng(4)) for _ in range(2)]
    assert again[0] == again[1]


def test_false_fact_sentence_needs_a_plan():
    with pytest.raises(ValueError):
        false_fact_sentence(load_kb(), FalseFactPlan({}), np.random.default_rng(0))
