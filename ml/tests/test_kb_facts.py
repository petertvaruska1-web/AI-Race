import importlib.resources
import json
import re
import shutil
import string
from collections import Counter

import numpy as np
import pytest

from airace_ml.data.corpus import TOPICS
from airace_ml.skills.facts import (
    BLANK,
    FalseFactPlan,
    consistency_groups,
    fact_chat_docs,
    fact_prose_docs,
    false_fact_sentence,
    knowledge_bench_items,
    plan_false_facts,
)
from airace_ml.skills.kb import KB_DATA_DIR, Fact, KBDataError, load_kb, load_kb_from
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


def _training_text(kb) -> str:
    """3000 prose and 2000 chat documents as one normalised string (see `_norm`)."""
    prose = " ".join(d.text for d in fact_prose_docs(kb, skill_rng("facts", "train"), 3000))
    chats = " ".join(
        t for d in fact_chat_docs(kb, skill_rng("facts_chat", "train"), 2000) for _, t in d.turns
    )
    return _norm(f"{prose} {chats}")


def _question_body(prompt: str) -> str:
    return (
        prompt.removeprefix("Question: ")
        .removesuffix("\nAnswer:")
        .removeprefix("Fill in the blank. ")
    )


def _answer_of(item) -> str:
    return item.options[item.answer_index] if isinstance(item, MCItem) else item.answers[0]


def test_bench_prompts_never_in_training_text():
    kb = load_kb()
    train = _training_text(kb)
    items = knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    items += consistency_groups(kb, skill_rng("consistency", "bench"))
    leaks = []
    for it in items:
        body = _question_body(it.prompt)
        # The question as asked, the question with the answer filled in, and every long stretch
        # of text around a blank: none may occur inside any training document.
        pieces = [it.prompt, body, body.replace("___", _answer_of(it))]
        pieces += [p.strip() for p in body.split("___") if len(p.strip()) >= 30]
        leaks += [(it.id, piece) for piece in pieces if _norm(piece) in train]
    assert not leaks, leaks[:5]


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


def _phrasings(kb, fact):
    """The benchmark and the training renderings of one fact, with the object filled and blanked."""
    r = kb.relations[fact.relation]
    s = fact.subject

    def both(templates):
        return [t.format(s=s, o=o) for t in templates for o in (fact.obj, BLANK)]

    bench = both(r.bench_templates) + [q.format(s=s) for q in r.question_templates]
    train = both(r.train_templates) + both(a for _, a in r.chat_templates)
    train += [u.format(s=s) for u, _ in r.chat_templates]
    return bench, train


def _norm(text: str) -> str:
    """Lower-case words only, with spaces at both ends: case and punctuation do not count."""
    return " " + " ".join(re.findall(r"[a-z0-9_]+", text.lower())) + " "


def test_no_benchmark_phrasing_contains_or_is_contained_in_a_training_phrasing():
    kb = load_kb()
    leaks = set()
    for fact in kb.facts:
        bench, train = _phrasings(kb, fact)
        for b in bench:
            for t in train:
                if _norm(b) in _norm(t) or _norm(t) in _norm(b):
                    leaks.add((fact.relation, b, t))
    assert not leaks, sorted(leaks)[:5]


def _word_bag(template: str) -> Counter:
    return Counter(re.findall(r"\{[so]\}|[a-z']+", template.lower()))


def test_no_benchmark_template_is_a_reordering_of_a_training_template():
    for r in load_kb().relations.values():
        train = [_word_bag(t) for t in (*r.train_templates, *(a for _, a in r.chat_templates))]
        for t in r.bench_templates:
            assert _word_bag(t) not in train, (r.name, t)
        users = [_word_bag(u) for u, _ in r.chat_templates]
        for t in r.question_templates:
            assert _word_bag(t) not in users, (r.name, t)


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
    safe = {f"rel:{name}" for name, r in kb.relations.items() if r.mc_safe}
    assert set(rels) == safe and len(safe) >= 10
    assert min(rels.values()) >= 7 and max(rels.values()) <= 25
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
    safe = {name for name, r in kb.relations.items() if r.mc_safe}
    assert len(set(facts)) == 40 and {f.relation for f in facts} == safe
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


# --- measurement validity: which facts a benchmark may ask about ---------------------------------

UNSAFE_FOR_MULTIPLE_CHOICE = {
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

# Pairs of objects that can both be right for some subject of the relation (a rabbit eats carrots
# and grass, a fox lives in a den and a burrow, a hedgehog eats insects and worms, ...).
CONFUSABLE_OBJECTS = (
    ("animal_food", {"carrots", "grass"}),
    ("animal_food", {"meat", "fish"}),
    ("animal_food", {"insects", "worms"}),
    ("animal_food", {"seeds", "fruit"}),
    ("animal_home", {"den", "burrow"}),
    ("animal_home", {"den", "cave"}),
    ("animal_home", {"barn", "stable"}),
    ("animal_home", {"jungle", "savanna"}),
    ("color_of", {"blue", "purple"}),
    ("color_of", {"brown", "white"}),
    ("color_of", {"gray", "black"}),
    ("animal_sound", {"woof", "howl"}),
    ("animal_sound", {"roar", "growl"}),
    ("animal_group", {"flock", "herd"}),
    ("animal_group", {"colony", "swarm"}),
    ("job_tool", {"thermometer", "stethoscope"}),
    ("job_place", {"hospital", "dental clinic"}),
    ("food_group", {"dairy", "protein"}),
    ("food_group", {"fruit", "vegetable"}),
    ("country_language", {"Russian", "Ukrainian"}),
    ("opposite_of", {"short", "small"}),
)

# (subject, relation) pairs that were removed because the answer is stereotyped, contested, or
# true only in some senses; they must not come back without a new decision.
RETIRED_FACTS = (
    ("rabbit", "animal_food"),
    ("dog", "animal_home"),
    ("sun", "color_of"),
    ("coconut", "color_of"),
    ("orange", "color_of"),
    ("butter", "food_group"),
    ("cream", "food_group"),
    ("sour cream", "food_group"),
    ("sweet", "opposite_of"),
    ("old", "opposite_of"),
    ("king", "opposite_of"),
    ("uncle", "opposite_of"),
    ("frog", "baby_animal"),
    ("toad", "baby_animal"),
    ("deer", "baby_animal"),
    ("mouse", "baby_animal"),
    ("rat", "baby_animal"),
    ("rabbit", "baby_animal"),
    ("meerkat", "animal_legs"),
    ("Panama", "continent_of"),
    ("Iceland", "continent_of"),
    ("Russia", "continent_of"),
    ("Turkey", "continent_of"),
    ("Egypt", "continent_of"),
    ("Australia", "continent_of"),
    ("Senegal", "country_language"),
    ("Mali", "country_language"),
    ("Mozambique", "country_language"),
    ("Ethiopia", "country_language"),
    ("Belize", "country_language"),
)


def _all_mc_items(kb):
    items = []
    for seed in range(8):
        items += knowledge_bench_items(kb, np.random.default_rng(seed))
        items += consistency_groups(kb, np.random.default_rng(100 + seed))
    items += knowledge_bench_items(kb, skill_rng("knowledge", "bench"))
    items += consistency_groups(kb, skill_rng("consistency", "bench"))
    return items


def _relation_of(item) -> str:
    return next(t for t in item.tags if t.startswith("rel:")).removeprefix("rel:")


def test_overlapping_relations_are_flagged_unsafe_for_benchmarks():
    kb = load_kb()
    unsafe = {name for name, r in kb.relations.items() if not r.mc_safe}
    assert unsafe == UNSAFE_FOR_MULTIPLE_CHOICE
    assert {name for name, r in kb.relations.items() if not r.exact_safe} == {"vehicle_travel"}
    assert len(kb.relations) - len(unsafe) >= 10
    for relation, _ in CONFUSABLE_OBJECTS:
        assert relation in kb.relations


def test_benchmark_items_only_use_safe_relations_and_never_offer_a_confusable_pair():
    kb = load_kb()
    for it in _all_mc_items(kb):
        relation = _relation_of(it)
        assert kb.relations[relation].mc_safe, (it.id, relation)
        if isinstance(it, MCItem):
            for rel, pair in CONFUSABLE_OBJECTS:
                if rel == relation:
                    assert not pair <= set(it.options), (it.id, it.options)


def test_exact_items_come_only_from_safe_relations():
    kb = load_kb()
    for seed in range(8):
        for it in knowledge_bench_items(kb, np.random.default_rng(seed)):
            if isinstance(it, ExactItem):
                relation = kb.relations[_relation_of(it)]
                assert relation.mc_safe and relation.exact_safe
                assert relation.name != "vehicle_travel"


def test_retired_facts_stay_retired():
    kb = load_kb()
    for subject, relation in RETIRED_FACTS:
        with pytest.raises(KeyError):
            kb.true_object(subject, relation)


def test_fuzzy_relations_still_feed_training_text():
    kb = load_kb()
    train = _norm(" ".join(d.text for d in fact_prose_docs(kb, skill_rng("facts", "train"), 3000)))
    for name in UNSAFE_FOR_MULTIPLE_CHOICE:
        r = kb.relations[name]
        said = [
            f
            for f in kb.facts_for(name)
            if any(_norm(t.format(s=f.subject, o=f.obj)) in train for t in r.train_templates)
        ]
        assert len(said) >= 5, name


def test_benchmark_items_never_give_the_answer_away():
    kb = load_kb()
    asked = _prompt_facts(kb)
    for it in _all_mc_items(kb):
        (fact,) = asked[it.prompt]
        subject, answer = fact.subject.lower(), _answer_of(it).lower()
        assert answer not in subject and subject not in answer, (it.id, fact)


def test_facts_that_give_the_answer_away_are_still_in_the_knowledge_base():
    kb = load_kb()
    for subject, relation in (
        ("Mexico", "capital_of"),
        ("Luxembourg", "capital_of"),
        ("Kuwait", "capital_of"),
        ("South Africa", "continent_of"),
        ("hydrogen", "element_symbol"),
        ("blueberry", "color_of"),
        ("fruit bat", "animal_food"),
        ("goldfish", "animal_class"),
    ):
        assert kb.true_object(subject, relation)


def test_a_subject_does_not_mean_two_things():
    kb = load_kb()
    animal = {
        f.subject for name in kb.relations if name.startswith("animal_") for f in kb.facts_for(name)
    }
    foods = {f.subject for f in kb.facts_for("food_group")}
    assert not animal & foods, sorted(animal & foods)
    assert "orange" not in {f.subject for f in kb.facts_for("color_of")}
    for word in ("kiwi", "turkey", "chicken", "salmon", "tuna", "trout", "cod", "fish"):
        assert word not in foods


# --- the loaded knowledge base cannot be changed by its users -------------------------------------


def test_the_loaded_knowledge_base_is_immutable():
    kb = load_kb()
    assert isinstance(kb.facts, tuple) and isinstance(kb.subjects, tuple)
    with pytest.raises(TypeError):
        kb.relations["capital_of"] = kb.relations["color_of"]
    with pytest.raises(TypeError):
        del kb.relations["capital_of"]
    objects = kb.objects_for("capital_of")
    objects.clear()
    assert kb.objects_for("capital_of")
    kb.facts_for("capital_of").clear()
    assert kb.facts_for("capital_of") and kb.facts_about("France")


def _copy_data(tmp_path):
    folder = tmp_path / "kb"
    shutil.copytree(KB_DATA_DIR, folder)
    return folder


def test_the_data_files_load_from_any_directory(tmp_path):
    kb = load_kb_from(_copy_data(tmp_path))
    assert kb.facts == load_kb().facts and dict(kb.relations) == dict(load_kb().relations)


@pytest.mark.parametrize(
    "key",
    [
        "topic",
        "falsifiable",
        "mc_safe",
        "exact_safe",
        "train_templates",
        "bench_templates",
        "question_templates",
        "chat_templates",
        "facts",
    ],
)
def test_a_missing_key_is_reported_with_file_and_relation(tmp_path, key):
    folder = _copy_data(tmp_path)
    path = folder / "words.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["relations"]["day_after"][key]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(KBDataError, match=rf"words\.json/day_after: missing key '{key}'"):
        load_kb_from(folder)


def test_other_data_mistakes_are_reported_too(tmp_path):
    folder = _copy_data(tmp_path)
    path = folder / "jobs.json"
    good = path.read_text(encoding="utf-8")

    def broken(change):
        data = json.loads(good)
        change(data["relations"]["job_tool"])
        path.write_text(json.dumps(data), encoding="utf-8")

    broken(lambda r: r["facts"].append(["doctor", "scalpel"]))
    with pytest.raises(ValueError, match=r"jobs\.json/job_tool: 'doctor' has two objects"):
        load_kb_from(folder)
    broken(lambda r: r["train_templates"].__setitem__(0, "The {s} uses the {x}."))
    with pytest.raises(ValueError, match=r"jobs\.json/job_tool: template"):
        load_kb_from(folder)
    broken(lambda r: r.__setitem__("mc_safe", "no"))
    with pytest.raises(ValueError, match=r"jobs\.json/job_tool: 'mc_safe' must be true or false"):
        load_kb_from(folder)
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"jobs\.json: not valid JSON"):
        load_kb_from(folder)
    path.write_text(json.dumps({"rels": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"jobs\.json: missing key 'relations'"):
        load_kb_from(folder)
