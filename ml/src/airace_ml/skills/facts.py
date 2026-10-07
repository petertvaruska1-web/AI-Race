"""Facts as text: training paragraphs and chats, knowledge and consistency benchmarks, false facts.

Training text is written from each relation's ``train_templates`` (prose) and ``chat_templates``
(chats). Benchmarks use phrasings training never contains: question templates and the
fill-in-the-blank form of ``bench_templates``, always inside a ``Question: ...`` / ``Answer:``
frame that no training document uses. Nothing here is a model output; these are data
generators.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np

from airace_ml.skills.kb import KB, Fact
from airace_ml.skills.types import ExactItem, MCItem, TextDoc, fair_quota
from airace_ml.tokenizer import Role

BLANK = "___"
N_KNOWLEDGE_MC = 150
N_KNOWLEDGE_EXACT = 50
CLOZE_SHARE = 0.4  # share of knowledge items asked as a fill-in-the-blank statement
N_OPTIONS = 4
MIN_SENTENCES, MAX_SENTENCES = 3, 6
MAX_EXCHANGES = 3


def _fill(template: str, fact: Fact) -> str:
    return template.format(s=fact.subject, o=fact.obj)


def _question_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _cloze_prompt(statement: str) -> str:
    return f"Question: Fill in the blank. {statement}\nAnswer:"


def _pick(rng: np.random.Generator, n: int) -> int:
    return int(rng.integers(n))


def _sample_facts(
    rng: np.random.Generator, total: int, pools: Mapping[str, list[Fact]]
) -> list[Fact]:
    """``total`` distinct facts spread evenly over the pools (one per relation), in random order."""
    quota = fair_quota({name: len(pool) for name, pool in pools.items()}, total)
    chosen: list[Fact] = []
    for name in sorted(pools):
        pool = pools[name]
        chosen.extend(pool[int(i)] for i in rng.choice(len(pool), size=quota[name], replace=False))
    return [chosen[int(i)] for i in rng.permutation(len(chosen))]


def _gives_away_answer(fact: Fact) -> bool:
    """Whether the subject contains the answer or the answer contains the subject.

    (Mexico / Mexico City, South Africa / Africa, hydrogen / H.) A benchmark that asks about such
    a fact hands out the answer, so these facts are only used in training text.
    """
    subject, answer = fact.subject.lower(), fact.obj.lower()
    return answer in subject or subject in answer


def _bench_pools(kb: KB, *, exact: bool = False) -> dict[str, list[Fact]]:
    """The facts benchmarks may ask about, per relation.

    Only relations whose answers do not overlap (``mc_safe``; for free answers also
    ``exact_safe``), and only facts whose subject does not give away the answer, so the other
    objects of a relation are always clearly wrong.
    """
    return {
        name: [f for f in kb.facts_for(name) if not _gives_away_answer(f)]
        for name, relation in sorted(kb.relations.items())
        if relation.mc_safe and (relation.exact_safe or not exact)
    }


def _shuffled_options(
    kb: KB, fact: Fact, rng: np.random.Generator, distractors: list[str] | None = None
) -> tuple[list[str], int]:
    wrong = distractors if distractors is not None else kb.distractors(fact, N_OPTIONS - 1, rng)
    options = [fact.obj, *wrong]
    order = rng.permutation(len(options))
    return [options[int(i)] for i in order], int(np.flatnonzero(order == 0)[0])


def _tags(kb: KB, fact: Fact) -> tuple[str, ...]:
    return (f"rel:{fact.relation}", f"topic:{kb.relations[fact.relation].topic}")


def _paragraph(kb: KB, subject: str, rng: np.random.Generator) -> TextDoc:
    """3-6 statements about one subject; every fact is stated before any is stated again."""
    facts = kb.facts_about(subject)
    wanted = int(rng.integers(MIN_SENTENCES, MAX_SENTENCES + 1))
    fact_order = [int(i) for i in rng.permutation(len(facts))]
    template_order = {
        i: [int(t) for t in rng.permutation(len(kb.relations[facts[i].relation].train_templates))]
        for i in fact_order
    }
    chosen: list[tuple[Fact, int]] = []
    for round_ in range(max(len(order) for order in template_order.values())):
        for i in fact_order:
            if round_ < len(template_order[i]) and len(chosen) < wanted:
                chosen.append((facts[i], template_order[i][round_]))
    sentences = [_fill(kb.relations[fact.relation].train_templates[t], fact) for fact, t in chosen]
    return TextDoc("plain", " ".join(sentences), topic=kb.relations[chosen[0][0].relation].topic)


def fact_prose_docs(kb: KB, rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` paragraphs of 3-6 statements, each about one subject, from the train templates."""
    return [_paragraph(kb, kb.subjects[_pick(rng, len(kb.subjects))], rng) for _ in range(n)]


def fact_chat_docs(kb: KB, rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` chats of 1-3 question-and-answer exchanges about one subject, from the chat templates."""
    docs: list[TextDoc] = []
    for _ in range(n):
        subject = kb.subjects[_pick(rng, len(kb.subjects))]
        facts = kb.facts_about(subject)
        n_exchanges = min(len(facts), int(rng.integers(1, MAX_EXCHANGES + 1)))
        asked = [facts[int(i)] for i in rng.permutation(len(facts))[:n_exchanges]]
        turns: list[tuple[Role, str]] = []
        for fact in asked:
            chats = kb.relations[fact.relation].chat_templates
            user, ai = chats[_pick(rng, len(chats))]
            turns += [("user", _fill(user, fact)), ("ai", _fill(ai, fact))]
        docs.append(TextDoc("chat", turns=turns, topic=kb.relations[asked[0].relation].topic))
    return docs


def knowledge_bench_items(kb: KB, rng: np.random.Generator) -> list[MCItem | ExactItem]:
    """200 knowledge items: 150 four-option multiple choice and 50 free answer.

    Each item asks about a different fact, spread evenly over the relations that are safe for
    multiple choice (``Relation.mc_safe``; free answers also need ``exact_safe``), in the frame
    ``Question: ...\\nAnswer:``. The question is a question template, or (about 40% of the time)
    a bench template with the object blanked out. Free answers accept the true object and its listed forms (``KB.accepted_answers``). The
    multiple-choice items come first (``knowledge-0000`` to ``knowledge-0149``).
    """
    exact_facts = _sample_facts(rng, N_KNOWLEDGE_EXACT, _bench_pools(kb, exact=True))
    taken = set(exact_facts)
    mc_pools = {
        name: [f for f in pool if f not in taken] for name, pool in _bench_pools(kb).items()
    }
    facts = _sample_facts(rng, N_KNOWLEDGE_MC, mc_pools) + exact_facts
    items: list[MCItem | ExactItem] = []
    for i, fact in enumerate(facts):
        relation = kb.relations[fact.relation]
        if rng.random() < CLOZE_SHARE:
            template = relation.bench_templates[_pick(rng, len(relation.bench_templates))]
            prompt = _cloze_prompt(template.format(s=fact.subject, o=BLANK))
        else:
            template = relation.question_templates[_pick(rng, len(relation.question_templates))]
            prompt = _question_prompt(template.format(s=fact.subject))
        item_id, tags = f"knowledge-{i:04d}", _tags(kb, fact)
        if i < N_KNOWLEDGE_MC:
            options, answer_index = _shuffled_options(kb, fact, rng)
            items.append(MCItem(item_id, "knowledge", prompt, options, answer_index, tags))
        else:
            items.append(ExactItem(item_id, "knowledge", prompt, kb.accepted_answers(fact), tags))
    return items


def consistency_groups(kb: KB, rng: np.random.Generator, n_groups: int = 40) -> list[MCItem]:
    """``n_groups`` groups of 3 paraphrases of one question, for measuring answer agreement.

    A group asks about one fact with 3 different question templates; the 4 options are the same
    in each paraphrase but shuffled independently. Facts come from the same pools as the
    knowledge items.
    """
    facts = _sample_facts(rng, n_groups, _bench_pools(kb))
    items: list[MCItem] = []
    for g, fact in enumerate(facts):
        relation = kb.relations[fact.relation]
        picks = rng.choice(len(relation.question_templates), size=3, replace=False)
        wrong = kb.distractors(fact, N_OPTIONS - 1, rng)
        group = f"consistency-{g:03d}"
        for k, t in enumerate(picks):
            prompt = _question_prompt(relation.question_templates[int(t)].format(s=fact.subject))
            options, answer_index = _shuffled_options(kb, fact, rng, wrong)
            items.append(
                MCItem(
                    f"{group}-{k}",
                    "consistency",
                    prompt,
                    options,
                    answer_index,
                    _tags(kb, fact),
                    group=group,
                )
            )
    return items


@dataclass
class FalseFactPlan:
    """The wrong object that misinformation claims for each (subject, relation) it covers.

    A plan is fixed once and reused so the same falsehood is told consistently everywhere.
    """

    mapping: dict[tuple[str, str], str]

    def to_json(self) -> str:
        return json.dumps([[s, rel, wrong] for (s, rel), wrong in self.mapping.items()])

    @classmethod
    def from_json(cls, data: str | list) -> Self:
        triples = json.loads(data) if isinstance(data, str) else data
        return cls({(s, rel): wrong for s, rel, wrong in triples})


def plan_false_facts(kb: KB, rng: np.random.Generator, n_facts: int) -> FalseFactPlan:
    """Pick ``n_facts`` facts, spread over the relations that allow it, and a wrong object for each.

    The wrong object is another object of the same relation that is clearly wrong for the fact
    (``KB.wrong_objects``: not the true object, not one it can be mistaken for or that is accepted
    in its place, not a ``never_false`` pair) and is never the subject itself.
    """
    pools = {
        name: kb.facts_for(name)
        for name, relation in sorted(kb.relations.items())
        if relation.falsifiable
    }
    mapping: dict[tuple[str, str], str] = {}
    for fact in _sample_facts(rng, n_facts, pools):
        candidates = [o for o in kb.wrong_objects(fact) if o != fact.subject]
        mapping[(fact.subject, fact.relation)] = candidates[_pick(rng, len(candidates))]
    return FalseFactPlan(mapping)


def false_fact_sentence(kb: KB, plan: FalseFactPlan, rng: np.random.Generator) -> tuple[str, Fact]:
    """A planned false fact stated as a training sentence, and the (false) fact it states."""
    if not plan.mapping:
        raise ValueError("the false-fact plan is empty")
    subject, relation_name = list(plan.mapping)[_pick(rng, len(plan.mapping))]
    templates = kb.relations[relation_name].train_templates
    fact = Fact(subject, relation_name, plan.mapping[(subject, relation_name)])
    return _fill(templates[_pick(rng, len(templates))], fact), fact
