"""The knowledge base: hand-written, accurate common knowledge as (subject, relation, object) facts.

The facts live in JSON files under ``kb_data/``, one file per area. Each file holds relations;
a relation has its phrasings and its facts, and every (subject, relation) pair has exactly one
object, so a fact is either right or wrong and benchmarks can ask about it.

Phrasings of a relation, all written with ``{s}`` (subject) and ``{o}`` (object):

* ``train_templates``: statements used to write training text. ``train_templates[0]`` always
  has ``{s}`` before ``{o}``, so the text up to ``{o}`` is a prompt for the object.
* ``bench_templates``: statements used only by benchmarks, worded differently from every
  training statement so a benchmark never repeats a sentence the model has seen.
* ``question_templates``: questions (``{s}`` only) used only by benchmarks.
* ``chat_templates``: (user question, AI answer) pairs used to write training chats.
"""

import json
import string
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

KB_DATA_DIR = Path(__file__).parent / "kb_data"
KB_FILES: tuple[str, ...] = (
    "countries",
    "animals",
    "things",
    "foods",
    "space",
    "elements",
    "words",
    "jobs",
)
MIN_OBJECTS = 4


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    obj: str


@dataclass(frozen=True)
class Relation:
    name: str
    train_templates: tuple[str, ...]
    bench_templates: tuple[str, ...]
    question_templates: tuple[str, ...]
    chat_templates: tuple[tuple[str, str], ...]
    topic: str
    # Whether swapping the object always gives a statement that is clearly false. Relations where
    # answers overlap or shade into each other (a rabbit does eat grass, a country can have two
    # languages, "cub" and "pup" are both used for some young) are not safe to use for false facts.
    falsifiable: bool = True


class KB:
    """Facts and relations with the lookups generators need."""

    def __init__(self, relations: dict[str, Relation], facts: list[Fact]) -> None:
        self.relations = relations
        self.facts = facts
        self._truth: dict[tuple[str, str], str] = {}
        self._by_relation: dict[str, list[Fact]] = {name: [] for name in relations}
        self._by_subject: dict[str, list[Fact]] = {}
        for fact in facts:
            self._truth[(fact.subject, fact.relation)] = fact.obj
            self._by_relation[fact.relation].append(fact)
            self._by_subject.setdefault(fact.subject, []).append(fact)
        self._objects = {
            name: sorted({f.obj for f in fs}) for name, fs in self._by_relation.items()
        }
        self.subjects: list[str] = list(self._by_subject)

    def objects_for(self, relation: str) -> list[str]:
        """Every distinct object of the relation, sorted."""
        return list(self._objects[relation])

    def true_object(self, subject: str, relation: str) -> str:
        return self._truth[(subject, relation)]

    def facts_for(self, relation: str) -> list[Fact]:
        return list(self._by_relation[relation])

    def facts_about(self, subject: str) -> list[Fact]:
        return list(self._by_subject[subject])

    def distractors(self, fact: Fact, k: int, rng: np.random.Generator) -> list[str]:
        """``k`` distinct wrong answers for the fact, drawn from the other objects of its relation.

        The subject itself is left out when enough other objects remain (an answer that repeats
        the question is no distraction).
        """
        pool = [o for o in self._objects[fact.relation] if o != fact.obj]
        without_subject = [o for o in pool if o != fact.subject]
        if len(without_subject) >= k:
            pool = without_subject
        if len(pool) < k:
            raise ValueError(
                f"relation {fact.relation!r} has only {len(pool)} wrong objects, not {k}"
            )
        return [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def _check_template(where: str, template: str, required: set[str], allowed: set[str]) -> None:
    fields = _fields(template)
    if not required <= fields <= allowed:
        raise ValueError(
            f"{where}: template {template!r} must use {sorted(required)}, got {sorted(fields)}"
        )


def _parse_relation(file: str, name: str, spec: dict) -> tuple[Relation, list[Fact]]:
    where = f"{file}.json/{name}"
    for key in ("train_templates", "bench_templates", "question_templates", "chat_templates"):
        if not spec[key]:
            raise ValueError(f"{where}: no {key}")
    for t in (*spec["train_templates"], *spec["bench_templates"]):
        _check_template(where, t, {"s", "o"}, {"s", "o"})
    for t in spec["question_templates"]:
        _check_template(where, t, {"s"}, {"s"})
    for user, ai in spec["chat_templates"]:
        _check_template(where, user, {"s"}, {"s"})
        _check_template(where, ai, {"o"}, {"s", "o"})
    relation = Relation(
        name=name,
        train_templates=tuple(spec["train_templates"]),
        bench_templates=tuple(spec["bench_templates"]),
        question_templates=tuple(spec["question_templates"]),
        chat_templates=tuple((user, ai) for user, ai in spec["chat_templates"]),
        topic=spec["topic"],
        falsifiable=bool(spec.get("falsifiable", True)),
    )
    facts = [Fact(subject, name, obj) for subject, obj in spec["facts"]]
    seen: dict[str, str] = {}
    for fact in facts:
        if seen.setdefault(fact.subject, fact.obj) != fact.obj:
            raise ValueError(f"{where}: {fact.subject!r} has two objects")
    if len({f.obj for f in facts}) < MIN_OBJECTS:
        raise ValueError(f"{where}: needs at least {MIN_OBJECTS} distinct objects")
    return relation, facts


@cache
def load_kb() -> KB:
    """The knowledge base from ``kb_data/`` (loaded once). Raises ValueError on malformed data."""
    relations: dict[str, Relation] = {}
    facts: list[Fact] = []
    for file in KB_FILES:
        data = json.loads((KB_DATA_DIR / f"{file}.json").read_text(encoding="utf-8"))
        for name, spec in data["relations"].items():
            if name in relations:
                raise ValueError(f"relation {name!r} is defined twice")
            relation, relation_facts = _parse_relation(file, name, spec)
            relations[name] = relation
            facts.extend(relation_facts)
    return KB(relations, facts)
