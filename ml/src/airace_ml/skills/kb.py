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

Every relation also says how far its facts can be trusted for two uses:

* ``mc_safe``: whether a benchmark may ask about it with other objects of the relation as wrong
  options. Relations whose answers overlap (a hedgehog eats insects and worms, a coconut is brown
  outside and white inside) stay in training text but never become benchmark items.
* ``exact_safe``: whether a benchmark may ask about it as a free answer that must match the
  object exactly. Objects that are phrases rather than names ("by road") are too fragile for that.
* ``falsifiable``: whether swapping the object always gives a clearly false statement, which is
  what makes a swapped object a safe false fact.
"""

import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

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
_RELATION_KEYS = (
    "topic",
    "falsifiable",
    "mc_safe",
    "exact_safe",
    "train_templates",
    "bench_templates",
    "question_templates",
    "chat_templates",
    "facts",
)


class KBDataError(ValueError):
    """The knowledge-base files are malformed; the message names the file, relation and key."""


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
    falsifiable: bool = True
    mc_safe: bool = True
    exact_safe: bool = True


class KB:
    """Facts and relations with the lookups generators need. Immutable once built."""

    def __init__(self, relations: Mapping[str, Relation], facts: tuple[Fact, ...]) -> None:
        self.relations: Mapping[str, Relation] = MappingProxyType(dict(relations))
        self.facts: tuple[Fact, ...] = tuple(facts)
        self._truth: dict[tuple[str, str], str] = {}
        self._by_relation: dict[str, list[Fact]] = {name: [] for name in relations}
        self._by_subject: dict[str, list[Fact]] = {}
        for fact in self.facts:
            self._truth[(fact.subject, fact.relation)] = fact.obj
            self._by_relation[fact.relation].append(fact)
            self._by_subject.setdefault(fact.subject, []).append(fact)
        self._objects = {
            name: sorted({f.obj for f in fs}) for name, fs in self._by_relation.items()
        }
        self.subjects: tuple[str, ...] = tuple(self._by_subject)

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
            raise KBDataError(
                f"relation {fact.relation!r} has only {len(pool)} wrong objects, not {k}"
            )
        return [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def _check_template(where: str, template: object, required: set[str], allowed: set[str]) -> None:
    if not isinstance(template, str):
        raise KBDataError(f"{where}: template {template!r} is not a string")
    fields = _fields(template)
    if not required <= fields <= allowed:
        raise KBDataError(
            f"{where}: template {template!r} must use {sorted(required)}, got {sorted(fields)}"
        )


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise KBDataError(f"{where}: {value!r} must be non-empty text without outer spaces")
    return value


def _parse_relation(file: str, name: str, spec: object) -> tuple[Relation, list[Fact]]:
    where = f"{file}.json/{name}"
    if not isinstance(spec, dict):
        raise KBDataError(f"{where}: a relation must be a JSON object")
    for key in _RELATION_KEYS:
        if key not in spec:
            raise KBDataError(f"{where}: missing key {key!r}")
    for key in ("falsifiable", "mc_safe", "exact_safe"):
        if not isinstance(spec[key], bool):
            raise KBDataError(f"{where}: {key!r} must be true or false")
    for key in ("train_templates", "bench_templates", "question_templates", "chat_templates"):
        if not isinstance(spec[key], list) or not spec[key]:
            raise KBDataError(f"{where}: {key!r} must be a non-empty list")
    for t in (*spec["train_templates"], *spec["bench_templates"]):
        _check_template(where, t, {"s", "o"}, {"s", "o"})
    for t in spec["question_templates"]:
        _check_template(where, t, {"s"}, {"s"})
    for pair in spec["chat_templates"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise KBDataError(f"{where}: chat template {pair!r} must be [user, ai]")
        _check_template(where, pair[0], {"s"}, {"s"})
        _check_template(where, pair[1], {"o"}, {"s", "o"})
    if not isinstance(spec["facts"], list):
        raise KBDataError(f"{where}: 'facts' must be a list")
    facts: list[Fact] = []
    for pair in spec["facts"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise KBDataError(f"{where}: fact {pair!r} must be [subject, object]")
        facts.append(Fact(_text(where, pair[0]), name, _text(where, pair[1])))
    seen: dict[str, str] = {}
    for fact in facts:
        if seen.setdefault(fact.subject, fact.obj) != fact.obj:
            raise KBDataError(f"{where}: {fact.subject!r} has two objects")
    if len({f.obj for f in facts}) < MIN_OBJECTS:
        raise KBDataError(f"{where}: needs at least {MIN_OBJECTS} distinct objects")
    relation = Relation(
        name=name,
        train_templates=tuple(spec["train_templates"]),
        bench_templates=tuple(spec["bench_templates"]),
        question_templates=tuple(spec["question_templates"]),
        chat_templates=tuple((user, ai) for user, ai in spec["chat_templates"]),
        topic=_text(where, spec["topic"]),
        falsifiable=spec["falsifiable"],
        mc_safe=spec["mc_safe"],
        exact_safe=spec["exact_safe"],
    )
    return relation, facts


def load_kb_from(directory: Path) -> KB:
    """Build a knowledge base from ``directory``. Raises KBDataError (a ValueError), naming the file and the key."""
    relations: dict[str, Relation] = {}
    facts: list[Fact] = []
    for file in KB_FILES:
        path = directory / f"{file}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise KBDataError(f"{file}.json: not valid JSON ({err})") from err
        if not isinstance(data, dict) or not isinstance(data.get("relations"), dict):
            raise KBDataError(f"{file}.json: missing key 'relations'")
        for name, spec in data["relations"].items():
            if name in relations:
                raise KBDataError(f"relation {name!r} is defined twice")
            relation, relation_facts = _parse_relation(file, name, spec)
            relations[name] = relation
            facts.extend(relation_facts)
    return KB(relations, tuple(facts))


@cache
def load_kb() -> KB:
    """The knowledge base from ``kb_data/`` (loaded once; it cannot be changed)."""
    return load_kb_from(KB_DATA_DIR)
