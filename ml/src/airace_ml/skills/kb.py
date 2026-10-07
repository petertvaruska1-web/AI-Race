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

Three more lists say which other objects count as right, or as not clearly wrong:

* ``confusable``: groups of objects that can be mistaken for each other (``pup`` and ``puppy``).
  A wrong option is never in a group with the true object, and two options are never in the same
  group.
* ``also_accepted``: extra spellings or forms a free answer may use for an object (``pup`` for
  ``puppy``, ``4`` for ``four``, ``Kiev`` for ``Kyiv``).
* ``never_false``: (subject, object) pairs that are not clearly false (some lizards have no legs),
  so they are never offered as a wrong option or used as a false fact.
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
    "confusable",
    "also_accepted",
    "never_false",
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
    confusable: tuple[tuple[str, ...], ...] = ()
    also_accepted: tuple[tuple[str, tuple[str, ...]], ...] = ()
    never_false: tuple[tuple[str, str], ...] = ()


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
        self._confusable: dict[str, dict[str, set[str]]] = {}
        self._accepted: dict[str, dict[str, tuple[str, ...]]] = {}
        self._never_false: dict[tuple[str, str], set[str]] = {}
        for name, relation in self.relations.items():
            mates: dict[str, set[str]] = {}
            for group in relation.confusable:
                for obj in group:
                    mates.setdefault(obj, set()).update(o for o in group if o != obj)
            self._confusable[name] = mates
            self._accepted[name] = dict(relation.also_accepted)
            for subject, obj in relation.never_false:
                self._never_false.setdefault((subject, name), set()).add(obj)
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

    def accepted_answers(self, fact: Fact) -> list[str]:
        """The true object first, then the other forms a free answer may use for it."""
        return [fact.obj, *self._accepted[fact.relation].get(fact.obj, ())]

    def wrong_objects(self, fact: Fact) -> list[str]:
        """The objects of the relation that are clearly wrong for the fact.

        Not the true object, not one that can be mistaken for it or is accepted in its place, and
        not one listed in the relation's ``never_false`` pairs for the subject.
        """
        banned = {fact.obj}
        banned |= self._confusable[fact.relation].get(fact.obj, set())
        banned |= set(self._accepted[fact.relation].get(fact.obj, ()))
        banned |= self._never_false.get((fact.subject, fact.relation), set())
        return [o for o in self._objects[fact.relation] if o not in banned]

    def distractors(self, fact: Fact, k: int, rng: np.random.Generator) -> list[str]:
        """``k`` distinct clearly wrong answers for the fact, from the other objects of its relation.

        Two answers that can be mistaken for each other are never both chosen. The subject itself
        is left out when enough other objects remain (an answer that repeats the question is no
        distraction).
        """
        mates = self._confusable[fact.relation]
        pool = self.wrong_objects(fact)
        for candidates in ([o for o in pool if o != fact.subject], pool):
            chosen: list[str] = []
            for i in rng.permutation(len(candidates)):
                option = candidates[int(i)]
                if not any(option in mates.get(c, ()) for c in chosen):
                    chosen.append(option)
                if len(chosen) == k:
                    return chosen
        raise ValueError(f"relation {fact.relation!r} has too few wrong objects for {k} options")


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


def _confusable_groups(where: str, raw: object, objects: set[str]) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list):
        raise KBDataError(f"{where}: 'confusable' must be a list of groups")
    groups = []
    for group in raw:
        if not isinstance(group, list) or len(group) < 2 or len(set(group)) != len(group):
            raise KBDataError(f"{where}: confusable group {group!r} needs 2+ different objects")
        unknown = [o for o in group if o not in objects]
        if unknown:
            raise KBDataError(f"{where}: confusable group {group!r} has unknown objects {unknown}")
        groups.append(tuple(group))
    return tuple(groups)


def _also_accepted(
    where: str, raw: object, objects: set[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(raw, dict):
        raise KBDataError(f"{where}: 'also_accepted' must be an object")
    accepted = []
    for obj, extras in raw.items():
        if obj not in objects:
            raise KBDataError(f"{where}: also_accepted key {obj!r} is not an object")
        if not isinstance(extras, list) or not extras:
            raise KBDataError(f"{where}: also_accepted for {obj!r} must be a non-empty list")
        forms = tuple(_text(where, extra) for extra in extras)
        if obj in forms or len(set(forms)) != len(forms):
            raise KBDataError(f"{where}: also_accepted for {obj!r} repeats a form")
        accepted.append((obj, forms))
    return tuple(accepted)


def _never_false(
    where: str, raw: object, truth: dict[str, str], objects: set[str]
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise KBDataError(f"{where}: 'never_false' must be a list of [subject, object]")
    pairs = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise KBDataError(f"{where}: never_false entry {pair!r} must be [subject, object]")
        subject, obj = pair
        if subject not in truth or obj not in objects or truth[subject] == obj:
            raise KBDataError(
                f"{where}: never_false entry {pair!r} must name a subject and a wrong object"
            )
        pairs.append((subject, obj))
    return tuple(pairs)


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
    objects = {f.obj for f in facts}
    truth = {f.subject: f.obj for f in facts}
    confusable = _confusable_groups(where, spec["confusable"], objects)
    also_accepted = _also_accepted(where, spec["also_accepted"], objects)
    never_false = _never_false(where, spec["never_false"], truth, objects)
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
        confusable=confusable,
        also_accepted=also_accepted,
        never_false=never_false,
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
