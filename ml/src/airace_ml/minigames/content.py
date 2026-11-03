"""What each Playground minigame asks (playtest pack §4.1-4.2).

Every game draws its items from the skill generators on their ``battles`` random stream, held-out
keys only, leaving out any prompt the benchmark, the customers or the creativity eval ask (the
battle rule), so a game never shows a scored item. The model never sees an answer: the options
are what it chooses between, and its choice is its own (see ``run.py``).

A run is a ladder: ``items[depth][retry]``. Depth ``d`` draws from difficulty band
``band(d)`` (each game's families ordered easy to hard), so a weak AI can always clear the first
steps; a miss costs a life and the next try at the same depth gets a fresh item. Both sides of a
duel get the same ladder.

Difficulty sets how many options a choice has (2, 3 or 4; the quiz always shows 4 and strikes
out 2 or 1). On Hard, the maze, the bridge and the quiz make the AI type its answer from step 7 on.
Simon Says is always typed; its difficulty starts it higher up the bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from airace_ml.battles import _SITUATIONS, _SUBJECTS, _taken_prompts
from airace_ml.skills.facts import knowledge_bench_items
from airace_ml.skills.instructions import instruction_bench_items
from airace_ml.skills.kb import load_kb
from airace_ml.skills.patterns import CYCLE_WORDS, _bench_pools, _space
from airace_ml.skills.reasoning import reasoning_bench_items
from airace_ml.skills.types import CheckItem, ExactItem, MCItem, skill_rng

GAMES = ("maze", "bridge", "quiz", "story", "simon")
DIFFICULTIES = ("easy", "medium", "hard")
LIVES = 3
MAX_STEPS = {"maze": 10, "bridge": 12, "quiz": 12, "story": 3, "simon": 10}
OPTIONS = {"easy": 2, "medium": 3, "hard": 4}
TYPED_FROM = 6  # on Hard, step 7 on is typed (0-based depth 6)
STORY_WORDS = (
    "key", "map", "hat", "cake", "star", "boat", "song", "door", "tree", "moon", "bell", "ring",
)  # fmt: skip

_MAZE_BANDS = ("syllogism", "compare", "count", "word_problem")
_BRIDGE_BANDS = ("cycle", "step", "countdown", "letters", "double")
_SIMON_BANDS = (
    ("one_word", "yes_no"),
    ("list_n", "repeat_word"),
    ("starts_with", "contains_any"),
    ("all_caps",),
)
_SIMON_START = {"easy": 0, "medium": 1, "hard": 2}
# Easiest first: what children learn early, then geography, then the sky and chemistry.
_QUIZ_ORDER = (
    "animal_legs", "shape_sides", "day_after", "month_after", "vehicle_travel", "baby_animal",
    "animal_class", "continent_of", "capital_of", "landmark_country", "planet_order",
    "element_symbol",
)  # fmt: skip
_COLORS = frozenset(CYCLE_WORDS["colors"])
_ANIMALS = frozenset(CYCLE_WORDS["animals"])


@dataclass(frozen=True)
class GameItem:
    """One step's challenge, as the AI is asked it and as the player sees it."""

    id: str
    band: int
    prompt: str
    chat: bool
    #: What the AI chooses between (None when it types its answer).
    options: tuple[str, ...] | None
    answer_index: int | None
    #: How a typed answer is checked (an ExactItem or a CheckItem); None for a choice or a story.
    check: ExactItem | CheckItem | None
    display: dict[str, Any] = field(default_factory=dict)

    @property
    def typed(self) -> bool:
        return self.options is None


def band(depth: int, bands: int, max_steps: int) -> int:
    """The difficulty band of a depth: the bands share the ladder evenly, easiest first."""
    return min(bands - 1, depth * bands // max_steps)


# -- pools -----------------------------------------------------------------------------------


def _fresh[T](items: list[T]) -> list[T]:
    taken = _taken_prompts()
    return [i for i in items if getattr(i, "prompt", None) not in taken]


@cache
def _reasoning() -> dict[str, tuple[MCItem, ...]]:
    items = _fresh(reasoning_bench_items(skill_rng("reasoning", "battles"), n=400))
    return {fam: tuple(i for i in items if f"fam:{fam}" in i.tags) for fam in _MAZE_BANDS}


@cache
def _knowledge() -> tuple[MCItem, ...]:
    items = _fresh(knowledge_bench_items(load_kb(), skill_rng("knowledge", "battles")))
    return tuple(i for i in items if isinstance(i, MCItem))


def _tag(item: MCItem, prefix: str) -> str:
    return next((t[len(prefix) :] for t in item.tags if t.startswith(prefix)), "")


def quiz_topics() -> list[str]:
    """The topics the Quiz Show can be about (besides "mixed")."""
    return sorted({_tag(i, "topic:") for i in _knowledge()} - {""})


@cache
def _quiz_bands(topic: str) -> tuple[tuple[MCItem, ...], ...]:
    """The topic's questions in 4 bands, easiest relations first."""
    pool = [i for i in _knowledge() if topic in ("mixed", _tag(i, "topic:"))]
    if not pool:
        raise ValueError(f"no quiz questions about {topic!r}; expected one of {quiz_topics()}")
    rank = {r: k for k, r in enumerate(_QUIZ_ORDER)}
    pool.sort(key=lambda i: (rank.get(_tag(i, "rel:"), len(rank)), i.id))
    cut = [len(pool) * k // 4 for k in range(5)]
    return tuple(tuple(pool[cut[k] : max(cut[k + 1], cut[k] + 1)]) for k in range(4))


PATTERNS_PER_FAMILY = 40


@cache
def _patterns() -> dict[str, tuple[tuple[tuple[str, ...], str, str], ...]]:
    """(shown terms, answer, prompt) per family: held-out keys first, never a benchmark prompt.

    The benchmark uses most held-out patterns of the small families (letters, double,
    countdown), so a family with fewer than ``PATTERNS_PER_FAMILY`` is topped up with ordinary
    practice patterns, spread evenly over its space. A game is not a measurement; it only must
    never show a scored item.
    """
    taken = _taken_prompts()
    out = {}
    for fam, pool in _bench_pools().items():
        fresh = [(p.shown, p.answer, p.key) for p in pool if p.key not in taken]
        if len(fresh) < PATTERNS_PER_FAMILY:
            reserved = {p.key for p in pool}
            others = [p for p in _space()[fam] if p.key not in reserved and p.key not in taken]
            need = min(len(others), PATTERNS_PER_FAMILY - len(fresh))
            stride = max(1, len(others) // max(1, need))
            fresh += [(p.shown, p.answer, p.key) for p in others[::stride][:need]]
        out[fam] = tuple(fresh)
    return out


@cache
def _instructions() -> tuple[CheckItem, ...]:
    items = _fresh(instruction_bench_items(load_kb(), skill_rng("instruction", "battles")))
    return tuple(i for i in items if isinstance(i, CheckItem))


# -- options ---------------------------------------------------------------------------------


def _picture(value: str) -> dict[str, str]:
    """How the client draws a pattern term: a colored disc, an animal, a number or a letter."""
    if value in _COLORS:
        return {"kind": "color", "value": value}
    if value in _ANIMALS:
        return {"kind": "animal", "value": value}
    if value.lstrip("-").isdigit():
        return {"kind": "number", "value": value}
    return {"kind": "letter", "value": value}


def _pattern_distractors(shown: tuple[str, ...], answer: str) -> list[str]:
    if answer.lstrip("-").isdigit():
        a, last = int(answer), int(shown[-1])
        step = a - last
        near = {a + 1, a - 1, a + 2, a - 2, last, a + step, a - 2 * step, 2 * a, a // 2}
        return sorted(str(x) for x in near if x >= 0 and x != a)
    if len(answer) == 1 and answer.isalpha():
        code = ord(answer)
        near = {chr(code + d) for d in (-2, -1, 1, 2) if "A" <= chr(code + d) <= "Z"}
        return sorted(near)
    kind = _COLORS if answer in _COLORS else _ANIMALS
    seen = [v for v in dict.fromkeys(shown) if v != answer]
    others = sorted(kind - set(shown))
    return seen + others


def _choice(
    rng: np.random.Generator, answer: str, distractors: list[str], n: int
) -> tuple[tuple[str, ...], int]:
    """``answer`` and ``n - 1`` of the distractors (the first ones preferred), shuffled."""
    pool = list(dict.fromkeys(d for d in distractors if d != answer))
    head = pool[: max(0, n - 1)] if len(pool) <= n - 1 else _prefer_first(rng, pool, n - 1)
    options = [answer, *head]
    order = rng.permutation(len(options))
    shuffled = tuple(options[int(k)] for k in order)
    return shuffled, shuffled.index(answer)


def _prefer_first(rng: np.random.Generator, pool: list[str], k: int) -> list[str]:
    """``k`` of ``pool``, drawn with the earlier ones (the more tempting) likelier."""
    weights = np.array([1.0 / (1 + i) for i in range(len(pool))])
    picks = rng.choice(len(pool), size=k, replace=False, p=weights / weights.sum())
    return [pool[int(i)] for i in picks]


def _typed_exact(item_id: str, prompt: str, answer: str, extract: str) -> ExactItem:
    return ExactItem(item_id, "minigame", prompt, [answer], (), False, 8, extract)  # type: ignore[arg-type]


# -- the games -------------------------------------------------------------------------------


def _draw[T](rng: np.random.Generator, pool: tuple[T, ...] | list[T], used: set[int]) -> T:
    """A random member of ``pool`` not drawn before in this run (any once all were)."""
    left = [k for k in range(len(pool)) if id(pool[k]) not in used] or list(range(len(pool)))
    item = pool[left[int(rng.integers(len(left)))]]
    used.add(id(item))
    return item


def _maze(rng, difficulty: str, depth: int, k: str, used: set[int]) -> GameItem:
    b = band(depth, len(_MAZE_BANDS), MAX_STEPS["maze"])
    item = _draw(rng, _reasoning()[_MAZE_BANDS[b]], used)
    question = item.prompt.removeprefix("Question: ").removesuffix("\nAnswer:")
    answer = item.options[item.answer_index]
    display = {"kind": "maze", "family": _MAZE_BANDS[b], "question": question}
    if difficulty == "hard" and depth >= TYPED_FROM:
        check = _typed_exact(k, item.prompt, answer, "first_line")
        return GameItem(k, b, item.prompt, False, None, None, check, {**display, "answer": answer})
    distractors = [o for o in item.options if o != answer]
    options, idx = _choice(rng, answer, distractors, min(OPTIONS[difficulty], len(item.options)))
    return GameItem(k, b, item.prompt, False, options, idx, None, display)


def _bridge(rng, difficulty: str, depth: int, k: str, used: set[int]) -> GameItem:
    b = band(depth, len(_BRIDGE_BANDS), MAX_STEPS["bridge"])
    fam = _BRIDGE_BANDS[b]
    shown, answer, prompt = _draw(rng, _patterns()[fam], used)
    display = {"kind": "bridge", "family": fam, "terms": [_picture(t) for t in shown]}
    if difficulty == "hard" and depth >= TYPED_FROM:
        check = _typed_exact(k, prompt, answer, "first_item")
        return GameItem(
            k, b, prompt, False, None, None, check, {**display, "answer": _picture(answer)}
        )
    options, idx = _choice(rng, answer, _pattern_distractors(shown, answer), OPTIONS[difficulty])
    display["options"] = [_picture(o) for o in options]
    return GameItem(k, b, prompt, False, options, idx, None, display)


def _quiz(rng, difficulty: str, depth: int, k: str, topic: str, used: set[int]) -> GameItem:
    bands = _quiz_bands(topic)
    b = band(depth, len(bands), MAX_STEPS["quiz"])
    item = _draw(rng, bands[b], used)
    answer = item.options[item.answer_index]
    question = item.prompt.removeprefix("Question: ").removesuffix("\nAnswer:")
    display = {"kind": "quiz", "topic": _tag(item, "topic:"), "question": question, "struck": []}
    if difficulty == "hard" and depth >= TYPED_FROM:
        check = _typed_exact(k, item.prompt, answer, "first_line")
        return GameItem(k, b, item.prompt, False, None, None, check, {**display, "answer": answer})
    # The AI chooses between the options left after the strike-outs; all four are shown.
    kept, idx = _choice(rng, answer, [o for o in item.options if o != answer], OPTIONS[difficulty])
    display["shown"] = list(item.options)
    display["struck"] = [o for o in item.options if o not in kept]
    return GameItem(k, b, item.prompt, False, kept, idx, None, display)


def _simon(rng, difficulty: str, depth: int, k: str, used: set[int]) -> GameItem:
    b = min(
        len(_SIMON_BANDS) - 1,
        _SIMON_START[difficulty] + band(depth, len(_SIMON_BANDS), MAX_STEPS["simon"]),
    )
    pool = [i for i in _instructions() if i.check in _SIMON_BANDS[b]]
    item = _draw(rng, pool, used)
    display = {"kind": "simon", "check": item.check, "instruction": item.prompt}
    return GameItem(k, b, item.prompt, item.chat, None, None, item, display)


def story_prompt(hero: int, situation: int, word: int | None) -> str:
    text = f"Write a short story about {_SUBJECTS[hero]} {_SITUATIONS[situation]}."
    return text if word is None else f'{text} Use the word "{STORY_WORDS[word]}".'


def _story(rng, depth: int, k: str, settings: dict[str, Any]) -> GameItem:
    rolls = settings.get("rolls")
    if isinstance(rolls, list) and depth < len(rolls) and rolls[depth] is not None:
        hero, situation, word = rolls[depth]
    else:
        hero = int(rng.integers(len(_SUBJECTS)))
        situation = int(rng.integers(len(_SITUATIONS)))
        word = int(rng.integers(len(STORY_WORDS))) if settings.get("rule", True) else None
    for value, size in ((hero, len(_SUBJECTS)), (situation, len(_SITUATIONS))):
        if not isinstance(value, int) or not 0 <= value < size:
            raise ValueError("a story die shows a face it doesn't have")
    if word is not None and not (isinstance(word, int) and 0 <= word < len(STORY_WORDS)):
        raise ValueError("a story die shows a face it doesn't have")
    display = {
        "kind": "story",
        "hero": _SUBJECTS[hero],
        "situation": _SITUATIONS[situation],
        "word": None if word is None else STORY_WORDS[word],
        "faces": [hero, situation, word],
    }
    return GameItem(k, 0, story_prompt(hero, situation, word), True, None, None, None, display)


def story_faces() -> dict[str, list[str]]:
    """Every face of the three Story Dice, for the client's picker."""
    return {"hero": list(_SUBJECTS), "situation": list(_SITUATIONS), "word": list(STORY_WORDS)}


def game_items(
    game: str, difficulty: str, seed: int, settings: dict[str, Any] | None = None
) -> list[list[GameItem]]:
    """The run's ladder ``items[depth][retry]`` (``LIVES`` tries per depth; one per story)."""
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r}; expected one of {GAMES}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}; expected one of {DIFFICULTIES}")
    settings = settings or {}
    rng = np.random.default_rng([seed, GAMES.index(game), DIFFICULTIES.index(difficulty)])
    ladder: list[list[GameItem]] = []
    used: set[int] = set()
    for depth in range(MAX_STEPS[game]):
        row = []
        for retry in range(1 if game == "story" else LIVES):
            k = f"{game}-{depth}-{retry}"
            if game == "maze":
                row.append(_maze(rng, difficulty, depth, k, used))
            elif game == "bridge":
                row.append(_bridge(rng, difficulty, depth, k, used))
            elif game == "quiz":
                topic = str(settings.get("topic", "mixed"))
                row.append(_quiz(rng, difficulty, depth, k, topic, used))
            elif game == "simon":
                row.append(_simon(rng, difficulty, depth, k, used))
            else:
                row.append(_story(rng, depth, k, settings))
        ladder.append(row)
    return ladder
