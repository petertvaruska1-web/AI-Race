"""Playing a minigame with real AIs (playtest pack §4.1).

Every move is the model's own (R1):
- **A choice** is the option it finds most likely after the prompt: the highest mean
  log-probability per token, as the benchmark decides multiple-choice items. Its sureness is the
  softmax of those means.
- **A typed answer** is its own greedy reply, checked as the benchmark checks its kind.
- **A story** is its own sampled reply, scored by the judge.

Nothing is rewritten; the reply text travels to the player exactly as generated.

A run climbs ``items[depth][retry]``. A right answer goes one step deeper; a miss costs a life and
tries a fresh item at the same depth. The run ends when the lives or the steps run out. In a duel,
both AIs take their turns one attempt at a time, on the same ladder.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from airace_ml.evals.scoring import exact_match, mean_logprob
from airace_ml.evals.suite import _prompt_ids
from airace_ml.minigames.content import LIVES, MAX_STEPS, GameItem
from airace_ml.skills.checkers import CHECKERS
from airace_ml.skills.types import ExactItem

STORY_POINTS, RULE_POINTS = 10.0, 3.0
STORY_MAX = MAX_STEPS["story"] * (STORY_POINTS + RULE_POINTS)

type Side = tuple[Any, contextlib.AbstractContextManager]  # (model, its lock)
type StepFn = Callable[[str, dict[str, Any]], None]
type TextFn = Callable[[str, int, str], None]


def choose(lm: Any, tok: Any, item: GameItem) -> tuple[int, list[float]]:
    """The option the model finds most likely, and how sure it is of each (summing to 1)."""
    assert item.options is not None
    context = _prompt_ids(tok, item.prompt, item.chat)
    lead = "" if item.chat else " "
    scores = lm.score_continuations(
        [context] * len(item.options), [tok.encode(lead + o) for o in item.options]
    )
    means = [mean_logprob(s) for s in scores]
    choice = means.index(max(means))
    finite = [m if math.isfinite(m) else -1e9 for m in means]
    top = max(finite)
    weights = [math.exp(m - top) for m in finite]
    total = sum(weights)
    return choice, [w / total for w in weights]


def _typed_passes(item: GameItem, reply: str) -> bool:
    check = item.check
    if isinstance(check, ExactItem):
        return exact_match(reply, check.answers, check.extract)
    assert check is not None
    return bool(CHECKERS[check.check](reply, check.check_args))


def _uses_word(story: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", story, re.IGNORECASE) is not None


def play(
    game: str,
    ladder: list[list[GameItem]],
    sides: Mapping[str, Side],
    *,
    story_score: Callable[[str, str], float] | None = None,
    seed: int = 0,
    on_step: StepFn | None = None,
    on_text: TextFn | None = None,
) -> dict[str, dict[str, Any]]:
    """Plays the ladder with every side's AI; returns ``{side: {steps, score, maxScore}}``."""
    if game == "story" and story_score is None:
        raise ValueError("Story Dice need the judge and the novelty index to score stories")
    state = {s: {"depth": 0, "retry": 0, "lives": LIVES, "steps": [], "score": 0.0} for s in sides}
    max_steps = len(ladder)
    attempt = 0
    while True:
        playing = [s for s in sorted(sides) if _going(game, state[s], max_steps)]
        if not playing:
            break
        for side in playing:
            st = state[side]
            item = ladder[st["depth"]][st["retry"]]
            lm, lock = sides[side]
            k = len(st["steps"])
            with lock:
                step = _attempt(
                    lm,
                    item,
                    seed + attempt,
                    story_score,
                    lambda t, s=side, k=k: _emit(on_text, s, k, t),
                )
            step |= {"attempt": k, "depth": st["depth"], "retry": st["retry"]}
            if game == "story":
                st["score"] += step["points"]
                st["depth"] += 1
            elif step["correct"]:
                st["score"] += 1
                st["depth"] += 1
                st["retry"] = 0
            else:
                st["lives"] -= 1
                st["retry"] += 1
            step["lives"] = st["lives"]
            st["steps"].append(step)
            if on_step is not None:
                on_step(side, step)
            attempt += 1
    max_score = STORY_MAX if game == "story" else max_steps
    return {
        s: {"steps": st["steps"], "score": st["score"], "maxScore": max_score}
        for s, st in state.items()
    }


def _going(game: str, st: dict[str, Any], max_steps: int) -> bool:
    if st["depth"] >= max_steps:
        return False
    return game == "story" or st["lives"] > 0


def _emit(on_text: TextFn | None, side: str, attempt: int, text: str) -> None:
    if on_text is not None:
        on_text(side, attempt, text)


def _attempt(
    lm: Any,
    item: GameItem,
    seed: int,
    story_score: Callable[[str, str], float] | None,
    on_text: Callable[[str], None],
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "itemId": item.id,
        "band": item.band,
        "typed": item.typed,
        "prompt": item.prompt,
        "display": item.display,
        "options": list(item.options) if item.options is not None else None,
        "answerIndex": item.answer_index,
        "choice": None,
        "sure": None,
        "reply": None,
        "points": None,
        "ruleMet": None,
    }
    if not item.typed:
        choice, sure = choose(lm, lm.tok, item)
        return step | {"choice": choice, "sure": sure, "correct": choice == item.answer_index}
    if item.check is None:  # a story
        assert story_score is not None
        reply = lm.stream_reply(
            item.prompt, chat=True, max_new_tokens=96, temperature=0.9, top_p=0.95, seed=seed,
            on_text=on_text,
        )  # fmt: skip
        word = item.display.get("word")
        rule = None if word is None else _uses_word(reply, word)
        audience = float(story_score(reply, item.prompt))
        points = round(STORY_POINTS * audience + (RULE_POINTS if rule else 0.0), 3)
        return step | {
            "reply": reply,
            "points": points,
            "ruleMet": rule,
            "audience": audience,
            "correct": True,
        }
    reply = lm.stream_reply(
        item.prompt,
        chat=item.chat,
        max_new_tokens=item.check.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
        on_text=on_text,
    )
    return step | {"reply": reply, "correct": _typed_passes(item, reply)}
