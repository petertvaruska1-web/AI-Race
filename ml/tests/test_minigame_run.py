import contextlib
import itertools
import math

import pytest

from airace_ml.infer.lm import ContinuationScore
from airace_ml.minigames.content import LIVES, MAX_STEPS, STORY_WORDS, game_items
from airace_ml.minigames.run import choose, play


class FakeLM:
    """Prefers options by a rule over their text; types a fixed reply or the right answer."""

    def __init__(self, tok, prefer, typed=lambda prompt: "") -> None:
        self.tok, self.prefer, self.typed = tok, prefer, typed
        self.asked: list[str] = []

    def score_continuations(self, contexts, continuations):
        out = []
        for ctx, cont in zip(contexts, continuations, strict=True):
            text, asked = self.tok.decode(cont).strip(), self.tok.decode(ctx)
            score = self.prefer(text, asked)
            out.append(ContinuationScore(score * max(1, len(cont)), max(1, len(cont))))
        return out

    def stream_reply(self, prompt, *, chat, max_new_tokens, temperature, top_p, seed, on_text):
        self.asked.append(prompt)
        reply = self.typed(prompt)
        for k in range(0, len(reply), 3):
            on_text(reply[k : k + 3])
        return reply


def _answers(ladder):
    """prompt -> the right answer, for an oracle."""
    out = {}
    for row in ladder:
        for item in row:
            if item.options is not None:
                out[item.prompt] = item.options[item.answer_index]
            elif item.check is not None and hasattr(item.check, "answers"):
                out[item.prompt] = item.check.answers[0]
            elif item.check is not None:
                out[item.prompt] = item.check.reference
    return out


def _oracle(tok, ladder):
    answers = _answers(ladder)

    def prefer(text, asked):
        right = next((a for p, a in answers.items() if p.strip() in asked), None)
        return 0.0 if text == right else -5.0

    return FakeLM(tok, prefer, lambda p: answers.get(p, ""))


def _alphabetical(tok):
    return FakeLM(tok, lambda text, asked: -float(sum(map(ord, text[:3]))) / 100, lambda p: "")


def _sides(**lms):
    return {side: (lm, contextlib.nullcontext()) for side, lm in lms.items()}


def test_a_choice_is_the_option_the_model_finds_most_likely_and_its_sureness_sums_to_one(tiny_tok):
    item = next(i for row in game_items("quiz", "hard", 1) for i in row if not i.typed)
    lm = FakeLM(tiny_tok, lambda text, asked: 0.0 if text == item.options[2] else -3.0)
    index, sure = choose(lm, tiny_tok, item)
    assert index == 2
    assert math.isclose(sum(sure), 1.0, rel_tol=1e-6)
    assert sure[2] == max(sure) and len(sure) == len(item.options)


def test_a_run_ends_after_three_misses(tiny_tok):
    ladder = game_items("maze", "hard", 3)
    result = play("maze", ladder, _sides(a=_alphabetical(tiny_tok)))
    a = result["a"]
    misses = [s for s in a["steps"] if not s["correct"]]
    assert len(misses) == LIVES or a["score"] == MAX_STEPS["maze"]
    assert len(a["steps"]) <= MAX_STEPS["maze"] + LIVES - 1
    assert a["score"] == sum(s["correct"] for s in a["steps"])
    assert a["maxScore"] == MAX_STEPS["maze"]


def test_an_ai_that_always_knows_clears_every_step_typed_ones_too(tiny_tok):
    for game in ("maze", "bridge", "quiz"):
        ladder = game_items(game, "hard", 4)
        a = play(game, ladder, _sides(a=_oracle(tiny_tok, ladder)))["a"]
        assert a["score"] == MAX_STEPS[game], game
        typed = [s for s in a["steps"] if s["typed"]]
        assert typed and all(s["reply"] and s["depth"] >= 6 for s in typed), game


def test_a_miss_tries_a_fresh_item_at_the_same_depth(tiny_tok):
    ladder = game_items("bridge", "easy", 5)
    steps = play("bridge", ladder, _sides(a=_alphabetical(tiny_tok)))["a"]["steps"]
    for prev, nxt in itertools.pairwise(steps):
        if not prev["correct"]:
            assert (nxt["depth"], nxt["retry"]) == (prev["depth"], prev["retry"] + 1)
        else:
            assert (nxt["depth"], nxt["retry"]) == (prev["depth"] + 1, 0)


def test_a_duel_asks_both_ais_the_same_items_and_streams_them_in_turn(tiny_tok):
    ladder = game_items("quiz", "medium", 6)
    seen = []
    result = play(
        "quiz",
        ladder,
        _sides(a=_oracle(tiny_tok, ladder), b=_alphabetical(tiny_tok)),
        on_step=lambda side, step: seen.append((side, step["itemId"])),
    )
    assert result["a"]["score"] == MAX_STEPS["quiz"]
    assert result["b"]["score"] < result["a"]["score"]
    a_items = {s["itemId"] for s in result["a"]["steps"]}
    assert all(s["itemId"] in a_items for s in result["b"]["steps"] if s["retry"] == 0)
    assert seen[0][0] == "a" and seen[1][0] == "b"


def test_simon_says_checks_the_typed_reply(tiny_tok):
    ladder = game_items("simon", "easy", 7)
    good = play("simon", ladder, _sides(a=_oracle(tiny_tok, ladder)))["a"]
    assert good["score"] == MAX_STEPS["simon"]
    bad = play("simon", ladder, _sides(a=_alphabetical(tiny_tok)))["a"]
    assert bad["score"] == 0 and len(bad["steps"]) == LIVES


def test_story_dice_score_the_audience_and_the_rule(tiny_tok):
    rolls = [[0, 0, 0], [1, 1, None], [2, 2, 1]]
    ladder = game_items("story", "easy", 0, {"rolls": rolls})
    word = STORY_WORDS[0]
    lm = FakeLM(tiny_tok, lambda t, a: 0.0, lambda p: f"The fox found a {word}. The end.")
    a = play("story", ladder, _sides(a=lm), story_score=lambda story, prompt: 0.5)["a"]
    assert a["maxScore"] == 39
    assert [s["points"] for s in a["steps"]] == [8.0, 5.0, 5.0]
    assert a["score"] == 18.0
    assert [s["ruleMet"] for s in a["steps"]] == [True, None, False]


def test_a_story_game_needs_the_judge(tiny_tok):
    ladder = game_items("story", "easy", 0)
    with pytest.raises(ValueError, match="judge"):
        play("story", ladder, _sides(a=_alphabetical(tiny_tok)))


def test_typed_text_streams_and_adds_up_to_the_reply(tiny_tok):
    ladder = game_items("simon", "easy", 8)
    pieces: dict[tuple[str, int], str] = {}

    def text(side, attempt, piece):
        pieces[(side, attempt)] = pieces.get((side, attempt), "") + piece

    a = play("simon", ladder, _sides(a=_oracle(tiny_tok, ladder)), on_text=text)["a"]
    for k, s in enumerate(a["steps"]):
        assert pieces.get(("a", k), "") == s["reply"]
