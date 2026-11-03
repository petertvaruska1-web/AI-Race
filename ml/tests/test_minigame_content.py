import pytest

from airace_ml.evals.creativity import STORY_PROMPTS
from airace_ml.evals.suite import build_suite
from airace_ml.feedback import GROUPS, customer_pool
from airace_ml.minigames.content import (
    DIFFICULTIES,
    GAMES,
    LIVES,
    MAX_STEPS,
    STORY_WORDS,
    game_items,
    quiz_topics,
    story_faces,
)

CHOICE_GAMES = ("maze", "bridge", "quiz")


def _all(game, difficulty="medium", seed=0, settings=None):
    return [i for row in game_items(game, difficulty, seed, settings) for i in row]


def test_a_ladder_has_a_row_per_step_and_a_try_per_life():
    for game in GAMES:
        ladder = game_items(game, "easy", 1)
        assert len(ladder) == MAX_STEPS[game], game
        assert all(len(row) == (1 if game == "story" else LIVES) for row in ladder), game


def test_items_never_repeat_a_benchmark_customer_or_story_prompt():
    seen = {getattr(i, "prompt", None) for items in build_suite().items.values() for i in items}
    seen |= {q.prompt for g in GROUPS for q in customer_pool(g)}
    seen |= set(STORY_PROMPTS)
    for game in GAMES:
        for seed in range(3):
            for item in _all(game, "hard", seed):
                assert item.prompt not in seen, (game, item.prompt)


def test_bands_rise_with_the_depth():
    for game in ("maze", "bridge", "quiz", "simon"):
        bands = [row[0].band for row in game_items(game, "easy", 2)]
        assert bands == sorted(bands), game
        assert bands[0] == 0 and bands[-1] > 0, game


def test_the_maze_starts_with_yes_or_no_puzzles_and_ends_with_word_problems():
    ladder = game_items("maze", "medium", 3)
    assert ladder[0][0].display["family"] == "syllogism"
    assert ladder[-1][0].display["family"] == "word_problem"


def test_difficulty_sets_the_number_of_options():
    for game in CHOICE_GAMES:
        for difficulty, n in zip(DIFFICULTIES, (2, 3, 4), strict=True):
            for item in _all(game, difficulty, 4)[: 6 * LIVES]:
                if item.display.get("family") == "syllogism":
                    assert len(item.options) == 2  # yes or no
                    continue
                assert len(item.options) == n, (game, difficulty, item.prompt)
                assert 0 <= item.answer_index < n


def test_the_quiz_always_shows_four_and_strikes_out_the_rest():
    for difficulty, struck in zip(DIFFICULTIES, (2, 1, 0), strict=True):
        for item in _all("quiz", difficulty, 5)[: 6 * LIVES]:
            assert len(item.display["shown"]) == 4
            assert len(item.display["struck"]) == struck
            assert set(item.options) | set(item.display["struck"]) == set(item.display["shown"])


def test_hard_makes_the_ai_type_its_answer_from_step_seven():
    for game in CHOICE_GAMES:
        ladder = game_items(game, "hard", 6)
        assert all(not row[0].typed for row in ladder[:6]), game
        late = [i for row in ladder[6:] for i in row]
        assert all(i.typed and i.check is not None for i in late), game
        assert all(not i.typed for i in _all(game, "medium", 6)), game


def test_pattern_options_hold_the_answer_once_and_show_as_pictures():
    for item in _all("bridge", "hard", 7):
        if item.typed:
            continue
        assert len(set(item.options)) == len(item.options)
        answer = item.options[item.answer_index]
        assert item.prompt.startswith("Next: ")
        assert answer not in item.prompt.split(", ")[-1:]
        kinds = {t["kind"] for t in item.display["terms"]}
        assert kinds <= {"color", "animal", "number", "letter"}
    cycles = [i for i in _all("bridge", "easy", 8) if i.display["family"] == "cycle"]
    assert cycles and all(t["kind"] in ("color", "animal") for t in cycles[0].display["terms"])


def test_the_same_seed_gives_the_same_ladder_and_another_seed_another():
    for game in GAMES:
        a = [i.prompt for i in _all(game, "medium", 9)]
        assert a == [i.prompt for i in _all(game, "medium", 9)], game
        assert a != [i.prompt for i in _all(game, "medium", 10)], game


def test_story_dice_ask_for_a_story_with_the_chosen_faces_and_rule():
    faces = story_faces()
    rolls = [[0, 1, 2], [3, 4, None], [5, 6, 0]]
    ladder = game_items("story", "easy", 0, {"rolls": rolls})
    first = ladder[0][0]
    assert first.prompt == (
        f"Write a short story about {faces['hero'][0]} {faces['situation'][1]}. "
        f'Use the word "{STORY_WORDS[2]}".'
    )
    assert first.chat and first.typed and first.check is None
    assert "Use the word" not in ladder[1][0].prompt
    assert ladder[1][0].display["word"] is None
    with pytest.raises(ValueError):
        game_items("story", "easy", 0, {"rolls": [[99, 0, 0]]})


def test_simon_says_asks_checked_instructions_and_hard_starts_harder():
    easy = [row[0] for row in game_items("simon", "easy", 1)]
    hard = [row[0] for row in game_items("simon", "hard", 1)]
    assert all(i.typed and i.check is not None and i.chat for i in easy)
    assert easy[0].check.check in ("one_word", "yes_no")
    assert hard[0].band > easy[0].band


def test_a_quiz_can_be_about_one_topic():
    topics = quiz_topics()
    assert len(topics) >= 3
    for item in _all("quiz", "easy", 1, {"topic": topics[0]}):
        assert item.display["topic"] == topics[0]
    with pytest.raises(ValueError):
        game_items("quiz", "easy", 1, {"topic": "astrology"})


def test_unknown_games_and_difficulties_are_refused():
    with pytest.raises(ValueError):
        game_items("chess", "easy", 0)
    with pytest.raises(ValueError):
        game_items("maze", "extreme", 0)


def test_every_bridge_family_has_enough_patterns_and_a_run_never_repeats_one():
    from airace_ml.minigames.content import PATTERNS_PER_FAMILY, _patterns

    assert all(len(pool) >= PATTERNS_PER_FAMILY for pool in _patterns().values())
    for game in ("maze", "bridge", "quiz"):
        prompts = [i.prompt for i in _all(game, "easy", 11)]
        assert len(set(prompts)) == len(prompts), game
