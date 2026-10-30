import pytest

from airace_ml.battles import (
    BATTLE_SKILLS,
    StoryPrompt,
    battle_challenges,
    score_challenge,
)
from airace_ml.evals.creativity import STORY_PROMPTS
from airace_ml.evals.suite import build_suite
from airace_ml.feedback import GROUPS, customer_pool
from airace_ml.skills.types import CheckItem, ExactItem


def test_the_same_seed_gives_the_same_challenges_and_another_seed_others():
    for skill in BATTLE_SKILLS:
        a = battle_challenges(skill, 5, seed=1)
        assert [c.id for c in a] == [c.id for c in battle_challenges(skill, 5, seed=1)], skill
        assert [c.id for c in a] != [c.id for c in battle_challenges(skill, 5, seed=2)], skill
        assert len({c.id for c in a}) == 5, skill


def test_challenges_are_fresh_never_benchmark_customer_or_story_prompts():
    seen = {getattr(i, "prompt", None) for items in build_suite().items.values() for i in items}
    seen |= {q.prompt for g in GROUPS for q in customer_pool(g)}
    seen |= set(STORY_PROMPTS)
    for skill in BATTLE_SKILLS:
        for seed in range(5):
            for c in battle_challenges(skill, 5, seed=seed):
                assert c.prompt not in seen, (skill, c.prompt)
                assert c.id.startswith(f"{skill}-battle-")


def test_kinds_of_challenge():
    assert all(isinstance(c, StoryPrompt) for c in battle_challenges("creativity", 3, seed=0))
    for skill in ("knowledge", "coding", "instruction", "pattern"):
        assert all(isinstance(c, (ExactItem, CheckItem)) for c in battle_challenges(skill, 3, 0))


def test_an_unknown_skill_is_refused():
    with pytest.raises(ValueError):
        battle_challenges("language", 3, seed=0)


def test_a_checked_challenge_is_scored_by_its_check():
    for skill in ("knowledge", "coding", "instruction", "pattern"):
        item = battle_challenges(skill, 1, seed=3)[0]
        right = item.answers[0] if isinstance(item, ExactItem) else item.reference
        assert score_challenge(item, right) == 1.0, skill
        assert score_challenge(item, "") == 0.0, skill


def test_a_story_needs_the_judge_and_the_novelty_index():
    story = battle_challenges("creativity", 1, seed=0)[0]
    with pytest.raises(ValueError, match="judge"):
        score_challenge(story, "Once upon a time.")
