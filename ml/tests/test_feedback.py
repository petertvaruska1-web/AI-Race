from airace_ml.evals.suite import build_suite, score_reply
from airace_ml.feedback import GROUP_CATEGORIES, GROUPS, customer_pool, pick_questions
from airace_ml.skills.types import CheckItem, ExactItem


def test_every_group_has_at_least_50_checked_questions_apart_from_the_benchmark():
    bench_ids = {item.id for items in build_suite().items.values() for item in items}
    assert set(GROUP_CATEGORIES) == set(GROUPS)
    for group in GROUPS:
        pool = customer_pool(group)
        assert len(pool) >= 50, group
        assert all(isinstance(item, (ExactItem, CheckItem)) for item in pool)
        assert all(item.category in GROUP_CATEGORIES[group] for item in pool)
        assert not {item.id for item in pool} & bench_ids
        assert len({item.id for item in pool}) == len(pool)


def test_questions_are_the_same_for_the_same_seed():
    a = pick_questions("developers", 3, seed=5)
    assert [q.id for q in a] == [q.id for q in pick_questions("developers", 3, seed=5)]
    assert len({q.id for q in a}) == 3
    assert [q.id for q in a] != [q.id for q in pick_questions("developers", 3, seed=6)]


def test_a_reply_is_scored_by_its_question_check():
    for group in GROUPS:
        item = customer_pool(group)[0]
        right = item.answers[0] if isinstance(item, ExactItem) else item.reference
        assert score_reply(item, right) == 1.0
        assert score_reply(item, "") == 0.0
