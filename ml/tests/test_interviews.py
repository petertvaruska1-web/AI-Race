import pytest

from airace_ml.evals.suite import build_suite
from airace_ml.interviews import interview_cards, score_answer


def test_twenty_four_cards_half_with_a_right_answer():
    cards = interview_cards()
    assert len(cards) == 24
    assert len({c.id for c in cards}) == 24
    checked = [c for c in cards if c.checked]
    assert len(checked) == 12
    for category in ("knowledge", "pattern", "instruction", "coding"):
        assert sum(c.id.startswith(f"interview-{category}-") for c in checked) == 3
    open_cards = [c for c in cards if not c.checked]
    assert [c.id for c in open_cards] == [f"interview-open-{k:02d}" for k in range(1, 13)]
    assert open_cards[0].prompt == "What would you like people to know about you?"
    assert open_cards[-1].prompt == "Describe the color blue."
    bench = {getattr(i, "prompt", None) for items in build_suite().items.values() for i in items}
    assert not any(c.prompt in bench for c in checked)
    assert interview_cards() == cards  # the same cards everywhere


def test_a_checked_card_is_scored_by_its_check():
    from airace_ml.interviews import _checked_items

    card_id, item = next(iter(_checked_items().items()))
    right = item.answers[0] if hasattr(item, "answers") else item.reference
    assert score_answer(card_id, right, None) == (1.0, True)
    assert score_answer(card_id, "", None) == (0.0, True)


def test_open_answers_are_scored_for_fluency():
    words = "I like to read books about the sea and the stars at night."
    assert score_answer("interview-open-01", words, None) == (1.0, False)
    assert score_answer("interview-open-01", "", None) == (0.0, False)
    assert score_answer(None, words, None) == (1.0, False)
    assert score_answer("interview-nope-99", words, None) == (1.0, False)  # unknown: open


class FakeJudge:
    """A judge whose loss is always the calibration's middle."""

    class calibration:
        coherence_lo = 1.0
        coherence_hi = 3.0

    class tok:
        @staticmethod
        def encode(text):
            return [1, 2, 3]

    @staticmethod
    def nll_per_token(docs):
        return [2.0 for _ in docs]


def test_with_the_judge_fluency_is_coherence_times_variety(monkeypatch):
    import airace_ml.interviews as iv

    monkeypatch.setattr(iv, "_coherence", lambda nll, cal: 0.5)
    words = "I like to read books about the sea and the stars at night."
    score, checked = score_answer("interview-open-02", words, FakeJudge())
    assert checked is False
    assert score == pytest.approx(0.5 * (1 - iv.repetitiveness(words)))
    assert score_answer("interview-open-02", "hi", FakeJudge()) == (0.0, False)
