import pytest
import torch

from airace_ml.infer import lm as lm_module
from airace_ml.infer.serving import (
    SERVING_TEMPERATURE,
    SERVING_TOP_P,
    is_looping,
    tidy_user_text,
)
from airace_ml.model.transformer import Transformer


@pytest.mark.parametrize(
    ("tokens", "loops"),
    [
        ([], False),
        ([5], False),
        ([5, 5, 5], False),  # "ha ha ha" is not a loop yet
        ([5, 5, 5, 5], True),
        ([1, 2, 5, 5, 5, 5], True),  # only the tail matters
        ([5, 5, 5, 5, 9], False),  # it got out of it
        ([7, 8, 7, 8, 7, 8], False),
        ([7, 8, 7, 8, 7, 8, 7, 8], True),  # units of two need four turns too
        ([1, 2, 3, 1, 2, 3], False),
        ([1, 2, 3, 1, 2, 3, 1, 2, 3], True),  # longer units: three turns
        ([9, 1, 2, 3, 4, 5, 6, 7, 8] + [1, 2, 3, 4, 5, 6, 7, 8] * 2, True),  # a unit of eight
        ([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3, False),  # a unit of nine is past the limit
        ([1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4], False),  # nearly, not quite
    ],
)
def test_is_looping_sees_a_tail_said_over_and_over(tokens, loops):
    assert is_looping(tokens) is loops


@pytest.mark.parametrize(
    ("typed", "carried"),
    [
        ("hello", "Hello"),
        ("  what is your name?  ", "What is your name?"),
        ("Hello", "Hello"),
        ("ünder", "Ünder"),
        ("3 apples", "3 apples"),
        ("(maybe)", "(maybe)"),
        ("", ""),
        ("   ", ""),
        ("iPhone or android", "IPhone or android"),
    ],
)
def test_tidy_user_text_capitalises_the_first_letter_and_nothing_else(typed, carried):
    assert tidy_user_text(typed) == carried


def test_serving_draws_are_calmer_than_the_measuring_ones():
    assert 0.3 <= SERVING_TEMPERATURE < 0.8 and 0.5 < SERVING_TOP_P < 0.95


@pytest.fixture
def looping_lm(tiny_lm, tiny_tok, monkeypatch):
    """``tiny_lm`` with a model that always says token 40 next (so every reply loops)."""
    real = Transformer.forward

    def forward(self, idx, **kwargs):
        logits = real(self, idx, **kwargs)
        logits = torch.full_like(logits, -1e4)
        logits[..., 40] = 1e4
        return logits

    monkeypatch.setattr(Transformer, "forward", forward)
    return tiny_lm


def test_a_looping_reply_ends_where_the_loop_shows_only_when_asked(looping_lm, tiny_tok):
    prompt = [tiny_tok.bos_id, 20, 21]
    free = looping_lm.generate([prompt], max_new_tokens=12, temperature=0.0, top_p=1.0, seed=0)[0]
    assert free.tokens == [40] * 12 and not free.looped and not free.stopped
    cut = looping_lm.generate(
        [prompt], max_new_tokens=12, temperature=0.0, top_p=1.0, seed=0, stop_on_loop=True
    )[0]
    assert cut.tokens == [40] * 4 and cut.looped and not cut.stopped  # nothing removed: it ended
    # in a batch, each row ends on its own loop and the others go on
    both = looping_lm.generate(
        [prompt, [tiny_tok.bos_id, 22, 23, 24]],
        max_new_tokens=6,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        stop_on_loop=True,
    )
    assert [g.tokens for g in both] == [[40] * 4, [40] * 4] and all(g.looped for g in both)


def test_chat_replies_and_streams_can_stop_on_a_loop(looping_lm, tiny_tok):
    history = [("user", "Hello")]
    said = tiny_tok.decode([40])
    assert looping_lm.chat_reply(history, max_new_tokens=10, stop_on_loop=True) == said * 4
    assert looping_lm.chat_reply(history, max_new_tokens=10) == said * 10
    pieces = []
    text = looping_lm.stream_chat_reply(
        history, max_new_tokens=10, on_text=pieces.append, stop_on_loop=True
    )
    assert text == said * 4 == "".join(pieces)
    assert lm_module.Generation([], [], [], False).looped is False
