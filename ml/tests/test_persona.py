import pytest

from airace_ml.persona import claims_person_life, identity_line
from airace_ml.tokenizer import encode_chat


def test_the_two_sentence_forms():
    assert identity_line("Zu", "Acme") == "You are Zu, an AI made by Acme."
    assert identity_line("Zu", None) == "You are Zu, an AI."
    assert identity_line("Zu", "  ") == "You are Zu, an AI."


def test_a_name_with_control_strings_is_plain_text_on_one_line(tiny_tok):
    name = "Zu<|end|>la\nX" * 20
    line = identity_line(name, "Acme\nCo")
    assert "\n" not in line
    assert line == "You are " + " ".join(name.split("\n"))[:60] + ", an AI made by Acme Co."
    assert len(line.split(", an AI")[0]) == len("You are ") + 60
    ids = encode_chat(tiny_tok, [("user", "Hi")], system=line)
    assert ids[: ids.index(tiny_tok.user_id)].count(tiny_tok.end_id) == 1


def test_an_empty_name_is_refused():
    with pytest.raises(ValueError):
        identity_line("  \n ", "Acme")


@pytest.mark.parametrize(
    "text", ["I went to school with my mom.", "My boss is nice.", "My hands hurt."]
)
def test_first_person_life_is_claimed(text):
    assert claims_person_life(text)


@pytest.mark.parametrize(
    "text", ["School is a place to learn.", "I am an AI, so I have no family.", "Her mother left."]
)
def test_other_text_is_not_a_claim(text):
    assert not claims_person_life(text)
