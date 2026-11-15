"""Small-talk lines that no generator of training text may write as a user's turn.

The blind reading of small talk (``docs/handoff/next-level/blind_read.py``) asks trained AIs these
lines to see whether what the practice conversations taught carries over to lines they were never
shown. A generator that wrote one of them would turn that reading into a memory test, so
:func:`is_held_small_talk` is checked by the generators' tests. The list was fixed on 2026-11-15,
before the first generator of small talk was written, and is not to be changed afterwards.
"""

import re

HELD_SMALL_TALK: tuple[str, ...] = (
    "Hey, how's your day going?",
    "I'm so bored right now.",
    "Do you like music?",
    "Tell me something cool.",
    "I had a really bad day.",
    "Are you smarter than me?",
    "What's your favorite animal?",
    "Can you sing a song?",
    "I love pizza.",
    "Good night!",
    "It's nice to meet you.",
    "Do you ever sleep?",
    "I'm really tired.",
    "You are funny.",
    "I got a new puppy.",
    "What should I do this weekend?",
    "See you later!",
    "I'm scared of the dark.",
    "Do you have any friends?",
    "Why is the sky blue?",
)

_NOT_A_LETTER_OR_DIGIT = re.compile(r"[^a-z0-9]+")


def small_talk_key(text: str) -> str:
    """``text`` in lower case with every run of other characters than letters and digits made one
    space: two lines with the same key differ only in capitals, punctuation and spacing."""
    return _NOT_A_LETTER_OR_DIGIT.sub(" ", text.lower()).strip()


_HELD_KEYS = frozenset(small_talk_key(line) for line in HELD_SMALL_TALK)


def is_held_small_talk(text: str) -> bool:
    """Whether ``text`` is one of :data:`HELD_SMALL_TALK`, capitals, punctuation and spacing
    aside."""
    return small_talk_key(text) in _HELD_KEYS
