"""How an AI talks to people: the settings its words are drawn with in chats, customer answers
and keynotes, and the form a typed message reaches it in.

The AI's words are never written or rewritten here (R1). Three things are decided:

* **How replies are drawn.** At temperature 0.8 a first AI's replies were well-formed about half
  the time, at 0.6 about four times in five (ADR 0009): brains this small need calmer draws than
  big models do. Benchmarks, battles, games and the personality fingerprint keep their own
  settings, so recorded scores stay comparable.
* **Loops end.** A reply whose tail is one short token sequence said over and over ("ra ra ra
  ra") stops there. Nothing is removed or replaced: generation simply ends, as at a length limit.
* **A typed message starts with a capital letter**, as every conversation the AIs learned from
  does. Engine-2 AIs also learn casual typing (:mod:`airace_ml.data.casual`); AIs trained before
  it have never seen a turn begin in lowercase and answer one with gibberish.
"""

from __future__ import annotations

from collections.abc import Sequence

SERVING_TEMPERATURE = 0.6
SERVING_TOP_P = 0.9

# A loop: the reply ends with LOOP_REPEATS copies of one unit of up to LOOP_MAX_UNIT tokens
# (LOOP_SHORT_REPEATS copies when the unit is one or two tokens: "ha ha ha" is not a loop yet).
LOOP_MAX_UNIT = 8
LOOP_REPEATS = 3
LOOP_SHORT_REPEATS = 4


def is_looping(tokens: Sequence[int]) -> bool:
    """Whether ``tokens`` end in a loop: one unit of 1 to :data:`LOOP_MAX_UNIT` tokens repeated
    :data:`LOOP_REPEATS` times in a row (:data:`LOOP_SHORT_REPEATS` for units of 1 or 2)."""
    n = len(tokens)
    for unit in range(1, LOOP_MAX_UNIT + 1):
        repeats = LOOP_SHORT_REPEATS if unit <= 2 else LOOP_REPEATS
        span = unit * repeats
        if span > n:
            break
        tail = tokens[n - unit :]
        if all(tokens[n - span + k] == tail[k % unit] for k in range(span - unit)):
            return True
    return False


def tidy_user_text(text: str) -> str:
    """A message a person typed, as the chat format carries it: without the blank space around
    it and with its first letter capitalised. Everything else stays exactly as typed."""
    text = text.strip()
    if text[:1].islower():
        return text[:1].upper() + text[1:]
    return text
