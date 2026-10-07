"""Answer checkers: each judges one model reply, ``CHECKERS[name](reply, args) -> bool``.

They back the benchmark items that are free answers judged by a rule instead of a fixed string
(instruction following and function completion). A checker never uses the real Python runtime on
model text: function replies run only in MiniPy.

Two rules hold for every checker. An empty or degenerate reply fails (a reply with no words, a
word repeated over and over). And harmless variants pass: capital letters, punctuation and
surrounding spaces never decide the result, so ``Yes.`` and ``yes`` are the same answer.

A *word* is a run of letters and digits, with ``'`` or ``-`` allowed inside it (``don't``,
``well-known``); punctuation around it is ignored.
"""

import re
from collections.abc import Callable
from typing import Any

from airace_ml.minipy.interpreter import call_function

_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_LIST_SPLIT = re.compile(r"[\n,]|\band\b", re.IGNORECASE)
MAX_WORDS_PER_LIST_ITEM = 3
_DEGENERATE_MIN_WORDS = 6


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _degenerate(words: list[str]) -> bool:
    """A reply that says one thing over and over: 6 or more words, a third or fewer different."""
    if len(words) < _DEGENERATE_MIN_WORDS:
        return False
    return 3 * len({w.casefold() for w in words}) <= len(words)


def _trim(text: str) -> str:
    """``text`` without spaces and punctuation at either end."""
    return re.sub(r"^\W+|\W+$", "", text.strip())


def one_word(reply: str, args: dict[str, Any] | None = None) -> bool:
    """Exactly one word (punctuation around it is ignored).

    ``args["not"]``, if given, lists words that do not count as an answer ("yes", "no" for a
    question that is not a yes-or-no one), so the same word cannot pass everything.
    """
    words = _words(reply)
    refused = {w.casefold() for w in (args or {}).get("not", ())}
    return len(words) == 1 and words[0].casefold() not in refused


def yes_no(reply: str, args: dict[str, Any] | None = None) -> bool:
    """The whole reply is "yes" or "no", in any case and with any punctuation."""
    words = _words(reply)
    return len(words) == 1 and words[0].casefold() in ("yes", "no")


def list_n(reply: str, args: dict[str, Any]) -> bool:
    """``args["n"]`` different items, separated by commas, "and" or new lines.

    An item is up to 3 words (a numbered or bulleted item is fine); a sentence is not an item.
    """
    items = []
    for part in _LIST_SPLIT.split(reply):
        item = _trim(_BULLET.sub("", part))
        if item:
            items.append(item)
    if len(items) != args["n"]:
        return False
    if len({item.casefold() for item in items}) != len(items):
        return False
    return all(0 < len(_words(item)) <= MAX_WORDS_PER_LIST_ITEM for item in items)


def starts_with(reply: str, args: dict[str, Any]) -> bool:
    """The reply starts with the word or phrase ``args["prefix"]`` (any case)."""
    prefix = args["prefix"].strip()
    text = reply.lstrip()
    match = re.match(re.escape(prefix), text, re.IGNORECASE) if prefix else None
    if match is None:
        return False
    if prefix[-1].isalnum() and text[match.end() : match.end() + 1].isalnum():
        return False  # "Hello" does not start "Hellos"
    return not _degenerate(_words(text))


def all_caps(reply: str, args: dict[str, Any] | None = None) -> bool:
    """There are letters, and every one of them is a capital.

    ``args["min_words"]``, if given, is how many words the reply needs at least (a sentence
    asked for in capitals is not one word).
    """
    words = _words(reply)
    has_letter = any(c.isalpha() for c in reply)
    enough = len(words) >= (args or {}).get("min_words", 1)
    return has_letter and enough and not any(c.islower() for c in reply) and not _degenerate(words)


def repeat_word(reply: str, args: dict[str, Any]) -> bool:
    """The reply is just ``args["word"]`` (any case, spaces and punctuation around it ignored)."""
    target = _trim(args["word"]).casefold()
    return bool(target) and _trim(reply).casefold() == target


def contains_any(reply: str, args: dict[str, Any]) -> bool:
    """The reply uses one of ``args["words"]`` as a word (a plural with "s" or "es" counts)."""
    words = _words(reply)
    if not words or _degenerate(words):
        return False
    for word in args["words"]:
        if word.strip() and re.search(
            rf"(?<!\w){re.escape(word.strip())}(?:e?s)?(?!\w)", reply, re.IGNORECASE
        ):
            return True
    return False


def function_body(reply: str) -> str:
    """The reply up to (not including) the first line that is not empty and starts without a space.

    A completed function ends where the next top-level statement begins, so whatever the model
    writes after its function is not part of it.
    """
    lines = reply.split("\n")
    for k, line in enumerate(lines):
        if line.strip() and line[0] not in " \t":
            return "".join(f"{kept}\n" for kept in lines[:k])
    return reply


def _same(value: object, expected: object) -> bool:
    """Equal in value and in type (``True`` is not ``1``), also inside lists."""
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(_same(v, e) for v, e in zip(value, expected))
        )
    return type(value) is type(expected) and value == expected


def minipy_function_tests(reply: str, args: dict[str, Any]) -> bool:
    """Run ``args["prompt"]`` + the function body in ``reply`` in MiniPy, against every test.

    ``args["tests"]`` is ``[[call_args, expected], ...]`` and ``args["name"]`` the function.
    Every call must return exactly the expected value; any MiniPy error (a wrong or missing
    body, an infinite loop) fails.
    """
    source = args["prompt"] + function_body(reply)
    for call_args, expected in args["tests"]:
        value, error = call_function(source, args["name"], list(call_args))
        if error is not None or not _same(value, expected):
            return False
    return True


CHECKERS: dict[str, Callable[[str, dict[str, Any]], bool]] = {
    "one_word": one_word,
    "yes_no": yes_no,
    "list_n": list_n,
    "starts_with": starts_with,
    "all_caps": all_caps,
    "repeat_word": repeat_word,
    "contains_any": contains_any,
    "minipy_function_tests": minipy_function_tests,
}
