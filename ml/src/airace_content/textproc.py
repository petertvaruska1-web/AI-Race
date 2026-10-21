"""Text processing shared by the corpus build: normalization, word statistics and quality scoring.

``quality_score`` is a measurement, not a judgement: it is the number the player's "cleaning"
level filters on, so each part of it looks for one thing that really is wrong with a document
(too few letters, unknown words, repeated lines, mojibake, spam, web-page furniture) and nothing
else.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

MOJIBAKE_CHARS = frozenset("ÃÂâ€™œ¢�¤¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾")
SPAM_MARKERS: tuple[str, ...] = (
    "click here",
    "buy now",
    "free!!!",
    "!!!",
    "$$$",
    "http://",
    "https://",
    "subscribe",
    "limited offer",
    "act now",
)

# Phrases of the navigation, share and footer lines of a web page. A line holding one is page
# furniture, like any line with a "|" separator ("Home | About | Contact"). Not the generic
# "terms of" (it is in "in terms of").
FURNITURE_PHRASES: tuple[str, ...] = (
    "privacy policy",
    "skip to",
    "sign in",
    "log in",
    "back to top",
    "share this",
    "read more",
    "cookies",
    "follow us",
    "powered by",
    "leave a comment",
    "older posts",
    "sitemap",
)
_FURNITURE_PHRASE = re.compile(r"\b(?:" + "|".join(FURNITURE_PHRASES) + r")\b")
_SPECIAL_TOKEN_LITERAL = re.compile(r"<\|[^|\s>]*\|>")  # "<|end|>" is text, not a separator

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # category Cc, except tab and newline
# A line's leading spaces and tabs (its indentation), or else a run of spaces; the indentation is
# matched first, and kept, so code survives normalization.
_INDENT_OR_SPACE_RUN = re.compile(r"^([ \t]*)|( {2,})", re.MULTILINE)
_TRAILING_SPACE = re.compile(r" +(?=\n|\Z)")
_NEWLINE_RUN = re.compile(r"\n{3,}")
_WORD = re.compile(r"[a-z']+")
_STRAIGHT_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'"})  # curly single quotes
_SENTENCE_END = re.compile(r"[.!?]+")

_LETTER_SHARE_RANGE = (0.6, 0.9)  # alpha_ratio at which `a` is 0 and 1
_KNOWN_SHARE_RANGE = (0.7, 0.95)  # known_ratio at which `k` is 0 and 1 (calibrated: ADR 0004)
_GARBLE_WEIGHT = 20.0  # a document with 5% mojibake characters scores 0
_SPAM_LIMIT = 3  # this many spam hits score 0
_FURNITURE_LIMIT = 3  # this many furniture lines score 0
_FURNITURE_MAX_WORDS = 20  # a longer line that holds a furniture phrase is prose ("bake cookies")
_SIMPLE_SENTENCE_WORDS = 18  # sentences longer than this lower `simplicity`


def normalize_text(s: str) -> str:
    """Canonical form of a document's text.

    NFC; ``\\r\\n`` and ``\\r`` become ``\\n``; control characters other than newline and tab are
    removed; runs of spaces collapse to one, except in a line's leading indentation (spaces and
    tabs before its first other character), which is kept; spaces at the end of a line (and of the
    text) are stripped; three or more newlines become two. Everything else, including non-ASCII
    text, is kept.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL.sub("", s)
    s = unicodedata.normalize("NFC", s)
    s = _INDENT_OR_SPACE_RUN.sub(lambda m: m.group(0) if m.group(1) is not None else " ", s)
    s = _TRAILING_SPACE.sub("", s)
    return _NEWLINE_RUN.sub("\n\n", s)


def words(s: str) -> list[str]:
    """Lowercase alphabetic words (``[a-z']+``), in order.

    Curly single quotes count as apostrophes ("isn\u2019t" is "isn't"), and apostrophes around a
    word are dropped ("'hello'" is "hello"; "dogs'" is "dogs"), so quoting does not change a word.
    """
    found = _WORD.findall(s.lower().translate(_STRAIGHT_APOSTROPHES))
    return [word for word in (w.strip("'") for w in found) if word]


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def build_vocab(texts: Iterable[str], top_n: int) -> set[str]:
    """The ``top_n`` most frequent words over ``texts``; ties go to the alphabetically first."""
    if top_n < 0:
        raise ValueError(f"top_n must not be negative, got {top_n}")
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(words(text))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {word for word, _ in ranked[:top_n]}


def simplicity(text: str, simple_vocab: set[str]) -> float:
    """How simple a text reads: the share of its words in ``simple_vocab``, lowered when sentences
    are long (``in_vocab_ratio * min(1, 18 / mean_sentence_words)``). 0 for text without words."""
    tokens = words(text)
    if not tokens:
        return 0.0
    in_vocab = sum(w in simple_vocab for w in tokens) / len(tokens)
    n_sentences = sum(1 for part in _SENTENCE_END.split(text) if words(part))
    mean_sentence_words = len(tokens) / n_sentences
    return in_vocab * min(1.0, _SIMPLE_SENTENCE_WORDS / mean_sentence_words)


def furniture_lines(text: str) -> int:
    """How many lines of ``text`` look like the furniture of a web page (menus, share buttons,
    footers): a non-empty line with a ``|`` separator, or a short one (at most 20 words) with a
    ``FURNITURE_PHRASES`` entry in it. A literal special token such as ``<|end|>`` is no separator,
    and a long line with a phrase in it is prose ("they made cookies in the kitchen ...")."""
    return sum(map(_is_furniture, text.split("\n")))


def _is_furniture(line: str) -> bool:
    if "|" in _SPECIAL_TOKEN_LITERAL.sub("", line):
        return True
    short = len(line.split()) <= _FURNITURE_MAX_WORDS
    return short and _FURNITURE_PHRASE.search(line.lower()) is not None


def quality_score(text: str, known_vocab: set[str]) -> float:
    """How clean a document is, from 0 to 1 (0 for empty text).

    ``k * f * g * p * (0.5 + 0.5 a) * (0.5 + 0.5 r)`` where

    * ``a`` rises from 0 to 1 as the share of letters among non-space characters goes 0.6 to 0.9,
    * ``k`` rises from 0 to 1 as the share of known words (length 2 or more) goes 0.7 to 0.95,
    * ``r`` is the share of non-empty lines that do not repeat an earlier line,
    * ``g`` falls to 0 once 5% of the characters are mojibake (``MOJIBAKE_CHARS``),
    * ``p`` falls to 0 at three spam hits: every occurrence of a ``SPAM_MARKERS`` entry counts,
    * ``f`` falls to 0 at three lines of web-page furniture (``furniture_lines``), counted like
      spam hits. The other factors are shares of the whole text, which a few lines of furniture
      move by under 1% on a web document of a few hundred words; ``f`` counts the lines instead.
      Only prose has it: a ``|`` is ordinary code, so ``structured_quality`` has no ``f``.
    """
    compact = "".join(text.split())
    if not compact:
        return 0.0
    alpha_ratio = sum(map(str.isalpha, compact)) / len(compact)
    lo, hi = _LETTER_SHARE_RANGE
    a = _clip((alpha_ratio - lo) / (hi - lo))

    long_words = [w for w in words(text) if len(w) >= 2]
    known_ratio = sum(w in known_vocab for w in long_words) / len(long_words) if long_words else 0.0
    lo, hi = _KNOWN_SHARE_RANGE
    k = _clip((known_ratio - lo) / (hi - lo))

    f = 1 - min(1.0, furniture_lines(text) / _FURNITURE_LIMIT)

    return float(k * f * (0.5 + 0.5 * a) * _language_free_quality(text))


def structured_quality(text: str) -> float:
    """``quality_score`` for structured text (programs, puzzles, number patterns), from 0 to 1
    (0 for empty text): ``g * p * (0.5 + 0.5 r)``.

    It drops the two factors that measure English prose (``k``, the share of known words, and
    ``a``, the share of letters), which clean code and puzzles fail by their nature, and keeps
    the ones that measure real damage: garbling, spam and repeated lines.
    """
    if not "".join(text.split()):
        return 0.0
    return float(_language_free_quality(text))


def _language_free_quality(text: str) -> float:
    """``g * p * (0.5 + 0.5 r)``, the factors of ``quality_score`` that hold for any kind of text
    (``text`` must have a non-space character)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    repeated_line_ratio = 1 - len(set(lines)) / len(lines)
    r = 1 - repeated_line_ratio

    garble_ratio = sum(map(MOJIBAKE_CHARS.__contains__, text)) / len(text)
    g = 1 - _clip(_GARBLE_WEIGHT * garble_ratio)

    lowered = text.lower()
    spam_hits = sum(lowered.count(marker) for marker in SPAM_MARKERS)
    p = 1 - min(1.0, spam_hits / _SPAM_LIMIT)

    return g * p * (0.5 + 0.5 * r)
