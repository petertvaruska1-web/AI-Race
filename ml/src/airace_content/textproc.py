"""Text processing shared by the corpus build: normalization, word statistics and quality scoring.

``quality_score`` is a measurement, not a judgement: it is the number the player's "cleaning"
level filters on, so each part of it looks for one thing that really is wrong with a document
(too few letters, unknown words, repeated lines, mojibake, spam) and nothing else.
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

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # category Cc, except tab and newline
_SPACE_RUN = re.compile(r" {2,}")
_TRAILING_SPACE = re.compile(r" +(?=\n|\Z)")
_NEWLINE_RUN = re.compile(r"\n{3,}")
_WORD = re.compile(r"[a-z']+")
_SENTENCE_END = re.compile(r"[.!?]+")

_LETTER_SHARE_RANGE = (0.6, 0.9)  # alpha_ratio at which `a` is 0 and 1
_KNOWN_SHARE_RANGE = (0.5, 0.9)  # known_ratio at which `k` is 0 and 1
_GARBLE_WEIGHT = 20.0  # a document with 5% mojibake characters scores 0
_SPAM_LIMIT = 3  # this many spam hits score 0
_SIMPLE_SENTENCE_WORDS = 18  # sentences longer than this lower `simplicity`


def normalize_text(s: str) -> str:
    """Canonical form of a document's text.

    NFC; ``\\r\\n`` and ``\\r`` become ``\\n``; control characters other than newline and tab are
    removed; runs of spaces collapse to one; spaces at the end of a line (and of the text) are
    stripped; three or more newlines become two. Everything else, including non-ASCII text, is kept.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL.sub("", s)
    s = unicodedata.normalize("NFC", s)
    s = _SPACE_RUN.sub(" ", s)
    s = _TRAILING_SPACE.sub("", s)
    return _NEWLINE_RUN.sub("\n\n", s)


def words(s: str) -> list[str]:
    """Lowercase alphabetic words (``[a-z']+``), in order."""
    return _WORD.findall(s.lower())


def _content_words(s: str) -> list[str]:
    """``words`` without tokens made only of apostrophes (a stray quote mark is not a word)."""
    return [w for w in words(s) if w.strip("'")]


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def build_vocab(texts: Iterable[str], top_n: int) -> set[str]:
    """The ``top_n`` most frequent words over ``texts``; ties go to the alphabetically first."""
    if top_n < 0:
        raise ValueError(f"top_n must not be negative, got {top_n}")
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_content_words(text))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {word for word, _ in ranked[:top_n]}


def simplicity(text: str, simple_vocab: set[str]) -> float:
    """How simple a text reads: the share of its words in ``simple_vocab``, lowered when sentences
    are long (``in_vocab_ratio * min(1, 18 / mean_sentence_words)``). 0 for text without words."""
    tokens = _content_words(text)
    if not tokens:
        return 0.0
    in_vocab = sum(w in simple_vocab for w in tokens) / len(tokens)
    n_sentences = sum(1 for part in _SENTENCE_END.split(text) if _content_words(part))
    mean_sentence_words = len(tokens) / n_sentences
    return in_vocab * min(1.0, _SIMPLE_SENTENCE_WORDS / mean_sentence_words)


def quality_score(text: str, known_vocab: set[str]) -> float:
    """How clean a document is, from 0 to 1 (0 for empty text).

    ``k * g * p * (0.5 + 0.5 a) * (0.5 + 0.5 r)`` where

    * ``a`` rises from 0 to 1 as the share of letters among non-space characters goes 0.6 to 0.9,
    * ``k`` rises from 0 to 1 as the share of known words (length 2 or more) goes 0.5 to 0.9,
    * ``r`` is the share of non-empty lines that do not repeat an earlier line,
    * ``g`` falls to 0 once 5% of the characters are mojibake (``MOJIBAKE_CHARS``),
    * ``p`` falls to 0 at three spam hits: every occurrence of a ``SPAM_MARKERS`` entry counts.
    """
    compact = "".join(text.split())
    if not compact:
        return 0.0
    alpha_ratio = sum(map(str.isalpha, compact)) / len(compact)
    lo, hi = _LETTER_SHARE_RANGE
    a = _clip((alpha_ratio - lo) / (hi - lo))

    long_words = [w for w in _content_words(text) if len(w) >= 2]
    known_ratio = sum(w in known_vocab for w in long_words) / len(long_words) if long_words else 0.0
    lo, hi = _KNOWN_SHARE_RANGE
    k = _clip((known_ratio - lo) / (hi - lo))

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    repeated_line_ratio = 1 - len(set(lines)) / len(lines)
    r = 1 - repeated_line_ratio

    garble_ratio = sum(map(MOJIBAKE_CHARS.__contains__, text)) / len(text)
    g = 1 - _clip(_GARBLE_WEIGHT * garble_ratio)

    lowered = text.lower()
    spam_hits = sum(lowered.count(marker) for marker in SPAM_MARKERS)
    p = 1 - min(1.0, spam_hits / _SPAM_LIMIT)

    return float(k * g * p * (0.5 + 0.5 * a) * (0.5 + 0.5 * r))
