"""Deliberate dirt, tagged: typos, spam, boilerplate, garbled encoding, duplicates and false facts.

``inject_noise`` dirties a share of a dataset's documents and says what it did to each one, so the
tags mean what they say: a document tagged "typo" really contains typos, a document tagged
"false_fact" really states a planned falsehood (and ``false_fact`` is True for exactly those), and
a document tagged "duplicate" really is a copy or near-copy of another document in the output.
A document that a noise function cannot change (typos in a text without letters) stays untagged.

How strongly a typo or garbled document is dirtied varies from document to document, so the
quality score (and with it the cleaning levels) separates mild damage from severe damage; spam is
always overt.
"""

import re
from dataclasses import dataclass

import numpy as np

from airace_ml.data.corpus import NOISE_KINDS
from airace_ml.skills.facts import FalseFactPlan, false_fact_sentence
from airace_ml.skills.kb import KB

TYPO_RATE_RANGE = (0.01, 0.15)  # per-letter typo rate of a typo document, drawn log-uniformly
GARBLE_RATE_RANGE = (0.005, 0.12)  # per-letter garble rate of a garbled document, likewise
MIN_COPIES, MAX_COPIES = 2, 6  # copies made of one source document
COPY_TYPO_SHARE = 0.5  # share of copies that carry one or two typos
MAX_FALSE_FACTS = 3  # false sentences inserted into one document

_NONE = NOISE_KINDS.index("none")
_DUPLICATE = NOISE_KINDS.index("duplicate")
_KIND_INDEX = {name: NOISE_KINDS.index(name) for name in NOISE_KINDS}
_PER_DOCUMENT_KINDS = ("typo", "spam", "boilerplate", "garbled", "false_fact")


@dataclass
class NoiseRates:
    """Each rate is the share of the *output* documents that get that kind of noise."""

    typo: float = 0.0
    spam: float = 0.0
    boilerplate: float = 0.0
    garbled: float = 0.0
    duplicate: float = 0.0
    false_fact: float = 0.0


@dataclass
class NoisyDoc:
    text: str
    noise_kind: int  # index into NOISE_KINDS
    false_fact: bool  # True exactly when noise_kind is "false_fact"


# --- typos -------------------------------------------------------------------------------------

_KEY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedzx", "d": "serfxc",
    "f": "drtgcv", "g": "ftyhvb", "h": "gyujbn", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}  # fmt: skip
_COMMON_LETTERS = "etaoinsrhl"
_TYPO_OPS = ("swap", "drop", "double", "substitute")


def _nearby_letter(c: str, rng: np.random.Generator) -> str:
    """A different letter, preferably one next to ``c`` on a keyboard, in the case of ``c``."""
    lower = c.lower()
    options = _KEY_NEIGHBORS.get(lower) or _COMMON_LETTERS.replace(lower, "")
    letter = options[int(rng.integers(len(options)))]
    return letter.upper() if c.isupper() else letter


def _misspell(text: str, rng: np.random.Generator, n_typos: int) -> str:
    """``text`` with ``n_typos`` letters misspelled (fewer if it has fewer letters).

    A typo swaps a letter with the next one, drops it, doubles it or replaces it with a nearby
    key. The result always differs from ``text`` when ``text`` has any letter.
    """
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    if not letters:
        return text
    picked = rng.choice(letters, size=min(n_typos, len(letters)), replace=False)
    chosen = sorted(int(i) for i in picked)
    chosen_set = set(chosen)
    parts = list(text)
    for i in chosen:
        c = text[i]
        op = _TYPO_OPS[int(rng.integers(len(_TYPO_OPS)))]
        if op == "swap":
            j = i + 1
            if j < len(text) and j not in chosen_set and text[j].isalpha() and text[j] != c:
                parts[i], parts[j] = text[j], c
                continue
            op = "substitute"
        if op == "drop":
            parts[i] = ""
        elif op == "double":
            parts[i] = c + c
        else:
            parts[i] = _nearby_letter(c, rng)
    out = "".join(parts)
    if out == text:  # a doubled letter and the dropped one beside it cancelled out
        i = letters[int(rng.integers(len(letters)))]
        out = text[:i] + _nearby_letter(text[i], rng) + text[i + 1 :]
    return out


def add_typos(text: str, rng: np.random.Generator, rate: float = 0.03) -> str:
    """Misspell about ``rate`` of the letters of ``text`` (at least one if it has any letter)."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"typo rate must be between 0 and 1, got {rate}")
    n_letters = sum(c.isalpha() for c in text)
    if n_letters == 0:
        return text
    return _misspell(text, rng, max(1, int(rng.binomial(n_letters, rate))))


# --- spam --------------------------------------------------------------------------------------

_SPAM_HOOKS = (  # each holds at least one of textproc.SPAM_MARKERS
    "CLICK HERE to claim your prize!!!",
    "Buy now and save 90% on everything.",
    "FREE!!! Limited offer, act now.",
    "Visit http://{site}.com for amazing deals.",
    "Subscribe to our newsletter and win $$$ today.",
    "Earn $$$ from home with no experience needed.",
    "Check out https://{site}.net/offer before it ends.",
    "Act now, this deal ends tonight.",
    "Limited offer for the first 100 customers.",
    "Congratulations, you are our lucky winner!!!",
    "Click here to subscribe or unsubscribe.",
    "Buy now, pay later, no credit check.",
)
_SPAM_FILLER = (
    "Best price guaranteed.",
    "Lose weight fast with this one weird trick.",
    "Cheap watches and bags online.",
    "Hot singles in your area want to meet you.",
    "Make money fast, work from home.",
    "The miracle cure doctors do not want you to know about.",
    "Get rich quick with our secret system.",
    "No obligation, 100% satisfaction.",
    "Amazing results in just 7 days.",
    "Cheap meds without a prescription.",
)
_SPAM_SITES = (
    "cheap-deals", "win-big-now", "bestpills24", "mega-offers", "luckyprize", "fast-cash-today",
    "hotdeals4u", "super-savings",
)  # fmt: skip


def make_spam(rng: np.random.Generator) -> str:
    """A spam text: three to six sales hooks (each with a spam marker) among some filler."""
    hook_ids = rng.choice(len(_SPAM_HOOKS), size=int(rng.integers(3, 7)), replace=False)
    filler_ids = rng.choice(len(_SPAM_FILLER), size=int(rng.integers(0, 4)), replace=False)
    pieces = [
        _SPAM_HOOKS[int(i)].format(site=_SPAM_SITES[int(rng.integers(len(_SPAM_SITES)))])
        for i in hook_ids
    ] + [_SPAM_FILLER[int(i)] for i in filler_ids]
    return " ".join(pieces[int(i)] for i in rng.permutation(len(pieces)))


# --- boilerplate -------------------------------------------------------------------------------

_PAGE_HEADERS = (
    "Home | About | Contact | Privacy Policy | Terms of Use",
    "Menu Search Skip to content",
    "This website uses cookies to give you the best experience. Accept all cookies",
    "Posted by admin on March 3, 2019 | 4 comments",
    "Sign in | Register | Cart (0)",
    "Skip to main content | Accessibility help",
    "You are here: Home > Blog > Archive",
)
_PAGE_FOOTERS = (
    "Copyright (c) 2019 All rights reserved.",
    "Privacy Policy | Terms of Service | Contact us | Sitemap",
    "Share this: Facebook Twitter Email Print",
    "Related posts: Read more >> Older posts",
    "Subscribe to our newsletter for updates.",
    "Click here to read more articles.",
    "Leave a comment | Back to top",
    "Follow us on Twitter and Facebook. Powered by WordPress.",
    "Visit our online store at https://example.com/shop",
    "Log in to reply | Report this post",
)


def add_boilerplate(text: str, rng: np.random.Generator) -> str:
    """``text`` between web-page furniture: up to two header lines and up to two footer lines
    (at least one line in all), each on its own line. The text itself is left as it was."""
    n_header, n_footer = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    if n_header + n_footer == 0:
        if rng.random() < 0.5:
            n_header = 1
        else:
            n_footer = 1
    headers = [
        _PAGE_HEADERS[int(i)] for i in rng.choice(len(_PAGE_HEADERS), n_header, replace=False)
    ]
    footers = [
        _PAGE_FOOTERS[int(i)] for i in rng.choice(len(_PAGE_FOOTERS), n_footer, replace=False)
    ]
    return "\n".join([*headers, text, *footers])


# --- garbled encoding --------------------------------------------------------------------------


def _misdecode(char: str) -> str:
    """What ``char`` looks like when its UTF-8 bytes are read as Windows-1252."""
    return char.encode("utf-8").decode("cp1252")


_ACCENTED = {
    "a": "äâáå", "e": "éèêë", "i": "ìïî",
    "o": "öôòó", "u": "üùûú", "n": "ñ",
    "c": "ç", "y": "ý", "'": "’", '"': "“", "-": "–—",
}  # fmt: skip
# The mojibake of an accented letter or typographic mark, per plain character it replaces; any
# other character becomes one of the generic sequences. All of them contain MOJIBAKE_CHARS.
_MOJIBAKE_OF = {plain: tuple(_misdecode(c) for c in marks) for plain, marks in _ACCENTED.items()}
_GENERIC_MOJIBAKE = (*(_misdecode(c) for c in "°´·º¼½¾«»©®±²³µ¶"), "\ufffd")


def garble(text: str, rng: np.random.Generator, rate: float = 0.08) -> str:
    """Replace about ``rate`` of the letters (and apostrophes, quotes and dashes) with mojibake,
    the junk that appears when UTF-8 text is read with the wrong encoding ("cafÃ©").

    At least one replacement is always made; text with nothing to replace gets a run of junk.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"garble rate must be between 0 and 1, got {rate}")
    candidates = [i for i, c in enumerate(text) if c.isalpha() or c in "'\"-"]
    if not candidates:
        run = rng.integers(len(_GENERIC_MOJIBAKE), size=int(rng.integers(3, 6)))
        junk = "".join(_GENERIC_MOJIBAKE[int(i)] for i in run)
        return f"{text} {junk}" if text else junk
    n = max(1, int(rng.binomial(len(candidates), rate)))
    parts = list(text)
    for i in rng.choice(candidates, size=n, replace=False):
        options = _MOJIBAKE_OF.get(text[int(i)].lower(), _GENERIC_MOJIBAKE)
        parts[int(i)] = options[int(rng.integers(len(options)))]
    return "".join(parts)


# --- false facts -------------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]*\s+")


def _insert_sentence(text: str, sentence: str, rng: np.random.Generator) -> str:
    """``text`` with ``sentence`` added at a random sentence boundary (start, between, or end)."""
    end = len(text.rstrip())
    if end == 0:
        return sentence + text
    cuts = [0, *(m.end() for m in _SENTENCE_BREAK.finditer(text) if m.end() < end), end]
    cut = cuts[int(rng.integers(len(cuts)))]
    if cut == 0:
        return f"{sentence} {text}"
    if cut == end:
        return f"{text[:end]} {sentence}{text[end:]}"
    return f"{text[:cut]}{sentence} {text[cut:]}"


def _plant_false_facts(text: str, rng: np.random.Generator, kb: KB, plan: FalseFactPlan) -> str:
    for _ in range(int(rng.integers(1, MAX_FALSE_FACTS + 1))):
        sentence, _fact = false_fact_sentence(kb, plan, rng)
        text = _insert_sentence(text, sentence, rng)
    return text


# --- injection ---------------------------------------------------------------------------------


def _log_uniform(rng: np.random.Generator, low_high: tuple[float, float]) -> float:
    low, high = low_high
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _check_rates(rates: NoiseRates) -> None:
    for name in (*_PER_DOCUMENT_KINDS, "duplicate"):
        value = getattr(rates, name)
        if not 0.0 <= value <= 1.0:  # also false for NaN
            raise ValueError(f"noise rate {name!r} must be between 0 and 1, got {value}")
    total = sum(getattr(rates, name) for name in (*_PER_DOCUMENT_KINDS, "duplicate"))
    if total > 1.0 + 1e-9:
        raise ValueError(
            f"noise rates add up to {total:.3f}, but each document gets at most one kind of "
            "noise, so they must add up to at most 1"
        )
    if rates.duplicate >= 1.0:
        raise ValueError("noise rate 'duplicate' must be below 1: the output cannot be all copies")


def _quotas(rates: NoiseRates, n_docs: int, n_out: int) -> dict[str, int]:
    """Documents per kind: each rate times the output count, but at most ``n_docs`` in all."""
    quotas = {kind: int(getattr(rates, kind) * n_out + 0.5) for kind in _PER_DOCUMENT_KINDS}
    while sum(quotas.values()) > n_docs:  # rounding up can overshoot when the rates add up to 1
        quotas[max(quotas, key=quotas.__getitem__)] -= 1
    return quotas


def _group_sizes(total: int, rng: np.random.Generator) -> list[int]:
    """Split ``total`` copies into groups of 2-6 (a lone copy only when ``total`` is 1)."""
    sizes: list[int] = []
    left = total
    while left > 0:
        if left <= MAX_COPIES:
            size = left
        else:
            size = int(rng.integers(MIN_COPIES, MAX_COPIES + 1))
            if left - size == 1:
                size -= 1
        sizes.append(size)
        left -= size
    return sizes


def inject_noise(
    texts: list[str],
    rates: NoiseRates,
    rng: np.random.Generator,
    kb: KB | None = None,
    plan: FalseFactPlan | None = None,
) -> list[NoisyDoc]:
    """Dirty ``texts`` and tag what was done; the result starts with one entry per input text, in
    order, followed by the duplicate copies.

    Rates are fractions of the output count (the inputs plus the copies). Each input document gets
    at most one kind of noise:

    * typo: some of its letters misspelled (``add_typos``);
    * spam: replaced by ``make_spam``;
    * boilerplate: wrapped in page furniture (``add_boilerplate``);
    * garbled: some characters turned into mojibake (``garble``);
    * false_fact: one to three sentences from ``plan`` inserted at sentence boundaries, and
      ``false_fact`` set.

    Duplicates are appended: groups of two to six copies of a random document that got no other
    noise, about half of them carrying one or two typos. Every document in the output is tagged
    "duplicate" for the copies and "none" for the inputs left alone.
    """
    _check_rates(rates)
    if rates.false_fact > 0:
        if kb is None or plan is None:
            raise ValueError("false_fact noise needs both the knowledge base (kb) and a plan")
        if not plan.mapping:
            raise ValueError("false_fact noise needs a non-empty plan; the given plan is empty")
    n_docs = len(texts)
    n_copies = int(n_docs * rates.duplicate / (1.0 - rates.duplicate) + 0.5)
    quotas = _quotas(rates, n_docs, n_docs + n_copies)

    kind_of: list[str | None] = [None] * n_docs
    shuffled = rng.permutation(n_docs).tolist()
    taken = 0
    for kind in _PER_DOCUMENT_KINDS:
        for i in shuffled[taken : taken + quotas[kind]]:
            kind_of[i] = kind
        taken += quotas[kind]
    untouched = [i for i in range(n_docs) if kind_of[i] is None]
    if n_copies and not untouched:
        raise ValueError(
            "the noise rates leave no clean document to copy for 'duplicate'; lower the other rates"
        )

    out: list[NoisyDoc] = []
    for text, kind in zip(texts, kind_of):
        if kind == "typo":
            new = add_typos(text, rng, _log_uniform(rng, TYPO_RATE_RANGE))
        elif kind == "spam":
            new = make_spam(rng)
        elif kind == "boilerplate":
            new = add_boilerplate(text, rng)
        elif kind == "garbled":
            new = garble(text, rng, _log_uniform(rng, GARBLE_RATE_RANGE))
        elif kind == "false_fact":
            new = _plant_false_facts(text, rng, kb, plan)
        else:
            new = text
        if kind is None or new == text:  # noise that changed nothing is not a noisy document
            out.append(NoisyDoc(text, _NONE, False))
        else:
            out.append(NoisyDoc(new, _KIND_INDEX[kind], kind == "false_fact"))

    sources = rng.permutation(untouched).tolist() if n_copies else []
    for turn, size in enumerate(_group_sizes(n_copies, rng)):
        source = texts[sources[turn % len(sources)]]
        for _ in range(size):
            copy = source
            if rng.random() < COPY_TYPO_SHARE:
                copy = _misspell(source, rng, int(rng.integers(1, 3)))
            out.append(NoisyDoc(copy, _DUPLICATE, False))
    return out
