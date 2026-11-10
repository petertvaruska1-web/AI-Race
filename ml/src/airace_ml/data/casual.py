"""Casual typing: chat documents are sometimes read with their user turns typed the way people
really type, so an AI learns to answer "what is the capital of france" as it answers "What is the
capital of France?".

The conversations in the corpora are written carefully: 99.5% of their user turns start with a
capital letter and every one ends with punctuation. Players do not type like that, and a model
that has only ever read careful turns answers a lowercase "hello" (three rare tokens instead of
one common one) with gibberish. So when the sampler hands out a document with user turns, with
probability :data:`CASUAL_SHARE` every user turn of that document is retyped casually. The AI's
replies are never touched: the lesson is "the same answer to the same question, however it is
typed".

It is the same conversation with real-world typing, applied alike to every base dataset (never to
a player's notebook or lessons, which are trained as written), decided by the run's own random
stream, and part of the training engine's version. Measured (ADR 0009): a first AI answers 74% of
lowercase fact questions instead of 62% (75% instead of 59% without the question mark), with
clean questions unchanged.
"""

from __future__ import annotations

import numpy as np

from airace_ml.tokenizer import SPECIAL_TOKENS, Tok

CASUAL_SHARE = 0.3  # of the chat documents a run reads
_FIRST_TEXT_ID = len(SPECIAL_TOKENS)  # ids below this are special tokens
_END_PUNCTUATION = ".?!"

# How a casual turn is typed, by share (cumulative bounds over a uniform draw in [0, 1)). The
# first three keep the first letter: the Playground capitalises it before the AI sees a message.
_REST_LOWER = 0.40  # "What is the capital of france?"
_REST_LOWER_NO_END = 0.70  # "What is the capital of france"
_NO_END = 0.85  # "What is the capital of France"
_ALL_LOWER = 0.925  # "what is the capital of france?"; above it also without the "?"


def _without_end(text: str) -> str:
    stripped = text.rstrip()
    cut = stripped.rstrip(_END_PUNCTUATION).rstrip()
    return cut if cut else text  # a turn that is only punctuation stays as it is


def casual_text(text: str, style: float) -> str:
    """``text`` (one user turn) typed casually; ``style`` in [0, 1) picks how. A turn is never
    emptied."""
    if style < _REST_LOWER:
        return text[:1] + text[1:].lower()
    if style < _REST_LOWER_NO_END:
        return _without_end(text[:1] + text[1:].lower())
    if style < _NO_END:
        return _without_end(text)
    if style < _ALL_LOWER:
        return text.lower()
    return _without_end(text.lower())


def casual_doc(doc: np.ndarray, tok: Tok, style: float) -> np.ndarray:
    """``doc`` (token ids) with every user turn retyped by :func:`casual_text`; everything else,
    the AI's replies above all, token for token as it was. A document without user turns is
    returned as it is (the same array)."""
    ids = doc.tolist()
    if tok.user_id not in ids:
        return doc
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        token = ids[i]
        out.append(token)
        i += 1
        if token != tok.user_id:
            continue
        end = i
        while end < n and ids[end] >= _FIRST_TEXT_ID:  # the turn's text: up to the next special
            end += 1
        if end > i:
            out.extend(tok.encode(casual_text(tok.decode(ids[i:end]), style)))
        i = end
    return np.asarray(out, dtype=doc.dtype)


class CasualTyping:
    """The sampler's document transform: with probability ``share`` a document is handed out as
    :func:`casual_doc` types it. Exactly one number is drawn from ``rng`` per document, whether
    or not it has user turns, so a run's random stream does not depend on what it happens to read.
    """

    def __init__(self, tok: Tok, share: float = CASUAL_SHARE) -> None:
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"share must be in [0, 1], got {share!r}")
        self.tok = tok
        self.share = share

    def __call__(self, doc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        u = rng.random()
        if u >= self.share:
            return doc
        return casual_doc(doc, self.tok, u / self.share)  # u / share is uniform in [0, 1) again
