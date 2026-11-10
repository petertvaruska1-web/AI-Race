"""Casual-typing augmentation of chat documents (experiment): some user turns are rewritten the
way people really type (lowercase, no final punctuation); the AI's replies stay as they are."""

import numpy as np

_END_PUNCT = ".?!"


def casual_text(text: str, style: float) -> str:
    """One user turn, typed casually. ``style`` in [0, 1) picks how."""
    if style < 0.40:  # all lowercase
        return text.lower()
    if style < 0.70:  # all lowercase, no final punctuation
        return text.lower().rstrip().rstrip(_END_PUNCT).rstrip()
    if style < 0.90:  # no final punctuation
        return text.rstrip().rstrip(_END_PUNCT).rstrip()
    return text[:1].lower() + text[1:]  # lowercase first letter only


def casual_doc(doc: np.ndarray, tok, style: float) -> np.ndarray:
    ids = doc.tolist()
    user, end = tok.user_id, tok.end_id
    if user not in ids:
        return doc
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        t = ids[i]
        out.append(t)
        i += 1
        if t == user:
            j = i
            while j < n and ids[j] >= 16:  # up to the next special token
                j += 1
            new = casual_text(tok.decode(ids[i:j]), style)
            if new.strip():
                out.extend(tok.encode(new))
            else:
                out.extend(ids[i:j])
            i = j
    return np.asarray(out, dtype=np.uint16)


def wrap_pools(pools: dict, tok, share: float) -> None:
    """Make every pool hand out a casual version of a chat document with probability ``share``."""
    for pool in pools.values():
        orig = pool.sample_doc

        def sample_doc(rng, _orig=orig):
            doc = _orig(rng)
            u = rng.random()
            if u < share:
                return casual_doc(doc, tok, u / share)
            return doc

        pool.sample_doc = sample_doc
