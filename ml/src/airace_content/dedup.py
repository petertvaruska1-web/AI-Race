"""Near-duplicate detection: MinHash signatures, banded LSH candidates and union-find.

Two documents are in the same cluster when their normalized texts are equal, or when the
estimated Jaccard similarity of their word 3-shingles reaches the threshold; clusters are the
transitive closure of those links. Signatures use blake2b and fixed per-permutation salts, never
Python's per-process ``hash()``, so the same text always gets the same signature and a corpus
build is reproducible.
"""

import hashlib
import re
from collections.abc import Sequence
from functools import cache

import numpy as np

from airace_content.textproc import normalize_text

NUM_PERM = 64
BANDS = 16
ROWS = 4
SHINGLE_WORDS = 3
assert BANDS * ROWS == NUM_PERM

_U64_MAX = np.iinfo(np.uint64).max
_BLOCK = 4096  # shingles hashed per block, to bound memory for very long documents
# A token is a run of word characters, or a single kana, CJK ideograph or Hangul syllable (those
# scripts put no spaces between words).
_UNSPACED_SCRIPTS = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF))
_TOKEN = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _UNSPACED_SCRIPTS) + r"]|\w+"
)
_S33 = np.uint64(33)
_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)


def _canonical(text: str) -> str:
    """Normalized, case-folded text with every run of whitespace as one space."""
    return " ".join(normalize_text(text).casefold().split())


def _blake64(data: str) -> int:
    return int.from_bytes(hashlib.blake2b(data.encode("utf-8"), digest_size=8).digest(), "little")


def _mix(x: np.ndarray) -> np.ndarray:
    """The murmur3 64-bit finalizer: a bijective scramble of every element."""
    x = x ^ (x >> _S33)
    x = x * _M1
    x = x ^ (x >> _S33)
    x = x * _M2
    return x ^ (x >> _S33)


@cache
def _salts(num_perm: int) -> np.ndarray:
    return np.array([_blake64(f"minhash-v1:{i}") for i in range(num_perm)], dtype=np.uint64)


def _shingles(canonical: str) -> list[str]:
    """Word 3-shingles; a text of fewer than three tokens is a single shingle (none if empty)."""
    tokens = _TOKEN.findall(canonical) or canonical.split()
    if len(tokens) < SHINGLE_WORDS:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + SHINGLE_WORDS]) for i in range(len(tokens) - SHINGLE_WORDS + 1)]


def _shingle_hashes(canonical: str) -> np.ndarray:
    return np.fromiter({_blake64(s) for s in _shingles(canonical)}, dtype=np.uint64)


def _signature(hashes: np.ndarray, num_perm: int) -> np.ndarray:
    signature = np.full(num_perm, _U64_MAX, dtype=np.uint64)
    salts = _salts(num_perm)
    for start in range(0, len(hashes), _BLOCK):
        block = hashes[start : start + _BLOCK, None] ^ salts[None, :]
        np.minimum(signature, _mix(block).min(axis=0), out=signature)
    return signature


def minhash_signature(text: str, num_perm: int = NUM_PERM) -> np.ndarray:
    """MinHash signature (uint64, ``num_perm`` values) of the text's word 3-shingles.

    The share of equal values in two signatures estimates the Jaccard similarity of the texts'
    shingle sets. The text is normalized and case-folded first, so spacing, line endings and case
    do not matter. Text without any word gets a signature of all-maximum values.
    """
    if num_perm < 1:
        raise ValueError(f"num_perm must be at least 1, got {num_perm}")
    return _signature(_shingle_hashes(_canonical(text)), num_perm)


class _UnionFind:
    """Union-find over ``n`` items in which the smallest index of a set is its root."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _band_keys(signatures: np.ndarray, band: int) -> np.ndarray:
    """One 64-bit key per signature for its ``ROWS`` values in ``band``; equal rows, equal keys."""
    key = np.zeros(len(signatures), dtype=np.uint64)
    for row in range(band * ROWS, (band + 1) * ROWS):
        key = _mix(key ^ signatures[:, row])
    return key


def _link_bucket(
    members: list[int],
    units: list[int],
    signatures: np.ndarray,
    needed: float,
    sets: _UnionFind,
) -> None:
    """Union the members of one LSH bucket that are similar enough.

    Each member is compared with the representatives found so far (members not yet linked to an
    earlier one), so a bucket costs about one comparison per member, not one per pair.
    """
    reps: list[int] = []
    for row in members:
        for rep in reps:
            if sets.find(units[row]) == sets.find(units[rep]):
                break
            if np.count_nonzero(signatures[row] == signatures[rep]) >= needed:
                sets.union(units[row], units[rep])
                break
        else:
            reps.append(row)


def _link_near_duplicates(
    signatures: np.ndarray, units: list[int], threshold: float, sets: _UnionFind
) -> None:
    """Link the texts whose signatures share a band and agree on at least ``threshold`` of values."""
    needed = threshold * signatures.shape[1]
    for band in range(BANDS):
        keys = _band_keys(signatures, band)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        cuts = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        stops = np.concatenate((cuts, [len(order)]))
        for k in np.flatnonzero(stops - starts > 1):
            members = order[starts[k] : stops[k]].tolist()
            _link_bucket(members, units, signatures, needed, sets)


def cluster_near_duplicates(
    texts: Sequence[str], threshold: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """Group duplicates and near-duplicates: ``(cluster, canonical)``, one entry per text.

    ``cluster`` (int32) is -1 for a text with no duplicate, otherwise a cluster id; ids are dense
    from 0 in order of first occurrence. ``canonical`` (bool) is True for singletons and for the
    first occurrence of each cluster, so keeping ``(cluster < 0) | canonical`` keeps one text per
    cluster.

    Texts are linked when their normalized texts are equal (exact matches), or when they share a
    band of the signature (16 bands of 4 rows) and the estimated Jaccard similarity of their word
    3-shingles is at least ``threshold``. Links are merged transitively.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    n = len(texts)
    cluster = np.full(n, -1, dtype=np.int32)
    canonical = np.ones(n, dtype=np.bool_)
    if n == 0:
        return cluster, canonical

    # Exact matches: every distinct normalized text gets one id (its "unit").
    unit_of_text: dict[bytes, int] = {}
    unit_of_doc = np.empty(n, dtype=np.int64)
    signatures: list[np.ndarray] = []
    units: list[int] = []  # the unit of each signature (texts without words have none)
    for i, text in enumerate(texts):
        canonical_text = _canonical(text)
        key = hashlib.blake2b(canonical_text.encode("utf-8"), digest_size=16).digest()
        unit = unit_of_text.get(key)
        if unit is None:
            unit = unit_of_text[key] = len(unit_of_text)
            hashes = _shingle_hashes(canonical_text)
            if len(hashes):
                signatures.append(_signature(hashes, NUM_PERM))
                units.append(unit)
        unit_of_doc[i] = unit

    # Near matches between the distinct texts.
    sets = _UnionFind(len(unit_of_text))
    if len(signatures) > 1:
        _link_near_duplicates(np.stack(signatures), units, threshold, sets)

    root_of_unit = np.array([sets.find(u) for u in range(len(unit_of_text))], dtype=np.int64)
    root_of_doc = root_of_unit[unit_of_doc]
    size = np.bincount(root_of_doc, minlength=len(unit_of_text))
    cluster_ids: dict[int, int] = {}
    for i, root in enumerate(root_of_doc.tolist()):
        if size[root] < 2:
            continue
        cluster_id = cluster_ids.get(root)
        if cluster_id is None:
            cluster_id = cluster_ids[root] = len(cluster_ids)
        else:
            canonical[i] = False
        cluster[i] = cluster_id
    return cluster, canonical
