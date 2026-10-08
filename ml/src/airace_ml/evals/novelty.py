"""The 8-gram novelty index: how much of a text is new compared with everything a model could read.

Every window of ``n`` consecutive token ids gets a polynomial hash (base 1,000,003, mod 2**64).
Only windows whose hash is divisible by ``sample_mod`` are kept. That is a sample chosen by
content, so a given window is either always kept or always dropped, in the corpora and in a
model's text alike, and the index is ``sample_mod`` times smaller than the full set.

The index of a data root (:func:`build_novelty_index`) holds the sorted, unique sampled hashes of
every document of the eight base corpora that is not held out, with no window spanning two
documents. :meth:`NoveltyIndex.novelty` is the fraction of a text's sampled windows that are not in
it: 0 for text copied from the corpora, near 1 for text no corpus contains.

On disk an index is a ``.npy`` array of the hashes plus a ``.json`` file beside it that records
how they were made.
"""

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.prep import heldout_docs
from airace_ml.paths import corpus_dir, novelty_path

HASH_BASE = 1_000_003
NGRAM = 8
SAMPLE_MOD = 8
_CHUNK = 1 << 22  # windows hashed at once while building: 32 MiB of hashes
_EMPTY = np.zeros(0, np.uint64)


def ngram_hashes(ids: Sequence[int] | np.ndarray, n: int = NGRAM) -> np.ndarray:
    """The hash of every window of ``n`` consecutive ids, in order (uint64; empty when there are
    fewer than ``n`` ids).

    The hash of ``ids[i:i + n]`` is ``sum(ids[i + j] * HASH_BASE ** (n - 1 - j))`` mod 2**64,
    computed with numpy's wrapping uint64 arithmetic over all windows at once.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError(f"ids must be one flat sequence of token ids, got shape {ids.shape}")
    count = ids.size - n + 1
    if count <= 0:
        return _EMPTY.copy()
    ids = ids.astype(np.uint64, copy=False)
    base = np.uint64(HASH_BASE)
    hashes = ids[:count].copy()
    for j in range(1, n):  # array arithmetic wraps mod 2**64 silently; only scalars would warn
        hashes *= base
        hashes += ids[j : j + count]
    return hashes


def _check_params(n: int, sample_mod: int) -> None:
    for name, value in (("n", n), ("sample_mod", sample_mod)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _sampled(hashes: np.ndarray, sample_mod: int) -> np.ndarray:
    return hashes[hashes % np.uint64(sample_mod) == 0]


def _sampled_hashes(
    tokens: np.ndarray, n: int, sample_mod: int, valid: np.ndarray | None = None
) -> np.ndarray:
    """The sampled hashes of the windows of ``tokens`` (those starting where ``valid`` is True,
    if given), hashed :data:`_CHUNK` windows at a time so memory stays bounded."""
    count = len(tokens) - n + 1
    parts = []
    for lo in range(0, max(count, 0), _CHUNK):
        hi = min(lo + _CHUNK, count)
        hashes = ngram_hashes(tokens[lo : hi + n - 1], n)
        keep = hashes % np.uint64(sample_mod) == 0
        if valid is not None:
            keep &= valid[lo:hi]
        parts.append(hashes[keep])
    return np.concatenate(parts) if parts else _EMPTY.copy()


def _valid_starts(offsets: np.ndarray, keep_doc: np.ndarray, n: int) -> np.ndarray:
    """For every window start in the concatenated tokens: True when the window lies wholly
    inside one document that ``keep_doc`` keeps."""
    count = int(offsets[-1]) - n + 1
    if count <= 0:
        return np.zeros(0, bool)
    starts, ends = offsets[:-1][keep_doc], offsets[1:][keep_doc]
    fits = ends - starts >= n
    # Each document of n or more tokens opens a run of valid starts at its first token and closes
    # it n - 1 tokens before its end. Documents never overlap, so the running sum is 0 or 1.
    delta = np.zeros(count + 1, np.int8)
    delta[starts[fits]] += 1
    delta[ends[fits] - n + 1] -= 1
    return np.cumsum(delta[:count], dtype=np.int8).astype(bool)


def _unique(parts: Iterable[np.ndarray]) -> np.ndarray:
    parts = list(parts)
    return np.unique(np.concatenate(parts)) if parts else _EMPTY.copy()


def _meta_path(path: Path) -> Path:
    return Path(path).with_suffix(".json")


class NoveltyIndex:
    """Sorted, unique sampled ``n``-gram hashes of a body of text.

    ``hashes`` are sorted and made unique here if they are not already.
    """

    def __init__(self, hashes: np.ndarray, n: int = NGRAM, sample_mod: int = SAMPLE_MOD) -> None:
        _check_params(n, sample_mod)
        hashes = np.asarray(hashes, dtype=np.uint64).reshape(-1)
        if hashes.size > 1 and not np.all(hashes[1:] > hashes[:-1]):
            hashes = np.unique(hashes)
        self.hashes = hashes
        self.n = n
        self.sample_mod = sample_mod

    def __len__(self) -> int:
        return int(self.hashes.size)

    @classmethod
    def build(
        cls, token_arrays: Iterable[np.ndarray], n: int = NGRAM, sample_mod: int = SAMPLE_MOD
    ) -> "NoveltyIndex":
        """The index of ``token_arrays``; no window spans two arrays."""
        _check_params(n, sample_mod)
        parts = (_sampled_hashes(np.asarray(a), n, sample_mod) for a in token_arrays)
        return cls(_unique(parts), n, sample_mod)

    def save(self, path: Path) -> None:
        """Write the hashes to ``path`` (``.npy``) and how they were made to the ``.json`` file
        beside it. The array is written to a temporary file first and then renamed into place."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "hash_base": HASH_BASE,
            "n": self.n,
            "sample_mod": self.sample_mod,
            "count": len(self),
        }
        _meta_path(path).write_text(json.dumps(meta), encoding="utf-8")
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:  # a file object, so numpy adds no ".npy" to the name
            np.save(f, self.hashes)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "NoveltyIndex":
        path = Path(path)
        meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
        if meta.get("hash_base") != HASH_BASE:
            raise ValueError(
                f"{path} was hashed with base {meta.get('hash_base')!r}, not {HASH_BASE}"
            )
        return cls(np.load(path), meta["n"], meta["sample_mod"])

    def novelty(self, ids: Sequence[int] | np.ndarray) -> float:
        """The fraction of the sampled windows of ``ids`` absent from the index (each occurrence
        counts); 0.0 when ``ids`` has no sampled window."""
        sampled = _sampled(ngram_hashes(ids, self.n), self.sample_mod)
        if sampled.size == 0:
            return 0.0
        if self.hashes.size == 0:
            return 1.0
        pos = np.minimum(np.searchsorted(self.hashes, sampled), self.hashes.size - 1)
        absent = np.count_nonzero(self.hashes[pos] != sampled)
        return absent / sampled.size


def build_novelty_index(data_root: Path) -> Path:
    """Index every document of the eight base corpora under ``data_root`` that is not held out,
    write it to :func:`~airace_ml.paths.novelty_path` and return that path."""
    parts = []
    for name in DATASET_IDS:
        corpus = Corpus.open(corpus_dir(data_root) / name)
        keep = np.ones(corpus.n_docs, bool)
        keep[heldout_docs(corpus)] = False
        valid = _valid_starts(corpus.offsets, keep, NGRAM)
        parts.append(_sampled_hashes(corpus.tokens, NGRAM, SAMPLE_MOD, valid))
    path = novelty_path(data_root)
    NoveltyIndex(_unique(parts)).save(path)
    return path
