"""Mixture sampling: turns per-dataset document pools and mixture weights into training batches."""

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from airace_ml.data.corpus import CUSTOM_DATASET_IDS, DATASET_IDS, Corpus
from airace_ml.data.prep import PrepConfig, eligible_docs
from airace_ml.paths import corpus_dir
from airace_ml.tokenizer import SPECIAL_TOKENS

# Ids below this are framing tokens (<|pad|> <|bos|> <|end|> <|user|> <|ai|>) that carry no text.
# A document made only of them is what encoding blank text produces, e.g. `[<|bos|>]`.
_FRAMING_IDS = SPECIAL_TOKENS.index("<|ai|>") + 1


class MixtureError(ValueError):
    """The requested mixture cannot be sampled. The message names the offending dataset."""


class DocPool:
    """The documents one dataset offers a model, after preparation and purchase filtering.

    Empty documents are dropped, and so (in :meth:`from_docs`) are documents with no content
    token, i.e. made only of the framing tokens (pad, bos, end, user, ai) such as the
    ``[<|bos|>]`` that encoding a blank text produces: training on those would teach nothing.
    A *balanced* pool draws a topic uniformly among the topics that are present, then a document
    of that topic; otherwise every document is equally likely.
    """

    def __init__(
        self,
        n_docs: int,
        fetch: Callable[[int], np.ndarray],
        topic_groups: list[np.ndarray] | None,
    ) -> None:
        self._n = n_docs
        self._fetch = fetch  # position in [0, n_docs) -> token array
        self._groups = topic_groups  # positions per present topic; None = unbalanced

    @classmethod
    def from_corpus(cls, corpus: Corpus, doc_ids: np.ndarray, balanced: bool) -> "DocPool":
        ids = np.asarray(doc_ids, dtype=np.int64)
        ids = ids[(corpus.offsets[ids + 1] - corpus.offsets[ids]) > 0]
        groups = None
        if balanced and len(ids):
            topics = corpus.tags.topic[ids]
            groups = [np.flatnonzero(topics == t) for t in np.unique(topics)]
        return cls(len(ids), lambda pos: corpus.doc(int(ids[pos])), groups)

    @classmethod
    def from_docs(cls, docs: list[list[int]]) -> "DocPool":
        arrays = [np.asarray(d, dtype=np.uint16) for d in docs]
        arrays = [a for a in arrays if a.size and int(a.max()) >= _FRAMING_IDS]
        return cls(len(arrays), arrays.__getitem__, None)

    def __len__(self) -> int:
        return self._n

    def sample_doc(self, rng: np.random.Generator) -> np.ndarray:
        if not self._n:
            raise MixtureError("cannot sample a document from an empty pool")
        if self._groups is None:
            return self._fetch(int(rng.integers(self._n)))
        group = self._groups[int(rng.integers(len(self._groups)))]
        return self._fetch(int(group[int(rng.integers(len(group)))]))


class MixtureSampler:
    """Draws fixed-length training rows: a dataset by weight, then documents packed to length.

    Every row is built from one dataset. A document longer than a row contributes a random
    window; shorter documents are appended (each freshly sampled from the same pool) until the
    row is full. ``stats()`` counts the tokens placed in each dataset's rows.
    """

    def __init__(
        self,
        pools: Mapping[str, DocPool],
        weights: Mapping[str, float],
        seq_len: int,
        seed: int,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")
        self._pools = dict(pools)
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)
        self._tokens_drawn = {name: 0 for name in self._pools}
        self._names: list[str] = []
        self._probs = np.zeros(0)
        self.set_weights(weights)

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Switch the mixture. Raises :class:`MixtureError` and keeps the old mixture if invalid."""
        for name, w in weights.items():
            if not math.isfinite(w) or w < 0:
                raise MixtureError(f"dataset {name!r} has an invalid weight {w!r}")
            if w > 0 and name not in self._pools:
                raise MixtureError(f"dataset {name!r} has weight but was not loaded (no pool)")
        total = float(sum(weights.values()))
        if not 0 < total < math.inf:
            raise MixtureError(f"mixture weights must have a positive, finite sum, got sum {total}")
        names = sorted(name for name, w in weights.items() if w > 0)
        for name in names:
            if not len(self._pools[name]):
                raise MixtureError(f"dataset {name!r} has weight but no usable documents")
        self._names = names
        self._probs = np.array([weights[name] for name in names], dtype=np.float64) / total

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return int64 ``(batch_size, seq_len)`` inputs and targets (inputs shifted by one)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        need = self.seq_len + 1
        rows = np.empty((batch_size, need), dtype=np.int64)
        for r, k in enumerate(self._rng.choice(len(self._names), size=batch_size, p=self._probs)):
            name = self._names[k]
            self._fill_row(rows[r], self._pools[name])
            self._tokens_drawn[name] += self.seq_len
        return np.ascontiguousarray(rows[:, :-1]), np.ascontiguousarray(rows[:, 1:])

    def _fill_row(self, row: np.ndarray, pool: DocPool) -> None:
        need = len(row)
        filled = 0
        while filled < need:
            doc = pool.sample_doc(self._rng)
            if filled == 0 and len(doc) > need:
                start = int(self._rng.integers(len(doc) - need + 1))
                doc = doc[start : start + need]
            take = min(len(doc), need - filled)
            row[filled : filled + take] = doc[:take]
            filled += take

    def stats(self) -> dict[str, int]:
        """Tokens drawn so far per dataset (every dataset of the pools appears, possibly at 0)."""
        return dict(self._tokens_drawn)

    def rng_state(self) -> dict:
        """JSON-safe snapshot of the generator and the draw counts, for exact resume."""
        return {"bit_generator": self._rng.bit_generator.state, "stats": dict(self._tokens_drawn)}

    def set_rng_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator"]
        self._tokens_drawn = {name: 0 for name in self._pools} | {
            name: int(n) for name, n in state["stats"].items()
        }


def load_pools(
    root: Path,
    weights: Mapping[str, float],
    prep: PrepConfig,
    purchases: Mapping[str, float],
    custom: Mapping[str, list[list[int]]],
) -> dict[str, DocPool]:
    """Build a pool for every dataset with positive weight; other datasets are never opened.

    ``purchases`` maps a dataset id to the fraction of it the player owns (default: all of it).
    Custom datasets (``notebook``, ``coaching``) come from ``custom`` token lists via
    :meth:`DocPool.from_docs` (blank documents dropped, no other filtering).
    """
    known = DATASET_IDS + CUSTOM_DATASET_IDS
    for name, w in weights.items():
        if name not in known:
            raise MixtureError(f"unknown dataset {name!r}; expected one of {list(known)}")
        if not math.isfinite(w) or w < 0:
            raise MixtureError(f"dataset {name!r} has an invalid weight {w!r}")
    pools: dict[str, DocPool] = {}
    for name, w in weights.items():
        if w <= 0:
            continue
        if name in CUSTOM_DATASET_IDS:
            if name not in custom:
                raise MixtureError(
                    f"dataset {name!r} has weight but no custom documents were given"
                )
            pools[name] = DocPool.from_docs(custom[name])
        else:
            corpus = Corpus.open(corpus_dir(root) / name)
            ids = eligible_docs(corpus, prep, purchases.get(name, 1.0))
            pools[name] = DocPool.from_corpus(corpus, ids, balanced=prep.variety == "balanced")
    return pools
