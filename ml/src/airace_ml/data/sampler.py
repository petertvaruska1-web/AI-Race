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


# What a pool may do to a document it hands out: (tokens, the sampler's rng) -> tokens.
DocTransform = Callable[[np.ndarray, np.random.Generator], np.ndarray]

# Pool seeds are drawn below this, so that they stay exact when a state is written as JSON.
_SEED_RANGE = 2**53


class MixtureError(ValueError):
    """The requested mixture cannot be sampled. The message names the offending dataset."""


class DocPool:
    """The documents one dataset offers a model, after preparation and purchase filtering.

    Empty documents are dropped, and so (in :meth:`from_docs`) are documents with no content
    token, i.e. made only of the framing tokens (pad, bos, end, user, ai) such as the
    ``[<|bos|>]`` that encoding a blank text produces: training on those would teach nothing.
    A pool hands its documents out *without replacement*: it walks a shuffled order of them and
    shuffles again when the order is used up, so within a run no document is read twice before
    every other one has been read once. A *balanced* pool draws a topic uniformly among the topics
    that are present, and each topic walks its own shuffled order of its documents. A pool with a
    ``transform`` hands each drawn document out as ``transform(doc, rng)`` returns it (casual
    typing, see :mod:`airace_ml.data.casual`).

    The state of a pool (:meth:`state`) is a seed and, for each order, how many passes are done and
    the place in the current pass; the order of a pass is made again from the seed and the pass
    number, so the state stays a few integers however many documents there are.

    A pool made with ``replacement=True`` draws every document independently at random
    instead, as every pool did up to engine 2, and takes nothing from the sampler's generator
    when the sampler is made. The fixed evaluation batches (the trainer's held-out rows and the
    scorecard's yardstick) are drawn this way, so that they are the same rows whatever the
    engine: a held-out loss means the same before and after the training sampler changed.
    """

    def __init__(
        self,
        n_docs: int,
        fetch: Callable[[int], np.ndarray],
        topic_groups: list[np.ndarray] | None,
        transform: DocTransform | None = None,
        replacement: bool = False,
    ) -> None:
        self.replacement = replacement
        self._n = n_docs
        self._fetch = fetch  # position in [0, n_docs) -> token array
        self._groups = topic_groups  # positions per present topic; None = unbalanced
        self._transform = transform
        self._seed: int | None = None  # set by :meth:`start`, or at the first draw
        n_orders = 1 if topic_groups is None else len(topic_groups)
        self._progress = [[0, 0] for _ in range(n_orders)]  # per order: [passes done, position]
        self._cached: list[tuple[int, np.ndarray] | None] = [None] * n_orders  # (pass, order)

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        doc_ids: np.ndarray,
        balanced: bool,
        transform: DocTransform | None = None,
        replacement: bool = False,
    ) -> "DocPool":
        ids = np.asarray(doc_ids, dtype=np.int64)
        ids = ids[(corpus.offsets[ids + 1] - corpus.offsets[ids]) > 0]
        groups = None
        if balanced and len(ids):
            topics = corpus.tags.topic[ids]
            groups = [np.flatnonzero(topics == t) for t in np.unique(topics)]
        return cls(
            len(ids), lambda pos: corpus.doc(int(ids[pos])), groups, transform, replacement
        )

    @classmethod
    def from_docs(cls, docs: list[list[int]]) -> "DocPool":
        arrays = [np.asarray(d, dtype=np.uint16) for d in docs]
        arrays = [a for a in arrays if a.size and int(a.max()) >= _FRAMING_IDS]
        return cls(len(arrays), arrays.__getitem__, None)

    def __len__(self) -> int:
        return self._n

    def start(self, seed: int) -> None:
        """Fix the seed the shuffled orders come from, and begin the first pass."""
        self._seed = int(seed)
        for progress in self._progress:
            progress[:] = [0, 0]
        self._cached = [None] * len(self._progress)

    def state(self) -> dict:
        """JSON-safe snapshot of where the pool is in its orders, for exact resume."""
        return {"seed": self._seed, "orders": [list(p) for p in self._progress]}

    def set_state(self, state: dict) -> None:
        orders = state["orders"]
        if len(orders) != len(self._progress):
            raise ValueError("the saved pool state belongs to a pool of other documents")
        seed = state["seed"]
        self._seed = None if seed is None else int(seed)
        self._progress = [[int(passes), int(pos)] for passes, pos in orders]
        self._cached = [None] * len(self._progress)

    def _next_position(self, which: int, size: int) -> int:
        """The next document position of order ``which`` (a permutation of ``size`` positions)."""
        progress = self._progress[which]
        if progress[1] >= size:  # this pass is used up: the next one is another shuffle
            progress[0] += 1
            progress[1] = 0
        cached = self._cached[which]
        if cached is None or cached[0] != progress[0]:
            order = np.random.default_rng([self._seed or 0, which, progress[0]]).permutation(size)
            cached = self._cached[which] = (progress[0], order)
        pos = int(cached[1][progress[1]])
        progress[1] += 1
        return pos

    def sample_doc(self, rng: np.random.Generator) -> np.ndarray:
        if not self._n:
            raise MixtureError("cannot sample a document from an empty pool")
        if self.replacement:
            if self._groups is None:
                doc = self._fetch(int(rng.integers(self._n)))
            else:
                group = self._groups[int(rng.integers(len(self._groups)))]
                doc = self._fetch(int(group[int(rng.integers(len(group)))]))
            return doc if self._transform is None else self._transform(doc, rng)
        if self._seed is None:  # a pool used without a sampler takes its seed from the generator
            self.start(int(rng.integers(_SEED_RANGE)))
        if self._groups is None:
            doc = self._fetch(self._next_position(0, self._n))
        else:
            which = int(rng.integers(len(self._groups)))
            group = self._groups[which]
            doc = self._fetch(int(group[self._next_position(which, len(group))]))
        return doc if self._transform is None else self._transform(doc, rng)


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
        for name in sorted(self._pools):  # a run is fully determined by its seed
            if not self._pools[name].replacement:
                self._pools[name].start(int(self._rng.integers(_SEED_RANGE)))
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
        """JSON-safe snapshot of the generator, the pools and the draw counts, for exact resume."""
        return {
            "bit_generator": self._rng.bit_generator.state,
            "stats": dict(self._tokens_drawn),
            "pools": {name: pool.state() for name, pool in self._pools.items()},
        }

    def set_rng_state(self, state: dict) -> None:
        """Put the sampler back as :meth:`rng_state` found it.

        A state without the pools was saved by the training engine before documents were drawn
        without replacement; it cannot be continued, so this raises :class:`ValueError`.
        """
        if "pools" not in state:
            raise ValueError("the saved sampler state has no pools (written by an older engine)")
        saved = state["pools"]
        missing = sorted(set(self._pools) - set(saved))
        if missing:
            raise ValueError(f"the saved sampler state has no state for the pools {missing}")
        for name, pool in self._pools.items():
            pool.set_state(saved[name])
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
    transform: DocTransform | None = None,
) -> dict[str, DocPool]:
    """Build a pool for every dataset with positive weight; other datasets are never opened.

    ``purchases`` maps a dataset id to the fraction of it the player owns (default: all of it).
    Custom datasets (``notebook``, ``coaching``) come from ``custom`` token lists via
    :meth:`DocPool.from_docs` (blank documents dropped, no other filtering). ``transform`` is
    given to the base datasets' pools only: the player's own text is read as written.
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
            pools[name] = DocPool.from_corpus(
                corpus, ids, balanced=prep.variety == "balanced", transform=transform
            )
    return pools
