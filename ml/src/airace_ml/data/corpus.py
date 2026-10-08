"""On-disk corpus shards: concatenated uint16 tokens, document offsets and per-document tags.

A corpus directory holds ``tokens.bin`` (all documents concatenated, uint16), ``offsets.npy``
(int64, ``N + 1`` entries so document ``i`` is ``tokens[offsets[i]:offsets[i + 1]]``),
``tags.npz`` (one array per :class:`DocTags` field) and ``info.json`` (free-form provenance).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

DATASET_IDS: tuple[str, ...] = (
    "web",
    "books",
    "educational",
    "conversations",
    "code",
    "reasoning",
    "facts",
    "creative",
)
CUSTOM_DATASET_IDS: tuple[str, ...] = ("notebook", "coaching")
TOPICS: tuple[str, ...] = (
    "animals",
    "food",
    "nature",
    "family",
    "school",
    "science",
    "sports",
    "technology",
    "places",
    "feelings",
    "other",
)
NOISE_KINDS: tuple[str, ...] = (
    "none",
    "typo",
    "spam",
    "boilerplate",
    "garbled",
    "duplicate",
    "false_fact",
)

TOKENS_FILE = "tokens.bin"
OFFSETS_FILE = "offsets.npy"
TAGS_FILE = "tags.npz"
INFO_FILE = "info.json"
_TOKEN_DTYPE = np.dtype(np.uint16)
_TOKEN_MAX = int(np.iinfo(_TOKEN_DTYPE).max)


@dataclass
class DocTags:
    """Per-document metadata. Every field is a 1-D array with one entry per document."""

    quality: np.ndarray  # float16 in [0, 1]
    dup_cluster: np.ndarray  # int32; -1 = unique
    dup_canonical: np.ndarray  # bool; the kept copy of a duplicate cluster
    false_fact: np.ndarray  # bool
    topic: np.ndarray  # uint8 index into TOPICS
    noise_kind: np.ndarray  # uint8 index into NOISE_KINDS
    heldout: np.ndarray  # bool; reserved for evaluation, never trained on
    purchase_rank: np.ndarray  # float32 in [0, 1); a document is bought once rank < fraction


_TAG_DTYPES: dict[str, np.dtype] = {
    "quality": np.dtype(np.float16),
    "dup_cluster": np.dtype(np.int32),
    "dup_canonical": np.dtype(np.bool_),
    "false_fact": np.dtype(np.bool_),
    "topic": np.dtype(np.uint8),
    "noise_kind": np.dtype(np.uint8),
    "heldout": np.dtype(np.bool_),
    "purchase_rank": np.dtype(np.float32),
}
assert tuple(_TAG_DTYPES) == tuple(f.name for f in fields(DocTags))


def _checked_tags(tags: DocTags, n_docs: int) -> DocTags:
    """Coerce every tag to its canonical dtype and require one entry per document."""
    arrays = {}
    for name, dtype in _TAG_DTYPES.items():
        array = np.asarray(getattr(tags, name), dtype=dtype)
        if array.shape != (n_docs,):
            raise ValueError(f"tag {name!r} has shape {array.shape}, expected ({n_docs},)")
        arrays[name] = array
    return DocTags(**arrays)


def write_corpus(
    out_dir: Path,
    docs: Sequence[Sequence[int]],
    tags: DocTags,
    info: dict,
) -> None:
    """Write ``docs`` (token-id sequences) with their ``tags`` and ``info`` to ``out_dir``."""
    n_docs = len(docs)
    checked = _checked_tags(tags, n_docs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = np.zeros(n_docs + 1, dtype=np.int64)
    with (out_dir / TOKENS_FILE).open("wb") as f:
        for i, doc in enumerate(docs):
            ids = np.asarray(doc, dtype=np.int64)
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) > _TOKEN_MAX):
                raise ValueError(f"document {i} has token ids outside the uint16 range")
            ids.astype(_TOKEN_DTYPE).tofile(f)
            offsets[i + 1] = offsets[i] + ids.size
    np.save(out_dir / OFFSETS_FILE, offsets)
    np.savez(out_dir / TAGS_FILE, **{name: getattr(checked, name) for name in _TAG_DTYPES})
    (out_dir / INFO_FILE).write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


class Corpus:
    """A read-only view of a corpus directory; token data is memory-mapped, never loaded whole."""

    def __init__(self, tokens: np.ndarray, offsets: np.ndarray, tags: DocTags, info: dict) -> None:
        self._tokens = tokens
        self.offsets = offsets
        self.tags = tags
        self.info = info
        self.n_docs: int = len(offsets) - 1
        self.n_tokens: int = int(offsets[-1])

    @classmethod
    def open(cls, dir: Path) -> "Corpus":
        dir = Path(dir)
        offsets = np.load(dir / OFFSETS_FILE)
        n_tokens = int(offsets[-1])
        tokens_path = dir / TOKENS_FILE
        if tokens_path.stat().st_size != n_tokens * _TOKEN_DTYPE.itemsize:
            raise ValueError(f"{tokens_path} does not match the offsets in {OFFSETS_FILE}")
        if n_tokens:
            tokens = np.memmap(tokens_path, dtype=_TOKEN_DTYPE, mode="r", shape=(n_tokens,))
        else:
            tokens = np.zeros(0, dtype=_TOKEN_DTYPE)  # np.memmap cannot map an empty file
        with np.load(dir / TAGS_FILE) as z:
            tags = _checked_tags(
                DocTags(**{name: z[name] for name in _TAG_DTYPES}), len(offsets) - 1
            )
        info = json.loads((dir / INFO_FILE).read_text(encoding="utf-8"))
        return cls(tokens, offsets, tags, info)

    @property
    def tokens(self) -> np.ndarray:
        """Every document's token ids, concatenated (uint16; the memory map itself, read-only),
        so document ``i`` is ``tokens[offsets[i]:offsets[i + 1]]``."""
        return self._tokens

    def doc(self, i: int) -> np.ndarray:
        """Token ids of document ``i`` as a uint16 view into the memory map."""
        if not 0 <= i < self.n_docs:
            raise IndexError(f"document {i} out of range for a corpus of {self.n_docs}")
        return self._tokens[self.offsets[i] : self.offsets[i + 1]]
