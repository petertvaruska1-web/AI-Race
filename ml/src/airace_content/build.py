"""The corpus build: from sources and generators to the 8 tagged, tokenized training corpora.

``build_corpus`` runs the pipeline once per corpus version:

1. ``simple_vocab``: the 6,000 most frequent words of TinyStories text;
2. a plan of 300 false facts, told the same way in every dataset;
3. each dataset assembled from its recipe (``assemble.assemble_dataset``);
4. ``known_vocab``: the 30,000 most frequent words of the books, educational, facts and creative
   text;
5. a quality score for every document (``structured_quality`` for code and reasoning, see
   ``STRUCTURED_DATASETS``);
6. near-duplicate clusters per dataset;
7. a topic per document (the generator's topic, else ``tag_topic``);
8. held-out documents for evaluation: whole duplicate clusters (and singletons), drawn in a seeded
   order until 1% of the documents (at least 8) are held out;
9. ``purchase_rank`` from a second seeded permutation;
10. the tokenizer, trained only when it is missing or ``retrain_tokenizer`` is set (it is frozen
    once built);
11. the documents encoded and written with their tags;
12. the manifest and the shared artifacts.

Chats count as their turn texts joined by newlines wherever a document is measured (quality,
duplicates, topic, tokenizer text). Every random choice comes from a stream named after its use
and derived from ``seed``, so the same seed gives the same corpus, whichever datasets a build
includes.
"""

import hashlib
import math
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from airace_content.assemble import (
    RECIPES,
    SCALES,
    AssembledDoc,
    assemble_dataset,
    content_text,
    estimate_tokens,
)
from airace_content.dedup import cluster_near_duplicates
from airace_content.manifest import (
    FALSE_FACTS_FILE,
    KNOWN_VOCAB_FILE,
    SIMPLE_VOCAB_FILE,
    make_manifest,
    read_false_facts,
    read_manifest,
    read_vocab,
    write_false_facts,
    write_manifest,
    write_vocab,
)
from airace_content.sources import SOURCES, Fetch, row_to_content
from airace_content.textproc import (
    build_vocab,
    normalize_text,
    quality_score,
    structured_quality,
)
from airace_content.topics import tag_topic
from airace_ml.data.corpus import DATASET_IDS, NOISE_KINDS, TOPICS, DocTags, write_corpus
from airace_ml.paths import CORPUS_VERSION, TOKENIZER_VERSION, corpus_dir, tokenizer_path
from airace_ml.skills.facts import plan_false_facts
from airace_ml.skills.kb import load_kb
from airace_ml.tokenizer import VOCAB_SIZE, Tok, encode_chat, encode_doc, train_tokenizer

SIMPLE_VOCAB_SIZE = 6_000
SIMPLE_VOCAB_ROWS = {"tiny": 5_000, "full": 200_000}  # TinyStories rows read for simple_vocab
KNOWN_VOCAB_SIZE = 30_000
KNOWN_VOCAB_DATASETS: tuple[str, ...] = ("books", "educational", "facts", "creative")
N_FALSE_FACTS = 300
HELDOUT_SHARE = 0.01
HELDOUT_MIN_DOCS = 8
HELDOUT_MAX_SHARE = 0.5  # a cluster that would push the held-out share past this is skipped
TOKENIZER_SAMPLE_BYTES = 20_000_000
# Datasets of programs and puzzles, scored by ``structured_quality``: cleaning is meant to remove
# noise, not a dataset's normal content, and clean code and puzzles fail the prose measures (known
# English words, share of letters) by their nature. Neither dataset gets typo, garble or spam
# noise (code gets duplicates only, reasoning none). Every other dataset uses ``quality_score``.
STRUCTURED_DATASETS: frozenset[str] = frozenset({"code", "reasoning"})


class BuildError(ValueError):
    """A build that cannot run as asked (the message says why and what to do)."""


@dataclass
class BuildSummary:
    datasets: dict[str, dict]
    tokenizer_trained: bool
    seconds: float


def stream(seed: int, name: str) -> np.random.Generator:
    """The random stream named ``name`` of a build seeded with ``seed``."""
    key = int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")
    return np.random.default_rng([seed, key])


def _units(dup_cluster: np.ndarray, origins: Sequence[bytes] | None) -> np.ndarray:
    """A unit id per document: its duplicate cluster, or the document alone, where documents of
    one origin (one text before noise: a typo, garbled or boilerplate variant, a copy) share a
    unit even when the clustering did not find them alike."""
    n_clusters = int(dup_cluster.max()) + 1 if (dup_cluster >= 0).any() else 0
    singles = dup_cluster < 0
    unit = np.where(singles, n_clusters + np.cumsum(singles) - 1, dup_cluster).astype(np.int64)
    if not origins:
        return unit
    parent = list(range(int(unit.max()) + 1))

    def root(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    first_with: dict[bytes, int] = {}
    for i, origin in enumerate(origins):
        first = first_with.setdefault(origin, i)
        if first != i:
            a, b = root(int(unit[i])), root(int(unit[first]))
            parent[max(a, b)] = min(a, b)
    return np.array([root(int(u)) for u in unit], dtype=np.int64)


def heldout_mask(
    dup_cluster: np.ndarray,
    rng: np.random.Generator,
    origins: Sequence[bytes] | None = None,
) -> np.ndarray:
    """Which documents are held out for evaluation.

    The units are whole duplicate clusters (every member) and single documents, where every
    document of one origin (``assemble.origin_key``: one text before noise) is in the same unit.
    Units are taken in a random order until ``max(HELDOUT_MIN_DOCS, ceil(HELDOUT_SHARE * n))``
    documents are held out, so no held-out text keeps a copy or a noised variant in training. A
    unit that would hold out more than ``HELDOUT_MAX_SHARE`` of the documents is skipped, so
    training always keeps the larger part.
    """
    n = len(dup_cluster)
    held = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return held
    unit = _units(dup_cluster, origins)
    sizes = np.bincount(unit)
    target = max(HELDOUT_MIN_DOCS, math.ceil(HELDOUT_SHARE * n))
    cap = int(HELDOUT_MAX_SHARE * n)
    chosen: list[int] = []
    count = 0
    for u in rng.permutation(np.flatnonzero(sizes)).tolist():
        if count >= target:
            break
        if count + sizes[u] <= cap:
            chosen.append(u)
            count += int(sizes[u])
    held[np.isin(unit, chosen)] = True
    return held


def purchase_ranks(n: int, rng: np.random.Generator) -> np.ndarray:
    """Each document's place in a random order, as ``rank / n`` (float32 in [0, 1))."""
    rank = np.empty(n, dtype=np.int64)
    rank[rng.permutation(n)] = np.arange(n)
    below_one = np.nextafter(np.float32(1), np.float32(0))
    return np.minimum((rank / max(n, 1)).astype(np.float32), below_one)


def tag_documents(
    docs: list[AssembledDoc], texts: list[str], known_vocab: set[str], seed: int, dataset_id: str
) -> DocTags:
    """Steps 5-9 for one dataset (quality by ``structured_quality`` for ``STRUCTURED_DATASETS``)."""
    if dataset_id in STRUCTURED_DATASETS:
        quality = [structured_quality(t) for t in texts]
    else:
        quality = [quality_score(t, known_vocab) for t in texts]
    cluster, canonical = cluster_near_duplicates(texts)
    return DocTags(
        quality=np.array(quality, dtype=np.float16),
        dup_cluster=cluster,
        dup_canonical=canonical,
        false_fact=np.array([d.false_fact for d in docs], dtype=np.bool_),
        topic=np.array(
            [d.topic if d.topic is not None else tag_topic(t) for d, t in zip(docs, texts)],
            dtype=np.uint8,
        ),
        noise_kind=np.array([d.noise_kind for d in docs], dtype=np.uint8),
        heldout=heldout_mask(
            cluster, stream(seed, f"heldout/{dataset_id}"), [d.origin for d in docs]
        ),
        purchase_rank=purchase_ranks(len(docs), stream(seed, f"purchase/{dataset_id}")),
    )


def tokenizer_texts(
    texts: dict[str, list[str]],
    heldout: dict[str, np.ndarray],
    seed: int,
    budget_bytes: int = TOKENIZER_SAMPLE_BYTES,
) -> list[str]:
    """Up to ``budget_bytes`` of training text, an equal share from each dataset, drawn in a seeded
    order; held-out documents are never used."""
    share = budget_bytes // max(1, len(texts))
    sample: list[str] = []
    for ds, ds_texts in texts.items():
        used = 0
        for i in stream(seed, f"tokenizer/{ds}").permutation(len(ds_texts)).tolist():
            size = len(ds_texts[i].encode("utf-8"))
            if heldout[ds][i] or used + size > share:
                continue
            sample.append(ds_texts[i])
            used += size
    return sample


def _simple_vocab(scale: str, fetch: Fetch) -> set[str]:
    spec = SOURCES["tinystories"]
    texts = [
        normalize_text(piece)
        for row in fetch(spec, SIMPLE_VOCAB_ROWS[scale])
        for piece in row_to_content(spec.id, row)
        if isinstance(piece, str)
    ]
    if not texts:
        raise BuildError("TinyStories gave no text, so there is no simple vocabulary to build")
    return build_vocab(texts, SIMPLE_VOCAB_SIZE)


def _selection(datasets: Iterable[str]) -> list[str]:
    wanted = list(datasets)
    unknown = sorted(set(wanted) - set(DATASET_IDS))
    if unknown:
        raise BuildError(f"unknown dataset(s) {unknown}; expected some of {list(DATASET_IDS)}")
    if not wanted:
        raise BuildError("no datasets to build")
    return [ds for ds in DATASET_IDS if ds in wanted]


def _encode(tok: Tok, doc: AssembledDoc) -> np.ndarray:
    """The document's token ids as a compact uint16 array (a full dataset as Python int lists
    would take about 40 bytes per token)."""
    if isinstance(doc.content, str):
        ids = encode_doc(tok, doc.content)
    else:
        ids = encode_chat(tok, doc.content)
    return np.array(ids, dtype=np.uint16)


def _entry(
    scale: str,
    seed: int,
    stats: dict,
    tags: DocTags,
    component_tokens: Counter,
    dataset_id: str,
) -> dict:
    """What the manifest and the summary record about one built dataset; each component also gets
    its actual ``tokens`` in the written corpus (with its noise and copies)."""
    n_tokens = sum(component_tokens.values())
    entry: dict = {"scale": scale, "seed": seed, "docs": len(tags.quality), "tokens": n_tokens}
    if scale == "full":
        entry["target_tokens"] = SCALES["full"]["target_tokens"][dataset_id]
    noise_counts = Counter(tags.noise_kind.tolist())
    topic_counts = Counter(tags.topic.tolist())
    return entry | {
        "components": [
            comp | {"tokens": component_tokens[comp["name"]]} for comp in stats["components"]
        ],
        "noise_rates": asdict(RECIPES[dataset_id].noise),
        "noise_counts": {kind: noise_counts[i] for i, kind in enumerate(NOISE_KINDS)},
        "false_fact_docs": int(tags.false_fact.sum()),
        "heldout_docs": int(tags.heldout.sum()),
        "dup_clusters": int(tags.dup_cluster.max()) + 1 if (tags.dup_cluster >= 0).any() else 0,
        "dup_extra_docs": int((~tags.dup_canonical).sum()),
        "topics": {topic: topic_counts[i] for i, topic in enumerate(TOPICS)},
    }


def build_corpus(
    scale: Literal["tiny", "full"],
    out_root: Path,
    fetch: Fetch,
    *,
    seed: int = 0,
    datasets: Sequence[str] = DATASET_IDS,
    retrain_tokenizer: bool = False,
) -> BuildSummary:
    """Build the corpora of ``datasets`` under ``corpus_dir(out_root)`` (see the module docstring).

    A build of only some datasets reuses the shared artifacts of the corpus when they are there
    (``false_facts.json``, ``simple_vocab.txt``, ``known_vocab.txt``), so the rebuilt datasets tell
    the same falsehoods and are scored like the others, and merges its datasets into the existing
    manifest. The tokenizer is trained on every dataset's text, so a build that has to train it
    must build every dataset.
    """
    started = time.perf_counter()
    if scale not in SCALES:
        raise BuildError(f"unknown scale {scale!r}; expected one of {list(SCALES)}")
    if seed < 0:
        raise BuildError(f"seed must not be negative, got {seed}")
    selected = _selection(datasets)
    partial = len(selected) < len(DATASET_IDS)
    out_root = Path(out_root)
    root, tok_path = corpus_dir(out_root), tokenizer_path(out_root)
    train_tok = retrain_tokenizer or not tok_path.exists()
    if train_tok and partial:
        reason = (
            "retraining it"
            if retrain_tokenizer
            else f"there is none yet at {tok_path}, and building it"
        )
        raise BuildError(
            f"the tokenizer is trained on the text of every dataset; {reason} needs a build of "
            "all datasets (leave out --datasets)"
        )
    frozen = None if train_tok else Tok.load(tok_path)

    def reusable(name: str) -> Path | None:
        path = root / name
        return path if partial and path.exists() else None

    kb = load_kb()
    path = reusable(SIMPLE_VOCAB_FILE)
    simple_vocab = read_vocab(path) if path else _simple_vocab(scale, fetch)
    path = reusable(FALSE_FACTS_FILE)
    plan = (
        read_false_facts(path)
        if path
        else plan_false_facts(kb, stream(seed, "false_facts"), N_FALSE_FACTS)
    )
    known_path = reusable(KNOWN_VOCAB_FILE)

    # Datasets the known vocabulary needs are assembled even when they are not written.
    needed = set(selected) | (set() if known_path else set(KNOWN_VOCAB_DATASETS))
    count_tokens = (lambda text: len(frozen.encode(text))) if frozen else estimate_tokens
    assembled: dict[str, list[AssembledDoc]] = {}
    stats: dict[str, dict] = {}
    for ds in (ds for ds in DATASET_IDS if ds in needed):
        stats[ds] = {}
        assembled[ds] = assemble_dataset(
            ds,
            scale,
            fetch,
            kb,
            simple_vocab,
            plan,
            stream(seed, f"assemble/{ds}"),
            count_tokens=count_tokens,
            stats=stats[ds],
        )
        if not assembled[ds]:
            raise BuildError(f"dataset {ds!r} got no documents from its sources and generators")
    texts = {ds: [content_text(d.content) for d in docs] for ds, docs in assembled.items()}

    if known_path:
        known_vocab = read_vocab(known_path)
    else:
        known_vocab = build_vocab(
            (t for ds in KNOWN_VOCAB_DATASETS for t in texts[ds]), KNOWN_VOCAB_SIZE
        )

    tags = {ds: tag_documents(assembled[ds], texts[ds], known_vocab, seed, ds) for ds in selected}

    if train_tok:
        sample = tokenizer_texts(
            {ds: texts[ds] for ds in selected},
            {ds: tags[ds].heldout for ds in selected},
            seed,
            TOKENIZER_SAMPLE_BYTES,
        )
        # The new tokenizer is written beside its final place and moved there only after every
        # corpus is written with it: a tokenizer at tok_path counts as frozen, so neither a
        # rejected one nor one whose corpora were never all written may be left there.
        trial = tok_path.with_name(tok_path.name + ".tmp")
        tok = train_tokenizer(sample, trial, vocab_size=VOCAB_SIZE)
    else:
        trial, tok = None, frozen

    entries: dict[str, dict] = {}
    try:
        if trial is not None and scale == "full" and tok.vocab_size != VOCAB_SIZE:
            raise RuntimeError(
                f"the tokenizer has {tok.vocab_size} entries, not {VOCAB_SIZE}: too little text"
            )
        for ds in selected:
            ids = [_encode(tok, doc) for doc in assembled[ds]]
            component_tokens: Counter = Counter()
            for doc, doc_ids in zip(assembled[ds], ids):
                component_tokens[doc.component] += len(doc_ids)
            info = {
                "dataset": ds,
                "corpus_version": CORPUS_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
                "scale": scale,
                "seed": seed,
                "docs": len(ids),
                "tokens": sum(component_tokens.values()),
            }
            write_corpus(root / ds, ids, tags[ds], info)
            entries[ds] = _entry(scale, seed, stats[ds], tags[ds], component_tokens, ds)
        if trial is not None:
            trial.replace(tok_path)
    finally:
        if trial is not None and trial.exists():
            trial.unlink()

    root.mkdir(parents=True, exist_ok=True)
    write_false_facts(root / FALSE_FACTS_FILE, plan)
    write_vocab(root / SIMPLE_VOCAB_FILE, simple_vocab)
    write_vocab(root / KNOWN_VOCAB_FILE, known_vocab)
    write_manifest(root, make_manifest(entries, read_manifest(root) if partial else None))
    return BuildSummary(entries, train_tok, time.perf_counter() - started)
