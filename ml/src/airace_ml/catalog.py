"""The game server's facts about the training data (``catalog.json``, plan Task 16).

The server never reads the corpora; it reads this file. It describes each base dataset as a
player sees it: a title and blurb, its size, how much of it each way of preparing it keeps, a
few clean sample passages and a few examples of what each preparation option removes. It also
records the data's identity, so the server only leases jobs to workers holding the same data
(ADR 0008 item 2), and the reference models' personality traits that personality words are
measured against.

The output matches ``packages/protocol/src/catalog.ts`` and is deterministic: the same data
gives byte-identical JSON.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.identity import data_identity
from airace_ml.data.prep import CLEANING_THRESHOLDS, PrepConfig, eligible_docs
from airace_ml.evals.suite import SUITE_VERSION
from airace_ml.paths import TOKENIZER_VERSION, corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

CATALOG_VERSION = "catalog-v1"
FINGERPRINT_FILE = "gate_fingerprint.json"
MAX_SAMPLE_CHARS = 600
SAMPLES = 5
REMOVED_EXAMPLES = 3
CLEANING_LEVELS = ("light", "standard", "thorough")

# Player-facing names (D2 is open: these are the temporary assumption, the same as the
# game's DATASET_TITLES).
TITLES = {
    "web": "Web Crawl",
    "books": "Books",
    "educational": "Educational",
    "conversations": "Conversations",
    "code": "Code",
    "reasoning": "Reasoning",
    "facts": "Reference Facts",
    "creative": "Creative Writing",
}
BLURBS = {
    "web": "Everyday pages from across the internet: news, how-tos and forums. Broad, but messy.",
    "books": "Old public-domain novels and stories. Long, careful writing.",
    "educational": "Simple lessons and explanations written for learners.",
    "conversations": "People talking and answering each other.",
    "code": "Small programs with their explanations.",
    "reasoning": "Puzzles and step-by-step problem solving.",
    "facts": "Short reference entries and quiz sheets about the world.",
    "creative": "Short stories, poems and story requests.",
}


def prep_key(cleaning: str, dedup: bool, fact_check: bool) -> str:
    return f"{cleaning}|{int(dedup)}|{int(fact_check)}"


def cut_text(text: str, limit: int = MAX_SAMPLE_CHARS) -> str:
    """``text`` shortened to at most ``limit`` characters, at a word boundary, marked with …."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    space = head.rfind(" ")
    if space >= limit // 2:
        head = head[:space]
    return head.rstrip() + "…"


def doc_text(tok: Tok, ids: Sequence[int]) -> str:
    """A document as readable text; a conversation shows one turn per line."""
    ids = [int(i) for i in ids]
    if tok.user_id not in ids and tok.ai_id not in ids:
        return tok.decode(ids)
    lines: list[str] = []
    role: str | None = None
    turn: list[int] = []
    for i in ids + [tok.end_id]:
        if i in (tok.user_id, tok.ai_id, tok.end_id):
            if role is not None and turn:
                lines.append(f"{role}: {tok.decode(turn).strip()}")
            turn = []
            role = {tok.user_id: "Person", tok.ai_id: "AI"}.get(
                i, role if i != tok.end_id else None
            )
        else:
            turn.append(i)
    return "\n".join(lines)


def _doc_tokens(corpus: Corpus, idx: np.ndarray) -> int:
    lengths = np.diff(corpus.offsets.astype(np.int64))
    return int(lengths[idx].sum())


def _pick(candidates: np.ndarray, n: int, seed: int) -> list[int]:
    """Up to ``n`` of ``candidates``, the same ones every time (in document order)."""
    if len(candidates) <= n:
        return [int(i) for i in candidates]
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(candidates, size=n, replace=False))


def _texts(tok: Tok, corpus: Corpus, idx: list[int]) -> list[str]:
    return [cut_text(doc_text(tok, corpus.doc(i))) for i in idx]


def describe_dataset(tok: Tok, corpus: Corpus, dataset: str, seed: int) -> dict:
    tags = corpus.tags
    trainable = np.flatnonzero(~tags.heldout)
    pools = {}
    for cleaning in CLEANING_LEVELS:
        for dedup in (False, True):
            for fact_check in (False, True):
                idx = eligible_docs(
                    corpus, PrepConfig(cleaning=cleaning, dedup=dedup, fact_check=fact_check)
                )
                pools[prep_key(cleaning, dedup, fact_check)] = {
                    "docs": len(idx),
                    "tokens": _doc_tokens(corpus, idx),
                }
    clean = eligible_docs(corpus, PrepConfig(cleaning="thorough", dedup=True, fact_check=True))
    cleaning_removes = {
        level: _texts(
            tok,
            corpus,
            _pick(
                np.flatnonzero(~tags.heldout & (tags.quality < np.float16(threshold))),
                REMOVED_EXAMPLES,
                seed + 1 + k,
            ),
        )
        for k, (level, threshold) in enumerate(CLEANING_THRESHOLDS.items())
    }
    duplicates = np.flatnonzero(~tags.heldout & (tags.dup_cluster >= 0) & ~tags.dup_canonical)
    false_facts = np.flatnonzero(~tags.heldout & tags.false_fact)
    return {
        "id": dataset,
        "title": TITLES[dataset],
        "blurb": BLURBS[dataset],
        "tokens": _doc_tokens(corpus, trainable),
        "docs": len(trainable),
        "pools": pools,
        "samples": _texts(tok, corpus, _pick(clean, SAMPLES, seed)),
        "removedExamples": {
            "cleaning": {level: cleaning_removes[level] for level in CLEANING_LEVELS},
            "duplicates": _texts(tok, corpus, _pick(duplicates, REMOVED_EXAMPLES, seed + 10)),
            "factCheck": _texts(tok, corpus, _pick(false_facts, REMOVED_EXAMPLES, seed + 11)),
        },
    }


def reference_traits(population_dirs: Sequence[Path]) -> list[dict[str, float]]:
    """The traits of every feasibility-gate fingerprint found under ``population_dirs``."""
    traits = []
    for folder in population_dirs:
        for path in sorted(Path(folder).rglob(FINGERPRINT_FILE)):
            value = json.loads(path.read_text(encoding="utf-8")).get("value", {})
            if isinstance(value.get("traits"), dict):
                traits.append({k: float(v) for k, v in sorted(value["traits"].items())})
    return traits


def build_catalog(root: Path, population_dirs: Sequence[Path] = ()) -> dict:
    """The catalog of the data under ``root`` (see the module docstring)."""
    tok = Tok.load(tokenizer_path(root))
    datasets = [
        describe_dataset(tok, Corpus.open(corpus_dir(root) / ds), ds, seed=1000 * k)
        for k, ds in enumerate(DATASET_IDS)
    ]
    return {
        "version": CATALOG_VERSION,
        "dataIdentity": data_identity(root),
        "tokenizer": TOKENIZER_VERSION,
        "suite": SUITE_VERSION,
        "datasets": datasets,
        "referenceTraits": reference_traits(population_dirs),
    }


def catalog_json(catalog: dict) -> str:
    return json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"


def catalog_path(root: Path) -> Path:
    return root / "catalog" / "v1" / "catalog.json"


def write_catalog(
    root: Path, population_dirs: Sequence[Path] = (), out: Path | None = None
) -> Path:
    """Builds the catalog and writes it atomically; returns where it went."""
    out = out or catalog_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    tmp.write_text(
        catalog_json(build_catalog(root, population_dirs)), encoding="utf-8", newline="\n"
    )
    tmp.replace(out)
    return out
