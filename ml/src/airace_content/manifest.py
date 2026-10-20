"""The corpus manifest and the shared build artifacts next to it.

``manifest.json`` records provenance: every source that feeds a built dataset, with its license,
homepage and the revision (commit) it was read at, and per dataset what went into it (components,
noise, counts). The shared artifacts
(``false_facts.json``, ``simple_vocab.txt``, ``known_vocab.txt``) are what every dataset of one
corpus is built against, so a partial rebuild reads them back instead of recomputing them.

Two small records say what scale of build made the files a later build may want to reuse: the one
beside the tokenizer (``build.json``: scale, seed, sample bytes and the tokenizer file's digest, so
a record of another tokenizer does not count) and the one beside the shared artifacts
(``shared.json``: scale and seed).
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from airace_content.sources import SOURCES, SourceSpec
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import CORPUS_VERSION, TOKENIZER_VERSION
from airace_ml.skills.facts import FalseFactPlan

MANIFEST_FILE = "manifest.json"
FALSE_FACTS_FILE = "false_facts.json"
SIMPLE_VOCAB_FILE = "simple_vocab.txt"
KNOWN_VOCAB_FILE = "known_vocab.txt"
SHARED_RECORD_FILE = "shared.json"  # beside the shared artifacts
TOKENIZER_RECORD_FILE = "build.json"  # beside tokenizer.json


def write_vocab(path: Path, vocab: set[str]) -> None:
    """One word per line, sorted."""
    Path(path).write_text("".join(f"{word}\n" for word in sorted(vocab)), encoding="utf-8")


def read_vocab(path: Path) -> set[str]:
    return {line for line in Path(path).read_text(encoding="utf-8").split("\n") if line}


def write_false_facts(path: Path, plan: FalseFactPlan) -> None:
    Path(path).write_text(plan.to_json(), encoding="utf-8")


def read_false_facts(path: Path) -> FalseFactPlan:
    return FalseFactPlan.from_json(Path(path).read_text(encoding="utf-8"))


def read_manifest(corpus_root: Path) -> dict | None:
    path = Path(corpus_root) / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_record(path: Path, record: dict) -> None:
    """``record`` as JSON at ``path``, written beside it and moved into place."""
    path = Path(path)
    trial = path.with_name(path.name + ".tmp")
    trial.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    trial.replace(path)


def read_record(path: Path) -> dict | None:
    """The JSON object at ``path``; None when there is no such file or it is not an object."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def tokenizer_record_path(tok_path: Path) -> Path:
    return Path(tok_path).with_name(TOKENIZER_RECORD_FILE)


def tokenizer_record(
    tok_file: Path, *, scale: str, seed: int, sample_bytes: int, vocab_size: int
) -> dict:
    """What a build records about the tokenizer it made: the scale and seed of the build, the
    bytes of text it was trained on, its size, and the digest of ``tok_file``."""
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "corpus_version": CORPUS_VERSION,
        "scale": scale,
        "seed": seed,
        "sample_bytes": sample_bytes,
        "vocab_size": vocab_size,
        "sha256": hashlib.sha256(Path(tok_file).read_bytes()).hexdigest(),
    }


def read_tokenizer_record(tok_path: Path) -> dict | None:
    """The record of the tokenizer at ``tok_path``; None when it has none, it cannot be read or it
    describes another tokenizer file (one put there since, or left by a build that stopped)."""
    record = read_record(tokenizer_record_path(tok_path))
    if record is None:
        return None
    try:
        digest = hashlib.sha256(Path(tok_path).read_bytes()).hexdigest()
    except OSError:
        return None
    return record if record.get("sha256") == digest else None


def shared_record(scale: str, seed: int) -> dict:
    return {"corpus_version": CORPUS_VERSION, "scale": scale, "seed": seed}


def read_shared_record(corpus_root: Path) -> dict | None:
    return read_record(Path(corpus_root) / SHARED_RECORD_FILE)


def make_manifest(
    datasets: dict[str, dict],
    previous: dict | None = None,
    revisions: Mapping[str, str] | None = None,
) -> dict:
    """The manifest for ``datasets`` (one entry per built dataset), merged into ``previous``:
    entries of datasets that were not rebuilt are kept. ``sources`` lists every source that feeds
    one of the datasets, with the datasets it feeds and its ``revision``: the one this build read
    it at (``revisions[id]``, else the one its spec declares, else None), and for a source that
    feeds only datasets that were not rebuilt, the revision ``previous`` recorded.

    A source has one revision, that of the latest build that read it. When a partial rebuild reads
    a source that also feeds datasets it did not rebuild, those datasets were built at an earlier
    revision, and the entry no longer says which."""
    merged = {
        ds: entry
        for ds, entry in (previous or {}).get("datasets", {}).items()
        if ds in DATASET_IDS and ds not in datasets
    }
    merged.update(datasets)
    ordered = {ds: merged[ds] for ds in DATASET_IDS if ds in merged}
    feeds: dict[str, list[str]] = {}
    for ds, entry in ordered.items():
        for comp in entry.get("components", []):
            if comp["kind"] == "source" and ds not in feeds.setdefault(comp["name"], []):
                feeds[comp["name"]].append(ds)
    revisions = revisions or {}
    kept = {src["id"]: src.get("revision") for src in (previous or {}).get("sources", [])}

    def revision_of(spec: SourceSpec) -> str | None:
        if any(ds in datasets for ds in feeds[spec.id]):  # read by this build
            return revisions.get(spec.id) or spec.revision
        return kept.get(spec.id) or spec.revision

    sources = [
        {**asdict(spec), "revision": revision_of(spec), "feeds": feeds[spec.id]}
        for spec in SOURCES.values()
        if spec.id in feeds
    ]
    return {
        "corpus_version": CORPUS_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "sources": sources,
        "datasets": ordered,
    }


def write_manifest(corpus_root: Path, manifest: dict) -> None:
    path = Path(corpus_root) / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
