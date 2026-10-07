"""The corpus manifest and the shared build artifacts next to it.

``manifest.json`` records provenance: every source that feeds a built dataset, with its license
and homepage, and per dataset what went into it (components, noise, counts). The shared artifacts
(``false_facts.json``, ``simple_vocab.txt``, ``known_vocab.txt``) are what every dataset of one
corpus is built against, so a partial rebuild reads them back instead of recomputing them.
"""

import json
from dataclasses import asdict
from pathlib import Path

from airace_content.sources import SOURCES
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import CORPUS_VERSION, TOKENIZER_VERSION
from airace_ml.skills.facts import FalseFactPlan

MANIFEST_FILE = "manifest.json"
FALSE_FACTS_FILE = "false_facts.json"
SIMPLE_VOCAB_FILE = "simple_vocab.txt"
KNOWN_VOCAB_FILE = "known_vocab.txt"


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


def make_manifest(datasets: dict[str, dict], previous: dict | None = None) -> dict:
    """The manifest for ``datasets`` (one entry per built dataset), merged into ``previous``:
    entries of datasets that were not rebuilt are kept. ``sources`` lists every source that feeds
    one of the datasets, with the datasets it feeds."""
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
    sources = [
        {**asdict(spec), "feeds": feeds[spec.id]} for spec in SOURCES.values() if spec.id in feeds
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
