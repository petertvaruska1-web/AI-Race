"""Filesystem layout for generated build artifacts (tokenizer, corpus, novelty index, judge)."""

import os
from pathlib import Path

# paths.py -> airace_ml -> src -> ml -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

TOKENIZER_VERSION = "tok-v1"
# The corpus folders a data root may hold, newest first: "v2" is text 2, "v1" is text 1.
CORPUS_VERSIONS = ("v2", "v1")


def data_root() -> Path:
    """Root of generated data. `AIRACE_DATA` overrides the default `<repo>/data`."""
    override = os.environ.get("AIRACE_DATA")
    return Path(override) if override else REPO_ROOT / "data"


def tokenizer_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "tokenizer" / TOKENIZER_VERSION / "tokenizer.json"


def corpus_version(root: Path | None = None) -> str:
    """The text a data root holds: the first of ``CORPUS_VERSIONS`` that has a manifest there,
    "v1" when none has (an empty root is text 1 until a build says otherwise)."""
    base = (root or data_root()) / "corpus"
    for version in CORPUS_VERSIONS:
        if (base / version / "manifest.json").exists():
            return version
    return "v1"


def text_version(root: Path | None = None) -> int:
    """The text a data root holds as a number: 2 for "v2", 1 for "v1"."""
    return int(corpus_version(root)[1:])


def corpus_dir(root: Path | None = None, version: str | None = None) -> Path:
    """The corpus folder of a data root: the text it holds, or ``version`` when one is given."""
    root = root or data_root()
    return root / "corpus" / (version or corpus_version(root))


def novelty_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "novelty" / "v1" / "index.npy"


def judge_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "judge" / "v1"


def rivals_dir(root: Path | None = None) -> Path:
    """Where the rival pack is built: ``rivals/v2``, the pack trained with engine 2.

    ``rivals/v1`` beside it is the pack of worlds made before engine 2 (the game's
    ``balance-v2`` worlds keep their rivals); it is never built again."""
    return (root or data_root()) / "rivals" / "v2"
