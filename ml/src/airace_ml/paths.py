"""Filesystem layout for generated build artifacts (tokenizer, corpus, novelty index, judge)."""

import os
from pathlib import Path

# paths.py -> airace_ml -> src -> ml -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

TOKENIZER_VERSION = "tok-v1"
CORPUS_VERSION = "v1"


def data_root() -> Path:
    """Root of generated data. `AIRACE_DATA` overrides the default `<repo>/data`."""
    override = os.environ.get("AIRACE_DATA")
    return Path(override) if override else REPO_ROOT / "data"


def tokenizer_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "tokenizer" / TOKENIZER_VERSION / "tokenizer.json"


def corpus_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "corpus" / CORPUS_VERSION


def novelty_path(root: Path | None = None) -> Path:
    return (root or data_root()) / "novelty" / "v1" / "index.npy"


def judge_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "judge" / "v1"
