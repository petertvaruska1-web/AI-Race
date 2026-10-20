"""F3: a build never reuses (and freezes) what a build of the other scale left in the same data
root, and never overwrites it."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import airace_ml.cli as ml_cli
from airace_content.assemble import SCALES
from airace_content.build import BuildError, build_corpus
from airace_content.cli import main
from airace_content.manifest import (
    SHARED_RECORD_FILE,
    read_record,
    tokenizer_record,
    tokenizer_record_path,
    write_record,
)
from airace_content.sources import fixture_fetch
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
FETCH = fixture_fetch(FIX)


@pytest.fixture
def small_full_targets(monkeypatch):
    """Full-scale targets small enough to build from the fixtures in seconds."""
    for ds in DATASET_IDS:
        monkeypatch.setitem(SCALES["full"]["target_tokens"], ds, 30_000)


@pytest.fixture(scope="module")
def built_tiny(tmp_path_factory) -> Path:
    """One tiny build (seed 3) that the tests copy."""
    root = tmp_path_factory.mktemp("tiny")
    build_corpus("tiny", root, FETCH, seed=3)
    return root


@pytest.fixture(scope="module")
def built_full(tmp_path_factory) -> Path:
    """One full build (seed 5, small targets) that the tests copy."""
    root = tmp_path_factory.mktemp("full")
    with pytest.MonkeyPatch.context() as patch:
        for ds in DATASET_IDS:
            patch.setitem(SCALES["full"]["target_tokens"], ds, 30_000)
        build_corpus("full", root, FETCH, seed=5)
    return root


@pytest.fixture
def tiny_root(built_tiny, tmp_path) -> Path:
    shutil.copytree(built_tiny, tmp_path / "data")
    return tmp_path / "data"


@pytest.fixture
def full_root(built_full, tmp_path) -> Path:
    shutil.copytree(built_full, tmp_path / "data")
    return tmp_path / "data"


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under the data root, by relative path."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_a_full_build_does_not_reuse_a_tiny_tokenizer(tiny_root, small_full_targets):
    before = _snapshot(tiny_root)
    with pytest.raises(BuildError, match="--retrain-tokenizer") as refused:
        build_corpus("full", tiny_root, FETCH, seed=0)
    assert "--out" in str(refused.value)
    assert _snapshot(tiny_root) == before  # refused before anything was read or written
    # asked to train a new one, the full build goes ahead and its tokenizer is the one to reuse
    summary = build_corpus("full", tiny_root, FETCH, seed=0, retrain_tokenizer=True)
    assert summary.tokenizer_trained
    assert Tok.load(tokenizer_path(tiny_root)).vocab_size == 4096
    assert not build_corpus("full", tiny_root, FETCH, seed=0, datasets=["code"]).tokenizer_trained


def test_a_partial_full_build_does_not_reuse_a_tiny_build_of_the_shared_files(
    full_root, small_full_targets
):
    """A full tokenizer in a root whose vocabularies and false facts are a tiny build's."""
    write_record(corpus_dir(full_root) / SHARED_RECORD_FILE, {"scale": "tiny", "seed": 0})
    before = _snapshot(full_root)
    with pytest.raises(BuildError, match="--out") as refused:
        build_corpus("full", full_root, FETCH, seed=0, datasets=["code"])
    assert "false_facts.json" in str(refused.value) and "tiny" in str(refused.value)
    assert _snapshot(full_root) == before


def test_a_tiny_build_does_not_overwrite_a_full_build(full_root):
    before = _snapshot(full_root)
    for kwargs in ({}, {"datasets": ["web"]}, {"retrain_tokenizer": True}):
        with pytest.raises(BuildError, match="own --out"):
            build_corpus("tiny", full_root, FETCH, seed=0, **kwargs)
    assert _snapshot(full_root) == before
    # the manifest alone says the corpora are full-scale, whatever else is missing
    (corpus_dir(full_root) / SHARED_RECORD_FILE).unlink()
    tokenizer_record_path(tokenizer_path(full_root)).unlink()
    with pytest.raises(BuildError, match="own --out"):
        build_corpus("tiny", full_root, FETCH, seed=0)


def test_the_cli_refuses_a_tiny_build_into_a_full_root(full_root, capsys):
    before = _snapshot(full_root)
    argv = ["build", "--scale", "tiny", "--fixtures", str(FIX), "--out", str(full_root)]
    assert main(argv) == 1
    assert capsys.readouterr().err.splitlines()[-1].startswith("error: ")
    assert _snapshot(full_root) == before


def test_a_reused_tokenizer_of_the_wrong_size_is_refused_at_full(
    full_root, small_full_targets, tiny_tok_path
):
    """The record says full, but the tokenizer has 512 entries, not 4096."""
    shutil.copyfile(tiny_tok_path, tokenizer_path(full_root))
    record = tokenizer_record(
        tokenizer_path(full_root), scale="full", seed=0, sample_bytes=1, vocab_size=512
    )
    write_record(tokenizer_record_path(tokenizer_path(full_root)), record)
    with pytest.raises(BuildError, match="4096"):
        build_corpus("full", full_root, FETCH, seed=0)


def test_the_records_say_how_the_files_were_made(built_tiny, built_full):
    for root, scale, seed in ((built_tiny, "tiny", 3), (built_full, "full", 5)):
        record = read_record(tokenizer_record_path(tokenizer_path(root)))
        assert record["scale"] == scale and record["seed"] == seed
        assert record["tokenizer_version"] == "tok-v1" and record["corpus_version"] == "v1"
        assert record["sample_bytes"] > 0 and record["vocab_size"] > 0
        assert record["sha256"] == hashlib.sha256(tokenizer_path(root).read_bytes()).hexdigest()
        shared = json.loads((corpus_dir(root) / SHARED_RECORD_FILE).read_text(encoding="utf-8"))
        assert shared["scale"] == scale and shared["seed"] == seed
    assert read_record(tokenizer_record_path(tokenizer_path(built_full)))["vocab_size"] == 4096


def test_a_record_of_another_tokenizer_does_not_count(full_root, small_full_targets, tiny_tok_path):
    """A tokenizer put in place after the record was written (hand-copied, or left by a build
    that stopped half way) is a tokenizer nothing records."""
    shutil.copyfile(tiny_tok_path, tokenizer_path(full_root))
    with pytest.raises(BuildError, match="--retrain-tokenizer"):
        build_corpus("full", full_root, FETCH, seed=0)


def test_a_tokenizer_with_no_record_is_not_reused_at_full(full_root, small_full_targets):
    tokenizer_record_path(tokenizer_path(full_root)).unlink()  # a root built before records
    with pytest.raises(BuildError, match="no record"):
        build_corpus("full", full_root, FETCH, seed=0)


def test_a_partial_full_build_with_no_shared_record_is_refused(full_root, small_full_targets):
    shared = corpus_dir(full_root) / SHARED_RECORD_FILE
    shared.unlink()
    with pytest.raises(BuildError, match="false_facts.json"):
        build_corpus("full", full_root, FETCH, seed=0, datasets=["code"])
    # but with every dataset it makes the shared files again, so there is nothing to refuse
    assert not build_corpus("full", full_root, FETCH, seed=0).tokenizer_trained
    assert shared.exists()
    assert build_corpus("full", full_root, FETCH, seed=0, datasets=["code"])


def test_a_partial_tiny_rebuild_keeps_the_record_of_the_shared_files(tiny_root):
    shared = corpus_dir(tiny_root) / SHARED_RECORD_FILE
    before = shared.read_bytes()
    build_corpus("tiny", tiny_root, FETCH, seed=9, datasets=["web"])  # reuses files of seed 3
    assert shared.read_bytes() == before


def test_the_missing_data_hint_does_not_point_at_a_tiny_build():
    assert "tiny" not in ml_cli.BUILD_DATA_HINT
    assert "--scale full" in ml_cli.BUILD_DATA_HINT
