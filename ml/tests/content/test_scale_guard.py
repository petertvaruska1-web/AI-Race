"""F3: a build never reuses (and freezes) what a build of the other scale left in the same data
root, and never overwrites it."""

import json
import shutil
from pathlib import Path

import pytest

import airace_ml.cli as ml_cli
from airace_content.assemble import SCALES
from airace_content.build import BuildError, build_corpus
from airace_content.cli import main
from airace_content.manifest import tokenizer_record, write_record
from airace_content.sources import fixture_fetch
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
FETCH = fixture_fetch(FIX)
SHARED = ("manifest.json", "false_facts.json", "simple_vocab.txt", "known_vocab.txt")


@pytest.fixture
def small_full_targets(monkeypatch):
    """Full-scale targets small enough to build from the fixtures in seconds."""
    for ds in DATASET_IDS:
        monkeypatch.setitem(SCALES["full"]["target_tokens"], ds, 30_000)


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under the data root, by relative path."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_a_full_build_does_not_reuse_a_tiny_tokenizer(tmp_path, small_full_targets):
    root = tmp_path / "data"
    build_corpus("tiny", root, FETCH, seed=0)
    before = _snapshot(root)
    with pytest.raises(BuildError, match="--retrain-tokenizer") as refused:
        build_corpus("full", root, FETCH, seed=0)
    assert "--out" in str(refused.value)
    assert _snapshot(root) == before  # refused before anything was read or written
    # asked to train a new one, the full build goes ahead and its tokenizer is the one to reuse
    summary = build_corpus("full", root, FETCH, seed=0, retrain_tokenizer=True)
    assert summary.tokenizer_trained
    assert Tok.load(tokenizer_path(root)).vocab_size == 4096
    assert not build_corpus("full", root, FETCH, seed=0).tokenizer_trained


def test_a_partial_full_build_does_not_reuse_a_tiny_build_of_the_shared_files(
    tmp_path, small_full_targets
):
    root = tmp_path / "data"
    build_corpus("tiny", root, FETCH, seed=0)
    full = tmp_path / "full"
    build_corpus("full", full, FETCH, seed=0)
    # a full tokenizer in a root whose vocabularies and false facts are a tiny build's
    shutil.copyfile(tokenizer_path(full), tokenizer_path(root))
    for path in (tokenizer_path(full),):
        shutil.copyfile(path.with_name("build.json"), tokenizer_path(root).with_name("build.json"))
    before = _snapshot(root)
    with pytest.raises(BuildError, match="--out"):
        build_corpus("full", root, FETCH, seed=0, datasets=["code"])
    assert _snapshot(root) == before


def test_a_tiny_build_does_not_overwrite_a_full_build(tmp_path, small_full_targets):
    root = tmp_path / "data"
    build_corpus("full", root, FETCH, seed=0)
    before = _snapshot(root)
    for kwargs in ({}, {"datasets": ["web"]}, {"retrain_tokenizer": True}):
        with pytest.raises(BuildError, match="own --out"):
            build_corpus("tiny", root, FETCH, seed=0, **kwargs)
    assert _snapshot(root) == before
    # the manifest alone says the corpora are full-scale, whatever else is missing
    for name in ("shared.json",):
        (corpus_dir(root) / name).unlink()
    tokenizer_path(root).with_name("build.json").unlink()
    with pytest.raises(BuildError, match="own --out"):
        build_corpus("tiny", root, FETCH, seed=0)


def test_the_cli_refuses_a_tiny_build_into_a_full_root(tmp_path, small_full_targets, capsys):
    root = tmp_path / "data"
    build_corpus("full", root, FETCH, seed=0)
    before = _snapshot(root)
    argv = ["build", "--scale", "tiny", "--fixtures", str(FIX), "--out", str(root)]
    assert main(argv) == 1
    assert capsys.readouterr().err.splitlines()[-1].startswith("error: ")
    assert _snapshot(root) == before


def test_a_reused_tokenizer_of_the_wrong_size_is_refused_at_full(
    tmp_path, small_full_targets, tiny_tok_path
):
    """The record says full, but the tokenizer has 512 entries, not 4096."""
    root = tmp_path / "data"
    build_corpus("full", root, FETCH, seed=0)
    shutil.copyfile(tiny_tok_path, tokenizer_path(root))
    record_path = tokenizer_path(root).with_name("build.json")
    record = tokenizer_record(
        tokenizer_path(root), scale="full", seed=0, sample_bytes=1, vocab_size=512
    )
    write_record(record_path, record)
    with pytest.raises(BuildError, match="4096"):
        build_corpus("full", root, FETCH, seed=0)


def test_the_tokenizer_record_says_how_the_tokenizer_was_made(tmp_path, small_full_targets):
    import hashlib

    tiny, full = tmp_path / "tiny", tmp_path / "full"
    build_corpus("tiny", tiny, FETCH, seed=3)
    build_corpus("full", full, FETCH, seed=5)
    for root, scale, seed in ((tiny, "tiny", 3), (full, "full", 5)):
        record = json.loads(
            tokenizer_path(root).with_name("build.json").read_text(encoding="utf-8")
        )
        assert record["scale"] == scale and record["seed"] == seed
        assert record["tokenizer_version"] == "tok-v1" and record["corpus_version"] == "v1"
        assert record["sample_bytes"] > 0
        assert record["sha256"] == hashlib.sha256(tokenizer_path(root).read_bytes()).hexdigest()
        shared = json.loads((corpus_dir(root) / "shared.json").read_text(encoding="utf-8"))
        assert shared["scale"] == scale and shared["seed"] == seed


def test_a_record_of_another_tokenizer_does_not_count(tmp_path, small_full_targets, tiny_tok_path):
    """A tokenizer put in place after the record was written (hand-copied, or left by a build
    that stopped half way) is a tokenizer nothing records."""
    root = tmp_path / "data"
    build_corpus("full", root, FETCH, seed=0)
    shutil.copyfile(tiny_tok_path, tokenizer_path(root))
    with pytest.raises(BuildError, match="--retrain-tokenizer"):
        build_corpus("full", root, FETCH, seed=0)


def test_a_tokenizer_with_no_record_is_not_reused_at_full(tmp_path, small_full_targets):
    root = tmp_path / "data"
    build_corpus("full", root, FETCH, seed=0)
    tokenizer_path(root).with_name("build.json").unlink()  # a root built before records existed
    with pytest.raises(BuildError, match="no record"):
        build_corpus("full", root, FETCH, seed=0)


def test_a_partial_full_build_with_no_shared_record_is_refused(tmp_path, small_full_targets):
    root = tmp_path / "data"
    build_corpus("full", root, FETCH, seed=0)
    (corpus_dir(root) / "shared.json").unlink()
    with pytest.raises(BuildError, match="false_facts.json"):
        build_corpus("full", root, FETCH, seed=0, datasets=["code"])
    # but with every dataset it rebuilds the shared files, so there is nothing to refuse
    assert not build_corpus("full", root, FETCH, seed=0).tokenizer_trained
    assert build_corpus("full", root, FETCH, seed=0, datasets=["code"])


def test_a_partial_tiny_rebuild_keeps_the_record_of_the_shared_files(tmp_path):
    root = tmp_path / "data"
    build_corpus("tiny", root, FETCH, seed=3)
    shared = corpus_dir(root) / "shared.json"
    before = shared.read_bytes()
    build_corpus("tiny", root, FETCH, seed=9, datasets=["web"])  # reuses files made with seed 3
    assert shared.read_bytes() == before


def test_the_missing_data_hint_does_not_point_at_a_tiny_build():
    assert "tiny" not in ml_cli.BUILD_DATA_HINT
    assert "--scale full" in ml_cli.BUILD_DATA_HINT
