"""A data root holds text 1 or text 2; a build says which, and a root never mixes them."""

from pathlib import Path

import pytest

from airace_content.build import BuildError, build_corpus
from airace_content.cli import main
from airace_content.manifest import read_manifest
from airace_content.sources import fixture_fetch
from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.paths import corpus_dir, corpus_version, text_version

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
FETCH = fixture_fetch(FIX)


@pytest.fixture(scope="module")
def text2_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("text2")
    build_corpus("tiny", root, FETCH, seed=0, text=2)
    return root


def test_a_text_2_build_lands_in_corpus_v2_and_reads_back_as_text_2(text2_root):
    assert (text2_root / "corpus" / "v2" / "manifest.json").exists()
    assert not (text2_root / "corpus" / "v1").exists()
    assert corpus_version(text2_root) == "v2" and text_version(text2_root) == 2
    assert read_manifest(corpus_dir(text2_root))["corpus_version"] == "v2"
    for ds in DATASET_IDS:
        assert Corpus.open(corpus_dir(text2_root) / ds).n_docs > 0


def test_a_build_without_a_text_is_text_1_in_an_empty_root_and_the_held_text_after(
    tmp_path, text2_root
):
    build_corpus("tiny", tmp_path, FETCH, seed=0)
    assert corpus_version(tmp_path) == "v1"
    assert read_manifest(corpus_dir(tmp_path))["corpus_version"] == "v1"
    # a rebuild without --text goes on with the text the root holds
    build_corpus("tiny", text2_root, FETCH, seed=0, datasets=["web"])
    assert corpus_version(text2_root) == "v2" and not (text2_root / "corpus" / "v1").exists()


def test_a_root_that_holds_one_text_refuses_a_build_of_the_other(tmp_path, text2_root):
    build_corpus("tiny", tmp_path, FETCH, seed=0)
    before = sorted(p.name for p in (tmp_path / "corpus").iterdir())
    with pytest.raises(BuildError, match="text 1"):
        build_corpus("tiny", tmp_path, FETCH, seed=0, text=2)
    with pytest.raises(BuildError, match="text 2"):
        build_corpus("tiny", text2_root, FETCH, seed=0, text=1)
    assert sorted(p.name for p in (tmp_path / "corpus").iterdir()) == before


def test_the_cli_builds_text_2_and_refuses_the_other_text_with_exit_2(tmp_path, capsys):
    root = tmp_path / "data2"
    full = ["build", "--scale", "tiny", "--fixtures", str(FIX), "--out", str(root)]
    assert main([*full, "--text", "2"]) == 0
    assert "corpus" in capsys.readouterr().out and corpus_version(root) == "v2"
    assert main([*full, "--text", "1"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "text 2" in err and "--out" in err
    assert main([*full, "--text", "3"]) == 2
    assert corpus_version(root) == "v2"
