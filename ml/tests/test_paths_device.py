from airace_ml import paths
from airace_ml.device import pick_device


def test_data_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRACE_DATA", str(tmp_path))
    assert paths.data_root() == tmp_path

def test_data_root_default(monkeypatch):
    monkeypatch.delenv("AIRACE_DATA", raising=False)
    assert paths.data_root() == paths.REPO_ROOT / "data"

def test_artifact_paths(tmp_path):
    assert paths.tokenizer_path(tmp_path) == tmp_path / "tokenizer" / "tok-v1" / "tokenizer.json"
    assert paths.corpus_dir(tmp_path) == tmp_path / "corpus" / "v1"
    assert paths.novelty_path(tmp_path) == tmp_path / "novelty" / "v1" / "index.npy"
    assert paths.judge_dir(tmp_path) == tmp_path / "judge" / "v1"

def test_pick_device_prefer_and_env(monkeypatch):
    assert pick_device("cpu").type == "cpu"
    monkeypatch.setenv("AIRACE_DEVICE", "cpu")
    assert pick_device().type == "cpu"


def test_a_data_root_reads_the_text_it_holds(tmp_path):
    assert paths.CORPUS_VERSIONS == ("v2", "v1")
    assert paths.corpus_version(tmp_path) == "v1" and paths.text_version(tmp_path) == 1
    assert paths.corpus_dir(tmp_path) == tmp_path / "corpus" / "v1"  # an empty root is text 1
    (tmp_path / "corpus" / "v1").mkdir(parents=True)
    (tmp_path / "corpus" / "v1" / "manifest.json").write_text("{}")
    assert paths.corpus_version(tmp_path) == "v1" and paths.text_version(tmp_path) == 1
    (tmp_path / "corpus" / "v2").mkdir()
    assert paths.corpus_version(tmp_path) == "v1"  # a folder without a manifest holds nothing
    (tmp_path / "corpus" / "v2" / "manifest.json").write_text("{}")
    assert paths.corpus_version(tmp_path) == "v2" and paths.text_version(tmp_path) == 2
    assert paths.corpus_dir(tmp_path) == tmp_path / "corpus" / "v2"
    assert paths.corpus_dir(tmp_path, "v1") == tmp_path / "corpus" / "v1"
