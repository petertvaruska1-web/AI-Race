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
