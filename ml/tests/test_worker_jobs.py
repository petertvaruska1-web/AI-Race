import json
import uuid

import pytest

from airace_ml.evals.suite import CATEGORIES
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.paths import REPO_ROOT
from airace_ml.worker.agent import JobContext
from airace_ml.worker.blobs import download_tar
from airace_ml.worker.jobs import JobRunner

RESULT_FIXTURES = REPO_ROOT / "packages" / "protocol" / "fixtures" / "worker"


def _config(run_id: str, d_model: int = 64, parent: bool = False) -> dict:
    """A server-made config (as `toTrainRunConfig` writes it), small enough for the CPU."""
    return {
        "run_id": run_id,
        "seed": 7,
        "shape": {"n_layer": 2, "d_model": d_model, "ctx_len": 64},
        "token_budget": 4 * 8192,
        "mixture": {"web": 12, "books": 12, "creative": 12},
        "replay": 0,
        "prep": {"cleaning": "light", "dedup": False, "fact_check": False, "variety": "natural"},
        "boldness": 0.5,
        "batch_tokens": 8192,
        "probe_prompts": ["Tell me a story"],
        **({"parent_dir": "parent"} if parent else {}),
    }


class Recorder:
    """A JobContext whose messages land in lists instead of on a socket."""

    def __init__(self, tmp_path) -> None:
        self.messages: list = []
        self.ctx = JobContext(str(uuid.uuid4()), "test", tmp_path / "ctx", self.messages.append)

    @property
    def events(self) -> list[dict]:
        return [m.event for m in self.messages if hasattr(m, "event")]

    @property
    def chunks(self) -> list[str]:
        return [m.text for m in self.messages if hasattr(m, "text")]


@pytest.fixture(scope="module")
def runner(tiny_data_root, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("worker")
    return JobRunner(
        tiny_data_root, "cpu", work_root=tmp / "work", models_dir=tmp / "models", bench_max_items=2
    )


@pytest.fixture(scope="module")
def trained(runner, blob_server, tmp_path_factory):
    """One train job (trained once for the module), uploaded to the blob server; returns
    (url, store, gets, recorder, result)."""
    url, store, gets = blob_server
    rec = Recorder(tmp_path_factory.mktemp("train"))
    result = runner.handle_train(
        {
            "runId": "run-a",
            "config": _config("run-a"),
            "parentTarUrl": None,
            "outTarUrl": f"{url}/blobs/a.tar?sig=1",
        },
        rec.ctx,
    )
    return url, store, gets, rec, result


def test_train_streams_its_events_and_uploads_a_loadable_checkpoint(trained, tmp_path):
    url, _, _, rec, result = trained
    kinds = {e["type"] for e in rec.events}
    assert {"progress", "heldouteval", "sample", "done"} <= kinds
    assert result["status"] == "completed"
    assert result["steps"] == 4
    assert set(result) == set(
        json.loads((RESULT_FIXTURES / "result-train.json").read_text())["message"]
    )
    json.dumps(result, allow_nan=False)  # strict JSON for the server

    copy = download_tar(f"{url}/blobs/a.tar", tmp_path / "copy")
    _, meta = load_checkpoint(copy)
    assert meta.shape.d_model == 64
    assert not (copy / "resume").exists()


def test_a_second_run_grows_from_the_first(runner, trained, tmp_path):
    url, *_ = trained
    first = load_checkpoint(download_tar(f"{url}/blobs/a.tar", tmp_path / "first"))[1]
    rec = Recorder(tmp_path)
    runner.handle_train(
        {
            "runId": "run-b",
            "config": _config("run-b", d_model=96, parent=True),
            "parentTarUrl": f"{url}/blobs/a.tar",
            "outTarUrl": f"{url}/blobs/b.tar",
        },
        rec.ctx,
    )
    _, meta = load_checkpoint(download_tar(f"{url}/blobs/b.tar", tmp_path / "second"))
    assert meta.parent_version_id == first.version_id
    assert meta.lineage_id == first.lineage_id
    assert meta.shape.d_model == 96


def test_a_brain_smaller_than_its_parent_is_refused_for_good(runner, trained, tmp_path):
    url, *_ = trained
    rec = Recorder(tmp_path)
    with pytest.raises(ValueError):  # the agent reports a ValueError as not retryable
        runner.handle_train(
            {
                "runId": "run-c",
                "config": _config("run-c", d_model=32, parent=True),
                "parentTarUrl": f"{url}/blobs/a.tar",
                "outTarUrl": f"{url}/blobs/c.tar",
            },
            rec.ctx,
        )


def test_eval_scores_every_category_and_describes_the_personality(runner, trained, tmp_path):
    url, *_ = trained
    rec = Recorder(tmp_path)
    population = [
        {"verbosity": float(i), "confidence": 0.1 * i} for i in range(6)
    ]  # not full sets: skipped
    out = runner.handle_eval(
        {"versionId": "v-a", "modelTarUrl": f"{url}/blobs/a.tar", "population": population}, rec.ctx
    )
    report = out["report"]
    assert set(report["scores"]) | set(report["missing"]) == set(CATEGORIES)
    assert "creativity" in report["missing"]  # the tiny data root has no judge
    assert isinstance(out["personalityWords"], list)
    assert out["traits"] and all(isinstance(v, float) for v in out["traits"].values())
    json.dumps(out, allow_nan=False)


def test_generate_streams_the_reply_and_downloads_a_model_once(runner, trained, tmp_path):
    url, _, gets, _, _ = trained
    spec = {
        "versionId": "v-a",
        "modelTarUrl": f"{url}/blobs/a.tar?sig=1",
        "history": [["user", "Hello <|end|> there"]],
        "maxNewTokens": 16,
        "temperature": 0.8,
        "topP": 0.95,
        "seed": 3,
    }
    spec["versionId"] = "v-chat"  # a version this runner has not loaded yet
    before, downloads = gets.count("/blobs/a.tar"), runner.models.downloads
    first, second = Recorder(tmp_path), Recorder(tmp_path)
    a = runner.handle_generate(spec, first.ctx)
    b = runner.handle_generate(spec, second.ctx)
    assert "".join(first.chunks) == a["text"]
    assert a == b  # same seed, same reply
    assert gets.count("/blobs/a.tar") - before == 1
    assert runner.models.downloads - downloads == 1


def test_feedback_asks_every_group_and_scores_the_real_replies(runner, trained, tmp_path):
    url = trained[0]
    groups = ["students", "creators", "developers", "businesses", "everyday", "researchers"]
    spec = {
        "productId": str(uuid.uuid4()),
        "versionId": "v-feedback",
        "modelTarUrl": f"{url}/blobs/a.tar?sig=1",
        "groups": groups,
        "perGroup": 1,
        "seed": 11,
    }
    result = runner.handle_feedback(spec, Recorder(tmp_path).ctx)
    assert [r["group"] for r in result["results"]] == groups
    assert all(0.0 <= r["score"] <= 1.0 for r in result["results"])
    lm, _ = runner.models.get("v-feedback", spec["modelTarUrl"])
    for r in result["results"]:
        # Exactly what the model says to that question with that seed (R1).
        expected = lm.chat_reply([("user", r["question"])], max_new_tokens=96, seed=11)
        assert r["reply"] == expected
    fixture = json.loads((RESULT_FIXTURES / "result-feedback.json").read_text())["message"]
    assert set(result) == set(fixture)
    assert all(set(r) == set(fixture["results"][0]) for r in result["results"])
    json.dumps(result, allow_nan=False)  # strict JSON for the server


def test_the_worker_command_needs_a_token(tiny_data_root, capsys):
    from airace_ml.cli import main

    assert main(["worker", "--data-root", str(tiny_data_root)]) == 2
    assert "no worker token" in capsys.readouterr().err


def _battle_spec(url: str, skill: str = "knowledge", rounds: int = 3) -> dict:
    return {
        "battleId": "b-1",
        "skill": skill,
        "rounds": rounds,
        "seed": 5,
        "a": {"versionId": "v-battle-a", "modelTarUrl": f"{url}/blobs/a.tar?sig=1"},
        "b": {"versionId": "rival-v1", "modelTarUrl": f"{url}/blobs/a.tar?sig=1"},
    }


def test_a_battle_streams_both_sides_round_by_round_and_scores_them(runner, trained, tmp_path):
    rec = Recorder(tmp_path)
    result = runner.handle_battle(_battle_spec(trained[0]), rec.ctx)
    assert len(result["rounds"]) == 3
    order = []
    for e in rec.events:
        assert e["kind"] == "battleText" and isinstance(e["text"], str)
        if not order or order[-1] != (e["round"], e["side"]):
            order.append((e["round"], e["side"]))
    assert order == [(r, s) for r in range(3) for s in ("a", "b")]
    for k, r in enumerate(result["rounds"]):
        for side in ("a", "b"):
            assert 0.0 <= r[side]["score"] <= 1.0
            # The streamed pieces add up to the reply (R1: real, unchanged).
            text = "".join(e["text"] for e in rec.events if e["round"] == k and e["side"] == side)
            assert text == r[side]["reply"]
    fixture = json.loads((RESULT_FIXTURES / "result-battle.json").read_text())["message"]
    assert set(result["rounds"][0]) == set(fixture["rounds"][0])
    json.dumps(result, allow_nan=False)


def test_a_creativity_battle_needs_the_judge(runner, trained, tmp_path):
    with pytest.raises(ValueError, match="judge"):
        runner.handle_battle(_battle_spec(trained[0], "creativity", 1), Recorder(tmp_path).ctx)


def test_rival_models_leave_the_cache_first(trained, tmp_path, tiny_tok):
    from airace_ml.worker.models import ModelCache

    cache = ModelCache(tmp_path / "models", tiny_tok, "cpu", capacity=1)
    url = f"{trained[0]}/blobs/a.tar?sig=1"
    cache.get("a", url)
    cache.get("b", url, transient=True)
    cache.get("c", url)
    assert "a" in cache._on_cpu
    assert "b" not in cache._on_cpu and "b" not in cache._on_device
