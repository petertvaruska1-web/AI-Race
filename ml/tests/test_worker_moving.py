"""Runs that move between cards (Stage 2, Fair mode): the helper saves the trainer's progress to
the host, and another helper continues from it to exactly the same AI."""

import json

import pytest
import torch

from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.trainer import TrainHooks, can_resume, has_resume_state, train_run
from airace_ml.worker.blobs import download_tar
from airace_ml.worker.jobs import JobRunner
from airace_ml.worker.moving import ResumeUploader
from tests.test_worker_jobs import Recorder, _config


def _same_weights(a, b) -> None:
    wa = load_checkpoint(a)[0].state_dict()
    wb = load_checkpoint(b)[0].state_dict()
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


def test_a_run_continued_on_another_card_ends_where_an_unbroken_run_does(
    tiny_data_root, blob_server, tmp_path
):
    url, store, _ = blob_server
    cfg = TrainRunConfig.from_json(json.dumps(_config("run-m")))

    # The reference: one card, start to finish.
    train_run(cfg, out_dir=tmp_path / "whole", data_root=tiny_data_root, device="cpu")

    # The first card saves its progress to the host, then goes away after step 2.
    uploader = ResumeUploader(f"{url}/blobs/resume/run-m.tar?sig=1", every_s=0.0)
    train_run(
        cfg,
        out_dir=tmp_path / "first" / "out",
        data_root=tiny_data_root,
        device="cpu",
        checkpoint_every=2,
        on_resume_saved=uploader.saved,
        _hooks=TrainHooks(stop_after_steps=2),
    )
    uploader.close()
    assert "/blobs/resume/run-m.tar" in store
    assert uploader.uploads == 1

    # The next card, with nothing of its own, continues from what the host kept.
    runner = JobRunner(
        tiny_data_root, "cpu", work_root=tmp_path / "second", models_dir=tmp_path / "models"
    )
    second = Recorder(tmp_path)
    result = runner.handle_train(
        {
            "runId": "run-m",
            "config": _config("run-m"),
            "parentTarUrl": None,
            "outTarUrl": f"{url}/blobs/m.tar?sig=1",
            "resumeUrl": None,
            "resumeFromUrl": f"{url}/blobs/resume/run-m.tar",
        },
        second.ctx,
    )
    assert result["status"] == "completed"
    # It ran only the steps the first card hadn't: it continued, it didn't start over.
    steps = [e["step"] for e in second.events if e["type"] == "progress"]
    assert steps and min(steps) > 2  # a run that started over reports step 1
    _same_weights(tmp_path / "whole", download_tar(f"{url}/blobs/m.tar", tmp_path / "moved"))


def test_a_fair_mode_run_saves_its_progress_while_it_trains(tiny_data_root, blob_server, tmp_path):
    url, store, _ = blob_server
    runner = JobRunner(
        tiny_data_root, "cpu", work_root=tmp_path / "work", models_dir=tmp_path / "models"
    )
    runner.checkpoint_seconds = 0.0  # a save after every step
    runner.resume_upload_every_s = 0.0
    runner.handle_train(
        {
            "runId": "run-s",
            "config": _config("run-s"),
            "parentTarUrl": None,
            "outTarUrl": f"{url}/blobs/s.tar?sig=1",
            "resumeUrl": f"{url}/blobs/resume/run-s.tar?sig=1",
            "resumeFromUrl": None,
        },
        Recorder(tmp_path).ctx,
    )
    assert "/blobs/resume/run-s.tar" in store
    saved = download_tar(f"{url}/blobs/resume/run-s.tar", tmp_path / "saved")
    assert (saved / "resume" / "state.json").is_file()
    assert (saved / "telemetry.jsonl").is_file()


def test_without_fair_mode_nothing_is_uploaded(tiny_data_root, blob_server, tmp_path):
    url, store, _ = blob_server
    runner = JobRunner(
        tiny_data_root, "cpu", work_root=tmp_path / "work", models_dir=tmp_path / "models"
    )
    runner.checkpoint_seconds = 0.0
    runner.resume_upload_every_s = 0.0
    runner.handle_train(
        {
            "runId": "run-n",
            "config": _config("run-n"),
            "parentTarUrl": None,
            "outTarUrl": f"{url}/blobs/n.tar?sig=1",
        },
        Recorder(tmp_path).ctx,
    )
    assert "/blobs/resume/run-n.tar" not in store


def test_uploads_wait_the_interval(tmp_path):
    clock = [0.0]
    uploader = ResumeUploader("http://unused", every_s=120.0, clock=lambda: clock[0])
    assert not uploader.due()
    clock[0] = 119.0
    assert not uploader.due()
    clock[0] = 120.0
    assert uploader.due()


# -- nothing lost, nothing done twice ---------------------------------------------------------------


def _train_spec(url: str, run_id: str, **more) -> dict:
    return {
        "runId": run_id,
        "config": _config(run_id),
        "parentTarUrl": None,
        "outTarUrl": f"{url}/blobs/{run_id}.tar?sig=1",
        **more,
    }


def test_progress_that_cannot_be_continued_is_dropped_and_the_run_starts_over(
    tiny_data_root, blob_server, tmp_path
):
    url, _, _ = blob_server
    cfg = TrainRunConfig.from_json(json.dumps(_config("run-d")))
    train_run(cfg, out_dir=tmp_path / "whole", data_root=tiny_data_root, device="cpu")
    # Left on this card under the run's name by other settings (or, in the same way, by an older
    # training engine): a complete state, but not this run's.
    other = TrainRunConfig.from_json(json.dumps({**_config("run-d"), "seed": 8}))
    stale = tmp_path / "work" / "runs" / "run-d" / "out"
    train_run(
        other,
        out_dir=stale,
        data_root=tiny_data_root,
        device="cpu",
        _hooks=TrainHooks(stop_after_steps=2),
    )
    assert has_resume_state(stale) and not can_resume(stale, cfg)
    runner = JobRunner(
        tiny_data_root, "cpu", work_root=tmp_path / "work", models_dir=tmp_path / "models"
    )
    rec = Recorder(tmp_path)
    result = runner.handle_train(_train_spec(url, "run-d"), rec.ctx)
    assert result["status"] == "completed"
    assert next(e["step"] for e in rec.events if e["type"] == "progress") == 1  # from the start
    _same_weights(tmp_path / "whole", download_tar(f"{url}/blobs/run-d.tar", tmp_path / "got"))


def test_progress_from_the_host_that_cannot_be_continued_is_dropped_too(
    tiny_data_root, blob_server, tmp_path
):
    url, store, _ = blob_server
    cfg = TrainRunConfig.from_json(json.dumps(_config("run-e")))
    train_run(cfg, out_dir=tmp_path / "whole", data_root=tiny_data_root, device="cpu")
    other = TrainRunConfig.from_json(json.dumps({**_config("run-e"), "seed": 8}))
    uploader = ResumeUploader(f"{url}/blobs/resume/run-e.tar?sig=1", every_s=0.0)
    train_run(
        other,
        out_dir=tmp_path / "elsewhere",
        data_root=tiny_data_root,
        device="cpu",
        checkpoint_every=2,
        on_resume_saved=uploader.saved,
        _hooks=TrainHooks(stop_after_steps=2),
    )
    uploader.close()
    assert "/blobs/resume/run-e.tar" in store
    runner = JobRunner(
        tiny_data_root, "cpu", work_root=tmp_path / "work", models_dir=tmp_path / "models"
    )
    rec = Recorder(tmp_path)
    result = runner.handle_train(
        _train_spec(url, "run-e", resumeFromUrl=f"{url}/blobs/resume/run-e.tar"), rec.ctx
    )
    assert result["status"] == "completed"
    assert next(e["step"] for e in rec.events if e["type"] == "progress") == 1
    _same_weights(tmp_path / "whole", download_tar(f"{url}/blobs/run-e.tar", tmp_path / "got"))


def test_a_finished_run_whose_upload_failed_is_uploaded_again_not_trained_again(
    tiny_data_root, blob_server, tmp_path, monkeypatch
):
    from airace_ml.worker import jobs as jobs_module

    url, store, _ = blob_server
    runner = JobRunner(
        tiny_data_root, "cpu", work_root=tmp_path / "work", models_dir=tmp_path / "models"
    )
    trained = []
    real_train, real_upload = jobs_module.train_run, jobs_module.upload_dir_tar

    def train(*args, **kwargs):
        trained.append(1)
        return real_train(*args, **kwargs)

    failing = [True]

    def upload(src, to, exclude=()):
        if failing[0]:
            failing[0] = False
            raise ConnectionError("the game went away")
        return real_upload(src, to, exclude=exclude)

    monkeypatch.setattr(jobs_module, "train_run", train)
    monkeypatch.setattr(jobs_module, "upload_dir_tar", upload)
    spec = _train_spec(url, "run-f")
    first = Recorder(tmp_path)
    with pytest.raises(ConnectionError):
        runner.handle_train(spec, first.ctx)
    assert "/blobs/run-f.tar" not in store
    second = Recorder(tmp_path)
    result = runner.handle_train(spec, second.ctx)
    assert trained == [1]  # the AI was already made: only the upload was missing
    assert second.events == []
    done = next(e for e in first.events if e["type"] == "done")
    assert result == {
        "steps": 4,
        "final_loss": done["summary"]["final_loss"],
        "heldout_losses": done["summary"]["heldout_losses"],
        "status": "completed",
    }
    load_checkpoint(download_tar(f"{url}/blobs/run-f.tar", tmp_path / "got"))
    assert not (tmp_path / "work" / "runs" / "run-f").exists()
