import contextlib
import io
import json
import math
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import airace_ml.evals.judge as judge_module
from airace_ml import cli
from airace_ml.cli import main
from airace_ml.evals.judge import JudgeBuildError
from airace_ml.infer.lm import TorchLM
from airace_ml.model.shape import ModelShape
from airace_ml.paths import judge_dir, novelty_path, tokenizer_path
from airace_ml.personality.fingerprint import TRAITS
from airace_ml.tokenizer import train_tokenizer
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import Done, Progress
from airace_ml.train.trainer import TrainHooks, train_run


def _cfg(tmp_path):
    c = TrainRunConfig(
        run_id="cli1",
        seed=1,
        shape=ModelShape(2, 64, 64),
        token_budget=1024 * 8,
        mixture={"creative": 1.0},
        batch_tokens=1024,
    )
    p = tmp_path / "c.json"
    p.write_text(c.to_json(), encoding="utf-8")
    return p


def test_train_bench_chat(tiny_data_root, tmp_path, monkeypatch, capsys):
    out = tmp_path / "m"
    assert (
        main(
            [
                "train",
                "--config",
                str(_cfg(tmp_path)),
                "--out",
                str(out),
                "--data-root",
                str(tiny_data_root),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (out / "result.json").exists()
    assert (
        main(
            [
                "bench",
                "--model",
                str(out),
                "--no-creativity",
                "--max-items",
                "3",
                "--out",
                str(tmp_path / "b.json"),
                "--data-root",
                str(tiny_data_root),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert "scores" in json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("héllo 🙂 <|end|>\n/raw\nOnce upon\n/quit\n")
    )  # Review Focus 1 & 5
    assert (
        main(
            [
                "chat",
                "--model",
                str(out),
                "--max-tokens",
                "8",
                "--data-root",
                str(tiny_data_root),
                "--device",
                "cpu",
            ]
        )
        == 0
    )


def test_bad_config_exit_2(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text('{"run_id": "x"}', encoding="utf-8")
    assert main(["train", "--config", str(p), "--out", str(tmp_path / "o")]) == 2
    assert "error:" in capsys.readouterr().err


# -- helpers --------------------------------------------------------------------------------------


def _common(root) -> list[str]:
    return ["--data-root", str(root), "--device", "cpu"]


def _run(argv: list[str]) -> tuple[int, str, str]:
    """``main(argv)``, with what it printed to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _small_config(run_id: str = "cli-small", steps: int = 6, **changes) -> TrainRunConfig:
    return TrainRunConfig(
        run_id=run_id,
        seed=1,
        shape=ModelShape(2, 64, 64),
        token_budget=1024 * steps,
        mixture={"creative": 1.0},
        batch_tokens=1024,
        **changes,
    )


def _write_config(tmp_path: Path, cfg: TrainRunConfig, name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(cfg.to_json(), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def trained(tiny_data_root, tmp_path_factory):
    """One tiny model, trained once through the CLI and shared by the tests that only read it."""
    work = tmp_path_factory.mktemp("cli_model")
    config = _write_config(work, _small_config("shared", steps=6))
    code, out, err = _run(
        ["train", "--config", str(config), "--out", str(work / "m"), *_common(tiny_data_root)]
    )
    assert code == 0, err
    return SimpleNamespace(dir=work / "m", output=out, config=config)


def _chat(model, root, text: str, monkeypatch, *extra: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    return _run(["chat", "--model", str(model), *extra, *_common(root)])


# -- train ----------------------------------------------------------------------------------------


def test_train_prints_progress_samples_and_a_summary(trained):
    lines = trained.output.splitlines()
    assert lines[0].startswith("training shared: 6 steps, 6,144 tokens, on cpu, into ")
    assert lines[1].startswith("step 1/6  ")  # the header comes once the data has loaded
    assert any(
        re.fullmatch(r"step 1/6  loss \d+\.\d{3}  lr \d\.\d{2}e[+-]\d{2}", ln) for ln in lines
    )
    assert any(
        re.fullmatch(r"step 6/6  loss \d+\.\d{3}  lr \d\.\d{2}e[+-]\d{2}", ln) for ln in lines
    )
    assert any(ln.startswith("  heldout [step 6] ") and "creative " in ln for ln in lines)
    assert any(ln.startswith("  sample [Once upon a time]: ") for ln in lines)
    assert "training finished" in trained.output
    assert re.search(r"tokens\s+6,144", trained.output)
    assert str(trained.dir) in trained.output
    result = json.loads((trained.dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed" and result["steps"] == 6


def test_train_continues_from_a_parent_given_on_the_command_line(trained, tiny_data_root, tmp_path):
    # The config names a parent that does not exist; --parent replaces it before validation.
    cfg = _small_config("child", steps=3, parent_dir=str(tmp_path / "nowhere"))
    code, _, err = _run(
        [
            "train",
            "--config",
            str(_write_config(tmp_path, cfg)),
            "--out",
            str(tmp_path / "child"),
            "--parent",
            str(trained.dir),
            *_common(tiny_data_root),
        ]
    )
    assert code == 0, err
    parent = json.loads((trained.dir / "meta.json").read_text(encoding="utf-8"))
    child = json.loads((tmp_path / "child" / "meta.json").read_text(encoding="utf-8"))
    assert child["parent_version_id"] == parent["version_id"]
    assert child["lineage_id"] == parent["lineage_id"]


def test_train_resume_continues_an_interrupted_run(tiny_data_root, tmp_path):
    cfg = _small_config("resumable", steps=6)
    out_dir = tmp_path / "run"
    train_run(  # what a run cut short on the owner's PC leaves behind
        cfg,
        out_dir=out_dir,
        data_root=tiny_data_root,
        device="cpu",
        _hooks=TrainHooks(stop_after_steps=3),
    )
    assert (out_dir / "resume").is_dir()
    argv = ["train", "--config", str(_write_config(tmp_path, cfg)), "--out", str(out_dir)]
    code, out, err = _run([*argv, "--resume", *_common(tiny_data_root)])
    assert code == 0, err
    assert "step 6/6" in out
    assert "step 1/6" not in out  # it carried on from step 3 rather than starting again
    assert json.loads((out_dir / "result.json").read_text(encoding="utf-8"))["steps"] == 6


def test_train_resume_without_a_run_to_resume_says_to_run_without_resume(tiny_data_root, tmp_path):
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config())), "--resume"]
    code, out, err = _run([*argv, "--out", str(tmp_path / "fresh"), *_common(tiny_data_root)])
    assert code == 2 and out == ""
    assert err.startswith("error: ") and "nothing saved to resume" in err
    assert "without --resume" in err and "Traceback" not in err
    assert not (tmp_path / "fresh").exists()


def _unfinished_run(root, tmp_path, steps: int = 6, stop_after: int = 3):
    """What a run cut short on the owner's PC leaves behind: (config file, out folder)."""
    cfg = _small_config("unfinished", steps=steps)
    out_dir = tmp_path / "run"
    train_run(
        cfg,
        out_dir=out_dir,
        data_root=root,
        device="cpu",
        _hooks=TrainHooks(stop_after_steps=stop_after),
    )
    assert (out_dir / "resume").is_dir()
    return _write_config(tmp_path, cfg), out_dir


def test_train_refuses_to_start_over_an_unfinished_run(tiny_data_root, tmp_path):
    config, out_dir = _unfinished_run(tiny_data_root, tmp_path)
    saved = sorted(p.name for p in (out_dir / "resume").iterdir())
    argv = ["train", "--config", str(config), "--out", str(out_dir), *_common(tiny_data_root)]
    code, out, err = _run(argv)
    assert code == 2 and out == ""  # refused before any training step
    assert err.startswith("error: ") and "unfinished run of this config" in err
    assert "--resume" in err and "delete the folder" in err and str(out_dir) in err
    assert sorted(p.name for p in (out_dir / "resume").iterdir()) == saved  # nothing was lost
    code, out, err = _run([*argv, "--resume"])  # and the way it suggests works
    assert code == 0, err
    assert "step 1/6" not in out and "step 6/6" in out


def test_train_into_a_folder_with_a_finished_run_is_not_refused(trained, tiny_data_root, tmp_path):
    finished = tmp_path / "finished"
    shutil.copytree(trained.dir, finished)  # a completed run: its resume state is gone
    argv = ["train", "--config", str(trained.config), "--out", str(finished)]
    code, _, err = _run([*argv, *_common(tiny_data_root)])
    assert code == 0, err


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{not json", "cannot be used"),
        ("[1, 2]", "must be an object"),
        ('{"run_id": "x"}', "missing"),
        (_small_config().to_json().replace('"creative"', '"nonsense"'), "nonsense"),
        (_small_config().to_json().replace('"run_id": "cli-small"', '"run_id": "a/b"'), "run_id"),
        ("﻿" + '{"run_id": "x"}', "missing"),  # a Windows editor's BOM is not the problem
    ],
)
def test_a_bad_config_says_what_is_wrong(text, expected, tiny_data_root, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    code, _, err = _run(
        ["train", "--config", str(path), "--out", str(tmp_path / "o"), *_common(tiny_data_root)]
    )
    assert code == 2
    assert err.startswith("error: ") and expected in err
    assert "Traceback" not in err and not (tmp_path / "o").exists()


def test_a_missing_config_file_is_named(tmp_path):
    missing = tmp_path / "no-such.json"
    code, _, err = _run(["train", "--config", str(missing), "--out", str(tmp_path / "o")])
    assert code == 2 and str(missing) in err


def test_a_missing_parent_model_is_named(tiny_data_root, tmp_path):
    parent = tmp_path / "no-parent"
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config())), "--parent"]
    code, _, err = _run(
        [*argv, str(parent), "--out", str(tmp_path / "o"), *_common(tiny_data_root)]
    )
    assert code == 2 and str(parent) in err and "parent model" in err


def test_training_data_that_was_never_built_is_a_plain_error(tiny_tok_path, tmp_path):
    root = tmp_path / "root"  # a tokenizer but no corpus
    tokenizer_path(root).parent.mkdir(parents=True)
    shutil.copyfile(tiny_tok_path, tokenizer_path(root))
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config()))]
    code, out, err = _run([*argv, "--out", str(tmp_path / "o"), *_common(root)])
    assert code == 2 and "cannot find" in err and "airace-content build" in err
    assert "Traceback" not in err
    assert out == ""  # no "training ..." header for a run that cannot start


def _interrupting_train_run(save_first: int | None):
    """A ``train_run`` that is interrupted, after saving resume state at ``save_first`` steps."""
    real = train_run

    def interrupted(cfg, **kwargs):
        if save_first is not None:
            real(cfg, **kwargs, _hooks=TrainHooks(stop_after_steps=save_first))
        raise KeyboardInterrupt

    return interrupted


def test_interrupting_train_after_a_save_says_to_add_resume(monkeypatch, tiny_data_root, tmp_path):
    monkeypatch.setattr(cli, "train_run", _interrupting_train_run(3))
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config()))]
    code, _, err = _run([*argv, "--out", str(tmp_path / "o"), *_common(tiny_data_root)])
    assert code == 130
    assert "interrupted" in err and "again with --resume" in err and "without" not in err


def test_interrupting_train_before_any_save_says_to_run_again_without_resume(
    monkeypatch, tiny_data_root, tmp_path
):
    monkeypatch.setattr(cli, "train_run", _interrupting_train_run(None))
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config()))]
    code, _, err = _run([*argv, "--out", str(tmp_path / "o"), *_common(tiny_data_root)])
    assert code == 130
    assert "interrupted" in err and "without --resume" in err and "200 steps" in err


# -- missing pieces are usage errors (exit 2) -----------------------------------------------------


@pytest.mark.parametrize("command", ["chat", "bench", "fingerprint"])
def test_a_missing_model_folder_is_named(command, tiny_data_root, tmp_path):
    missing = tmp_path / "no-model"
    code, _, err = _run([command, "--model", str(missing), *_common(tiny_data_root)])
    assert code == 2
    assert err.startswith("error: ") and str(missing) in err and "airace-ml train" in err


@pytest.mark.parametrize("command", ["train", "chat", "bench", "fingerprint", "build-judge"])
def test_a_missing_tokenizer_is_named_and_exit_2(command, trained, tmp_path):
    empty = tmp_path / "empty-root"
    extra = {
        "train": ["--config", str(trained.config), "--out", str(tmp_path / "o")],
        "build-judge": [],
    }.get(command, ["--model", str(trained.dir)])
    code, _, err = _run([command, *extra, *_common(empty)])
    assert code == 2
    assert err.startswith("error: ")
    assert str(tokenizer_path(empty)) in err and "airace-content build" in err
    assert "Traceback" not in err


def test_a_model_made_with_another_tokenizer_is_refused(trained, fixture_texts, tmp_path):
    root = tmp_path / "other-root"
    tokenizer_path(root).parent.mkdir(parents=True)
    train_tokenizer(fixture_texts, tokenizer_path(root), vocab_size=300)
    code, _, err = _run(["fingerprint", "--model", str(trained.dir), *_common(root)])
    assert code == 2 and "vocabulary" in err and "--data-root" in err


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has a GPU")
def test_asking_for_a_gpu_that_is_not_there_is_a_plain_error(trained, tiny_data_root):
    argv = ["fingerprint", "--model", str(trained.dir), "--data-root", str(tiny_data_root)]
    code, _, err = _run([*argv, "--device", "cuda"])
    assert code == 2 and "--device cuda" in err and "no GPU" in err


def test_usage_errors_return_2_and_never_exit():
    for argv in (
        [],
        ["train"],
        ["no-such-command"],
        ["chat", "--model", "m", "--temperature", "-1"],
    ):
        code, out, err = _run(argv)
        assert code == 2, argv
        assert err.startswith("error: ") and "--help" in err
    code, out, err = _run(["train", "--help"])
    assert code == 0 and "--config" in out and err == ""
    code, out, err = _run(["--help"])
    assert code == 0 and all(c in out for c in ("train", "chat", "bench", "gate"))


# -- chat -----------------------------------------------------------------------------------------


def test_chat_keeps_history_and_reset_clears_it(trained, tiny_data_root, monkeypatch):
    calls = []

    def fake_reply(self, history, **options):
        calls.append((list(history), options))
        return f"reply{len(calls) - 1}"

    monkeypatch.setattr(TorchLM, "chat_reply", fake_reply)
    typed = "hello <|end|>\nagain\n/reset\nfresh\n/quit\n"
    code, out, err = _chat(
        trained.dir, tiny_data_root, typed, monkeypatch, "--max-tokens", "7", "--temperature", "0.5"
    )
    assert code == 0, err
    assert [h for h, _ in calls] == [
        [("user", "hello <|end|>")],  # special-token text goes through as plain text
        [("user", "hello <|end|>"), ("ai", "reply0"), ("user", "again")],
        [("user", "fresh")],  # /reset started the conversation over
    ]
    assert [o["seed"] for _, o in calls] == [0, 1, 0]
    assert all(o["max_new_tokens"] == 7 and o["temperature"] == 0.5 for _, o in calls)
    assert out.count("> ") == 5  # a prompt for each line read
    assert "reply0" in out and "reply1" in out and "reply2" in out and "cleared" in out


def test_chat_raw_toggles_plain_completion(trained, tiny_data_root, monkeypatch):
    chats, completions = [], []

    def fake_reply(self, history, **options):
        chats.append(list(history))
        return "chatted"

    def fake_complete(self, text, **options):
        completions.append((text, options["seed"]))
        return " and so on"

    monkeypatch.setattr(TorchLM, "chat_reply", fake_reply)
    monkeypatch.setattr(TorchLM, "complete", fake_complete)
    typed = "one\n/raw\nOnce upon\n/raw\ntwo\n/quit\n"
    code, out, err = _chat(trained.dir, tiny_data_root, typed, monkeypatch)
    assert code == 0, err
    assert completions == [("Once upon", 1)]
    assert chats == [[("user", "one")], [("user", "one"), ("ai", "chatted"), ("user", "two")]]
    assert "raw mode on" in out and "chat mode on" in out and " and so on" in out


def test_chat_ignores_blank_lines_flags_typos_and_stops_at_end_of_input(
    trained, tiny_data_root, monkeypatch
):
    monkeypatch.setattr(TorchLM, "chat_reply", lambda self, history, **kw: "ok")
    code, out, err = _chat(trained.dir, tiny_data_root, "\n   \n/quitt\nhi\n", monkeypatch)
    assert code == 0, err  # the input ended without /quit
    assert "unknown command" in out and out.count("> ok\n") == 1


def test_chat_says_so_when_the_model_says_nothing(trained, tiny_data_root, monkeypatch):
    histories = []

    def fake_reply(self, history, **options):
        histories.append(list(history))
        return ""

    monkeypatch.setattr(TorchLM, "chat_reply", fake_reply)
    code, out, _ = _chat(trained.dir, tiny_data_root, "hi\nagain\n/quit\n", monkeypatch)
    assert code == 0 and "(the model said nothing)" in out
    assert histories[1] == [("user", "hi"), ("ai", ""), ("user", "again")]  # kept as it was


def test_chat_with_a_real_model_survives_text_and_commands(trained, tiny_data_root, monkeypatch):
    typed = "héllo 🙂 <|end|> <|ai|>\n/raw\nÇa va? 日本語\n/reset\n/quit\n"
    code, _, err = _chat(trained.dir, tiny_data_root, typed, monkeypatch, "--max-tokens", "6")
    assert code == 0, err


def test_interrupting_chat_returns_130(trained, tiny_data_root, monkeypatch):
    class Interrupting(io.StringIO):
        def readline(self, *args):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", Interrupting())
    code, _, err = _run(["chat", "--model", str(trained.dir), *_common(tiny_data_root)])
    assert code == 130 and "interrupted" in err


# -- non-ASCII text (Review Focus 5) --------------------------------------------------------------

TRICKY = "café 🙂 日本語 \udc80"  # accents, emoji, CJK and a lone surrogate no codec can encode


def _cp1252_console():
    """A stream like the Windows console or a redirect on a Windows PC: it encodes as cp1252."""
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", write_through=True)


def test_reconfiguring_stops_a_cp1252_console_from_crashing(monkeypatch):
    raw, stream = _cp1252_console()
    with pytest.raises(UnicodeEncodeError):  # the hazard: this is what printing did before
        stream.write(TRICKY)
    monkeypatch.setattr(sys, "stdout", stream)
    cli._use_utf8_output()
    print(TRICKY, flush=True)
    assert raw.getvalue().decode("utf-8").endswith("café 🙂 日本語 ?\n")


def test_model_text_that_cp1252_cannot_show_does_not_crash_the_cli(
    trained, tiny_data_root, monkeypatch
):
    out_bytes, stdout = _cp1252_console()
    err_bytes, stderr = _cp1252_console()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(TorchLM, "chat_reply", lambda self, history, **kw: TRICKY)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\n/quit\n"))
    assert main(["chat", "--model", str(trained.dir), *_common(tiny_data_root)]) == 0
    assert "café 🙂 日本語 ?" in out_bytes.getvalue().decode("utf-8")
    # an error that names a non-ASCII path is printed as well
    nowhere = trained.dir.parent / "modèle-🙂"
    assert main(["bench", "--model", str(nowhere), *_common(tiny_data_root)]) == 2
    assert "modèle-🙂" in err_bytes.getvalue().decode("utf-8")


def test_train_progress_survives_a_cp1252_console(tiny_data_root, tmp_path, monkeypatch):
    out_bytes, stdout = _cp1252_console()
    monkeypatch.setattr(sys, "stdout", stdout)
    cfg = _small_config("cp1252", steps=2, probe_prompts=["Voilà: café 🙂 日本語"])
    argv = ["train", "--config", str(_write_config(tmp_path, cfg))]
    assert main([*argv, "--out", str(tmp_path / "o"), *_common(tiny_data_root)]) == 0
    text = out_bytes.getvalue().decode("utf-8")
    assert "sample [Voilà: café 🙂 日本語]: " in text


def test_chat_reads_piped_input_as_utf8_even_on_a_cp1252_machine(
    trained, tiny_data_root, monkeypatch
):
    seen = []

    def fake_reply(self, history, **options):
        seen.append(history[-1][1])
        return "ok"

    monkeypatch.setattr(TorchLM, "chat_reply", fake_reply)
    piped = b"h\xc3\xa9llo \xf0\x9f\x99\x82 \x81\n/quit\n"  # UTF-8, then a byte cp1252 lacks
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(piped), encoding="cp1252"))
    code, _, err = _run(["chat", "--model", str(trained.dir), *_common(tiny_data_root)])
    assert code == 0, err
    assert seen == ["héllo 🙂 �"]


# -- bench ----------------------------------------------------------------------------------------


def test_bench_prints_a_table_and_notes_that_creativity_is_missing(
    trained, tiny_data_root, tmp_path
):
    report_path = tmp_path / "reports" / "nested" / "bench.json"
    code, out, err = _run(
        ["bench", "--model", str(trained.dir), "--max-items", "2", "--out", str(report_path)]
        + _common(tiny_data_root)
    )
    assert code == 0, err
    assert re.search(r"^language\s+-?\d+\.\d\s+2$", out, re.MULTILINE)
    assert re.search(r"^overall\s+-?\d+\.\d$", out, re.MULTILINE)
    assert "not measured: creativity" in out
    assert "airace-ml build-judge" in out and "airace-ml build-novelty-index" in out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["missing"] == ["creativity"] and "creativity" not in report["scores"]
    assert all(cat in report["scores"] for cat in ("language", "reasoning", "instruction"))


def test_bench_without_creativity_does_not_call_it_missing(trained, tiny_data_root, tmp_path):
    report_path = tmp_path / "bench.json"
    code, out, err = _run(
        ["bench", "--model", str(trained.dir), "--no-creativity", "--max-items", "1"]
        + ["--out", str(report_path)]
        + _common(tiny_data_root)
    )
    assert code == 0, err
    assert "not measured" not in out and "note:" not in out
    assert json.loads(report_path.read_text(encoding="utf-8"))["missing"] == []


# -- fingerprint ----------------------------------------------------------------------------------


def test_fingerprint_prints_every_trait_in_order(trained, tiny_data_root):
    code, out, err = _run(
        ["fingerprint", "--model", str(trained.dir), "--k", "1", "--seed", "3"]
        + _common(tiny_data_root)
    )
    assert code == 0, err
    rows = [line.split() for line in out.splitlines()]
    assert [r[0] for r in rows] == list(TRAITS)
    assert all(len(r) == 2 and float(r[1]) == float(r[1]) for r in rows)  # a number, never NaN


# -- build-novelty-index, build-judge and bench with creativity -----------------------------------


@pytest.fixture(scope="module")
def judged(tiny_data_root, tmp_path_factory):
    """A copy of the tiny data root with a novelty index and a (very small) judge built through
    the CLI. The judge recipe is shrunk to a 2-layer, 64-wide model, as the judge tests do."""
    root = tmp_path_factory.mktemp("judged_root") / "root"
    shutil.copytree(tiny_data_root, root)
    real = judge_module.judge_train_config
    patch = pytest.MonkeyPatch()
    patch.setattr(
        judge_module,
        "judge_train_config",
        lambda seed=1234: replace(real(seed), shape=ModelShape(2, 64, 64), batch_tokens=1024),
    )
    try:
        index = _run(["build-novelty-index", "--data-root", str(root)])
        judge = _run(["build-judge", "--budget-tokens", "3072", *_common(root)])
    finally:
        patch.undo()
    return SimpleNamespace(root=root, index=index, judge=judge)


def test_build_novelty_index_writes_the_index(judged):
    code, out, err = judged.index
    assert code == 0, err
    assert novelty_path(judged.root).is_file() and str(novelty_path(judged.root)) in out


def test_build_judge_shows_training_progress_and_where_the_judge_is(judged):
    code, out, err = judged.judge
    assert code == 0, err
    assert re.search(r"^step 3/3  loss \d+\.\d{3}  lr ", out, re.MULTILINE)
    assert out.index("step 3/3") < out.index("calibrating") < out.index("ready in")
    assert f"ready in {judge_dir(judged.root)}" in out
    assert (judge_dir(judged.root) / "calibration.json").is_file()


def test_bench_scores_creativity_once_the_judge_and_index_exist(trained, judged):
    code, out, err = _run(
        ["bench", "--model", str(trained.dir), "--max-items", "1"] + _common(judged.root)
    )
    assert code == 0, err
    assert re.search(r"^creativity\s+-?\d+\.\d\s+\d+$", out, re.MULTILINE)
    assert "not measured" not in out and "note:" not in out


def test_build_novelty_index_without_a_corpus_is_a_plain_error(tmp_path):
    code, _, err = _run(["build-novelty-index", "--data-root", str(tmp_path)])
    assert code == 2 and err.startswith("error: cannot find ") and "Traceback" not in err


def test_build_judge_passes_its_budget_and_shows_progress(monkeypatch, tiny_data_root, tmp_path):
    seen = {}

    def fake_build(root, *, device, on_event, token_budget):
        seen.update(root=root, budget=token_budget, device=device)
        on_event(Progress(1, 2, 1024, 5.5, 1e-3))
        return tmp_path / "judge"

    monkeypatch.setattr(cli, "build_judge", fake_build)
    code, out, err = _run(["build-judge", "--budget-tokens", "65536", *_common(tiny_data_root)])
    assert code == 0, err
    assert seen == {"root": tiny_data_root, "budget": 65536, "device": torch.device("cpu")}
    assert "step 1/2  loss 5.500  lr 1.00e-03" in out and "judge" in out


def test_a_judge_that_did_not_finish_training_exits_1(monkeypatch, tiny_data_root):
    def failing(*args, **kwargs):
        raise JudgeBuildError("the judge's training ended 'unstable_stopped' after 3 of 9 steps")

    monkeypatch.setattr(cli, "build_judge", failing)
    code, _, err = _run(["build-judge", *_common(tiny_data_root)])
    assert code == 1
    assert err.startswith("error: the judge's training ended") and "Traceback" not in err


# -- unstable runs, bad paths, damaged files: plain errors, never a traceback ----------------------


def test_a_run_that_goes_unstable_stops_early_and_says_what_to_try(
    tiny_data_root, tmp_path, monkeypatch
):
    real = cli.train_run
    spikes = TrainHooks(loss_override=lambda step, loss: math.nan if step in (2, 3, 4) else loss)
    monkeypatch.setattr(cli, "train_run", lambda cfg, **kw: real(cfg, **kw, _hooks=spikes))
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config("wobbly", steps=6)))]
    code, out, err = _run([*argv, "--out", str(tmp_path / "o"), *_common(tiny_data_root)])
    assert code == 0, err  # the last good weights were saved, so the run is a result
    assert out.count("  warning [step") == 3
    assert "training stopped early because it became unstable" in out
    assert re.search(r"after 4 of 6 steps", out)
    assert "last good weights were kept" in out and str(tmp_path / "o") in out
    assert "lower boldness" in out and "more careful learning style" in out
    result = json.loads((tmp_path / "o" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "unstable_stopped" and result["steps"] == 4


@pytest.mark.parametrize("where", ["file", "inside-a-file"])
def test_an_output_path_that_is_a_file_is_a_plain_error(where, tiny_data_root, tmp_path):
    taken = tmp_path / "taken"
    taken.write_text("keep me", encoding="utf-8")
    target = taken if where == "file" else taken / "run"
    argv = ["train", "--config", str(_write_config(tmp_path, _small_config()))]
    code, out, err = _run([*argv, "--out", str(target), *_common(tiny_data_root)])
    assert code == 2 and out == ""
    assert err.startswith("error: ") and str(taken) in err and "folder" in err
    assert "Traceback" not in err and taken.read_text(encoding="utf-8") == "keep me"


def _no_benchmark(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("the benchmark must not start")

    monkeypatch.setattr(cli, "run_benchmarks", refused)


def test_a_report_path_that_is_a_folder_is_refused_before_the_benchmark(
    trained, tiny_data_root, tmp_path, monkeypatch
):
    _no_benchmark(monkeypatch)
    reports = tmp_path / "reports"
    reports.mkdir()
    argv = ["bench", "--model", str(trained.dir), "--no-creativity", "--out", str(reports)]
    code, out, err = _run([*argv, *_common(tiny_data_root)])
    assert code == 2 and out == ""
    assert err.startswith("error: ") and str(reports) in err and "folder" in err
    assert "Traceback" not in err


def test_a_report_path_inside_a_file_is_refused_before_the_benchmark(
    trained, tiny_data_root, tmp_path, monkeypatch
):
    _no_benchmark(monkeypatch)
    taken = tmp_path / "taken"
    taken.write_text("x", encoding="utf-8")
    argv = ["bench", "--model", str(trained.dir), "--no-creativity", "--out", str(taken / "b.json")]
    code, out, err = _run([*argv, *_common(tiny_data_root)])
    assert code == 2 and out == "" and str(taken) in err and "Traceback" not in err


def test_a_report_that_cannot_be_written_is_a_plain_error_after_the_table(
    trained, tiny_data_root, tmp_path, monkeypatch
):
    real = Path.write_text

    def deny(self, *args, **kwargs):
        if self.name == "bench.json":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", deny)
    argv = ["bench", "--model", str(trained.dir), "--no-creativity", "--max-items", "1"]
    code, out, err = _run([*argv, "--out", str(tmp_path / "bench.json"), *_common(tiny_data_root)])
    assert code == 2 and "overall" in out  # the scores were still shown
    assert err.startswith("error: ") and "bench.json" in err and "Traceback" not in err


@pytest.fixture
def damaged(judged, tmp_path):
    """A private copy of the judged data root, for tests that break something in it."""
    root = tmp_path / "damaged_root"
    shutil.copytree(judged.root, root)
    return root


def _damage_index_record(root):
    record = novelty_path(root).with_suffix(".json")
    meta = json.loads(record.read_text(encoding="utf-8"))
    meta["count"] += 1  # a save cut short: the record no longer matches the array
    record.write_text(json.dumps(meta), encoding="utf-8")


def _damage_index_array(root):
    novelty_path(root).write_bytes(b"this is not an array")


def _damage_index_empty(root):
    novelty_path(root).write_bytes(b"")  # np.load raises EOFError on a zero-byte file


def _damage_judge_calibration(root):
    (judge_dir(root) / "calibration.json").write_text("{}", encoding="utf-8")


def _damage_judge_json(root):
    (judge_dir(root) / "calibration.json").write_text("{ half a file", encoding="utf-8")


def _damage_judge_weights(root):
    (judge_dir(root) / "model.safetensors").write_bytes(b"not a model at all")


@pytest.mark.parametrize(
    "damage, rebuild",
    [
        (_damage_index_record, "airace-ml build-novelty-index"),
        (_damage_index_array, "airace-ml build-novelty-index"),
        (_damage_index_empty, "airace-ml build-novelty-index"),
        (_damage_judge_calibration, "airace-ml build-judge"),
        (_damage_judge_json, "airace-ml build-judge"),
        (_damage_judge_weights, "airace-ml build-judge"),
    ],
)
def test_a_damaged_judge_or_index_is_an_error_that_names_the_rebuild(
    damage, rebuild, trained, damaged
):
    damage(damaged)
    argv = ["bench", "--model", str(trained.dir), "--max-items", "1", *_common(damaged)]
    code, out, err = _run(argv)
    assert code == 2 and out == ""  # refused before the benchmark ran
    assert err.startswith("error: ") and rebuild in err and "Traceback" not in err
    # creativity is the only thing that needs them
    code, out, err = _run([*argv, "--no-creativity"])
    assert code == 0, err


@pytest.mark.parametrize("command", ["chat", "bench", "fingerprint"])
@pytest.mark.parametrize("damage", ["garbage", "truncated"])
def test_a_damaged_model_file_is_a_plain_error(command, damage, trained, tiny_data_root, tmp_path):
    model = tmp_path / "model"
    shutil.copytree(trained.dir, model)
    weights = model / "model.safetensors"
    weights.write_bytes(b"garbage" * 50 if damage == "garbage" else weights.read_bytes()[:200])
    code, out, err = _run([command, "--model", str(model), *_common(tiny_data_root)])
    assert code == 2 and out == ""
    assert err.startswith("error: ") and "cannot load the model" in err and str(model) in err
    assert "Traceback" not in err


# -- build-judge budget, and options with a minimum -----------------------------------------------


def _no_judge_build(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("no training may start")

    monkeypatch.setattr(cli, "build_judge", refused)


def test_a_judge_budget_below_one_step_of_the_real_recipe_is_refused_up_front(
    tiny_data_root, monkeypatch
):
    _no_judge_build(monkeypatch)  # the real recipe: 32,768 tokens per step, not shrunk
    code, out, err = _run(["build-judge", "--budget-tokens", "2048", *_common(tiny_data_root)])
    assert code == 2 and out == ""
    assert err.startswith("error: ") and "2048" in err and "32,768" in err
    assert "Traceback" not in err


def test_a_judge_budget_of_exactly_one_step_is_accepted(tiny_data_root, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "build_judge", lambda root, **kw: seen.append(kw) or tmp_path)
    argv = ["build-judge", "--budget-tokens", "32768", *_common(tiny_data_root)]
    code, _, err = _run(argv)
    assert code == 0, err
    assert seen[0]["token_budget"] == 32768


def test_a_judge_build_that_hits_a_config_error_is_a_plain_error(tiny_data_root, monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("token_budget (5) must be at least batch_tokens (32768)")

    monkeypatch.setattr(cli, "build_judge", invalid)
    code, _, err = _run(["build-judge", *_common(tiny_data_root)])
    assert code == 2 and err.startswith("error: token_budget") and "Traceback" not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "--model", "m", "--max-tokens", "0"],
        ["bench", "--model", "m", "--max-items", "0"],
        ["bench", "--model", "m", "--max-items", "-1"],
        ["fingerprint", "--model", "m", "--k", "0"],
        ["fingerprint", "--model", "m", "--seed", "-1"],
        ["fingerprint", "--model", "m", "--seed", "soon"],
        ["build-judge", "--budget-tokens", "0"],
    ],
)
def test_options_with_a_minimum_reject_values_below_it(argv):
    code, out, err = _run(argv)
    assert code == 2 and out == ""
    assert err.startswith("error: ") and "whole number" in err and "--help" in err


def test_a_seed_of_zero_and_a_max_items_of_one_are_fine(trained, tiny_data_root):
    argv = ["fingerprint", "--model", str(trained.dir), "--k", "1", "--seed", "0"]
    assert _run([*argv, *_common(tiny_data_root)])[0] == 0
    argv = ["bench", "--model", str(trained.dir), "--no-creativity", "--max-items", "1"]
    assert _run([*argv, *_common(tiny_data_root)])[0] == 0


def test_build_judge_says_when_it_moves_on_to_calibrating(monkeypatch, tiny_data_root, tmp_path):
    def fake_build(root, *, device, on_event, token_budget):
        on_event(Progress(1, 2, 1024, 5.5, 1e-3))
        on_event(Progress(2, 2, 2048, 5.1, 1e-4))
        on_event(Done("completed", {}))
        return tmp_path / "judge"

    monkeypatch.setattr(cli, "build_judge", fake_build)
    code, out, err = _run(["build-judge", *_common(tiny_data_root)])
    assert code == 0, err
    lines = out.splitlines()
    step2 = lines.index("step 2/2  loss 5.100  lr 1.00e-04")
    calibrating = next(i for i, ln in enumerate(lines) if "calibrating" in ln)
    ready = next(i for i, ln in enumerate(lines) if "ready in" in ln)
    assert step2 < calibrating < ready


def test_build_judge_does_not_say_calibrating_when_training_did_not_complete(
    monkeypatch, tiny_data_root
):
    def fake_build(root, *, device, on_event, token_budget):
        on_event(Done("unstable_stopped", {}))
        raise JudgeBuildError("the judge's training ended 'unstable_stopped' after 3 of 9 steps")

    monkeypatch.setattr(cli, "build_judge", fake_build)
    code, out, _ = _run(["build-judge", *_common(tiny_data_root)])
    assert code == 1 and "calibrating" not in out


# -- gate -----------------------------------------------------------------------------------------


def test_gate_is_not_available_yet(tmp_path):
    argv = ["gate", "--out", str(tmp_path / "g"), "--seeds", "1,2", "--cpu-speed", "--quick"]
    code, _, err = _run(argv)
    assert code == 2
    assert err.startswith("error: ") and "gate not available" in err


def test_gate_checks_its_seeds():
    code, _, err = _run(["gate", "--out", "g", "--seeds", "1,x"])
    assert code == 2 and "--seeds" in err and "1,2,3" in err
