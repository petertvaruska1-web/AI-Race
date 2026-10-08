"""``airace-ml``: train, talk to, measure and benchmark models on the local machine.

    airace-ml train --config FILE --out DIR [--parent DIR] [--resume]
    airace-ml chat --model DIR [--temperature F] [--max-tokens N]
    airace-ml bench --model DIR [--out FILE] [--no-creativity] [--max-items N]
    airace-ml fingerprint --model DIR [--k N] [--seed N]
    airace-ml build-novelty-index
    airace-ml build-judge [--budget-tokens N]
    airace-ml gate --out DIR [--seeds 1,2,3] [--cpu-speed] [--quick]

Every command also takes ``--data-root PATH`` (default: ``AIRACE_DATA`` or ``<repo>/data``) and
``--device {cpu,cuda}`` (default: the GPU when there is one).

Exit codes: 0 on success; 2 when the request or the state of the files it names is wrong (a bad
option, a malformed or invalid config, a missing, damaged or unwritable model, tokenizer, data,
index, judge or output path, an unfinished run that would be overwritten, a command that is not
available yet); 1 when a run fails for a reason that is not the user's input (the judge's training
went unstable, the disk filled up); 130 when the user interrupts a run. Each of those prints one
``error: ...`` line on stderr and never a traceback; anything unexpected is a bug and propagates.
A training run that goes unstable and stops early still saved its last good weights, so it exits 0
and says so.

``main`` returns the exit code and never calls ``sys.exit`` (argparse's own exits are turned into
return values), so it can be driven from tests and from other tools.

Output is always UTF-8 and never fails on a character the console cannot show: a model that has
barely trained emits arbitrary bytes, and the Windows console defaults to a code page that cannot
print most of Unicode.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import SafetensorError

import airace_ml.evals.judge as judge_module
from airace_ml.device import pick_device
from airace_ml.evals.judge import Judge, JudgeBuildError, build_judge
from airace_ml.evals.novelty import NoveltyIndex, build_novelty_index
from airace_ml.evals.scoring import BenchReport
from airace_ml.evals.suite import CATEGORIES, run_benchmarks
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME, load_checkpoint
from airace_ml.paths import data_root, judge_dir, novelty_path, tokenizer_path
from airace_ml.personality.fingerprint import TRAITS, measure_fingerprint
from airace_ml.tokenizer import Role, Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import (
    Done,
    HeldoutEval,
    Instability,
    Progress,
    Sample,
    TrainEvent,
    json_safe,
)
from airace_ml.train.trainer import (
    CHECKPOINT_EVERY,
    TrainResult,
    can_resume,
    has_resume_state,
    train_run,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 96
CHAT_COMMANDS = ("/reset", "/raw", "/quit")
_COMMAND_LIKE = re.compile(r"/[A-Za-z]+")  # looks like a chat command, so a typo is not chatted
GATE_SEEDS = (1, 2, 3)
# Everything that can be wrong with a file someone else wrote or that a crash left half-saved.
_DAMAGED = (OSError, ValueError, KeyError, TypeError, EOFError, SafetensorError)
# Loading a checkpoint adds one more: torch's RuntimeError when meta.json's shape does not match
# the weights. Catch this only around the load itself, never around a whole run.
_CHECKPOINT_DAMAGED = (*_DAMAGED, RuntimeError)

BUILD_DATA_HINT = "airace-content build --scale tiny (a quick test set) or --scale full"


class _UsageError(Exception):
    """The request is wrong: reported as ``error: ...``, exit code 2."""


class _RunFailed(Exception):
    """The request was fine but the run failed: reported as ``error: ...``, exit code 1."""


class _Interrupted(Exception):
    """The user stopped a run (Ctrl-C); ``hint`` says what to do next. Exit code 130."""

    def __init__(self, hint: str) -> None:
        super().__init__(hint)
        self.hint = hint


class _Exit(Exception):
    """argparse wanted to exit (``--help``): carried to ``main`` as a return value."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class _Parser(argparse.ArgumentParser):
    """An argparse parser that reports instead of exiting, so ``main`` can return the code."""

    def error(self, message: str):
        raise _UsageError(f"{message} (run '{self.prog} --help' to see the options)")

    def exit(self, status: int = 0, message: str | None = None):
        if message:
            self._print_message(message, sys.stderr)
        raise _Exit(status)


# -- argument types -------------------------------------------------------------------------------


def _whole_number(minimum: int):
    """An argparse type: a whole number that is at least ``minimum``."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            value = minimum - 1
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of {minimum} or more")
        return value

    return parse


def _temperature(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of 0 or more")
    return value


def _seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in text.split(","))
    except ValueError:
        seeds = ()
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of whole numbers, like 1,2,3"
        )
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="airace-ml", description="Train, talk to, measure and benchmark AI Race models."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-root",
        type=Path,
        metavar="PATH",
        help="folder holding the built training data (default: AIRACE_DATA or <repo>/data)",
    )
    common.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="where to compute (default: the GPU when there is one, else the CPU)",
    )

    def add(name: str, run, help: str, hint: str = "") -> argparse.ArgumentParser:
        sub = commands.add_parser(name, parents=[common], help=help, description=help)
        sub.set_defaults(run=run, interrupt_hint=hint)
        return sub

    train = add("train", _cmd_train, "train a model from a run config")
    train.add_argument("--config", type=Path, required=True, help="the run config (JSON file)")
    train.add_argument("--out", type=Path, required=True, help="folder to save the model in")
    train.add_argument("--parent", type=Path, help="folder of the model to continue from")
    train.add_argument(
        "--resume", action="store_true", help="continue an interrupted run of this config"
    )

    chat = add("chat", _cmd_chat, "talk to a trained model")
    chat.add_argument("--model", type=Path, required=True, help="folder of the trained model")
    chat.add_argument(
        "--temperature",
        type=_temperature,
        default=DEFAULT_TEMPERATURE,
        help=f"0 is always the likeliest word, higher is looser (default {DEFAULT_TEMPERATURE})",
    )
    chat.add_argument(
        "--max-tokens",
        type=_whole_number(1),
        default=DEFAULT_MAX_TOKENS,
        help=f"longest reply, in tokens (default {DEFAULT_MAX_TOKENS})",
    )

    bench = add("bench", _cmd_bench, "score a trained model on the benchmark suite")
    bench.add_argument("--model", type=Path, required=True, help="folder of the trained model")
    bench.add_argument("--out", type=Path, help="write the full report here (JSON)")
    bench.add_argument(
        "--no-creativity",
        action="store_true",
        help="skip the creativity category (it needs the judge and the novelty index)",
    )
    bench.add_argument(
        "--max-items",
        type=_whole_number(1),
        metavar="N",
        help="score at most N items per category (a quick, rough look)",
    )

    fingerprint = add("fingerprint", _cmd_fingerprint, "measure a trained model's personality")
    fingerprint.add_argument(
        "--model", type=Path, required=True, help="folder of the trained model"
    )
    fingerprint.add_argument(
        "--k", type=_whole_number(1), default=3, help="replies per question (default 3)"
    )
    fingerprint.add_argument(
        "--seed", type=_whole_number(0), default=0, help="random seed (default 0)"
    )

    add(
        "build-novelty-index",
        _cmd_build_novelty_index,
        "index the training text, so creativity can tell new writing from copied writing",
    )

    judge = add(
        "build-judge",
        _cmd_build_judge,
        "train the reference judge that creativity is scored against",
        f"Run the same command again: it carries on from the last saved point if there was one "
        f"(progress is saved every {CHECKPOINT_EVERY} steps), and starts over if not.",
    )
    judge.add_argument(
        "--budget-tokens",
        type=_whole_number(1),
        metavar="N",
        help="train on N tokens instead of the full recipe (for a quick test)",
    )

    gate = add("gate", _cmd_gate, "run the feasibility gate")
    gate.add_argument("--out", type=Path, required=True, help="folder for the gate's results")
    gate.add_argument(
        "--seeds",
        type=_seeds,
        default=GATE_SEEDS,
        help="comma-separated seeds, one run each (default 1,2,3)",
    )
    gate.add_argument("--cpu-speed", action="store_true", help="also measure the CPU's speed")
    gate.add_argument("--quick", action="store_true", help="a short run, for checking the setup")
    return parser


# -- entry point ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``airace-ml`` command and return its exit code (see the module docstring)."""
    _use_utf8_output()
    args = None
    try:
        args = _parser().parse_args(argv)
        return args.run(args)
    except _Exit as e:
        return e.status
    except _UsageError as e:
        return _fail(str(e), EXIT_USAGE)
    except _RunFailed as e:
        return _fail(str(e), EXIT_FAILED)
    except _Interrupted as e:
        return _interrupted(e.hint)
    except KeyboardInterrupt:
        return _interrupted(getattr(args, "interrupt_hint", ""))


def _interrupted(hint: str) -> int:
    print(f"\ninterrupted. {hint}".rstrip(), file=sys.stderr, flush=True)
    return EXIT_INTERRUPTED


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr, flush=True)
    return code


def _use_utf8_output() -> None:
    """Make stdout and stderr UTF-8 and unable to fail on an unencodable character.

    Without this, printing a model's text on a console or a redirect that uses a legacy code page
    (Windows cp1252) raises ``UnicodeEncodeError``. A stream without ``reconfigure`` (a test
    harness's capture object) is left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # a stream that has already been read or written cannot change; keep it


def _use_utf8_input() -> None:
    """Read piped or redirected stdin as UTF-8 that never fails on a stray byte. A console is left
    alone: Python already reads it as Unicode."""
    reconfigure = getattr(sys.stdin, "reconfigure", None)
    if reconfigure is None or sys.stdin.isatty():
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass


# -- shared helpers -------------------------------------------------------------------------------


def _data_root(args: argparse.Namespace) -> Path:
    return Path(args.data_root) if args.data_root is not None else data_root()


def _device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise _UsageError(
            "--device cuda was asked for, but this computer has no GPU that can be used. "
            "Leave --device out to use whatever is available, or pass --device cpu."
        )
    return pick_device(args.device)


def _missing(error: FileNotFoundError) -> str:
    if not error.filename:
        return str(error)
    return (
        f"cannot find {error.filename}. Check the paths, and that the training data has been "
        f"built (airace-content build)."
    )


def _require_tokenizer(root: Path) -> Tok:
    path = tokenizer_path(root)
    if not path.is_file():
        raise _UsageError(
            f"cannot find the tokenizer at {path}. Build the training data first with "
            f"{BUILD_DATA_HINT}, or point --data-root at the folder that holds it."
        )
    return Tok.load(path)


def _why(error: BaseException) -> str:
    """A short reason for a damaged file, for the people who have to fix it."""
    shown = (OSError, ValueError, SafetensorError, RuntimeError)
    text = str(error) if isinstance(error, shown) else ""
    reason = " ".join((text or f"{type(error).__name__}: {error}").split())  # one line
    return reason if len(reason) <= 300 else reason[:297] + "..."


def _os_problem(error: OSError, what: str) -> Exception:
    """A plain error for a file system problem, ``what`` first and then the operating system's own
    words. A path the user chose that cannot be used is a usage error (2); anything else (a full
    disk, say) is a failed run (1)."""
    message = f"{what}: {error.strerror or error}"
    if error.filename and str(error.filename) not in what:
        message += f": {error.filename}"
    if isinstance(error, (PermissionError, NotADirectoryError, IsADirectoryError, FileExistsError)):
        return _UsageError(message)
    return _RunFailed(message)


def _files_problem(error: OSError) -> Exception:
    """A file system problem somewhere in a whole run, which may be a read or a write."""
    return _os_problem(error, "could not read or write files")


def _require_folder_path(path: Path, option: str) -> None:
    """``path`` can be a folder: neither it nor any folder above it is a file."""
    for candidate in (path, *path.parents):
        if candidate.exists() and not candidate.is_dir():
            what = "it" if candidate == path else str(candidate)
            raise _UsageError(
                f"cannot use {path} for {option}: {what} is a file, not a folder. "
                f"Choose another folder name."
            )


def _require_report_path(path: Path) -> None:
    """``path`` can be a report file: not a folder, and no folder above it is a file."""
    if path.is_dir():
        raise _UsageError(
            f"--out {path} is a folder; give a file name instead, such as {path / 'bench.json'}"
        )
    _require_folder_path(path.parent, "the folder of --out")


def _require_model_dir(path: Path, what: str = "model") -> None:
    if not (path / META_NAME).is_file() or not (path / WEIGHTS_NAME).is_file():
        raise _UsageError(
            f"no trained {what} found in {path} (it should hold {WEIGHTS_NAME} and {META_NAME}). "
            f"Train one first with: airace-ml train --config FILE --out {path}"
        )


def _load_checkpoint(folder: Path, device: torch.device, what: str = "model"):
    """``load_checkpoint``, with every way the files can be wrong turned into a plain error that
    names ``folder``. Wraps the load and nothing else."""
    try:
        return load_checkpoint(folder, device)
    except _CHECKPOINT_DAMAGED as e:
        raise _UsageError(
            f"cannot load the {what} in {folder} ({_why(e)}). "
            f"Is it a folder made by 'airace-ml train', and is the file complete?"
        ) from e


def _load_lm(model_dir: Path, root: Path, device: torch.device) -> TorchLM:
    _require_model_dir(model_dir)
    tok = _require_tokenizer(root)
    net, _ = _load_checkpoint(model_dir, device)
    if net.tok_emb.num_embeddings != tok.vocab_size:
        raise _UsageError(
            f"the model in {model_dir} was trained with a {net.tok_emb.num_embeddings}-word "
            f"vocabulary, but the tokenizer under {root} has {tok.vocab_size}. "
            f"Point --data-root at the data the model was trained with."
        )
    return TorchLM(net, tok, device)


# -- train and build-judge: progress output -------------------------------------------------------


def _print_event(event: TrainEvent) -> None:
    """One line (or so) per training event, flushed so a redirected log keeps up."""
    match event:
        case Progress():
            line = (
                f"step {event.step}/{event.total_steps}  loss {event.loss:.3f}  lr {event.lr:.2e}"
            )
        case HeldoutEval():
            losses = "  ".join(f"{name} {loss:.3f}" for name, loss in event.losses.items())
            line = f"  heldout [step {event.step}] {losses}"
        case Sample():
            line = f"  sample [{event.prompt}]: {event.text}"
        case Instability(action="rollback"):
            line = (
                f"  warning [step {event.step}]: training became unstable, so it went back to "
                f"the last good weights and slowed learning to x{event.lr_scale:g} of the plan"
            )
        case Instability():
            line = (
                f"  warning [step {event.step}]: training became unstable again, so the run is "
                f"stopping and keeping the last good weights"
            )
        case _:
            return
    print(line, flush=True)


def _print_summary(result: TrainResult, total_steps: int) -> None:
    print()
    if result.status == "completed":
        print("training finished")
    else:
        print(
            f"training stopped early because it became unstable, after {result.steps} of "
            f"{total_steps} steps."
        )
        print("The last good weights were kept, so the model can still be used.")
        print(
            "To try again, use a more careful learning style: lower boldness in the config, "
            "which also makes the learning steps smaller."
        )
    print(f"  run         {result.run_id}")
    print(f"  steps       {result.steps}")
    print(f"  tokens      {result.tokens:,}")
    print(f"  final loss  {result.final_loss:.3f}")
    print(f"  time        {result.wall_seconds:.1f} s")
    print(f"  saved to    {result.out_dir}", flush=True)


# -- commands -------------------------------------------------------------------------------------


def _read_config(path: Path) -> TrainRunConfig:
    try:
        # utf-8-sig: still UTF-8, but also accepts the BOM that Windows editors like to add.
        return TrainRunConfig.from_json(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as e:
        raise _UsageError(f"cannot find the config file {path}") from e
    except (OSError, ValueError) as e:
        raise _UsageError(f"the config file {path} cannot be used: {e}") from e


def _holds_progress(out: Path) -> bool:
    """Whether ``out`` holds resume state of any run. A state too broken to read counts as held:
    it may be someone's progress, and a fresh run would delete it."""
    try:
        return has_resume_state(out)
    except _DAMAGED:
        return True


def _saved_progress(out: Path, cfg: TrainRunConfig) -> bool:
    """Whether ``out`` holds resume state saved by a run of exactly ``cfg``."""
    try:
        return can_resume(out, cfg)
    except _DAMAGED:
        return False


def _interrupted_train_hint(out: Path, cfg: TrainRunConfig) -> str:
    if _saved_progress(out, cfg):
        return (
            "Its progress up to the last saved point is kept: run the same command again "
            "with --resume to carry on from there."
        )
    return (
        f"No progress had been saved yet (a run saves its progress every {CHECKPOINT_EVERY} "
        f"steps), so run the same command again without --resume to start over."
    )


def _cmd_train(args: argparse.Namespace) -> int:
    device = _device(args)
    root = _data_root(args)
    cfg = _read_config(args.config)
    if args.parent is not None:
        cfg.parent_dir = str(args.parent)
    try:
        cfg.validate()
        if cfg.parent_dir is not None:
            _require_model_dir(Path(cfg.parent_dir), "parent model")
            _load_checkpoint(Path(cfg.parent_dir), torch.device("cpu"), "parent model")
        _require_tokenizer(root)
        _require_folder_path(args.out, "--out")
        saved = _saved_progress(args.out, cfg)
        if _holds_progress(args.out) and not saved:
            raise _UsageError(
                f"{args.out} holds unfinished progress from a different run config; running "
                f"there would discard it. Use another --out folder, or delete this one to "
                f"start over."
            )
        if args.resume and not saved:
            raise _UsageError(
                f"there is nothing saved to resume in {args.out} for this config (a run saves "
                f"its progress every {CHECKPOINT_EVERY} steps, so one stopped sooner has "
                f"none). Run the command again without --resume to start the run."
            )
        if saved and not args.resume:
            raise _UsageError(
                f"{args.out} holds an unfinished run of this config; add --resume to carry "
                f"on, or delete the folder to start over"
            )
        header = (
            f"{'resuming' if args.resume else 'training'} {cfg.run_id}: {cfg.steps} steps, "
            f"{cfg.token_budget:,} tokens, on {device.type}, into {args.out}"
        )
        result = train_run(
            cfg,
            out_dir=args.out,
            data_root=root,
            device=device,
            on_event=_announcing(header, _print_event),
            resume=args.resume,
        )
    except KeyboardInterrupt:
        raise _Interrupted(_interrupted_train_hint(args.out, cfg)) from None
    except FileNotFoundError as e:
        raise _UsageError(_missing(e)) from e
    except OSError as e:
        raise _files_problem(e) from e
    except ValueError as e:
        raise _UsageError(str(e)) from e
    _print_summary(result, cfg.steps)
    return EXIT_OK


def _announcing(header: str, handler):
    """An event handler that prints ``header`` just before the first event. The first event
    comes once the data has loaded and the model is set up, so a run that cannot start (no
    corpus, a bad mix) never prints a header as if it had."""
    shown = False

    def on_event(event: TrainEvent) -> None:
        nonlocal shown
        if not shown:
            shown = True
            print(header, flush=True)
        handler(event)

    return on_event


def _cmd_chat(args: argparse.Namespace) -> int:
    lm = _load_lm(args.model, _data_root(args), _device(args))
    _use_utf8_input()
    print(
        f"Talking to the model in {args.model}. Type /reset to start the conversation over, "
        f"/raw to switch between chat and plain text continuation, /quit to leave.",
        flush=True,
    )
    history: list[tuple[Role, str]] = []
    raw = False
    turn = 0  # generations since the conversation began; seeds the next one
    while True:
        print("> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:  # end of input
            print(flush=True)
            return EXIT_OK
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            return EXIT_OK
        if text == "/reset":
            history.clear()
            turn = 0
            print("(conversation cleared)", flush=True)
        elif text == "/raw":
            raw = not raw
            print(
                "(raw mode on: the model now continues what you type as plain text)"
                if raw
                else "(chat mode on)",
                flush=True,
            )
        elif _COMMAND_LIKE.fullmatch(text):
            print(f"(unknown command; the commands are {', '.join(CHAT_COMMANDS)})", flush=True)
        else:
            options = {
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": turn,
            }
            if raw:
                reply = lm.complete(text, **options)
            else:
                history.append(("user", text))
                reply = lm.chat_reply(history, **options)
                history.append(("ai", reply))
            turn += 1
            print(reply if reply.strip() else "(the model said nothing)", flush=True)


def _creativity_tools(root: Path, device: torch.device) -> tuple[Judge, NoveltyIndex] | None:
    """The judge and the novelty index under ``root``. ``None`` (with a note saying what is
    missing and how to build it) if either is missing; an error naming the command that rebuilds
    it if either is damaged or out of date."""
    judge = index = None
    todo = []
    try:
        judge = Judge.load(judge_dir(root), device)
    except FileNotFoundError:
        todo.append("  - the reference judge: build it with 'airace-ml build-judge'")
    except _CHECKPOINT_DAMAGED as e:
        raise _UsageError(
            f"the reference judge in {judge_dir(root)} is damaged or out of date "
            f"({_why(e)}). Rebuild it with: airace-ml build-judge"
        ) from e
    try:
        index = NoveltyIndex.load(novelty_path(root))
    except FileNotFoundError:
        todo.append("  - the novelty index: build it with 'airace-ml build-novelty-index'")
    except _DAMAGED as e:
        raise _UsageError(
            f"the novelty index at {novelty_path(root)} is damaged or out of date "
            f"({_why(e)}). Rebuild it with: airace-ml build-novelty-index"
        ) from e
    if judge is None or index is None:
        print(
            "note: creativity was left out of this run. Still to build:",
            *todo,
            sep="\n",
            flush=True,
        )
        return None
    return judge, index


def _print_report(report: BenchReport) -> None:
    print(f"{'category':<14}{'score':>8}{'items':>7}")
    for category, score in report.scores.items():
        print(f"{category:<14}{score.score:>8.1f}{score.n:>7}")
    print(f"{'overall':<14}{report.overall:>8.1f}")
    print("(score: 0 is no better than guessing, 100 is perfect)")
    if report.missing:
        print(f"not measured: {', '.join(report.missing)}")
    print(f"took {report.seconds:.1f} s", flush=True)


def _cmd_bench(args: argparse.Namespace) -> int:
    device = _device(args)
    root = _data_root(args)
    if args.out is not None:
        _require_report_path(args.out)  # before the benchmark, which can take a long time
    lm = _load_lm(args.model, root, device)
    categories = [c for c in CATEGORIES if not (args.no_creativity and c == "creativity")]
    tools = None if args.no_creativity else _creativity_tools(root, device)
    judge, novelty = tools if tools is not None else (None, None)
    report = run_benchmarks(
        lm,
        lm.tok,
        judge=judge,
        novelty=novelty,
        categories=categories,
        max_items_per_category=args.max_items,
    )
    _print_report(report)
    if args.out is not None:
        document = json.dumps(json_safe(report.to_dict()), allow_nan=False, indent=2)
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(document, encoding="utf-8")
        except OSError as e:
            raise _os_problem(e, f"could not write the report to {args.out}") from e
        print(f"full report written to {args.out}", flush=True)
    return EXIT_OK


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    lm = _load_lm(args.model, _data_root(args), _device(args))
    fp = measure_fingerprint(lm, lm.tok, k=args.k, seed=args.seed)
    for name in TRAITS:
        print(f"{name:<15}{fp.traits[name]:>8.3f}", flush=True)
    return EXIT_OK


def _cmd_build_novelty_index(args: argparse.Namespace) -> int:
    root = _data_root(args)
    try:
        path = build_novelty_index(root)
    except FileNotFoundError as e:
        raise _UsageError(_missing(e)) from e
    print(f"novelty index written to {path}", flush=True)
    return EXIT_OK


def _print_judge_event(event: TrainEvent) -> None:
    _print_event(event)
    if isinstance(event, Done) and event.status == "completed":
        print(
            "training finished; now calibrating the judge on real held-out text "
            "(this takes a little while)",
            flush=True,
        )


def _cmd_build_judge(args: argparse.Namespace) -> int:
    device = _device(args)
    root = _data_root(args)
    one_step = judge_module.judge_train_config().batch_tokens
    if args.budget_tokens is not None and args.budget_tokens < one_step:
        raise _UsageError(
            f"--budget-tokens {args.budget_tokens} is too small: the judge learns in steps of "
            f"{one_step:,} tokens, so the budget must be at least {one_step:,}."
        )
    _require_tokenizer(root)
    try:
        path = build_judge(
            root, device=device, on_event=_print_judge_event, token_budget=args.budget_tokens
        )
    except FileNotFoundError as e:
        raise _UsageError(_missing(e)) from e
    except JudgeBuildError as e:
        raise _RunFailed(str(e)) from e
    except OSError as e:
        raise _files_problem(e) from e
    print(f"\nreference judge ready in {path}", flush=True)
    return EXIT_OK


def _cmd_gate(args: argparse.Namespace) -> int:
    # Task 20 replaces this body with a call to airace_ml.experiments.gate.run_gate.
    raise _UsageError("gate not available yet: the feasibility gate has not been installed.")
