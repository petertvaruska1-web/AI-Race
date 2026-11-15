"""Lessons lab: do a player's lessons stick after ONE run?

In the game a player writes lessons (a question and the answer the AI should give), switches on
"Train on your lessons" and trains a run. This script builds that run exactly as the game server
does and then asks the AI the lessons back. The target is at least 7 of 10 answered.

Run it from ``ml/`` (it uses the GPU; the lead runs it)::

    python -m uv run python ../docs/handoff/next-level/lessons_lab.py \\
        --parent runs/lab-replacement-off/runs/first-s1 --out runs/lessons-lab \\
        [--shares 0.05 0.1 0.2] [--seed 30001]

Hidden options for a dry run on the tiny test data: ``--data-root DIR``, ``--device cpu``,
``--tokens N``, ``--batch-tokens N`` and ``--shape LAYERS D_MODEL CTX``.

What is built, for each share
-----------------------------
The player keeps the first AI's brain (Tiny), picks the Short length, leaves the mix at its even
default and switches on nothing but "Train on your lessons", with the first 10 of
``chatprobes.LESSONS``. The server turns that plan into a ``TrainRunConfig`` in
``toTrainRunConfig`` (``apps/server/src/training/config.ts``). The values taken from there and
from the rules ``balance-v3`` (``packages/sim/src/balance.ts``, whose lengths and Tiny brain are
those of ``BALANCE_V2``) are:

* ``BATCH_TOKENS = 8192`` (config.ts) gives ``batch_tokens``.
* ``COACHING_SHARE = 0.05`` (config.ts) is the share of the run's text that the lessons get; it
  is this script's variable ``--shares`` (0.05 is the game's value today). One source of lessons
  (no "Learn from customers") gives a coaching share of exactly the share.
* Short is ``{ id: "short", tokens: 6_291_456 }`` (balance.ts, ``BALANCE_V2.lengths``, inherited
  by ``BALANCE_V3``). With no tutoring packs the paid budget equals the length, and the server
  sends the run's recorded ``preview.tokens`` (training.ts ``previewPlan``: the paid text times
  one plus the Research bonus, which is zero without a Research team), so ``token_budget`` is
  6_291_456, that is 768 steps of 8192 tokens.
* Tiny is ``{ nLayer: 4, dModel: 128 }`` (balance.ts); the attention span of the default plan
  is 256 (``defaultPlan("tiny", 256, ...)``): ``shape = ModelShape(4, 128, 256)``.
* The default plan (``defaultPlan`` in balance.ts): even mix of 12 for each of the eight
  datasets and 0 for the notebook (``EVEN_MIX``), ``cleaning = "standard"``,
  ``removeDuplicates = false``, ``factCheck = false``, ``variety = "natural"``,
  ``learningStyle = 0.5``, ``review = 0``, ``probePrompt = ""``, ``tutoring = null``. They become
  ``prep = PrepConfig("standard", dedup=False, fact_check=False, variety="natural")``,
  ``boldness = 0.5``, ``replay = 0.0`` and ``probe_prompts = []``.
* The mixture (config.ts): each base weight is ``own * base * w / total`` with
  ``own = 1 - coachingShare`` and ``base = length / budget = 1`` (no packs), which is
  ``(1 - share) / 8`` for each dataset, and ``mixture.coaching = share``.
* The server leaves out ``notebook``, ``purchases``, ``finishing_mixture`` (Python defaults:
  none) and sends ``parent_dir`` (here the parent checkpoint folder). The seed is the run's own
  (``run.seed``); here it is ``--seed``.

Where the server differs from the scorecard's recipes (``experiments/scorecard.py``): the
scorecard trains a recipe step of 12_582_912 tokens on the plain mix, with no coaching, and
gives a continuation a seed of its own; the server's Short run is half as long and always carries
the coaching dataset. This script follows the server.

What is measured, for the parent and for each trained child
-----------------------------------------------------------
(a) ``chatprobes.lessons``: the lesson questions asked as written, greedy; right when the reply
holds the answer's last word. (b) The same questions typed casually (lower case, no ending
punctuation). The game reads the lessons as written but base text with casual typing, so (b) asks
whether the lessons survive how people really type. (c) ``scorecard.yardstick_loss``: what the
lessons cost the rest. (d) How many times each lesson was read: the run's tokens drawn for the
coaching dataset (``tokens_per_dataset``) over the tokens of the 10 lessons as the trainer
encodes them (one pass reads each lesson once); estimated from share x tokens when the run did not
report it.

Everything goes to ``OUT/results.json``; each share trains into ``OUT/share-<share>/`` and a
finished run is reused on a re-run (``lab_run.json`` beside it names the parent, share, seed and
setup, and a mismatch is refused).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from airace_ml.data.prep import PrepConfig
from airace_ml.experiments import chatprobes
from airace_ml.experiments.scorecard import yardstick_loss
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.paths import data_root as default_data_root
from airace_ml.paths import tokenizer_path
from airace_ml.tokenizer import Tok, encode_chat
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.trainer import can_resume, finished_run, train_run

DATASETS = ("web", "books", "educational", "conversations", "code", "reasoning", "facts", "creative")
EVEN_WEIGHT = 12.0  # balance.ts EVEN_MIX: every dataset 12, the notebook 0
BATCH_TOKENS = 8192  # config.ts BATCH_TOKENS
SHORT_TOKENS = 6_291_456  # balance.ts BALANCE_V2.lengths, "short" (also in balance-v3)
TINY_SHAPE = ModelShape(4, 128, 256)  # balance.ts "tiny" 4 x 128, default plan span 256
N_LESSONS = 10
RECORD_FILE = "lab_run.json"
CASUAL = chatprobes.STYLES["lower_nopunct"]


def server_mixture(share: float) -> dict[str, float]:
    """The mixture ``toTrainRunConfig`` builds for the even default mix, no packs and one source
    of lessons with the given coaching share: each dataset gets ``(1 - share) * 12 / 96`` and
    ``coaching`` gets ``share``."""
    if not 0.0 < share < 1.0:
        raise ValueError(f"the coaching share must be between 0 and 1, got {share!r}")
    total = EVEN_WEIGHT * len(DATASETS)
    own = 1.0 - share
    base = 1.0  # length.tokens / budget; no tutoring packs
    mixture = {ds: own * base * EVEN_WEIGHT / total for ds in DATASETS}
    mixture["coaching"] = share
    return mixture


def server_config(
    share: float,
    *,
    run_id: str,
    seed: int,
    shape: ModelShape,
    tokens: int,
    batch_tokens: int,
    parent_dir: str,
    lessons: list[tuple[str, str]],
) -> TrainRunConfig:
    """The ``TrainRunConfig`` the server sends for the plan described in the module docstring."""
    return TrainRunConfig(
        run_id=run_id,
        seed=seed,
        shape=shape,
        token_budget=tokens,
        mixture=server_mixture(share),
        coaching=list(lessons),
        replay=0.0,
        prep=PrepConfig("standard", False, False, "natural"),
        boldness=0.5,
        batch_tokens=batch_tokens,
        probe_prompts=[],
        parent_dir=parent_dir,
    )


class _Asker:
    """A model seen through the lessons probe: questions are re-typed with ``style`` and every
    question with its reply is recorded. It only reads replies (R1)."""

    def __init__(self, lm: TorchLM, style=lambda text: text) -> None:
        self.lm = lm
        self.style = style
        self.log: list[tuple[str, str]] = []

    def chat_reply(self, history, **kwargs) -> str:
        typed = [(role, self.style(text) if role == "user" else text) for role, text in history]
        reply = self.lm.chat_reply(typed, **kwargs)
        self.log.append((typed[-1][1], reply))
        return reply


def probe_lessons(lm: TorchLM, style=lambda text: text) -> dict:
    """Ask the lessons of ``lm`` (already trained on them, if at all) and score the replies."""
    asker = _Asker(lm, style)
    score = chatprobes.lessons(lambda pairs: asker, n=N_LESSONS)  # type: ignore[arg-type]
    rows = []
    for (question, answer), (asked, reply) in zip(
        chatprobes.LESSONS[:N_LESSONS], asker.log, strict=True
    ):
        key = chatprobes._key_word(answer)
        rows.append(
            {
                "asked": asked,
                "question": question,
                "expected": answer,
                "reply": reply,
                "right": chatprobes._holds_word(reply, key),
            }
        )
    assert sum(r["right"] for r in rows) == score.passed
    return {"passed": score.passed, "total": score.total, "rows": rows}


def measure(
    ckpt: Path, tok: Tok, device: torch.device, root: Path, label: str
) -> dict:
    """The lessons asked as written and typed casually, and the yardstick loss, of one checkpoint."""
    model, _ = load_checkpoint(ckpt, device)
    lm = TorchLM(model, tok, device)
    written = probe_lessons(lm)
    casual = probe_lessons(lm, CASUAL)
    loss = yardstick_loss(model, root)
    for name, probe in (("as written", written), ("typed casually", casual)):
        print(f"\n[{label}] lessons {name}: {probe['passed']} of {probe['total']}")
        for row in probe["rows"]:
            mark = "ok  " if row["right"] else "MISS"
            print(f"  {mark} Q: {row['asked']}\n       want: {row['expected']}\n       got:  {row['reply']!r}")
    print(f"[{label}] yardstick loss {loss:.4f}")
    del lm, model
    return {"written": written, "casual": casual, "yardstick_loss": loss}


def lesson_tokens(tok: Tok) -> int:
    """Tokens of the lessons as the trainer reads them: one chat document per lesson."""
    return sum(
        len(encode_chat(tok, [("user", q), ("ai", a)]))
        for q, a in chatprobes.LESSONS[:N_LESSONS]
    )


def _json_load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def train_share(
    share: float, args, out: Path, parent: Path, root: Path, device: torch.device
) -> tuple[Path, dict]:
    """Train (or reuse) the run of one share; returns its folder and its record."""
    folder = out / f"share-{share}"
    cfg = server_config(
        share,
        run_id=f"lessons-{str(share).replace('.', 'p')}",
        seed=args.seed,
        shape=args.shape,
        tokens=args.tokens,
        batch_tokens=args.batch_tokens,
        parent_dir=str(parent),
        lessons=list(chatprobes.LESSONS[:N_LESSONS]),
    )
    cfg.validate()
    identity = {
        "parent": str(parent.resolve()),
        "share": share,
        "seed": args.seed,
        "tokens": args.tokens,
        "batch_tokens": args.batch_tokens,
        "shape": cfg.shape.to_dict(),
    }
    record_path = folder / RECORD_FILE
    if folder.exists() and any(folder.iterdir()):
        record = _json_load(record_path)
        if not isinstance(record, dict) or record.get("identity") != identity:
            sys.exit(
                f"{folder} holds a run of another setup ({record.get('identity') if isinstance(record, dict) else 'no record'}); "
                f"use another --out or remove the folder"
            )
        result = finished_run(folder, cfg)
        if result is not None:
            print(f"share {share}: finished earlier, reusing it")
            return folder, {"identity": identity, "result": result, "reused": True}
        resume = can_resume(folder, cfg)
    else:
        resume = False
        folder.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps({"identity": identity}, indent=2), encoding="utf-8")
    print(f"share {share}: {'resuming' if resume else 'training'} {cfg.steps} steps, {cfg.token_budget:,} tokens")
    done = train_run(cfg, out_dir=folder, data_root=root, device=device, resume=resume)
    result = finished_run(folder, cfg)
    if result is None:
        sys.exit(f"share {share}: the run ended {done.status}, not finished; run again to resume it")
    return folder, {"identity": identity, "result": result, "reused": False}


def passes(result: dict, share: float, tokens: int, per_pass: int) -> dict:
    """How many times each lesson was read, from the run's tokens drawn for the lessons if it
    reports them, else from share x tokens."""
    drawn = (result.get("tokens_per_dataset") or {}).get("coaching")
    if drawn is not None:
        return {"coaching_tokens": drawn, "passes": drawn / per_pass, "source": "run"}
    return {"coaching_tokens": share * tokens, "passes": share * tokens / per_pass, "source": "estimate"}


def table(results: dict) -> str:
    parent_loss = results["parent"]["yardstick_loss"]
    lines = [
        f"{'share':>6} {'answered':>9} {'answered casual':>16} {'yardstick':>10} {'(parent)':>9} {'passes':>8}",
        f"{'parent':>6} {results['parent']['written']['passed']:>7}/10 "
        f"{results['parent']['casual']['passed']:>13}/10 {parent_loss:>10.4f} {parent_loss:>9.4f} {'-':>8}",
    ]
    for share, r in results["shares"].items():
        lines.append(
            f"{share:>6} {r['written']['passed']:>7}/10 {r['casual']['passed']:>13}/10 "
            f"{r['yardstick_loss']:>10.4f} {parent_loss:>9.4f} {r['passes']['passes']:>8.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Do a player's lessons stick after one run?")
    p.add_argument("--parent", required=True, type=Path, help="the first AI's checkpoint folder")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--shares", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--seed", type=int, default=30001)
    p.add_argument("--data-root", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--device", default="auto", help=argparse.SUPPRESS)
    p.add_argument("--tokens", type=int, default=SHORT_TOKENS, help=argparse.SUPPRESS)
    p.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS, help=argparse.SUPPRESS)
    p.add_argument("--shape", type=int, nargs=3, metavar=("LAYERS", "D_MODEL", "CTX"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.shape = ModelShape(*args.shape) if args.shape else TINY_SHAPE

    root = args.data_root or default_data_root()
    if args.device == "auto":
        from airace_ml.device import pick_device

        device = pick_device()
    else:
        device = torch.device(args.device)
    tok = Tok.load(tokenizer_path(root))
    args.out.mkdir(parents=True, exist_ok=True)
    per_pass = lesson_tokens(tok)
    print(f"parent {args.parent}; data {root}; device {device}; the 10 lessons are {per_pass} tokens")

    results: dict = {
        "setup": {
            "parent": str(args.parent),
            "seed": args.seed,
            "tokens": args.tokens,
            "batch_tokens": args.batch_tokens,
            "shape": args.shape.to_dict(),
            "lesson_tokens_per_pass": per_pass,
            "data_root": str(root),
        },
        "shares": {},
    }
    results["parent"] = measure(args.parent, tok, device, root, "parent")
    for share in args.shares:
        folder, info = train_share(share, args, args.out, args.parent, root, device)
        m = measure(folder, tok, device, root, f"share {share}")
        m["passes"] = passes(info["result"], share, args.tokens, per_pass)
        m["config_mixture"] = server_mixture(share)
        m["reused"] = info["reused"]
        results["shares"][str(share)] = m
        (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + table(results))
    print(f"\nwritten to {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
