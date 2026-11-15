"""Lab: how much should a chat count the person's turns in the training loss?

A chat document trains the model on every token, the person's turns included, although the model
never writes a person's turn when it is used. This script sets the weight of the person's turns
(``airace_ml.train.trainer.USER_TURN_LOSS_WEIGHT``; 1.0 is today's loss), runs the scorecard
exactly as ``airace-ml scorecard`` does (the same entry point, so the PC stays awake the same
way), and then measures every model of the card on the AI's turns alone.

Run it from ``ml/``:

    python -m uv run python ../docs/handoff/next-level/user_weight_lab.py --weight 0.3 \\
        --out runs/uw-0.3 [--recipes first,small,chatty] [--baseline runs/uw-1.0/scorecard.json]

Every weight needs its own ``--out``: the scorecard's run cache knows the code, the data and the
recipe, but not the weight, so a folder used for one weight would hand its trained models to the
next. Weight 1.0 is the baseline to compare the others against.

Afterwards ``DIR/ai_turn_loss.json`` holds, for every model, the yardstick loss over the same 128
packed rows the scorecard's ``yardstick_loss`` uses, split three ways: on the AI's turns (the
AI's text and the ``<|end|>`` that closes it), on the person's turns (the ``<|user|>`` marker,
the text and its ``<|end|>``) and on all targets, which must equal the card's ``yardstick_loss``
to three decimals. The tokens of neither kind (the ``<|bos|>``, a ``<|sys|>`` line, the
``<|ai|>`` marker, plain text) are in the third figure only.

Exit codes are the scorecard's (0 fine, 1 a measure is worse than the baseline, 2 usage).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from airace_ml import cli
from airace_ml.awake import stay_awake
from airace_ml.data.corpus import Corpus
from airace_ml.data.prep import heldout_docs
from airace_ml.data.sampler import DocPool, MixtureSampler
from airace_ml.experiments.scorecard import RUNS_DIR, YARDSTICK_DATASET, YARDSTICK_SEQS
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok
from airace_ml.train import trainer

REPORT_NAME = "ai_turn_loss.json"
CARD_NAME = "scorecard.json"
MATCH_PLACES = 3  # the all-targets loss must equal the card's yardstick loss to this many places


def ai_turn_mask(inputs: torch.Tensor, targets: torch.Tensor, tok: Tok) -> torch.Tensor:
    """Which targets are an AI's text or the ``<|end|>`` that closes its turn: those after an
    ``<|ai|>`` marker (not the marker itself) up to and including the turn's ``<|end|>``. The rule
    runs along each packed row, as :func:`airace_ml.train.trainer._loss_weights` does for the
    person's turns."""
    seq = torch.cat([inputs[:, :1], targets], dim=1)
    is_ai = seq == tok.ai_id
    changes = is_ai | (seq == tok.user_id) | (seq == tok.bos_id) | (seq == tok.sys_id)
    changes |= seq == tok.end_id
    position = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
    last = torch.where(changes, position, -1).cummax(dim=1).values
    inside = (last >= 0) & is_ai.gather(1, last.clamp(min=0))
    before = torch.cat([torch.zeros_like(inside[:, :1]), inside[:, :-1]], dim=1)
    return torch.where(seq == tok.end_id, before, inside & ~is_ai)[:, 1:]


def user_turn_mask(inputs: torch.Tensor, targets: torch.Tensor, tok: Tok) -> torch.Tensor:
    """Which targets are in a person's turn, by the trainer's own rule."""
    weights = trainer._loss_weights(
        inputs, targets, tok.user_id, tok.ai_id, tok.end_id, tok.bos_id, tok.sys_id, 0.0
    )
    return weights == 0.0


def yardstick_rows(model: Transformer, yardstick_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows the scorecard's ``yardstick_loss`` takes its mean over (its own row building)."""
    corpus = Corpus.open(corpus_dir(Path(yardstick_root)) / YARDSTICK_DATASET)
    pool = DocPool.from_corpus(corpus, heldout_docs(corpus), balanced=False, replacement=True)
    sampler = MixtureSampler(
        {YARDSTICK_DATASET: pool}, {YARDSTICK_DATASET: 1.0}, model.shape.ctx_len, 0
    )
    x, y = sampler.next_batch(YARDSTICK_SEQS)
    device = next(model.parameters()).device
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.inference_mode()
def turn_losses(model: Transformer, tok: Tok, yardstick_root: Path) -> dict[str, float | int]:
    """The mean loss per target token of ``model`` on the AI's turns, the person's turns and all
    targets of the yardstick rows, and how many targets each is taken over."""
    x, y = yardstick_rows(model, yardstick_root)
    model.eval()
    rows = max(1, trainer.MICRO_BATCH_TOKENS // x.shape[1])
    pieces = []
    for start in range(0, len(x), rows):
        with trainer._autocast(x.device):
            logits = model(x[start : start + rows])
            pieces.append(
                torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), y[start : start + rows].flatten(), reduction="none"
                )
            )
    nll = torch.cat(pieces).float().view(y.shape)
    ai, user = ai_turn_mask(x, y, tok), user_turn_mask(x, y, tok)

    def mean(mask: torch.Tensor | None) -> float:
        picked = nll if mask is None else nll[mask]
        return float(picked.mean()) if picked.numel() else float("nan")

    return {
        "ai_turn_loss": mean(ai),
        "user_turn_loss": mean(user),
        "all_loss": mean(None),
        "ai_targets": int(ai.sum()),
        "user_targets": int(user.sum()),
        "all_targets": int(nll.numel()),
    }


def measure_card(out: Path, device: torch.device | str = "cpu") -> dict[str, dict]:
    """Measure every model of the card saved in ``out``, check the all-targets loss against the
    card's own and write ``ai_turn_loss.json`` beside it; the figures by model name."""
    out = Path(out)
    card = json.loads((out / CARD_NAME).read_text(encoding="utf-8"))
    tok = Tok.load(tokenizer_path(Path(card["data_root"])))
    result = {}
    for name, entry in card["models"].items():
        model, _ = load_checkpoint(out / RUNS_DIR / name, device)
        figures = turn_losses(model, tok, Path(card["yardstick_root"]))
        wanted = entry["measures"]["yardstick_loss"]
        assert round(figures["all_loss"], MATCH_PLACES) == round(wanted, MATCH_PLACES), (
            f"{name}: the loss over all targets is {figures['all_loss']:.4f} but the card's "
            f"yardstick loss is {wanted:.4f}"
        )
        result[name] = figures
        print(
            f"{name}: AI turns {figures['ai_turn_loss']:.3f}, person's turns "
            f"{figures['user_turn_loss']:.3f}, all {figures['all_loss']:.3f}",
            flush=True,
        )
    (out / REPORT_NAME).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weight", type=float, required=True, help="weight of the person's turns")
    parser.add_argument("--out", type=Path, required=True, help="a folder of its own per weight")
    parser.add_argument("--recipes", help="comma-separated recipes (default: the scorecard's)")
    parser.add_argument("--baseline", type=Path, help="a saved scorecard.json to compare against")
    parser.add_argument("--data-root", type=Path, help="the data root (default: the CLI's)")
    args = parser.parse_args(argv)
    if not 0.0 <= args.weight <= 1.0:
        parser.error("--weight must be between 0 and 1")
    trainer.USER_TURN_LOSS_WEIGHT = args.weight
    command = ["scorecard", "--out", str(args.out)]
    if args.recipes:
        command += ["--recipes", args.recipes]
    if args.baseline:
        command += ["--baseline", str(args.baseline)]
    if args.data_root:
        command += ["--data-root", str(args.data_root)]
    code = cli.main(command)
    if code not in (0, 1) or not (args.out / CARD_NAME).is_file():
        return code
    with stay_awake():
        measure_card(args.out, "cuda" if torch.cuda.is_available() else "cpu")
    return code


if __name__ == "__main__":
    sys.exit(main())
