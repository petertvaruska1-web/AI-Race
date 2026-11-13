"""The scorecard (next-level spec 5.1): one measuring stick for every change to the training text
or the training code.

:func:`run_scorecard` trains the fixed :data:`RECIPES` on the data root's text and measures every
model the same way: its held-out losses (as its own training recorded them), its *yardstick* loss
on the yardstick root's held-out conversations (so models trained on different text compare),
bench-v1, and the talk probes of :mod:`~airace_ml.experiments.chatprobes`, drawn as the game
talks. The yardstick root also supplies the judge (the well-formed rule and creativity) and the
novelty index, so the stick never moves while the training text does.

Runs train under ``out/runs/<model>`` through :class:`~airace_ml.experiments.runs.GateRuns`, so a
finished run is reused while its recipe, data, training code and device are unchanged. Each
model's measurements are cached in ``out/measured/<model>.json`` and reused while the run, the
measuring code (:data:`CARD_SOURCES`), the yardstick and the scale are unchanged.
:func:`write_reports` writes ``scorecard.json`` and ``scorecard.md``; :func:`compare` lists what got
worse than a saved baseline by more than its :data:`TOLERANCE`.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from airace_ml.data.corpus import Corpus
from airace_ml.data.identity import data_identity
from airace_ml.data.prep import heldout_docs
from airace_ml.data.sampler import DocPool, MixtureSampler
from airace_ml.engine import ENGINE_VERSION
from airace_ml.evals.judge import CALIBRATION_NAME, Judge, is_well_formed
from airace_ml.evals.novelty import NoveltyIndex
from airace_ml.evals.suite import SUITE_VERSION, build_suite, run_benchmarks
from airace_ml.experiments import chatprobes
from airace_ml.experiments.configs import BALANCED_MIX, mix_heavy
from airace_ml.experiments.gate import (
    CONTINUATION_SEED_OFFSET,
    GateSetupError,
    _load_judge,
    _load_novelty,
)
from airace_ml.experiments.runs import (
    GATE_RUN_FILE,
    MEASURING_SOURCES,
    TRAINING_SOURCES,
    GateRun,
    GateRuns,
    _cached,
    _read_json,
    _write_json,
    source_digest,
)
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir, judge_dir, novelty_path, tokenizer_path
from airace_ml.skills.kb import KB, load_kb
from airace_ml.tokenizer import Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.trainer import _heldout_losses

CARD_VERSION = "scorecard-v1"
# What a measurement depends on besides its run: the measuring code and this module (how the
# probes are asked and the yardstick loss is taken).
CARD_SOURCES: tuple[str, ...] = (*MEASURING_SOURCES, "experiments/scorecard.py")
RUNS_DIR = "runs"
MEASURED_DIR = "measured"
YARDSTICK_DATASET = "conversations"
YARDSTICK_SEQS = 128  # held-out rows the yardstick loss is taken over (the trainer takes 8)
MEASURE_SEED = 0

TINY = ModelShape(4, 128, 256)
SMALL = ModelShape(6, 192, 256)
MEDIUM = ModelShape(8, 256, 256)
LARGE = ModelShape(10, 384, 256)
STEP_TOKENS = 12_582_912
CHATTY_TOKENS = 8_388_608


@dataclass(frozen=True)
class Step:
    """One model of a recipe: ``name`` is its folder and its row on the card. ``parent`` names
    the model it continues from (trained first if its recipe was not asked for)."""

    name: str
    shape: ModelShape
    tokens: int
    mix: Mapping[str, float]
    seed: int = 1
    parent: str | None = None


# A continuation reads text new to it: with its parent's seed it would read its parent's batches
# again (the gate's G5 found this), so each generation takes a seed of its own.
RECIPES: dict[str, list[Step]] = {
    "first": [
        Step("first-s1", TINY, STEP_TOKENS, BALANCED_MIX, seed=1),
        Step("first-s2", TINY, STEP_TOKENS, BALANCED_MIX, seed=2),
    ],
    "small": [Step("small", SMALL, STEP_TOKENS, BALANCED_MIX)],
    "lineage": [
        Step(
            "lineage-small",
            SMALL,
            STEP_TOKENS,
            BALANCED_MIX,
            seed=1 + CONTINUATION_SEED_OFFSET,
            parent="first-s1",
        ),
        Step(
            "lineage-medium",
            MEDIUM,
            STEP_TOKENS,
            BALANCED_MIX,
            seed=1 + 2 * CONTINUATION_SEED_OFFSET,
            parent="lineage-small",
        ),
    ],
    "chatty": [Step("chatty", SMALL, CHATTY_TOKENS, mix_heavy("conversations"))],  # the gate's G3
    "large": [Step("large", LARGE, 2 * STEP_TOKENS, BALANCED_MIX)],
}

# Every measure on the card, in the order of its columns; RATES are shares from 0 to 1.
MEASURES: tuple[str, ...] = (
    "heldout",
    "heldout_conversations",
    "yardstick_loss",
    "bench_overall",
    "well_formed",
    "casual",
    "followups",
    "identity_line",
    "identity_no_line",
    "person_life",
    "cut_off",
    "looped",
)
RATES = frozenset(
    {
        "well_formed",
        "casual",
        "followups",
        "identity_line",
        "identity_no_line",
        "person_life",
        "cut_off",
        "looped",
    }
)
LOSSES = frozenset({"heldout", "heldout_conversations", "yardstick_loss"})
LABELS: dict[str, str] = {
    "heldout": "held-out loss",
    "heldout_conversations": "held-out conversation loss",
    "yardstick_loss": "yardstick conversation loss",
    "bench_overall": "bench-v1 overall",
    "well_formed": "well-formed replies",
    "casual": "facts typed casually",
    "followups": "follow-ups answered",
    "identity_line": "knows its name and that it is an AI",
    "identity_no_line": "claims no human life without its name",
    "person_life": "claims a human life in small talk",
    "cut_off": "replies cut off",
    "looped": "replies stopped in a loop",
}
COLUMNS: dict[str, str] = {
    "heldout": "Held-out loss",
    "heldout_conversations": "Held-out conv. loss",
    "yardstick_loss": "Yardstick loss",
    "bench_overall": "Bench",
    "well_formed": "Well-formed",
    "casual": "Casual facts",
    "followups": "Follow-ups",
    "identity_line": "Identity (named)",
    "identity_no_line": "No life claim (unnamed)",
    "person_life": "Life claims",
    "cut_off": "Cut off",
    "looped": "Loops",
}
# How much worse than the baseline a measure may get before compare() reports it (rates as
# shares, bench in points, losses in nats per token).
TOLERANCE: dict[str, float] = {
    "well_formed": 0.04,
    "casual": 0.05,
    "followups": 0.05,
    "bench_overall": 2.0,
    "yardstick_loss": 0.03,
    "heldout": 0.02,
}


class ScorecardSetupError(ValueError):
    """The scorecard cannot start: something it needs is missing, damaged or does not match. The
    message says what, and the command that builds it."""


@dataclass(frozen=True)
class CardScale:
    """How big the recipes and the probes are. :data:`FULL` is the real card; a smaller scale
    (tests) puts ``shapes[s]`` in place of each recipe shape ``s``, ``tokens`` in place of every
    budget and ``batch_tokens`` in every step, and asks fewer probe items."""

    name: str
    shapes: Mapping[ModelShape, ModelShape] | None = None
    tokens: int | None = None
    batch_tokens: int | None = None
    bench_items: int | None = None  # items per benchmark category (None: all)
    relations: int | None = None  # knowledge-base relations the fact probes ask about (None: all)
    casual_per_relation: int = 5
    followups: int = 60
    identity_names: tuple[str, ...] = chatprobes.IDENTITY_NAMES
    small_talk_samples: int = 3

    def probes(self) -> dict[str, Any]:
        """The part of the scale that changes a measurement of a given run."""
        return {
            "bench_items": self.bench_items,
            "relations": self.relations,
            "casual_per_relation": self.casual_per_relation,
            "followups": self.followups,
            "identity_names": list(self.identity_names),
            "small_talk_samples": self.small_talk_samples,
        }


FULL = CardScale("full")


@dataclass
class Instruments:
    """What judges the replies: the judge and novelty index creativity needs (``None``: creativity
    is not measured), the well-formed rule, and a ``label`` that tells these instruments apart in
    the measurement caches."""

    judge: Judge | None
    novelty: NoveltyIndex | None
    well_formed: Callable[[str], bool]
    label: str

    def key(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "judge": asdict(self.judge.calibration) if self.judge is not None else None,
            "novelty": len(self.novelty) if self.novelty is not None else None,
        }


# -- the yardstick ------------------------------------------------------------------------------


def yardstick_loss(model: Transformer, yardstick_root: Path) -> float:
    """The mean loss of ``model`` on held-out conversations of the yardstick root's corpus, taken
    by the trainer's held-out rule (:func:`~airace_ml.train.trainer._heldout_batches`): rows of the
    model's attention span packed from held-out documents by a sampler seeded with 0, each row's
    mean token loss, then the mean over :data:`YARDSTICK_SEQS` rows."""
    corpus = Corpus.open(corpus_dir(Path(yardstick_root)) / YARDSTICK_DATASET)
    pool = DocPool.from_corpus(corpus, heldout_docs(corpus), balanced=False)
    if not len(pool):
        raise ValueError(f"the yardstick root {yardstick_root} holds no held-out conversations")
    sampler = MixtureSampler(
        {YARDSTICK_DATASET: pool}, {YARDSTICK_DATASET: 1.0}, model.shape.ctx_len, 0
    )
    x, y = sampler.next_batch(YARDSTICK_SEQS)
    device = next(model.parameters()).device
    batch = {YARDSTICK_DATASET: (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))}
    return _heldout_losses(model, batch, device)[YARDSTICK_DATASET]


def _identity(root: Path, what: str) -> str:
    try:
        return data_identity(root)
    except (OSError, ValueError) as e:
        raise ScorecardSetupError(
            f"{what} {root} holds no complete training data ({e}); build it with: "
            f"airace-content build --scale full"
        ) from e


def _load_instruments(root: Path, device: torch.device, identity: str) -> Instruments:
    """The yardstick root's judge and novelty index, both built from its data."""
    missing = []
    if not (judge_dir(root) / CALIBRATION_NAME).is_file():
        missing.append(
            f"the reference judge (build it with: airace-ml build-judge --data-root {root})"
        )
    index = novelty_path(root)
    if not (index.is_file() and index.with_suffix(".json").is_file()):
        missing.append(
            f"the novelty index (build it with: airace-ml build-novelty-index --data-root {root})"
        )
    if missing:
        raise ScorecardSetupError(f"the yardstick root {root} lacks {'; '.join(missing)}")
    try:
        judge = _load_judge(root, device, identity)
        novelty = _load_novelty(root, identity)
    except GateSetupError as e:
        raise ScorecardSetupError(str(e)) from e
    return Instruments(
        judge, novelty, lambda reply: is_well_formed(reply, judge), "reference judge"
    )


def _check_recipes(recipes: Sequence[str]) -> list[str]:
    chosen = list(recipes)
    unknown = [r for r in chosen if r not in RECIPES]
    if not chosen or unknown or len(set(chosen)) != len(chosen):
        raise ScorecardSetupError(
            f"recipes must be different names from {', '.join(RECIPES)}; got {', '.join(chosen)}"
        )
    return chosen


def _small_kb(kb: KB, relations: int | None) -> KB:
    if relations is None:
        return kb
    kept = dict(list(kb.relations.items())[:relations])
    return KB(kept, tuple(f for f in kb.facts if f.relation in kept))


# -- measuring ----------------------------------------------------------------------------------


@dataclass
class _Card:
    """One scorecard call's state."""

    data_root: Path
    yardstick_root: Path
    out: Path
    device: torch.device
    scale: CardScale
    tok: Tok
    instruments: Instruments
    yardstick: str  # the yardstick root's data identity
    runs: GateRuns
    on_status: Callable[[str], None]
    trained: dict[str, GateRun] = field(default_factory=dict)
    _suite: Any = None
    _kb: KB | None = None

    def config(self, step: Step) -> TrainRunConfig:
        s = self.scale
        shape = s.shapes[step.shape] if s.shapes is not None else step.shape
        extra = {} if s.batch_tokens is None else {"batch_tokens": s.batch_tokens}
        return TrainRunConfig(
            run_id=step.name,
            seed=step.seed,
            shape=shape,
            token_budget=s.tokens if s.tokens is not None else step.tokens,
            mixture=dict(step.mix),
            **extra,
        )

    def train(self, step: Step) -> GateRun:
        """The trained run of ``step``, its parent trained first; each is trained once a call."""
        if step.name not in self.trained:
            parent = self.train(_STEPS[step.parent]) if step.parent is not None else None
            self.trained[step.name] = self.runs.train(self.config(step), self.device, parent)
        return self.trained[step.name]

    def measured(self, run: GateRun) -> dict[str, Any]:
        """The (cached) measurements of ``run``."""
        key = {
            "run": run.token,
            "code": source_digest(CARD_SOURCES),
            "yardstick": self.yardstick,
            "instruments": self.instruments.key(),
            "probes": self.scale.probes(),
            "suite": SUITE_VERSION,
            "yardstick_seqs": YARDSTICK_SEQS,
            "seed": MEASURE_SEED,
        }
        fresh: list[int] = []

        def compute() -> dict[str, Any]:
            fresh.append(1)
            self.on_status(f"  {run.name}: measuring")
            return self.measure(run)

        value = _cached(self.out / MEASURED_DIR / f"{run.name}.json", key, compute)
        if not fresh:
            self.on_status(f"  {run.name}: measured earlier, reusing it")
        return value

    def measure(self, run: GateRun) -> dict[str, Any]:
        model, _ = load_checkpoint(run.dir, self.device)
        lm = TorchLM(model, self.tok, self.device)
        if self._suite is None:
            self._suite = build_suite()
        if self._kb is None:
            self._kb = _small_kb(load_kb(), self.scale.relations)
        s, kb = self.scale, self._kb
        report = run_benchmarks(
            lm,
            self.tok,
            suite=self._suite,
            judge=self.instruments.judge,
            novelty=self.instruments.novelty,
            max_items_per_category=s.bench_items,
            seed=MEASURE_SEED,
        )
        talk = chatprobes.small_talk(lm, self.instruments.well_formed, samples=s.small_talk_samples)
        casual = chatprobes.casual_facts(
            lm, kb, per_relation=s.casual_per_relation, seed=MEASURE_SEED
        )
        follow = chatprobes.followups(lm, kb, n=s.followups, seed=MEASURE_SEED)
        named = chatprobes.identity(lm, with_line=True, names=s.identity_names)
        unnamed = chatprobes.identity(lm, with_line=False, names=s.identity_names)
        casual_total = sum(score.total for score in casual.values())
        return {
            "measures": {
                "yardstick_loss": yardstick_loss(model, self.yardstick_root),
                "bench_overall": report.overall,
                "well_formed": talk["well_formed"],
                "casual": sum(score.passed for score in casual.values()) / casual_total
                if casual_total
                else 0.0,
                "followups": follow.rate,
                "identity_line": named.rate,
                "identity_no_line": unnamed.rate,
                "person_life": talk["person_life"],
                "cut_off": talk["cut_off"],
                "looped": talk["looped"],
            },
            "bench": {c: score.score for c, score in report.scores.items()},
            "casual_by_style": {style: score.rate for style, score in casual.items()},
            "examples": {
                "followups": follow.examples,
                "identity_line": named.examples,
                "identity_no_line": unnamed.examples,
            },
        }

    def model_entry(self, recipe: str, step: Step, run: GateRun) -> dict[str, Any]:
        record = _read_json(run.dir / GATE_RUN_FILE) or {}
        losses = (record.get("result") or {}).get("heldout_losses") or {}
        measured = self.measured(run)
        cfg = self.config(step)
        measures = {
            "heldout": run.heldout_mean,
            "heldout_conversations": _finite(losses.get(YARDSTICK_DATASET)),
            **measured["measures"],
        }
        return {
            "recipe": recipe,
            "parent": step.parent,
            "shape": [cfg.shape.n_layer, cfg.shape.d_model, cfg.shape.ctx_len],
            "tokens": cfg.token_budget,
            "seed": step.seed,
            "status": run.status,
            "reused": run.reused,
            "measures": {k: measures[k] for k in MEASURES},
            "bench": measured["bench"],
            "casual_by_style": measured["casual_by_style"],
            "examples": measured["examples"],
        }


_STEPS: dict[str, Step] = {step.name: step for recipe in RECIPES.values() for step in recipe}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def run_scorecard(
    data_root: Path,
    out: Path,
    *,
    yardstick_root: Path,
    recipes: Sequence[str],
    device: torch.device | str,
    timing: bool = False,
    on_status: Callable[[str], None] = print,
    scale: CardScale = FULL,
    instruments: Instruments | None = None,
) -> dict[str, Any]:
    """Train ``recipes`` on the data root's text into ``out`` (reusing finished runs) and measure
    every model against the yardstick root (see the module docstring); the card as a dict.

    ``timing`` records the wall-clock seconds of the ``first`` recipe's seed-1 run, marked
    ``reused`` when it was not trained in this call (the seconds are then its own training's).
    ``instruments`` replaces the yardstick root's judge and novelty index (tests). Raises
    :class:`ScorecardSetupError` before training anything if an input is missing or does not
    match."""
    data_root, yardstick_root, out = Path(data_root), Path(yardstick_root), Path(out)
    device = torch.device(device)
    chosen = _check_recipes(recipes)
    identity = _identity(data_root, "the data root")
    same = data_root.resolve() == yardstick_root.resolve()
    yardstick = identity if same else _identity(yardstick_root, "the yardstick root")
    if (
        not same
        and tokenizer_path(data_root).read_bytes() != tokenizer_path(yardstick_root).read_bytes()
    ):
        raise ScorecardSetupError(
            f"the yardstick root {yardstick_root} holds another tokenizer than the data root "
            f"{data_root}; both must use the same one"
        )
    tok = Tok.load(tokenizer_path(data_root))
    if instruments is None:
        instruments = _load_instruments(yardstick_root, device, yardstick)
    on_status(
        f"scorecard: {', '.join(chosen)} on {device.type}; data {data_root}, yardstick "
        f"{yardstick_root}; into {out}"
    )
    card = _Card(
        data_root=data_root,
        yardstick_root=yardstick_root,
        out=out,
        device=device,
        scale=scale,
        tok=tok,
        instruments=instruments,
        yardstick=yardstick,
        runs=GateRuns(
            data_root, out / RUNS_DIR, identity, on_status, source_digest(TRAINING_SOURCES)
        ),
        on_status=on_status,
    )
    models: dict[str, Any] = {}
    for recipe in chosen:
        for step in RECIPES[recipe]:
            models[step.name] = card.model_entry(recipe, step, card.train(step))
    first = card.trained.get("first-s1")
    clock = None
    if timing and first is not None:
        clock = {
            "model": first.name,
            "seconds": first.wall_seconds,
            "device": first.device,
            "reused": first.reused,
        }
    return {
        "scorecard": CARD_VERSION,
        "data_root": str(data_root),
        "yardstick_root": str(yardstick_root),
        "data_identity": identity,
        "yardstick_identity": yardstick,
        "engine": ENGINE_VERSION,
        "date": datetime.datetime.now(datetime.UTC).astimezone().date().isoformat(),
        "device": device.type,
        "scale": scale.name,
        "recipes": {r: [s.name for s in RECIPES[r]] for r in chosen},
        "models": models,
        "timing": clock,
    }


# -- reports ------------------------------------------------------------------------------------


def _shown(key: str, value: object) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    if key in RATES:
        return f"{number:.0%}"
    if key in LOSSES:
        return f"{number:.3f}"
    return f"{number:.1f}"


def _difference(key: str, amount: float) -> str:
    if key in RATES:
        return f"{100 * amount:.0f} points"
    if key in LOSSES:
        return f"{amount:.3f}"
    return f"{amount:.1f} points"


def compare(card: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """One sentence for each measure of :data:`TOLERANCE` on which a model of ``card`` is worse
    than the same model of ``baseline`` by more than its tolerance (a loss going up, anything
    else going down). Models or measures the baseline lacks are not compared."""
    found = []
    before_models = baseline.get("models") or {}
    for name, model in (card.get("models") or {}).items():
        before = (before_models.get(name) or {}).get("measures") or {}
        now = model.get("measures") or {}
        for key, tolerance in TOLERANCE.items():
            new, old = _finite(now.get(key)), _finite(before.get(key))
            if new is None or old is None:
                continue
            worse = new - old if key in LOSSES else old - new
            if worse > tolerance + 1e-9:
                found.append(
                    f"{name}: {LABELS[key]} {_shown(key, new)} against {_shown(key, old)} in the "
                    f"baseline, worse by {_difference(key, worse)} (allowed "
                    f"{_difference(key, tolerance)})."
                )
    return found


def _markdown(card: Mapping[str, Any]) -> str:
    lines = [
        "# Scorecard",
        "",
        (
            f"Data root: {card['data_root']}; yardstick root: {card['yardstick_root']}; "
            f"engine {card['engine']}; {card['date']}"
        ),
        "",
    ]
    if card.get("scale") != FULL.name:
        lines += [f"Scale: {card.get('scale')} (shrunk recipes: a check of the setup only).", ""]
    clock = card.get("timing")
    if clock:
        when = "reused: the time of its own earlier training" if clock["reused"] else "trained now"
        lines += [
            (
                f"First AI ({clock['model']}) trained in {clock['seconds']:.1f} s on "
                f"{clock['device']} ({when})."
            ),
            "",
        ]
    heading = ["Model", "Recipe", *(COLUMNS[key] for key in MEASURES)]
    lines.append("| " + " | ".join(heading) + " |")
    lines.append("| " + " | ".join("---" for _ in heading) + " |")
    for name, model in card["models"].items():
        shown = name if model.get("status") == "completed" else f"{name} (stopped early)"
        cells = [shown, model["recipe"], *(_shown(k, model["measures"].get(k)) for k in MEASURES)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of "
            "replies: well-formed, casual facts, follow-ups and identity are better higher; life "
            "claims, cut-off replies and loops better lower."
        ),
        "",
    ]
    return "\n".join(lines)


def write_reports(card: Mapping[str, Any], out: Path) -> None:
    """``scorecard.json`` (the whole card) and ``scorecard.md`` (one row per model, one column per
    measure) in ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "scorecard.json", card)
    (out / "scorecard.md").write_text(_markdown(card), encoding="utf-8")
