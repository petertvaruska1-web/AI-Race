"""Build the rival pack: train, measure and pre-answer every rival version (game-layer design §9).

``build_rival_pack`` trains each lineage of :data:`~airace_ml.rivals.recipe.LINEAGES` in order,
each version continuing from the one before. Every version is measured like a player's AI
(benchmarks, personality traits and words, the tag breakdown) and answers the first
:data:`REPLIES_PER_GROUP` questions of each customer group's pool; the game later quotes those
answers as the rival's customer posts (R1: real, stored unchanged).

The pack lives under ``out`` (``data/rivals/v1`` by default):

- ``rivals.json``: the index the game server loads, written last so a half-built pack never loads
- ``models/<key>.tar``: each version's checkpoint, packed like a player version's blob
- ``replies/<key>.json``: its answers to the customer questions
- ``measured/<key>.json`` and ``work/``: what a rerun reuses (finished trainings and measurements
  whose recipe, data and training are unchanged)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from airace_ml.data.identity import data_identity
from airace_ml.evals.judge import Judge, check_judge_provenance
from airace_ml.evals.novelty import NoveltyIndex, check_index_provenance
from airace_ml.evals.suite import CATEGORIES, run_benchmarks, score_reply
from airace_ml.experiments.configs import (
    QUICK_BATCH_TOKENS,
    QUICK_BUDGET,
    QUICK_GROWN_SHAPE,
    QUICK_SHAPE,
)
from airace_ml.experiments.runs import GateRun, GateRuns, _cached, _write_json
from airace_ml.feedback import GROUPS, customer_pool
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.paths import judge_dir, novelty_path, tokenizer_path
from airace_ml.personality.fingerprint import Fingerprint, describe, measure_fingerprint
from airace_ml.rivals.recipe import LINEAGES, LineageRecipe, VersionStep, recipe_digest
from airace_ml.tokenizer import Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.worker.blobs import make_tar

PACK_VERSION = "rivals-v1"
REPLIES_PER_GROUP = 24
QUICK_REPLIES_PER_GROUP = 2
QUICK_BENCH_ITEMS = 40  # enough for some tags to reach MIN_TAG_ITEMS, so breakdowns exist
INDEX_NAME = "rivals.json"


@dataclass(frozen=True)
class _Built:
    key: str
    recipe: LineageRecipe
    shape: ModelShape
    trained_tokens: int
    measured: dict[str, Any]


def version_steps(recipe: LineageRecipe, quick: bool) -> tuple[VersionStep, ...]:
    """The recipe's versions, or two tiny ones for a quick build."""
    if quick:
        return (
            VersionStep(QUICK_SHAPE, QUICK_BUDGET),
            VersionStep(QUICK_GROWN_SHAPE, QUICK_BUDGET),
        )
    return recipe.steps


def _creativity_tools(
    root: Path, device: torch.device, identity: str, say: Callable[[str], None]
) -> tuple[Judge | None, NoveltyIndex | None]:
    """The installed judge and novelty index, when both are there and built from this data."""
    try:
        judge = Judge.load(judge_dir(root), device)
        check_judge_provenance(judge_dir(root), identity)
        index = NoveltyIndex.load(novelty_path(root))
        check_index_provenance(index, identity)
    except (OSError, ValueError, TypeError) as e:
        say(
            f"creativity will be 'not measured' ({e}); run build-novelty-index and build-judge "
            "first if rivals should battle in creativity"
        )
        return None, None
    return judge, index


def _reference_traits(root: Path) -> list[dict[str, float]]:
    try:
        catalog = json.loads((root / "catalog" / "v1" / "catalog.json").read_text("utf-8"))
        return [t for t in catalog.get("referenceTraits", []) if isinstance(t, dict)]
    except (OSError, ValueError):
        return []


def _measure(
    run: GateRun,
    tok: Tok,
    device: torch.device,
    judge: Judge | None,
    novelty: NoveltyIndex | None,
    quick: bool,
) -> dict[str, Any]:
    net, _ = load_checkpoint(run.dir, device)
    lm = TorchLM(net, tok, device)
    report = run_benchmarks(
        lm,
        tok,
        judge=judge,
        novelty=novelty,
        max_items_per_category=QUICK_BENCH_ITEMS if quick else None,
    )
    fp = measure_fingerprint(lm, tok, k=1 if quick else 8)
    scores: dict[str, float | None] = {c: None for c in CATEGORIES}
    scores.update({c: s.score for c, s in report.scores.items()})
    per_group = QUICK_REPLIES_PER_GROUP if quick else REPLIES_PER_GROUP
    replies = []
    for group in GROUPS:
        for k, q in enumerate(customer_pool(group)[:per_group]):
            reply = lm.chat_reply([("user", q.prompt)], max_new_tokens=96, temperature=0.8, seed=k)
            replies.append(
                {
                    "group": group,
                    "questionId": q.id,
                    "question": q.prompt,
                    "reply": reply,
                    "score": score_reply(q, reply),
                }
            )
    return {
        "scores": scores,
        "overall": report.overall if report.scores else None,
        "traits": dict(fp.traits),
        "breakdown": {k: [v, n] for k, (v, n) in report.tag_breakdown().items()},
        "replies": replies,
    }


def version_entry(built: _Built, words: list[str]) -> dict[str, Any]:
    """One version as ``rivals.json`` lists it (the protocol's ``RivalPackVersion``)."""
    m = built.measured
    return {
        "key": built.key,
        "nLayer": built.shape.n_layer,
        "dModel": built.shape.d_model,
        "ctxLen": built.shape.ctx_len,
        "trainedTokens": built.trained_tokens,
        "scores": m["scores"],
        "overall": m["overall"],
        "traits": m["traits"],
        "personalityWords": words,
        "breakdown": m["breakdown"],
    }


def pack_document(
    identity: str, quick: bool, built: list[_Built], words: dict[str, list[str]]
) -> dict:
    """The whole ``rivals.json`` (the protocol's ``RivalPack``)."""
    lineages = []
    for recipe in LINEAGES:
        versions = [version_entry(b, words.get(b.key, [])) for b in built if b.recipe is recipe]
        lineages.append(
            {"id": recipe.id, "style": recipe.style, "focus": recipe.focus, "versions": versions}
        )
    return {
        "version": PACK_VERSION,
        "dataIdentity": identity,
        "tokenizer": "tok-v1",
        "suite": "bench-v1",
        "quick": quick,
        "lineages": lineages,
    }


def build_rival_pack(
    data_root: Path,
    out: Path,
    device: torch.device,
    *,
    quick: bool = False,
    on_status: Callable[[str], None] = print,
) -> Path:
    """Train, measure and pre-answer every rival version into ``out``; returns ``rivals.json``."""
    data_root, out = Path(data_root), Path(out)
    identity = data_identity(data_root)
    tok = Tok.load(tokenizer_path(data_root))
    judge, novelty = _creativity_tools(data_root, device, identity, on_status)
    runs = GateRuns(data_root, out / "work", identity, on_status)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "replies").mkdir(parents=True, exist_ok=True)
    built: list[_Built] = []
    for i, recipe in enumerate(LINEAGES):
        parent: GateRun | None = None
        trained = 0
        for k, step in enumerate(version_steps(recipe, quick), start=1):
            key = f"{recipe.id}-v{k}"
            cfg_kwargs = {"batch_tokens": QUICK_BATCH_TOKENS} if quick else {}
            cfg = TrainRunConfig(
                run_id=key,
                seed=i + 1,
                shape=step.shape,
                token_budget=step.tokens,
                mixture=dict(recipe.mix),
                **cfg_kwargs,
            )
            run = runs.train(cfg, device, parent)
            parent = run
            trained += step.tokens
            measure_key = {
                "run": run.token,
                "recipe": recipe_digest(recipe),
                "data": identity,
                "quick": quick,
                "creativity": judge is not None,
            }
            fresh: list[int] = []

            def compute(run: GateRun = run, fresh: list[int] = fresh) -> dict[str, Any]:
                fresh.append(1)
                return _measure(run, tok, device, judge, novelty, quick)

            measured = _cached(out / "measured" / f"{key}.json", measure_key, compute)
            tar = out / "models" / f"{key}.tar"
            if fresh or run.reused is False or not tar.is_file():
                make_tar(run.dir, tar, exclude=("resume",))
            _write_json(out / "replies" / f"{key}.json", measured["replies"])
            on_status(f"reused {key}" if run.reused and not fresh else f"built {key}")
            built.append(_Built(key, recipe, step.shape, trained, measured))
    population = [
        Fingerprint.from_dict({"traits": b.measured["traits"], "samples": []}) for b in built
    ]
    for traits in _reference_traits(data_root):
        try:
            population.append(Fingerprint.from_dict({"traits": traits, "samples": []}))
        except (TypeError, ValueError):
            continue  # traits from another version of the measure
    words = {}
    for n, b in enumerate(built):
        others = population[:n] + population[n + 1 :]
        words[b.key] = describe(population[n], others)
    index = out / INDEX_NAME
    _write_json(index, pack_document(identity, quick, built, words))
    return index
