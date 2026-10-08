"""The feasibility gate (spec 4.12): seven measured criteria that decide whether tiny models make
a game, and the experiments that measure them.

==  ===============  =========================================================================
G1  Speed            the first model (balanced mix, starter shape and budget) meets every
                     first-model target of spec 4.11 on the GPU: it trains in at most 90 s (and,
                     when asked, in at most 8 minutes on the CPU), its full benchmark takes at
                     most 30 s, its first chat token at most 300 ms at 50 tokens/s or more, and
                     growing it at most 2 s
G2  Legibility       at least 70% of that model's replies to 20 probe prompts are well-formed
G3  Differentiation  four contrasting mixes (story, code, fact and conversation heavy) at equal
                     compute each lead their benchmark category by more than twice the seed
                     spread
G4  Seed variation   different seeds of one mix give measurably different personalities, but
                     less different than different mixes do
G5  Growth           growing the first model changes none of its outputs, and the grown model
                     then learns to a lower held-out loss than the ungrown one on the same tokens
G6  Forgetting       a code-only continuation of the story-heavy model loses creativity and
                     language, and a 30% replay of the old mix wins back at least half of it
G7  Preparation      thorough cleaning lowers the garbled-word rate, and fact-checking lowers the
                     false-fact rate, each to at most 80% of the unprepared model's
==  ===============  =========================================================================

The evaluators (``eval_*``) are pure: they turn measurements into a :class:`GateCriterion` and
never raise, whatever they are given. Missing, non-finite or degenerate measurements (no seed
pairs, a single seed, nothing to lower) fail with a detail that says why, and ``data`` holds only
finite numbers (``None`` where nothing could be measured). A criterion that uses a run which
stopped early (unstable) fails and names that run (:func:`require_completed`): its arms did not
get the compute they were meant to.

:func:`run_gate` trains every run under ``out_dir/runs/<name>`` and is safe to stop and start
again (:mod:`airace_ml.experiments.runs`). A run that finished is reused while its config, the
training data, the training code, the device and the exact parent model it continued from are
unchanged; an interrupted run carries on from its last saved point. Benchmark and fingerprint
results are cached beside each run and reused while the run, the measuring code, the judge and
the novelty index are unchanged. So the same command can simply be run again after any change.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import SafetensorError

from airace_ml.data.corpus import (
    DATASET_IDS,
    INFO_FILE,
    OFFSETS_FILE,
    TAGS_FILE,
    TOKENS_FILE,
    Corpus,
)
from airace_ml.data.prep import PrepConfig, heldout_docs
from airace_ml.device import pick_device
from airace_ml.evals.creativity import words
from airace_ml.evals.judge import CALIBRATION_NAME, Judge, is_well_formed
from airace_ml.evals.novelty import NoveltyIndex
from airace_ml.evals.scoring import mean_logprob
from airace_ml.evals.suite import CATEGORIES, SUITE_VERSION, build_suite, run_benchmarks
from airace_ml.experiments.configs import (
    BALANCED_MIX,
    FULL_SCALE,
    QUICK_SCALE,
    TARGET_CATEGORY,
    GateScale,
    mix_heavy,
)
from airace_ml.experiments.runs import (
    MEASURING_SOURCES,
    TRAINING_SOURCES,
    GateRun,
    GateRuns,
    _cached,
    _plain,
    source_digest,
)
from airace_ml.infer.lm import LanguageModel, TorchLM
from airace_ml.model.checkpoint import load_checkpoint
from airace_ml.model.growth import grow
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir, judge_dir, novelty_path, tokenizer_path
from airace_ml.personality.fingerprint import (
    Fingerprint,
    describe,
    fingerprint_distance,
    load_probes,
    measure_fingerprint,
)
from airace_ml.skills.facts import FalseFactPlan
from airace_ml.skills.kb import KB, load_kb
from airace_ml.tokenizer import Tok, encode_chat, encode_doc
from airace_ml.train.config import TrainRunConfig

TITLES: dict[str, str] = {
    "G1": "Speed",
    "G2": "Legibility",
    "G3": "Differentiation",
    "G4": "Seed variation",
    "G5": "Growth",
    "G6": "Forgetting",
    "G7": "Preparation",
}

GPU_SECONDS_LIMIT = 90.0
CPU_SECONDS_LIMIT = 480.0
NO_CUDA_DETAIL = "no CUDA device; GPU speed target not measured"
LEGIBILITY_PROBES = 20
LEGIBILITY_MIN_RATE = 0.7
SPREAD_FACTOR = 2.0  # a mix must lead its category by more than this many seed spreads
SEED_MIN_DISTANCE = 0.05
GROWTH_MAX_LOGIT_DIFF = 1e-4
FORGETTING_CATEGORIES: tuple[str, ...] = ("creativity", "language")
FORGETTING_MIN_DROP = 3.0
FORGETTING_MIN_RECOVERY = 0.5
PREP_MAX_RATIO = 0.8

REPLAY_SHARE = 0.3
WEB_HEAVY_SHARE = 0.7
FACTS_MIX: dict[str, float] = {"web": 0.5, "facts": 0.5}
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_P = 0.95
LEGIBILITY_REPLY_TOKENS = 64
GARBLE_SAMPLES = 60
GARBLE_PREFIX_TOKENS = 32
GARBLE_NEW_TOKENS = 48
GROWTH_SEQUENCES = 4
MEASURE_SEED = 0  # benchmarks, fingerprints and the choice of held-out prompts
G3_TRANSCRIPT_PROBES: tuple[str, ...] = ("creative-10", "factual-01")

RUNS_DIR = "runs"
BENCH_FILE = "gate_bench.json"
FINGERPRINT_FILE = "gate_fingerprint.json"
KNOWN_VOCAB_FILE = "known_vocab.txt"
FALSE_FACTS_FILE = "false_facts.json"
# A file that cannot be read or does not parse. Only these blame (and ask to rebuild) a file;
# anything else (a CUDA error, a bug) propagates.
_LOAD_ERRORS = (OSError, ValueError, EOFError, SafetensorError)


class GateSetupError(ValueError):
    """The gate cannot start: something it needs is missing or damaged. The message says what,
    and the command that builds it."""


@dataclass
class GateCriterion:
    id: str
    title: str
    passed: bool
    detail: str
    data: dict


@dataclass
class GateOutcome:
    criteria: list[GateCriterion]
    transcripts: dict[str, list[tuple[str, str]]]
    raw: dict

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)


# -- evaluators (pure) ----------------------------------------------------------------------------


def _finite(value: object) -> float | None:
    """``value`` as a float if it is a finite number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _criterion(gid: str, passed: bool, detail: str, data: dict) -> GateCriterion:
    return GateCriterion(gid, TITLES[gid], passed, detail, data)


@dataclass(frozen=True)
class SpeedTarget:
    """One first-model target of spec 4.11: ``key`` names the measurement, which must be at most
    ``limit`` (at least ``limit`` when ``at_least``)."""

    key: str
    label: str
    unit: str
    limit: float
    digits: int
    at_least: bool = False

    def show(self, value: float) -> str:
        return f"{self.label} {value:.{self.digits}f} {self.unit}"

    def met(self, value: float) -> bool:
        return value >= self.limit if self.at_least else value <= self.limit


SPEED_TARGETS: tuple[SpeedTarget, ...] = (
    SpeedTarget("gpu_seconds", "trained on the GPU in", "s", GPU_SECONDS_LIMIT, 1),
    SpeedTarget("cpu_seconds", "trained on the CPU in", "s", CPU_SECONDS_LIMIT, 1),
    SpeedTarget("bench_seconds", "full benchmark suite", "s", 30.0, 1),
    SpeedTarget("first_token_ms", "chat first token", "ms", 300.0, 0),
    SpeedTarget("tokens_per_second", "chat throughput", "tokens/s", 50.0, 0, at_least=True),
    SpeedTarget("growth_seconds", "growth op", "s", 2.0, 2),
)


def _speed_data(values: Mapping[str, float | None]) -> dict:
    data: dict = {}
    for target in SPEED_TARGETS:
        given = values.get(target.key)
        data[target.key] = None if given is None else _finite(given)
        data[f"{target.key}_target"] = target.limit
    return data


def eval_speed(
    gpu_seconds: float,
    cpu_seconds: float | None,
    *,
    bench_seconds: float | None = None,
    first_token_ms: float | None = None,
    tokens_per_second: float | None = None,
    growth_seconds: float | None = None,
) -> GateCriterion:
    """G1: every first-model target of spec 4.11 that was measured is met (:data:`SPEED_TARGETS`):
    training in at most 90 s on the GPU (and 480 s on the CPU, if timed), the full benchmark in
    at most 30 s, a first chat token in at most 300 ms at 50 tokens/s or more, and growth in at
    most 2 s. A measurement not given is not checked; the GPU training time is required."""
    values = {
        "gpu_seconds": gpu_seconds,
        "cpu_seconds": cpu_seconds,
        "bench_seconds": bench_seconds,
        "first_token_ms": first_token_ms,
        "tokens_per_second": tokens_per_second,
        "growth_seconds": growth_seconds,
    }
    data = _speed_data(values)
    broken = [
        t.label
        for t in SPEED_TARGETS
        if (values[t.key] is not None or t.key == "gpu_seconds") and data[t.key] is None
    ]
    if broken:
        detail = f"{', '.join(broken)}: a measurement is not a finite number"
        return _criterion("G1", False, detail, data)
    parts, missed = [], []
    for t in SPEED_TARGETS:
        value = data[t.key]
        if value is None:
            continue
        bound = "at least" if t.at_least else "at most"
        parts.append(f"{t.show(value)} ({bound} {t.limit:g} {t.unit})")
        if not t.met(value):
            missed.append(t.label)
    if data["cpu_seconds"] is None:
        parts.append("CPU not timed")
    detail = "; ".join(parts)
    if missed:
        detail = f"missed: {', '.join(missed)}. {detail}"
    return _criterion("G1", not missed, detail, data)


def speed_criterion(
    device_type: str, seconds: float, cpu_seconds: float | None, **measurements: float | None
) -> GateCriterion:
    """G1 for a first model timed on ``device_type``, with the other first-model measurements of
    :func:`eval_speed` as keywords: :func:`eval_speed` when that is CUDA. Any other device fails,
    reporting every measurement: a CPU time is never passed off as a GPU time."""
    if device_type == "cuda":
        return eval_speed(seconds, cpu_seconds, **measurements)
    measured = _finite(seconds)
    data = _speed_data({**measurements, "cpu_seconds": cpu_seconds})
    data.update({"measured_seconds": measured, "device": device_type})
    took = "an unmeasurable time" if measured is None else f"{measured:.1f} s"
    shown = [f"the first model trained in {took} on the {device_type.upper()}"]
    for t in SPEED_TARGETS[2:]:
        if data[t.key] is not None:
            shown.append(t.show(data[t.key]))
    detail = f"{NO_CUDA_DETAIL} ({'; '.join(shown)})"
    if data["cpu_seconds"] is not None:
        detail += f"; CPU {data['cpu_seconds']:.1f} s (at most {CPU_SECONDS_LIMIT:g} s)"
    return _criterion("G1", False, detail, data)


def require_completed(criterion: GateCriterion, runs: Sequence[GateRun]) -> GateCriterion:
    """``criterion`` unchanged if every run it used completed; otherwise a failed copy whose detail
    first names each run that stopped early and how far it got (its arms did not get the
    compute they were meant to, so the comparison does not hold)."""
    unfinished = list({r.name: r for r in runs if r.status != "completed"}.values())
    if not unfinished:
        return criterion
    named = "; ".join(
        f"{r.name} stopped early ({r.status} after {r.steps} of {r.planned_steps} steps)"
        for r in unfinished
    )
    return replace(
        criterion,
        passed=False,
        detail=f"{named}, so this cannot pass. {criterion.detail}",
        data={**criterion.data, "unfinished_runs": [r.name for r in unfinished]},
    )


def eval_legibility(flags: Sequence[bool]) -> GateCriterion:
    """G2: exactly 20 replies were judged, and at least 70% of them are well-formed."""
    n = len(flags)
    good = sum(bool(flag) for flag in flags)
    rate = good / n if n else 0.0
    data = {"well_formed": good, "replies": n, "rate": rate, "min_rate": LEGIBILITY_MIN_RATE}
    detail = f"{good} of {n} replies well-formed ({rate:.0%}; needs {LEGIBILITY_MIN_RATE:.0%})"
    if n != LEGIBILITY_PROBES:
        return _criterion("G2", False, f"needs {LEGIBILITY_PROBES} replies, got {n}", data)
    return _criterion("G2", rate >= LEGIBILITY_MIN_RATE, detail, data)


def _spread(values: Sequence[float]) -> float:
    """The seed standard deviation (sample, n - 1); 0 for fewer than two seeds."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def eval_differentiation(scores: Mapping[str, Sequence[Mapping[str, float]]]) -> GateCriterion:
    """G3: every mix ``t`` of :data:`TARGET_CATEGORY` leads its category ``k`` over the best rival
    mix ``u*``: ``mean(t, k) > mean(u*, k) + 2 * max(std(t, k), std(u*, k))``.

    ``scores[t]`` holds one ``{category: score}`` per seed. The spread is the sample standard
    deviation over seeds, so fewer than two seeds cannot show a margin and fail.
    """
    problems: list[str] = []
    stats: dict[tuple[str, str], tuple[float, float, int]] = {}
    for t in TARGET_CATEGORY:
        runs = list(scores.get(t, ()))
        if not runs:
            problems.append(f"no {t}-heavy runs")
        for k in TARGET_CATEGORY.values():
            values = [_finite(run.get(k)) for run in runs]
            kept = [v for v in values if v is not None]
            if len(kept) < len(values):
                problems.append(f"a {t}-heavy run has no {k} score")
            stats[(t, k)] = (_mean(kept), _spread(kept), len(kept))
    seeds = min(len(scores.get(t, ())) for t in TARGET_CATEGORY)
    if 0 < seeds < 2:
        problems.append("one seed per mix cannot show a margin over the seed spread")
    per_mix: dict[str, dict] = {}
    for t, k in TARGET_CATEGORY.items():
        mean, std, n = stats[(t, k)]
        rival = max((u for u in TARGET_CATEGORY if u != t), key=lambda u: stats[(u, k)][0])
        rival_mean, rival_std, _ = stats[(rival, k)]
        needed = SPREAD_FACTOR * max(std, rival_std)
        per_mix[t] = {
            "category": k,
            "mean": mean,
            "std": std,
            "seeds": n,
            "rival": rival,
            "rival_mean": rival_mean,
            "rival_std": rival_std,
            "lead": mean - rival_mean,
            "needed_lead": needed,
            "leads": mean > rival_mean + needed,
        }
    lines = [
        f"{t}-heavy {m['category']} {m['mean']:.1f} vs {m['rival']}-heavy {m['rival_mean']:.1f} "
        f"(lead {m['lead']:+.1f}, needs more than {m['needed_lead']:.1f})"
        for t, m in per_mix.items()
    ]
    detail = "; ".join(lines)
    if problems:
        detail = "; ".join(dict.fromkeys(problems)) + ". " + detail
    passed = not problems and all(m["leads"] for m in per_mix.values())
    return _criterion("G3", passed, detail, per_mix)


def eval_seed_variation(seed_dists: Sequence[float], mix_dists: Sequence[float]) -> GateCriterion:
    """G4: the mean fingerprint distance between seeds of one mix is above 0.05 and below the
    mean distance between different mixes."""
    seed = [v for v in map(_finite, seed_dists) if v is not None]
    mix = [v for v in map(_finite, mix_dists) if v is not None]
    problems = []
    if not seed_dists:
        problems.append("no pairs of seeds to compare (needs at least 2 seeds)")
    if not mix_dists:
        problems.append("no pairs of mixes to compare")
    if len(seed) < len(seed_dists) or len(mix) < len(mix_dists):
        problems.append("a distance is not a finite number")
    seed_mean, mix_mean = _mean(seed), _mean(mix)
    data = {
        "seed_mean": seed_mean,
        "mix_mean": mix_mean,
        "min_seed_mean": SEED_MIN_DISTANCE,
        "seed_distances": seed,
        "mix_distances": mix,
    }
    detail = (
        f"same mix, different seeds: mean distance {seed_mean:.3f} (needs more than "
        f"{SEED_MIN_DISTANCE:g}); different mixes: {mix_mean:.3f} (seeds must differ less)"
    )
    if problems:
        detail = "; ".join(problems) + ". " + detail
    passed = not problems and seed_mean > SEED_MIN_DISTANCE and seed_mean < mix_mean
    return _criterion("G4", passed, detail, data)


def eval_growth(max_logit_diff: float, grown_loss: float, ungrown_loss: float) -> GateCriterion:
    """G5: growth moves no output logit by more than 1e-4, and the grown model ends with the lower
    held-out loss."""
    diff, grown, ungrown = _finite(max_logit_diff), _finite(grown_loss), _finite(ungrown_loss)
    data = {
        "max_logit_diff": diff,
        "max_allowed": GROWTH_MAX_LOGIT_DIFF,
        "grown_loss": grown,
        "ungrown_loss": ungrown,
    }
    if diff is None or grown is None or ungrown is None:
        return _criterion("G5", False, "a growth measurement is not a finite number", data)
    detail = (
        f"growing moved no output by more than {diff:.2g} (allowed {GROWTH_MAX_LOGIT_DIFF:g}); "
        f"after the same extra training the held-out loss is {grown:.3f} grown vs {ungrown:.3f} "
        f"not grown"
    )
    passed = diff <= GROWTH_MAX_LOGIT_DIFF and grown < ungrown
    return _criterion("G5", passed, detail, data)


def eval_forgetting(
    base: Mapping[str, float], no_replay: Mapping[str, float], replay: Mapping[str, float]
) -> GateCriterion:
    """G6: summed over creativity and language, code-only training drops the story model's score
    by at least 3 points, and 30% replay wins back at least half of that drop."""
    problems = []

    def scores(model: Mapping[str, float], name: str) -> dict[str, float | None]:
        found = {c: _finite(model.get(c)) for c in FORGETTING_CATEGORIES}
        missing = [c for c, v in found.items() if v is None]
        if missing:
            problems.append(f"the {name} model has no {' or '.join(missing)} score")
        return found

    found = {
        "base": scores(base, "story"),
        "no_replay": scores(no_replay, "code-only"),
        "replay": scores(replay, "replay"),
    }
    total = {name: math.fsum(v or 0.0 for v in s.values()) for name, s in found.items()}
    drop = total["base"] - total["no_replay"]
    recovered = total["replay"] - total["no_replay"]
    recovery = _finite(recovered / drop) if drop > 0 else None
    data = {
        **found,
        "drop": drop,
        "recovered": recovered,
        "recovery": recovery or 0.0,
        "min_drop": FORGETTING_MIN_DROP,
        "min_recovery": FORGETTING_MIN_RECOVERY,
    }
    detail = (
        f"creativity + language {total['base']:.1f} for the story-heavy model, "
        f"{total['no_replay']:.1f} after code-only training (a drop of {drop:.1f}; needs "
        f"{FORGETTING_MIN_DROP:g} or more), {total['replay']:.1f} with {REPLAY_SHARE:.0%} replay"
    )
    if recovery is None:
        detail += " (no drop to win back)"
    else:
        detail += f" (won back {recovery:.0%} of the drop; needs {FORGETTING_MIN_RECOVERY:.0%})"
    if problems:
        detail = "; ".join(problems) + ". " + detail
    passed = (
        not problems
        and drop >= FORGETTING_MIN_DROP
        and recovery is not None
        and recovery >= FORGETTING_MIN_RECOVERY
    )
    return _criterion("G6", passed, detail, data)


def eval_prep(
    garble_dirty: float, garble_clean: float, false_unchecked: float, false_checked: float
) -> GateCriterion:
    """G7: thorough cleaning brings the garbled-word rate to at most 80% of light cleaning's, and
    fact-checking brings the false-fact rate to at most 80% of no checking's. A rate of 0 without
    the preparation leaves nothing to lower, so nothing is shown and the criterion fails."""
    dirty, clean = _finite(garble_dirty), _finite(garble_clean)
    unchecked, checked = _finite(false_unchecked), _finite(false_checked)
    data = {
        "garble_light": dirty,
        "garble_thorough": clean,
        "false_unchecked": unchecked,
        "false_checked": checked,
        "max_ratio": PREP_MAX_RATIO,
    }
    if None in (dirty, clean, unchecked, checked):
        return _criterion("G7", False, "a preparation measurement is not a finite number", data)
    problems = []
    if dirty <= 0:
        problems.append("the lightly cleaned model wrote no garbled word, so nothing could fall")
    if unchecked <= 0:
        problems.append("the unchecked model preferred no false fact, so nothing could fall")
    detail = (
        f"garbled words {clean:.1%} after thorough cleaning vs {dirty:.1%} after light cleaning; "
        f"false facts preferred {checked:.1%} with fact-checking vs {unchecked:.1%} without "
        f"(each must fall to {PREP_MAX_RATIO:.0%} or less)"
    )
    if problems:
        detail = "; ".join(problems) + ". " + detail
    passed = (
        not problems and clean <= PREP_MAX_RATIO * dirty and checked <= PREP_MAX_RATIO * unchecked
    )
    return _criterion("G7", passed, detail, data)


# -- measurements -------------------------------------------------------------------------------


def garble_rate(texts: Sequence[str], known_vocab: set[str]) -> float:
    """The fraction of the words of ``texts`` (all pooled) that are not in ``known_vocab``; 0.0
    when there are no words.

    Words are counted as everywhere a model's text is (:func:`~airace_ml.evals.creativity.words`:
    lowercase runs of letters). The vocabulary is read with the same rule, so an entry with an
    apostrophe (``don't``) vouches for both of its parts.
    """
    known = {w for entry in known_vocab for w in words(entry)}
    said = [w for text in texts for w in words(text)]
    return sum(w not in known for w in said) / len(said) if said else 0.0


def _object_context(template: str, subject: str) -> str | None:
    """``template`` with its subject filled in, cut just before ``{o}`` and right-stripped; None
    when the subject does not come before ``{o}`` (the cut would not ask about it)."""
    cut = template.find("{o}")
    if cut < 0 or "{s}" not in template[:cut]:
        return None
    context = template[:cut].replace("{s}", subject).rstrip()
    return context or None


def false_fact_probes(kb: KB, plan: FalseFactPlan) -> tuple[list[tuple[str, str, str]], int]:
    """``(context, true object, false object)`` for each pair of ``plan`` that can be asked, and
    how many pairs were skipped.

    The context is the relation's first training template, cut at ``{o}``. A pair is skipped
    when its relation or subject is not in ``kb``, its template does not put the subject before
    ``{o}``, or its false object is the true one.
    """
    probes: list[tuple[str, str, str]] = []
    skipped = 0
    for (subject, name), false_obj in plan.mapping.items():
        relation = kb.relations.get(name)
        try:
            true_obj = kb.true_object(subject, name)
        except KeyError:
            true_obj = None
        context = _object_context(relation.train_templates[0], subject) if relation else None
        if context is None or true_obj is None or true_obj == false_obj:
            skipped += 1
            continue
        probes.append((context, true_obj, false_obj))
    return probes, skipped


def false_fact_rate(lm: LanguageModel, tok: Tok, kb: KB, plan: FalseFactPlan) -> float:
    """The fraction of the askable pairs of ``plan`` (:func:`false_fact_probes`) on which ``lm``
    prefers the false object.

    Each object is scored as a continuation (with its leading space) of the plain-document
    context, and the false one is preferred when its mean log-probability per token is strictly
    higher, as multiple-choice benchmarks decide. 0.0 when no pair can be asked.
    """
    probes, _ = false_fact_probes(kb, plan)
    if not probes:
        return 0.0
    contexts, continuations = [], []
    for context, true_obj, false_obj in probes:
        ids = encode_doc(tok, context)
        contexts += [ids, ids]
        continuations += [tok.encode(" " + true_obj), tok.encode(" " + false_obj)]
    scores = lm.score_continuations(contexts, continuations)
    preferred = sum(
        mean_logprob(scores[2 * i + 1]) > mean_logprob(scores[2 * i]) for i in range(len(probes))
    )
    return preferred / len(probes)


CHAT_SPEED_PROMPT = "Hello! How are you today?"
CHAT_SPEED_TOKENS = 64


def _clock(device: object) -> float:
    """``time.perf_counter()``, once ``device`` (if it is a CUDA device) has finished its queued
    work, so GPU work is timed when it is done rather than when it was queued."""
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def chat_speed(lm: LanguageModel, tok: Tok) -> tuple[float, float]:
    """The first-token latency (ms) and the throughput (tokens/s) of a chat reply on ``lm``'s
    device: a one-token reply is generated once to warm up and once timed, then a
    :data:`CHAT_SPEED_TOKENS`-token reply that never stops early (an attention span too short
    for all of them caps it) is timed and its tokens counted."""
    prompt = encode_chat(tok, [("user", CHAT_SPEED_PROMPT)], add_generation_prompt=True)
    options = {"temperature": SAMPLE_TEMPERATURE, "top_p": SAMPLE_TOP_P, "seed": MEASURE_SEED}
    device = getattr(lm, "device", None)
    lm.generate([prompt], max_new_tokens=1, **options)
    started = _clock(device)
    lm.generate([prompt], max_new_tokens=1, **options)
    first = _clock(device) - started
    started = _clock(device)
    reply = lm.generate([prompt], max_new_tokens=CHAT_SPEED_TOKENS, stop_ids=(), **options)[0]
    elapsed = max(_clock(device) - started, 1e-9)
    return 1000 * first, len(reply.tokens) / elapsed


@torch.inference_mode()
def max_logit_diff(a: Transformer, b: Transformer, sequences: Sequence[Sequence[int]]) -> float:
    """The largest absolute difference between the two models' logits over ``sequences`` (both in
    eval mode, full precision); ``inf`` if either gives a non-finite logit or there is no sequence
    to compare on (nothing measured is nothing shown)."""
    if not sequences:
        return math.inf
    modes = (a.training, b.training)
    a.eval()
    b.eval()
    device = next(a.parameters()).device
    worst = 0.0
    try:
        for seq in sequences:
            x = torch.as_tensor(list(seq), dtype=torch.long, device=device)[None]
            diff = (a(x).float() - b(x).float()).abs().max().item()
            if not math.isfinite(diff):
                return math.inf
            worst = max(worst, diff)
    finally:
        a.train(modes[0])
        b.train(modes[1])
    return worst


# -- inputs -------------------------------------------------------------------------------------


@dataclass
class _Inputs:
    tok: Tok
    judge: Judge
    novelty: NoveltyIndex
    known_vocab: set[str]
    plan: FalseFactPlan
    kb: KB
    identity: str  # a digest of the tokenizer and the corpora
    web_prefixes: list[str]
    growth_sequences: list[list[int]]


def _missing_inputs(root: Path) -> list[str]:
    corpus = corpus_dir(root)
    missing = []
    built = tokenizer_path(root).is_file() and all(
        (corpus / ds / name).is_file()
        for ds in DATASET_IDS
        for name in (INFO_FILE, OFFSETS_FILE, TAGS_FILE, TOKENS_FILE)
    )
    shared = all((corpus / name).is_file() for name in (KNOWN_VOCAB_FILE, FALSE_FACTS_FILE))
    if not (built and shared):
        missing.append(
            f"the training data under {root} (build it with: airace-content build --scale full)"
        )
    if not (judge_dir(root) / CALIBRATION_NAME).is_file():
        missing.append("the reference judge (build it with: airace-ml build-judge)")
    index = novelty_path(root)
    if not (index.is_file() and index.with_suffix(".json").is_file()):
        missing.append("the novelty index (build it with: airace-ml build-novelty-index)")
    return missing


def _data_identity(root: Path) -> str:
    """A digest of the tokenizer and every corpus's documents and tags, so runs trained on other
    data are never reused."""
    digest = hashlib.sha256()
    paths = [tokenizer_path(root)]
    for ds in DATASET_IDS:
        folder = corpus_dir(root) / ds
        paths += [folder / INFO_FILE, folder / OFFSETS_FILE, folder / TAGS_FILE]
        digest.update(f"{ds}/{TOKENS_FILE}:{(folder / TOKENS_FILE).stat().st_size};".encode())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def trim_to_word(text: str) -> str:
    """``text`` cut back to its last whitespace (dropping the whitespace), so a prompt never ends
    inside a word. Text with no whitespace after its first non-space character is kept whole."""
    for i in range(len(text) - 1, -1, -1):
        if text[i].isspace():
            head = text[:i].rstrip()
            return head if head else text
    return text


def _web_prefixes(root: Path, tok: Tok) -> list[str]:
    """:data:`GARBLE_SAMPLES` prompts: the first :data:`GARBLE_PREFIX_TOKENS` tokens (after
    ``<|bos|>``) of held-out web documents in a seeded order, as text cut back to a whole word
    (:func:`trim_to_word`). With fewer held-out documents than samples, the documents are used
    again in the same order."""
    corpus = Corpus.open(corpus_dir(root) / "web")
    order = np.random.default_rng(MEASURE_SEED).permutation(heldout_docs(corpus))
    texts = []
    for i in order[:GARBLE_SAMPLES]:
        doc = corpus.doc(int(i))
        start = 1 if len(doc) and doc[0] == tok.bos_id else 0
        prefix = tok.decode(doc[start : start + GARBLE_PREFIX_TOKENS].tolist())
        texts.append(trim_to_word(prefix))
    return [texts[k % len(texts)] for k in range(GARBLE_SAMPLES)] if texts else []


def _growth_sequences(root: Path, n: int, length: int) -> list[list[int]]:
    """``n`` held-out sequences of at most ``length`` tokens, one dataset at a time in
    :data:`DATASET_IDS` order (each dataset's first held-out documents first)."""
    pools = []
    for ds in DATASET_IDS:
        corpus = Corpus.open(corpus_dir(root) / ds)
        docs = (corpus.doc(int(i))[:length].tolist() for i in heldout_docs(corpus))
        pools.append([d for d in docs if len(d) >= 2][:n])
    picked: list[list[int]] = []
    for depth in range(n):
        picked += [pool[depth] for pool in pools if depth < len(pool)]
    return picked[:n]


def _load_judge(root: Path, device: torch.device) -> Judge:
    """The reference judge; a judge whose files cannot be read or parsed is a
    :class:`GateSetupError` naming the command that rebuilds it."""
    try:
        return Judge.load(judge_dir(root), device)
    except _LOAD_ERRORS as e:
        raise GateSetupError(
            f"the reference judge in {judge_dir(root)} is damaged or out of date ({e}). "
            f"Rebuild it with: airace-ml build-judge"
        ) from e


def _load_novelty(root: Path) -> NoveltyIndex:
    """The novelty index; one whose files cannot be read or parsed is a :class:`GateSetupError`
    naming the command that rebuilds it."""
    try:
        return NoveltyIndex.load(novelty_path(root))
    except _LOAD_ERRORS as e:
        raise GateSetupError(
            f"the novelty index at {novelty_path(root)} is damaged or out of date ({e}). "
            f"Rebuild it with: airace-ml build-novelty-index"
        ) from e


def _load_inputs(root: Path, device: torch.device, length: int) -> _Inputs:
    """Everything the gate reads besides its own runs, checked before any training starts."""
    missing = _missing_inputs(root)
    if missing:
        raise GateSetupError(f"the gate cannot start; still to build: {'; '.join(missing)}")
    data_hint = f"rebuild the training data under {root} with: airace-content build"
    try:
        tok = Tok.load(tokenizer_path(root))
        identity = _data_identity(root)
        known_vocab = set((corpus_dir(root) / KNOWN_VOCAB_FILE).read_text(encoding="utf-8").split())
        plan = FalseFactPlan.from_json(
            (corpus_dir(root) / FALSE_FACTS_FILE).read_text(encoding="utf-8")
        )
        prefixes = _web_prefixes(root, tok)
        sequences = _growth_sequences(root, GROWTH_SEQUENCES, length)
    except _LOAD_ERRORS as e:
        raise GateSetupError(f"the training data cannot be read ({e}); {data_hint}") from e
    if not prefixes or len(sequences) < GROWTH_SEQUENCES:
        raise GateSetupError(f"the corpora hold too few held-out documents to measure; {data_hint}")
    judge = _load_judge(root, device)
    novelty = _load_novelty(root)
    return _Inputs(tok, judge, novelty, known_vocab, plan, load_kb(), identity, prefixes, sequences)


# -- the gate -----------------------------------------------------------------------------------


@dataclass
class _Gate:
    """One gate run's state. Each experiment returns its criterion and the runs it used."""

    inputs: _Inputs
    runs: GateRuns
    scale: GateScale
    seeds: tuple[int, ...]
    device: torch.device
    measuring_code: str
    include_cpu_speed: bool
    on_progress: Callable[[str], None]
    raw: dict = field(default_factory=dict)
    transcripts: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    starter_run: GateRun | None = None
    g3_runs: dict[str, list[GateRun]] = field(default_factory=dict)
    growth_diff: float = math.inf
    growth_seconds: float | None = None
    _loaded: tuple[str, TorchLM] | None = None
    _suite: Any = None

    # -- helpers ----------------------------------------------------------------------------

    def config(
        self,
        run_id: str,
        *,
        shape: ModelShape,
        budget: int,
        mixture: Mapping[str, float],
        seed: int,
        prep: PrepConfig | None = None,
        replay: float = 0.0,
    ) -> TrainRunConfig:
        extra = {} if self.scale.batch_tokens is None else {"batch_tokens": self.scale.batch_tokens}
        return TrainRunConfig(
            run_id=run_id,
            seed=seed,
            shape=shape,
            token_budget=budget,
            mixture=dict(mixture),
            replay=replay,
            prep=prep if prep is not None else PrepConfig(),
            **extra,
        )

    def train(
        self,
        cfg: TrainRunConfig,
        parent: GateRun | None = None,
        device: torch.device | None = None,
    ) -> GateRun:
        run = self.runs.train(cfg, device if device is not None else self.device, parent)
        self.raw.setdefault("runs", {})[run.name] = {
            "status": run.status,
            "device": run.device,
            "steps": run.steps,
            "planned_steps": run.planned_steps,
            "wall_seconds": run.wall_seconds,
            "heldout_loss": run.heldout_mean,
            "reused": run.reused,
        }
        return run

    def lm(self, run: GateRun) -> TorchLM:
        if self._loaded is None or self._loaded[0] != run.token:
            model, _ = load_checkpoint(run.dir, self.device)
            self._loaded = (run.token, TorchLM(model, self.inputs.tok, self.device))
        return self._loaded[1]

    def bench_report(self, run: GateRun, categories: Sequence[str] = CATEGORIES) -> dict:
        """The (cached) benchmark report of ``run``: ``BenchReport.to_dict()``, whose ``seconds``
        time the benchmark alone (the suite is built beforehand)."""
        judge, novelty = self.inputs.judge, self.inputs.novelty
        key = {
            "run": run.token,
            "code": self.measuring_code,
            "suite": SUITE_VERSION,
            "categories": list(categories),
            "seed": MEASURE_SEED,
            "judge": asdict(judge.calibration),
            "novelty": len(novelty),
        }

        def compute() -> dict:
            if self._suite is None:
                self._suite = build_suite()
            self.on_progress(f"  {run.name}: benchmarks ({', '.join(categories)})")
            report = run_benchmarks(
                self.lm(run),
                self.inputs.tok,
                suite=self._suite,
                judge=judge,
                novelty=novelty,
                categories=categories,
                seed=MEASURE_SEED,
            )
            return report.to_dict()

        report = _cached(run.dir / BENCH_FILE, key, compute)
        self.raw.setdefault("bench", {})[run.name] = {
            c: s["score"] for c, s in report["scores"].items()
        }
        return report

    def bench(self, run: GateRun, categories: Sequence[str] = CATEGORIES) -> dict[str, float]:
        report = self.bench_report(run, categories)
        return {c: s["score"] for c, s in report["scores"].items()}

    def fingerprint(self, run: GateRun) -> Fingerprint:
        key = {
            "run": run.token,
            "code": self.measuring_code,
            "k": self.scale.fingerprint_k,
            "seed": MEASURE_SEED,
        }

        def compute() -> dict:
            self.on_progress(f"  {run.name}: personality fingerprint")
            fp = measure_fingerprint(
                self.lm(run), self.inputs.tok, k=self.scale.fingerprint_k, seed=MEASURE_SEED
            )
            return fp.to_dict()

        fp = Fingerprint.from_dict(_cached(run.dir / FINGERPRINT_FILE, key, compute))
        self.raw.setdefault("fingerprints", {})[run.name] = dict(fp.traits)
        return fp

    def announce(self, gid: str, what: str) -> None:
        self.on_progress(f"{gid} {TITLES[gid]}: {what}")

    # -- experiments ------------------------------------------------------------------------

    def speed(self) -> tuple[GateCriterion, list[GateRun]]:
        """G1: train the first model, then time everything spec 4.11 sets a first-model target
        for: training, the full benchmark, chat, and growing it (kept for G5)."""
        self.announce("G1", "training the first model (balanced mix)")
        s = self.scale
        cfg = self.config(
            "starter",
            shape=s.starter_shape,
            budget=s.starter_budget,
            mixture=BALANCED_MIX,
            seed=self.seeds[0],
        )
        run = self.starter_run = self.train(cfg)
        used = [run]
        cpu_seconds = None
        if self.include_cpu_speed:
            if self.device.type == "cpu":
                cpu_seconds = run.wall_seconds
            else:
                self.on_progress("  timing the same run on the CPU")
                cpu_run = self.train(replace(cfg, run_id="starter-cpu"), device=torch.device("cpu"))
                cpu_seconds = cpu_run.wall_seconds
                used.append(cpu_run)
        bench_seconds = self.bench_report(run)["seconds"]
        self.on_progress("  starter: timing a chat reply")
        first_token_ms, tokens_per_second = chat_speed(self.lm(run), self.inputs.tok)
        grown_to = f"{s.grown_shape.n_layer} layers x {s.grown_shape.d_model} wide"
        self.on_progress(f"  starter: timing its growth to {grown_to}")
        model, _ = load_checkpoint(run.dir, self.device)
        on = next(model.parameters()).device  # where the growth's copies run
        started = _clock(on)
        grown = grow(model, s.grown_shape, seed=self.seeds[0])
        self.growth_seconds = _clock(on) - started
        self.growth_diff = max_logit_diff(model, grown, self.inputs.growth_sequences)
        del model, grown
        criterion = speed_criterion(
            run.device,
            run.wall_seconds,
            cpu_seconds,
            bench_seconds=bench_seconds,
            first_token_ms=first_token_ms,
            tokens_per_second=tokens_per_second,
            growth_seconds=self.growth_seconds,
        )
        return criterion, used

    def legibility(self) -> tuple[GateCriterion, list[GateRun]]:
        self.announce("G2", "the first model answers 20 probe prompts")
        probes = load_probes()
        chosen = [p for p in probes if p.kind == "open"][:10] + [
            p for p in probes if p.kind == "help"
        ][:10]
        lm = self.lm(self.starter_run)
        replies = [
            lm.chat_reply(
                [("user", probe.text)],
                max_new_tokens=LEGIBILITY_REPLY_TOKENS,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                seed=i,
            )
            for i, probe in enumerate(chosen)
        ]
        flags = [is_well_formed(reply, self.inputs.judge) for reply in replies]
        self.transcripts["G2"] = [(p.text, r) for p, r in zip(chosen, replies, strict=True)]
        self.raw["legibility_flags"] = flags
        return eval_legibility(flags), [self.starter_run]

    def differentiation(self) -> tuple[GateCriterion, list[GateRun]]:
        self.announce("G3", "four contrasting mixes at equal compute")
        s = self.scale
        scores: dict[str, list[dict[str, float]]] = {}
        for target in TARGET_CATEGORY:
            for seed in self.seeds:
                cfg = self.config(
                    f"g3-{target}-s{seed}",
                    shape=s.early_shape,
                    budget=s.early_budget,
                    mixture=mix_heavy(target),
                    seed=seed,
                )
                run = self.train(cfg)
                self.g3_runs.setdefault(target, []).append(run)
                scores.setdefault(target, []).append(self.bench(run))
        self.raw["differentiation_models"] = {t: rs[0].name for t, rs in self.g3_runs.items()}
        used = [run for rs in self.g3_runs.values() for run in rs]
        return eval_differentiation(scores), used

    def seed_variation(self) -> tuple[GateCriterion, list[GateRun]]:
        self.announce("G4", "one balanced mix, different seeds")
        s, g3 = self.scale, self.g3_runs
        balanced = [
            self.train(
                self.config(
                    f"g4-balanced-s{seed}",
                    shape=s.early_shape,
                    budget=s.early_budget,
                    mixture=BALANCED_MIX,
                    seed=seed,
                )
            )
            for seed in self.seeds
        ]
        models = [run for rs in g3.values() for run in rs] + balanced
        fps = {run.name: self.fingerprint(run) for run in models}
        population = list(fps.values())
        self.raw["describe"] = {name: describe(fp, population) for name, fp in fps.items()}
        self.transcripts["G3"] = self._g3_transcript(g3, fps)

        def distances(group: Sequence[GateRun]) -> list[float]:
            return [
                fingerprint_distance(fps[a.name], fps[b.name], population)
                for a, b in combinations(group, 2)
            ]

        firsts = [rs[0] for rs in g3.values()]
        return eval_seed_variation(distances(balanced), distances(firsts)), models

    @staticmethod
    def _g3_transcript(
        g3: dict[str, list[GateRun]], fps: Mapping[str, Fingerprint]
    ) -> list[tuple[str, str]]:
        """Each mix's first model answering :data:`G3_TRANSCRIPT_PROBES`, from its fingerprint."""
        texts = {p.id: p.text for p in load_probes()}
        exchanges = []
        for probe_id in G3_TRANSCRIPT_PROBES:
            for target, rs in g3.items():
                reply = next(
                    (
                        s["text"]
                        for s in fps[rs[0].name].samples
                        if s["probe"] == probe_id and s["sample"] == 0
                    ),
                    "",
                )
                exchanges.append((f"[{target}-heavy] {texts[probe_id]}", reply))
        return exchanges

    def growth(self) -> tuple[GateCriterion, list[GateRun]]:
        """G5: the growth measured on the first model (by G1), and the grown and ungrown
        continuations of it."""
        self.announce("G5", "continuing the first model, grown and not grown")
        s, seed, starter = self.scale, self.seeds[0], self.starter_run
        continued = {}
        for name, shape in (("g5-grown", s.grown_shape), ("g5-ungrown", s.starter_shape)):
            cfg = self.config(
                name, shape=shape, budget=s.starter_budget, mixture=BALANCED_MIX, seed=seed
            )
            continued[name] = self.train(cfg, parent=starter)
        criterion = eval_growth(
            self.growth_diff,
            continued["g5-grown"].heldout_mean,
            continued["g5-ungrown"].heldout_mean,
        )
        criterion.data["growth_seconds"] = self.growth_seconds
        return criterion, [starter, *continued.values()]

    def forgetting(self) -> tuple[GateCriterion, list[GateRun]]:
        self.announce("G6", "code-only continuations of the story-heavy model")
        s, seed = self.scale, self.seeds[0]
        base = self.g3_runs["creative"][0]
        children, used = {}, [base]
        for name, replay in (("g6-code-replay0", 0.0), ("g6-code-replay30", REPLAY_SHARE)):
            cfg = self.config(
                name,
                shape=s.early_shape,
                budget=s.starter_budget,
                mixture={"code": 1.0},
                seed=seed,
                replay=replay,
            )
            run = self.train(cfg, parent=base)
            used.append(run)
            children[name] = self.bench(run, FORGETTING_CATEGORIES)
        criterion = eval_forgetting(
            self.bench(base), children["g6-code-replay0"], children["g6-code-replay30"]
        )
        return criterion, used

    def preparation(self) -> tuple[GateCriterion, list[GateRun]]:
        self.announce("G7", "light vs thorough cleaning, and fact-checking")
        s, seed = self.scale, self.seeds[0]
        used: list[GateRun] = []

        def prep_run(name: str, mixture: Mapping[str, float], prep: PrepConfig) -> GateRun:
            cfg = self.config(
                name,
                shape=s.starter_shape,
                budget=s.prep_budget,
                mixture=mixture,
                seed=seed,
                prep=prep,
            )
            run = self.train(cfg)
            used.append(run)
            return run

        web = mix_heavy("web", WEB_HEAVY_SHARE)
        garble = {}
        for name, prep in (
            ("g7-web-light", PrepConfig("light")),
            ("g7-web-thorough", PrepConfig("thorough", dedup=True)),
        ):
            lm = self.lm(prep_run(name, web, prep))
            self.on_progress(f"  {name}: {GARBLE_SAMPLES} completions of held-out web text")
            completions = [
                lm.complete(
                    prefix,
                    max_new_tokens=GARBLE_NEW_TOKENS,
                    temperature=SAMPLE_TEMPERATURE,
                    top_p=SAMPLE_TOP_P,
                    seed=i,
                )
                for i, prefix in enumerate(self.inputs.web_prefixes)
            ]
            garble[name] = garble_rate(completions, self.inputs.known_vocab)
        false = {}
        for name, checked in (("g7-facts-unchecked", False), ("g7-facts-checked", True)):
            lm = self.lm(prep_run(name, FACTS_MIX, PrepConfig(fact_check=checked)))
            false[name] = false_fact_rate(lm, self.inputs.tok, self.inputs.kb, self.inputs.plan)
        criterion = eval_prep(
            garble["g7-web-light"],
            garble["g7-web-thorough"],
            false["g7-facts-unchecked"],
            false["g7-facts-checked"],
        )
        probes, skipped = false_fact_probes(self.inputs.kb, self.inputs.plan)
        criterion.data.update(
            {
                "completions": len(self.inputs.web_prefixes),
                "distinct_prompts": len(set(self.inputs.web_prefixes)),
                "fact_pairs": len(probes),
                "fact_pairs_skipped": skipped,
            }
        )
        return criterion, used


def _check_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(seeds)
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError(f"seeds must be one or more whole numbers >= 0, got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must all be different, got {seeds!r}")
    return seeds


def run_gate(
    data_root: Path,
    out_dir: Path,
    *,
    seeds: Sequence[int] = (1, 2, 3),
    device: torch.device | str | None = None,
    include_cpu_speed: bool = False,
    quick: bool = False,
    on_progress: Callable[[str], None] = print,
) -> GateOutcome:
    """Run the gate's experiments on ``data_root`` into ``out_dir/runs`` and measure its seven
    criteria (see the module docstring).

    The first seed plays "seed 1": the first model, the growth, forgetting and preparation runs
    use it, and so does each mix's model in the seed-variation and transcript comparisons.
    ``quick`` swaps in :data:`~airace_ml.experiments.configs.QUICK_SCALE` and keeps only the
    first seed. ``device=None`` picks one with :func:`~airace_ml.device.pick_device`. Raises
    :class:`GateSetupError` before training anything if an input is missing or damaged.
    """
    seeds = _check_seeds(seeds)
    if quick:
        seeds = seeds[:1]
    scale = QUICK_SCALE if quick else FULL_SCALE
    device = torch.device(device) if device is not None else pick_device()
    root = Path(data_root)
    out = Path(out_dir).resolve()
    inputs = _load_inputs(root, device, scale.starter_shape.ctx_len)
    training_code = source_digest(TRAINING_SOURCES)
    mode = "quick (tiny models: a check of the setup, not of feasibility)" if quick else "full"
    on_progress(f"gate: {mode}; seeds {', '.join(map(str, seeds))}; on {device.type}; into {out}")
    gate = _Gate(
        inputs=inputs,
        runs=GateRuns(root, out / RUNS_DIR, inputs.identity, on_progress, code=training_code),
        scale=scale,
        seeds=seeds,
        device=device,
        measuring_code=source_digest(MEASURING_SOURCES),
        include_cpu_speed=include_cpu_speed,
        on_progress=on_progress,
    )
    gate.raw.update({"quick": quick, "seeds": list(seeds), "device": device.type})
    criteria: list[GateCriterion] = []
    for experiment in (
        gate.speed,
        gate.legibility,
        gate.differentiation,
        gate.seed_variation,
        gate.growth,
        gate.forgetting,
        gate.preparation,
    ):
        criterion = require_completed(*experiment())
        criteria.append(criterion)
        verdict = "PASS" if criterion.passed else "FAIL"
        on_progress(f"{criterion.id} {criterion.title}: {verdict} - {criterion.detail}")
    return GateOutcome(criteria, gate.transcripts, _plain(gate.raw))
