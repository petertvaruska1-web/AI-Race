"""The benchmark suite ``bench-v1`` and how a model is run through it.

Every item comes from the skill generators' benchmark split, each category from its own
``skill_rng(<category>, "bench")`` stream, so the suite is the same on every machine. How each
kind of item is put to the model:

========================  ===================================================  ===============
Item                      Asked as                                             Chance
========================  ===================================================  ===============
``PairItem``              total log-prob of each text after ``<|bos|>``;       0.5
                          right if the good one is strictly higher
``MCItem``                mean log-prob per token of each option after the     1 / options
                          prompt; the best one is the answer (first on ties)
``ExactItem``             greedy reply; the extracted answer must match        0
``CheckItem``             greedy reply; the item's checker must pass it        0
consistency group         its members' multiple-choice answers; the score is   mean 1 / options
                          the fraction of pairs that chose the same text
========================  ===================================================  ===============

A plain prompt is ``encode_doc(prompt)`` and its options start with a space; a chat prompt is
``encode_chat([("user", prompt)], True)`` and its options do not (AI turns start without one).
Each category asks the model in as few calls as possible: one scoring call, and one generation
call per reply length.
"""

import copy
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from statistics import fmean
from typing import Any

from airace_ml.evals.scoring import (
    BenchReport,
    CategoryScore,
    ItemResult,
    agreement,
    category_score,
    exact_match,
    mc_choice,
    pair_correct,
)
from airace_ml.infer.lm import ContinuationScore, LanguageModel
from airace_ml.skills.checkers import CHECKERS
from airace_ml.skills.code import code_bench_items
from airace_ml.skills.facts import consistency_groups, knowledge_bench_items
from airace_ml.skills.grammar import grammar_pairs
from airace_ml.skills.instructions import instruction_bench_items
from airace_ml.skills.kb import load_kb
from airace_ml.skills.patterns import pattern_bench_items
from airace_ml.skills.reasoning import reasoning_bench_items
from airace_ml.skills.types import CheckItem, ExactItem, MCItem, PairItem, skill_rng
from airace_ml.tokenizer import Tok, encode_chat, encode_doc

CATEGORIES = (
    "language",
    "reasoning",
    "pattern",
    "knowledge",
    "coding",
    "creativity",
    "consistency",
    "instruction",
)
SUITE_VERSION = "bench-v1"
PAIR_CHANCE = 0.5

type BenchItem = PairItem | MCItem | ExactItem | CheckItem


@dataclass
class Suite:
    """A versioned set of benchmark items, by category, in the order they are run."""

    version: str
    items: dict[str, list]


@cache
def _bench_items() -> dict[str, list[BenchItem]]:
    """The bench-v1 items (built once per process; :func:`build_suite` hands out copies)."""
    kb = load_kb()
    builders = {
        "language": lambda rng: grammar_pairs(rng, n=200),
        "reasoning": lambda rng: reasoning_bench_items(rng, n=200),
        "pattern": lambda rng: pattern_bench_items(rng, n=150),
        "knowledge": lambda rng: knowledge_bench_items(kb, rng),
        "coding": lambda rng: code_bench_items(rng, n_output=100, n_func=50),
        "consistency": lambda rng: consistency_groups(kb, rng, n_groups=40),
        "instruction": lambda rng: instruction_bench_items(kb, rng, n=120),
    }
    return {category: build(skill_rng(category, "bench")) for category, build in builders.items()}


def build_suite() -> Suite:
    """The ``bench-v1`` suite: 200 language pairs, 200 reasoning, 150 pattern, 200 knowledge,
    150 coding, 120 consistency (40 groups of 3) and 120 instruction items.

    Each call returns its own copy, so changing it never changes the next one.
    """
    return Suite(SUITE_VERSION, copy.deepcopy(_bench_items()))


# -- asking the model ------------------------------------------------------------------------


@dataclass
class _Outcome:
    score: float
    chance: float
    output: str | None


def _prompt_ids(tok: Tok, prompt: str, chat: bool) -> list[int]:
    if chat:
        return encode_chat(tok, [("user", prompt)], add_generation_prompt=True)
    return encode_doc(tok, prompt)


def _scoring_requests(tok: Tok, item: PairItem | MCItem) -> list[tuple[list[int], list[int]]]:
    """The (context, continuation) pairs whose log-probabilities decide ``item``."""
    if isinstance(item, PairItem):
        return [([tok.bos_id], tok.encode(item.good)), ([tok.bos_id], tok.encode(item.bad))]
    context = _prompt_ids(tok, item.prompt, item.chat)
    lead = "" if item.chat else " "
    return [(context, tok.encode(lead + option)) for option in item.options]


def _judge_scores(item: PairItem | MCItem, scores: Sequence[ContinuationScore]) -> _Outcome:
    if isinstance(item, PairItem):
        return _Outcome(float(pair_correct(*scores)), PAIR_CHANCE, None)
    choice = mc_choice(scores)
    return _Outcome(float(choice == item.answer_index), 1 / len(item.options), item.options[choice])


def _judge_reply(item: ExactItem | CheckItem, reply: str) -> _Outcome:
    if isinstance(item, ExactItem):
        passed = exact_match(reply, item.answers, item.extract)
    else:
        passed = CHECKERS[item.check](reply, item.check_args)
    return _Outcome(float(passed), 0.0, reply)


def _ask(lm: LanguageModel, tok: Tok, items: Sequence[BenchItem], seed: int) -> list[_Outcome]:
    """Every item's outcome, from one scoring call and one generation call per reply length."""
    outcomes: list[_Outcome | None] = [None] * len(items)
    contexts: list[list[int]] = []
    continuations: list[list[int]] = []
    spans: list[tuple[int, int, int]] = []  # item index, first request, request count
    by_length: dict[int, list[int]] = {}  # max_new_tokens -> item indices
    for k, item in enumerate(items):
        if isinstance(item, (ExactItem, CheckItem)):
            by_length.setdefault(item.max_new_tokens, []).append(k)
            continue
        if not isinstance(item, (PairItem, MCItem)):
            raise TypeError(f"not a benchmark item: {item!r}")
        requests = _scoring_requests(tok, item)
        spans.append((k, len(contexts), len(requests)))
        contexts.extend(context for context, _ in requests)
        continuations.extend(continuation for _, continuation in requests)
    if contexts:
        scores = lm.score_continuations(contexts, continuations)
        if len(scores) != len(contexts):
            raise ValueError(f"asked for {len(contexts)} scores, got {len(scores)}")
        for k, first, count in spans:
            outcomes[k] = _judge_scores(items[k], scores[first : first + count])
    for max_new_tokens, indices in by_length.items():
        prompts = [_prompt_ids(tok, items[k].prompt, items[k].chat) for k in indices]
        replies = lm.generate(
            prompts, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, seed=seed
        )
        for k, reply in zip(indices, replies, strict=True):
            outcomes[k] = _judge_reply(items[k], tok.decode(reply.tokens))
    return outcomes  # every index was filled above


# -- categories ------------------------------------------------------------------------------


def _group_of(item: Any) -> str:
    if not isinstance(item, MCItem) or item.group is None:
        raise ValueError(f"consistency items are grouped multiple choice, not {item!r}")
    return item.group


def _limited(category: str, items: Sequence[BenchItem], limit: int | None) -> list[BenchItem]:
    """The first ``limit`` items; for consistency, as many whole groups as fit (at least one)."""
    if limit is None:
        return list(items)
    if category != "consistency":
        return list(items[:limit])
    kept: set[str] = set()
    total = 0
    for group, size in Counter(_group_of(item) for item in items).items():  # first-seen order
        if limit == 0 or (kept and total + size > limit):
            break
        kept.add(group)
        total += size
    return [item for item in items if _group_of(item) in kept]


def _consistency(
    category: str, items: Sequence[MCItem], outcomes: Sequence[_Outcome]
) -> tuple[CategoryScore, list[ItemResult]]:
    """One result per group: the agreement of its members' chosen option texts."""
    groups: dict[str, list[int]] = {}
    for k, item in enumerate(items):
        groups.setdefault(_group_of(item), []).append(k)
    results, chances = [], []
    for group, members in groups.items():
        if len(members) < 2:
            raise ValueError(f"consistency group {group!r} has fewer than 2 items")
        score = agreement([outcomes[k].output for k in members])
        tags = tuple(dict.fromkeys(tag for k in members for tag in items[k].tags))
        results.append(ItemResult(group, category, score, tags))
        chances.append(fmean(outcomes[k].chance for k in members))
    return category_score([r.score for r in results], chances), results


def _run_category(
    lm: LanguageModel, tok: Tok, category: str, items: Sequence[BenchItem], seed: int
) -> tuple[CategoryScore, list[ItemResult]]:
    outcomes = _ask(lm, tok, items, seed)
    if category == "consistency":
        return _consistency(category, items, outcomes)
    results = [
        ItemResult(item.id, category, outcome.score, tuple(item.tags), outcome.output)
        for item, outcome in zip(items, outcomes, strict=True)
    ]
    return category_score([o.score for o in outcomes], [o.chance for o in outcomes]), results


def _creativity(
    lm: LanguageModel, tok: Tok, judge: Any, novelty: Any, seed: int
) -> tuple[CategoryScore, list[ItemResult]] | None:
    """Creativity needs the reference judge and the novelty index; ``None`` without both."""
    if judge is None or novelty is None:
        return None
    from airace_ml.evals.creativity import score_creativity  # the module arrives with Task 17

    return score_creativity(lm, tok, judge, novelty, seed=seed)


def run_benchmarks(
    lm: LanguageModel,
    tok: Tok,
    *,
    suite: Suite | None = None,
    judge: Any = None,
    novelty: Any = None,
    categories: Sequence[str] = CATEGORIES,
    max_items_per_category: int | None = None,
    seed: int = 0,
) -> BenchReport:
    """Score ``lm`` on ``categories`` of ``suite`` (default: :func:`build_suite`), run and
    reported in :data:`CATEGORIES` order.

    ``max_items_per_category`` keeps the first items of each category in suite order;
    consistency keeps whole groups, at least one. A requested category that cannot be measured
    (creativity without both ``judge`` and ``novelty``, or no items) is listed in ``missing``.
    Replies are greedy, seeded with ``seed``. ``seconds`` is the wall-clock time of the run.
    """
    start = time.perf_counter()
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown benchmark categories {unknown}; expected some of {CATEGORIES}")
    if max_items_per_category is not None and max_items_per_category < 0:
        raise ValueError(f"max_items_per_category must be >= 0, got {max_items_per_category}")
    if suite is None:
        suite = build_suite()
    scores: dict[str, CategoryScore] = {}
    missing: list[str] = []
    results: list[ItemResult] = []
    for category in CATEGORIES:
        if category not in categories:
            continue
        if category == "creativity":
            measured = _creativity(lm, tok, judge, novelty, seed)
        else:
            items = _limited(category, suite.items.get(category, []), max_items_per_category)
            measured = _run_category(lm, tok, category, items, seed) if items else None
        if measured is None:
            missing.append(category)
            continue
        scores[category], category_results = measured
        results.extend(category_results)
    overall = fmean(s.score for s in scores.values()) if scores else 0.0
    return BenchReport(
        suite.version, scores, overall, missing, results, time.perf_counter() - start
    )
