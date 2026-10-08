"""The creativity category: stories a model writes, scored as coherence x novelty x diversity.

The model writes one story for each of the 24 :data:`STORY_PROMPTS`, all sampled in one batched
call (temperature 0.9, top-p 0.95, seeded). Each story is measured four ways:

* **coherence** ``c``: its per-token loss under the reference judge, placed on the judge's own
  scale for real stories, ``clip((p90 - loss) / (p90 - p10), 0, 1)``. As fluent as the best real
  stories is 1; stranger than 90% of them is 0.
* **novelty** ``nov``: the fraction of its sampled 8-grams that no corpus contains.
* **repetitiveness** ``rep``: ``1 - unique word 3-grams / all word 3-grams`` within the story (0
  when it has no 3-gram). A loop reads as easy to a judge and every repeat of a new phrase counts
  as new, so without it "the the the ..." would score about 50; with it, a phrase said ``k``
  times keeps about ``1 / k`` and endless repetition scores about 0.
* together, ``story = c * (0.4 + 0.6 * nov) * (1 - rep)``: an incoherent story is worth nothing
  however new, a fluent copy of the training data keeps 40%, and a loop keeps almost nothing.

**Diversity** ``d`` is distinct-2 over all the stories: unique word bigrams over all word bigrams
(words are runs of lowercase letters). The category score is ``100 * mean(story) * (0.5 + 0.5 *
d)``, so a model that tells the same story to every prompt loses up to half. A story with no words
scores 0, and every score is finite whatever the model writes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from itertools import pairwise
from statistics import fmean
from typing import TYPE_CHECKING

from airace_ml.evals.scoring import CategoryScore, ItemResult
from airace_ml.infer.lm import LanguageModel
from airace_ml.tokenizer import Tok, encode_chat

if TYPE_CHECKING:
    from airace_ml.evals.judge import Judge, JudgeCalibration
    from airace_ml.evals.novelty import NoveltyIndex

CATEGORY = "creativity"
TEMPERATURE = 0.9
TOP_P = 0.95
STORY_TOKENS = 96
NOVELTY_FLOOR = 0.4  # what a fluent story keeps when nothing in it is new
_WORD = re.compile(r"[^\W\d_]+")  # a run of letters

_STORIES: tuple[tuple[str, str], ...] = (
    ("animals", "Write a short story about a dog who is afraid of the dark."),
    ("animals", "Write a short story about a cat who wants to fly."),
    ("animals", "Write a short story about a turtle who finds a lost key."),
    ("animals", "Write a short story about a little bird who cannot sing."),
    ("places", "Write a short story about a quiet town by the sea."),
    ("places", "Write a short story about a house at the top of a hill."),
    ("places", "Write a short story about a day at a busy market."),
    ("places", "Write a short story about a secret room in a school."),
    ("feelings", "Write a short story about a girl who feels lonely on her first day."),
    ("feelings", "Write a short story about a boy who is proud of his drawing."),
    ("feelings", "Write a short story about feeling scared before a big test."),
    ("feelings", "Write a short story about two friends who are angry with each other."),
    ("objects", "Write a short story about a red ball that rolls away."),
    ("objects", "Write a short story about an old clock that stops."),
    ("objects", "Write a short story about a lost shoe."),
    ("objects", "Write a short story about a magic box under the bed."),
    ("people", "Write a short story about a baker who makes a giant cake."),
    ("people", "Write a short story about a grandmother and her garden."),
    ("people", "Write a short story about a new teacher on a rainy day."),
    ("people", "Write a short story about a farmer who finds a strange seed."),
    ("nature", "Write a short story about a tree that grows very fast."),
    ("nature", "Write a short story about the first snow of winter."),
    ("nature", "Write a short story about a storm on the mountain."),
    ("nature", "Write a short story about a river that changes color."),
)
STORY_PROMPTS: tuple[str, ...] = tuple(prompt for _, prompt in _STORIES)


def _words(text: str) -> list[str]:
    """The words of ``text``: runs of letters, lowercased."""
    return _WORD.findall(text.lower())


def distinct_2(texts: Iterable[str]) -> float:
    """Unique word bigrams over all word bigrams, pooled over ``texts`` (bigrams never span two
    texts); 0.0 when there are none."""
    bigrams = [pair for text in texts for pair in pairwise(_words(text))]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def repetitiveness(text: str) -> float:
    """``1 - unique word 3-grams / all word 3-grams`` of ``text``; 0.0 when it has no 3-gram."""
    words = _words(text)
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0


def _coherence(nll: float, calibration: JudgeCalibration) -> float:
    """``clip((p90 - nll) / (p90 - p10), 0, 1)``; a NaN loss is 0, and a calibration with no
    spread (p90 <= p10) is a plain test against p90."""
    p10, p90 = calibration.creative_p10, calibration.creative_p90
    value = (p90 - nll) / (p90 - p10) if p90 > p10 else float(nll <= p90)
    return 0.0 if math.isnan(value) else min(max(value, 0.0), 1.0)


def score_creativity(
    lm: LanguageModel,
    tok: Tok,
    judge: Judge,
    novelty: NoveltyIndex,
    *,
    seed: int = 0,
    max_new_tokens: int = STORY_TOKENS,
) -> tuple[CategoryScore, list[ItemResult]]:
    """Have ``lm`` write a story for every prompt of :data:`STORY_PROMPTS` and score them (see
    the module docstring).

    ``raw`` is the mean story score (0-1). Each :class:`ItemResult` carries one story's score
    and text.
    """
    prompts = [
        encode_chat(tok, [("user", prompt)], add_generation_prompt=True) for prompt in STORY_PROMPTS
    ]
    generations = lm.generate(
        prompts, max_new_tokens=max_new_tokens, temperature=TEMPERATURE, top_p=TOP_P, seed=seed
    )
    stories = [tok.decode(g.tokens).strip() for g in generations]
    losses = judge.nll_per_token([judge.tok.encode(story) for story in stories])
    results = []
    for k, ((topic, _), story, nll) in enumerate(zip(_STORIES, stories, losses, strict=True)):
        score = 0.0
        if _words(story):
            new = novelty.novelty(tok.encode(story))
            score = (
                _coherence(nll, judge.calibration)
                * (NOVELTY_FLOOR + (1 - NOVELTY_FLOOR) * new)
                * (1 - repetitiveness(story))
            )
        results.append(
            ItemResult(f"story-{k:02d}", CATEGORY, score, ("fmt:story", f"topic:{topic}"), story)
        )
    raw = fmean(r.score for r in results)
    d = distinct_2(stories)
    return CategoryScore(100 * raw * (0.5 + 0.5 * d), raw, len(results)), results
