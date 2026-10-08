"""The personality fingerprint: ten traits measured from how a model answers 60 fixed probes.

:func:`measure_fingerprint` asks the model every probe of :func:`load_probes` ``k`` times, all in
one batched ``generate`` call (so identical prompts get independent samples), and measures the
replies. Personality is a measurement (R5): nothing here scripts, edits or filters what the model
says, and a trait is only ever computed from the model's own replies and token probabilities.

Words are lowercase runs of letters, the same words creativity scoring counts; everywhere below
"word", "bigram" and "3-gram" mean those. Every trait is finite, and 0 when there is nothing to
measure (no probes, no replies, no words), whatever the model writes: empty replies, only special
tokens, or endless repetition.

==============  ==========================================================================
verbosity       mean word count of a reply
confidence      mean, over replies with at least one token, of the mean top-1 probability of
                the reply's tokens (the model's own temperature-1 confidence)
inventiveness   distinct-2: unique word bigrams over all word bigrams, pooled over all replies
steadiness      mean over probes of the mean pairwise Jaccard similarity of the word sets of
                the probe's ``k`` replies (two empty sets agree fully). With ``k == 1`` there
                are no pairs to compare, so steadiness is 0.0
precision       fraction of factual probes where at least one reply contains a right answer
boldness        fraction of replies with at least 3 words and a repetitiveness below 0.5
slip_rate       fraction of factual replies that contain a wrong answer
register        ``(formal - casual) / (formal + casual + 1)`` over the lexicon words of all
                replies pooled: +1 is formal, -1 is casual
warmth          ``10 * (positive words + "!") / max(1, words)`` over all replies pooled
repetitiveness  mean over replies of ``1 - unique word 3-grams / all word 3-grams`` (0 for a
                reply with no 3-gram)
==============  ==========================================================================

An answer counts when it appears as a whole word, or a whole phrase for a multi-word answer,
ignoring case (``cat`` is not found in ``category``; ``Port-au-Prince`` is found in ``port au
prince``). A dotted abbreviation of single letters is the joined form, in the answers and in the
reply alike: ``Washington D.C.`` is ``Washington DC`` (see :func:`answer_words`).

Traits only mean something relative to other models: :func:`describe` puts a fingerprint in plain
words against a population and :func:`fingerprint_distance` compares two fingerprints in units of
how much the population varies.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np

from airace_ml.evals.creativity import distinct_2, repetitiveness, words
from airace_ml.infer.lm import LanguageModel
from airace_ml.tokenizer import Tok, encode_chat

TRAITS: tuple[str, ...] = (
    "verbosity",
    "confidence",
    "inventiveness",
    "steadiness",
    "precision",
    "boldness",
    "slip_rate",
    "register",
    "warmth",
    "repetitiveness",
)

ProbeKind = Literal["open", "factual", "creative", "help", "opinion"]
PROBE_KINDS: tuple[ProbeKind, ...] = ("open", "factual", "creative", "help", "opinion")
PROBES_PATH = Path(__file__).with_name("probes.json")

BOLD_MIN_WORDS = 3
BOLD_MAX_REPETITIVENESS = 0.5  # a reply that repeats half its 3-grams or more is not a real try
WARMTH_SCALE = 10
MIN_POPULATION = 5  # fewer fingerprints than this say nothing about what is unusual
STD_FLOOR = 1e-6  # a trait nobody varies in must not make every difference infinite

# The lexicons hold words that mark a register or an affect and nothing else. Words that are just
# as common in children's stories and plain speech ("once upon a time", "a warm day", "a good
# friend", "it is cool") are left out, so a model is not called formal, casual or warm for
# telling an ordinary story or repeating a question.
# fmt: off
FORMAL_WORDS: frozenset[str] = frozenset({
    "therefore", "however", "additionally", "furthermore", "moreover", "consequently", "thus",
    "hence", "nevertheless", "nonetheless", "accordingly", "subsequently", "regarding",
    "whereas", "whereby", "thereby", "hereby", "herein", "thereof", "therein", "wherein",
    "thereafter", "henceforth", "hitherto", "whilst", "pursuant", "notwithstanding",
    "aforementioned", "approximately", "sufficient", "obtain", "assist", "assistance",
    "demonstrate", "utilize", "commence", "endeavor", "facilitate", "comprehend", "inquire",
    "respectively", "numerous", "ascertain", "forthwith",
})
CASUAL_WORDS: frozenset[str] = frozenset({
    "yeah", "yep", "yup", "nope", "nah", "lol", "haha", "hehe", "omg", "hey", "hiya", "yo",
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme", "gimme", "cuz", "tho", "btw",
    "ya", "dude", "bro", "idk", "imo", "thx", "pls", "ppl", "lmao", "rofl", "gotcha", "sup",
})
POSITIVE_WORDS: frozenset[str] = frozenset({
    "love", "loved", "loves", "loving", "adore", "happy", "happily", "glad", "joy", "joyful",
    "delighted", "cheerful", "great", "wonderful", "lovely", "amazing", "fantastic", "excellent",
    "kind", "kindly", "caring", "thank", "thanks", "thankful", "grateful", "appreciate",
    "appreciated", "enjoy", "enjoyed", "enjoys", "smile", "smiles", "hug", "proud", "pleasure",
    "pleased", "thrilled", "excited", "cherish", "adorable", "affection", "fond", "delight",
})
# fmt: on

# (low phrase, high phrase) per trait, in the plain words a player uses (R7).
TRAIT_PHRASES: dict[str, tuple[str, str]] = {
    "verbosity": ("terse", "talkative"),
    "confidence": ("hesitant", "self-assured"),
    "inventiveness": ("predictable", "inventive"),
    "steadiness": ("changeable", "steady"),
    "precision": ("vague on facts", "careful with facts"),
    "boldness": ("timid", "bold"),
    "slip_rate": ("rarely slips", "often slips"),
    "register": ("casual", "formal"),
    "warmth": ("reserved", "warm"),
    "repetitiveness": ("varied", "repetitive"),
}


@dataclass
class Probe:
    """One prompt of the fixed probe set.

    A ``factual`` probe also carries ``answers`` (the KB's true object and the other forms it
    accepts) and ``wrong_answers`` (objects of the same relation that are clearly wrong for it).
    """

    id: str
    text: str
    kind: ProbeKind
    answers: list[str] | None = None
    wrong_answers: list[str] | None = None


def _parse_probe(raw: object) -> Probe:
    if not isinstance(raw, dict):
        raise TypeError(f"probe {raw!r} must be a JSON object")
    probe = Probe(**raw)
    where = f"probe {probe.id!r}"
    if probe.kind not in PROBE_KINDS:
        raise ValueError(f"{where}: unknown kind {probe.kind!r}")
    if not isinstance(probe.text, str) or not probe.text.strip():
        raise ValueError(f"{where}: text must be non-empty")
    if probe.kind == "factual" and not (probe.answers and probe.wrong_answers):
        raise ValueError(f"{where}: a factual probe needs answers and wrong_answers")
    return probe


def load_probes() -> list[Probe]:
    """The 60 probes (open 20, factual 15, creative 10, help 10, opinion 5) from ``probes.json``.

    The set is frozen, and a new list is returned on every call.
    """
    probes = [_parse_probe(raw) for raw in json.loads(PROBES_PATH.read_text(encoding="utf-8"))]
    if len({p.id for p in probes}) != len(probes):
        raise ValueError("probes.json: probe ids must be unique")
    return probes


@dataclass
class Fingerprint:
    """A model's measured personality.

    ``traits`` maps every name of :data:`TRAITS` to a finite number and is the public part.
    ``samples`` records every reply as ``{"probe": id, "kind": kind, "sample": j, "text": reply}``.
    """

    traits: dict[str, float]
    samples: list[dict]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe copy (plain dicts, lists, floats, ints and strings)."""
        return {"traits": dict(self.traits), "samples": [dict(s) for s in self.samples]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Fingerprint:
        """Rebuild what :meth:`to_dict` wrote. Raises ``ValueError`` if ``data`` does not hold
        exactly the traits of :data:`TRAITS` or a trait is not finite, ``TypeError`` if a trait is
        not a number, and ``ValueError`` if the samples are not a list of objects."""
        traits = data.get("traits") if isinstance(data, Mapping) else None
        samples = data.get("samples") if isinstance(data, Mapping) else None
        if not isinstance(traits, Mapping) or set(traits) != set(TRAITS):
            raise ValueError(f"a fingerprint needs exactly the traits {list(TRAITS)}")
        for name, value in traits.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"trait {name!r} must be a number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"trait {name!r} must be finite, got {value!r}")
        if not isinstance(samples, list) or not all(isinstance(s, Mapping) for s in samples):
            raise ValueError("a fingerprint's samples must be a list of objects")
        return cls({name: float(traits[name]) for name in TRAITS}, [dict(s) for s in samples])


@dataclass(frozen=True)
class _Reply:
    text: str
    words: list[str]
    confidence: float | None  # mean top-1 probability; None when no token has a finite one


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# Single letters joined by periods: "D.C.", "U.S.", "U.S.A" (not "Mr.", "3.5" or "ph.D").
_ABBREVIATION = re.compile(r"(?<![^\W\d_])[^\W\d_](?:\.[^\W\d_])+\.?(?![^\W\d_])")


def answer_words(text: str) -> list[str]:
    """The words of ``text`` as answers are matched: :func:`words`, with a dotted abbreviation
    read as its joined form, so ``Washington D.C.`` and ``Washington DC`` are the same."""
    return words(_ABBREVIATION.sub(lambda m: m.group().replace(".", ""), text))


def _phrases(answers: Sequence[str] | None) -> set[tuple[str, ...]]:
    """Each answer as its words; an answer with no letters can never be found."""
    return {tuple(ws) for answer in answers or () if (ws := answer_words(answer))}


def _spans(ws: Sequence[str], longest: int) -> set[tuple[str, ...]]:
    """Every run of 1 to ``longest`` consecutive words."""
    return {tuple(ws[i : i + n]) for n in range(1, longest + 1) for i in range(len(ws) - n + 1)}


def _answer_flags(probe: Probe, replies: Sequence[_Reply]) -> list[tuple[bool, bool]]:
    """Per reply: does it contain a right answer, and does it contain a wrong one."""
    right, wrong = _phrases(probe.answers), _phrases(probe.wrong_answers)
    longest = max((len(p) for p in right | wrong), default=0)
    flags = []
    for reply in replies:
        found = _spans(answer_words(reply.text), longest)
        flags.append((bool(found & right), bool(found & wrong)))
    return flags


def _traits(groups: Sequence[tuple[Probe, list[_Reply]]]) -> dict[str, float]:
    """The ten traits from every probe with its replies."""
    replies = [r for _, group in groups for r in group]
    texts = [r.text for r in replies]
    counts = [len(r.words) for r in replies]
    total_words = sum(counts)
    pairs_per_probe = [
        [_jaccard(set(a.words), set(b.words)) for a, b in combinations(group, 2)]
        for _, group in groups
    ]
    factual = [_answer_flags(p, g) for p, g in groups if p.kind == "factual"]
    flat = [flag for flags in factual for flag in flags]
    pooled = [w for r in replies for w in r.words]
    formal = sum(w in FORMAL_WORDS for w in pooled)
    casual = sum(w in CASUAL_WORDS for w in pooled)
    positive = sum(w in POSITIVE_WORDS for w in pooled)
    repeats = [repetitiveness(t) for t in texts]
    return {
        "verbosity": _mean(counts),
        "confidence": _mean([r.confidence for r in replies if r.confidence is not None]),
        "inventiveness": distinct_2(texts),
        "steadiness": _mean([_mean(pairs) for pairs in pairs_per_probe]),
        "precision": _mean([float(any(right for right, _ in flags)) for flags in factual]),
        "boldness": _mean(
            [
                float(n >= BOLD_MIN_WORDS and rep < BOLD_MAX_REPETITIVENESS)
                for n, rep in zip(counts, repeats, strict=True)
            ]
        ),
        "slip_rate": _mean([float(wrong) for _, wrong in flat]),
        "register": (formal - casual) / (formal + casual + 1),
        "warmth": WARMTH_SCALE
        * (positive + sum(t.count("!") for t in texts))
        / max(1, total_words),
        "repetitiveness": _mean(repeats),
    }


def measure_fingerprint(
    lm: LanguageModel,
    tok: Tok,
    *,
    probes: Sequence[Probe] | None = None,
    k: int = 3,
    seed: int = 0,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
) -> Fingerprint:
    """Ask ``lm`` every probe ``k`` times and measure its traits (see the module docstring).

    ``probes`` defaults to :func:`load_probes`. All ``len(probes) * k`` prompts go to the model in
    one ``generate`` call, so a given ``seed`` gives the same fingerprint. A reply is the decoded
    tokens, stripped. ``k`` must be at least 1; with ``k == 1`` steadiness is 0.0 (no pairs).
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    probe_list = list(load_probes() if probes is None else probes)
    prompts = [
        encode_chat(tok, [("user", probe.text)], add_generation_prompt=True)
        for probe in probe_list
        for _ in range(k)
    ]
    generations = (
        lm.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        if prompts
        else []
    )
    if len(generations) != len(prompts):
        raise ValueError(f"asked for {len(prompts)} replies, the model gave {len(generations)}")
    replies: list[_Reply] = []
    samples: list[dict] = []
    for n, generation in enumerate(generations):
        probe, j = probe_list[n // k], n % k
        text = tok.decode(generation.tokens).strip()
        finite = [p for p in generation.top1_probs if math.isfinite(p)]
        replies.append(_Reply(text, words(text), _mean(finite) if finite else None))
        samples.append({"probe": probe.id, "kind": probe.kind, "sample": j, "text": text})
    groups = [(probe, replies[i * k : (i + 1) * k]) for i, probe in enumerate(probe_list)]
    return Fingerprint(_traits(groups), samples)


def describe(
    fp: Fingerprint,
    population: Sequence[Fingerprint],
    low_pct: float = 15,
    high_pct: float = 85,
) -> list[str]:
    """Plain words for the traits of ``fp`` that are unusual in ``population``.

    A trait below the ``low_pct`` percentile of the population gives its low phrase of
    :data:`TRAIT_PHRASES`, one above the ``high_pct`` percentile its high phrase (strict
    comparisons, so ties and a trait nobody varies in say nothing), in :data:`TRAITS` order. A
    population of fewer than :data:`MIN_POPULATION` fingerprints cannot tell what is unusual and
    gives ``[]``.
    """
    if not 0 <= low_pct <= high_pct <= 100:
        raise ValueError(f"need 0 <= low_pct <= high_pct <= 100, got {low_pct} and {high_pct}")
    if len(population) < MIN_POPULATION:
        return []
    phrases = []
    for trait in TRAITS:
        low, high = np.percentile([p.traits[trait] for p in population], [low_pct, high_pct])
        low_phrase, high_phrase = TRAIT_PHRASES[trait]
        value = fp.traits[trait]
        if value < low:
            phrases.append(low_phrase)
        elif value > high:
            phrases.append(high_phrase)
    return phrases


def fingerprint_distance(
    a: Fingerprint, b: Fingerprint, population: Sequence[Fingerprint]
) -> float:
    """Euclidean distance between two fingerprints, each trait in units of the population's
    standard deviation (``ddof=0``, at least :data:`STD_FLOOR`; an empty population has none, so
    every trait uses the floor). Symmetric, and 0 for identical fingerprints."""
    total = 0.0
    for trait in TRAITS:
        values = [p.traits[trait] for p in population]
        std = max(float(np.std(values)) if values else 0.0, STD_FLOOR)
        total += ((a.traits[trait] - b.traits[trait]) / std) ** 2
    return math.sqrt(total)
