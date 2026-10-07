"""Dataset recipes and assembly: which sources and generators make up each dataset, and in what
shares, and how much deliberate noise each dataset carries.

``assemble_dataset`` collects one dataset's documents component by component (normalized, and
filtered for simplicity where the recipe says so), then dirties a share of them with tagged noise.
Every random choice comes from the ``rng`` it is given, so a dataset is a pure function of its
inputs and seed.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from airace_content.noise import TYPO_RATE_RANGE, NoiseRates, add_typos, inject_noise
from airace_content.sources import (
    FORMAT_CHECK_ROWS,
    MIN_CHAT_TURNS,
    SOURCES,
    Content,
    Fetch,
    format_error,
    row_has_format,
    row_to_content,
)
from airace_content.textproc import normalize_text, simplicity
from airace_ml.data.corpus import DATASET_IDS, NOISE_KINDS, TOPICS
from airace_ml.skills.code import code_train_docs
from airace_ml.skills.facts import FalseFactPlan, fact_chat_docs, fact_prose_docs
from airace_ml.skills.instructions import instruction_train_docs
from airace_ml.skills.kb import KB
from airace_ml.skills.patterns import pattern_train_docs
from airace_ml.skills.reasoning import reasoning_train_docs
from airace_ml.skills.types import TextDoc


@dataclass
class Component:
    kind: Literal["source", "generator"]
    name: str
    share: float
    simplicity_min: float | None = None


@dataclass
class Recipe:
    dataset_id: str
    components: list[Component]
    noise: NoiseRates


def _source(name: str, share: float, simplicity_min: float | None = None) -> Component:
    return Component("source", name, share, simplicity_min)


def _generator(name: str, share: float) -> Component:
    return Component("generator", name, share)


RECIPES: dict[str, Recipe] = {
    recipe.dataset_id: recipe
    for recipe in (
        Recipe(
            "web",
            [_source("fineweb", 1.0, 0.35)],
            NoiseRates(
                typo=0.15,
                spam=0.06,
                boilerplate=0.10,
                garbled=0.05,
                duplicate=0.08,
                false_fact=0.06,
            ),
        ),
        Recipe(
            "books",
            [_source("gutenberg", 1.0, 0.30)],
            NoiseRates(typo=0.02, garbled=0.01, duplicate=0.02),
        ),
        Recipe(
            "educational",
            [
                _source("cosmo_khan", 0.25),
                _source("cosmo_wikihow", 0.25),
                _source("cosmo_openstax", 0.20),
                _source("fineweb_edu", 0.30, 0.35),
            ],
            NoiseRates(duplicate=0.02, false_fact=0.01),
        ),
        Recipe(
            "conversations",
            [
                _source("soda", 0.55),
                _source("everyday_conv", 0.15),
                _generator("fact_chat", 0.15),
                _generator("instructions", 0.15),
            ],
            NoiseRates(typo=0.03),
        ),
        Recipe(
            "code", [_generator("code", 0.80), _source("mbpp", 0.20)], NoiseRates(duplicate=0.03)
        ),
        Recipe(
            "reasoning",
            [_generator("reasoning", 0.60), _generator("patterns", 0.30), _source("gsm8k", 0.10)],
            NoiseRates(),
        ),
        Recipe(
            "facts",
            [_generator("fact_prose", 0.60), _source("simplewiki", 0.40, 0.30)],
            NoiseRates(duplicate=0.02),
        ),
        Recipe(
            "creative",
            [_source("tinystories", 0.85), _source("cosmo_stories", 0.15)],
            NoiseRates(typo=0.02, duplicate=0.03),
        ),
    )
}

SCALES: dict[str, dict] = {
    "tiny": {"docs_per_dataset": 150},
    "full": {
        "target_tokens": {
            "web": 30e6,
            "books": 15e6,
            "educational": 20e6,
            "conversations": 20e6,
            "code": 10e6,
            "reasoning": 10e6,
            "facts": 12e6,
            "creative": 20e6,
        }
    },
}

# Full scale counts the tokens of a component's first EST_SAMPLE_DOCS documents and, from their
# mean, the number of documents that reach the component's token quota.
EST_SAMPLE_DOCS = 200
# Tokens per UTF-8 byte of English text under a 4096-entry byte-level BPE, used to count tokens
# before the tokenizer exists (about 3.6 bytes per token; the summary reports the real counts).
BYTES_PER_TOKEN_ESTIMATE = 3.6
# Rows one pass over a source may read. Tiny scale cycles passes until it has its documents.
MAX_ROWS_PER_PASS = {"tiny": 20_000, "full": 1_000_000_000}

GeneratorFn = Callable[[KB, np.random.Generator, int], list[TextDoc]]

GENERATORS: dict[str, GeneratorFn] = {
    "fact_prose": fact_prose_docs,
    "fact_chat": fact_chat_docs,
    "instructions": instruction_train_docs,
    "code": lambda kb, rng, n: code_train_docs(rng, n),
    "reasoning": lambda kb, rng, n: reasoning_train_docs(rng, n),
    "patterns": lambda kb, rng, n: pattern_train_docs(rng, n),
}

# Spam replaces a document, boilerplate and garbling rewrite it as plain text, and duplicates and
# false facts are made from plain text: on chats only typos keep the chat a chat.
CHAT_NOISE_KINDS: tuple[str, ...] = ("typo",)
_RATE_KINDS = tuple(f.name for f in fields(NoiseRates))
_NONE = NOISE_KINDS.index("none")
_TYPO = NOISE_KINDS.index("typo")
_SPAM = NOISE_KINDS.index("spam")


@dataclass
class Collected:
    """A document as its component delivered it, before any noise."""

    content: Content
    topic: int | None = None  # the generator's topic (index into TOPICS); None: tag it by its text
    component: str = ""  # the name of the recipe component it came from


@dataclass
class AssembledDoc:
    content: Content
    noise_kind: int  # index into NOISE_KINDS
    false_fact: bool
    topic: int | None = None  # the generator's topic (index into TOPICS); None: tag it by its text
    origin: bytes = b""  # digest of the text before noise (``origin_key``); a copy has its source's
    component: str = ""  # the recipe component it came from (a copy: its source's)


def content_text(content: Content) -> str:
    """A document as one text: the text itself, or a chat's turn texts joined by newlines."""
    if isinstance(content, str):
        return content
    return "\n".join(text for _, text in content)


def origin_key(content: Content) -> bytes:
    """A digest of a document's content before noise. Documents with one origin are one text in
    different states of noise, so they are held out (or trained on) together."""
    if isinstance(content, str):
        data = "text:" + content
    else:
        data = "chat:" + json.dumps(content, ensure_ascii=False)
    return hashlib.blake2b(data.encode("utf-8"), digest_size=16).digest()


def estimate_tokens(text: str) -> int:
    """Tokens in ``text`` before the tokenizer exists: UTF-8 bytes / ``BYTES_PER_TOKEN_ESTIMATE``."""
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN_ESTIMATE)


def doc_tokens(content: Content, count_tokens: Callable[[str], int]) -> int:
    """Tokens of an encoded document: ``<|bos|>`` plus the text, or per turn a role, the text and
    ``<|end|>``."""
    if isinstance(content, str):
        return 1 + count_tokens(content)
    return 1 + sum(2 + count_tokens(text) for _, text in content)


def prepare_content(content: Content) -> Content | None:
    """``content`` normalized (each chat turn on its own) and trimmed; None when nothing is left
    (or a chat has fewer than ``MIN_CHAT_TURNS`` non-empty turns)."""
    if isinstance(content, str):
        text = normalize_text(content).strip()
        return text or None
    turns = [(role, normalize_text(text).strip()) for role, text in content]
    turns = [(role, text) for role, text in turns if text]
    return turns if len(turns) >= MIN_CHAT_TURNS else None


def validate_recipe(recipe: Recipe) -> None:
    """Raise ``ValueError`` unless the recipe is one the build can assemble."""
    where = f"recipe {recipe.dataset_id!r}"
    if recipe.dataset_id not in DATASET_IDS:
        raise ValueError(f"{where}: unknown dataset; expected one of {list(DATASET_IDS)}")
    if not recipe.components:
        raise ValueError(f"{where}: has no components")
    names = [comp.name for comp in recipe.components]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"{where}: {twice} named twice; each component needs its own name")
    for comp in recipe.components:
        known = SOURCES if comp.kind == "source" else GENERATORS
        if comp.kind not in ("source", "generator") or comp.name not in known:
            raise ValueError(f"{where}: unknown {comp.kind} {comp.name!r}")
        if not comp.share > 0:
            raise ValueError(f"{where}: {comp.name!r} has share {comp.share}; shares must be > 0")
        if comp.simplicity_min is not None:
            if recipe.dataset_id == "code":  # the simplicity measure is for prose, not programs
                raise ValueError(f"{where}: code is never filtered by prose simplicity")
            if not 0.0 <= comp.simplicity_min <= 1.0:
                raise ValueError(f"{where}: {comp.name!r} simplicity_min must be in [0, 1]")
    total = sum(comp.share for comp in recipe.components)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{where}: shares add up to {total}, not 1")


def split_count(total: int, shares: list[float]) -> list[int]:
    """``total`` split in proportion to ``shares`` (largest remainders; ties to the earlier)."""
    raw = [total * share / sum(shares) for share in shares]
    counts = [math.floor(x) for x in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: (counts[i] - raw[i], i))
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


def _share_out(amount: float, shares: list[float], scale: str) -> list[float]:
    """``amount`` of quota split in proportion to ``shares``: whole documents at tiny scale."""
    if scale == "tiny":
        return list(split_count(round(amount), shares))
    return [amount * share / sum(shares) for share in shares]


class _Goal:
    """Whether a component holds enough documents for its quota.

    A tiny-scale quota is a number of documents. A full-scale quota is a number of tokens: the
    tokens of the first ``EST_SAMPLE_DOCS`` documents are counted and, from their mean, the number
    of documents that reach the quota (before that, the counted tokens themselves).
    """

    def __init__(self, scale: str, count_tokens: Callable[[str], int]) -> None:
        self.scale = scale
        self.count_tokens = count_tokens
        self.sample_tokens = 0
        self.sample_docs = 0

    def add(self, content: Content) -> None:
        if self.scale == "full" and self.sample_docs < EST_SAMPLE_DOCS:
            self.sample_tokens += doc_tokens(content, self.count_tokens)
            self.sample_docs += 1

    def met(self, quota: float, n_docs: int) -> bool:
        if self.scale == "tiny":
            return n_docs >= quota
        if self.sample_docs < EST_SAMPLE_DOCS:
            return self.sample_tokens >= quota
        return n_docs >= math.ceil(quota * EST_SAMPLE_DOCS / self.sample_tokens)


def _simple_enough(content: Content, comp: Component, simple_vocab: set[str]) -> bool:
    if comp.simplicity_min is None:
        return True
    return simplicity(content_text(content), simple_vocab) >= comp.simplicity_min


class _SourceCollector:
    """One source component's documents, in row order. ``fill`` collects until the quota is met
    and resumes where it stopped when the component is given more quota later.

    Full scale reads the source once and is ``exhausted`` at its end. Tiny scale starts another
    pass over the rows instead (fixture files are short), unless a whole pass accepted no
    document, which no further pass would change: the component is exhausted then too. A source
    whose first ``FORMAT_CHECK_ROWS`` rows (or all its rows, when it has fewer) all lack the
    fields its adapter reads raises ``SourceFormatError`` instead of being filtered away.
    """

    def __init__(
        self,
        comp: Component,
        scale: str,
        fetch: Fetch,
        simple_vocab: set[str],
        count_tokens: Callable[[str], int],
    ) -> None:
        self.comp, self.scale, self.fetch = comp, scale, fetch
        self.simple_vocab, self.count_tokens = simple_vocab, count_tokens
        self.spec = SOURCES[comp.name]
        self.goal = _Goal(scale, count_tokens)
        self.docs: list[Collected] = []
        self.exhausted = False
        self.rows_read = 0
        self.passes = 0
        self._format_seen = False
        self._accepted = self._accept()

    def fill(self, quota: float) -> None:
        while not self.exhausted and not self.goal.met(quota, len(self.docs)):
            content = next(self._accepted, None)
            if content is None:
                self.exhausted = True
            else:
                self.docs.append(Collected(content, None, self.comp.name))
                self.goal.add(content)

    def amount(self) -> float:
        """What the component holds, in its quota's unit: documents (tiny) or tokens (full)."""
        if self.scale == "tiny":
            return len(self.docs)
        return sum(doc_tokens(doc.content, self.count_tokens) for doc in self.docs)

    def close(self) -> None:
        self._accepted.close()

    def _accept(self) -> Iterator[Content]:
        while True:
            self.passes += 1
            accepted = 0
            rows = self.fetch(self.spec, MAX_ROWS_PER_PASS[self.scale])
            try:
                for row in rows:
                    self._check_format(row)
                    for piece in row_to_content(self.spec.id, row):
                        content = prepare_content(piece)
                        if content is not None and _simple_enough(
                            content, self.comp, self.simple_vocab
                        ):
                            accepted += 1
                            yield content
            finally:
                close = getattr(rows, "close", None)
                if close is not None:
                    close()
            if self.rows_read and not self._format_seen:
                raise format_error(self.spec, self.rows_read)
            if self.scale == "full" or accepted == 0:
                return

    def _check_format(self, row: dict) -> None:
        self.rows_read += 1
        if self._format_seen:
            return
        if row_has_format(self.spec.id, row):
            self._format_seen = True
        elif self.rows_read >= FORMAT_CHECK_ROWS:
            raise format_error(self.spec, self.rows_read)


def _generated(comp: Component, kb: KB, rng: np.random.Generator, n: int) -> list[Collected]:
    """``n`` documents of a generator, prepared, each with its topic (None for "other")."""
    out: list[Collected] = []
    for doc in GENERATORS[comp.name](kb, rng, n) if n > 0 else []:
        content = prepare_content(doc.text if doc.kind == "plain" else list(doc.turns or []))
        if content is None:
            continue
        if doc.topic not in TOPICS:
            raise ValueError(f"generator {comp.name!r} gave unknown topic {doc.topic!r}")
        topic = None if doc.topic == "other" else TOPICS.index(doc.topic)
        out.append(Collected(content, topic, comp.name))
    return out


class _GeneratorCollector:
    """One generator component's documents. A generator never runs dry, so ``fill`` always meets
    the quota; at full scale it estimates tokens per document from a separate 200-document sample.
    """

    exhausted = False

    def __init__(
        self,
        comp: Component,
        scale: str,
        kb: KB,
        rng: np.random.Generator,
        count_tokens: Callable[[str], int],
    ) -> None:
        self.comp, self.scale, self.kb, self.count_tokens = comp, scale, kb, count_tokens
        self.sample_rng, self.rng = (np.random.default_rng(s) for s in rng.integers(2**63, size=2))
        self.docs: list[Collected] = []
        self._mean_tokens: float | None = None

    def fill(self, quota: float) -> None:
        if self.scale == "tiny":
            wanted = int(quota)
        else:
            if self._mean_tokens is None:
                sample = _generated(self.comp, self.kb, self.sample_rng, EST_SAMPLE_DOCS)
                tokens = sum(doc_tokens(doc.content, self.count_tokens) for doc in sample)
                self._mean_tokens = tokens / max(1, len(sample))
            wanted = math.ceil(quota / self._mean_tokens) if self._mean_tokens else 0
        if wanted > len(self.docs):
            self.docs += _generated(self.comp, self.kb, self.rng, wanted - len(self.docs))

    def amount(self) -> float:
        return len(self.docs)

    def close(self) -> None:
        pass


# --- noise -------------------------------------------------------------------------------------

_NON_LETTER = re.compile(r"[\W\d_]")  # every character str.isalpha() rejects, and a few more
MAX_COPY_TYPOS = 2  # noise.inject_noise misspells at most two letters of a duplicate copy


def _letter_runs(text: str) -> list[str]:
    """The runs of letters between the non-letter characters of ``text`` (some may be empty)."""
    return _NON_LETTER.split(text)


def _skeleton_key(text: str) -> bytes:
    """A digest of the non-letter characters of ``text``, in order."""
    skeleton = "".join(_NON_LETTER.findall(text))
    return hashlib.blake2b(skeleton.encode("utf-8"), digest_size=16).digest()


def source_of_copies(
    texts: list[str], copies: list[str], candidates: list[int]
) -> list[int | None]:
    """For each duplicate copy made by ``inject_noise``, the index of the text among
    ``candidates`` it was copied from (None when no candidate fits).

    A copy is its source with at most two letters misspelled, and a misspelling only ever changes
    letters, so the copy keeps every non-letter character of its source and differs from it in at
    most two runs of letters. Candidates are indexed by a digest of their non-letter characters,
    so memory stays small for large datasets.
    """
    by_skeleton: dict[bytes, list[int]] = {}
    for i in candidates:
        by_skeleton.setdefault(_skeleton_key(texts[i]), []).append(i)
    found: list[int | None] = []
    for copy in copies:
        match = None
        same_skeleton = by_skeleton.get(_skeleton_key(copy), [])
        if same_skeleton:
            runs = _letter_runs(copy)
            for i in same_skeleton:
                source_runs = _letter_runs(texts[i])
                if len(runs) == len(source_runs) and (
                    sum(a != b for a, b in zip(runs, source_runs)) <= MAX_COPY_TYPOS
                ):
                    match = i
                    break
        found.append(match)
    return found


def _typo_noise(
    docs: list[Collected], origins: list[bytes], rate: float, rng: np.random.Generator
) -> list[AssembledDoc]:
    """Typos in ``rate`` of the documents; a chat gets them in each of its turns and stays a chat."""
    quota = int(rate * len(docs) + 0.5)
    chosen = set(rng.permutation(len(docs))[:quota].tolist())
    out: list[AssembledDoc] = []
    for i, (doc, origin) in enumerate(zip(docs, origins)):
        new = doc.content
        if i in chosen:
            low, high = TYPO_RATE_RANGE
            letter_rate = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            if isinstance(new, str):
                new = add_typos(new, rng, letter_rate)
            else:
                new = [(role, add_typos(text, rng, letter_rate)) for role, text in new]
        kind = _TYPO if new != doc.content else _NONE
        out.append(AssembledDoc(new, kind, False, doc.topic, origin, doc.component))
    return out


def apply_noise(
    docs: list[Collected],
    rates: NoiseRates,
    rng: np.random.Generator,
    kb: KB | None = None,
    plan: FalseFactPlan | None = None,
    dataset_id: str = "",
) -> list[AssembledDoc]:
    """Dirty a dataset's documents and tag what was done.

    Every output document records its ``origin``, the digest of its text before noise, and its
    component. Plain documents go through ``inject_noise``; each duplicate copy it appends takes
    the origin, topic and component of the document it was copied from. A spam document replaced
    its original, so it loses the original's topic (not its origin). A dataset with chats may only
    have typo noise (a chat with typos keeps its turns); any other kind raises ``ValueError``.
    """
    origins = [origin_key(doc.content) for doc in docs]
    if any(not isinstance(doc.content, str) for doc in docs):
        refused = [
            kind
            for kind in _RATE_KINDS
            if kind not in CHAT_NOISE_KINDS and getattr(rates, kind) > 0
        ]
        if refused:
            raise ValueError(
                f"dataset {dataset_id!r} has chat documents, but its recipe adds {refused} noise: "
                f"those kinds rewrite or copy a document as plain text, and chats take only "
                f"{list(CHAT_NOISE_KINDS)} noise"
            )
        return _typo_noise(docs, origins, rates.typo, rng)

    texts = [doc.content for doc in docs]
    noisy = inject_noise(texts, rates, rng, kb, plan)
    out = [
        AssembledDoc(
            nd.text,
            nd.noise_kind,
            nd.false_fact,
            None if nd.noise_kind == _SPAM else doc.topic,
            origin,
            doc.component,
        )
        for nd, doc, origin in zip(noisy, docs, origins)
    ]
    copies = noisy[len(docs) :]
    untouched = [i for i, nd in enumerate(noisy[: len(docs)]) if nd.noise_kind == _NONE]
    sources = source_of_copies(texts, [nd.text for nd in copies], untouched) if copies else []
    for nd, source in zip(copies, sources):
        if source is None:  # not expected: a copy always keeps its source's non-letters
            out.append(
                AssembledDoc(nd.text, nd.noise_kind, nd.false_fact, None, origin_key(nd.text))
            )
        else:
            doc = docs[source]
            out.append(
                AssembledDoc(
                    nd.text, nd.noise_kind, nd.false_fact, doc.topic, origins[source], doc.component
                )
            )
    return out


# --- assembly ----------------------------------------------------------------------------------


def assemble_dataset(
    dataset_id: str,
    scale: str,
    fetch: Fetch,
    kb: KB,
    simple_vocab: set[str],
    plan: FalseFactPlan,
    rng: np.random.Generator,
    *,
    count_tokens: Callable[[str], int] = estimate_tokens,
    stats: dict | None = None,
) -> list[AssembledDoc]:
    """One dataset's documents, by its recipe: each component's share of the scale's documents
    (tiny) or tokens (full), in component order, then the recipe's noise.

    A source that runs out before its quota is met is ``exhausted``, and its unmet quota goes to
    the dataset's other components that are not exhausted, in proportion to their shares, until
    the dataset's total is met or every component is exhausted.

    ``count_tokens`` counts the tokens of a text for the full-scale quotas (the frozen tokenizer
    when there is one). ``stats``, when given, receives per component its planned and assigned
    quota (documents at tiny scale, tokens at full scale), its documents and whether it ran out.
    """
    if dataset_id not in RECIPES:
        raise ValueError(f"unknown dataset {dataset_id!r}; expected one of {list(DATASET_IDS)}")
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {list(SCALES)}")
    recipe = RECIPES[dataset_id]
    validate_recipe(recipe)
    shares = [comp.share for comp in recipe.components]
    if scale == "tiny":
        planned: list[float] = list(split_count(SCALES["tiny"]["docs_per_dataset"], shares))
    else:
        target = SCALES["full"]["target_tokens"][dataset_id]
        planned = [target * share for share in shares]
    seeds = rng.integers(2**63, size=len(recipe.components) + 1)
    collectors: list[_SourceCollector | _GeneratorCollector] = [
        _SourceCollector(comp, scale, fetch, simple_vocab, count_tokens)
        if comp.kind == "source"
        else _GeneratorCollector(comp, scale, kb, np.random.default_rng(seed), count_tokens)
        for comp, seed in zip(recipe.components, seeds[:-1])
    ]
    assigned = list(planned)
    try:
        while True:
            unmet = 0.0
            for i, collector in enumerate(collectors):
                if not collector.exhausted:
                    collector.fill(assigned[i])
                    if collector.exhausted:
                        unmet += max(0.0, assigned[i] - collector.amount())
            remaining = [i for i, collector in enumerate(collectors) if not collector.exhausted]
            if unmet <= 0 or not remaining:
                break
            extra = _share_out(unmet, [shares[i] for i in remaining], scale)
            for i, more in zip(remaining, extra):
                assigned[i] += more
    finally:
        for collector in collectors:
            collector.close()

    if stats is not None:
        unit = "docs" if scale == "tiny" else "tokens"
        stats["components"] = []
        for comp, collector, plan_i, assigned_i in zip(
            recipe.components, collectors, planned, assigned
        ):
            info = {"kind": comp.kind, "name": comp.name, "share": comp.share}
            if comp.simplicity_min is not None:
                info["simplicity_min"] = comp.simplicity_min
            info |= {
                f"planned_{unit}": plan_i,
                f"assigned_{unit}": assigned_i,
                "docs": len(collector.docs),
                "exhausted": collector.exhausted,
            }
            if isinstance(collector, _SourceCollector):
                info |= {"rows_read": collector.rows_read, "passes": collector.passes}
            stats["components"].append(info)
    docs = [doc for collector in collectors for doc in collector.docs]
    return apply_noise(docs, recipe.noise, np.random.default_rng(seeds[-1]), kb, plan, dataset_id)
