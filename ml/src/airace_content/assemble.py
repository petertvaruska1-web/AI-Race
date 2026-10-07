"""Dataset recipes and assembly: which sources and generators make up each dataset, and in what
shares, and how much deliberate noise each dataset carries.

``assemble_dataset`` collects one dataset's documents component by component (normalized, and
filtered for simplicity where the recipe says so), then dirties a share of them with tagged noise.
Every random choice comes from the ``rng`` it is given, so a dataset is a pure function of its
inputs and seed.
"""

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from airace_content.noise import TYPO_RATE_RANGE, NoiseRates, add_typos, inject_noise
from airace_content.sources import SOURCES, Content, Fetch, row_to_content
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

# Full scale counts tokens per document over this many first documents of a component, then
# keeps collecting until the documents' estimated tokens reach the component's target.
EST_SAMPLE_DOCS = 200
# Tokens per UTF-8 byte of English text under a 4096-entry byte-level BPE, used to count tokens
# before the tokenizer exists (about 3.6 bytes per token; the summary reports the real counts).
BYTES_PER_TOKEN_ESTIMATE = 3.6
# Rows one pass over a source may read. Tiny scale cycles passes until it has its documents.
MAX_ROWS_PER_PASS = {"tiny": 20_000, "full": 1_000_000_000}
MIN_CHAT_TURNS = 2

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
_RATE_KINDS = ("typo", "spam", "boilerplate", "garbled", "duplicate", "false_fact")
_NONE = NOISE_KINDS.index("none")
_TYPO = NOISE_KINDS.index("typo")
_SPAM = NOISE_KINDS.index("spam")


@dataclass
class AssembledDoc:
    content: Content
    noise_kind: int  # index into NOISE_KINDS
    false_fact: bool
    topic: int | None = None  # the generator's topic (index into TOPICS); None: tag it by its text
    copy_of: int | None = None  # a duplicate copy: index of the document it was copied from


def content_text(content: Content) -> str:
    """A document as one text: the text itself, or a chat's turn texts joined by newlines."""
    if isinstance(content, str):
        return content
    return "\n".join(text for _, text in content)


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


class _Goal:
    """When a component has collected enough documents.

    Tiny scale wants a number of documents. Full scale wants a number of tokens: it counts the
    tokens of the first ``EST_SAMPLE_DOCS`` documents, and from their mean, the number of
    documents that reach the target.
    """

    def __init__(
        self,
        *,
        docs: int | None = None,
        tokens: float = 0.0,
        count_tokens: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self.docs = docs
        self.tokens = tokens
        self.count_tokens = count_tokens
        self.sample_tokens = 0
        self.sample_docs = 0

    def add(self, content: Content) -> None:
        if self.docs is None:
            self.sample_tokens += doc_tokens(content, self.count_tokens)
            self.sample_docs += 1
            if self.sample_docs == EST_SAMPLE_DOCS:
                self.docs = math.ceil(self.tokens * EST_SAMPLE_DOCS / self.sample_tokens)

    def met(self, n_docs: int) -> bool:
        if self.docs is not None:
            return n_docs >= self.docs
        return self.sample_tokens >= self.tokens


def _simple_enough(content: Content, comp: Component, simple_vocab: set[str]) -> bool:
    if comp.simplicity_min is None:
        return True
    return simplicity(content_text(content), simple_vocab) >= comp.simplicity_min


def _collect_source(
    comp: Component,
    scale: str,
    goal: _Goal,
    fetch: Fetch,
    simple_vocab: set[str],
    stats: dict,
) -> list[Content]:
    """Documents from one source, in row order, until the goal is met.

    Full scale stops when the source is exhausted. Tiny scale starts another pass over the rows
    instead (fixture files are short), unless a whole pass accepted no document, which no further
    pass would change.
    """
    spec = SOURCES[comp.name]
    docs: list[Content] = []
    while not goal.met(len(docs)):
        accepted = 0
        rows = fetch(spec, MAX_ROWS_PER_PASS[scale])
        try:
            for row in rows:
                stats["rows_read"] += 1
                for piece in row_to_content(spec.id, row):
                    content = prepare_content(piece)
                    if content is None or not _simple_enough(content, comp, simple_vocab):
                        continue
                    docs.append(content)
                    goal.add(content)
                    accepted += 1
                    if goal.met(len(docs)):
                        break
                if goal.met(len(docs)):
                    break
        finally:
            close = getattr(rows, "close", None)
            if close is not None:
                close()
        stats["passes"] += 1
        if scale == "full" or accepted == 0:
            break
    return docs


def _generated(comp: Component, kb: KB, rng: np.random.Generator, n: int) -> list[tuple]:
    """``n`` documents of a generator, prepared, each with its topic (None for "other")."""
    out: list[tuple[Content, int | None]] = []
    for doc in GENERATORS[comp.name](kb, rng, n) if n > 0 else []:
        content = prepare_content(doc.text if doc.kind == "plain" else list(doc.turns or []))
        if content is None:
            continue
        if doc.topic not in TOPICS:
            raise ValueError(f"generator {comp.name!r} gave unknown topic {doc.topic!r}")
        out.append((content, None if doc.topic == "other" else TOPICS.index(doc.topic)))
    return out


def _collect_generator(
    comp: Component,
    scale: str,
    quota: float,
    kb: KB,
    rng: np.random.Generator,
    count_tokens: Callable[[str], int],
) -> list[tuple[Content, int | None]]:
    sample_rng, docs_rng = (np.random.default_rng(s) for s in rng.integers(2**63, size=2))
    if scale == "tiny":
        return _generated(comp, kb, docs_rng, int(quota))
    sample = _generated(comp, kb, sample_rng, EST_SAMPLE_DOCS)
    mean = sum(doc_tokens(content, count_tokens) for content, _ in sample) / max(1, len(sample))
    return _generated(comp, kb, docs_rng, math.ceil(quota / mean) if mean else 0)


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
    docs: list[tuple[Content, int | None]], rate: float, rng: np.random.Generator
) -> list[AssembledDoc]:
    """Typos in ``rate`` of the documents; a chat gets them in each of its turns and stays a chat."""
    quota = int(rate * len(docs) + 0.5)
    chosen = set(rng.permutation(len(docs))[:quota].tolist())
    out: list[AssembledDoc] = []
    for i, (content, topic) in enumerate(docs):
        new = content
        if i in chosen:
            low, high = TYPO_RATE_RANGE
            letter_rate = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            if isinstance(content, str):
                new = add_typos(content, rng, letter_rate)
            else:
                new = [(role, add_typos(text, rng, letter_rate)) for role, text in content]
        out.append(AssembledDoc(new, _TYPO if new != content else _NONE, False, topic))
    return out


def apply_noise(
    docs: list[tuple[Content, int | None]],
    rates: NoiseRates,
    rng: np.random.Generator,
    kb: KB | None = None,
    plan: FalseFactPlan | None = None,
    dataset_id: str = "",
) -> list[AssembledDoc]:
    """Dirty a dataset's documents (``(content, topic)`` pairs) and tag what was done.

    Plain documents go through ``inject_noise``; each duplicate copy it appends records the
    document it was copied from (``copy_of``) and keeps that document's topic. A spam document
    replaced its original, so it loses the original's topic. A dataset with chats may only have
    typo noise (a chat with typos keeps its turns); any other kind raises ``ValueError``.
    """
    if any(not isinstance(content, str) for content, _ in docs):
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
        return _typo_noise(docs, rates.typo, rng)

    texts = [content for content, _ in docs]
    noisy = inject_noise(texts, rates, rng, kb, plan)
    out = [
        AssembledDoc(
            nd.text, nd.noise_kind, nd.false_fact, None if nd.noise_kind == _SPAM else topic
        )
        for nd, (_, topic) in zip(noisy, docs)
    ]
    copies = noisy[len(docs) :]
    untouched = [i for i, nd in enumerate(noisy[: len(docs)]) if nd.noise_kind == _NONE]
    sources = source_of_copies(texts, [nd.text for nd in copies], untouched) if copies else []
    for nd, source in zip(copies, sources):
        topic = None if source is None else docs[source][1]
        out.append(AssembledDoc(nd.text, nd.noise_kind, nd.false_fact, topic, source))
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

    ``count_tokens`` counts the tokens of a text for the full-scale targets (the frozen tokenizer
    when there is one). ``stats``, when given, receives what each component contributed.
    """
    if dataset_id not in RECIPES:
        raise ValueError(f"unknown dataset {dataset_id!r}; expected one of {list(DATASET_IDS)}")
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {list(SCALES)}")
    recipe = RECIPES[dataset_id]
    validate_recipe(recipe)
    shares = [comp.share for comp in recipe.components]
    if scale == "tiny":
        quotas: list[float] = list(split_count(SCALES["tiny"]["docs_per_dataset"], shares))
    else:
        target = SCALES["full"]["target_tokens"][dataset_id]
        quotas = [target * share for share in shares]
    seeds = rng.integers(2**63, size=len(recipe.components) + 1)

    docs: list[tuple[Content, int | None]] = []
    components: list[dict] = []
    for comp, quota, seed in zip(recipe.components, quotas, seeds[:-1]):
        info = {"kind": comp.kind, "name": comp.name, "share": comp.share}
        if comp.simplicity_min is not None:
            info["simplicity_min"] = comp.simplicity_min
        info["target_docs" if scale == "tiny" else "target_tokens"] = quota
        if comp.kind == "source":
            info |= {"rows_read": 0, "passes": 0}
            goal = (
                _Goal(docs=int(quota))
                if scale == "tiny"
                else _Goal(tokens=quota, count_tokens=count_tokens)
            )
            found = _collect_source(comp, scale, goal, fetch, simple_vocab, info)
            docs += [(content, None) for content in found]
        else:
            found = _collect_generator(
                comp, scale, quota, kb, np.random.default_rng(seed), count_tokens
            )
            docs += found
        info["docs"] = len(found)
        components.append(info)
    if stats is not None:
        stats["components"] = components
    return apply_noise(docs, recipe.noise, np.random.default_rng(seeds[-1]), kb, plan, dataset_id)
