import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from airace_ml.data.corpus import DATASET_IDS, DocTags, write_corpus
from airace_ml.infer.lm import TorchLM
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok, encode_chat, encode_doc, train_tokenizer

FIXTURE_TEXT_DIR = Path(__file__).parent / "fixtures" / "text"
FIXTURE_SOURCES_DIR = Path(__file__).parent / "fixtures" / "sources"
FIXTURE_TEXT_FILES = ("stories.txt", "chats.txt", "code.txt", "facts.txt", "unicode.txt")


def _file_paragraphs(name: str) -> list[str]:
    raw = (FIXTURE_TEXT_DIR / name).read_text(encoding="utf-8")
    return [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]


@pytest.fixture(scope="session")
def fixture_texts() -> list[str]:
    """Hand-authored fixture corpus: one string per blank-line-separated paragraph."""
    return [p for name in FIXTURE_TEXT_FILES for p in _file_paragraphs(name)]


@pytest.fixture(scope="session")
def tiny_tok_path(tmp_path_factory, fixture_texts) -> Path:
    """Where the session's small (vocab 512) tokenizer, trained on the fixture corpus, is saved."""
    out = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    train_tokenizer(fixture_texts, out, vocab_size=512)
    return out


@pytest.fixture(scope="session")
def tiny_tok(tiny_tok_path) -> Tok:
    """A small (vocab 512) tokenizer trained on the fixture corpus, shared by the session."""
    return Tok.load(tiny_tok_path)


# Fixture files that feed each dataset of `tiny_data_root`.
_DATASET_SOURCES = {
    "web": ("facts.txt", "unicode.txt", "stories.txt"),
    "books": ("stories.txt",),
    "educational": ("facts.txt",),
    "conversations": ("chats.txt",),
    "code": ("code.txt",),
    "reasoning": ("code.txt", "facts.txt"),
    "facts": ("facts.txt",),
    "creative": ("stories.txt", "unicode.txt"),
}
_TINY_DOCS_PER_DATASET = 50
_TINY_TOPICS = (0, 2, 5, 7)  # indices into TOPICS: animals, nature, science, technology


def _parse_chat(paragraph: str) -> list[tuple[str, str]]:
    """``User: ...`` / ``AI: ...`` lines to turns; a line with no prefix continues the last turn."""
    turns: list[tuple[str, str]] = []
    for line in paragraph.split("\n"):
        if line.startswith("User: "):
            turns.append(("user", line[len("User: ") :]))
        elif line.startswith("AI: "):
            turns.append(("ai", line[len("AI: ") :]))
        else:
            role, text = turns[-1]
            turns[-1] = (role, text + "\n" + line)
    return turns


def _tiny_tags(n: int, seed: int) -> DocTags:
    """Deterministic tags with a guaranteed spread of every kind (see `tiny_data_root`)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    heldout_ids = order[: max(1, n // 10)]
    cluster_ids = order[len(heldout_ids) : len(heldout_ids) + 9]  # three clusters of three
    false_ids = order[len(heldout_ids) + 9 : len(heldout_ids) + 13]
    heldout = np.zeros(n, bool)
    heldout[heldout_ids] = True
    false_fact = np.zeros(n, bool)
    false_fact[false_ids] = True
    dup_cluster = np.full(n, -1, np.int32)
    dup_canonical = np.zeros(n, bool)
    for c in range(3):
        members = cluster_ids[3 * c : 3 * c + 3]
        dup_cluster[members] = c
        dup_canonical[members[0]] = True
    quality = (rng.permutation(n) + rng.uniform(0, 1, n)) / n  # one document per quality band
    noise = np.where(quality < 0.4, rng.integers(1, 5, n), 0)
    noise[(dup_cluster >= 0) & ~dup_canonical] = 5  # "duplicate"
    noise[false_fact] = 6  # "false_fact"
    return DocTags(
        quality=quality.astype(np.float16),
        dup_cluster=dup_cluster,
        dup_canonical=dup_canonical,
        false_fact=false_fact,
        topic=np.array(_TINY_TOPICS, np.uint8)[rng.permutation(np.arange(n) % len(_TINY_TOPICS))],
        noise_kind=noise.astype(np.uint8),
        heldout=heldout,
        purchase_rank=(rng.permutation(n) / n).astype(np.float32),
    )


@pytest.fixture(scope="session")
def tiny_data_root(tmp_path_factory, tiny_tok, tiny_tok_path) -> Path:
    """A data root with the tiny tokenizer and all 8 corpora, built from the fixture texts.

    Each corpus has 50 documents (cycled from its source paragraphs; chats become `encode_chat`
    documents) and deterministic tags: qualities spread over [0, 1), three duplicate clusters
    (members share the canonical's tokens), four false facts, four topics, 10% held out and a
    random-permutation `purchase_rank`.
    """
    root = tmp_path_factory.mktemp("data_root")
    tokenizer_path(root).parent.mkdir(parents=True)
    shutil.copyfile(tiny_tok_path, tokenizer_path(root))
    n = _TINY_DOCS_PER_DATASET
    for k, ds in enumerate(DATASET_IDS):
        paragraphs = [p for f in _DATASET_SOURCES[ds] for p in _file_paragraphs(f)]
        rotated = [paragraphs[(i + 7 * k) % len(paragraphs)] for i in range(n)]
        if ds == "conversations":
            docs = [encode_chat(tiny_tok, _parse_chat(p)) for p in rotated]
        else:
            docs = [encode_doc(tiny_tok, p) for p in rotated]
        tags = _tiny_tags(n, seed=1000 + k)
        for c in range(3):  # a duplicate cluster repeats its canonical document verbatim
            canonical = int(np.flatnonzero((tags.dup_cluster == c) & tags.dup_canonical)[0])
            for i in np.flatnonzero(tags.dup_cluster == c):
                docs[i] = docs[canonical]
        write_corpus(corpus_dir(root) / ds, docs, tags, {"dataset": ds, "source": "test fixtures"})
    return root


@pytest.fixture(scope="session")
def gate_data_root(tmp_path_factory) -> Path:
    """Everything the feasibility gate reads, built for real at tiny scale: the corpora and
    tokenizer of ``airace-content build --scale tiny`` (from the source fixtures, with its
    ``known_vocab.txt`` and ``false_facts.json``), the novelty index and a reference judge.

    The judge's recipe is shrunk to 2 layers x 64 wide on a 64-token span, as the CLI's judge
    tests do, so it builds in seconds. Built only when a test asks for it (slow tests only).
    """
    from airace_content.build import build_corpus
    from airace_content.sources import fixture_fetch
    from airace_ml.evals import judge as judge_module
    from airace_ml.evals.novelty import build_novelty_index

    root = tmp_path_factory.mktemp("gate_data_root")
    build_corpus("tiny", root, fixture_fetch(FIXTURE_SOURCES_DIR), seed=0)
    build_novelty_index(root)
    real = judge_module.judge_train_config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            judge_module,
            "judge_train_config",
            lambda seed=1234: replace(real(seed), shape=ModelShape(2, 64, 64), batch_tokens=1024),
        )
        judge_module.build_judge(root, device="cpu", token_budget=1024 * 8)
    return root


@pytest.fixture
def tiny_lm(tiny_tok) -> TorchLM:
    """A seeded, untrained 2-layer / d_model 64 / ctx 64 language model on the CPU."""
    torch.manual_seed(0)
    model = Transformer(ModelShape(2, 64, 64), tiny_tok.vocab_size)
    return TorchLM(model, tiny_tok, torch.device("cpu"))
