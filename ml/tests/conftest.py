import re
from pathlib import Path

import pytest
import torch

from airace_ml.infer.lm import TorchLM
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.tokenizer import Tok, train_tokenizer

FIXTURE_TEXT_DIR = Path(__file__).parent / "fixtures" / "text"
FIXTURE_TEXT_FILES = ("stories.txt", "chats.txt", "code.txt", "facts.txt", "unicode.txt")


@pytest.fixture(scope="session")
def fixture_texts() -> list[str]:
    """Hand-authored fixture corpus: one string per blank-line-separated paragraph."""
    texts: list[str] = []
    for name in FIXTURE_TEXT_FILES:
        raw = (FIXTURE_TEXT_DIR / name).read_text(encoding="utf-8")
        texts.extend(p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip())
    return texts


@pytest.fixture(scope="session")
def tiny_tok(tmp_path_factory, fixture_texts) -> Tok:
    """A small (vocab 512) tokenizer trained on the fixture corpus, shared by the session."""
    out = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    return train_tokenizer(fixture_texts, out, vocab_size=512)


@pytest.fixture
def tiny_lm(tiny_tok) -> TorchLM:
    """A seeded, untrained 2-layer / d_model 64 / ctx 64 language model on the CPU."""
    torch.manual_seed(0)
    model = Transformer(ModelShape(2, 64, 64), tiny_tok.vocab_size)
    return TorchLM(model, tiny_tok, torch.device("cpu"))
