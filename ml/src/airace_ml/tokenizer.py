"""Frozen byte-level BPE tokenizer (``tok-v1``) shared by every model, dataset and benchmark.

Design rules:

* Byte-level BPE with **no normalizer**, so ``decode(encode(text)) == text`` exactly for any
  Unicode string (accents, emoji, CJK, tabs, CRLF).
* The 16 special tokens occupy IDs 0-15 in a fixed order and enter token streams **only by ID**
  (see :func:`encode_doc` and :func:`encode_chat`). ``Tok.encode`` treats special-token strings
  such as ``<|end|>`` as plain text, so a player (or a dataset) can never inject control tokens.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|end|>",
    "<|user|>",
    "<|ai|>",
    "<|sep|>",
    "<|sys|>",
    "<|r0|>",
    "<|r1|>",
    "<|r2|>",
    "<|r3|>",
    "<|r4|>",
    "<|r5|>",
    "<|r6|>",
    "<|r7|>",
    "<|r8|>",
)
VOCAB_SIZE = 4096

Role = Literal["user", "ai"]


class Tok:
    """Thin wrapper over a HF ``tokenizers.Tokenizer`` that enforces the frozen token layout."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        # Special-token strings in text are plain text; they only enter as IDs. This flag is a
        # runtime property (not stored in tokenizer.json), so it is set on every construction.
        tokenizer.encode_special_tokens = True
        self._tk = tokenizer
        for i, token in enumerate(SPECIAL_TOKENS):
            if tokenizer.token_to_id(token) != i:
                raise ValueError(f"special token {token!r} must have id {i}")
        self.vocab_size: int = tokenizer.get_vocab_size()
        self.pad_id: int = 0
        self.bos_id: int = 1
        self.end_id: int = 2
        self.user_id: int = 3
        self.ai_id: int = 4

    @classmethod
    def load(cls, path: Path) -> "Tok":
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs. Never produces special-token IDs (0-15 stay reserved)."""
        return self._tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self._tk.decode(list(ids), skip_special_tokens=skip_special)

    def id_of(self, token: str) -> int:
        token_id = self._tk.token_to_id(token)
        if token_id is None:
            raise KeyError(token)
        return token_id


def train_tokenizer(texts: Iterable[str], out_path: Path, vocab_size: int = VOCAB_SIZE) -> Tok:
    """Train a byte-level BPE on ``texts``, save it to ``out_path`` and return it."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return Tok(tokenizer)


def encode_doc(tok: Tok, text: str) -> list[int]:
    """A plain document: ``<|bos|>`` followed by the text."""
    return [tok.bos_id, *tok.encode(text)]


def encode_chat(
    tok: Tok,
    turns: Sequence[tuple[Role, str]],
    add_generation_prompt: bool = False,
) -> list[int]:
    """``<|bos|>`` then ``[<|user|>|<|ai|>] text <|end|>`` per turn.

    With ``add_generation_prompt`` a trailing ``<|ai|>`` asks the model to write the next reply.
    """
    role_ids = {"user": tok.user_id, "ai": tok.ai_id}
    ids = [tok.bos_id]
    for role, text in turns:
        if role not in role_ids:
            raise ValueError(f"unknown chat role {role!r}; expected 'user' or 'ai'")
        ids.append(role_ids[role])
        ids.extend(tok.encode(text))
        ids.append(tok.end_id)
    if add_generation_prompt:
        ids.append(tok.ai_id)
    return ids
