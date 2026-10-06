"""Model shape: the three numbers a player's model grows along, and the rules they obey."""

from __future__ import annotations

from dataclasses import dataclass

from airace_ml.tokenizer import VOCAB_SIZE

HEAD_DIM = 32
MAX_PARAMS = 85_000_000

_N_LAYER_RANGE = (1, 24)
_D_MODEL_RANGE = (64, 1024)
_CTX_LEN_RANGE = (64, 1024)
_CTX_STEP = 64
_FFN_STEP = 64


class ShapeError(ValueError):
    """A model shape breaks one of the global shape constraints."""


@dataclass(frozen=True)
class ModelShape:
    n_layer: int
    d_model: int
    ctx_len: int

    @property
    def n_head(self) -> int:
        return self.d_model // HEAD_DIM

    @property
    def ffn_hidden(self) -> int:
        # 64 * ceil(8 * d_model / 3 / 64), in exact integer arithmetic.
        return _FFN_STEP * ((8 * self.d_model + 3 * _FFN_STEP - 1) // (3 * _FFN_STEP))

    def param_count(self, vocab_size: int = VOCAB_SIZE) -> int:
        """Unique trainable parameters (the output head is tied to the embedding)."""
        d, f = self.d_model, self.ffn_hidden
        per_block = 4 * d * d + 3 * d * f + 2 * d  # q/k/v/o, gate/up/down, two norms
        return vocab_size * d + self.n_layer * per_block + d  # embedding, blocks, final norm

    def validate(self, vocab_size: int = VOCAB_SIZE) -> None:
        fields = (("n_layer", self.n_layer), ("d_model", self.d_model), ("ctx_len", self.ctx_len))
        for name, value in fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ShapeError(f"{name} must be an integer, got {value!r}")
        lo, hi = _N_LAYER_RANGE
        if not lo <= self.n_layer <= hi:
            raise ShapeError(f"n_layer must be in [{lo}, {hi}], got {self.n_layer}")
        lo, hi = _D_MODEL_RANGE
        if not lo <= self.d_model <= hi or self.d_model % HEAD_DIM:
            raise ShapeError(
                f"d_model must be a multiple of {HEAD_DIM} in [{lo}, {hi}], got {self.d_model}"
            )
        lo, hi = _CTX_LEN_RANGE
        if not lo <= self.ctx_len <= hi or self.ctx_len % _CTX_STEP:
            raise ShapeError(
                f"ctx_len must be a multiple of {_CTX_STEP} in [{lo}, {hi}], got {self.ctx_len}"
            )
        params = self.param_count(vocab_size)
        if params > MAX_PARAMS:
            raise ShapeError(f"model has {params:,} parameters, over the {MAX_PARAMS:,} limit")

    def to_dict(self) -> dict:
        return {"n_layer": self.n_layer, "d_model": self.d_model, "ctx_len": self.ctx_len}

    @classmethod
    def from_dict(cls, d: dict) -> ModelShape:
        return cls(n_layer=d["n_layer"], d_model=d["d_model"], ctx_len=d["ctx_len"])
