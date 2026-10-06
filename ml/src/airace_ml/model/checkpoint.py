"""Checkpoint = ``model.safetensors`` (weights) + ``meta.json`` (lineage and shape)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_model, save_model

from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer

CHECKPOINT_FORMAT = 1
WEIGHTS_NAME = "model.safetensors"
META_NAME = "meta.json"


@dataclass
class CheckpointMeta:
    tokenizer: str
    shape: ModelShape
    lineage_id: str
    version_id: str
    parent_version_id: str | None
    tokens_trained_total: int
    last_mixture: dict[str, float]
    runs: list[dict]
    format: int = CHECKPOINT_FORMAT

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": self.format,
                "tokenizer": self.tokenizer,
                "shape": self.shape.to_dict(),
                "lineage_id": self.lineage_id,
                "version_id": self.version_id,
                "parent_version_id": self.parent_version_id,
                "tokens_trained_total": self.tokens_trained_total,
                "last_mixture": self.last_mixture,
                "runs": self.runs,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> CheckpointMeta:
        d = json.loads(s)
        if d["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format {d['format']!r}")
        return cls(
            tokenizer=d["tokenizer"],
            shape=ModelShape.from_dict(d["shape"]),
            lineage_id=d["lineage_id"],
            version_id=d["version_id"],
            parent_version_id=d["parent_version_id"],
            tokens_trained_total=d["tokens_trained_total"],
            last_mixture=d["last_mixture"],
            runs=d["runs"],
            format=d["format"],
        )


def save_checkpoint(model: Transformer, meta: CheckpointMeta, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_dir / WEIGHTS_NAME))
    (out_dir / META_NAME).write_text(meta.to_json(), encoding="utf-8")


def _vocab_size(weights: Path) -> int:
    """Vocabulary size read from the file; the tied weight is stored under one of two names."""
    with safe_open(str(weights), framework="pt") as f:
        names = set(f.keys())
        for name in ("tok_emb.weight", "lm_head.weight"):
            if name in names:
                return f.get_slice(name).get_shape()[0]
    raise ValueError(f"{weights} holds neither tok_emb.weight nor lm_head.weight")


def load_checkpoint(
    ckpt_dir: Path, device: torch.device | str = "cpu"
) -> tuple[Transformer, CheckpointMeta]:
    ckpt_dir = Path(ckpt_dir)
    meta = CheckpointMeta.from_json((ckpt_dir / META_NAME).read_text(encoding="utf-8"))
    weights = ckpt_dir / WEIGHTS_NAME
    model = Transformer(meta.shape, _vocab_size(weights))
    load_model(model, str(weights))
    return model.to(device), meta
