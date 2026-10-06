"""Compute device selection."""

import os

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return CUDA when available, unless `prefer` or `AIRACE_DEVICE` asks for CPU."""
    choice = (prefer or os.environ.get("AIRACE_DEVICE") or "").lower()
    if choice != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
