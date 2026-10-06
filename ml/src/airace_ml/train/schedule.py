"""Warmup-stable-decay learning-rate schedule."""

from __future__ import annotations

from airace_ml.train.config import LearningStyle

DECAY_FRAC = 0.2  # the last fifth of the run is the decay phase
MIN_LR_RATIO = 0.1  # the decay ends at this fraction of the peak rate


def wsd_lr(step: int, total_steps: int, style: LearningStyle) -> float:
    """Learning rate at ``step`` (0-based) of a ``total_steps`` run.

    Linear warmup to ``style.peak_lr`` over ``warmup_frac * total_steps`` steps (at least one;
    step 0 is already above zero), flat at the peak, then a linear decay over the last
    ``DECAY_FRAC`` of the steps that ends at ``MIN_LR_RATIO * peak_lr`` on the final step.
    Steps past the end stay at that floor.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")
    if step < 0:
        raise ValueError(f"step must not be negative, got {step}")
    warmup = max(1, int(style.warmup_frac * total_steps))
    decay = max(1, int(DECAY_FRAC * total_steps))
    up = min(1.0, (step + 1) / warmup)
    decay_start = total_steps - decay
    if step < decay_start:
        down = 1.0
    else:
        progress = 1.0 if decay == 1 else min(1.0, (step - decay_start) / (decay - 1))
        down = 1.0 - progress * (1.0 - MIN_LR_RATIO)
    # Only a run of a step or two can have both phases on one step; the lower rate wins.
    return style.peak_lr * min(up, down)
