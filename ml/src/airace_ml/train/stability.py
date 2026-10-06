"""Loss-spike detection: the signal that a training run has become unstable."""

from __future__ import annotations

import math


class SpikeDetector:
    """Flags a step whose loss jumps far above the recent trend.

    After ``warmup_steps`` a loss is a spike when it exceeds ``mean + z * std`` of an exponential
    moving average **and** ``ratio * mean``. A non-finite loss is always a spike. Spikes are not
    folded into the statistics, so they cannot raise the bar for the next step.
    """

    def __init__(
        self, warmup_steps: int, alpha: float = 0.02, z: float = 6.0, ratio: float = 1.3
    ) -> None:
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.z = z
        self.ratio = ratio
        self._mean = 0.0
        self._var = 0.0
        self._count = 0  # losses folded into the statistics so far

    def update(self, step: int, loss: float) -> bool:
        """Feed the loss of ``step``; returns True when it is a spike."""
        if not math.isfinite(loss):
            return True
        if step >= self.warmup_steps and self._count > 0:
            std = math.sqrt(self._var)
            if loss > self._mean + self.z * std and loss > self.ratio * self._mean:
                return True
        self._fold(loss)
        return False

    def _fold(self, loss: float) -> None:
        if self._count == 0:
            self._mean, self._var = loss, 0.0
        else:
            delta = loss - self._mean
            self._mean += self.alpha * delta
            self._var = (1.0 - self.alpha) * (self._var + self.alpha * delta * delta)
        self._count += 1

    def state(self) -> dict:
        """The moving statistics, JSON-safe, for :meth:`load_state` (a resumed run)."""
        return {"mean": self._mean, "var": self._var, "count": self._count}

    def load_state(self, d: dict) -> None:
        self._mean = float(d["mean"])
        self._var = float(d["var"])
        self._count = int(d["count"])
