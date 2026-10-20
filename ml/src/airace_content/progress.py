"""Progress lines of a long build.

A full build reads for an hour or more, and with no word from it a stalled stream, a rate-limit
loop and a healthy read look the same. The build and the assembly take an optional ``Progress``
callback (one line of text per call); the command line passes one that prints to stderr, and
everything else stays silent. A line never changes what is built.
"""

import time
from collections.abc import Callable

Progress = Callable[[str], None]

EVERY_ROWS = 20_000  # a source reports every this many rows it has read ...
EVERY_SECONDS = 30.0  # ... or when this long has passed since its last line (big rows are slow)


def silent(message: str) -> None:
    """The default ``Progress``: says nothing."""


def _now() -> float:
    return time.monotonic()


class Pace:
    """Tells a loop over rows when to report: every ``EVERY_ROWS`` rows, or ``EVERY_SECONDS`` after
    the last report (or the start), whichever comes first."""

    def __init__(self) -> None:
        self.rows = 0
        self._last = _now()

    def tick(self) -> bool:
        """Count one more row; whether to report now."""
        self.rows += 1
        now = _now()
        if self.rows % EVERY_ROWS == 0 or now - self._last >= EVERY_SECONDS:
            self._last = now
            return True
        return False
