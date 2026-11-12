"""Keeps the computer awake while real work runs.

A PC that goes to sleep for idleness in the middle of a training run stops the run (a gate run
once lost hours this way, and a friend's helper would drop out of a shared world). While a
:func:`stay_awake` block runs, Windows is told that the system is needed, exactly as a video
player does: the screen may still turn off, the user can still put the PC to sleep by hand, and
nothing about the PC's power settings is changed. On other systems it does nothing.
"""

import contextlib
import logging
import sys
import threading
from collections.abc import Iterator

log = logging.getLogger(__name__)

# SetThreadExecutionState flags (winbase.h). The state belongs to the calling thread and lasts
# until that thread clears it or ends.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

_depth = threading.local()


def _set_state(flags: int) -> None:
    """Tells Windows what this thread needs of the system; nothing elsewhere."""
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


@contextlib.contextmanager
def stay_awake() -> Iterator[None]:
    """While the block runs in this thread, the computer does not go to sleep for idleness.

    Blocks may nest and several threads may each hold one: the request is made by the outermost
    block of a thread and withdrawn when that block ends, however it ends. Failing to make the
    request is never a reason to fail the work.
    """
    depth = getattr(_depth, "n", 0)
    _depth.n = depth + 1
    if depth == 0:
        _request(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        _depth.n = depth
        if depth == 0:
            _request(ES_CONTINUOUS)


def _request(flags: int) -> None:
    try:
        _set_state(flags)
    except Exception:  # an odd Windows build: the work matters more than the request
        log.debug("could not tell the system to stay awake", exc_info=True)
