"""Runs that move between graphics cards (Stage 2, Fair mode).

In a Fair-mode world a training run may continue on another player's card when its card goes
away. While it trains, the helper copies the trainer's resume state (taken between steps, so it
is consistent) and uploads it to the host every two minutes, in the background so training never
waits for the network. The next card downloads it and the trainer continues exactly where the
saved state stopped (the same final AI as an unbroken run).
"""

import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from airace_ml.train.trainer import copy_resume_state
from airace_ml.worker.blobs import download_tar, upload_dir_tar

log = logging.getLogger(__name__)

RESUME_UPLOAD_EVERY_S = 120.0
# Seconds between the trainer's saves of its resume state on this computer. By the clock, not
# by steps: 200 steps are 5 seconds of a small brain and over a minute of a big one.
RESUME_SAVE_EVERY_S = 30.0


class ResumeUploader:
    """Uploads a run's resume state to ``url`` at most every ``every_s`` seconds.

    ``saved(out_dir)`` is the trainer's ``on_resume_saved``: when an upload is due and none is
    under way, it copies the state aside and uploads the copy on a background thread. A failed
    upload is logged and tried again at the next save. ``close()`` waits for the last upload.
    """

    def __init__(
        self,
        url: str,
        every_s: float = RESUME_UPLOAD_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.every_s = every_s
        self.clock = clock
        self.last = clock()
        self.uploads = 0
        self._thread: threading.Thread | None = None

    def due(self) -> bool:
        return self.clock() - self.last >= self.every_s

    def saved(self, out_dir: Path) -> None:
        if not self.due() or (self._thread is not None and self._thread.is_alive()):
            return
        staging = Path(tempfile.mkdtemp(prefix="airace-resume-"))
        try:
            if not copy_resume_state(out_dir, staging / "state"):
                shutil.rmtree(staging, ignore_errors=True)
                return
        except OSError as e:
            log.warning("could not copy the training progress: %s", e)
            shutil.rmtree(staging, ignore_errors=True)
            return
        self.last = self.clock()
        self._thread = threading.Thread(target=self._upload, args=(staging,), daemon=True)
        self._thread.start()

    def _upload(self, staging: Path) -> None:
        try:
            upload_dir_tar(staging / "state", self.url)
            self.uploads += 1
        except Exception as e:  # noqa: BLE001 - the run goes on; the next save tries again
            log.warning("could not save the training progress to the game: %s", e)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join()


def restore_resume(url: str, out: Path) -> bool:
    """Puts the progress another card saved at ``url`` into ``out`` (which must be empty or
    absent). False, with ``out`` left empty, when it can't be had: the run then starts over."""
    try:
        download_tar(url, out)
        return True
    except Exception as e:  # noqa: BLE001 - starting over is always possible
        log.warning("could not get the saved training progress, starting over: %s", e)
        shutil.rmtree(out, ignore_errors=True)
        return False
