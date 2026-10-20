"""Console output shared by the command-line tools (``airace-ml`` and ``airace-content``)."""

import sys


def use_utf8_output() -> None:
    """Make stdout and stderr UTF-8 and unable to fail on an unencodable character.

    Without this, printing a model's text, or a path with letters outside the legacy code page, on
    a console or a redirect that uses one (Windows cp1252) raises ``UnicodeEncodeError``. A stream
    without ``reconfigure`` (a test harness's capture object) is left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # a stream that has already been read or written cannot change; keep it
