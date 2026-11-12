"""Moving checkpoints through the server's signed blob URLs (one ``.tar`` per model).

Plain ``urllib``: the URLs are short-lived and already carry their permission, so nothing else
is needed. A downloaded tar is extracted with strict path checks, since its bytes come over the
network.
"""

import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

CHUNK = 1 << 20
UPLOAD_ATTEMPTS = 3
# A bigger file is uploaded in parts of this size: the free tunnel friends play through refuses
# a single upload over 100 MB, and a big brain's files are larger (the game takes parts up to
# 64 MB: ``apps/server/src/blobs.ts``).
PART_BYTES = 32 << 20
TIMEOUT_S = 60


class UnsafeTarError(ValueError):
    """A tar member that would land outside the destination, or is not a plain file or folder."""


def _check_member(m: tarfile.TarInfo) -> None:
    path = PurePosixPath(m.name)
    if path.is_absolute() or ".." in path.parts or "\\" in m.name or ":" in m.name:
        raise UnsafeTarError(f"unsafe path in checkpoint tar: {m.name!r}")
    if not (m.isfile() or m.isdir()):
        raise UnsafeTarError(f"checkpoint tar may hold only files and folders: {m.name!r}")


def download_tar(url: str, dest_dir: Path) -> Path:
    """Downloads the tar at ``url`` and extracts it into ``dest_dir`` (created; must be empty)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.iterdir()):
        raise FileExistsError(f"{dest_dir} is not empty")
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "download.tar"
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(tar_path, "wb") as f:
            shutil.copyfileobj(r, f, CHUNK)
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            for m in members:
                _check_member(m)
            # The data filter repeats the checks above inside the standard library.
            tar.extractall(dest_dir, members=members, filter="data")
    return dest_dir


def make_tar(src_dir: Path, tar_path: Path, exclude: tuple[str, ...] = ()) -> Path:
    """Tars ``src_dir``'s contents (paths relative to it, sorted), leaving out top-level ``exclude``."""
    with tarfile.open(tar_path, "w") as tar:
        for path in sorted(src_dir.rglob("*")):
            rel = path.relative_to(src_dir)
            if rel.parts[0] in exclude:
                continue
            tar.add(path, arcname=rel.as_posix(), recursive=False)
    return tar_path


def upload_file(
    path: Path, url: str, attempts: int = UPLOAD_ATTEMPTS, part_bytes: int = PART_BYTES
) -> None:
    """PUTs ``path`` to ``url``, retrying network errors and server errors with a short wait.

    A file over ``part_bytes`` goes in parts of that size, in order (``&part=i&parts=n`` on the
    same signed address); the game joins them and the blob appears when the last one is in. If
    the game lost track of the parts (it restarted: 409), they are all sent again.
    """
    size = path.stat().st_size
    if size <= part_bytes:
        _put(path, 0, size, url, attempts)
        return
    parts = -(-size // part_bytes)
    joiner = "&" if "?" in url else "?"
    for attempt in range(1, attempts + 1):
        try:
            for i in range(parts):
                start = i * part_bytes
                length = min(part_bytes, size - start)
                _put(path, start, length, f"{url}{joiner}part={i}&parts={parts}", attempts)
            return
        except urllib.error.HTTPError as e:
            if e.code != 409 or attempt == attempts:
                raise


def _put(path: Path, start: int, length: int, url: str, attempts: int) -> None:
    """PUTs ``length`` bytes of ``path`` from ``start`` to ``url``."""
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                method="PUT",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(data)),
                },
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                r.read()
            return
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == attempts:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(min(2**attempt, 10))


def upload_dir_tar(src_dir: Path, url: str, exclude: tuple[str, ...] = ()) -> None:
    """Tars ``src_dir`` and uploads it to ``url``."""
    with tempfile.TemporaryDirectory() as tmp:
        upload_file(make_tar(src_dir, Path(tmp) / "upload.tar", exclude), url)
