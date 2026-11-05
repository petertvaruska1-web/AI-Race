"""The game's data for a helper on a friend's PC (Stage 2): ``airace-ml fetch-data``.

The host lists its data with checksums (``/helper/data/manifest.json``). Each file whose checksum
differs here is downloaded again; a file cut short carries on from where it stopped (a range
request), and every file is checked once it is whole. At the end the data's identity must be
the game's, or the helper would train on other data than the host's.
"""

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

from airace_ml.data.identity import data_identity

DATA_DIRS = ("corpus", "tokenizer", "judge", "novelty", "rivals")
IDENTITY_CHANGED = "The game's data changed; run the helper setup again."
TIMEOUT_S = 60
_CHUNK = 1 << 20


class FetchError(Exception):
    """The data couldn't be fetched or doesn't match the game."""


class IdentityChanged(FetchError):
    """Everything downloaded, but it isn't the game's data."""


def game_address(link: str) -> tuple[str, str | None]:
    """The game's web address and the helper key in a link: an ``https://``/``http://`` address
    (no key), or an ``airace-helper://connect?url=…&key=…`` game link."""
    parsed = urlparse(link)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return link.rstrip("/"), None
    if parsed.scheme == "airace-helper" and parsed.netloc == "connect":
        query = parse_qs(parsed.query)
        url = (query.get("url") or [""])[0]
        if urlparse(url).scheme in ("http", "https"):
            return url.rstrip("/"), (query.get("key") or [None])[0]
    raise ValueError(f"that isn't the game's address or a game link: {link!r}")


def _get(url: str, key: str, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", **(headers or {})})
    return urllib.request.urlopen(req, timeout=TIMEOUT_S)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local(root: Path, rel: str) -> Path:
    """Where a manifest file goes; only inside the game's data folders."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or "\\" in rel or ":" in rel or not p.parts:
        raise FetchError(f"the game's file list isn't valid: {rel!r}")
    if p.parts[0] not in DATA_DIRS:
        raise FetchError(f"the game's file list isn't valid: {rel!r}")
    return root.joinpath(*p.parts)


def _download(url: str, key: str, path: Path, size: int) -> None:
    """Downloads into ``path``, carrying on after the bytes already there when it is shorter."""
    have = path.stat().st_size if path.is_file() else 0
    if have >= size:
        have = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Range": f"bytes={have}-"} if have else {}
    with _get(url, key, headers) as r:
        resumed = have > 0 and r.status == 206
        with open(path, "ab" if resumed else "wb") as f:
            while chunk := r.read(_CHUNK):
                f.write(chunk)


def fetch_data(server: str, key: str, root: Path, *, say: Callable[[str], None] = print) -> None:
    """Brings ``root`` up to the game's data at ``server`` (see the module docstring).

    :raises FetchError: the game can't be reached, a file won't check out, or (``IdentityChanged``)
        the data isn't the game's."""
    base, _ = game_address(server)
    root = Path(root)
    try:
        with _get(f"{base}/helper/data/manifest.json", key) as r:
            manifest = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise FetchError("the game didn't accept this helper's key") from e
        raise FetchError(f"the game's data list couldn't be read ({e.code})") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise FetchError(f"the game couldn't be reached at {base} ({e})") from e

    files = manifest.get("files") or []
    plan = [(f, _local(root, f["path"])) for f in files]
    total = sum(int(f["size"]) for f in files)
    say(f"Checking the game's data ({len(files)} files, {total // 2**20:,} MB)...")
    todo = [
        (f, p)
        for f, p in plan
        if not (p.is_file() and p.stat().st_size == f["size"] and _sha256(p) == f["sha256"])
    ]
    if not todo:
        say("The game's data is up to date.")
    need = sum(int(f["size"]) for f, _ in todo)
    done = 0
    for k, (f, path) in enumerate(todo, start=1):
        say(
            f"Downloading the game's data: file {k} of {len(todo)} "
            f"({done // 2**20:,} of {need // 2**20:,} MB)"
        )
        url = f"{base}/helper/data/{f['path']}"
        try:
            _download(url, key, path, int(f["size"]))
            if _sha256(path) != f["sha256"]:
                # A resumed file that doesn't check out is downloaded again from the start.
                path.unlink()
                _download(url, key, path, int(f["size"]))
        except (urllib.error.URLError, OSError) as e:
            raise FetchError(f"{f['path']} couldn't be downloaded ({e}); run this again") from e
        if _sha256(path) != f["sha256"]:
            raise FetchError(f"{f['path']} arrived damaged; run this again")
        done += int(f["size"])
    expected = manifest.get("dataIdentity")
    if expected:
        try:
            ours = data_identity(root)
        except FileNotFoundError as e:
            raise IdentityChanged(IDENTITY_CHANGED) from e
        if ours != expected:
            raise IdentityChanged(IDENTITY_CHANGED)
    if todo:
        say("The game's data is ready.")
