"""`airace-ml fetch-data`: a helper downloads the game's data from the host and checks it."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from airace_ml.cli import main
from airace_ml.data.identity import data_identity
from airace_ml.worker.fetch import IDENTITY_CHANGED, FetchError, fetch_data, game_address

KEY = "user-1.secret"


@pytest.fixture
def host(tiny_data_root):
    """The game's download routes over the tiny data root: a manifest and its files, ranges
    and the player's key included. `gets` lists every file download (with its Range header)."""
    files = {}
    for path in sorted(p for p in tiny_data_root.rglob("*") if p.is_file()):
        rel = path.relative_to(tiny_data_root).as_posix()
        if rel.split("/")[0] in ("corpus", "tokenizer"):
            files[rel] = path.read_bytes()
    state = {
        "identity": data_identity(tiny_data_root),
        "files": files,
        "gets": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != f"Bearer {KEY}":
                return self._send(401, b'{"error": "Please sign in again."}')
            if self.path == "/helper/data/manifest.json":
                manifest = {
                    "dataIdentity": state["identity"],
                    "files": [
                        {"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                        for p, b in state["files"].items()
                    ],
                }
                return self._send(200, json.dumps(manifest).encode())
            rel = self.path.removeprefix("/helper/data/")
            body = state["files"].get(rel)
            if body is None:
                return self._send(404, b"")
            rng = self.headers.get("Range")
            state["gets"].append((rel, rng))
            if rng:
                start = int(rng.removeprefix("bytes=").split("-")[0])
                return self._send(206, body[start:])
            return self._send(200, body)

        def _send(self, code, body):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield state
    httpd.shutdown()


def _fetch(host, root: Path) -> list[str]:
    said: list[str] = []
    fetch_data(host["url"], KEY, root, say=said.append)
    return said


def test_a_fresh_folder_gets_everything_and_matches_the_game(host, tmp_path):
    root = tmp_path / "data"
    said = _fetch(host, root)
    assert data_identity(root) == host["identity"]
    assert len(host["gets"]) == len(host["files"])
    assert any("game's data" in s for s in said)


def test_a_second_run_downloads_nothing(host, tmp_path):
    root = tmp_path / "data"
    _fetch(host, root)
    host["gets"].clear()
    said = _fetch(host, root)
    assert host["gets"] == []
    assert any("up to date" in s for s in said)


def test_a_cut_short_download_carries_on_from_where_it_stopped(host, tmp_path):
    root = tmp_path / "data"
    _fetch(host, root)
    big = max(host["files"], key=lambda p: len(host["files"][p]))
    full = host["files"][big]
    (root / big).write_bytes(full[: len(full) // 2])
    host["gets"].clear()
    _fetch(host, root)
    assert host["gets"] == [(big, f"bytes={len(full) // 2}-")]
    assert (root / big).read_bytes() == full


def test_a_damaged_file_is_downloaded_again(host, tmp_path):
    root = tmp_path / "data"
    _fetch(host, root)
    name = next(iter(host["files"]))
    damaged = bytearray(host["files"][name])
    damaged[0] ^= 0xFF
    (root / name).write_bytes(bytes(damaged))
    host["gets"].clear()
    _fetch(host, root)
    assert host["gets"] == [(name, None)]
    assert (root / name).read_bytes() == host["files"][name]


def test_a_file_outside_the_game_data_is_refused(host, tmp_path):
    host["files"]["../evil.txt"] = b"x"
    with pytest.raises(FetchError, match="isn't valid"):
        _fetch(host, tmp_path / "data")
    assert not (tmp_path / "evil.txt").exists()


def test_the_cli_says_when_the_game_data_changed(host, tmp_path, capsys):
    host["identity"] = "something-else"
    code = main(
        ["fetch-data", "--server", host["url"], "--key", KEY, "--data-root", str(tmp_path / "d")]
    )
    assert code == 3
    assert IDENTITY_CHANGED in capsys.readouterr().err


def test_the_cli_takes_a_game_link_too(host, tmp_path):
    link = f"airace-helper://connect?url={host['url']}&key={KEY}"
    assert main(["fetch-data", "--server", link, "--data-root", str(tmp_path / "d")]) == 0


def test_game_address():
    assert game_address("https://a-b.trycloudflare.com/") == ("https://a-b.trycloudflare.com", None)
    assert game_address("airace-helper://connect?url=https://x.example&key=k") == (
        "https://x.example",
        "k",
    )
    with pytest.raises(ValueError):
        game_address("ftp://nope")
