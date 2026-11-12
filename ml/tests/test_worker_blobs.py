"""Uploads to the game's blob store: whole when small, in parts when big (the free tunnel friends
play through refuses a single upload over 100 MB)."""

import os
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from airace_ml.worker import blobs
from airace_ml.worker.blobs import upload_file


class Game:
    """The blob store's side of an upload, as the game server keeps it (``apps/server/src/blobs.ts``)."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.requests: list[tuple[str, int | None, int | None, int]] = []
        self.answers: list[int] = []  # status codes to give instead, one request each
        self._arriving: dict[str, tuple[int, int, bytes]] = {}

    def put(self, key: str, part: int | None, parts: int | None, body: bytes) -> int:
        self.requests.append((key, part, parts, len(body)))
        if self.answers:
            status = self.answers.pop(0)
            if status == 409:
                self._arriving.pop(key, None)  # it lost track of the parts (a restart)
            return status
        if part is None or parts is None:
            self.blobs[key] = body
            return 204
        nxt, total, held = self._arriving.get(key, (0, parts, b""))
        if part == 0:
            nxt, total, held = 0, parts, b""
        elif key not in self._arriving or total != parts or part != nxt:
            return 409
        held += body
        if part + 1 == parts:
            self._arriving.pop(key, None)
            self.blobs[key] = held
        else:
            self._arriving[key] = (part + 1, parts, held)
        return 204


@pytest.fixture
def game(monkeypatch):
    game = Game()

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            assert q["sig"] == ["s"]  # the signed address is kept whole
            body = self.rfile.read(int(self.headers["Content-Length"]))
            part = int(q["part"][0]) if "part" in q else None
            parts = int(q["parts"][0]) if "parts" in q else None
            self.send_response(game.put(url.path, part, parts, body))
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(blobs.time, "sleep", lambda s: None)
    game.url = f"http://127.0.0.1:{httpd.server_address[1]}/blobs/a.tar?exp=1&sig=s"
    yield game
    httpd.shutdown()


def _file(tmp_path, size: int):
    path = tmp_path / "upload.tar"
    path.write_bytes(os.urandom(size))
    return path


def test_a_small_file_goes_in_one_upload(game, tmp_path):
    path = _file(tmp_path, 900)
    upload_file(path, game.url, part_bytes=1000)
    assert game.requests == [("/blobs/a.tar", None, None, 900)]
    assert game.blobs["/blobs/a.tar"] == path.read_bytes()


def test_a_big_file_goes_in_parts_no_bigger_than_the_limit(game, tmp_path):
    path = _file(tmp_path, 2500)
    upload_file(path, game.url, part_bytes=1000)
    assert game.requests == [
        ("/blobs/a.tar", 0, 3, 1000),
        ("/blobs/a.tar", 1, 3, 1000),
        ("/blobs/a.tar", 2, 3, 500),
    ]
    assert game.blobs["/blobs/a.tar"] == path.read_bytes()


def test_a_file_of_exactly_the_limit_is_one_upload_and_one_byte_more_is_two(game, tmp_path):
    upload_file(_file(tmp_path, 1000), game.url, part_bytes=1000)
    assert [r[1:] for r in game.requests] == [(None, None, 1000)]
    game.requests.clear()
    path = _file(tmp_path, 1001)
    upload_file(path, game.url, part_bytes=1000)
    assert [r[1:] for r in game.requests] == [(0, 2, 1000), (1, 2, 1)]
    assert game.blobs["/blobs/a.tar"] == path.read_bytes()


def test_a_part_the_game_could_not_take_is_sent_again(game, tmp_path):
    path = _file(tmp_path, 2500)
    game.answers = [204, 503]  # part 0 is "taken" by the script, part 1 fails once
    game._arriving["/blobs/a.tar"] = (1, 3, path.read_bytes()[:1000])
    upload_file(path, game.url, part_bytes=1000)
    assert [r[1] for r in game.requests] == [0, 1, 1, 2]
    assert game.blobs["/blobs/a.tar"] == path.read_bytes()


def test_the_upload_starts_over_when_the_game_lost_track_of_the_parts(game, tmp_path):
    path = _file(tmp_path, 2500)
    game.answers = [204, 409]  # the game restarted between part 0 and part 1
    upload_file(path, game.url, part_bytes=1000)
    assert [r[1] for r in game.requests] == [0, 1, 0, 1, 2]
    assert game.blobs["/blobs/a.tar"] == path.read_bytes()


def test_an_upload_the_game_keeps_refusing_fails(game, tmp_path):
    game.answers = [403]
    with pytest.raises(urllib.error.HTTPError):
        upload_file(_file(tmp_path, 2500), game.url, part_bytes=1000)
    assert len(game.requests) == 1  # a refusal is final: asking again won't help
    game.answers = [409] * 10
    with pytest.raises(urllib.error.HTTPError):
        upload_file(_file(tmp_path, 2500), game.url, part_bytes=1000)
