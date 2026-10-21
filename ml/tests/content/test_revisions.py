"""F29: every source is read at one commit of its Hub repository, and the manifest says which."""

import json
import re
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import HfHubHTTPError

from airace_content import sources
from airace_content.build import BuildError, build_corpus
from airace_content.cli import main
from airace_content.manifest import make_manifest, read_manifest
from airace_content.sources import (
    SOURCES,
    DatasetsUnavailable,
    SourceSpec,
    SourceUnavailable,
    fixture_fetch,
    hf_fetch,
    resolve_revision,
)
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import corpus_dir

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
HEAD = "1f2e3d4c5b6a79880011223344556677deadbeef"
ROWS = [{"question": f"Q{i}", "answer": f"a {i}\n#### {i}"} for i in range(6)]
# The sources are pinned (ADR 0004); the lookup of a head is tested on an unpinned copy.
GSM8K = replace(SOURCES["gsm8k"], revision=None)


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/openai/gsm8k")
    return HfHubHTTPError(f"HTTP {status}", response=httpx.Response(status, request=request))


@dataclass
class FakeHub:
    """A fake ``datasets`` module and a fake ``HfApi`` over one set of rows."""

    rows: list[dict] = field(default_factory=lambda: list(ROWS))
    head: str | None = HEAD
    info_errors: list[BaseException] = field(default_factory=list)  # raised by the first calls
    stream_errors: list[int] = field(default_factory=list)  # the open numbers that drop at row 3
    loads: list[dict] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    info_kwargs: list[dict] = field(default_factory=list)  # the keywords of each lookup

    def load_dataset(self, path, config, *, split, streaming, **kwargs):
        self.loads.append({"path": path, "config": config, **kwargs})
        drop = len(self.loads) in self.stream_errors
        return self._stream(drop)

    def _stream(self, drop: bool):
        for i, row in enumerate(self.rows):
            if drop and i == 3:
                raise ConnectionError("Server Disconnected")
            yield row

    def api(self):
        hub = self

        class FakeApi:
            def dataset_info(self, repo_id, *, timeout=None, **kwargs):
                hub.infos.append(repo_id)
                hub.info_kwargs.append({"timeout": timeout, **kwargs})
                if hub.info_errors:
                    raise hub.info_errors.pop(0)
                return types.SimpleNamespace(sha=hub.head)

        return FakeApi


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(sources, "_pause", recorded.append)
    return recorded


def install(monkeypatch, hub: FakeHub) -> FakeHub:
    fake = types.ModuleType("datasets")
    fake.load_dataset = hub.load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(huggingface_hub, "HfApi", hub.api())
    return hub


# --- the spec, the fetch ---------------------------------------------------------------------


def test_every_source_is_pinned_and_a_spec_declares_none_by_default():
    assert all(re.fullmatch("[0-9a-f]{40}", spec.revision or "") for spec in SOURCES.values())
    assert SourceSpec("x", "o/x", None, "train", "mit", "h").revision is None
    spec = SourceSpec(
        "x", "o/x", None, "train", "mit", "https://huggingface.co/datasets/o/x", "abc"
    )
    assert spec.revision == "abc"


def test_a_declared_revision_is_passed_on_and_nothing_is_looked_up(monkeypatch, waits):
    hub = install(monkeypatch, FakeHub())
    pinned = replace(GSM8K, revision="cafe1234")
    assert list(hf_fetch(pinned, 3)) == ROWS[:3]
    assert [call["revision"] for call in hub.loads] == ["cafe1234"] and hub.infos == []


def test_the_head_is_looked_up_once_and_used_for_the_whole_read(monkeypatch, waits):
    hub = install(monkeypatch, FakeHub(stream_errors=[1, 2]))
    assert list(hf_fetch(GSM8K, 6)) == ROWS  # dropped twice, resumed twice
    assert hub.infos == ["openai/gsm8k"]
    assert [call["revision"] for call in hub.loads] == [HEAD, HEAD, HEAD]


def test_resolve_revision(monkeypatch, waits):
    hub = install(monkeypatch, FakeHub())
    assert resolve_revision(GSM8K) == HEAD
    assert resolve_revision(replace(GSM8K, revision="abc")) == "abc"
    assert hub.infos == ["openai/gsm8k"]
    install(monkeypatch, FakeHub(head=None))
    assert resolve_revision(GSM8K) is None  # no commit to name: the head, as before


def test_the_head_lookup_cannot_hang_for_ever(monkeypatch, waits):
    """The hub's metadata call has no timeout of its own: the lookup passes one, every time."""
    assert 0 < sources.INFO_TIMEOUT_S <= 300
    hub = install(monkeypatch, FakeHub(info_errors=[httpx.ConnectError("no route")]))
    assert resolve_revision(GSM8K) == HEAD  # the second try, after a failed one
    assert hub.info_kwargs == [{"timeout": sources.INFO_TIMEOUT_S}] * 2
    hub = install(monkeypatch, FakeHub())
    assert list(hf_fetch(GSM8K, 2)) == ROWS[:2]  # the lookup a read makes
    assert hub.info_kwargs == [{"timeout": sources.INFO_TIMEOUT_S}]


def test_a_lookup_without_the_content_extra_fails_before_the_network(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)  # importing it fails, installed or not
    asked = []

    class FakeApi:
        def dataset_info(self, repo_id, **kwargs):
            asked.append(repo_id)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(DatasetsUnavailable, match="python -m uv sync --extra content"):
        resolve_revision(GSM8K)
    assert asked == []


def test_a_lookup_that_fails_for_a_moment_is_retried(monkeypatch, waits, capsys):
    hub = install(
        monkeypatch, FakeHub(info_errors=[_http_error(503), httpx.ConnectError("no route")])
    )
    assert resolve_revision(GSM8K) == HEAD
    assert len(hub.infos) == 3 and waits == [30, 60]
    assert len(capsys.readouterr().err.splitlines()) == 2


def test_a_lookup_that_cannot_work_is_not_retried(monkeypatch, waits):
    hub = install(monkeypatch, FakeHub(info_errors=[_http_error(404)]))
    with pytest.raises(HfHubHTTPError):
        resolve_revision(GSM8K)
    assert len(hub.infos) == 1 and waits == []


def test_a_lookup_that_never_works_gives_up(monkeypatch, waits):
    install(monkeypatch, FakeHub(info_errors=[_http_error(502)] * 20))
    with pytest.raises(SourceUnavailable, match="gsm8k"):
        resolve_revision(GSM8K)
    assert len(waits) == sources.RETRY_ATTEMPTS - 1


# --- the build and the manifest ----------------------------------------------------------------


class Spy:
    """A fetch over the fixtures that notes the revision of every spec it is asked for."""

    def __init__(self):
        self.inner = fixture_fetch(FIX)
        self.seen: list[tuple[str, str | None]] = []

    def __call__(self, spec, max_rows):
        self.seen.append((spec.id, spec.revision))
        return self.inner(spec, max_rows)


@pytest.fixture
def unpinned(monkeypatch):
    """The sources as they were before they were pinned: a build looks each head up."""
    for sid, spec in list(SOURCES.items()):
        monkeypatch.setitem(SOURCES, sid, replace(spec, revision=None))


def _pin(looked_up: list[str] | None = None):
    def pin(spec):
        if looked_up is not None:
            looked_up.append(spec.id)
        return f"{spec.id}-rev"

    return pin


def test_a_build_reads_every_source_at_its_pinned_revision_and_records_it(tmp_path, unpinned):
    spy, looked_up = Spy(), []
    root = tmp_path / "data"
    lines: list[str] = []
    build_corpus("tiny", root, spy, seed=0, pin=_pin(looked_up), progress=lines.append)
    assert sorted(looked_up) == sorted(SOURCES)  # once per source, however often it is read
    assert all(revision == f"{sid}-rev" for sid, revision in spy.seen) and spy.seen
    assert any(line.startswith("pinning fineweb at fineweb-rev") for line in lines)
    manifest = read_manifest(corpus_dir(root))
    assert {s["id"]: s["revision"] for s in manifest["sources"]} == {
        sid: f"{sid}-rev" for sid in SOURCES
    }


def test_a_build_without_pins_records_no_revision(tmp_path, unpinned):
    spy = Spy()
    build_corpus("tiny", tmp_path / "data", spy, seed=0)
    assert all(revision is None for _, revision in spy.seen)
    manifest = read_manifest(corpus_dir(tmp_path / "data"))
    assert all(s["revision"] is None for s in manifest["sources"])


def test_a_declared_revision_is_read_and_recorded_without_a_pin(tmp_path, monkeypatch):
    monkeypatch.setitem(SOURCES, "gsm8k", replace(SOURCES["gsm8k"], revision="declared"))
    spy = Spy()
    build_corpus("tiny", tmp_path / "data", spy, seed=0)
    assert {rev for sid, rev in spy.seen if sid == "gsm8k"} == {"declared"}
    manifest = read_manifest(corpus_dir(tmp_path / "data"))
    assert {s["id"]: s["revision"] for s in manifest["sources"]}["gsm8k"] == "declared"


def test_a_partial_rebuild_keeps_the_revisions_of_the_sources_it_does_not_read(tmp_path, unpinned):
    root = tmp_path / "data"
    build_corpus("tiny", root, fixture_fetch(FIX), seed=0, pin=_pin())
    build_corpus(
        "tiny", root, fixture_fetch(FIX), seed=0, datasets=["code"], pin=lambda s: f"{s.id}-new"
    )
    revisions = {s["id"]: s["revision"] for s in read_manifest(corpus_dir(root))["sources"]}
    assert revisions["mbpp"] == "mbpp-new"  # read again, at its new head
    assert all(rev == f"{sid}-rev" for sid, rev in revisions.items() if sid != "mbpp")


def test_make_manifest_takes_the_revision_actually_used(unpinned):
    entries = {"code": {"components": [{"kind": "source", "name": "mbpp"}]}}
    sources_ = make_manifest(entries, None, {"mbpp": "abc"})["sources"]
    assert sources_[0]["id"] == "mbpp" and sources_[0]["revision"] == "abc"
    assert make_manifest(entries)["sources"][0]["revision"] is None


# --- the command line, through the fake hub ----------------------------------------------------


def test_the_command_line_pins_every_source_and_records_it(monkeypatch, tmp_path, waits, unpinned):
    by_repo = {(s.hf_path, s.hf_config): s for s in SOURCES.values()}
    fixtures = fixture_fetch(FIX)
    loads: list[dict] = []

    def load_dataset(path, config, *, split, streaming, revision=None):
        loads.append({"path": path, "config": config, "revision": revision})
        return iter(list(fixtures(by_repo[(path, config)], 10**9)))

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    infos: list[str] = []

    class FakeApi:
        def dataset_info(self, repo_id, *, timeout=None, **kwargs):
            infos.append(repo_id)
            return types.SimpleNamespace(sha=f"sha-of-{repo_id.replace('/', '-')}")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    root = tmp_path / "data"
    assert main(["build", "--scale", "tiny", "--out", str(root)]) == 0
    assert sorted(set(infos)) == sorted({s.hf_path for s in SOURCES.values()})
    assert len(infos) == len(SOURCES)  # one lookup per source, not per read
    assert loads and all(
        load["revision"] == f"sha-of-{load['path'].replace('/', '-')}" for load in loads
    )
    manifest = json.loads((corpus_dir(root) / "manifest.json").read_text(encoding="utf-8"))
    used = {s["id"]: s["revision"] for s in manifest["sources"]}
    assert set(used) == set(SOURCES) and all(
        used[s.id] == f"sha-of-{s.hf_path.replace('/', '-')}" for s in SOURCES.values()
    )
    assert set(manifest["datasets"]) == set(DATASET_IDS)


def test_a_source_that_cannot_be_looked_up_fails_in_plain_words(tmp_path, unpinned):
    def pin(spec):
        raise _http_error(401)

    with pytest.raises(BuildError, match="'tinystories'.*cannot be read.*HTTP 401"):
        build_corpus("tiny", tmp_path / "data", fixture_fetch(FIX), seed=0, pin=pin)
    assert not (tmp_path / "data").exists()


def test_a_lookup_that_hits_a_network_problem_is_left_for_the_command_line(tmp_path, unpinned):
    def pin(spec):
        raise SourceUnavailable("network problem while looking up the current version of x")

    with pytest.raises(SourceUnavailable):
        build_corpus("tiny", tmp_path / "data", fixture_fetch(FIX), seed=0, pin=pin)
