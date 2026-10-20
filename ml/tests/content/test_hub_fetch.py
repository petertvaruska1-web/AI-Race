"""F11: the hour-long full build survives a flaky Hub (retry and exact resume inside ``hf_fetch``),
turns a hopeless network into a plain message, and checks every source before it reads any."""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

import airace_content.cli as cli_module
from airace_content import sources
from airace_content.assemble import RECIPES
from airace_content.build import BuildError, build_corpus
from airace_content.cli import main
from airace_content.manifest import KNOWN_VOCAB_FILE
from airace_content.sources import (
    FORMAT_CHECK_ROWS,
    SOURCES,
    SourceFormatError,
    SourceUnavailable,
    fixture_fetch,
    hf_fetch,
    is_transient_error,
)
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import corpus_dir, tokenizer_path

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
ROWS = [{"question": f"Q{i}", "answer": f"a {i}\n#### {i}"} for i in range(10)]
SPEC = SOURCES["gsm8k"]


def _http_error(status: int | None, cls=HfHubHTTPError) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/openai/gsm8k")
    return cls(f"HTTP {status}", response=httpx.Response(status, request=request))


class _NoResponse(HfHubHTTPError):
    def __init__(self):  # an error with no response at all (status None)
        Exception.__init__(self, "no response")
        self.response = None


@dataclass
class Plan:
    """What one ``load_dataset`` call does: raise ``open_error`` at once, or stream the rows and
    raise ``error`` when row ``fail_at`` is asked for."""

    open_error: BaseException | None = None
    fail_at: int | None = None
    error: BaseException | None = None


@dataclass
class FakeHub:
    rows: list[dict]
    plans: list[Plan] = field(default_factory=list)  # one per call; later calls run to the end
    forever: BaseException | None = None  # every call raises this at once
    calls: list[dict] = field(default_factory=list)

    def load_dataset(self, path, config, *, split, streaming, **kwargs):
        self.calls.append({"path": path, "config": config, "split": split, **kwargs})
        if self.forever is not None:
            raise self.forever
        plan = self.plans[len(self.calls) - 1] if len(self.calls) <= len(self.plans) else Plan()
        if plan.open_error is not None:
            raise plan.open_error
        return self._stream(plan)

    def _stream(self, plan: Plan):
        for i, row in enumerate(self.rows):
            if plan.fail_at == i:
                raise plan.error
            yield row


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """The pauses the retries ask for, without waiting for them."""
    recorded: list[float] = []
    monkeypatch.setattr(sources, "_pause", recorded.append)
    return recorded


def install(monkeypatch, hub: FakeHub) -> FakeHub:
    fake = types.ModuleType("datasets")
    fake.load_dataset = hub.load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return hub


# --- retry and exact resume --------------------------------------------------------------------


def test_a_stream_that_drops_resumes_at_the_next_row(monkeypatch, waits, capsys):
    hub = install(
        monkeypatch,
        FakeHub(
            ROWS,
            [
                Plan(fail_at=4, error=ConnectionError("Server Disconnected")),
                Plan(fail_at=7, error=TimeoutError("read timed out")),
            ],
        ),
    )
    assert list(hf_fetch(SPEC, 10)) == ROWS  # no row lost, none twice
    assert len(hub.calls) == 3 and waits == [30, 30]
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2 and all("gsm8k" in line and "30 s" in line for line in lines)
    assert "ConnectionError" in lines[0] and "TimeoutError" in lines[1]


def test_a_resumed_stream_still_stops_at_max_rows(monkeypatch, waits):
    install(monkeypatch, FakeHub(ROWS, [Plan(fail_at=4, error=ConnectionError("gone"))]))
    assert list(hf_fetch(SPEC, 6)) == ROWS[:6]


def test_a_failure_when_the_source_is_opened_is_retried(monkeypatch, waits):
    hub = install(monkeypatch, FakeHub(ROWS, [Plan(open_error=_http_error(503))]))
    assert list(hf_fetch(SPEC, 3)) == ROWS[:3]
    assert len(hub.calls) == 2 and waits == [30]


def test_failures_with_no_row_in_between_wait_longer_and_longer_then_give_up(
    monkeypatch, waits, capsys
):
    hub = install(monkeypatch, FakeHub(ROWS, forever=ConnectionError("Server Disconnected")))
    with pytest.raises(SourceUnavailable, match="gsm8k") as stopped:
        list(hf_fetch(SPEC, 10))
    assert isinstance(stopped.value.__cause__, ConnectionError)
    assert len(hub.calls) == sources.RETRY_ATTEMPTS == 8
    assert waits == [30, 60, 120, 240, 300, 300, 300]  # doubling, up to five minutes
    assert len(capsys.readouterr().err.splitlines()) == 7  # one line per retry


def test_a_row_between_failures_starts_the_count_again(monkeypatch, waits):
    plans = [Plan(fail_at=i, error=ConnectionError("blip")) for i in range(1, 10)]
    install(monkeypatch, FakeHub(ROWS, plans))
    assert list(hf_fetch(SPEC, 10)) == ROWS  # nine failures in all, never two in a row
    assert waits == [30] * 9


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Server Disconnected"),
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
        httpx.ConnectError("no route"),
        httpx.ReadTimeout("slow"),
        _NoResponse(),
        *[_http_error(status) for status in (408, 429, 500, 502, 503, 504)],
        RuntimeError("wrapped"),  # given a transient cause below
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(getattr(e, 'response', None), 'status_code', '')}",
)
def test_transient_errors_are_retried(error, monkeypatch, waits):
    if type(error) is RuntimeError:
        error.__cause__ = ConnectionError("Server Disconnected")
    assert is_transient_error(error)
    hub = install(monkeypatch, FakeHub(ROWS, [Plan(fail_at=2, error=error)]))
    assert list(hf_fetch(SPEC, 5)) == ROWS[:5]
    assert len(hub.calls) == 2 and waits == [30]


@pytest.mark.parametrize(
    "error",
    [
        _http_error(401),
        _http_error(403),
        _http_error(404),
        _http_error(400),
        FileNotFoundError("Dataset 'x' doesn't exist on the Hub or cannot be accessed."),
        ValueError("BuilderConfig 'nope' not found"),
        SourceFormatError("not in the expected format"),
        OfflineModeIsEnabled("HF_HUB_OFFLINE is set"),
        KeyError("text"),
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(getattr(e, 'response', None), 'status_code', '')}",
)
@pytest.mark.parametrize("when", ["open", "stream"])
def test_other_errors_are_never_retried(error, when, monkeypatch, waits, capsys):
    assert not is_transient_error(error)
    plan = Plan(open_error=error) if when == "open" else Plan(fail_at=2, error=error)
    hub = install(monkeypatch, FakeHub(ROWS, [plan]))
    with pytest.raises(type(error)) as raised:
        list(hf_fetch(SPEC, 5))
    assert raised.value is error
    assert len(hub.calls) == 1 and waits == []
    assert capsys.readouterr().err == ""


def test_a_wrapped_error_follows_its_cause():
    permanent = RuntimeError("could not load")
    permanent.__cause__ = _http_error(404)
    assert not is_transient_error(permanent)
    transient = OSError("could not load")
    transient.__cause__ = _http_error(502)
    assert is_transient_error(transient)


def test_closing_the_stream_stops_reading_and_never_retries(monkeypatch, waits):
    hub = install(monkeypatch, FakeHub(ROWS))
    rows = hf_fetch(SPEC, 10)
    assert next(rows) == ROWS[0]
    rows.close()
    assert len(hub.calls) == 1 and waits == []


def test_a_dataset_that_ends_early_ends_the_stream(monkeypatch, waits):
    install(monkeypatch, FakeHub(ROWS[:3]))
    assert list(hf_fetch(SPEC, 10)) == ROWS[:3]


# --- the command line --------------------------------------------------------------------------


def test_the_cli_turns_a_hopeless_network_into_a_plain_message(
    monkeypatch, waits, tmp_path, capsys
):
    install(monkeypatch, FakeHub([], forever=httpx.ConnectError("no route to host")))
    assert main(["build", "--scale", "tiny", "--out", str(tmp_path / "data")]) == 1
    err = capsys.readouterr().err
    last = err.strip().splitlines()[-1]
    assert last.startswith("error: network problem") and "run the same command again" in last
    assert "Traceback" not in err and "tinystories" in last
    assert not (tmp_path / "data").exists()


def test_the_cli_also_turns_a_bare_connection_error_into_that_message(
    monkeypatch, tmp_path, capsys
):
    def fetch(spec, max_rows):
        raise ConnectionError("Server Disconnected")
        yield

    monkeypatch.setattr(cli_module, "hf_fetch", fetch)
    assert main(["build", "--scale", "tiny", "--out", str(tmp_path)]) == 1
    assert "run the same command again" in capsys.readouterr().err


# --- every source is checked before any is read ------------------------------------------------


class Recording:
    """A fetch over the fixtures that notes every call, and can break one source."""

    def __init__(self, broken: str | None = None, error: Exception | None = None, rows=None):
        self.inner = fixture_fetch(FIX)
        self.calls: list[tuple[str, int]] = []
        self.broken, self.error, self.rows = broken, error, rows

    def __call__(self, spec, max_rows):
        self.calls.append((spec.id, max_rows))
        if spec.id == self.broken:
            if self.error is not None:
                raise self.error
            return iter(self.rows)
        return self.inner(spec, max_rows)


def _nothing_written(root: Path) -> bool:
    return not root.exists() or not any(p.is_file() for p in root.rglob("*"))


def test_a_late_source_in_the_wrong_format_fails_before_anything_is_read(tmp_path):
    root = tmp_path / "data"
    fetch = Recording("cosmo_stories", rows=[{"title": "x", "body": "y"}])
    with pytest.raises(SourceFormatError, match="cosmo_stories"):
        build_corpus("tiny", root, fetch, seed=0)
    assert len(fetch.calls) <= len(SOURCES)  # one look at each source, no assembly
    assert all(rows <= FORMAT_CHECK_ROWS for _, rows in fetch.calls)
    assert _nothing_written(root)


def test_a_late_source_that_cannot_be_opened_fails_in_plain_words(tmp_path):
    root = tmp_path / "data"
    fetch = Recording("cosmo_stories", error=ValueError("BuilderConfig 'stories' not found"))
    with pytest.raises(BuildError, match="cosmo_stories.*BuilderConfig 'stories' not found"):
        build_corpus("tiny", root, fetch, seed=0)
    assert all(rows <= FORMAT_CHECK_ROWS for _, rows in fetch.calls)
    assert _nothing_written(root)


def test_a_late_network_failure_still_reads_as_a_network_problem(tmp_path):
    fetch = Recording("simplewiki", error=ConnectionError("Server Disconnected"))
    with pytest.raises(ConnectionError):  # passed on as it is, for the command line to word
        build_corpus("tiny", tmp_path / "data", fetch, seed=0)


def test_every_source_a_full_build_needs_is_opened_once_before_it_reads_any(tmp_path):
    fetch = Recording()
    build_corpus("tiny", tmp_path / "data", fetch, seed=0)
    needed = {c.name for r in RECIPES.values() for c in r.components if c.kind == "source"}
    first = fetch.calls[: len(needed)]
    assert {sid for sid, _ in first} == needed and len(first) == len(needed)
    assert all(rows <= FORMAT_CHECK_ROWS for _, rows in first)
    assert fetch.calls[len(needed)][0] == "tinystories"  # then the real reading starts


def test_a_partial_rebuild_checks_only_the_sources_it_reads(tmp_path):
    root = tmp_path / "data"
    build_corpus("tiny", root, fixture_fetch(FIX), seed=0)
    fetch = Recording()
    build_corpus("tiny", root, fetch, seed=0, datasets=["code"])
    assert [sid for sid, rows in fetch.calls if rows <= FORMAT_CHECK_ROWS] == ["mbpp"]
    (corpus_dir(root) / KNOWN_VOCAB_FILE).unlink()  # the known vocabulary needs four datasets
    fetch = Recording()
    build_corpus("tiny", root, fetch, seed=0, datasets=["code"])
    checked = [sid for sid, rows in fetch.calls if rows <= FORMAT_CHECK_ROWS]
    assert checked == [
        sid
        for sid in SOURCES
        if sid
        in {
            c.name
            for ds in ("books", "educational", "facts", "creative", "code")
            for c in RECIPES[ds].components
            if c.kind == "source"
        }
    ]


def test_a_source_with_no_rows_is_left_to_the_build(tmp_path):
    """An empty source is not a format problem: the build says so when it runs dry."""
    root = tmp_path / "data"
    fetch = Recording("cosmo_openstax", rows=[])
    summary = build_corpus("tiny", root, fetch, seed=0)
    assert summary.datasets["educational"]["components"][2]["exhausted"]
    assert tokenizer_path(root).exists() and set(summary.datasets) == set(DATASET_IDS)
