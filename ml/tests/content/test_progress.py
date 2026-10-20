"""F17: a long build says what it is doing, so a stall looks like one."""

import re
import shutil
from pathlib import Path

from airace_content import progress as progress_module
from airace_content.assemble import RECIPES, SCALES
from airace_content.build import build_corpus, sources_needed
from airace_content.cli import main
from airace_content.sources import SOURCES, fixture_fetch
from airace_ml.data.corpus import DATASET_IDS

FIX = Path(__file__).parents[1] / "fixtures" / "sources"
FETCH = fixture_fetch(FIX)


def _lines(**kwargs) -> list[str]:
    lines: list[str] = []
    root = kwargs.pop("root")
    build_corpus(kwargs.pop("scale", "tiny"), root, FETCH, seed=0, progress=lines.append, **kwargs)
    return lines


def _index(lines: list[str], pattern: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return i
    raise AssertionError(f"no line matches {pattern!r} after line {start}:\n" + "\n".join(lines))


def test_a_build_reports_each_stage_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(progress_module, "EVERY_ROWS", 40)
    lines = _lines(root=tmp_path / "data")
    # every source is checked first
    needed = sources_needed(DATASET_IDS, simple_vocab=True)
    checks = [i for i, line in enumerate(lines) if line.startswith("checking source ")]
    assert len(checks) == len(needed) == len(SOURCES)
    assert checks == list(range(len(needed)))
    # then the simple vocabulary, and each dataset assembled: a line when each source opens
    simple = _index(lines, r"^simple vocabulary: opening tinystories")
    assert simple > checks[-1]
    at = simple
    for ds in DATASET_IDS:
        at = _index(lines, rf"^{ds}: assembling", at)
        for comp in RECIPES[ds].components:
            if comp.kind == "source":
                at = _index(lines, rf"^{ds}: opening {comp.name} ", at)
        at = _index(lines, rf"^{ds}: [\d,]+ documents assembled", at)
    # the known vocabulary, then dedup and scoring of each dataset
    at = _index(lines, r"^known vocabulary", at)
    for ds in DATASET_IDS:
        at = _index(
            lines, rf"^{ds}: scoring and finding near-duplicates among [\d,]+ documents", at
        )
    # the tokenizer, then each corpus written, then the manifest
    at = _index(lines, r"^training the tokenizer on [\d.]+ MB of text", at)
    for ds in DATASET_IDS:
        at = _index(lines, rf"^{ds}: encoding and writing [\d,]+ documents", at)
    _index(lines, r"^writing the manifest", at)


def test_rows_are_reported_every_n_rows_with_what_was_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(progress_module, "EVERY_ROWS", 40)
    lines = _lines(root=tmp_path / "data")
    rows = [
        m
        for line in lines
        if (m := re.match(r"web: fineweb: ([\d,]+) rows read, ([\d,]+) docu", line))
    ]
    assert rows, "\n".join(lines)
    counts = [int(m.group(1).replace(",", "")) for m in rows]
    assert counts == sorted(counts) and all(c % 40 == 0 for c in counts)
    assert all(int(m.group(2).replace(",", "")) <= c for m, c in zip(rows, counts))


def test_a_full_build_reports_estimated_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(progress_module, "EVERY_ROWS", 10)
    for ds in DATASET_IDS:
        monkeypatch.setitem(SCALES["full"]["target_tokens"], ds, 30_000)
    lines = _lines(root=tmp_path / "data", scale="full")
    rows = [line for line in lines if re.match(r"web: fineweb: [\d,]+ rows read", line)]
    assert rows and all(re.search(r", ~[\d,]+ tokens$", line) for line in rows)


def test_a_slow_source_is_reported_by_the_clock_too(tmp_path, monkeypatch):
    """Rows that are big and slow (a whole book each) must not leave the log silent."""
    monkeypatch.setattr(progress_module, "EVERY_ROWS", 10**9)
    clock = iter(range(0, 10**6, 31))  # 31 s pass between any two readings
    monkeypatch.setattr(progress_module, "_now", lambda: next(clock))
    lines = _lines(root=tmp_path / "data")
    assert any(re.match(r"books: gutenberg: [\d,]+ rows read", line) for line in lines)


def test_a_source_that_ends_reports_what_it_gave(tmp_path):
    lines = _lines(root=tmp_path / "data")
    done = [line for line in lines if re.match(r"web: finished with fineweb: ", line)]
    assert len(done) == 1 and re.search(r"[\d,]+ rows read, [\d,]+ documents kept", done[0])


def test_a_build_without_a_callback_prints_nothing(tmp_path, capsys):
    build_corpus("tiny", tmp_path / "data", FETCH, seed=0)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_progress_does_not_change_the_corpus(tmp_path):
    quiet = build_corpus("tiny", tmp_path / "quiet", FETCH, seed=0)
    loud = build_corpus("tiny", tmp_path / "loud", FETCH, seed=0, progress=lambda line: None)
    assert quiet.datasets == loud.datasets
    for name in ("tokens.bin", "offsets.npy"):
        a = tmp_path / "quiet" / "corpus" / "v1" / "web" / name
        assert a.read_bytes() == (tmp_path / "loud" / "corpus" / "v1" / "web" / name).read_bytes()


def test_the_cli_puts_progress_on_stderr_and_the_summary_on_stdout(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(progress_module, "EVERY_ROWS", 40)
    argv = ["build", "--scale", "tiny", "--fixtures", str(FIX), "--out", str(tmp_path / "data")]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("corpus written to ")
    assert "checking source" not in captured.out and "rows read" not in captured.out
    err = captured.err.splitlines()
    assert err[0].startswith("checking source ") and err[-1] == "writing the manifest"
    assert any("rows read" in line for line in err)


def test_warnings_still_come_after_the_progress(tmp_path, capsys):
    fixtures = tmp_path / "fixtures"
    shutil.copytree(FIX, fixtures)
    (fixtures / "cosmo_openstax.jsonl").write_text("", encoding="utf-8")  # runs dry at once
    argv = ["build", "--scale", "tiny", "--fixtures", str(fixtures), "--out", str(tmp_path / "d")]
    assert main(argv) == 0
    err = capsys.readouterr().err.splitlines()
    warnings = [i for i, line in enumerate(err) if line.startswith("warning:")]
    assert len(warnings) == 1 and warnings[0] == len(err) - 1
    assert err[warnings[0]].startswith("warning: educational: cosmo_openstax ran out at 0 of 30 ")


def test_pace_reports_every_n_rows(monkeypatch):
    monkeypatch.setattr(progress_module, "EVERY_ROWS", 3)
    pace = progress_module.Pace()
    assert [pace.tick() for _ in range(7)] == [False, False, True, False, False, True, False]
