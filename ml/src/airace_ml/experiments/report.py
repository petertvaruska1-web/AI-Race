"""The feasibility gate's report: Markdown a person reads to judge the gate's verdict.

It holds a summary of the seven criteria, each criterion's measurements, sample transcripts for
human review, the four differentiation models' benchmark scores side by side, what each model's
personality is measured to be, every run the gate trained (and whether it completed), and an
"Iterations" section that later attempts fill in.

Model text is arbitrary (an untrained model writes anything), so every table cell is escaped:
Markdown and HTML punctuation is backslash-escaped, line breaks become ``<br>``, and transcript
text is cut to :data:`TRANSCRIPT_CHARS` characters.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from airace_ml.evals.suite import CATEGORIES
from airace_ml.experiments.gate import GateOutcome, stopped_prefix

MAX_EXCHANGES = 10  # transcript exchanges shown per criterion
TRANSCRIPT_CHARS = 300  # longest prompt or reply shown
CELL_CHARS = 400  # longest other cell
_ESCAPED = re.compile(r"([\\`*_\[\]<>|#~])")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_TRANSCRIPT_TITLES = {
    "G2": "G2: the first model's replies to probe prompts",
    "G3": "G3: each contrasting mix's model on the same prompts",
}
ITERATIONS_PLACEHOLDER = (
    "_None yet. Each attempt to fix a failing criterion is recorded here: the hypothesis, the "
    "change and the result._"
)


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def cell(value: object, limit: int = CELL_CHARS) -> str:
    """``value`` as the text of one Markdown table cell: cut to ``limit`` characters, Markdown
    and HTML punctuation escaped, line breaks as ``<br>`` and control characters dropped."""
    text = value if isinstance(value, str) else _format(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _CONTROL.sub("", _cut(text, limit))
    return _ESCAPED.sub(r"\\\1", text).replace("\n", "<br>")


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}" if math.isfinite(value) else "n/a"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format(v) for v in value) if value else "none"
    return str(value)


def _near(value: object, target: object) -> str:
    """``value`` with the usual 4 significant digits, or as many more as it takes to see on which
    side of ``target`` it lies (300.04 must not read as its 300 target)."""
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (value, target)):
        return _format(value)
    if not (math.isfinite(value) and math.isfinite(target)):
        return _format(value)
    side = (value > target) - (value < target)
    for digits in range(4, 18):
        text = f"{value:.{digits}g}"
        if (float(text) > target) - (float(text) < target) == side:
            return text
    return repr(value)


def _flatten(data: Mapping, prefix: str = "") -> Iterable[tuple[str, object]]:
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _flatten(value, f"{name}.")
        else:
            yield name, value


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _summary(outcome: GateOutcome) -> list[str]:
    passed = sum(c.passed for c in outcome.criteria)
    lines = [f"**{passed} of {len(outcome.criteria)} criteria passed.**", ""]
    # A stopped-run prefix gets its own room, so it never cuts the detail it opens.
    rows = [
        (
            cell(c.id),
            cell(c.title),
            "PASS" if c.passed else "FAIL",
            cell(c.detail, CELL_CHARS + len(stopped_prefix(c.unfinished_runs))),
        )
        for c in outcome.criteria
    ]
    return lines + _table(("ID", "Criterion", "Result", "Detail"), rows)


def _criteria(outcome: GateOutcome) -> list[str]:
    lines = ["## Measurements"]
    for c in outcome.criteria:
        lines += ["", f"### {cell(c.id)} {cell(c.title)}: {'PASS' if c.passed else 'FAIL'}", ""]
        lines.append(cell(c.detail, limit=4 * CELL_CHARS))
        if c.unfinished_runs:
            stopped = _stopped(c.unfinished_runs, outcome.raw)
            lines += ["", "**Cannot pass:** " + cell(stopped, limit=4 * CELL_CHARS)]
        if c.data:
            lines += ["", *_data_table(c.data)]
    return lines


def _stopped(names: Sequence[str], raw: Mapping) -> str:
    """Which runs did not complete and how far they got, as one sentence."""
    runs = raw.get("runs") or {}
    said = []
    for name in names:
        run = runs.get(name) or {}
        if run:
            said.append(
                f"{name} did not complete ({_format(run.get('status'))} after "
                f"{_format(run.get('steps'))} of {_format(run.get('planned_steps'))} steps)"
            )
        else:
            said.append(f"{name} did not complete")
    return "; ".join(said) + ", so what this criterion compares did not get its full compute."


def _data_table(data: Mapping) -> list[str]:
    """One row per measure; or, when every measure is a group of named values (one per mix,
    say), one row per group with a column per name."""
    if all(isinstance(v, Mapping) for v in data.values()):
        names = list(dict.fromkeys(name for group in data.values() for name in group))
        rows = [
            [cell(key), *(cell(group.get(name)) for name in names)] for key, group in data.items()
        ]
        return _table(["", *(cell(name) for name in names)], rows)
    flat = dict(_flatten(data))
    rows = ((cell(k), cell(_near(v, flat.get(f"{k}_target")))) for k, v in flat.items())
    return _table(("Measure", "Value"), rows)


def _runs(outcome: GateOutcome) -> list[str]:
    runs = outcome.raw.get("runs") or {}
    if not runs:
        return []
    lines = [
        "## Runs",
        "",
        "Every model the gate trained. A run that did not complete fails each criterion that uses it.",
        "",
    ]
    rows = []
    for name, run in runs.items():
        steps = f"{_format(run.get('steps'))}/{_format(run.get('planned_steps'))}"
        rows.append(
            (
                cell(name),
                cell(run.get("status")),
                cell(steps),
                cell(run.get("wall_seconds")),
                cell(run.get("heldout_loss")),
                cell(run.get("reused")),
            )
        )
    header = ("Run", "Status", "Steps", "Wall time (s)", "Held-out loss", "Reused")
    return lines + _table(header, rows)


def _transcripts(outcome: GateOutcome) -> list[str]:
    if not outcome.transcripts:
        return []
    lines = ["## Transcripts"]
    flags = outcome.raw.get("legibility_flags")
    for key, exchanges in outcome.transcripts.items():
        shown = list(exchanges)[:MAX_EXCHANGES]
        title = _TRANSCRIPT_TITLES.get(key, key)
        lines += ["", f"### {cell(title)}", ""]
        if len(exchanges) > len(shown):
            lines += [f"The first {len(shown)} of {len(exchanges)} exchanges.", ""]
        judged = key == "G2" and isinstance(flags, list) and len(flags) == len(exchanges)
        header = ("#", "Prompt", "Reply", "Well-formed") if judged else ("#", "Prompt", "Reply")
        rows = []
        for i, (prompt, reply) in enumerate(shown):
            row = [str(i + 1), cell(prompt, TRANSCRIPT_CHARS)]
            row.append(cell(reply, TRANSCRIPT_CHARS) if reply.strip() else "_(nothing)_")
            if judged:
                row.append("yes" if flags[i] else "no")
            rows.append(row)
        lines += _table(header, rows)
    return lines


def _benchmarks(outcome: GateOutcome) -> list[str]:
    models = outcome.raw.get("differentiation_models") or {}
    bench = outcome.raw.get("bench") or {}
    columns = [(target, name) for target, name in models.items() if name in bench]
    if not columns:
        return []
    lines = [
        "## Differentiation models: benchmark scores",
        "",
        "0 is no better than guessing, 100 is perfect.",
        "",
    ]
    header = ["Category", *(cell(f"{target}-heavy ({name})") for target, name in columns)]
    rows = [
        [cell(category), *(cell(bench[name].get(category)) for _, name in columns)]
        for category in CATEGORIES
    ]
    return lines + _table(header, rows)


def _personalities(outcome: GateOutcome) -> list[str]:
    described = outcome.raw.get("describe") or {}
    if not described:
        return []
    lines = [
        "## Personalities",
        "",
        (
            "What stands out in each model's measured fingerprint, compared with all the gate's "
            "models."
        ),
        "",
    ]
    rows = [
        (cell(name), cell(", ".join(words) if words else "nothing unusual"))
        for name, words in described.items()
    ]
    return lines + _table(("Model", "Described as"), rows)


def _setup(raw: Mapping) -> list[str]:
    lines = []
    if raw.get("quick"):
        lines += [
            (
                "> **Quick run.** Tiny models on a smoke-test budget: this checks that the gate "
                "runs end to end, and its results say nothing about feasibility."
            ),
            "",
        ]
    if raw.get("seeds") is not None or raw.get("device") is not None:
        seeds = ", ".join(str(s) for s in raw.get("seeds") or []) or "n/a"
        lines += [f"Seeds: {cell(seeds)}. Device: {cell(raw.get('device'))}.", ""]
    return lines


def write_gate_report(outcome: GateOutcome, path: Path) -> None:
    """Write ``outcome`` to ``path`` as Markdown (UTF-8), creating its folder if needed."""
    raw = outcome.raw or {}
    lines = ["# M1 feasibility gate", "", *_setup(raw), "## Summary", "", *_summary(outcome)]
    for section in (_criteria, _runs, _transcripts, _benchmarks, _personalities):
        part = section(outcome)
        if part:
            lines += ["", *part]
    lines += ["", "## Iterations", "", ITERATIONS_PLACEHOLDER, ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
