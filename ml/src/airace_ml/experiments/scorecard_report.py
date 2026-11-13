"""The scorecard's reports and its comparison with a saved baseline.

Kept apart from :mod:`~airace_ml.experiments.scorecard` and out of its digested sources
(:data:`~airace_ml.experiments.scorecard.CARD_SOURCES`): changing a column title, a sentence or a
tolerance here never measures a model again.

:func:`unlike` says what makes two cards incomparable (another yardstick, judge, novelty index,
row count or length, or scale; different engines and different training text still compare).
:func:`compare` lists what got worse than the baseline by more than its :data:`TOLERANCE`, a
measure that has no value any more and a model that stopped early this time; :func:`notes` says
what was not compared and why.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from airace_ml.experiments.runs import _write_json
from airace_ml.experiments.scorecard import FULL, LOSSES, MEASURES, RATES

LABELS: dict[str, str] = {
    "heldout": "held-out loss",
    "heldout_conversations": "held-out conversation loss",
    "yardstick_loss": "yardstick conversation loss",
    "bench_overall": "bench-v1 overall",
    "well_formed": "well-formed replies",
    "casual": "facts typed casually",
    "followups": "follow-ups answered",
    "identity_line": "knows its name and that it is an AI",
    "identity_no_line": "says it is an AI without its name",
    "person_life": "claims a human life in small talk",
    "cut_off": "replies cut off",
    "looped": "replies stopped in a loop",
}
COLUMNS: dict[str, str] = {
    "heldout": "Held-out loss",
    "heldout_conversations": "Held-out conv. loss",
    "yardstick_loss": "Yardstick loss",
    "bench_overall": "Bench",
    "well_formed": "Well-formed",
    "casual": "Casual facts",
    "followups": "Follow-ups",
    "identity_line": "Identity (named)",
    "identity_no_line": "Says it is an AI (unnamed)",
    "person_life": "Life claims",
    "cut_off": "Cut off",
    "looped": "Loops",
}
# How much worse than the baseline a measure may get before compare() reports it (rates as
# shares, bench in points, losses in nats per token).
TOLERANCE: dict[str, float] = {
    "well_formed": 0.04,
    "casual": 0.05,
    "followups": 0.05,
    "bench_overall": 2.0,
    "yardstick_loss": 0.03,
    "heldout": 0.02,
}
# Measured on the data root's own held-out text: comparable only on the same training text.
OWN_TEXT = frozenset({"heldout", "heldout_conversations"})


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _models(card: Mapping[str, Any]) -> Mapping[str, Any]:
    models = card.get("models")
    return models if isinstance(models, Mapping) else {}


def _same_text(card: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return card.get("data_identity") == baseline.get("data_identity")


# -- comparing ------------------------------------------------------------------------------------


def unlike(card: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """What makes ``baseline`` measured with another stick than ``card``, one phrase each; empty
    when the two compare. A baseline that does not say what it was measured with (saved before
    cards did) never compares."""
    before = baseline.get("measured_with")
    if not isinstance(before, Mapping):
        return [
            (
                "the baseline was saved before scorecards recorded what measured them; measure "
                "it again with this version"
            )
        ]
    now = card.get("measured_with") or {}
    tools, old_tools = now.get("instruments") or {}, before.get("instruments") or {}
    found = []
    if now.get("yardstick") != before.get("yardstick"):
        found.append("the yardstick data differs")
    if tools.get("judge") != old_tools.get("judge"):
        found.append("the judge differs")
    if tools.get("novelty") != old_tools.get("novelty"):
        found.append("the novelty index differs")
    if tools.get("label") != old_tools.get("label"):
        found.append(f"the instruments differ ({tools.get('label')} vs {old_tools.get('label')})")
    if now.get("yardstick_rows") != before.get("yardstick_rows"):
        found.append(
            f"the yardstick rows differ ({now.get('yardstick_rows')} here, "
            f"{before.get('yardstick_rows')} in the baseline)"
        )
    if now.get("scale") != before.get("scale"):
        found.append("the scale differs (the recipes or the probe sizes are not the same)")
    before_models = _models(baseline)
    for name, model in _models(card).items():
        if name in before_models:
            span, old_span = model.get("yardstick_span"), before_models[name].get("yardstick_span")
            if span != old_span:
                found.append(
                    f"{name}'s yardstick rows are {span} tokens long here and {old_span} in the "
                    f"baseline"
                )
    return found


def compare(card: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """One sentence for each thing on ``card`` that got worse than in ``baseline``: a model that
    completed there and stopped early now; a measure that had a value there and has none now; a
    measure of :data:`TOLERANCE` worse by more than its tolerance (a loss going up, anything else
    going down). Models the baseline lacks, measures it lacks and, on other training text, the
    own-text losses (:data:`OWN_TEXT`) are not compared (:func:`notes` says so). Call
    :func:`unlike` first: cards measured with another stick do not compare at all."""
    found = []
    same_text = _same_text(card, baseline)
    before_models = _models(baseline)
    for name, model in _models(card).items():
        if name not in before_models:
            continue
        old_model = before_models[name]
        if old_model.get("status") == "completed" and model.get("status") != "completed":
            found.append(
                f"{name}: its training stopped early this time ({model.get('status')}); it "
                f"completed in the baseline."
            )
        now, before = model.get("measures") or {}, old_model.get("measures") or {}
        for key in MEASURES:
            if key in OWN_TEXT and not same_text:
                continue
            old = _finite(before.get(key))
            if old is None:
                continue
            new = _finite(now.get(key))
            if new is None:
                found.append(
                    f"{name}: {LABELS[key]} has no value this time ({_shown(key, old)} in the "
                    f"baseline)."
                )
                continue
            tolerance = TOLERANCE.get(key)
            if tolerance is None:
                continue
            worse = new - old if key in LOSSES else old - new
            if worse > tolerance + 1e-9:
                found.append(
                    f"{name}: {LABELS[key]} {_shown(key, new)} against {_shown(key, old)} in the "
                    f"baseline, worse by {_difference(key, worse)} (allowed "
                    f"{_difference(key, tolerance)})."
                )
    return found


def notes(card: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """What :func:`compare` left out, one line each."""
    said = []
    if not _same_text(card, baseline):
        said.append(
            "The training text differs from the baseline's, so the held-out losses on it are not "
            "compared (the yardstick loss and everything else are)."
        )
    missing = [name for name in _models(card) if name not in _models(baseline)]
    if missing:
        said.append(f"Not in the baseline, so not compared: {', '.join(missing)}.")
    return said


# -- reports --------------------------------------------------------------------------------------


def _shown(key: str, value: object) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    if key in RATES:
        return f"{number:.0%}"
    if key in LOSSES:
        return f"{number:.3f}"
    return f"{number:.1f}"


def _difference(key: str, amount: float) -> str:
    if key in RATES:
        return f"{100 * amount:.0f} points"
    if key in LOSSES:
        return f"{amount:.3f}"
    return f"{amount:.1f} points"


def _markdown(card: Mapping[str, Any]) -> str:
    lines = [
        "# Scorecard",
        "",
        (
            f"Data root: {card['data_root']}; yardstick root: {card['yardstick_root']}; "
            f"engine {card['engine']}; {card['date']}"
        ),
        "",
    ]
    if card.get("scale") != FULL.name:
        lines += [f"Scale: {card.get('scale')} (shrunk recipes: a check of the setup only).", ""]
    clock = card.get("timing")
    if clock:
        when = "reused: the time of its own earlier training" if clock["reused"] else "trained now"
        lines += [
            (
                f"First AI ({clock['model']}) trained in {clock['seconds']:.1f} s on "
                f"{clock['device']} ({when})."
            ),
            "",
        ]
    heading = ["Model", "Recipe", *(COLUMNS[key] for key in MEASURES)]
    lines.append("| " + " | ".join(heading) + " |")
    lines.append("| " + " | ".join("---" for _ in heading) + " |")
    for name, model in card["models"].items():
        shown = name if model.get("status") == "completed" else f"{name} (stopped early)"
        measures = model.get("measures") or {}
        cells = [shown, model["recipe"], *(_shown(k, measures.get(k)) for k in MEASURES)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "Losses are per token (lower is better); bench-v1 is 0-100. Rates are shares of "
            "replies: well-formed, casual facts, follow-ups and identity are better higher; life "
            "claims, cut-off replies and loops better lower."
        ),
        "",
    ]
    return "\n".join(lines)


def write_reports(card: Mapping[str, Any], out: Path) -> None:
    """``scorecard.json`` (the whole card) and ``scorecard.md`` (one row per model, one column per
    measure) in ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "scorecard.json", card)
    (out / "scorecard.md").write_text(_markdown(card), encoding="utf-8")
