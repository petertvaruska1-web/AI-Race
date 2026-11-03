"""What an eval found that is worth telling the player: milestone signals and discoveries.

Both are measurements (R5). :func:`eval_signals` counts what the milestones need from one eval.
:func:`find_discoveries` compares this version's tag breakdown and traits with the version it
was trained from and with the world's other AIs, and keeps only differences large enough not to
be luck. Each finding carries one real question and the AI's reply to it, verbatim (R1).
"""

from statistics import fmean, pstdev
from typing import Any

from airace_ml.evals.creativity import STORY_PROMPTS
from airace_ml.evals.judge import is_well_formed
from airace_ml.evals.scoring import BenchReport, ItemResult
from airace_ml.evals.suite import build_suite
from airace_ml.personality.fingerprint import TRAIT_PHRASES, TRAITS, Fingerprint, load_probes
from airace_ml.skills.types import CheckItem, ExactItem

PERSONALITY = "personality"
TAG_FLOOR = 5.0
"""The smallest peer spread a tag's z-score divides by, in points: a few peers that happen to
agree must not make a small difference look like a discovery."""
TRAIT_UNIT = 10.0
"""A trait's change in peer spreads counts this many points when findings are ranked."""

CATEGORY_WORDS = {
    "language": "grammar",
    "reasoning": "reasoning",
    "pattern": "patterns",
    "knowledge": "facts",
    "coding": "coding",
    "creativity": "stories",
    "consistency": "steady facts",
    "instruction": "following instructions",
}

TAG_WORDS = {
    "fmt:story": "stories",
    "fmt:output": "predicting what code prints",
    "fmt:function": "writing working functions",
    "rel:animal_class": "animal groups",
    "rel:animal_legs": "how many legs animals have",
    "rel:baby_animal": "baby animal names",
    "rel:capital_of": "capital cities",
    "rel:continent_of": "continents",
    "rel:element_symbol": "chemical symbols",
    "rel:landmark_country": "famous landmarks",
    "rel:month_after": "the order of the months",
    "rel:shape_sides": "the sides of shapes",
    "rel:vehicle_travel": "how vehicles travel",
    "fam:agreement": "making words agree",
    "fam:article": "using a and an",
    "fam:plural": "plurals",
    "fam:tense": "verb tenses",
    "fam:word_order": "word order",
    "fam:compare": "comparing things",
    "fam:count": "counting",
    "fam:syllogism": "logic puzzles",
    "fam:word_problem": "word problems",
    "fam:countdown": "counting down",
    "fam:cycle": "repeating patterns",
    "fam:letters": "letter patterns",
    "fam:step": "number steps",
    "fam:call": "code that calls functions",
    "fam:if": "code with choices",
    "fam:list": "code with lists",
    "fam:loop": "code with loops",
    "fam:straight": "simple code",
    "fam:string": "code with text",
    "fam:all_caps": "writing in capitals",
    "fam:contains_any": "using the words asked for",
    "fam:list_n": "making lists of the right length",
    "fam:one_word": "one-word answers",
    "fam:repeat_word": "repeating a word on request",
    "fam:starts_with": "starting with the word asked for",
    "fam:yes_no": "yes-or-no answers",
}

_TOPIC_FRAMES = {
    "knowledge": "facts about {}",
    "consistency": "{} questions asked different ways",
    "instruction": "instructions about {}",
}

_TITLES = {
    "strength": "Surprisingly good at {about}",
    "weakness": "Struggles with {about}",
    "better": "Much better at {about} than {label}",
    "worse": "Worse at {about} than {label}",
    "changed": "More {about} than {label}",
}


def tag_words(key: str) -> str:
    """Plain words for a breakdown key ``"<category>/<tag>"`` (R7)."""
    category, _, tag = key.partition("/")
    if tag.startswith("topic:"):
        topic = tag.removeprefix("topic:").replace("_", " ")
        return _TOPIC_FRAMES.get(category, "{}").format(topic)
    if tag in TAG_WORDS:
        return TAG_WORDS[tag]
    value = tag.partition(":")[2] or tag
    return f"{CATEGORY_WORDS.get(category, category)} ({value.replace('_', ' ')})"


# -- signals ---------------------------------------------------------------------------------


def _items_by_id() -> dict[str, Any]:
    return {item.id: item for items in build_suite().items.values() for item in items}


def eval_signals(report: BenchReport, fingerprint: Fingerprint, judge: Any) -> dict[str, Any]:
    """What the milestones need from one eval.

    ``legibleReplies``: the fingerprint's replies that read as real conversation under the
    judge (``None`` without one: legibility unknown). ``correctFacts``: free-answer knowledge
    questions answered right (multiple choice can be right by luck). ``passingPrograms``: written
    functions that pass their hidden tests.
    """
    items = _items_by_id()

    def count(category: str, kind: type) -> int:
        return sum(
            1
            for r in report.items
            if r.category == category and r.score == 1 and isinstance(items.get(r.item_id), kind)
        )

    legible = None
    if judge is not None:
        legible = sum(
            1 for s in fingerprint.samples if is_well_formed(str(s.get("text", "")), judge)
        )
    return {
        "legibleReplies": legible,
        "correctFacts": count("knowledge", ExactItem),
        "passingPrograms": count("coding", CheckItem),
    }


# -- discoveries -----------------------------------------------------------------------------


def _prompt_of(result: ItemResult, items: dict[str, Any]) -> str | None:
    if result.category == "creativity" and result.item_id.startswith("story-"):
        k = int(result.item_id.removeprefix("story-"))
        return STORY_PROMPTS[k] if 0 <= k < len(STORY_PROMPTS) else None
    if result.category == "consistency":
        member = items.get(f"{result.item_id}-0")
        return member.prompt if member is not None else None
    item = items.get(result.item_id)
    return getattr(item, "prompt", None)


def _evidence(
    report: BenchReport, key: str, good: bool, items: dict[str, Any]
) -> dict[str, str] | None:
    """One result with the tag, quoted verbatim: for a good finding the first right one (else
    the best), for a bad one the first wrong one (else the worst). ``None`` when no result has
    the tag, or it has no reply to show (a grammar pair)."""
    category, _, tag = key.partition("/")
    tagged = [r for r in report.items if r.category == category and tag in r.tags]
    if not tagged:
        return None
    target = 1.0 if good else 0.0
    hit = next((r for r in tagged if r.score == target), None)
    if hit is None:
        hit = (max if good else min)(tagged, key=lambda r: r.score)
    prompt = _prompt_of(hit, items)
    if prompt is None or hit.output is None:
        return None
    return {"prompt": prompt, "reply": hit.output}


def _sample_evidence(fingerprint: Fingerprint) -> dict[str, str] | None:
    probes = {p.id: p.text for p in load_probes()}
    for sample in fingerprint.samples:
        text = sample.get("text")
        if isinstance(text, str) and text.strip() and sample.get("probe") in probes:
            return {"prompt": probes[sample["probe"]], "reply": text}
    return None


def _finding(kind: str, key: str, about: str, value: float, compared: float, label: str) -> dict:
    category, _, tag = key.partition("/")
    return {
        "key": key,
        "kind": kind,
        "about": about,
        "title": _TITLES[kind].format(about=about, label=label),
        "category": category,
        "tag": tag,
        "value": float(value),
        "compared": float(compared),
        "evidence": None,
    }


def find_discoveries(
    report: BenchReport,
    fingerprint: Fingerprint,
    previous: dict | None,
    peers: list[dict],
    thresholds: dict,
) -> list[dict]:
    """The largest real differences between this version and the one before it (``previous``:
    ``{label, breakdown, traits}``) or the world's other AIs (``peers``: ``{breakdown, traits}``).

    Tags (those with enough items, :meth:`BenchReport.tag_breakdown`): ``minPoints`` up or down
    against the previous version is "better" or "worse"; ``minPoints`` and ``minZ`` spreads
    above or below the mean of at least ``minPeers`` peers is a "strength" or "weakness".
    Traits: a move from the previous version of at least ``minZ`` peer spreads is "changed".
    One finding per tag (the previous-version one on a tie), the ``perEval`` largest kept.
    """
    min_points = float(thresholds.get("minPoints", 25))
    min_z = float(thresholds.get("minZ", 2))
    min_peers = int(thresholds.get("minPeers", 3))
    per_eval = int(thresholds.get("perEval", 3))
    label = str(previous.get("label", "")) if previous else ""
    candidates: list[tuple[float, int, dict]] = []  # (size, previous first, finding)

    for key, (value, _n) in report.tag_breakdown().items():
        about = tag_words(key)
        before = (previous or {}).get("breakdown", {}).get(key)
        if before is not None:
            d = value - float(before[0])
            if d >= min_points or d <= -min_points:
                kind = "better" if d > 0 else "worse"
                candidates.append((abs(d), 0, _finding(kind, key, about, value, before[0], label)))
        others = [float(p["breakdown"][key][0]) for p in peers if key in p.get("breakdown", {})]
        if len(others) >= min_peers:
            m = fmean(others)
            z = (value - m) / max(pstdev(others), TAG_FLOOR)
            if value - m >= min_points and z >= min_z:
                candidates.append((value - m, 1, _finding("strength", key, about, value, m, label)))
            elif m - value >= min_points and z <= -min_z:
                candidates.append((m - value, 1, _finding("weakness", key, about, value, m, label)))

    if previous is not None:
        for trait in TRAITS:
            before = previous.get("traits", {}).get(trait)
            others = [float(p["traits"][trait]) for p in peers if trait in p.get("traits", {})]
            if before is None or len(others) < min_peers:
                continue
            spread = pstdev(others)
            change = fingerprint.traits[trait] - float(before)
            if spread <= 0 or change == 0 or abs(change) < min_z * spread:
                continue
            low, high = TRAIT_PHRASES[trait]
            finding = _finding(
                "changed",
                f"{PERSONALITY}/{trait}",
                high if change > 0 else low,
                fingerprint.traits[trait],
                before,
                label,
            )
            candidates.append((abs(change) / spread * TRAIT_UNIT, 0, finding))

    best: dict[str, tuple[float, int, dict]] = {}
    for size, order, finding in candidates:
        held = best.get(finding["key"])
        if held is None or (size, -order) > (held[0], -held[1]):
            best[finding["key"]] = (size, order, finding)
    kept = sorted(best.values(), key=lambda c: (-c[0], c[1], c[2]["key"]))[:per_eval]

    items = _items_by_id()
    found = []
    for _size, _order, finding in kept:
        if finding["category"] == PERSONALITY:
            finding["evidence"] = _sample_evidence(fingerprint)
        else:
            good = finding["kind"] in ("strength", "better")
            finding["evidence"] = _evidence(report, finding["key"], good, items)
        found.append(finding)
    return found
