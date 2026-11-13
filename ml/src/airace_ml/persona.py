"""The one line that tells an AI who it is, and a check for replies that claim a human life."""

import re

_MAX_PART = 60

_PERSON_LIFE = re.compile(
    r"\bmy (?:mom|mother|dad|father|wife|husband|son|daughter|brother|sister|kids?|boss|teacher"
    r"|job|house|car)\b"
    r"|\bi (?:went|go|am going) to (?:school|work|church|the gym)\b"
    r"|\bmy (?:hands?|legs?|eyes?|hair)\b",
    re.IGNORECASE,
)


def _one_line(text: str) -> str:
    """``text`` on one line, stripped and cut to 60 characters."""
    return " ".join(text.splitlines()).strip()[:_MAX_PART].strip()


def identity_line(name: str, company: str | None) -> str:
    """``You are {name}, an AI made by {company}.`` (``You are {name}, an AI.`` without a company).

    Name and company are plain text: newlines become spaces and each is cut to 60 characters.
    """
    name = _one_line(name)
    if not name:
        raise ValueError("an AI needs a name")
    company = _one_line(company) if company else ""
    if company:
        return f"You are {name}, an AI made by {company}."
    return f"You are {name}, an AI."


def claims_person_life(text: str) -> bool:
    """True when the text speaks in the first person of family, school, job, home or body."""
    return _PERSON_LIFE.search(text) is not None
