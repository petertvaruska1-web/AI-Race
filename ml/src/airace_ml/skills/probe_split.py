"""Which knowledge-base subjects the follow-up probe keeps for itself.

The follow-up probe asks a second question about a new subject ("And Peru?"). To measure
whether the AI follows the conversation, that subject must never be one the AI was taught in
a second turn. Training text for follow-ups stays away from the subjects this module picks.
The pick is a stable hash of the relation and the subject: no random state, the same on every
machine.
"""

import hashlib

PROBE_SUBJECT_SHARE = 0.1
_BUCKETS = 10_000


def is_probe_subject(relation: str, subject: str) -> bool:
    """Whether ``subject`` of ``relation`` is held back for the follow-up probe (about
    :data:`PROBE_SUBJECT_SHARE` of all subjects)."""
    digest = hashlib.sha256(f"{relation}\x00{subject}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _BUCKETS < PROBE_SUBJECT_SHARE * _BUCKETS
