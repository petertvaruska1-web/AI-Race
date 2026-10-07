"""Where the real text comes from: the Hugging Face sources and the adapters that turn their rows
into documents.

A source row becomes zero or more pieces of content (``row_to_content``): a plain text, or a chat
as ``(role, text)`` turns. The adapters only select, cut and lay out what the dataset says; they
never write text of their own beyond the fixed frames the recipes call for (``Question:`` /
``Answer:`` for GSM8K, a ``#`` comment line for MBPP, the title line for Simple Wikipedia).

``hf_fetch`` streams rows from the Hugging Face hub and imports ``datasets`` only when it runs,
so nothing else needs the library. ``fixture_fetch`` reads the same rows from local JSONL files,
which keeps every test offline.
"""

import itertools
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from airace_ml.tokenizer import Role

Content = str | list[tuple[Role, str]]


@dataclass(frozen=True)
class SourceSpec:
    id: str
    hf_path: str
    hf_config: str | None
    split: str
    license: str
    homepage: str


def _spec(id: str, hf_path: str, hf_config: str | None, split: str, license: str) -> SourceSpec:
    return SourceSpec(
        id, hf_path, hf_config, split, license, f"https://huggingface.co/datasets/{hf_path}"
    )


# Licenses as the dataset cards state them; Task 15 re-verifies each one against the hub.
SOURCES: dict[str, SourceSpec] = {
    spec.id: spec
    for spec in (
        _spec("tinystories", "roneneldan/TinyStories", None, "train", "cdla-sharing-1.0"),
        _spec("fineweb", "HuggingFaceFW/fineweb", "sample-10BT", "train", "odc-by"),
        _spec("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "odc-by"),
        _spec("cosmo_khan", "HuggingFaceTB/cosmopedia", "khanacademy", "train", "apache-2.0"),
        _spec("cosmo_wikihow", "HuggingFaceTB/cosmopedia", "wikihow", "train", "apache-2.0"),
        _spec("cosmo_openstax", "HuggingFaceTB/cosmopedia", "openstax", "train", "apache-2.0"),
        _spec("cosmo_stories", "HuggingFaceTB/cosmopedia", "stories", "train", "apache-2.0"),
        _spec("soda", "allenai/soda", None, "train", "cc-by-4.0"),
        _spec(
            "everyday_conv",
            "HuggingFaceTB/everyday-conversations-llama3.1-2k",
            None,
            "train_sft",
            "apache-2.0",
        ),
        _spec("gutenberg", "sedthh/gutenberg_english", None, "train", "mit"),
        _spec("gsm8k", "openai/gsm8k", "main", "train", "mit"),
        _spec("mbpp", "google-research-datasets/mbpp", "full", "train", "cc-by-4.0"),
        _spec("simplewiki", "wikimedia/wikipedia", "20231101.simple", "train", "cc-by-sa-3.0"),
    )
}

Fetch = Callable[[SourceSpec, int], Iterator[dict]]


def hf_fetch(spec: SourceSpec, max_rows: int) -> Iterator[dict]:
    """Stream up to ``max_rows`` rows of ``spec`` from the Hugging Face hub (needs the network and
    the ``content`` extra, which provides ``datasets``)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "fetching sources needs the 'datasets' package; install the content extra "
            "(uv sync --extra content)"
        ) from e
    rows = load_dataset(spec.hf_path, spec.hf_config, split=spec.split, streaming=True)
    yield from itertools.islice(rows, max_rows)


def fixture_fetch(fixture_dir: Path) -> Fetch:
    """A ``Fetch`` that reads ``<fixture_dir>/<source id>.jsonl`` (one JSON row per line)."""
    fixture_dir = Path(fixture_dir)

    def fetch(spec: SourceSpec, max_rows: int) -> Iterator[dict]:
        with (fixture_dir / f"{spec.id}.jsonl").open(encoding="utf-8") as f:
            rows = (json.loads(line) for line in f if line.strip())
            yield from itertools.islice(rows, max_rows)

    return fetch


# --- Gutenberg ---------------------------------------------------------------------------------

CHILDREN_SUBJECTS: tuple[str, ...] = ("juvenile", "fairy tales", "fables", "children")
PASSAGE_MAX_WORDS = 400
PASSAGE_MIN_WORDS = 20  # shorter leftovers ("THE END", a lone heading) are not passages
WRAPPED_LINE_MIN = 45  # a paragraph whose lines (but the last) are this long is hard-wrapped prose

# The licence header ends at the START marker (or, in old files, at the end of the "small print");
# the footer begins at the END marker or the "End of the Project Gutenberg EBook" line before it.
_PG_HEADER_END = re.compile(
    r"^.*(?:\*{3}[ \t]*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK|\*END\*THE SMALL PRINT).*$",
    re.IGNORECASE | re.MULTILINE,
)
_PG_FOOTER_START = re.compile(
    r"^(?:.*\*{3}[ \t]*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK|[ \t]*End of (?:the )?Project Gutenberg)",
    re.IGNORECASE | re.MULTILINE,
)
# Credit lines right after the header ("Produced by ...") belong to the header too.
_PG_CREDITS = re.compile(
    r"^\s*(?:produced by|e-?text prepared by|this e-?(?:text|book) was produced|transcribed (?:by|from)|"
    r"transcriber's note)",
    re.IGNORECASE,
)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _subjects(metadata: object) -> str:
    """The subjects listed in a Gutenberg ``METADATA`` value (a JSON string, or already a dict)."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return ""
    if not isinstance(metadata, dict):
        return ""
    subjects = metadata.get("subjects", metadata.get("subject", ""))
    if isinstance(subjects, (list, tuple)):
        return "; ".join(str(s) for s in subjects)
    return str(subjects or "")


def is_childrens_book(metadata: object) -> bool:
    """Whether the book's subjects mention juvenile literature, fairy tales, fables or children."""
    subjects = _subjects(metadata).lower()
    return any(marker in subjects for marker in CHILDREN_SUBJECTS)


def strip_gutenberg_boilerplate(text: str) -> str:
    """The book without the Project Gutenberg licence header, credit lines and footer."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    headers = list(_PG_HEADER_END.finditer(text))
    if headers:
        text = text[headers[-1].end() :]
    footer = _PG_FOOTER_START.search(text)
    if footer:
        text = text[: footer.start()]
    paragraphs = [p for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    while paragraphs and _PG_CREDITS.match(paragraphs[0]):
        paragraphs.pop(0)
    return "\n\n".join(p.strip("\n") for p in paragraphs)


def _unwrap(paragraph: str) -> str:
    """Join the hard-wrapped lines of a prose paragraph; verse and lists keep their lines."""
    lines = [line.strip() for line in paragraph.strip().split("\n")]
    if len(lines) > 1 and all(len(line) >= WRAPPED_LINE_MIN for line in lines[:-1]):
        return " ".join(lines)
    return "\n".join(lines)


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Pieces of at most ``max_words`` words, cut between sentences (or, failing that, words)."""
    pieces: list[str] = []
    current: list[str] = []
    n = 0
    for sentence in _SENTENCE_END.split(paragraph):
        sentence_words = sentence.split()
        while len(sentence_words) > max_words:  # one sentence longer than a passage
            if current:
                pieces.append(" ".join(current))
                current, n = [], 0
            pieces.append(" ".join(sentence_words[:max_words]))
            sentence_words = sentence_words[max_words:]
        if n + len(sentence_words) > max_words and current:
            pieces.append(" ".join(current))
            current, n = [], 0
        if sentence_words:
            current.append(" ".join(sentence_words))
            n += len(sentence_words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_passages(text: str, max_words: int = PASSAGE_MAX_WORDS) -> list[str]:
    """Consecutive paragraphs packed into passages of at most ``max_words`` words; a paragraph that
    is longer on its own is cut between sentences. Passages under ``PASSAGE_MIN_WORDS`` are dropped."""
    paragraphs: list[str] = []
    for raw in _PARAGRAPH_BREAK.split(text):
        if not raw.strip():
            continue
        paragraph = _unwrap(raw)
        if len(paragraph.split()) > max_words:
            paragraphs.extend(_split_long(paragraph, max_words))
        else:
            paragraphs.append(paragraph)
    passages: list[str] = []
    current: list[str] = []
    n = 0
    for paragraph in paragraphs:
        k = len(paragraph.split())
        if n + k > max_words and current:
            passages.append("\n\n".join(current))
            current, n = [], 0
        current.append(paragraph)
        n += k
    if current:
        passages.append("\n\n".join(current))
    return [p for p in passages if len(p.split()) >= PASSAGE_MIN_WORDS]


# --- the other adapters ------------------------------------------------------------------------

SIMPLEWIKI_MAX_WORDS = 150
MIN_CHAT_TURNS = 2
_CALCULATOR = re.compile(r"<<[^<>\n]*>>")  # GSM8K's calculator annotations, "<<48/2=24>>"


def _first_words(text: str, max_words: int) -> str:
    """At most ``max_words`` words of ``text``, ending at a sentence end when there is one."""
    kept = " ".join(text.split()[:max_words])
    ends = [m.start() for m in re.finditer(r"[.!?](?=\s|$)", kept)]
    return kept[: ends[-1] + 1] if ends else kept


def _simplewiki(row: dict) -> list[Content]:
    title = str(row.get("title") or "").strip()
    paragraphs = [p.strip() for p in str(row.get("text") or "").split("\n") if p.strip()]
    chosen: list[str] = []
    n = 0
    for paragraph in paragraphs:
        k = len(paragraph.split())
        if n + k > SIMPLEWIKI_MAX_WORDS:
            if not chosen:
                chosen.append(_first_words(paragraph, SIMPLEWIKI_MAX_WORDS))
            break
        chosen.append(paragraph)
        n += k
    if not title or not chosen:
        return []
    return [title + "\n" + "\n".join(chosen)]


def _gsm8k(row: dict) -> list[Content]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "")
    if not question or "####" not in answer:
        return []
    reasoning, final = answer.rsplit("####", 1)
    reasoning = _CALCULATOR.sub("", reasoning).strip()
    final = final.strip().rstrip(".").strip()
    if not final:
        return []
    body = f"{reasoning} " if reasoning else ""
    return [f"Question: {question}\nAnswer: {body}The answer is {final}."]


def _mbpp(row: dict) -> list[Content]:
    text = " ".join(str(row.get("text") or "").split())
    code = str(row.get("code") or "").replace("\r\n", "\n").replace("\t", "    ").strip("\n")
    if not text or not code.strip():
        return []
    return [f"# {text}\n{code}"]


def _soda(row: dict) -> list[Content]:
    dialogue, speakers = row.get("dialogue") or [], row.get("speakers") or []
    if not speakers:
        return []
    first = speakers[0]
    turns: list[tuple[Role, str]] = [
        ("user" if speaker == first else "ai", str(text).strip())
        for speaker, text in zip(speakers, dialogue)
        if str(text).strip()
    ]
    return [turns] if len(turns) >= MIN_CHAT_TURNS else []


_EVERYDAY_ROLES: dict[str, Role] = {"user": "user", "assistant": "ai"}


def _everyday(row: dict) -> list[Content]:
    turns: list[tuple[Role, str]] = [
        (_EVERYDAY_ROLES[m["role"]], str(m.get("content") or "").strip())
        for m in row.get("messages") or []
        if m.get("role") in _EVERYDAY_ROLES and str(m.get("content") or "").strip()
    ]
    return [turns] if len(turns) >= MIN_CHAT_TURNS else []


def _gutenberg(row: dict) -> list[Content]:
    if not is_childrens_book(row.get("METADATA")):
        return []
    return list(chunk_passages(strip_gutenberg_boilerplate(str(row.get("TEXT") or ""))))


def _fineweb_edu(row: dict) -> list[Content]:
    score = row.get("int_score")
    if score is None or score < 3:
        return []
    return _text(row)


def _text(row: dict) -> list[Content]:
    text = str(row.get("text") or "")
    return [text] if text.strip() else []


_ADAPTERS: dict[str, Callable[[dict], list[Content]]] = {
    "tinystories": _text,
    "fineweb": _text,
    "fineweb_edu": _fineweb_edu,
    "cosmo_khan": _text,
    "cosmo_wikihow": _text,
    "cosmo_openstax": _text,
    "cosmo_stories": _text,
    "soda": _soda,
    "everyday_conv": _everyday,
    "gutenberg": _gutenberg,
    "gsm8k": _gsm8k,
    "mbpp": _mbpp,
    "simplewiki": _simplewiki,
}
assert set(_ADAPTERS) == set(SOURCES)


def row_to_content(source_id: str, row: dict) -> list[Content]:
    """The documents in one row of a source: plain texts or chats of ``(role, text)`` turns.

    * gutenberg: children's books only (subjects mention "Juvenile", "Fairy tales", "Fables" or
      "Children"), without the Project Gutenberg header and footer, in passages of at most 400
      words;
    * fineweb_edu: rows with ``int_score >= 3``;
    * soda: ``speakers[0]`` is the user, every other speaker the AI; at least 2 turns;
    * everyday_conv: "user" is the user, "assistant" the AI, system turns dropped; at least 2 turns;
    * gsm8k: ``Question: {q}\\nAnswer: {reasoning} The answer is {final}.`` without the
      calculator annotations;
    * mbpp: ``# {text}\\n{code}`` with tabs as 4 spaces and LF line endings;
    * simplewiki: the title, then the first paragraphs, up to 150 words;
    * the rest: their ``text``.
    """
    if source_id not in _ADAPTERS:
        raise ValueError(f"unknown source {source_id!r}; expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[source_id](row)
