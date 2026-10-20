"""Where the real text comes from: the Hugging Face sources and the adapters that turn their rows
into documents.

A source row becomes zero or more pieces of content (``row_to_content``): a plain text, or a chat
as ``(role, text)`` turns. The adapters only select, cut and lay out what the dataset says; they
never write text of their own beyond the fixed frames the recipes call for (``Question:`` /
``Answer:`` for GSM8K, a ``#`` comment line for MBPP, the title line for Simple Wikipedia).

``hf_fetch`` streams rows from the Hugging Face hub and imports ``datasets`` only when it runs,
so nothing else needs the library. It rides out a flaky connection: a network error (or a 408, 429
or 5xx from the hub) at any point is waited out and the stream reopened and brought back to the
row where it broke, so the same seed still gives the same corpus; an error that waiting cannot fix
is raised as it is. ``fixture_fetch`` reads the same rows from local JSONL files, which keeps
every test offline. ``row_has_format`` says whether a row has the fields its adapter
reads, so a build can stop early when a dataset's format is not what the adapter expects (instead
of filtering out every row of a whole dataset).
"""

import itertools
import json
import re
import sys
import time
from collections import Counter
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


class DatasetsUnavailable(ImportError):
    """The ``datasets`` package, which streams the sources from the hub, is not installed."""


class SourceFormatError(ValueError):
    """A source's rows lack the fields its adapter reads: the dataset is not in the expected
    format."""


# A source that cannot be read gets this many tries in a row (a row read in between starts the
# count again); the waits between them double from the first to the longest.
RETRY_ATTEMPTS = 8
RETRY_FIRST_WAIT_S = 30.0
RETRY_MAX_WAIT_S = 300.0
_TRANSIENT_STATUSES = frozenset({408, 429})  # and every 5xx


class SourceUnavailable(ConnectionError):
    """A source could not be read, after the retries: a network problem, not a problem with the
    data."""


def _verdict(error: BaseException) -> bool | None:
    """True when waiting may fix ``error``, False when it cannot, None when it has no opinion."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        try:
            from huggingface_hub.errors import OfflineModeIsEnabled
        except ImportError:  # pragma: no cover - the hub client always comes with ``datasets``
            return True
        return not isinstance(error, OfflineModeIsEnabled)  # offline mode is asked for
    try:
        import httpx
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # pragma: no cover
        return None
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, (HfHubHTTPError, httpx.HTTPStatusError)):
        status = getattr(getattr(error, "response", None), "status_code", None)
        return status is None or status in _TRANSIENT_STATUSES or 500 <= status <= 599
    return None


def is_transient_error(error: BaseException) -> bool:
    """Whether waiting and trying again may fix ``error``: a dropped connection or timeout, or an
    HTTP error from the hub with no status, 408, 429 or 5xx. Not-found, 401/403/404, a config or
    format problem and every other error are not. A wrapped error is judged by its first cause
    that says."""
    for _ in range(8):
        verdict = _verdict(error)
        if verdict is not None:
            return verdict
        if error.__cause__ is None:
            return False
        error = error.__cause__
    return False


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def _one_line(error: BaseException, limit: int = 160) -> str:
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def hf_fetch(spec: SourceSpec, max_rows: int) -> Iterator[dict]:
    """Stream up to ``max_rows`` rows of ``spec`` from the Hugging Face hub (needs the network and
    the ``content`` extra, which provides ``datasets``).

    A transient error (``is_transient_error``), whether the source is being opened or a row read,
    is waited out (``RETRY_FIRST_WAIT_S``, doubling up to ``RETRY_MAX_WAIT_S``, one line on stderr
    each time) and the source opened again and read past the rows already given, so none is lost or
    repeated. After ``RETRY_ATTEMPTS`` failures in a row (a row given in between starts the count
    again) it raises ``SourceUnavailable``. Any other error is raised as it is, at once.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetsUnavailable(
            "fetching sources needs the 'datasets' package; install the content extra "
            "(uv sync --extra content)"
        ) from e
    given, failures = 0, 0
    rows: Iterator[dict] | None = None
    while given < max_rows:
        try:
            if rows is None:
                reopened = iter(
                    load_dataset(spec.hf_path, spec.hf_config, split=spec.split, streaming=True)
                )
                for _ in itertools.islice(reopened, given):  # read past what was already given
                    pass
                rows = reopened
            row = next(rows, None)
        except Exception as e:
            rows = None
            if not is_transient_error(e):
                raise
            failures += 1
            if failures >= RETRY_ATTEMPTS:
                raise SourceUnavailable(
                    f"network problem while reading source {spec.id!r} ({spec.hf_path}): "
                    f"{_one_line(e)}"
                ) from e
            wait = min(RETRY_FIRST_WAIT_S * 2 ** (failures - 1), RETRY_MAX_WAIT_S)
            print(
                f"network problem reading {spec.id} after {given:,} rows ({_one_line(e)}); "
                f"waiting {wait:.0f} s, then trying again (try {failures + 1} of "
                f"{RETRY_ATTEMPTS})",
                file=sys.stderr,
                flush=True,
            )
            _pause(wait)
            continue
        if row is None:
            return
        failures = 0
        given += 1
        yield row


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
_NEWLINE_RUN = re.compile(r"\n+")


def _subjects(metadata: object) -> str | None:
    """The subjects listed in a Gutenberg ``METADATA`` value (a JSON string, or already a dict);
    None when the value cannot be read or has no subjects entry."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    if not isinstance(metadata, dict):
        return None
    for key in ("subjects", "subject"):
        if key in metadata:
            subjects = metadata[key]
            if isinstance(subjects, (list, tuple)):
                return "; ".join(str(s) for s in subjects)
            return str(subjects or "")
    return None


def is_childrens_book(metadata: object) -> bool:
    """Whether the book's subjects mention juvenile literature, fairy tales, fables or children."""
    subjects = (_subjects(metadata) or "").lower()
    return any(marker in subjects for marker in CHILDREN_SUBJECTS)


def restore_plain_layout(text: str) -> str:
    """The text of a book as a plain-text file lays it out: the wrapped lines of a paragraph
    separated by one newline, the paragraphs by a blank line.

    The real ``sedthh/gutenberg_english`` rows are parsed from each book's HTML and lay their text
    out otherwise: every wrapped line is followed by a blank line (CRLF CRLF), and the paragraphs
    are set off by three newlines (or by a lone one). When the runs of exactly two newlines
    outnumber all the other runs, the text is in that layout: every run of two becomes one newline
    and every other run a blank line, so ``_unwrap`` joins the wrapped lines as it does for a
    plain-text file. Text in the plain-text layout is returned as it is, with line ends as LF.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lengths = Counter(len(run.group()) for run in _NEWLINE_RUN.finditer(text))
    if lengths[2] <= sum(lengths.values()) - lengths[2]:
        return text
    return _NEWLINE_RUN.sub(lambda run: "\n" if len(run.group()) == 2 else "\n\n", text)


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
SIMPLEWIKI_HEADING_MAX_WORDS = 8  # a line this short without a sentence end is a heading
MIN_CHAT_TURNS = 2
_CLOSING_MARKS = "\"'\u201d\u2019)]\u00bb "  # may follow a sentence's final punctuation
_CALCULATOR = re.compile(r"<<[^<>\n]*>>")  # GSM8K's calculator annotations, "<<48/2=24>>"


def _first_words(text: str, max_words: int) -> str:
    """At most ``max_words`` words of ``text``, ending at a sentence end when there is one."""
    kept = " ".join(text.split()[:max_words])
    ends = [m.start() for m in re.finditer(r"[.!?](?=\s|$)", kept)]
    return kept[: ends[-1] + 1] if ends else kept


def _is_heading(line: str) -> bool:
    """A short line that does not end a sentence: a section heading ("Related pages", "History")."""
    return len(line.split()) <= SIMPLEWIKI_HEADING_MAX_WORDS and not line.rstrip(
        _CLOSING_MARKS
    ).endswith((".", "!", "?"))


def _lead_section(text: str) -> list[str]:
    """The paragraphs before the article's first section heading."""
    lead: list[str] = []
    for line in (line.strip() for line in text.split("\n")):
        if not line:
            continue
        if _is_heading(line):
            break
        lead.append(line)
    return lead


def _simplewiki(row: dict) -> list[Content]:
    title = str(row.get("title") or "").strip()
    paragraphs = _lead_section(str(row.get("text") or ""))
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
    text = restore_plain_layout(str(row.get("TEXT") or ""))
    return list(chunk_passages(strip_gutenberg_boilerplate(text)))


def _fineweb_edu(row: dict) -> list[Content]:
    score = row.get("int_score")
    if score is None or score < 3:
        return []
    return _text(row)


def _text(row: dict) -> list[Content]:
    text = str(row.get("text") or "")
    return [text] if text.strip() else []


# About 7% of the TinyStories rows the hub serves have their punctuation misdecoded: UTF-8 bytes
# read as Windows-1252. A curly quote or dash is E2 80 xx, which shows as "a-circumflex, euro,
# <xx as cp1252>"; a closing double quote (E2 80 9D) has lost its last byte, which cp1252 has no
# character for, and is left as just the first two characters. The clean stories use straight
# ASCII quotes, so the repair gives those.
_TINYSTORIES_MOJIBAKE = re.compile("\u00e2\u20ac[\u0153\u2122\u02dc\u201c\u201d]?")
_TINYSTORIES_REPAIRS = {
    "\u00e2\u20ac\u0153": '"',  # left double quote
    "\u00e2\u20ac\u2122": "'",  # right single quote, the apostrophe
    "\u00e2\u20ac\u02dc": "'",  # left single quote
    "\u00e2\u20ac\u201c": "-",  # en dash
    "\u00e2\u20ac\u201d": "-",  # em dash
}


def repair_tinystories_mojibake(text: str) -> str:
    """``text`` with the misdecoded punctuation of the TinyStories rows turned into ASCII: the
    left double quote into ``"``, the apostrophe and the left single quote into ``'``, the dashes
    into ``-``, and the closing double quote (the two characters left of it) into ``"``. Text
    without them is returned as it is."""
    return _TINYSTORIES_MOJIBAKE.sub(lambda m: _TINYSTORIES_REPAIRS.get(m.group(), '"'), text)


def _tinystories(row: dict) -> list[Content]:
    return _text({"text": repair_tinystories_mojibake(str(row.get("text") or ""))})


_ADAPTERS: dict[str, Callable[[dict], list[Content]]] = {
    "tinystories": _tinystories,
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


# --- row formats -------------------------------------------------------------------------------

# A source whose first rows (or all rows, when it has fewer) all lack the fields its adapter reads
# is not in the expected format; the build stops there instead of filtering out the whole dataset.
FORMAT_CHECK_ROWS = 200


def _strings(*keys: str) -> Callable[[dict], bool]:
    return lambda row: all(isinstance(row.get(key), str) for key in keys)


def _scored(row: dict) -> bool:
    score = row.get("int_score")
    return _strings("text")(row) and isinstance(score, (int, float)) and not isinstance(score, bool)


def _dialogue(row: dict) -> bool:
    return isinstance(row.get("dialogue"), list) and isinstance(row.get("speakers"), list)


def _messages(row: dict) -> bool:
    messages = row.get("messages")
    return isinstance(messages, list) and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in messages
    )


def _book(row: dict) -> bool:
    return _strings("TEXT")(row) and _subjects(row.get("METADATA")) is not None


_ROW_FORMATS: dict[str, tuple[Callable[[dict], bool], str]] = {
    "tinystories": (_strings("text"), "'text' as a string"),
    "fineweb": (_strings("text"), "'text' as a string"),
    "fineweb_edu": (_scored, "'text' as a string and 'int_score' as a number"),
    "cosmo_khan": (_strings("text"), "'text' as a string"),
    "cosmo_wikihow": (_strings("text"), "'text' as a string"),
    "cosmo_openstax": (_strings("text"), "'text' as a string"),
    "cosmo_stories": (_strings("text"), "'text' as a string"),
    "soda": (_dialogue, "'dialogue' and 'speakers' as lists"),
    "everyday_conv": (_messages, "'messages' as a list of {role, content}"),
    "gutenberg": (_book, "'TEXT' as a string and 'METADATA' as JSON with 'subjects'"),
    "gsm8k": (_strings("question", "answer"), "'question' and 'answer' as strings"),
    "mbpp": (_strings("text", "code"), "'text' and 'code' as strings"),
    "simplewiki": (_strings("title", "text"), "'title' and 'text' as strings"),
}
assert set(_ROW_FORMATS) == set(SOURCES)


def row_has_format(source_id: str, row: dict) -> bool:
    """Whether ``row`` has the fields the source's adapter reads, in the expected form."""
    return _ROW_FORMATS[source_id][0](row)


def format_error(spec: SourceSpec, rows_read: int) -> SourceFormatError:
    """The error for a source none of whose first ``rows_read`` rows has the expected format."""
    return SourceFormatError(
        f"source {spec.id!r} ({spec.hf_path}): none of its first {rows_read} rows has "
        f"{_ROW_FORMATS[spec.id][1]}, which its adapter reads; the dataset is not in the "
        "expected format"
    )


def row_to_content(source_id: str, row: dict) -> list[Content]:
    """The documents in one row of a source: plain texts or chats of ``(role, text)`` turns.

    * gutenberg: children's books only (subjects mention "Juvenile", "Fairy tales", "Fables" or
      "Children"), without the Project Gutenberg header and footer, in the layout of a plain-text
      file (``restore_plain_layout``), in passages of at most 400 words;
    * fineweb_edu: rows with ``int_score >= 3``;
    * soda: ``speakers[0]`` is the user, every other speaker the AI; at least 2 turns;
    * everyday_conv: "user" is the user, "assistant" the AI, system turns dropped; at least 2 turns;
    * gsm8k: ``Question: {q}\\nAnswer: {reasoning} The answer is {final}.`` without the
      calculator annotations;
    * mbpp: ``# {text}\\n{code}`` with tabs as 4 spaces and LF line endings;
    * simplewiki: the title, then the first paragraphs of the lead section (before the first
      heading), up to 150 words;
    * tinystories: its ``text`` with the misdecoded quotes, apostrophes and dashes of the hub's
      rows repaired to ASCII (``repair_tinystories_mojibake``);
    * the rest: their ``text``.
    """
    if source_id not in _ADAPTERS:
        raise ValueError(f"unknown source {source_id!r}; expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[source_id](row)
