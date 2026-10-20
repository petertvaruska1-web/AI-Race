"""The reference judge: one fixed language model, trained once on all eight datasets and then
frozen, that measures how natural a text reads to a model that has read everything.

It is a measuring instrument, never a player's model. Creativity scores a story's coherence by its
per-token loss under the judge, and the feasibility gate calls a reply *well-formed* only if the
judge finds it no stranger than most real conversation replies (:func:`is_well_formed`).

:func:`build_judge` trains it into :func:`~airace_ml.paths.judge_dir`, which ends up
self-contained: the checkpoint, a copy of the frozen tokenizer (``tokenizer.json``) and
``calibration.json``, the loss levels the judge gives real held-out text and how varied that text
is (:func:`calibrate`).
``calibration.json`` is written last, so a directory with it holds a complete judge. A build cut
short (about an hour on the reference GPU) resumes from its last saved state when run again, but
only on the data it began on: ``building.json`` says what a build set out to produce.

A judge's token ids mean what its tokenizer says, and its calibration describes the corpora it
was calibrated on, so ``train_config.json`` records which data the judge was built from
(:func:`~airace_ml.data.identity.data_identity`) and whether it was trained on the recipe's full
budget. :func:`build_judge` trains again whenever the data has been rebuilt since, and
:func:`check_judge_provenance` is what the gate and ``bench`` use to refuse a judge built from
other data or from a shortened budget.
"""

import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from safetensors import SafetensorError

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.identity import data_identity
from airace_ml.data.prep import PrepConfig, heldout_docs
from airace_ml.device import pick_device
from airace_ml.evals.creativity import STORY_PROMPTS, STORY_TOKENS, distinct_2
from airace_ml.infer.lm import ContinuationScore, LanguageModel, TorchLM
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME, load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.paths import corpus_dir, judge_dir, tokenizer_path
from airace_ml.skills.checkers import words
from airace_ml.tokenizer import SPECIAL_TOKENS, Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import TrainEvent
from airace_ml.train.trainer import RESUME_DIR, can_resume, train_run

TOKENIZER_NAME = "tokenizer.json"
CALIBRATION_NAME = "calibration.json"
TRAIN_CONFIG_NAME = "train_config.json"
# Written before training starts and removed once TRAIN_CONFIG_NAME is: the record this build is
# about to produce. A resume state is continued only when the marker next to it is the same record.
BUILD_MARKER_NAME = "building.json"
CALIBRATION_DOCS = 500  # held-out creative documents, and conversation replies, calibrated on
CALIBRATION_SEED = 0
# A creative document is judged on its first 96 tokens after <|bos|>, as long as a story
# (creativity.py defines the length once, and calibrate reads real stories' variety at it).
CREATIVE_TOKENS = STORY_TOKENS
MIN_WORDS = 4
MIN_DISTINCT_WORDS = 3  # "go go go go" and "yes yes yes yes" say nothing
MAX_TRIGRAM_COUNT = 2  # a word 3-gram said 3 or more times makes a reply degenerate
_FIRST_TEXT_ID = len(SPECIAL_TOKENS)  # ids below this are special tokens
# Everything that can be wrong with a saved checkpoint, torch's RuntimeError (meta.json describes
# another shape than the weights) included. Catch it only around a load on the CPU.
_CHECKPOINT_DAMAGED = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    EOFError,
    SafetensorError,
    RuntimeError,
)


class JudgeBuildError(RuntimeError):
    """The judge's training did not finish, so it was neither calibrated nor frozen."""


@dataclass
class JudgeCalibration:
    """Per-token losses the judge gives real held-out text.

    ``creative_p10`` and ``creative_p90`` are the 10th and 90th percentiles over the openings of
    held-out creative documents; ``conv_reply_p90`` is the 90th percentile over the AI replies of
    held-out conversations. ``creative_distinct2`` is how varied real stories are, the
    distinct-2 (:func:`~airace_ml.evals.creativity.distinct_2`) of a set of held-out openings as
    big as a creativity run's set of stories (:func:`_reference_distinct_2`); a creativity run's
    variety is measured against it. A calibration without it comes from an older build and is
    refused (it raises ``TypeError``): the judge must be calibrated again, not guessed at.
    """

    creative_p10: float
    creative_p90: float
    conv_reply_p90: float
    creative_distinct2: float


def _nll(score: ContinuationScore) -> float:
    """Mean negative log-probability per token; ``inf`` for nothing scored or a NaN."""
    if score.n_tokens <= 0 or math.isnan(score.sum_logprob):
        return math.inf
    return -score.sum_logprob / score.n_tokens


class Judge:
    """The judge model behind :class:`~airace_ml.infer.lm.LanguageModel`, its tokenizer and its
    calibration (``None`` only while :func:`build_judge` is calibrating it)."""

    def __init__(
        self, lm: LanguageModel, tok: Tok, calibration: JudgeCalibration | None = None
    ) -> None:
        self.lm = lm
        self.tok = tok
        self.calibration = calibration

    @classmethod
    def load(cls, dir: Path, device: torch.device | str | None = None) -> "Judge":
        """The judge in ``dir``, which needs nothing outside it. ``device=None`` picks one with
        :func:`~airace_ml.device.pick_device`. A directory without ``calibration.json`` (written
        last) holds no complete judge: ``FileNotFoundError``."""
        dir = Path(dir)
        device = torch.device(device) if device is not None else pick_device()
        if not (dir / CALIBRATION_NAME).is_file():
            raise FileNotFoundError(f"{dir} holds no complete judge: {CALIBRATION_NAME} is missing")
        calibration = JudgeCalibration(
            **json.loads((dir / CALIBRATION_NAME).read_text(encoding="utf-8"))
        )
        tok = Tok.load(dir / TOKENIZER_NAME)
        model, _ = load_checkpoint(dir, device)
        if model.tok_emb.num_embeddings != tok.vocab_size:
            raise ValueError(
                f"the judge in {dir} has a {model.tok_emb.num_embeddings}-token vocabulary, "
                f"its tokenizer {tok.vocab_size}"
            )
        return cls(TorchLM(model, tok, device), tok, calibration)

    def nll_per_token(self, token_lists: Sequence[Sequence[int]]) -> list[float]:
        """Each list's mean negative log-probability per token, read as a document after
        ``<|bos|>``; ``inf`` for a list with nothing to score.

        Special-token ids are dropped first (text never contains them), and a list longer than
        the judge's attention span is judged on its first ``ctx_len - 1`` tokens. All lists go to
        the model in one call.
        """
        limit = self.lm.ctx_len - 1
        texts = [[int(t) for t in ids if t >= _FIRST_TEXT_ID][:limit] for ids in token_lists]
        scores = self.lm.score_continuations([[self.tok.bos_id] for _ in texts], texts)
        return [_nll(score) for score in scores]


def judge_train_config(seed: int = 1234) -> TrainRunConfig:
    """The judge's training run: 8 layers x 384 wide, a 256-token span, 120M tokens of an equal
    mix of all eight datasets, thoroughly cleaned, deduplicated and fact-checked."""
    return TrainRunConfig(
        run_id="judge-v1",
        seed=seed,
        shape=ModelShape(8, 384, 256),
        token_budget=120_000_000,
        mixture={ds: 1.0 for ds in DATASET_IDS},
        prep=PrepConfig("thorough", dedup=True, fact_check=True),
        boldness=0.4,
        batch_tokens=32768,
    )


# -- word checks ------------------------------------------------------------------------------


def _has_reply_words(reply: str) -> bool:
    """The word checks of a well-formed reply, shared by :func:`is_well_formed` and
    :func:`calibrate` so the two never drift apart: at least :data:`MIN_WORDS` words, at least
    :data:`MIN_DISTINCT_WORDS` different ones, and no word 3-gram said more than
    :data:`MAX_TRIGRAM_COUNT` times (case aside)."""
    said = [w.casefold() for w in words(reply)]
    if len(said) < MIN_WORDS or len(set(said)) < MIN_DISTINCT_WORDS:
        return False
    trigrams = Counter(zip(said, said[1:], said[2:]))
    return max(trigrams.values(), default=0) <= MAX_TRIGRAM_COUNT


# -- calibration ------------------------------------------------------------------------------


def _pick(indices: np.ndarray, k: int) -> np.ndarray:
    """All of ``indices`` if there are at most ``k``, else a seeded sample of ``k`` (ascending)."""
    if len(indices) <= k:
        return indices
    rng = np.random.default_rng(CALIBRATION_SEED)
    return np.sort(rng.choice(indices, size=k, replace=False))


def _creative_opening(doc: np.ndarray, bos_id: int) -> list[int]:
    """The first :data:`CREATIVE_TOKENS` tokens after the document's ``<|bos|>``."""
    start = 1 if len(doc) and doc[0] == bos_id else 0
    return doc[start : start + CREATIVE_TOKENS].tolist()


def _ai_replies(doc: np.ndarray, ai_id: int, end_id: int) -> list[list[int]]:
    """The tokens between each ``<|ai|>`` and the ``<|end|>`` that closes it (empty or unclosed
    replies left out)."""
    ends = np.flatnonzero(doc == end_id)
    replies = []
    for start in np.flatnonzero(doc == ai_id):
        k = np.searchsorted(ends, start)
        if k == len(ends):
            break
        if ends[k] > start + 1:
            replies.append(doc[start + 1 : ends[k]].tolist())
    return replies


def _percentiles(values: list[float], qs: list[float], what: str) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        raise ValueError(f"cannot calibrate the judge: no held-out {what} to score")
    return [float(v) for v in np.percentile(finite, qs)]


def _reference_distinct_2(openings: list[list[int]], tok: Tok) -> float:
    """How varied real stories are: the distinct-2 of as many held-out openings as a creativity
    run has stories (``len(STORY_PROMPTS)``), averaged over the sets that a seeded shuffle of
    ``openings`` cuts into (openings left over are dropped). With fewer openings than one set, the
    distinct-2 of all of them. ``ValueError`` if they hold no word pair.

    Distinct-2 falls as more text is pooled, so the openings are measured a run of stories at a
    time: a run is held to what real stories do in as many words, not in 20 times as many.
    """
    texts = [tok.decode(opening) for opening in openings]
    size = len(STORY_PROMPTS)
    order = np.random.default_rng(CALIBRATION_SEED).permutation(len(texts))
    sets = [order[i : i + size] for i in range(0, len(order) - size + 1, size)] or [order]
    value = fmean(distinct_2([texts[j] for j in chosen]) for chosen in sets)
    if not value > 0:
        raise ValueError(
            "cannot calibrate the judge: the held-out creative openings hold no word pairs to "
            "measure variety by"
        )
    return value


def calibrate(judge: Judge, data_root: Path) -> JudgeCalibration:
    """Score real held-out text under ``judge``: the openings of up to
    :data:`CALIBRATION_DOCS` creative documents and up to as many conversation AI replies, and
    measure how varied those openings are (:func:`_reference_distinct_2`).

    A reply is scored exactly as :func:`is_well_formed` scores one: replies that fail its word
    checks are left out (it rejects those without asking the judge), and the rest are scored as
    their own stripped text after ``<|bos|>``, with no chat context. Where there are more than
    enough, a fixed-seed sample of documents is used, so calibration is deterministic. With no
    creative opening or no eligible reply to score, it raises ``ValueError``.
    """
    tok = judge.tok
    creative = Corpus.open(corpus_dir(data_root) / "creative")
    openings = [
        _creative_opening(creative.doc(int(i)), tok.bos_id)
        for i in _pick(heldout_docs(creative), CALIBRATION_DOCS)
    ]
    conversations = Corpus.open(corpus_dir(data_root) / "conversations")
    order = np.random.default_rng(CALIBRATION_SEED).permutation(heldout_docs(conversations))
    replies: list[list[int]] = []
    for i in order:
        if len(replies) >= CALIBRATION_DOCS:
            break
        for segment in _ai_replies(conversations.doc(int(i)), tok.ai_id, tok.end_id):
            text = tok.decode(segment)
            if _has_reply_words(text):
                replies.append(tok.encode(text.strip()))
    replies = replies[:CALIBRATION_DOCS]
    p10, p90 = _percentiles(judge.nll_per_token(openings), [10, 90], "creative text")
    (reply_p90,) = _percentiles(
        judge.nll_per_token(replies), [90], "conversation replies that pass the word checks"
    )
    return JudgeCalibration(p10, p90, reply_p90, _reference_distinct_2(openings, tok))


# -- building ---------------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` to a temporary file, then rename it into place."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_record(cfg: TrainRunConfig, data_root: Path) -> dict:
    """What ``train_config.json`` says of a judge trained as ``cfg`` on the data under
    ``data_root``: the recipe itself, which data it was built from, and whether ``cfg`` is the
    full recipe (:func:`judge_train_config`, token budget included) and not a shortened test."""
    return {
        **json.loads(cfg.to_json()),
        "data_identity": data_identity(data_root),
        "full_recipe": cfg.token_budget == judge_train_config().token_budget,
    }


def _trained(out: Path, record: dict) -> bool:
    """``out`` holds the checkpoint of a completed training run recorded as ``record`` (see
    :func:`_run_record`): the record :func:`build_judge` writes once such a run completes, and the
    checkpoint beside it. A record of another recipe, or of other data, does not count."""
    try:
        saved = json.loads((out / TRAIN_CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    complete = (out / WEIGHTS_NAME).is_file() and (out / META_NAME).is_file()
    return complete and saved == record


def _may_resume(out: Path, cfg: TrainRunConfig, record: dict) -> bool:
    """``out`` holds the resume state of an interrupted build of exactly this judge: of the recipe
    ``cfg`` (:func:`~airace_ml.train.trainer.can_resume`) and, as its marker says, on the data and
    budget ``record`` describes. A resume state on other data, or with no marker (a build that
    predates markers), is not continued: the trainer's resume state knows the recipe, not the
    data, and weights trained on old data would be recorded as trained on the new."""
    try:
        marked = json.loads((out / BUILD_MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return marked == record and can_resume(out, cfg)


def _discard_resume_states(out: Path) -> None:
    """Delete every resume state in ``out`` (``resume``, and the ``resume.tmp.*`` and
    ``resume.old.*`` of a save cut short), as far as the files can be deleted.

    The trainer does this too when it starts a run that does not resume, but only after it has
    loaded the data. A build that is killed in between would leave the new build's marker next to
    the old build's state, which would then look resumable."""
    if not out.is_dir():
        return
    for path in list(out.iterdir()):  # listed first: the folder changes as it is deleted from
        if path.is_dir() and (path.name == RESUME_DIR or path.name.startswith(f"{RESUME_DIR}.")):
            shutil.rmtree(path, ignore_errors=True)


def check_judge_provenance(folder: Path, identity: str) -> None:
    """Raise ``ValueError`` (saying why) unless the judge in ``folder`` was built from the data
    whose :func:`~airace_ml.data.identity.data_identity` is ``identity`` and trained on the
    recipe's full budget, as its ``train_config.json`` records."""
    try:
        record = json.loads((folder / TRAIN_CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{TRAIN_CONFIG_NAME} cannot be read ({e})") from e
    if not isinstance(record, dict) or "data_identity" not in record:
        raise ValueError(f"{TRAIN_CONFIG_NAME} does not say which training data it was built from")
    if record["data_identity"] != identity:
        raise ValueError(
            "it was built from other training data than the data under the data root "
            "(the tokenizer or the corpora were rebuilt since)"
        )
    if record.get("full_recipe") is not True:
        raise ValueError("it was trained on a shortened budget (a quick test), not the full recipe")


def build_judge(
    data_root: Path,
    *,
    device: torch.device | str | None = None,
    on_event: Callable[[TrainEvent], None] | None = None,
    token_budget: int | None = None,
) -> Path:
    """Train the judge (:func:`judge_train_config`, with ``token_budget`` in place of its own if
    given) into :func:`~airace_ml.paths.judge_dir`, copy the tokenizer in, calibrate it and write
    ``calibration.json``. Returns the directory.

    ``on_event`` receives the run's training telemetry. ``train_config.json`` records the recipe,
    the data it was built from (:func:`~airace_ml.data.identity.data_identity`) and whether it is
    the full recipe. Running it again recovers from wherever the last build stopped:

    * if training of this exact run on this exact data completed (``train_config.json`` records
      it, written once the run completes) and its checkpoint can be loaded, the model is not
      trained again, only calibrated. A judge built before the record named its data has no such
      record, so it is trained again (about an hour on the reference GPU), not only calibrated;
    * else if the directory holds the resume state of this exact run on this exact data (as the
      marker ``building.json``, written before training starts, says), training resumes from it;
    * else training starts fresh, and any other state (another budget, other data, damaged
      weights, an incomplete save) is deleted.

    A training run that does not complete (it stopped as unstable) raises
    :class:`JudgeBuildError`: the model is neither calibrated nor recorded as trained. A previous
    calibration is removed first, so the directory never pairs a model with an old calibration.
    """
    data_root = Path(data_root)
    device = torch.device(device) if device is not None else pick_device()
    cfg = judge_train_config()
    if token_budget is not None:
        cfg = replace(cfg, token_budget=token_budget)
    record = _run_record(cfg, data_root)
    out = judge_dir(data_root)
    (out / CALIBRATION_NAME).unlink(missing_ok=True)
    model = None
    if _trained(out, record):
        try:
            # On the CPU, where only a damaged file can fail (a busy GPU is no reason to train)
            model, _ = load_checkpoint(out, "cpu")
        except _CHECKPOINT_DAMAGED:
            model = None  # weights that cannot be read are no judge: train it again
    if model is None:
        (out / TRAIN_CONFIG_NAME).unlink(missing_ok=True)
        resume = _may_resume(out, cfg, record)
        if not resume:
            _discard_resume_states(out)  # before the marker: see _discard_resume_states
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / BUILD_MARKER_NAME, record)  # before any training can be interrupted
        result = train_run(
            cfg,
            out_dir=out,
            data_root=data_root,
            device=device,
            on_event=on_event,
            resume=resume,
        )
        if result.status != "completed":
            raise JudgeBuildError(
                f"the judge's training ended {result.status!r} after {result.steps} of "
                f"{cfg.steps} steps, so it was not calibrated; its telemetry is in {out}"
            )
        _write_json(out / TRAIN_CONFIG_NAME, record)
        (out / BUILD_MARKER_NAME).unlink(missing_ok=True)
        model, _ = load_checkpoint(out, device)
    shutil.copyfile(tokenizer_path(data_root), out / TOKENIZER_NAME)
    tok = Tok.load(out / TOKENIZER_NAME)
    calibration = calibrate(Judge(TorchLM(model, tok, device), tok), data_root)
    _write_json(out / CALIBRATION_NAME, asdict(calibration))
    return out


# -- well-formed replies ----------------------------------------------------------------------


def is_well_formed(reply: str, judge: Judge) -> bool:
    """A reply that reads like real conversation: at least :data:`MIN_WORDS` words, at least
    :data:`MIN_DISTINCT_WORDS` different words, no word 3-gram said 3 or more times (case aside),
    and a per-token loss under the judge no higher than the 90th percentile of real held-out
    conversation replies.

    The distinct-word rule extends spec 4.12's definition (a controller ruling): without it,
    "go go go go" passes every word check and is left to the judge alone.
    """
    if not _has_reply_words(reply):
        return False
    nll = judge.nll_per_token([judge.tok.encode(reply.strip())])[0]
    return nll <= judge.calibration.conv_reply_p90
