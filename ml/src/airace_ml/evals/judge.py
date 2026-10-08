"""The reference judge: one fixed language model, trained once on all eight datasets and then
frozen, that measures how natural a text reads to a model that has read everything.

It is a measuring instrument, never a player's model. Creativity scores a story's coherence by its
per-token loss under the judge, and the feasibility gate calls a reply *well-formed* only if the
judge finds it no stranger than most real conversation replies (:func:`is_well_formed`).

:func:`build_judge` trains it into :func:`~airace_ml.paths.judge_dir`, which ends up
self-contained: the checkpoint, a copy of the frozen tokenizer (``tokenizer.json``) and
``calibration.json``, the loss levels the judge gives real held-out text (:func:`calibrate`).
``calibration.json`` is written last, so a directory with it holds a complete judge. A build cut
short (about an hour on the reference GPU) resumes from its last saved state when run again.
"""

import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.prep import PrepConfig, heldout_docs
from airace_ml.device import pick_device
from airace_ml.infer.lm import ContinuationScore, LanguageModel, TorchLM
from airace_ml.model.checkpoint import META_NAME, WEIGHTS_NAME, load_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.paths import corpus_dir, judge_dir, tokenizer_path
from airace_ml.skills.checkers import words
from airace_ml.tokenizer import SPECIAL_TOKENS, Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import TrainEvent
from airace_ml.train.trainer import can_resume, train_run

TOKENIZER_NAME = "tokenizer.json"
CALIBRATION_NAME = "calibration.json"
TRAIN_CONFIG_NAME = "train_config.json"
CALIBRATION_DOCS = 500  # held-out creative documents, and conversation replies, calibrated on
CALIBRATION_SEED = 0
CREATIVE_TOKENS = 96  # a creative document is judged on its first 96 tokens after <|bos|>
MIN_WORDS = 4
MIN_DISTINCT_WORDS = 3  # "go go go go" and "yes yes yes yes" say nothing
MAX_TRIGRAM_COUNT = 2  # a word 3-gram said 3 or more times makes a reply degenerate
_FIRST_TEXT_ID = len(SPECIAL_TOKENS)  # ids below this are special tokens


class JudgeBuildError(RuntimeError):
    """The judge's training did not finish, so it was neither calibrated nor frozen."""


@dataclass
class JudgeCalibration:
    """Per-token losses the judge gives real held-out text.

    ``creative_p10`` and ``creative_p90`` are the 10th and 90th percentiles over the openings of
    held-out creative documents; ``conv_reply_p90`` is the 90th percentile over the AI replies of
    held-out conversations.
    """

    creative_p10: float
    creative_p90: float
    conv_reply_p90: float


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


def calibrate(judge: Judge, data_root: Path) -> JudgeCalibration:
    """Score real held-out text under ``judge``: the openings of up to
    :data:`CALIBRATION_DOCS` creative documents and up to as many conversation AI replies.

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
    return JudgeCalibration(p10, p90, reply_p90)


# -- building ---------------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` to a temporary file, then rename it into place."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _trained(out: Path, cfg: TrainRunConfig) -> bool:
    """``out`` holds the checkpoint of a completed training run of exactly ``cfg``: the record
    :func:`build_judge` writes once such a run completes, and the checkpoint beside it."""
    try:
        record = json.loads((out / TRAIN_CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    complete = (out / WEIGHTS_NAME).is_file() and (out / META_NAME).is_file()
    return complete and record == json.loads(cfg.to_json())


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

    ``on_event`` receives the run's training telemetry. Running it again recovers from wherever
    the last build stopped:

    * if training of this exact run completed (``train_config.json`` records it, written once the
      run completes), the model is not trained again, only calibrated;
    * else if the directory holds the resume state of this exact run, training resumes from it;
    * else training starts fresh, and any other state (another budget, an incomplete save) is
      deleted.

    A training run that does not complete (it stopped as unstable) raises
    :class:`JudgeBuildError`: the model is neither calibrated nor recorded as trained. A previous
    calibration is removed first, so the directory never pairs a model with an old calibration.
    """
    data_root = Path(data_root)
    device = torch.device(device) if device is not None else pick_device()
    cfg = judge_train_config()
    if token_budget is not None:
        cfg = replace(cfg, token_budget=token_budget)
    out = judge_dir(data_root)
    (out / CALIBRATION_NAME).unlink(missing_ok=True)
    if not _trained(out, cfg):
        (out / TRAIN_CONFIG_NAME).unlink(missing_ok=True)
        result = train_run(
            cfg,
            out_dir=out,
            data_root=data_root,
            device=device,
            on_event=on_event,
            resume=can_resume(out, cfg),
        )
        if result.status != "completed":
            raise JudgeBuildError(
                f"the judge's training ended {result.status!r} after {result.steps} of "
                f"{cfg.steps} steps, so it was not calibrated; its telemetry is in {out}"
            )
        _write_json(out / TRAIN_CONFIG_NAME, json.loads(cfg.to_json()))
    shutil.copyfile(tokenizer_path(data_root), out / TOKENIZER_NAME)
    model, _ = load_checkpoint(out, device)
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
