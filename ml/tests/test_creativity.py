import json
import math
import os
import re
import shutil
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import airace_content
import airace_ml.skills
from airace_ml.data.corpus import DATASET_IDS, Corpus, DocTags, write_corpus
from airace_ml.data.prep import heldout_docs
from airace_ml.evals import judge as judge_module
from airace_ml.evals import novelty as novelty_module
from airace_ml.evals.creativity import STORY_PROMPTS, score_creativity, words
from airace_ml.evals.judge import (
    CALIBRATION_NAME,
    TOKENIZER_NAME,
    TRAIN_CONFIG_NAME,
    Judge,
    JudgeBuildError,
    JudgeCalibration,
    build_judge,
    calibrate,
    is_well_formed,
    judge_train_config,
)
from airace_ml.evals.novelty import (
    HASH_BASE,
    NoveltyIndex,
    build_novelty_index,
    ngram_hashes,
)
from airace_ml.evals.suite import run_benchmarks
from airace_ml.infer.lm import Generation, TorchLM
from airace_ml.model.checkpoint import CheckpointMeta, save_checkpoint
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.paths import corpus_dir, judge_dir, novelty_path
from airace_ml.skills.checkers import words as checker_words
from airace_ml.tokenizer import encode_chat, encode_doc
from airace_ml.train.events import Done, Progress
from airace_ml.train.trainer import TrainHooks, train_run
from tests.fakes import ScriptedLM


class StubJudge:
    def __init__(self, tok, nll):
        self.tok, self._nll = tok, nll

    calibration = JudgeCalibration(creative_p10=2.0, creative_p90=6.0, conv_reply_p90=5.0)

    def nll_per_token(self, lists):
        return [self._nll(l) for l in lists]


CORPUS_TEXT = "once upon a time a little bunny hopped to the big green hill and ate a carrot"


def test_novelty_index(tiny_tok, tmp_path):
    ids = np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)
    idx = NoveltyIndex.build([ids], sample_mod=1)
    assert idx.novelty(ids.tolist()) == 0.0 and idx.novelty([5, 6]) == 0.0
    rnd = np.random.default_rng(0).integers(20, 500, 200).tolist()
    assert idx.novelty(rnd) > 0.95
    idx.save(tmp_path / "i.npy")
    assert NoveltyIndex.load(tmp_path / "i.npy").novelty(rnd) == idx.novelty(rnd)


def test_creativity_ordering(tiny_tok):
    idx = NoveltyIndex.build([np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)], sample_mod=1)
    fluent = StubJudge(tiny_tok, lambda l: 2.5)
    gib = StubJudge(tiny_tok, lambda l: 9.0)
    novel = ScriptedLM(
        tiny_tok,
        reply=lambda p: f"a curious robot {len(p)} painted purple stars across the quiet ocean sky",
    )
    copy = ScriptedLM(tiny_tok, reply=lambda p: CORPUS_TEXT)
    s_novel = score_creativity(novel, tiny_tok, fluent, idx)[0].score
    s_copy = score_creativity(copy, tiny_tok, fluent, idx)[0].score
    s_gib = score_creativity(novel, tiny_tok, gib, idx)[0].score
    # Review Focus 4
    s_empty = score_creativity(ScriptedLM(tiny_tok), tiny_tok, fluent, idx)[0].score
    assert s_novel > s_copy > s_gib >= 0 and s_empty == 0 and len(STORY_PROMPTS) == 24


def test_well_formed(tiny_tok):
    j = StubJudge(tiny_tok, lambda l: 3.0)
    assert is_well_formed("I like to play in the park.", j)
    assert not is_well_formed("hi there", j) and not is_well_formed("go go go go go go go go go", j)
    assert not is_well_formed("I like to play in the park.", StubJudge(tiny_tok, lambda l: 8.0))


def test_words_is_the_public_lowercase_letter_run_extractor():
    # creativity scoring and personality both count words this way: runs of letters, lowercased
    assert words("Hello, WORLD 42! x_y") == ["hello", "world", "x", "y"]
    assert words("Caf\u00e9 d\u00e9j\u00e0-vu na\u00efve") == [
        "caf\u00e9",
        "d\u00e9j\u00e0",
        "vu",
        "na\u00efve",
    ]
    assert words("\u65e5\u672c\u8a9e \U0001f600 \ufffd") == ["\u65e5\u672c\u8a9e"]
    assert words("") == [] and words("123 _ !?") == []


def test_judge_config_valid():
    c = judge_train_config()
    c.validate()
    assert c.shape.n_layer == 8 and c.token_budget == 120_000_000


@pytest.mark.slow
def test_build_judge_smoke(tiny_data_root):
    from airace_ml.evals.judge import Judge, build_judge

    d = build_judge(tiny_data_root, token_budget=32768 * 4)
    j = Judge.load(d)
    assert math.isfinite(j.calibration.creative_p90)


# -- helpers for the tests below -----------------------------------------------------------------

CPU = torch.device("cpu")
INF = math.inf


class RecordingLM(ScriptedLM):
    """A :class:`ScriptedLM` that also records every scoring and generation call."""

    def __init__(self, tok, **kwargs):
        super().__init__(tok, **kwargs)
        self.scored = []
        self.generated = []

    def score_continuations(self, contexts, continuations):
        self.scored.append(([list(c) for c in contexts], [list(c) for c in continuations]))
        return super().score_continuations(contexts, continuations)

    def generate(self, prompts, **kwargs):
        self.generated.append(([list(p) for p in prompts], kwargs))
        return super().generate(prompts, **kwargs)


class TokenLM:
    """Replies to every prompt with the same raw token ids (special tokens included)."""

    ctx_len = 256

    def __init__(self, tokens):
        self.tokens = list(tokens)

    def generate(self, prompts, **kwargs):
        n = len(self.tokens)
        return [Generation(list(self.tokens), [0.5] * n, [0.5] * n, False) for _ in prompts]


def ref_hashes(ids, n):
    """The window hashes computed slowly with Python integers."""
    ids = [int(t) for t in ids]
    return [
        sum(ids[i + j] * HASH_BASE ** (n - 1 - j) for j in range(n)) % 2**64
        for i in range(len(ids) - n + 1)
    ]


def ref_words(text):
    return re.findall(r"[a-z]+", text.lower())


def ref_repetitiveness(text):
    """1 - unique word 3-grams / all word 3-grams of one text; 0 with no 3-grams."""
    words = ref_words(text)
    trigrams = list(zip(words, words[1:], words[2:]))
    return 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0


def ref_distinct_2(texts):
    pairs = [p for t in texts for p in zip(ref_words(t), ref_words(t)[1:])]
    return len(set(pairs)) / len(pairs) if pairs else 0.0


def ref_replies(doc, ai_id, end_id):
    """Every AI turn's tokens, read one token at a time."""
    replies, current = [], None
    for t in doc:
        if t == ai_id:
            current = []
        elif t == end_id:
            if current:
                replies.append(current)
            current = None
        elif current is not None:
            current.append(t)
    return replies


def ref_word_checks(reply):
    """The word rules of a well-formed reply, written out: 4+ words, 3+ different words, and no
    word 3-gram said 3 or more times (case aside)."""
    said = [w.casefold() for w in checker_words(reply)]
    trigrams = list(zip(said, said[1:], said[2:]))
    return len(said) >= 4 and len(set(said)) >= 3 and all(trigrams.count(g) < 3 for g in trigrams)


def _write_corpus(root, ds, docs, heldout=None):
    n = len(docs)
    tags = DocTags(
        quality=np.ones(n),
        dup_cluster=np.full(n, -1),
        dup_canonical=np.zeros(n, bool),
        false_fact=np.zeros(n, bool),
        topic=np.zeros(n),
        noise_kind=np.zeros(n),
        heldout=np.ones(n, bool) if heldout is None else heldout,
        purchase_rank=np.zeros(n),
    )
    write_corpus(corpus_dir(root) / ds, docs, tags, {"dataset": ds})


def tiny_judge_model(vocab_size, seed=0):
    torch.manual_seed(seed)
    return Transformer(ModelShape(2, 64, 64), vocab_size)


def tiny_meta(model):
    return CheckpointMeta("tok-v1", model.shape, "lineage", "judge-v1", None, 0, {}, [])


# -- novelty -------------------------------------------------------------------------------------


def test_ngram_hashes_match_a_slow_reference_without_warnings():
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 65536, 300).astype(np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow RuntimeWarning fails the test
        for n in (1, 2, 3, 8):
            got = ngram_hashes(ids, n)
            assert got.dtype == np.uint64 and got.tolist() == ref_hashes(ids, n)
            assert ngram_hashes(ids.tolist(), n).tolist() == got.tolist()
        assert ngram_hashes(ids[:7]).dtype == np.uint64 and ngram_hashes(ids[:7]).size == 0
        assert ngram_hashes([]).size == 0 and ngram_hashes(ids[:8]).size == 1
    assert ngram_hashes([1, 2, 3, 4, 5, 6, 7, 8]).tolist() == ref_hashes(range(1, 9), 8)
    with pytest.raises(ValueError):
        ngram_hashes(ids, 0)
    with pytest.raises(ValueError):
        ngram_hashes(ids.reshape(2, -1))


def test_novelty_index_samples_by_content_and_never_spans_arrays():
    rng = np.random.default_rng(2)
    a, b = (rng.integers(16, 4096, 400).astype(np.uint16) for _ in range(2))
    idx = NoveltyIndex.build([a, b])  # n = 8, sample_mod = 8
    expected = {h for arr in (a, b) for h in ref_hashes(arr, 8) if h % 8 == 0}
    assert idx.hashes.tolist() == sorted(expected) and len(idx) == len(expected)
    spanning = {h for h in ref_hashes(np.concatenate([a, b]), 8) if h % 8 == 0} - expected
    assert spanning and not spanning & set(idx.hashes.tolist())
    # One window counts once per occurrence: a copied stretch plus a new one.
    copied = a[:100].tolist()
    fresh = rng.integers(16, 4096, 100).tolist()
    windows = [h for h in ref_hashes(copied + fresh, 8) if h % 8 == 0]
    absent = sum(h not in expected for h in windows)
    assert idx.novelty(copied + fresh) == pytest.approx(absent / len(windows))
    assert 0 < idx.novelty(copied + fresh) < 1 and idx.novelty(copied) == 0.0


def test_novelty_edge_cases(tiny_tok):
    empty = NoveltyIndex(np.zeros(0, np.uint64), sample_mod=1)
    assert empty.novelty(list(range(20, 40))) == 1.0 and empty.novelty([]) == 0.0
    unsorted = NoveltyIndex(np.array([9, 3, 3, 7], np.uint64), n=2, sample_mod=1)
    assert unsorted.hashes.tolist() == [3, 7, 9]
    for bad in ({"n": 0}, {"sample_mod": 0}, {"n": 1.5}):
        with pytest.raises(ValueError):
            NoveltyIndex(np.zeros(0, np.uint64), **bad)


def test_novelty_index_save_load_keeps_its_parameters(tmp_path):
    rng = np.random.default_rng(3)
    idx = NoveltyIndex.build([rng.integers(16, 4096, 500)], n=3, sample_mod=2)
    path = tmp_path / "deep" / "index.npy"
    idx.save(path)
    loaded = NoveltyIndex.load(path)
    assert (loaded.n, loaded.sample_mod) == (3, 2)
    assert loaded.hashes.tolist() == idx.hashes.tolist()
    assert sorted(p.name for p in path.parent.iterdir()) == ["index.json", "index.npy"]
    meta = json.loads((path.parent / "index.json").read_text(encoding="utf-8"))
    meta["hash_base"] = 31
    (path.parent / "index.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="base"):
        NoveltyIndex.load(path)


def test_novelty_save_writes_the_array_first_and_its_record_last(tmp_path, monkeypatch):
    idx = NoveltyIndex.build([np.arange(16, 400)], sample_mod=2)
    replaced = []
    real_replace = os.replace

    def recording_replace(src, dst):
        replaced.append(Path(dst).name)
        real_replace(src, dst)

    monkeypatch.setattr(novelty_module.os, "replace", recording_replace)
    idx.save(tmp_path / "index.npy")
    assert replaced == ["index.npy", "index.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json", "index.npy"]
    meta = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert meta["count"] == len(idx) > 0


def test_novelty_load_rejects_a_record_that_does_not_match_the_array(tmp_path, monkeypatch):
    path = tmp_path / "index.npy"
    NoveltyIndex.build([np.arange(16, 400)], sample_mod=2).save(path)
    bigger = NoveltyIndex.build([np.arange(16, 900)], sample_mod=2)
    real_replace = os.replace

    def crash_on_the_record(src, dst):  # the new array lands, then the build dies
        if Path(dst).suffix == ".json":
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(novelty_module.os, "replace", crash_on_the_record)
    with pytest.raises(OSError):
        bigger.save(path)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="count"):
        NoveltyIndex.load(path)
    bigger.save(path)  # a complete save repairs it
    assert NoveltyIndex.load(path).hashes.tolist() == bigger.hashes.tolist()
    meta = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    del meta["count"]
    (tmp_path / "index.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        NoveltyIndex.load(path)


def test_novelty_counts_windows_seen_in_the_prompt_as_not_new():
    rng = np.random.default_rng(5)
    corpus, prompt, fresh = (rng.integers(16, 4096, k).tolist() for k in (200, 30, 40))
    idx = NoveltyIndex.build([np.array(corpus)], sample_mod=1)
    text = corpus[:20] + prompt + fresh
    known = set(ref_hashes(corpus, 8)) | set(ref_hashes(prompt, 8))
    windows = ref_hashes(text, 8)
    expected = sum(h not in known for h in windows) / len(windows)
    assert idx.novelty(text, seen=prompt) == pytest.approx(expected)
    assert idx.novelty(text, seen=prompt) < idx.novelty(text)  # the prompt's windows were new
    assert idx.novelty(text, seen=prompt[:7]) == idx.novelty(text)  # too short for a window
    assert idx.novelty(prompt, seen=prompt) == 0.0
    empty = NoveltyIndex(np.zeros(0, np.uint64), sample_mod=1)
    assert empty.novelty(prompt + fresh, seen=prompt) == pytest.approx(
        sum(h not in set(ref_hashes(prompt, 8)) for h in ref_hashes(prompt + fresh, 8))
        / (len(prompt + fresh) - 7)
    )


def _write_root(root, rng):
    """All 8 corpora of random documents (some shorter than 8 tokens, one empty), every fourth
    document held out. Returns ``{dataset: (docs, heldout)}``."""
    written = {}
    for ds in DATASET_IDS:
        docs = [rng.integers(16, 4096, rng.integers(0, 30)).tolist() for _ in range(12)]
        docs[5] = []
        n = len(docs)
        heldout = np.arange(n) % 4 == 1
        tags = DocTags(
            quality=np.ones(n),
            dup_cluster=np.full(n, -1),
            dup_canonical=np.zeros(n, bool),
            false_fact=np.zeros(n, bool),
            topic=np.zeros(n),
            noise_kind=np.zeros(n),
            heldout=heldout,
            purchase_rank=np.zeros(n),
        )
        write_corpus(corpus_dir(root) / ds, docs, tags, {"dataset": ds})
        written[ds] = (docs, heldout)
    return written


def test_build_novelty_index_skips_heldout_docs_and_boundaries(tmp_path, monkeypatch):
    written = _write_root(tmp_path, np.random.default_rng(4))
    path = build_novelty_index(tmp_path)
    assert path == novelty_path(tmp_path)
    idx = NoveltyIndex.load(path)
    assert (idx.n, idx.sample_mod) == (8, 8)

    def sampled(ids):
        return {h for h in ref_hashes(ids, 8) if h % 8 == 0}

    kept = set().union(
        *(sampled(d) for docs, held in written.values() for d, h in zip(docs, held) if not h)
    )
    assert idx.hashes.tolist() == sorted(kept)
    held_only = set().union(
        *(sampled(d) for docs, held in written.values() for d, h in zip(docs, held) if h)
    )
    every_window = set().union(
        *(sampled([t for d in docs for t in d]) for docs, _ in written.values())
    )
    spanning = every_window - kept - held_only
    assert held_only - kept and spanning  # both kinds exist, and neither is in the index

    monkeypatch.setattr(novelty_module, "_CHUNK", 3)  # chunk edges in every corpus
    assert NoveltyIndex.load(build_novelty_index(tmp_path)).hashes.tolist() == sorted(kept)


# -- the judge -----------------------------------------------------------------------------------


def test_nll_per_token_scores_text_after_bos_in_one_call(tiny_tok):
    lm = RecordingLM(tiny_tok, ctx_len=10, score=lambda ctx, cont: -2.0 * len(cont))
    judge = Judge(lm, tiny_tok)
    lists = [[20, 21, 22], [], [0, 1, 2, 4, 15], list(range(20, 40)), np.array([1, 30, 4, 31])]
    assert judge.nll_per_token(lists) == [2.0, INF, INF, 2.0, 2.0]
    [(contexts, continuations)] = lm.scored
    assert contexts == [[tiny_tok.bos_id]] * 5
    assert continuations == [[20, 21, 22], [], [], list(range(20, 29)), [30, 31]]
    broken = Judge(RecordingLM(tiny_tok, score=lambda ctx, cont: math.nan), tiny_tok)
    assert broken.nll_per_token([[20, 21]]) == [INF]
    assert judge.nll_per_token([]) == []


def test_nll_per_token_on_a_real_model(tiny_tok):
    model = tiny_judge_model(tiny_tok.vocab_size)
    lm = TorchLM(model, tiny_tok, CPU)
    judge = Judge(lm, tiny_tok)
    text = tiny_tok.encode("The little dog ran to the park and played all day.")
    longer = list(range(20, 220))  # beyond the 64-token span
    got = judge.nll_per_token([text, longer, [], [1, 2, 4]])
    [direct] = lm.score_continuations([[tiny_tok.bos_id]], [text])
    assert got[0] == pytest.approx(-direct.sum_logprob / len(text))
    assert got[1] == pytest.approx(judge.nll_per_token([longer[:63]])[0])
    assert math.isfinite(got[0]) and math.isfinite(got[1]) and got[2:] == [INF, INF]


def test_judge_load_needs_only_its_directory(tmp_path, tiny_tok, tiny_tok_path):
    model = tiny_judge_model(tiny_tok.vocab_size)
    built = tmp_path / "built"
    save_checkpoint(model, tiny_meta(model), built)
    shutil.copyfile(tiny_tok_path, built / TOKENIZER_NAME)
    cal = JudgeCalibration(2.5, 5.5, 4.75)
    (built / CALIBRATION_NAME).write_text(json.dumps(asdict(cal)), encoding="utf-8")
    moved = Path(shutil.move(built, tmp_path / "elsewhere"))
    judge = Judge.load(moved, device="cpu")
    assert judge.calibration == cal and judge.tok.vocab_size == tiny_tok.vocab_size
    ids = tiny_tok.encode("A cat sat on a warm mat.")
    expected = Judge(TorchLM(model, tiny_tok, CPU), tiny_tok).nll_per_token([ids])
    assert judge.nll_per_token([ids]) == pytest.approx(expected)

    other = tiny_judge_model(600)
    save_checkpoint(other, tiny_meta(other), moved)
    with pytest.raises(ValueError, match="vocabulary"):
        Judge.load(moved, device="cpu")
    (moved / CALIBRATION_NAME).unlink()  # no calibration: not a complete judge
    with pytest.raises(FileNotFoundError):
        Judge.load(moved, device="cpu")


def test_calibrate_scores_heldout_openings_and_ai_replies(tiny_data_root, tiny_tok):
    lm = RecordingLM(tiny_tok, score=lambda ctx, cont: -(float(len(cont)) ** 2))  # nll = length
    cal = calibrate(Judge(lm, tiny_tok), tiny_data_root)
    (_, openings), (_, replies) = lm.scored
    creative = Corpus.open(corpus_dir(tiny_data_root) / "creative")
    expected_openings = [creative.doc(int(i))[1:97].tolist() for i in heldout_docs(creative)]
    assert all(creative.doc(int(i))[0] == tiny_tok.bos_id for i in heldout_docs(creative))
    conv = Corpus.open(corpus_dir(tiny_data_root) / "conversations")
    expected_replies = [
        r
        for i in heldout_docs(conv)
        for r in ref_replies(conv.doc(int(i)).tolist(), tiny_tok.ai_id, tiny_tok.end_id)
        if ref_word_checks(tiny_tok.decode(r))
    ]
    assert sorted(openings) == sorted(expected_openings)
    assert sorted(replies) == sorted(expected_replies) and len(replies) >= 5
    # The calibration scores a reply exactly as is_well_formed scores the reply's text.
    texts = [tiny_tok.decode(r) for r in replies]
    assert [tiny_tok.encode(t.strip()) for t in texts] == replies
    lengths = [len(o) for o in expected_openings]
    p10, p90 = np.percentile(lengths, [10, 90])
    reply_p90 = np.percentile([len(r) for r in expected_replies], 90)
    assert asdict(cal) == pytest.approx(
        {"creative_p10": p10, "creative_p90": p90, "conv_reply_p90": reply_p90}
    )


def test_calibration_scores_only_replies_that_pass_the_word_checks(tmp_path, tiny_tok):
    good = ["I like to play in the park.", "The red bus goes up the hill.", "We had 2 pies."]
    bad = [
        "Yes.",
        "Thank you so",
        "go go go go",
        "yes yes yes yes",
        "the cat sat the cat sat the cat sat",
    ]
    turns = [t for reply in good + bad for t in (("user", "Hi"), ("ai", reply))]
    _write_corpus(tmp_path, "conversations", [encode_chat(tiny_tok, turns)])
    _write_corpus(tmp_path, "creative", [encode_doc(tiny_tok, CORPUS_TEXT)])
    assert all(ref_word_checks(r) for r in good) and not any(ref_word_checks(r) for r in bad)
    lm = RecordingLM(tiny_tok, score=lambda ctx, cont: -2.0 * len(cont))
    calibrate(Judge(lm, tiny_tok), tmp_path)
    (_, openings), (_, replies) = lm.scored
    assert openings == [tiny_tok.encode(CORPUS_TEXT)[:96]]
    assert replies == [tiny_tok.encode(r) for r in good]
    for reply in good + bad:  # is_well_formed applies the very same word checks
        judge = StubJudge(tiny_tok, lambda ids: 0.0)
        assert is_well_formed(reply, judge) == (reply in good)


def test_calibration_without_eligible_replies_is_a_clear_error(
    tiny_data_root, tiny_tok, monkeypatch
):
    monkeypatch.setattr(judge_module, "MIN_WORDS", 1000)  # no real reply is that long
    lm = RecordingLM(tiny_tok, score=lambda ctx, cont: -2.0 * len(cont))
    with pytest.raises(ValueError, match="no held-out conversation replies"):
        calibrate(Judge(lm, tiny_tok), tiny_data_root)


def test_calibrate_samples_deterministically(tiny_data_root, tiny_tok, monkeypatch):
    monkeypatch.setattr(judge_module, "CALIBRATION_DOCS", 2)
    runs = []
    for _ in range(2):
        lm = RecordingLM(tiny_tok, score=lambda ctx, cont: -float(len(cont)))
        calibrate(Judge(lm, tiny_tok), tiny_data_root)
        runs.append(lm.scored)
    assert runs[0] == runs[1]
    (_, openings), (_, replies) = runs[0]
    assert len(openings) == 2 and len(replies) == 2
    broken = Judge(RecordingLM(tiny_tok, score=lambda ctx, cont: math.nan), tiny_tok)
    with pytest.raises(ValueError, match="calibrate"):
        calibrate(broken, tiny_data_root)


def test_build_judge_writes_a_self_contained_judge(tiny_data_root, tiny_tok, tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    out = judge_dir(root)
    out.mkdir(parents=True)
    (out / CALIBRATION_NAME).write_text("{}", encoding="utf-8")  # a previous judge's
    calls = []

    def fake_train_run(cfg, **kwargs):  # what the real run leaves behind: a checkpoint
        assert not (kwargs["out_dir"] / CALIBRATION_NAME).exists()
        calls.append((cfg, kwargs))
        model = tiny_judge_model(tiny_tok.vocab_size)
        save_checkpoint(model, tiny_meta(model), kwargs["out_dir"])
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(judge_module, "train_run", fake_train_run)
    events = []
    built = build_judge(root, device="cpu", on_event=events.append, token_budget=65536)
    assert built == out
    [(cfg, kwargs)] = calls
    assert cfg == replace(judge_train_config(), token_budget=65536)
    assert kwargs == {
        "out_dir": out,
        "data_root": root,
        "device": CPU,
        "on_event": events.append,
        "resume": False,  # nothing to resume: a previous judge left no resume state
    }
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "calibration.json",
        "meta.json",
        "model.safetensors",
        "tokenizer.json",
        "train_config.json",
    ]
    record = json.loads((out / TRAIN_CONFIG_NAME).read_text(encoding="utf-8"))
    assert record == json.loads(cfg.to_json())
    moved = Path(shutil.move(out, tmp_path / "moved"))
    judge = Judge.load(moved, device="cpu")
    assert judge.calibration == calibrate(judge, root)
    assert judge.calibration.creative_p10 <= judge.calibration.creative_p90


def _small_judge_config(monkeypatch):
    """Make build_judge train a 2-layer, 64-wide judge on 1024-token steps (everything else as
    the real recipe), so a build takes about a second."""
    real = judge_module.judge_train_config

    def small(seed=1234):
        return replace(real(seed), shape=ModelShape(2, 64, 64), batch_tokens=1024)

    monkeypatch.setattr(judge_module, "judge_train_config", small)
    return small


def _spy_train_run(monkeypatch):
    """Record the ``resume`` flag of every train_run that build_judge starts."""
    flags = []

    def spy(cfg, **kwargs):
        flags.append(kwargs.get("resume", False))
        return train_run(cfg, **kwargs)

    monkeypatch.setattr(judge_module, "train_run", spy)
    return flags


def _interrupted_judge_build(root, cfg, after_steps):
    """What a judge build leaves behind when it stops after ``after_steps`` steps."""
    train_run(
        cfg,
        out_dir=judge_dir(root),
        data_root=root,
        device="cpu",
        _hooks=TrainHooks(stop_after_steps=after_steps),
    )
    assert (judge_dir(root) / "resume").is_dir()


def test_build_judge_resumes_an_interrupted_build(tiny_data_root, tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    small = _small_judge_config(monkeypatch)
    _interrupted_judge_build(root, replace(small(), token_budget=1024 * 6), after_steps=3)
    flags = _spy_train_run(monkeypatch)
    events = []
    built = build_judge(root, device="cpu", on_event=events.append, token_budget=1024 * 6)
    assert flags == [True]
    assert [e.step for e in events if isinstance(e, Progress)] == [6]  # steps 4-6 only
    assert [e.status for e in events if isinstance(e, Done)] == ["completed"]
    assert not [p for p in built.iterdir() if p.name.startswith("resume")]
    result = json.loads((built / "result.json").read_text(encoding="utf-8"))
    assert result["steps"] == 6 and result["status"] == "completed"
    judge = Judge.load(built, device="cpu")
    assert math.isfinite(judge.calibration.creative_p90)
    assert judge.calibration == calibrate(judge, root)


def test_build_judge_starts_fresh_over_another_builds_state(tiny_data_root, tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    small = _small_judge_config(monkeypatch)
    _interrupted_judge_build(root, replace(small(), token_budget=1024 * 6), after_steps=3)
    flags = _spy_train_run(monkeypatch)
    events = []
    built = build_judge(root, device="cpu", on_event=events.append, token_budget=1024 * 4)
    assert flags == [False]  # a 4-step build cannot continue a 6-step one
    assert [e.step for e in events if isinstance(e, Progress)] == [1, 4]
    assert not [p for p in built.iterdir() if p.name.startswith("resume")]
    assert math.isfinite(Judge.load(built, device="cpu").calibration.conv_reply_p90)


@pytest.mark.slow
def test_build_judge_resume_smoke(tiny_data_root, tmp_path, monkeypatch):
    """The real judge recipe: interrupted after 1 of 2 steps, then finished by build_judge."""
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    _interrupted_judge_build(root, replace(judge_train_config(), token_budget=32768 * 2), 1)
    flags = _spy_train_run(monkeypatch)
    events = []
    built = build_judge(root, device="cpu", on_event=events.append, token_budget=32768 * 2)
    assert flags == [True] and [e.step for e in events if isinstance(e, Progress)] == [2]
    judge = Judge.load(built, device="cpu")
    assert math.isfinite(judge.calibration.creative_p90)


def test_build_judge_recalibrates_a_trained_judge_without_retraining(
    tiny_data_root, tmp_path, monkeypatch
):
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    _small_judge_config(monkeypatch)
    real_calibrate = judge_module.calibrate

    def failing_calibrate(judge, data_root):
        raise RuntimeError("the machine went to sleep")

    monkeypatch.setattr(judge_module, "calibrate", failing_calibrate)
    with pytest.raises(RuntimeError, match="sleep"):
        build_judge(root, device="cpu", token_budget=1024 * 4)
    out = judge_dir(root)
    assert (out / "model.safetensors").exists() and not (out / CALIBRATION_NAME).exists()
    weights = (out / "model.safetensors").read_bytes()
    monkeypatch.setattr(judge_module, "calibrate", real_calibrate)
    flags = _spy_train_run(monkeypatch)
    assert build_judge(root, device="cpu", token_budget=1024 * 4) == out
    assert flags == []  # trained already: only calibrated
    assert (out / "model.safetensors").read_bytes() == weights
    judge = Judge.load(out, device="cpu")
    assert judge.calibration == real_calibrate(judge, root)
    build_judge(root, device="cpu", token_budget=1024 * 4)  # a complete judge: recalibrated only
    assert flags == [] and Judge.load(out, device="cpu").calibration == judge.calibration
    build_judge(root, device="cpu", token_budget=1024 * 5)  # another recipe: trained afresh
    assert flags == [False]
    record = json.loads((out / TRAIN_CONFIG_NAME).read_text(encoding="utf-8"))
    assert record["token_budget"] == 1024 * 5


def test_build_judge_refuses_a_run_that_did_not_complete(tiny_data_root, tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    _small_judge_config(monkeypatch)

    def unstable_train_run(cfg, **kwargs):  # every loss is NaN: the third spike stops the run
        return train_run(cfg, **kwargs, _hooks=TrainHooks(loss_override=lambda s, x: math.nan))

    monkeypatch.setattr(judge_module, "train_run", unstable_train_run)
    with pytest.raises(JudgeBuildError, match="unstable_stopped"):
        build_judge(root, device="cpu", token_budget=1024 * 6)
    out = judge_dir(root)
    assert (out / "model.safetensors").exists()  # the trainer kept its last good weights
    for name in (CALIBRATION_NAME, TRAIN_CONFIG_NAME, TOKENIZER_NAME):
        assert not (out / name).exists()  # not calibrated, not frozen
    with pytest.raises(FileNotFoundError):
        Judge.load(out, device="cpu")
    flags = _spy_train_run(monkeypatch)  # the real trainer again
    build_judge(root, device="cpu", token_budget=1024 * 6)
    assert flags == [False]  # an unfinished run is never taken as trained
    assert math.isfinite(Judge.load(out, device="cpu").calibration.creative_p90)


def test_a_failed_rebuild_never_passes_for_the_earlier_judge(tiny_data_root, tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(tiny_data_root, root)
    _small_judge_config(monkeypatch)
    build_judge(root, device="cpu", token_budget=1024 * 4)  # a complete judge of recipe A
    out = judge_dir(root)

    def unstable_train_run(cfg, **kwargs):
        return train_run(cfg, **kwargs, _hooks=TrainHooks(loss_override=lambda s, x: math.nan))

    monkeypatch.setattr(judge_module, "train_run", unstable_train_run)
    with pytest.raises(JudgeBuildError):  # recipe B overwrites the checkpoint, then fails
        build_judge(root, device="cpu", token_budget=1024 * 6)
    assert not (out / TRAIN_CONFIG_NAME).exists()  # recipe A's record went before training
    flags = _spy_train_run(monkeypatch)
    build_judge(root, device="cpu", token_budget=1024 * 4)
    assert flags == [False]  # recipe A is trained again, not taken from B's leftover weights


def test_judge_config_is_the_reference_recipe():
    c = judge_train_config()
    assert (c.shape.n_layer, c.shape.d_model, c.shape.ctx_len) == (8, 384, 256)
    assert c.mixture == {ds: 1.0 for ds in DATASET_IDS} and c.run_id == "judge-v1"
    assert (c.prep.cleaning, c.prep.dedup, c.prep.fact_check) == ("thorough", True, True)
    assert (c.boldness, c.batch_tokens, c.seed) == (0.4, 32768, 1234)
    assert judge_train_config(seed=7).seed == 7
    assert 15_000_000 < c.shape.param_count() < 16_500_000


# -- well-formed replies -------------------------------------------------------------------------


def test_well_formed_edge_cases(tiny_tok):
    seen = []
    judge = StubJudge(tiny_tok, lambda ids: seen.append(ids) or 3.0)
    assert is_well_formed("  I like big dogs \n", judge)  # exactly 4 words
    assert seen == [tiny_tok.encode("I like big dogs")]  # judged on the stripped reply
    assert is_well_formed("I have 3 cats", judge)  # a number is a word
    assert is_well_formed("the cat sat and the cat sat on a mat", judge)  # a 3-gram twice
    assert is_well_formed("I like the park", judge) and is_well_formed("go go stop now", judge)
    seen.clear()
    for reply in (
        "I like dogs",
        "",
        "  ... !!! ???",
        "The cat sat, the cat sat, THE CAT SAT.",
        "go go go go",  # 4 words but only 1 different
        "yes yes yes yes",
        "Go go GO stop",  # 2 different words
    ):
        assert not is_well_formed(reply, judge)
    assert seen == []  # the judge is only asked about replies that pass the word checks
    at_limit = StubJudge(tiny_tok, lambda ids: 5.0)  # conv_reply_p90 is 5.0
    assert is_well_formed("I like to play outside.", at_limit)
    for nll in (math.nan, math.inf, 5.0001):
        assert not is_well_formed(
            "I like to play outside.", StubJudge(tiny_tok, lambda ids, v=nll: v)
        )


# -- creativity ----------------------------------------------------------------------------------

ANIMALS = ["dog", "cat", "owl", "fox", "bee", "cow", "pig", "hen", "ant", "elk", "yak", "emu"]
PLACES = ["river", "forest", "garden", "harbor", "meadow", "castle"]


def _story(k):
    if k == 5:
        return ""
    if k == 6:
        return "  123 456 !!!  "  # no words
    if k == 7:
        return "the red cat sat, the red cat sat"  # 6 word 3-grams, 4 different
    if k == 8:
        return "hello friend"  # no 3-gram: nothing repeated
    return f"  the {ANIMALS[k % 12]} found a shiny stone near the {PLACES[k % 6]} today  "


def test_creativity_scores_every_story_and_the_mix(tiny_tok):
    prompt_index = {
        tuple(encode_chat(tiny_tok, [("user", p)], add_generation_prompt=True)): k
        for k, p in enumerate(STORY_PROMPTS)
    }
    lm = RecordingLM(tiny_tok, reply=lambda prompt: _story(prompt_index[tuple(prompt)]))
    idx = NoveltyIndex.build([np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)], sample_mod=1)
    seen = []
    judge = StubJudge(tiny_tok, lambda ids: seen.append(ids) or 3.0)  # coherence 0.75
    category, items = score_creativity(lm, tiny_tok, judge, idx, seed=7)

    [(prompts, kwargs)] = lm.generated  # one batched call
    assert list(map(tuple, prompts)) == list(prompt_index)
    assert kwargs == {"max_new_tokens": 96, "temperature": 0.9, "top_p": 0.95, "seed": 7}
    stories = [_story(k).strip() for k in range(24)]
    assert seen == [tiny_tok.encode(s) for s in stories]
    expected = [
        0.75
        * (0.4 + 0.6 * idx.novelty(tiny_tok.encode(s), seen=tiny_tok.encode(p)))
        * (1 - ref_repetitiveness(s))
        if ref_words(s)
        else 0.0
        for s, p in zip(stories, STORY_PROMPTS)
    ]
    assert [r.score for r in items] == pytest.approx(expected)
    assert items[5].score == items[6].score == 0.0 and min(expected[:5]) > 0.3
    assert ref_repetitiveness(stories[7]) == pytest.approx(1 / 3)
    assert ref_repetitiveness(stories[8]) == ref_repetitiveness(stories[0]) == 0.0
    assert items[8].score == pytest.approx(
        0.75 * (0.4 + 0.6 * idx.novelty(tiny_tok.encode("hello friend")))
    )
    assert [r.item_id for r in items] == [f"story-{k:02d}" for k in range(24)]
    assert [r.output for r in items] == stories and {r.category for r in items} == {"creativity"}
    topics = [r.tags for r in items]
    assert all(t[0] == "fmt:story" and t[1].startswith("topic:") for t in topics)
    assert len({t[1] for t in topics}) == 6
    raw = sum(expected) / 24
    d = ref_distinct_2(stories)
    assert 0 < d < 1
    assert category.raw == pytest.approx(raw) and category.n == 24
    assert category.score == pytest.approx(100 * raw * (0.5 + 0.5 * d))

    score_creativity(lm, tiny_tok, judge, idx, seed=1, max_new_tokens=40)
    assert lm.generated[-1][1]["max_new_tokens"] == 40 and lm.generated[-1][1]["seed"] == 1


def test_creativity_degenerate_outputs_score_finite(tiny_tok):
    idx = NoveltyIndex.build([np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)])
    fluent = StubJudge(tiny_tok, lambda ids: 2.5)
    specials = score_creativity(TokenLM([4, 2, 1, 0, 3, 15, 15]), tiny_tok, fluent, idx)
    assert specials[0].score == 0.0 and specials[0].raw == 0.0
    assert all(r.score == 0.0 and r.output == "" for r in specials[1])
    punct = ScriptedLM(tiny_tok, reply=lambda p: "... !!! ???")
    assert score_creativity(punct, tiny_tok, fluent, idx)[0].score == 0.0
    repeated = ScriptedLM(tiny_tok, reply=lambda p: "the " * 60)
    flat = StubJudge(tiny_tok, lambda ids: 4.0)
    flat.calibration = JudgeCalibration(4.0, 4.0, 4.0)  # no spread at all
    weird = [StubJudge(tiny_tok, lambda ids, v=v: v) for v in (math.nan, INF, -INF, 0.0, 1e9)]
    for lm in (repeated, punct, ScriptedLM(tiny_tok, reply=lambda p: CORPUS_TEXT)):
        for judge in (fluent, flat, *weird):
            category, items = score_creativity(lm, tiny_tok, judge, idx)
            assert math.isfinite(category.score) and 0 <= category.score <= 100
            assert all(math.isfinite(r.score) and 0 <= r.score <= 1 for r in items)
    assert score_creativity(repeated, tiny_tok, weird[0], idx)[0].score == 0.0  # NaN loss


def test_a_story_that_echoes_its_prompt_is_not_new(tiny_tok):
    idx = NoveltyIndex.build([np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)])
    fluent = StubJudge(tiny_tok, lambda ids: 2.5)  # coherence 0.875
    echo = ScriptedLM(tiny_tok, reply=lambda prompt: tiny_tok.decode(prompt))  # the user text
    category, items = score_creativity(echo, tiny_tok, fluent, idx)
    assert [r.output for r in items] == list(STORY_PROMPTS)
    # No corpus holds the prompts, so without the prompt check every echo would count as new.
    assert all(idx.novelty(tiny_tok.encode(p)) > 0.5 for p in STORY_PROMPTS)
    assert [r.score for r in items] == pytest.approx(
        [0.875 * 0.4 * (1 - ref_repetitiveness(p)) for p in STORY_PROMPTS]
    )  # novelty 0: an echo is worth what a fluent copy of the training data is
    novel = ScriptedLM(
        tiny_tok,
        reply=lambda p: f"a curious robot {len(p)} painted purple stars across the quiet ocean sky",
    )
    assert category.score < 0.65 * score_creativity(novel, tiny_tok, fluent, idx)[0].score


def test_endless_repetition_scores_about_zero(tiny_tok):
    idx = NoveltyIndex.build([np.array(tiny_tok.encode(CORPUS_TEXT * 3), np.uint16)])
    fluent = StubJudge(tiny_tok, lambda ids: 2.5)  # the judge finds loops easy: coherence 0.875

    def score(reply):
        return score_creativity(ScriptedLM(tiny_tok, reply=lambda p: reply), tiny_tok, fluent, idx)

    looping, items = score("the the the " * 30)
    assert looping.score < 1 and all(r.score < 0.01 for r in items)
    novel = score("a curious robot painted purple stars across the quiet ocean sky")[0].score
    repeated, items = score("purple robots dance on frozen moons " * 16)  # ~ the story budget
    assert novel > 40 and repeated.score < 5 and repeated.score < novel / 10
    assert all(r.score == pytest.approx(0.875 * 6 / 94, rel=0.05) for r in items)


def test_story_prompts_are_simple_varied_and_not_in_any_generator():
    assert len(STORY_PROMPTS) == len(set(STORY_PROMPTS)) == 24
    assert all(re.fullmatch(r"Write a short story about [a-z ]+\.", p) for p in STORY_PROMPTS)
    assert all(len(p.split()) <= 14 for p in STORY_PROMPTS)
    roots = (Path(airace_content.__file__).parent, Path(airace_ml.skills.__file__).parent)
    sources = [
        f.read_text(encoding="utf-8").lower()
        for root in roots
        for f in root.rglob("*")
        if f.suffix in (".py", ".json", ".txt")
    ]
    assert len(sources) > 10
    for prompt in STORY_PROMPTS:
        subject = prompt.removeprefix("Write a short story about ").removesuffix(".").lower()
        assert not any(prompt.lower()[:-1] in s or subject in s for s in sources), prompt


def test_run_benchmarks_measures_creativity_with_a_real_judge(tiny_lm, tiny_tok, fixture_texts):
    judge_lm = TorchLM(tiny_judge_model(tiny_tok.vocab_size, seed=1), tiny_tok, CPU)
    judge = Judge(judge_lm, tiny_tok, JudgeCalibration(5.0, 7.0, 6.0))
    idx = NoveltyIndex.build([np.array(encode_doc(tiny_tok, t), np.uint16) for t in fixture_texts])

    def run():
        return run_benchmarks(
            tiny_lm,
            tiny_tok,
            categories=["language", "creativity"],
            judge=judge,
            novelty=idx,
            max_items_per_category=4,
            seed=3,
        )

    a, b = run(), run()
    assert a.missing == [] and list(a.scores) == ["language", "creativity"]
    creativity = a.scores["creativity"]
    assert creativity.n == 24 and math.isfinite(creativity.score) and 0 <= creativity.score <= 100
    assert creativity == score_creativity(tiny_lm, tiny_tok, judge, idx, seed=3)[0]
    stories = [(r.item_id, r.score, r.output) for r in a.items if r.category == "creativity"]
    assert len(stories) == 24 and any(r[2] for r in stories)
    assert a.scores == b.scores
    assert stories == [
        (r.item_id, r.score, r.output) for r in b.items if r.category == "creativity"
    ]
