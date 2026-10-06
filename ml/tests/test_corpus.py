import json

import numpy as np
import pytest

from airace_ml.data.corpus import (
    DATASET_IDS,
    NOISE_KINDS,
    TOPICS,
    Corpus,
    DocTags,
    write_corpus,
)
from airace_ml.data.prep import PrepConfig, eligible_docs, heldout_docs
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok


def _tags(n, **over):
    base = {"quality": np.full(n, 0.9, np.float16), "dup_cluster": np.full(n, -1, np.int32),
            "dup_canonical": np.zeros(n, bool), "false_fact": np.zeros(n, bool), "topic": np.zeros(n, np.uint8),
            "noise_kind": np.zeros(n, np.uint8), "heldout": np.zeros(n, bool),
            "purchase_rank": (np.arange(n, dtype=np.float32) / n)}
    base.update(over); return DocTags(**base)

def test_roundtrip(tmp_path):
    docs = [[1, 5, 6], [1, 7], [1, 8, 9, 10]]
    write_corpus(tmp_path, docs, _tags(3), {"dataset": "web"})
    c = Corpus.open(tmp_path)
    assert c.n_docs == 3 and c.n_tokens == 9 and c.doc(2).tolist() == [1, 8, 9, 10] and c.doc(0).dtype == np.uint16

def test_filters(tmp_path):
    q = np.array([0.1, 0.3, 0.5, 0.7], np.float16)
    write_corpus(tmp_path, [[1, 2]] * 4, _tags(4, quality=q), {}); c = Corpus.open(tmp_path)
    assert [len(eligible_docs(c, PrepConfig(cleaning=k))) for k in ("light", "standard", "thorough")] == [3, 2, 1]

def test_dedup_factcheck_heldout(tmp_path):
    t = _tags(5, dup_cluster=np.array([0, 0, 0, -1, -1], np.int32), dup_canonical=np.array([1, 0, 0, 0, 0], bool),
              false_fact=np.array([0, 0, 0, 1, 0], bool), heldout=np.array([0, 0, 0, 0, 1], bool))
    write_corpus(tmp_path, [[1, 2]] * 5, t, {}); c = Corpus.open(tmp_path)
    assert eligible_docs(c, PrepConfig(dedup=True, fact_check=True)).tolist() == [0]
    assert 4 not in eligible_docs(c, PrepConfig()).tolist()

def test_purchase_nested(tmp_path):
    write_corpus(tmp_path, [[1, 2]] * 100, _tags(100), {}); c = Corpus.open(tmp_path)
    a, b, f = (set(eligible_docs(c, PrepConfig(), x).tolist()) for x in (0.25, 0.5, 1.0))
    assert a <= b <= f and len(a) == 25 and len(b) == 50


def test_dedup_and_factcheck_are_independent_switches(tmp_path):
    t = _tags(4, dup_cluster=np.array([0, 0, -1, -1], np.int32), dup_canonical=np.array([1, 0, 0, 0], bool),
              false_fact=np.array([0, 0, 1, 0], bool))
    write_corpus(tmp_path, [[1, 2]] * 4, t, {}); c = Corpus.open(tmp_path)
    assert eligible_docs(c, PrepConfig()).tolist() == [0, 1, 2, 3]
    assert eligible_docs(c, PrepConfig(dedup=True)).tolist() == [0, 2, 3]
    assert eligible_docs(c, PrepConfig(fact_check=True)).tolist() == [0, 1, 3]

def test_quality_threshold_boundary_survives_float16(tmp_path):
    # A document stored at exactly the "standard" threshold must pass despite float16 rounding.
    q = np.array([0.40, 0.39], np.float16)
    write_corpus(tmp_path, [[1, 2]] * 2, _tags(2, quality=q), {}); c = Corpus.open(tmp_path)
    assert eligible_docs(c, PrepConfig(cleaning="standard")).tolist() == [0]

def test_heldout_docs_are_excluded_from_training_and_listed(tmp_path):
    h = np.array([0, 1, 0, 1], bool)
    write_corpus(tmp_path, [[1, 2]] * 4, _tags(4, heldout=h), {}); c = Corpus.open(tmp_path)
    assert heldout_docs(c).tolist() == [1, 3]
    assert set(heldout_docs(c)).isdisjoint(eligible_docs(c, PrepConfig(cleaning="light")))

def test_purchase_zero_is_empty(tmp_path):
    write_corpus(tmp_path, [[1, 2]] * 10, _tags(10), {}); c = Corpus.open(tmp_path)
    assert len(eligible_docs(c, PrepConfig(), 0.0)) == 0

def test_prep_config_roundtrip_and_validation():
    p = PrepConfig(cleaning="thorough", dedup=True, fact_check=True, variety="balanced")
    assert PrepConfig.from_dict(p.to_dict()) == p
    assert PrepConfig.from_dict({}) == PrepConfig()
    with pytest.raises(ValueError, match="cleaning"):
        PrepConfig(cleaning="extreme")
    with pytest.raises(ValueError, match="variety"):
        PrepConfig(variety="diverse")

def test_doc_out_of_range_and_empty_corpus(tmp_path):
    write_corpus(tmp_path / "a", [[1, 2]], _tags(1), {}); c = Corpus.open(tmp_path / "a")
    for bad in (-1, 1):
        with pytest.raises(IndexError):
            c.doc(bad)
    write_corpus(tmp_path / "e", [], _tags(0), {}); e = Corpus.open(tmp_path / "e")
    assert e.n_docs == 0 and e.n_tokens == 0
    assert len(eligible_docs(e, PrepConfig())) == 0 and len(heldout_docs(e)) == 0

def test_write_corpus_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        write_corpus(tmp_path / "big", [[1, 70000]], _tags(1), {})
    with pytest.raises(ValueError, match="quality"):
        write_corpus(tmp_path / "len", [[1, 2], [1, 3]], _tags(1), {})

def test_info_roundtrip_and_tag_dtypes(tmp_path):
    write_corpus(tmp_path, [[1, 2]], _tags(1), {"dataset": "web", "note": "ünï"})
    c = Corpus.open(tmp_path)
    assert c.info == {"dataset": "web", "note": "ünï"}
    assert json.loads((tmp_path / "info.json").read_text(encoding="utf-8")) == c.info
    t = c.tags
    assert (t.quality.dtype, t.dup_cluster.dtype, t.dup_canonical.dtype, t.false_fact.dtype) == (
        np.float16, np.int32, np.bool_, np.bool_)
    assert (t.topic.dtype, t.noise_kind.dtype, t.heldout.dtype, t.purchase_rank.dtype) == (
        np.uint8, np.uint8, np.bool_, np.float32)


def test_tiny_data_root_layout(tiny_data_root, tiny_tok):
    assert tokenizer_path(tiny_data_root).is_file()
    assert Tok.load(tokenizer_path(tiny_data_root)).encode("The cat sat.") == tiny_tok.encode("The cat sat.")
    for ds in DATASET_IDS:
        c = Corpus.open(corpus_dir(tiny_data_root) / ds)
        t = c.tags
        assert c.n_docs >= 40 and c.info["dataset"] == ds
        assert all(len(c.doc(i)) > 1 and c.doc(i)[0] == tiny_tok.bos_id for i in range(c.n_docs))
        assert float(t.quality.min()) < 0.15 and float(t.quality.max()) > 0.65
        assert len(set(t.dup_cluster[t.dup_cluster >= 0].tolist())) >= 2
        assert t.dup_canonical[t.dup_cluster >= 0].any() and not t.dup_canonical[t.dup_cluster < 0].any()
        assert t.false_fact.sum() >= 3
        assert len(set(t.topic.tolist())) >= 2 and int(t.topic.max()) < len(TOPICS)
        assert int(t.noise_kind.max()) < len(NOISE_KINDS)
        assert t.heldout.sum() >= 1 and abs(t.heldout.mean() - 0.1) < 0.05
        assert np.array_equal(np.sort(t.purchase_rank), np.arange(c.n_docs, dtype=np.float32) / c.n_docs)
        assert len(heldout_docs(c)) == t.heldout.sum()
        for prep in (PrepConfig(), PrepConfig(cleaning="thorough", dedup=True, fact_check=True)):
            assert len(eligible_docs(c, prep)) >= 5

def test_tiny_data_root_conversations_are_chats(tiny_data_root, tiny_tok):
    c = Corpus.open(corpus_dir(tiny_data_root) / "conversations")
    for i in range(c.n_docs):
        d = c.doc(i).tolist()
        assert d[:2] == [tiny_tok.bos_id, tiny_tok.user_id] and d[-1] == tiny_tok.end_id
        assert tiny_tok.ai_id in d
