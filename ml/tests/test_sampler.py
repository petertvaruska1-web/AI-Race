import json

import numpy as np
import pytest

from airace_ml.data.corpus import DATASET_IDS, Corpus, DocTags, write_corpus
from airace_ml.data.prep import PrepConfig, eligible_docs
from airace_ml.data.sampler import DocPool, MixtureError, MixtureSampler, load_pools
from airace_ml.paths import corpus_dir
from airace_ml.tokenizer import encode_chat, encode_doc


def _pool(n, length=10, topics=None):
    return DocPool.from_docs([[1] + list(range(2, 2 + length - 1))] * n)

def test_proportions_and_shapes():
    s = MixtureSampler({"a": _pool(5), "b": _pool(5)}, {"a": 0.75, "b": 0.25}, seq_len=32, seed=0)
    for _ in range(100):
        x, y = s.next_batch(40)
    assert x.shape == (40, 32) and (y[:, :-1] == x[:, 1:]).all()
    st = s.stats(); assert 0.72 < st["a"] / (st["a"] + st["b"]) < 0.78

@pytest.mark.parametrize(
    ("weights", "match"),
    [  # the error names the offending dataset (not a letter every message has), or the sum
        ({"alpha": 0.0}, "sum"),
        ({"alpha": -1.0, "beta": 1.0}, "'alpha'"),
        ({"zzz": 1.0}, "'zzz'"),
        ({"empty": 1.0}, "'empty'"),
        ({"blank": 1.0}, "'blank'"),
    ],
)
def test_invalid_mixtures(weights, match):  # Review Focus 3
    pools = {"alpha": _pool(3), "beta": _pool(3), "empty": DocPool.from_docs([]), "blank": DocPool.from_docs([[], []])}
    with pytest.raises(MixtureError, match=match) as e:
        MixtureSampler(pools, weights, seq_len=16, seed=0)
    assert not any(f"'{k}'" in str(e.value) for k in weights if f"'{k}'" != match)  # no other name

def test_zero_weight_empty_pool_is_fine():
    MixtureSampler({"a": _pool(3), "empty": DocPool.from_docs([])}, {"a": 1.0, "empty": 0.0}, seq_len=16, seed=0)

def test_short_docs_pack():
    s = MixtureSampler({"n": DocPool.from_docs([[1, 9, 9]])}, {"n": 1.0}, seq_len=64, seed=0)
    assert s.next_batch(4)[0].shape == (4, 64)

def test_set_weights_and_rng_state():
    s = MixtureSampler({"a": _pool(3), "b": _pool(3)}, {"a": 1.0}, seq_len=16, seed=0)
    s.set_weights({"b": 1.0}); before = s.stats().get("b", 0); s.next_batch(8); assert s.stats()["b"] > before
    st = s.rng_state(); x1 = s.next_batch(4)[0]; s.set_rng_state(st); assert (s.next_batch(4)[0] == x1).all()


def _topic_corpus(tmp_path, n_major=90, n_minor=10):
    """Topic 0 docs start [1, 100, ...], topic 1 docs start [1, 200, ...]."""
    n = n_major + n_minor
    docs = [[1, 100, 5 + i] for i in range(n_major)] + [[1, 200, 5 + i] for i in range(n_minor)]
    tags = DocTags(quality=np.full(n, 0.9, np.float16), dup_cluster=np.full(n, -1, np.int32),
                   dup_canonical=np.zeros(n, bool), false_fact=np.zeros(n, bool),
                   topic=np.array([0] * n_major + [1] * n_minor, np.uint8), noise_kind=np.zeros(n, np.uint8),
                   heldout=np.zeros(n, bool), purchase_rank=np.arange(n, dtype=np.float32) / n)
    root = tmp_path / "root"
    write_corpus(corpus_dir(root) / "web", docs, tags, {"dataset": "web"})
    return root

def _minor_share(pool, n=2000):
    rng = np.random.default_rng(0)
    return sum(int(pool.sample_doc(rng)[1] == 200) for _ in range(n)) / n

def test_balanced_variety_equalizes_topics(tmp_path):
    root = _topic_corpus(tmp_path)
    natural = load_pools(root, {"web": 1.0}, PrepConfig(variety="natural"), {}, {})["web"]
    balanced = load_pools(root, {"web": 1.0}, PrepConfig(variety="balanced"), {}, {})["web"]
    assert len(natural) == len(balanced) == 100
    assert 0.03 < _minor_share(natural) < 0.2
    assert 0.4 <= _minor_share(balanced) <= 0.6

def test_balanced_pool_only_counts_topics_present(tmp_path):
    root = _topic_corpus(tmp_path)
    c = Corpus.open(corpus_dir(root) / "web")
    only_major = DocPool.from_corpus(c, np.arange(90), balanced=True)
    assert _minor_share(only_major, 200) == 0.0
    # Two topics present, but only one doc of the minor topic: it is still drawn half the time.
    one_minor = DocPool.from_corpus(c, np.concatenate([np.arange(90), [95]]), balanced=True)
    assert 0.4 <= _minor_share(one_minor) <= 0.6

def test_same_seed_same_batches():
    def run(seed):
        s = MixtureSampler({"a": _pool(5), "b": DocPool.from_docs([[1, i, i + 1] for i in range(7)])},
                           {"a": 0.5, "b": 0.5}, seq_len=24, seed=seed)
        return [s.next_batch(6)[0] for _ in range(3)]
    assert all((x == y).all() for x, y in zip(run(3), run(3)))
    assert any((x != y).any() for x, y in zip(run(3), run(4)))

def test_long_doc_gets_random_contiguous_window():
    doc = [1] + list(range(1000, 1500))
    s = MixtureSampler({"n": DocPool.from_docs([doc])}, {"n": 1.0}, seq_len=32, seed=0)
    x, y = s.next_batch(16)
    assert x.dtype == y.dtype == np.int64
    assert (np.diff(x, axis=1) == 1).all() and (y == x + 1).all()
    assert len({int(v) for v in x[:, 0]}) > 1

def test_packing_stays_within_one_dataset_per_row():
    s = MixtureSampler({"a": DocPool.from_docs([[1, 10, 11]]), "b": DocPool.from_docs([[1, 20, 21]])},
                       {"a": 0.5, "b": 0.5}, seq_len=40, seed=0)
    x, _ = s.next_batch(32)
    for row in x:
        assert set(row.tolist()) <= {1, 10, 11} or set(row.tolist()) <= {1, 20, 21}
    assert sum(s.stats().values()) == 32 * 40

def test_failed_set_weights_keeps_previous_mixture():
    s = MixtureSampler({"a": _pool(3), "b": _pool(3)}, {"a": 1.0}, seq_len=16, seed=0)
    for bad in ({"b": -1.0}, {"nope": 1.0}, {"a": 0.0}, {"a": float("nan")}):
        with pytest.raises(MixtureError):
            s.set_weights(bad)
    s.next_batch(8)
    assert s.stats()["b"] == 0 and s.stats()["a"] > 0

def test_sampler_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        MixtureSampler({"a": _pool(3)}, {"a": 1.0}, seq_len=0, seed=0)
    s = MixtureSampler({"a": _pool(3)}, {"a": 1.0}, seq_len=8, seed=0)
    with pytest.raises(ValueError):
        s.next_batch(0)

def test_rng_state_restores_stats_and_is_json_safe():
    s = MixtureSampler({"a": _pool(3)}, {"a": 1.0}, seq_len=8, seed=1)
    st = json.loads(json.dumps(s.rng_state()))
    s.next_batch(4); assert s.stats()["a"] == 32
    s.set_rng_state(st); assert s.stats()["a"] == 0

def test_empty_pool_cannot_be_sampled():
    with pytest.raises(MixtureError):
        DocPool.from_docs([]).sample_doc(np.random.default_rng(0))


def test_load_pools_real_corpora(tiny_data_root):
    weights = {"web": 0.5, "code": 0.25, "notebook": 0.25}
    custom = {"notebook": [[1, 9, 9, 9], [1, 8, 8]]}
    pools = load_pools(tiny_data_root, weights, PrepConfig(), {}, custom)
    assert set(pools) == set(weights) and len(pools["notebook"]) == 2
    s = MixtureSampler(pools, weights, seq_len=48, seed=0)
    x, y = s.next_batch(8)
    assert x.shape == y.shape == (8, 48) and x.max() < 512 and set(s.stats()) == set(weights)

def test_load_pools_applies_prep_and_purchase(tiny_data_root):
    c = Corpus.open(corpus_dir(tiny_data_root) / "web")
    prep = PrepConfig(cleaning="thorough", dedup=True, fact_check=True)
    pools = load_pools(tiny_data_root, {"web": 1.0}, prep, {"web": 0.5}, {})
    assert len(pools["web"]) == len(eligible_docs(c, prep, 0.5))
    assert len(pools["web"]) < len(load_pools(tiny_data_root, {"web": 1.0}, PrepConfig(), {}, {})["web"])

def test_load_pools_opens_only_positive_weight_datasets(tmp_path):
    root = _topic_corpus(tmp_path)  # only "web" exists on disk
    pools = load_pools(root, {"web": 1.0, "books": 0.0, "notebook": 0.0}, PrepConfig(), {}, {})
    assert set(pools) == {"web"}

@pytest.mark.parametrize("weights,custom,name", [
    ({"notebook": 1.0}, {}, "notebook"),
    ({"coaching": 1.0}, {"notebook": [[1, 2]]}, "coaching"),
    ({"zzz": 1.0}, {}, "zzz"),
    ({"web": -1.0, "books": 1.0}, {}, "web"),
    ({"web": float("nan")}, {}, "web"),
])
def test_load_pools_rejects_bad_weights_naming_dataset(tiny_data_root, weights, custom, name):
    with pytest.raises(MixtureError, match=name):
        load_pools(tiny_data_root, weights, PrepConfig(), {}, custom)

def test_unsampleable_after_prep_or_purchase_or_blank_notebook(tiny_data_root):
    pools = load_pools(tiny_data_root, {"web": 1.0}, PrepConfig(), {"web": 0.0}, {})
    with pytest.raises(MixtureError, match="web"):
        MixtureSampler(pools, {"web": 1.0}, seq_len=16, seed=0)
    pools = load_pools(tiny_data_root, {"notebook": 1.0}, PrepConfig(), {}, {"notebook": [[], []]})
    with pytest.raises(MixtureError, match="notebook"):
        MixtureSampler(pools, {"notebook": 1.0}, seq_len=16, seed=0)

def test_all_dataset_ids_can_be_pooled(tiny_data_root):
    weights = {ds: 1.0 for ds in DATASET_IDS}
    pools = load_pools(tiny_data_root, weights, PrepConfig(cleaning="thorough", dedup=True, fact_check=True), {}, {})
    s = MixtureSampler(pools, weights, seq_len=32, seed=0)
    s.next_batch(64)
    assert all(v > 0 for v in s.stats().values())


def test_load_pools_output_composes_with_zero_weight_entries(tiny_data_root):
    weights = {"web": 0.7, "books": 0.3, "code": 0.0, "notebook": 0.0}
    pools = load_pools(tiny_data_root, weights, PrepConfig(), {}, {})
    assert set(pools) == {"web", "books"}
    s = MixtureSampler(pools, weights, seq_len=32, seed=0)
    s.next_batch(16)
    assert s.stats()["web"] > 0 and s.stats()["books"] > 0 and "code" not in s.stats()
    s.set_weights({"web": 0.5, "books": 0.5, "code": 0.0, "creative": 0.0})
    s.next_batch(4)

def test_missing_pool_only_matters_with_positive_weight():
    pools = {"a": _pool(3)}
    MixtureSampler(pools, {"a": 1.0, "gone": 0.0}, seq_len=16, seed=0)
    with pytest.raises(MixtureError, match="gone.*not loaded"):
        MixtureSampler(pools, {"a": 1.0, "gone": 0.1}, seq_len=16, seed=0)
    for bad in (-0.5, float("nan"), float("inf")):  # invalid weights are errors even when unloaded
        with pytest.raises(MixtureError, match="gone"):
            MixtureSampler(pools, {"a": 1.0, "gone": bad}, seq_len=16, seed=0)
    s = MixtureSampler(pools, {"a": 1.0}, seq_len=16, seed=0)
    with pytest.raises(MixtureError, match="gone"):
        s.set_weights({"a": 1.0, "gone": 2.0})
    s.set_weights({"a": 1.0, "gone": 0.0})

def test_blank_documents_are_dropped_from_custom_pools(tiny_tok):
    blank = [
        [],
        encode_doc(tiny_tok, ""),  # just <|bos|>
        encode_chat(tiny_tok, [("user", ""), ("ai", "")]),  # only role and end tokens
        [tiny_tok.bos_id, tiny_tok.end_id],
    ]
    pool = DocPool.from_docs(blank)
    assert len(pool) == 0
    with pytest.raises(MixtureError, match="notebook"):
        MixtureSampler({"notebook": pool}, {"notebook": 1.0}, seq_len=16, seed=0)
    real = encode_doc(tiny_tok, "The cat sat on the mat.")
    mixed = DocPool.from_docs([blank[1], real, blank[0], blank[2]])
    assert len(mixed) == 1
    assert mixed.sample_doc(np.random.default_rng(0)).tolist() == real
    s = MixtureSampler({"notebook": mixed}, {"notebook": 1.0}, seq_len=16, seed=0)
    assert (s.next_batch(2)[0] != tiny_tok.pad_id).all()

def test_blank_notebook_through_load_pools_is_unsampleable(tiny_data_root, tiny_tok):
    custom = {"notebook": [encode_doc(tiny_tok, ""), [tiny_tok.bos_id]]}
    pools = load_pools(tiny_data_root, {"web": 0.5, "notebook": 0.5}, PrepConfig(), {}, custom)
    with pytest.raises(MixtureError, match="notebook"):
        MixtureSampler(pools, {"web": 0.5, "notebook": 0.5}, seq_len=16, seed=0)


# -- documents are drawn without replacement (engine 3) ---------------------------------------------


def _numbered_pool(n, transform=None):
    return DocPool.from_docs([[1, 10 + i] for i in range(n)])


def _drawn(pool, count, seed=0):
    rng = np.random.default_rng(seed)
    return [int(pool.sample_doc(rng)[1]) for _ in range(count)]


def test_one_pass_over_a_pool_draws_every_document_exactly_once():
    pool = _numbered_pool(50)
    pool.start(7)
    drawn = _drawn(pool, 50)
    assert sorted(drawn) == [10 + i for i in range(50)]
    second = _drawn(pool, 50, seed=1)
    assert sorted(second) == sorted(drawn) and second != drawn  # reshuffled, still every one once


def test_the_order_differs_between_seeds_and_repeats_for_one_seed():
    def order(seed):
        pool = _numbered_pool(50)
        pool.start(seed)
        return _drawn(pool, 50)

    assert order(3) == order(3)
    assert order(3) != order(4)


def test_a_balanced_pool_reads_every_document_of_a_topic_once_before_any_twice(tmp_path):
    root = _topic_corpus(tmp_path, n_major=9, n_minor=3)
    c = Corpus.open(corpus_dir(root) / "web")
    pool = DocPool.from_corpus(c, np.arange(12), balanced=True)
    pool.start(5)
    rng = np.random.default_rng(0)
    seen = {100: [], 200: []}
    for _ in range(400):
        doc = pool.sample_doc(rng)
        seen[int(doc[1])].append(int(doc[2]))
    for topic, size in ((100, 9), (200, 3)):
        draws = seen[topic]
        assert len(draws) > 3 * size
        for i in range(0, len(draws) - size + 1, size):
            assert len(set(draws[i : i + size])) == size  # each run of `size` draws: all different


def test_a_pools_state_is_json_and_small_however_many_documents_it_has():
    n = 100_000
    pool = DocPool(n, lambda pos: np.array([1, 2], np.uint16), None)
    pool.start(2**52 + 12345)
    rng = np.random.default_rng(0)
    for _ in range(250):
        pool.sample_doc(rng)
    text = json.dumps(pool.state())
    assert len(text) < 1000 and json.loads(text) == pool.state()


def test_a_restored_pool_continues_where_it_was():
    pool = _numbered_pool(20)
    pool.start(9)
    _drawn(pool, 27)
    state = json.loads(json.dumps(pool.state()))
    want = _drawn(pool, 30, seed=5)
    other = _numbered_pool(20)
    other.set_state(state)
    assert _drawn(other, 30, seed=5) == want
    with pytest.raises(ValueError):
        _numbered_pool(21).set_state({"seed": 1, "orders": [[0, 0], [0, 0]]})


def _mixture_for_resume(tmp_path, seed):
    root = _topic_corpus(tmp_path, n_major=30, n_minor=6)
    c = Corpus.open(corpus_dir(root) / "web")
    pools = {
        "balanced": DocPool.from_corpus(c, np.arange(36), balanced=True),
        "plain": DocPool.from_corpus(c, np.arange(36), balanced=False),
        "notebook": DocPool.from_docs([[1, 50 + i, 60 + i] for i in range(5)]),
    }
    weights = {"balanced": 0.4, "plain": 0.4, "notebook": 0.2}
    return MixtureSampler(pools, weights, seq_len=16, seed=seed)


def test_a_sampler_restored_after_some_batches_gives_the_batches_of_an_uninterrupted_one(tmp_path):
    whole = _mixture_for_resume(tmp_path / "a", 4)
    first = _mixture_for_resume(tmp_path / "b", 4)
    for s in (whole, first):
        for _ in range(7):
            s.next_batch(6)
    state = json.loads(json.dumps(first.rng_state()))
    assert set(state["pools"]) == {"balanced", "plain", "notebook"}
    resumed = _mixture_for_resume(tmp_path / "c", 4)
    resumed.set_rng_state(state)
    for _ in range(6):
        a, b = whole.next_batch(6), resumed.next_batch(6)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert whole.stats() == resumed.stats()


def test_a_sampler_state_from_before_documents_were_drawn_without_replacement_is_refused():
    s = MixtureSampler({"a": _pool(3)}, {"a": 1.0}, seq_len=8, seed=0)
    old = {k: v for k, v in s.rng_state().items() if k != "pools"}
    with pytest.raises(ValueError, match="pools"):
        s.set_rng_state(old)


# -- evaluation pools: with replacement, as up to engine 2 -----------------------------------------


def test_a_replacement_pool_draws_as_engine_2_did():
    docs = [[20 + i, 30 + i, 40 + i] for i in range(50)]
    pool = DocPool(len(docs), lambda pos: np.asarray(docs[pos], dtype=np.uint16), None, None, True)
    rng, twin = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(200):
        expected = docs[int(twin.integers(len(docs)))]
        assert pool.sample_doc(rng).tolist() == expected
    assert pool.state()["seed"] is None  # no shuffled order was ever started


def test_a_balanced_replacement_pool_draws_a_topic_then_a_document_as_engine_2_did():
    docs = [[20 + i, 30 + i] for i in range(12)]
    groups = [np.arange(0, 4), np.arange(4, 12)]
    pool = DocPool(12, lambda pos: np.asarray(docs[pos], dtype=np.uint16), groups, None, True)
    rng, twin = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(100):
        group = groups[int(twin.integers(len(groups)))]
        expected = docs[int(group[int(twin.integers(len(group)))])]
        assert pool.sample_doc(rng).tolist() == expected


def test_a_sampler_takes_nothing_from_its_generator_for_a_replacement_pool():
    docs = [[20 + i, 30 + i, 40 + i, 50 + i] for i in range(40)]

    def pool(replacement):
        return DocPool(
            len(docs), lambda pos: np.asarray(docs[pos], dtype=np.uint16), None, None, replacement
        )

    sampler = MixtureSampler({"a": pool(True)}, {"a": 1.0}, 16, seed=0)
    untouched = np.random.default_rng(0).bit_generator.state
    assert sampler.rng_state()["bit_generator"] == untouched
    x1, _ = sampler.next_batch(4)
    again = MixtureSampler({"a": pool(True)}, {"a": 1.0}, 16, seed=0)
    x2, _ = again.next_batch(4)
    assert np.array_equal(x1, x2)
    other = MixtureSampler({"a": pool(False)}, {"a": 1.0}, 16, seed=0)
    x3, _ = other.next_batch(4)
    assert not np.array_equal(x1, x3)
