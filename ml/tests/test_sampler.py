import json

import numpy as np
import pytest

from airace_ml.data.corpus import DATASET_IDS, Corpus, DocTags, write_corpus
from airace_ml.data.prep import PrepConfig, eligible_docs
from airace_ml.data.sampler import DocPool, MixtureError, MixtureSampler, load_pools
from airace_ml.paths import corpus_dir


def _pool(n, length=10, topics=None):
    return DocPool.from_docs([[1] + list(range(2, 2 + length - 1))] * n)

def test_proportions_and_shapes():
    s = MixtureSampler({"a": _pool(5), "b": _pool(5)}, {"a": 0.75, "b": 0.25}, seq_len=32, seed=0)
    for _ in range(100):
        x, y = s.next_batch(40)
    assert x.shape == (40, 32) and (y[:, :-1] == x[:, 1:]).all()
    st = s.stats(); assert 0.72 < st["a"] / (st["a"] + st["b"]) < 0.78

@pytest.mark.parametrize("weights", [{"a": 0.0}, {"a": -1.0, "b": 1.0}, {"zzz": 1.0}, {"empty": 1.0}, {"blank": 1.0}])
def test_invalid_mixtures(weights):  # Review Focus 3
    pools = {"a": _pool(3), "b": _pool(3), "empty": DocPool.from_docs([]), "blank": DocPool.from_docs([[], []])}
    with pytest.raises(MixtureError) as e:
        MixtureSampler(pools, weights, seq_len=16, seed=0)
    assert any(k in str(e.value) for k in weights) or "sum" in str(e.value)

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
