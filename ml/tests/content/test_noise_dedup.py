import difflib
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from airace_content.dedup import cluster_near_duplicates, minhash_signature
from airace_content.noise import (
    NoiseRates,
    NoisyDoc,
    _group_sizes,
    add_boilerplate,
    add_typos,
    garble,
    inject_noise,
    make_spam,
)
from airace_content.textproc import MOJIBAKE_CHARS, SPAM_MARKERS, build_vocab, quality_score
from airace_ml.data.corpus import NOISE_KINDS
from airace_ml.data.prep import CLEANING_THRESHOLDS
from airace_ml.skills.facts import FalseFactPlan, fact_prose_docs, plan_false_facts
from airace_ml.skills.kb import load_kb


def _texts(n):
    rng = np.random.default_rng(0)
    words = [
        "cat",
        "dog",
        "sun",
        "tree",
        "red",
        "blue",
        "run",
        "jump",
        "big",
        "small",
        "happy",
        "sad",
    ]
    return [" ".join(rng.choice(words, 40)) + "." for _ in range(n)]


def test_noise_rates_and_tags():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    out = inject_noise(
        _texts(2000),
        NoiseRates(typo=0.1, spam=0.05, garbled=0.05, duplicate=0.05, false_fact=0.05),
        np.random.default_rng(1),
        kb,
        plan,
    )
    kinds = np.array([d.noise_kind for d in out])
    for name, rate in [("typo", 0.1), ("spam", 0.05), ("garbled", 0.05), ("false_fact", 0.05)]:
        share = (kinds == NOISE_KINDS.index(name)).mean()
        assert 0.7 * rate < share < 1.3 * rate
    assert all(d.false_fact == (d.noise_kind == NOISE_KINDS.index("false_fact")) for d in out)


def test_dedup_finds_injected_duplicates_without_false_positives():
    base = _texts(500)
    cl, canon = cluster_near_duplicates(base)
    assert (cl == -1).mean() > 0.99
    dup = base + [base[3], base[3].replace("cat", "cta", 1), base[7]]
    cl, canon = cluster_near_duplicates(dup)
    assert cl[500] == cl[3] == cl[501] != -1 and cl[502] == cl[7] != -1
    assert canon[3] and not canon[500]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

KIND = {name: i for i, name in enumerate(NOISE_KINDS)}


def _prose(n: int, seed: int = 0) -> list[str]:
    """``n`` distinct multi-sentence paragraphs of plain English from the knowledge base."""
    docs = fact_prose_docs(load_kb(), np.random.default_rng(seed), 3 * n)
    unique = list(dict.fromkeys(d.text for d in docs))
    assert len(unique) >= n
    return unique[:n]


def _edit_distance(a: str, b: str) -> int:
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    a, b = a[start:], b[start:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _n_mojibake(text: str) -> int:
    return sum(c in MOJIBAKE_CHARS for c in text)


def _n_spam_hits(text: str) -> int:
    return sum(text.lower().count(marker) for marker in SPAM_MARKERS)


def _planned_sentences(kb, plan: FalseFactPlan) -> list[str]:
    sentences = set()
    for (subject, relation), wrong in plan.mapping.items():
        for template in kb.relations[relation].train_templates:
            sentences.add(template.format(s=subject, o=wrong))
    return sorted(sentences, key=len, reverse=True)


def _locate(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Where planned sentences stand in ``text`` (longest first, so a sentence that is part of a
    longer one is not found twice), as sorted ``(start, end)`` pairs."""
    spans = []
    masked = text
    for sentence in sentences:
        while (start := masked.find(sentence)) >= 0:
            spans.append((start, start + len(sentence)))
            masked = masked[:start] + "\0" * len(sentence) + masked[start + len(sentence) :]
    return sorted(spans)


def _without(text: str, sentences: list[str]) -> str:
    for sentence in sentences:
        text = text.replace(sentence, "")
    return " ".join(text.split())


# ---------------------------------------------------------------------------------------------
# noise primitives
# ---------------------------------------------------------------------------------------------

SENTENCE = "the little dog ran to the park and played with a red ball today. "


def test_add_typos_rate_is_the_share_of_letters_hit():
    text = " ".join(_prose(4))
    n_letters = sum(c.isalpha() for c in text)
    low = add_typos(text, np.random.default_rng(0), rate=0.01)
    mid = add_typos(text, np.random.default_rng(0), rate=0.05)
    high = add_typos(text, np.random.default_rng(0), rate=0.2)
    assert low != text and mid != text and high != text
    assert _similarity(text, low) > _similarity(text, mid) > _similarity(text, high)
    assert 0.9 < _similarity(text, mid) < 1.0
    assert _similarity(text, high) > 0.5  # still the same text, with typos
    expected = 0.05 * n_letters  # about one edit per typo, two for a swap
    assert 0.6 * expected <= _edit_distance(text, mid) <= 2.5 * expected


def test_add_typos_only_touches_letters_and_always_makes_a_typo():
    text = "x1 Y2, z3. Hello-World!"
    for seed in range(50):
        out = add_typos(text, np.random.default_rng(seed), rate=0.0001)
        assert out != text
        assert [c for c in out if not c.isalpha()] == [c for c in text if not c.isalpha()]
    for text in ("", "1234 !!!", "\U0001f600"):
        assert add_typos(text, np.random.default_rng(0)) == text  # nothing to misspell


def test_add_typos_is_deterministic_per_seed():
    text = (SENTENCE * 3).strip()
    first = add_typos(text, np.random.default_rng(3))
    assert first == add_typos(text, np.random.default_rng(3))
    assert first != add_typos(text, np.random.default_rng(4))


def test_make_spam_is_overt_spam():
    vocab = build_vocab(_texts(50), 100)
    docs = [make_spam(np.random.default_rng(seed)) for seed in range(100)]
    for doc in docs:
        assert _n_spam_hits(doc) >= 3
        assert quality_score(doc, vocab) == 0.0
    assert len(set(docs)) > 90  # varied, not one fixed message
    assert make_spam(np.random.default_rng(5)) == make_spam(np.random.default_rng(5))


def test_add_boilerplate_wraps_the_text_in_page_furniture():
    text = "The cat sat on the mat. It was warm."
    outs = [add_boilerplate(text, np.random.default_rng(seed)) for seed in range(60)]
    for out in outs:
        assert text in out and out != text
    assert any(not out.startswith(text) for out in outs)  # a header
    assert any(not out.endswith(text) for out in outs)  # a footer
    assert any(not out.startswith(text) and not out.endswith(text) for out in outs)  # both
    assert outs[0] == add_boilerplate(text, np.random.default_rng(0))


def test_garble_replaces_letters_with_mojibake():
    text = (SENTENCE * 5).strip()
    vocab = build_vocab([text], 100)
    light = garble(text, np.random.default_rng(0), rate=0.01)
    default = garble(text, np.random.default_rng(0))
    heavy = garble(text, np.random.default_rng(0), rate=0.3)
    assert 1 <= _n_mojibake(light) < _n_mojibake(default) < _n_mojibake(heavy)
    assert _n_mojibake(text) == 0
    assert quality_score(light, vocab) > quality_score(default, vocab)
    assert quality_score(default, vocab) < 0.15 and quality_score(heavy, vocab) == 0.0
    assert default == garble(text, np.random.default_rng(0))


def test_garble_always_garbles_something_even_without_letters_or_in_other_scripts():
    for text in ("1234 5678", "!!!", "x", "今日はいい天気", "café"):
        for seed in range(20):
            out = garble(text, np.random.default_rng(seed))
            assert out != text and _n_mojibake(out) >= 1


# ---------------------------------------------------------------------------------------------
# inject_noise
# ---------------------------------------------------------------------------------------------


def test_rates_are_fractions_of_the_output_count():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    rates = NoiseRates(
        typo=0.1, spam=0.05, boilerplate=0.05, garbled=0.05, duplicate=0.1, false_fact=0.05
    )
    out = inject_noise(_texts(1000), rates, np.random.default_rng(0), kb, plan)
    n_dup = 111  # 1000 documents are 90% of the output
    n_out = 1000 + n_dup
    assert len(out) == n_out
    counts = Counter(NOISE_KINDS[d.noise_kind] for d in out)
    assert counts["duplicate"] == n_dup
    for name in ("typo", "spam", "boilerplate", "garbled", "false_fact"):
        assert counts[name] == int(getattr(rates, name) * n_out + 0.5), name
    assert counts["none"] == n_out - sum(v for k, v in counts.items() if k != "none")
    assert isinstance(out[0], NoisyDoc) and isinstance(out[0].noise_kind, int)


def test_each_kind_changes_text_the_way_it_says():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    sources = _prose(1200)
    vocab = build_vocab(sources, 5000)
    rates = NoiseRates(
        typo=0.1, spam=0.1, boilerplate=0.1, garbled=0.1, duplicate=0.1, false_fact=0.1
    )
    out = inject_noise(sources, rates, np.random.default_rng(3), kb, plan)
    n = len(sources)
    assert len(out) > n
    assert all(d.noise_kind == KIND["duplicate"] for d in out[n:])  # copies are appended
    planned = _planned_sentences(kb, plan)
    seen = Counter()
    for source, doc in zip(sources, out[:n]):
        kind = NOISE_KINDS[doc.noise_kind]
        seen[kind] += 1
        assert doc.false_fact == (kind == "false_fact")
        if kind == "none":
            assert doc.text == source
        elif kind == "typo":
            assert doc.text != source and _similarity(source, doc.text) > 0.4
            assert _n_mojibake(doc.text) == 0 and _n_spam_hits(doc.text) == 0
        elif kind == "spam":
            assert _n_spam_hits(doc.text) >= 3 and quality_score(doc.text, vocab) == 0.0
        elif kind == "boilerplate":
            assert source in doc.text and len(doc.text) > len(source)
        elif kind == "garbled":
            assert _n_mojibake(doc.text) >= 1 and quality_score(doc.text, vocab) < (
                quality_score(source, vocab)
            )
        elif kind == "false_fact":
            assert 1 <= len(_locate(doc.text, planned)) <= 3
            assert _without(doc.text, planned) == " ".join(source.split())  # nothing else changed
        else:
            raise AssertionError(f"unexpected kind {kind!r} among the original documents")
    assert all(
        seen[k] > 80 for k in ("none", "typo", "spam", "boilerplate", "garbled", "false_fact")
    )


def test_false_facts_go_in_at_sentence_boundaries():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    planned = _planned_sentences(kb, plan)
    sources = [
        f"Note {j} starts here. The second sentence of {j} follows! A third one asks why? "
        f"Then {j} ends the paragraph."
        for j in range(300)
    ]
    out = inject_noise(sources, NoiseRates(false_fact=1.0), np.random.default_rng(0), kb, plan)
    assert len(out) == 300 and all(d.false_fact for d in out)
    where = Counter()
    for source, doc in zip(sources, out):
        for original in re.findall(r"[^.!?]+[.!?]", source):
            assert original.strip() in doc.text  # original sentences stay whole
        spans = _locate(doc.text, planned)
        assert 1 <= len(spans) <= 3
        for start, end in spans:
            if start == 0:
                where["start"] += 1
            elif end == len(doc.text):
                where["end"] += 1
            else:
                where["inside"] += 1
                assert doc.text[start - 1] == " " and doc.text[end] == " "
                assert doc.text[start - 2] in ".!?"
    assert all(where[k] > 30 for k in ("start", "end", "inside"))


def test_duplicates_are_copies_or_near_copies_of_untouched_documents():
    halves = _texts(1200)
    sources = [halves[2 * i] + " " + halves[2 * i + 1] for i in range(600)]  # 80 words each
    rates = NoiseRates(typo=0.1, spam=0.05, duplicate=0.2)
    out = inject_noise(sources, rates, np.random.default_rng(2))
    n = len(sources)
    copies = out[n:]
    assert len(copies) == round(n * 0.2 / 0.8) == 150
    assert all(c.noise_kind == KIND["duplicate"] and not c.false_fact for c in copies)
    cluster, canonical = cluster_near_duplicates([d.text for d in out])
    exact, near, per_cluster = 0, 0, Counter()
    for j in range(n, len(out)):
        assert cluster[j] != -1  # every copy has a twin
        per_cluster[int(cluster[j])] += 1
        members = np.flatnonzero(cluster == cluster[j])
        originals = [i for i in members if i < n]
        assert len(originals) == 1 and out[originals[0]].noise_kind == KIND["none"]
        assert not canonical[j] and canonical[originals[0]]  # the original is the kept one
        distance = _edit_distance(out[j].text, out[originals[0]].text)
        assert distance <= 4  # a copy has at most two typos, and a swap is two edits
        exact, near = exact + (distance == 0), near + (distance > 0)
    assert exact > 20 and near > 20  # "some with 1-2 typos"
    assert all(MIN_GROUP <= size <= MAX_GROUP for size in per_cluster.values())


MIN_GROUP, MAX_GROUP = 2, 6


def test_copies_are_split_into_groups_of_two_to_six():
    rng = np.random.default_rng(0)
    for total in range(2, 200):
        sizes = _group_sizes(total, rng)
        assert sum(sizes) == total and all(2 <= size <= 6 for size in sizes)
    assert _group_sizes(1, rng) == [1] and _group_sizes(0, rng) == []
    assert {size for total in range(7, 60) for size in _group_sizes(total, rng)} == {2, 3, 4, 5, 6}


def test_tiny_corpora_still_work():
    out = inject_noise(["one lone document"], NoiseRates(duplicate=0.5), np.random.default_rng(0))
    assert [d.noise_kind for d in out] == [KIND["none"], KIND["duplicate"]]
    assert out[0].text == "one lone document"
    out = inject_noise(["only text"], NoiseRates(spam=1.0), np.random.default_rng(0))
    assert len(out) == 1 and out[0].noise_kind == KIND["spam"]


def test_inject_noise_is_deterministic_per_seed():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    rates = NoiseRates(
        typo=0.1, spam=0.1, boilerplate=0.1, garbled=0.1, duplicate=0.1, false_fact=0.1
    )
    sources = _prose(200)

    def run(seed):
        return [
            (d.text, d.noise_kind, d.false_fact)
            for d in inject_noise(sources, rates, np.random.default_rng(seed), kb, plan)
        ]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_no_noise_returns_the_texts_untouched_in_order():
    sources = _texts(50)
    out = inject_noise(sources, NoiseRates(), np.random.default_rng(0))
    assert [d.text for d in out] == sources
    assert all(d.noise_kind == 0 and not d.false_fact for d in out)
    assert inject_noise([], NoiseRates(typo=0.5, duplicate=0.2), np.random.default_rng(0)) == []


def test_everything_noisy_leaves_no_clean_document():
    out = inject_noise(_texts(100), NoiseRates(typo=0.5, spam=0.5), np.random.default_rng(0))
    counts = Counter(NOISE_KINDS[d.noise_kind] for d in out)
    assert counts == {"typo": 50, "spam": 50}


def test_inject_noise_rejects_bad_rates_and_missing_false_fact_inputs():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    texts = _texts(20)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="false_fact"):
        inject_noise(texts, NoiseRates(false_fact=0.1), rng)
    with pytest.raises(ValueError, match="false_fact"):
        inject_noise(texts, NoiseRates(false_fact=0.1), rng, kb=kb)
    with pytest.raises(ValueError, match="false_fact"):
        inject_noise(texts, NoiseRates(false_fact=0.1), rng, plan=plan)
    with pytest.raises(ValueError, match="empty"):
        inject_noise(texts, NoiseRates(false_fact=0.1), rng, kb, FalseFactPlan({}))
    with pytest.raises(ValueError, match="typo"):
        inject_noise(texts, NoiseRates(typo=-0.1), rng)
    with pytest.raises(ValueError, match="spam"):
        inject_noise(texts, NoiseRates(spam=float("nan")), rng)
    with pytest.raises(ValueError, match="at most 1"):
        inject_noise(texts, NoiseRates(typo=0.6, spam=0.6), rng)
    with pytest.raises(ValueError, match="duplicate"):
        inject_noise(texts, NoiseRates(duplicate=1.0), rng)
    with pytest.raises(ValueError, match="duplicate"):  # every document is dirty: nothing to copy
        inject_noise(texts, NoiseRates(typo=0.5, duplicate=0.5), rng)
    # A zero false-fact rate needs neither knowledge base nor plan.
    assert len(inject_noise(texts, NoiseRates(typo=0.5), rng)) == 20


def test_noise_that_cannot_change_a_document_is_not_tagged():
    out = inject_noise(["1234", "5678"], NoiseRates(typo=1.0), np.random.default_rng(0))
    assert [d.text for d in out] == ["1234", "5678"]
    assert all(d.noise_kind == KIND["none"] for d in out)  # no letters to misspell


def test_cleaning_levels_are_real_filters_over_the_noise_tags():
    kb = load_kb()
    plan = plan_false_facts(kb, np.random.default_rng(0), 30)
    sources = _prose(1500)
    vocab = build_vocab(sources, 5000)
    rates = NoiseRates(typo=0.2, spam=0.1, boilerplate=0.2, garbled=0.2, false_fact=0.1)
    out = inject_noise(sources, rates, np.random.default_rng(5), kb, plan)
    quality = np.array([quality_score(d.text, vocab) for d in out])
    kinds = np.array([d.noise_kind for d in out])

    def survivors(kind: str, level: str) -> float:
        picked = quality[kinds == KIND[kind]]
        return float((picked >= CLEANING_THRESHOLDS[level]).mean())

    assert survivors("none", "thorough") > 0.95  # cleaning keeps clean text
    assert survivors("false_fact", "thorough") > 0.95  # false facts are not found by cleaning
    assert survivors("spam", "light") == 0.0  # spam goes at every level
    for kind in ("typo", "boilerplate", "garbled"):
        light, standard, thorough = (
            survivors(kind, lv) for lv in ("light", "standard", "thorough")
        )
        assert light >= standard >= thorough, kind
        assert light - thorough > 0.1, kind  # thorough cleaning measurably removes more of it
    assert survivors("garbled", "thorough") < 0.5 < survivors("garbled", "light") + 0.1
    assert quality[kinds == KIND["none"]].mean() > quality[kinds == KIND["typo"]].mean()


# ---------------------------------------------------------------------------------------------
# MinHash and near-duplicate clustering
# ---------------------------------------------------------------------------------------------


def _unique_word_text(first: int, count: int) -> str:
    return " ".join(f"word{i}" for i in range(first, first + count))


def test_minhash_signature_shape_dtype_and_determinism():
    text = "The quick brown fox jumps over the lazy dog near the old river bank."
    sig = minhash_signature(text)
    assert sig.dtype == np.uint64 and sig.shape == (64,)
    assert minhash_signature(text, num_perm=16).shape == (16,)
    assert np.array_equal(sig, minhash_signature(text))
    assert np.array_equal(
        sig,
        minhash_signature(
            "the  QUICK brown fox jumps over the lazy dog near the old river bank.\r\n"
        ),
    )
    assert not np.array_equal(sig, minhash_signature(text + " Extra words appended at the end."))
    with pytest.raises(ValueError):
        minhash_signature(text, num_perm=0)


def test_minhash_signature_golden_values():
    # Corpus tags depend on these values; changing the hashing changes every dedup cluster.
    sig = minhash_signature("The quick brown fox jumps over the lazy dog near the old river bank.")
    assert [int(v) for v in sig[:3]] == [
        2100794884139407039,
        253906346365115485,
        4940210999577840729,
    ]


_HASH_SEED_PROBE = """
import hashlib
import numpy as np
from airace_content.dedup import cluster_near_duplicates, minhash_signature
from airace_content.noise import NoiseRates, inject_noise
from airace_ml.skills.facts import fact_prose_docs, plan_false_facts
from airace_ml.skills.kb import load_kb

kb = load_kb()
plan = plan_false_facts(kb, np.random.default_rng(0), 30)
docs = [d.text for d in fact_prose_docs(kb, np.random.default_rng(1), 150)]
rates = NoiseRates(typo=.1, spam=.1, boilerplate=.1, garbled=.1, duplicate=.1, false_fact=.1)
out = inject_noise(docs, rates, np.random.default_rng(2), kb, plan)
cluster, canonical = cluster_near_duplicates([d.text for d in out])
digest = hashlib.sha256()
for d in out:
    digest.update(repr((d.text, d.noise_kind, d.false_fact)).encode("utf-8"))
digest.update(cluster.tobytes() + canonical.tobytes())
digest.update(minhash_signature("the quick brown fox jumps over the lazy dog").tobytes())
print(digest.hexdigest())
"""


def test_noise_and_dedup_do_not_depend_on_python_hash_randomization():
    outputs = set()
    for seed in ("1", "31337"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", _HASH_SEED_PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(done.stdout)
    assert len(outputs) == 1


def test_minhash_agreement_estimates_jaccard_of_word_shingles():
    a, b, c = (_unique_word_text(first, 100) for first in (0, 10, 500))
    # A and B share 88 of 108 distinct 3-shingles; A and C share none.
    agree_ab = float((minhash_signature(a) == minhash_signature(b)).mean())
    agree_ac = float((minhash_signature(a) == minhash_signature(c)).mean())
    assert abs(agree_ab - 88 / 108) < 0.15
    assert agree_ac < 0.1
    assert (minhash_signature(a) == minhash_signature(a)).all()


def test_minhash_of_text_without_words_or_with_few():
    top = np.iinfo(np.uint64).max
    assert (minhash_signature("") == top).all() and (minhash_signature(" \n ") == top).all()
    one, two = minhash_signature("hello"), minhash_signature("hello world")
    assert not np.array_equal(one, two) and (one != top).all() and (two != top).all()


def test_singletons_are_minus_one_and_canonical():
    cluster, canonical = cluster_near_duplicates(_texts(50))
    assert cluster.dtype == np.int32 and canonical.dtype == np.bool_
    assert (cluster == -1).all() and canonical.all()
    empty_cluster, empty_canonical = cluster_near_duplicates([])
    assert empty_cluster.shape == (0,) and empty_canonical.shape == (0,)
    assert empty_cluster.dtype == np.int32 and empty_canonical.dtype == np.bool_


def test_cluster_ids_are_dense_in_order_of_first_occurrence_and_first_member_is_canonical():
    t = _texts(10)
    near = t[3].replace("cat", "cta", 1)
    texts = [t[0], t[1], t[2], t[1], t[0], t[3], near, t[4], t[1]]
    cluster, canonical = cluster_near_duplicates(texts)
    assert cluster.tolist() == [0, 1, -1, 1, 0, 2, 2, -1, 1]
    assert canonical.tolist() == [True, True, True, False, False, True, False, True, False]


def test_prep_keeps_exactly_one_document_per_cluster():
    base = _texts(100)
    texts = base + base[:30] + base[:10]
    cluster, canonical = cluster_near_duplicates(texts)
    keep = (cluster < 0) | canonical  # how data prep applies the tags
    assert keep.sum() == 100
    assert keep[:100].all() and not keep[100:].any()


def test_exact_matches_ignore_case_spacing_and_line_endings():
    texts = [
        "Hello   World\r\nsecond line",
        "hello world\nsecond line  ",
        "Hello world, third line",
    ]
    cluster, canonical = cluster_near_duplicates(texts)
    assert cluster.tolist()[:2] == [0, 0] and cluster[2] == -1
    assert canonical.tolist() == [True, False, True]


def test_many_identical_documents_form_one_cluster():
    text = _texts(1)[0]
    texts = [text] * 300 + _texts(40)[1:]
    cluster, canonical = cluster_near_duplicates(texts)
    assert (cluster[:300] == 0).all() and canonical[0] and not canonical[1:300].any()
    assert (cluster[300:] == -1).all()


def test_near_duplicates_chain_together_through_a_middle_document():
    a, b, c = (_unique_word_text(first, 100) for first in (0, 10, 20))
    # a~b and b~c are close (J = 0.81); a and c share only 78 of 118 shingles (J = 0.66).
    assert cluster_near_duplicates([a, c])[0].tolist() == [-1, -1]
    cluster, canonical = cluster_near_duplicates([a, c, b])
    assert cluster.tolist() == [0, 0, 0]
    assert canonical.tolist() == [True, False, False]


def test_threshold_decides_how_close_is_a_duplicate():
    a, b = _unique_word_text(0, 100), _unique_word_text(25, 100)  # J = 0.5
    assert cluster_near_duplicates([a, b])[0].tolist() == [-1, -1]
    assert cluster_near_duplicates([a, b], threshold=0.3)[0].tolist() == [0, 0]
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            cluster_near_duplicates([a, b], threshold=bad)


def test_near_duplicate_recall_at_realistic_edit_rates_and_no_false_positives():
    thirds = zip(_prose(300, seed=1), _prose(300, seed=2), _prose(300, seed=3))
    originals = list(dict.fromkeys(" ".join(parts) for parts in thirds))
    assert min(len(o.split()) for o in originals) >= 60  # documents of realistic length
    rng = np.random.default_rng(0)
    one = [add_typos(o, rng, rate=0.0001) for o in originals]  # a single typo
    two = [add_typos(add_typos(o, rng, rate=0.0001), rng, rate=0.0001) for o in originals]
    n = len(originals)
    cluster, canonical = cluster_near_duplicates(originals + one + two)
    recall_one = np.mean([cluster[i] != -1 and cluster[i] == cluster[n + i] for i in range(n)])
    recall_two = np.mean([cluster[i] != -1 and cluster[i] == cluster[2 * n + i] for i in range(n)])
    assert recall_one > 0.98 and recall_two > 0.95
    assert len({int(cluster[i]) for i in range(n)}) == n  # no two different originals merged
    assert canonical[:n].all()


def test_real_prose_paragraphs_are_not_duplicates_of_each_other():
    folder = Path(__file__).parents[1] / "fixtures" / "text"
    paragraphs = []
    for name in ("stories.txt", "facts.txt", "unicode.txt", "chats.txt"):
        raw = (folder / name).read_text(encoding="utf-8")
        paragraphs += [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    paragraphs = list(dict.fromkeys(paragraphs))
    cluster, canonical = cluster_near_duplicates(paragraphs)
    assert len(paragraphs) > 60
    assert (cluster == -1).all() and canonical.all()


def test_non_ascii_text_clusters_correctly():
    rng = np.random.default_rng(0)
    pool = [chr(0x4E00 + i) for i in range(400)]
    cjk = ["".join(rng.choice(pool, 150)) for _ in range(30)]
    edited = cjk[0][:70] + pool[399] + cjk[0][71:]  # one character changed
    cluster, _ = cluster_near_duplicates(cjk + [edited, cjk[1]])
    assert cluster[30] == cluster[0] != -1 and cluster[31] == cluster[1] != -1
    assert (cluster[2:30] == -1).all()  # the other texts have no twin
    cafe = "Caf\u00e9 society in Vienna \u2014 na\u00efve travelers order \u00e9clairs every day"
    shouted = cafe.upper().replace("\u00c9", "\u00e9")  # same words, other case
    cluster, _ = cluster_near_duplicates([cafe, "Something else entirely: boring words", shouted])
    assert cluster.tolist() == [0, -1, 0]


def test_texts_without_words_do_not_collapse_into_one_cluster():
    texts = [
        "!!!",
        "???",
        "\U0001f600\U0001f600",
        "\U0001f600\U0001f601",
        "...",
        "!!!",
        "",
        " ",
        "x",
    ]
    cluster, canonical = cluster_near_duplicates(texts)
    assert cluster.tolist() == [0, -1, -1, -1, -1, 0, 1, 1, -1]
    assert canonical.tolist() == [True, True, True, True, True, False, True, False, True]
