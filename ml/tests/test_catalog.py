import json
from pathlib import Path

import numpy as np

from airace_ml.catalog import (
    CLEANING_LEVELS,
    MAX_SAMPLE_CHARS,
    build_catalog,
    catalog_json,
    cut_text,
    doc_text,
    prep_key,
    reference_traits,
    write_catalog,
)
from airace_ml.data.corpus import DATASET_IDS, Corpus
from airace_ml.data.identity import data_identity
from airace_ml.data.prep import PrepConfig, eligible_docs
from airace_ml.paths import REPO_ROOT, corpus_dir
from airace_ml.tokenizer import encode_chat

SAMPLE = REPO_ROOT / "packages" / "protocol" / "fixtures" / "catalog.sample.json"
PREP_KEYS = [
    prep_key(c, d, f) for c in CLEANING_LEVELS for d in (False, True) for f in (False, True)
]


def test_every_dataset_has_all_twelve_pools(tiny_data_root):
    catalog = build_catalog(tiny_data_root)
    assert [d["id"] for d in catalog["datasets"]] == list(DATASET_IDS)
    assert catalog["dataIdentity"] == data_identity(tiny_data_root)
    for d in catalog["datasets"]:
        assert sorted(d["pools"]) == sorted(PREP_KEYS)
        for dedup in "01":
            for fact in "01":
                light = d["pools"][f"light|{dedup}|{fact}"]
                thorough = d["pools"][f"thorough|{dedup}|{fact}"]
                assert thorough["docs"] <= light["docs"]
                assert thorough["tokens"] <= light["tokens"]
        assert d["pools"]["light|0|0"]["docs"] <= d["docs"]


def test_samples_are_short_and_clean(tiny_data_root, tiny_tok):
    catalog = build_catalog(tiny_data_root)
    for d in catalog["datasets"]:
        corpus = Corpus.open(corpus_dir(tiny_data_root) / d["id"])
        clean = eligible_docs(corpus, PrepConfig(cleaning="thorough", dedup=True, fact_check=True))
        clean_texts = {doc_text(tiny_tok, corpus.doc(int(i))).strip() for i in clean}
        assert 1 <= len(d["samples"]) <= 5
        for s in d["samples"]:
            assert len(s) <= MAX_SAMPLE_CHARS
            assert s in clean_texts or (
                s.endswith("…") and any(t.startswith(s[:-1]) for t in clean_texts)
            )
        removed = d["removedExamples"]
        assert sorted(removed["cleaning"]) == sorted(CLEANING_LEVELS)
        for examples in [
            *removed["cleaning"].values(),
            removed["duplicates"],
            removed["factCheck"],
        ]:
            assert len(examples) <= 3
            assert all(len(e) <= MAX_SAMPLE_CHARS for e in examples)
        assert removed["duplicates"], "the tiny corpora have duplicate clusters"
        assert removed["factCheck"], "the tiny corpora have planted false facts"


def test_matches_the_game_servers_catalog_shape(tiny_data_root):
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    catalog = build_catalog(tiny_data_root)
    assert set(catalog) == set(sample)
    assert {k: catalog[k] for k in ("version", "tokenizer", "suite")} == {
        k: sample[k] for k in ("version", "tokenizer", "suite")
    }
    for ours, theirs in zip(catalog["datasets"], sample["datasets"], strict=True):
        assert set(ours) == set(theirs)
        assert ours["title"] == theirs["title"]
        assert set(ours["removedExamples"]) == set(theirs["removedExamples"])
        assert set(ours["removedExamples"]["cleaning"]) == set(
            theirs["removedExamples"]["cleaning"]
        )
        assert set(ours["pools"]) == set(theirs["pools"])


def test_the_same_data_gives_the_same_bytes(tiny_data_root, tmp_path):
    a = write_catalog(tiny_data_root, out=tmp_path / "a.json").read_bytes()
    b = write_catalog(tiny_data_root, out=tmp_path / "b.json").read_bytes()
    assert a == b
    assert a.decode("utf-8") == catalog_json(json.loads(a))


def test_reference_traits_come_from_gate_fingerprints(tmp_path):
    for name, curious in (("run-b", 0.3), ("run-a", 0.1)):
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        (folder / "gate_fingerprint.json").write_text(
            json.dumps(
                {"key": {}, "value": {"traits": {"warmth": 0.5, "curious": curious}, "samples": []}}
            )
        )
    (tmp_path / "runs" / "other.json").write_text("{}")
    assert reference_traits([tmp_path / "runs"]) == [
        {"curious": 0.1, "warmth": 0.5},
        {"curious": 0.3, "warmth": 0.5},
    ]
    assert reference_traits([]) == []


def test_cut_text_stops_at_a_word():
    assert cut_text("short") == "short"
    long = "word " * 200
    cut = cut_text(long)
    assert len(cut) <= MAX_SAMPLE_CHARS
    assert cut.endswith("word…")


def test_conversations_read_as_turns(tiny_tok):
    ids = encode_chat(tiny_tok, [("user", "Hi there"), ("ai", "Hello!")])
    assert doc_text(tiny_tok, np.array(ids)) == "Person: Hi there\nAI: Hello!"


def test_writes_to_the_data_root_by_default(tiny_data_root):
    path = write_catalog(tiny_data_root)
    assert path == Path(tiny_data_root) / "catalog" / "v1" / "catalog.json"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == "catalog-v1"


def test_cli_exports_the_catalog(tiny_data_root, tmp_path, capsys):
    from airace_ml.cli import main

    out = tmp_path / "catalog.json"
    assert main(["export-catalog", "--data-root", str(tiny_data_root), "--out", str(out)]) == 0
    assert "Wrote the dataset catalog" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["dataIdentity"] == data_identity(
        tiny_data_root
    )
