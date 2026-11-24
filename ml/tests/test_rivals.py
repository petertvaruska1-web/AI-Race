import json
from pathlib import Path

import pytest
import torch

from airace_ml.rivals.build import (
    REPLIES_PER_GROUP,
    _Built,
    build_rival_pack,
    pack_document,
    version_key,
)
from airace_ml.rivals.recipe import LINEAGES

FIXTURE = (
    Path(__file__).parents[2] / "packages" / "protocol" / "fixtures" / "rivals" / "rivals.json"
)
STYLES = ("premium", "budget", "specialist", "consumer")


def test_eight_lineages_two_per_style_that_only_grow():
    assert len(LINEAGES) == 8
    assert sorted(r.style for r in LINEAGES) == sorted(STYLES * 2)
    assert len({r.id for r in LINEAGES}) == 8
    for r in LINEAGES:
        assert 1 <= len(r.steps) <= 5
        for a, b in zip(r.steps, r.steps[1:]):
            assert b.shape.n_layer >= a.shape.n_layer and b.shape.d_model >= a.shape.d_model
        assert sum(r.mix.values()) == pytest.approx(1.0)
    assert REPLIES_PER_GROUP == 24


def _keys(value):
    """The field names at every level of a pack document (first list entry stands for all)."""
    if isinstance(value, dict):
        return {k: _keys(v) for k, v in value.items() if k not in ("scores", "traits", "breakdown")}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [_keys(value[0])]
    return None


def test_the_pack_document_has_exactly_the_protocol_fixtures_fields():
    measured = {
        "scores": {"language": 50.0, "creativity": None},
        "overall": 50.0,
        "traits": {"verbosity": 0.5},
        "breakdown": {"knowledge/topic:animals": [40.0, 12]},
        "replies": [],
    }
    built = [
        _Built(version_key(r, k), r, step.shape, step.tokens * k, measured)
        for r in LINEAGES
        for k, step in enumerate(r.steps[:2], start=1)
    ]
    doc = pack_document("identity", False, built, {})
    assert _keys(doc) == _keys(json.loads(FIXTURE.read_text("utf-8")))
    assert [len(lin["versions"]) for lin in doc["lineages"]] == [2] * 8


@pytest.mark.slow
def test_a_quick_build_makes_a_whole_pack_and_a_rerun_reuses_it(tiny_data_root, tmp_path):
    out = tmp_path / "rivals"
    said = []
    index = build_rival_pack(
        tiny_data_root, out, torch.device("cpu"), quick=True, on_status=said.append
    )
    pack = json.loads(index.read_text("utf-8"))
    assert pack["quick"] is True and len(pack["lineages"]) == 8
    keys = [v["key"] for lin in pack["lineages"] for v in lin["versions"]]
    assert len(keys) == 16
    for key in keys:
        assert (out / "models" / f"{key}.tar").stat().st_size > 0
        replies = json.loads((out / "replies" / f"{key}.json").read_text("utf-8"))
        assert len(replies) == 12
        assert all(r["question"] and isinstance(r["reply"], str) for r in replies)
        assert {r["group"] for r in replies} == {
            "students",
            "creators",
            "developers",
            "businesses",
            "everyday",
            "researchers",
        }
    v = pack["lineages"][0]["versions"][1]
    assert v["trainedTokens"] > pack["lineages"][0]["versions"][0]["trainedTokens"]
    assert v["breakdown"] and v["scores"]["creativity"] is None  # no judge in this data
    assert all(isinstance(n, int) and n >= 10 for _, n in v["breakdown"].values())
    said.clear()
    build_rival_pack(tiny_data_root, out, torch.device("cpu"), quick=True, on_status=said.append)
    assert sorted(m for m in said if m.startswith("reused ")) == sorted(f"reused {k}" for k in keys)


def test_a_rivals_product_is_named_as_the_game_names_a_rivals_releases():
    from airace_ml.rivals.recipe import product_name

    first, second = (r for r in LINEAGES if r.style == "premium")
    assert [product_name(first, k) for k in (1, 2, 3)] == ["Titan", "Titan 2", "Titan 3"]
    assert product_name(second, 1) == "Apex"
    names = {product_name(r, 1) for r in LINEAGES}
    assert len(names) == 8  # no two lineages share a first product's name


def test_the_product_words_are_the_games(tmp_path):
    import re

    from airace_ml.rivals.recipe import PRODUCT_WORDS

    ts = (Path(__file__).parents[2] / "packages" / "sim" / "src" / "rivalCast.ts").read_text("utf-8")
    block = ts[ts.index("export const PRODUCT_WORDS") :].split("};")[0]
    for style, words in PRODUCT_WORDS.items():
        match = re.search(rf"{style}: \[(.*?)\]", block)
        assert match is not None
        assert tuple(re.findall(r'"(\w+)"', match.group(1))) == words


def _checkpoint_of(text, tiny_tok, path):
    from airace_ml.model.checkpoint import CheckpointMeta, save_checkpoint
    from airace_ml.model.shape import ModelShape
    from airace_ml.model.transformer import Transformer

    shape = ModelShape(2, 64, 64)
    run = {"run_id": "r"} if text is None else {"run_id": "r", "text": text}
    meta = CheckpointMeta("tok-v1", shape, "l", "r", None, 0, {}, [run])
    save_checkpoint(Transformer(shape, tiny_tok.vocab_size), meta, path)
    return path


def test_a_text_2_rival_is_told_its_products_name_and_a_text_1_rival_is_not(tiny_tok, tmp_path):
    from airace_ml.rivals.build import version_system

    recipe = next(r for r in LINEAGES if r.style == "premium")
    two = _checkpoint_of(2, tiny_tok, tmp_path / "two")
    assert version_system(two, recipe, 1) == "You are Titan, an AI."
    assert version_system(two, recipe, 2) == "You are Titan 2, an AI."
    for text in (1, None):
        old = _checkpoint_of(text, tiny_tok, tmp_path / f"old-{text}")
        assert version_system(old, recipe, 1) is None


def test_a_rivals_pre_drawn_answers_use_its_line(tiny_tok, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from airace_ml.rivals import build

    path = _checkpoint_of(2, tiny_tok, tmp_path / "ck")
    monkeypatch.setattr(
        build,
        "run_benchmarks",
        lambda *a, **k: SimpleNamespace(scores={}, overall=0.0, tag_breakdown=dict),
    )
    monkeypatch.setattr(
        build, "measure_fingerprint", lambda *a, **k: SimpleNamespace(traits={})
    )
    asked = []
    original = build.TorchLM.chat_reply

    def spy(self, history, **kwargs):
        asked.append(kwargs.get("system"))
        return original(self, history, **kwargs)

    monkeypatch.setattr(build.TorchLM, "chat_reply", spy)
    run = SimpleNamespace(dir=path)
    build._measure(run, tiny_tok, torch.device("cpu"), None, None, True, "You are Titan, an AI.")
    assert asked and set(asked) == {"You are Titan, an AI."}
    asked.clear()
    build._measure(run, tiny_tok, torch.device("cpu"), None, None, True)  # text 1: as before
    assert asked and set(asked) == {None}


def test_version_keys_of_pack_3_carry_r3():
    from airace_ml.rivals.build import PACK_VERSION

    assert PACK_VERSION == "rivals-v3"
    assert version_key(LINEAGES[0], 2) == f"{LINEAGES[0].id}-r3-v2"


def test_rivals_dir_is_the_third_pack(tmp_path):
    from airace_ml.paths import rivals_dir

    assert rivals_dir(tmp_path) == tmp_path / "rivals" / "v3"


def test_a_folder_holding_another_pack_is_refused(tmp_path):
    from airace_ml.rivals.build import PackFolderError, check_pack_folder

    check_pack_folder(tmp_path / "not-there")  # a new folder
    check_pack_folder(tmp_path)  # an empty one
    index = tmp_path / "rivals.json"
    index.write_text(json.dumps({"version": "rivals-v3"}), "utf-8")
    check_pack_folder(tmp_path)  # this pack, built again
    index.write_text(json.dumps({"version": "rivals-v2"}), "utf-8")
    with pytest.raises(PackFolderError, match="rivals-v2"):
        check_pack_folder(tmp_path)
    index.write_text("{not json", "utf-8")
    with pytest.raises(PackFolderError, match="rivals.json"):
        check_pack_folder(tmp_path)
