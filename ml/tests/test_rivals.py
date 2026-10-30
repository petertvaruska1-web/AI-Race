import json
from pathlib import Path

import pytest
import torch

from airace_ml.rivals.build import REPLIES_PER_GROUP, _Built, build_rival_pack, pack_document
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
        _Built(f"{r.id}-v{k}", r, step.shape, step.tokens * k, measured)
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
