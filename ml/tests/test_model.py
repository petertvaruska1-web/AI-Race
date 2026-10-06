import pytest
import torch

from airace_ml.model.checkpoint import CheckpointMeta, load_checkpoint, save_checkpoint
from airace_ml.model.shape import ModelShape, ShapeError
from airace_ml.model.transformer import Transformer


def test_reference_param_counts():
    assert ModelShape(4, 128, 256).param_count() == 1_377_408
    assert ModelShape(8, 256, 256).param_count() == 7_475_456
    assert Transformer(ModelShape(2, 64, 64), 512).num_params() == ModelShape(2, 64, 64).param_count(512)

@pytest.mark.parametrize("s", [(2, 63, 64), (0, 64, 64), (2, 64, 100), (2, 64, 2048), (25, 64, 64), (24, 1024, 1024)])
def test_invalid_shapes(s):
    with pytest.raises(ShapeError):
        ModelShape(*s).validate()

def test_forward_and_causality():
    torch.manual_seed(0); m = Transformer(ModelShape(2, 64, 64), 512).eval()
    x = torch.randint(0, 512, (3, 20)); y = x.clone(); y[:, 15:] = 7
    assert m(x).shape == (3, 20, 512)
    torch.testing.assert_close(m(x)[:, :15], m(y)[:, :15])

def test_kv_cache_matches_full_forward():
    torch.manual_seed(0); m = Transformer(ModelShape(2, 64, 64), 512).eval()
    x = torch.randint(0, 512, (2, 12)); full = m(x)
    cache = m.new_kv_cache(); out = [m(x[:, :8], kv_cache=cache)]
    for t in range(8, 12):
        out.append(m(x[:, t:t + 1], kv_cache=cache))
    torch.testing.assert_close(torch.cat(out, 1), full, atol=1e-5, rtol=0)

def test_left_padding_matches_unpadded():
    torch.manual_seed(0); m = Transformer(ModelShape(2, 64, 64), 512).eval()
    a = torch.tensor([[5, 6, 7]]); padded = torch.tensor([[0, 0, 5, 6, 7]])
    mask = torch.tensor([[False, False, True, True, True]]); pos = torch.tensor([[0, 0, 0, 1, 2]])
    torch.testing.assert_close(m(padded, positions=pos, key_padding_mask=mask)[:, -1], m(a)[:, -1], atol=1e-5, rtol=0)

def test_checkpoint_roundtrip(tmp_path):
    m = Transformer(ModelShape(2, 64, 64), 512)
    meta = CheckpointMeta(tokenizer="tok-v1", shape=m.shape, lineage_id="L", version_id="v1",
                          parent_version_id=None, tokens_trained_total=0, last_mixture={"web": 1.0}, runs=[])
    save_checkpoint(m, meta, tmp_path)
    m2, meta2 = load_checkpoint(tmp_path)
    x = torch.randint(0, 512, (1, 10))
    torch.testing.assert_close(m2(x), m(x)); assert meta2 == meta
    assert m2.lm_head.weight is m2.tok_emb.weight


# --- Beyond the brief: the contracts Task 4 (growth) and Task 5 (batched generation) rely on ---


def test_shape_derived_values_and_dict_roundtrip():
    s = ModelShape(4, 128, 256)
    assert (s.n_head, s.ffn_hidden) == (4, 384)
    assert ModelShape(8, 256, 256).ffn_hidden == 704
    assert ModelShape.from_dict(s.to_dict()) == s


def test_block_layout_is_bias_free_and_named_for_growth():
    m = Transformer(ModelShape(2, 64, 64), 512)
    for b in m.blocks:
        for lin in (b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo,
                    b.mlp.w_gate, b.mlp.w_up, b.mlp.w_down):
            assert isinstance(lin, torch.nn.Linear) and lin.bias is None
        assert b.attn_norm.weight.shape == b.mlp_norm.weight.shape == (64,)
    assert m.norm_f.weight.shape == (64,) and m.lm_head.bias is None


def test_rmsnorm_eps_is_a_saved_buffer(tmp_path):
    m = Transformer(ModelShape(2, 64, 64), 512)
    assert "blocks.0.attn_norm.eps" in m.state_dict() and "norm_f.eps" in m.state_dict()
    m.blocks[1].mlp_norm.eps.fill_(3e-4)
    meta = CheckpointMeta(tokenizer="tok-v1", shape=m.shape, lineage_id="L", version_id="v1",
                          parent_version_id=None, tokens_trained_total=0, last_mixture={}, runs=[])
    save_checkpoint(m, meta, tmp_path)
    m2, _ = load_checkpoint(tmp_path)
    assert m2.blocks[1].mlp_norm.eps.item() == pytest.approx(3e-4)
    assert m2.blocks[0].mlp_norm.eps.item() == pytest.approx(1e-5)


def test_cached_left_padded_batch_matches_unpadded_rows():
    torch.manual_seed(0); m = Transformer(ModelShape(2, 64, 64), 512).eval()
    rows = [[5, 6, 7], [1, 2, 3, 4, 5]]
    padded = torch.tensor([[0, 0, 5, 6, 7], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[False, False, True, True, True], [True] * 5])
    pos = torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])
    cache = m.new_kv_cache()
    prefill = m(padded, positions=pos, key_padding_mask=mask, kv_cache=cache)
    nxt = torch.tensor([[9], [10]])
    mask = torch.cat([mask, torch.ones(2, 1, dtype=torch.bool)], dim=1)
    step = m(nxt, positions=torch.tensor([[3], [5]]), key_padding_mask=mask, kv_cache=cache)
    assert torch.isfinite(prefill).all() and torch.isfinite(step).all()  # no NaN from pad rows
    assert cache.length == 6 and cache.k[0].shape == (2, 2, 6, 32)
    for i, row in enumerate(rows):
        ref = m(torch.tensor([row + [int(nxt[i, 0])]]))[:, -1]
        torch.testing.assert_close(step[i, -1], ref[0], atol=1e-5, rtol=0)


def test_cache_prefill_continuation_and_bad_mask_shape():
    torch.manual_seed(0); m = Transformer(ModelShape(2, 64, 64), 512).eval()
    x = torch.randint(0, 512, (2, 12)); cache = m.new_kv_cache()
    out = torch.cat([m(x[:, :5], kv_cache=cache), m(x[:, 5:], kv_cache=cache)], dim=1)
    torch.testing.assert_close(out, m(x), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        m(x, key_padding_mask=torch.ones(2, 11, dtype=torch.bool))
