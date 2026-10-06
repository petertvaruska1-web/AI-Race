import pytest
import torch

from airace_ml.model.growth import GrowthError, grow, insertion_layout
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer


def _perturbed(shape):
    torch.manual_seed(0)
    m = Transformer(ModelShape(*shape), 512)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))  # make norm gains != 1 so the test is meaningful
    return m.eval()


@pytest.mark.parametrize(
    "src,dst",
    [
        ((2, 64, 64), (2, 128, 64)),
        ((2, 64, 64), (4, 64, 64)),
        ((2, 64, 64), (5, 96, 128)),
        ((1, 64, 64), (1, 64, 64)),
    ],
)
def test_growth_preserves_function(src, dst):
    m = _perturbed(src)
    x = torch.randint(0, 512, (2, 48))
    torch.testing.assert_close(grow(m, ModelShape(*dst), seed=1).eval()(x), m(x), atol=1e-4, rtol=0)


def test_growth_rejects_shrinking():
    m = _perturbed((2, 96, 64))
    for bad in [ModelShape(1, 96, 64), ModelShape(2, 64, 64)]:
        with pytest.raises(GrowthError):
            grow(m, bad, seed=0)


def test_insertion_layout():
    assert insertion_layout(4, 6) == [0, 1, None, 2, 3, None]
    assert insertion_layout(2, 2) == [0, 1]


def test_new_parameters_receive_gradient():
    g = grow(_perturbed((2, 64, 64)), ModelShape(3, 96, 64), seed=1)
    g(torch.randint(0, 512, (2, 16))).logsumexp(-1).mean().backward()
    new_block = next(i for i, v in enumerate(insertion_layout(2, 3)) if v is None)
    assert g.blocks[new_block].attn.wo.weight.grad.abs().sum() > 0
    assert g.tok_emb.weight.grad[:, 64:].abs().sum() > 0


def test_seed_controls_new_parameters_and_leaves_global_rng_alone():
    m = _perturbed((2, 64, 64))
    target = ModelShape(3, 96, 64)
    torch.manual_seed(5)
    expected_next = torch.rand(3)
    torch.manual_seed(5)
    a = grow(m, target, seed=3)
    assert torch.equal(torch.rand(3), expected_next)  # growing did not advance the global RNG
    b = grow(m, target, seed=3)
    c = grow(m, target, seed=4)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    assert not torch.equal(a.blocks[0].attn.wq.weight, c.blocks[0].attn.wq.weight)


def test_grown_model_keeps_dtype_tying_and_mode():
    m = _perturbed((2, 64, 64)).to(torch.bfloat16)
    g = grow(m, ModelShape(2, 96, 64), seed=1)
    assert {p.dtype for p in g.parameters()} == {torch.bfloat16}
    assert g.norm_f.eps.dtype == m.norm_f.eps.dtype
    assert g.lm_head.weight is g.tok_emb.weight
    assert g.training == m.training
