import copy

import pytest
import torch
import torch.nn.functional as F

from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.train.optim import MuonAdamW, orthogonalize


def _params(*shapes, seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s)) for s in shapes]


def _set_grads(params, seed):
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


@pytest.mark.parametrize("shape", [(3, 8, 8), (2, 6, 16), (4, 16, 6)])
def test_orthogonalize_brings_every_singular_value_near_one(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape) * torch.tensor([0.01, 1.0, 50.0, 3.0][: shape[0]])[:, None, None]
    out = orthogonalize(g, steps=5)
    assert out.shape == g.shape and out.dtype == g.dtype
    s = torch.linalg.svdvals(out)
    assert s.min() > 0.5 and s.max() < 1.3  # whatever the gradient's own size or shape
    # each matrix of the batch is treated on its own
    for i in range(shape[0]):
        torch.testing.assert_close(orthogonalize(g[i : i + 1], steps=5)[0], out[i], atol=1e-5, rtol=0)
    # and it keeps the gradient's directions: U V^T of its SVD, up to the iteration's tolerance
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    assert (out - u @ vh).abs().max() < 0.35


def test_orthogonalize_leaves_a_zero_matrix_zero():
    out = orthogonalize(torch.zeros(2, 4, 6))
    assert torch.equal(out, torch.zeros(2, 4, 6))


def test_a_muon_step_moves_each_matrix_by_its_orthogonalized_gradient():
    square, tall, wide = _params((8, 8), (16, 4), (4, 16))
    opt = MuonAdamW([square, tall, wide], [], [], matrix_lr=0.1, other_lr=0.0, momentum=0.0)
    for group in opt.param_groups:
        group["lr"] = group["peak_lr"]
    before = [p.detach().clone() for p in (square, tall, wide)]
    _set_grads([square, tall, wide], seed=1)
    grads = [p.grad.clone() for p in (square, tall, wide)]
    opt.step()
    for p, old, g, scale in zip((square, tall, wide), before, grads, (1.0, 2.0, 1.0), strict=True):
        want = old - 0.1 * scale * orthogonalize(g[None])[0]  # tall: sqrt(16 / 4) = 2
        torch.testing.assert_close(p.detach(), want, atol=1e-6, rtol=0)


def test_matrices_of_one_shape_are_updated_together_as_if_each_were_alone():
    a, b, c = _params((6, 6), (6, 6), (6, 6))
    together = MuonAdamW([a, b, c], [], [], matrix_lr=0.05, other_lr=0.0)
    alone_params = [torch.nn.Parameter(p.detach().clone()) for p in (a, b, c)]
    alone = [MuonAdamW([p], [], [], matrix_lr=0.05, other_lr=0.0) for p in alone_params]
    for step in range(4):
        _set_grads([a, b, c], seed=10 + step)
        for p, q in zip((a, b, c), alone_params, strict=True):
            q.grad = p.grad.clone()
        for opt in (together, *alone):
            for group in opt.param_groups:
                group["lr"] = group["peak_lr"]
            opt.step()
    for p, q in zip((a, b, c), alone_params, strict=True):
        torch.testing.assert_close(p, q, atol=1e-5, rtol=0)


def test_the_adam_part_is_torchs_adamw():
    emb, gain1, gain2 = _params((20, 8), (8,), (8,))
    ours = MuonAdamW([], [emb], [gain1, gain2], matrix_lr=0.0, other_lr=3e-3, weight_decay=0.1)
    ref_params = [torch.nn.Parameter(p.detach().clone()) for p in (emb, gain1, gain2)]
    ref = torch.optim.AdamW(
        [
            {"params": ref_params[:1], "weight_decay": 0.1},
            {"params": ref_params[1:], "weight_decay": 0.0},
        ],
        lr=3e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    for step in range(6):
        lr = 3e-3 * (step + 1) / 6  # a changing rate, as the schedule gives it
        _set_grads([emb, gain1, gain2], seed=20 + step)
        for p, q in zip((emb, gain1, gain2), ref_params, strict=True):
            q.grad = p.grad.clone()
        for group in (*ours.param_groups, *ref.param_groups):
            group["lr"] = lr
        ours.step()
        ref.step()
    for p, q in zip((emb, gain1, gain2), ref_params, strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def _model_optimizer(model):
    emb = [model.tok_emb.weight]
    matrices = [p for p in model.parameters() if p.dim() == 2 and p is not model.tok_emb.weight]
    gains = [p for p in model.parameters() if p.dim() < 2]
    return MuonAdamW(matrices, emb, gains, matrix_lr=0.02, other_lr=0.01)


def _train(model, opt, steps, seed):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, 64, (8, 17))
        ids[:, 1::2] = ids[:, 0:-1:2]  # every second token repeats the one before: learnable
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        for group in opt.param_groups:
            group["lr"] = group["peak_lr"]
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def test_a_model_learns_and_the_state_round_trips_exactly():
    torch.manual_seed(0)
    model = Transformer(ModelShape(2, 64, 64), 64)
    opt = _model_optimizer(model)
    assert [g["kind"] for g in opt.param_groups] == ["muon", "adam", "adam"]
    losses = _train(model, opt, 30, seed=1)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.3
    # a copy of the model with the optimizer's state loaded continues exactly as the original does
    twin = copy.deepcopy(model)
    twin_opt = _model_optimizer(twin)
    twin_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    _train(model, opt, 5, seed=2)
    _train(twin, twin_opt, 5, seed=2)
    for p, q in zip(model.parameters(), twin.parameters(), strict=True):
        assert torch.equal(p, q)


def test_no_rate_no_change_and_a_missing_gradient_counts_as_zero():
    a, b = _params((5, 5), (5, 5))
    emb, gain = _params((9, 5), (5,), seed=1)
    opt = MuonAdamW([a, b], [emb], [gain], matrix_lr=0.05, other_lr=0.01)
    before = [p.detach().clone() for p in (a, b, emb, gain)]
    _set_grads([a, b, emb, gain], seed=3)
    opt.step()  # every group's rate is still 0
    for p, old in zip((a, b, emb, gain), before, strict=True):
        assert torch.equal(p.detach(), old)
    for group in opt.param_groups:
        group["lr"] = group["peak_lr"]
    b.grad = None
    opt.step()
    assert not torch.equal(a.detach(), before[0]) and torch.isfinite(b).all()


def test_bad_arguments_are_refused():
    (vector,) = _params((5,))
    (matrix,) = _params((5, 5))
    with pytest.raises(ValueError, match="matrices"):
        MuonAdamW([vector], [], [], matrix_lr=0.1, other_lr=0.1)
    with pytest.raises(ValueError, match="momentum"):
        MuonAdamW([matrix], [], [], matrix_lr=0.1, other_lr=0.1, momentum=1.0)
    with pytest.raises(ValueError, match="ns_steps"):
        MuonAdamW([matrix], [], [], matrix_lr=0.1, other_lr=0.1, ns_steps=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_cuda_the_iteration_runs_in_bf16_and_tracks_the_cpu():
    torch.manual_seed(0)
    g = torch.randn(4, 32, 96)
    cpu = orthogonalize(g, steps=5)
    gpu = orthogonalize(g.cuda(), steps=5)
    assert gpu.dtype == torch.float32 and gpu.is_cuda
    assert (gpu.cpu() - cpu).abs().max() < 0.05
