"""The trainer's optimizer (engine 2): orthogonalised momentum for the weight matrices, AdamW for
the embedding and the norm gains.

Adam moves every weight by about the learning rate whatever its gradient, one weight at a time.
For a weight *matrix* a better unit of change is the whole matrix: the momentum of its gradient is
replaced by the nearest "rotation-like" matrix (all singular values near 1, found with a few
Newton-Schulz iterations) before it is applied, so every direction the matrix can learn in moves
at the same rate. This is the Muon optimizer (Jordan et al., 2024). On this game's models it
learns from 12.6M tokens what AdamW learns from about 19M (ADR 0009).

The embedding (tied to the output head) and the norm gains are not matrices in that sense and
keep AdamW.

One :class:`torch.optim.Optimizer`, so the trainer's snapshots, rollbacks and resume states work
as with any optimizer. Parameters of one shape are updated together (a small brain has 28 weight
matrices in 3 shapes; updating them one by one would cost more in launches than in arithmetic):
their state lives stacked under the first parameter of each shape.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

MUON_MOMENTUM = 0.9
NS_STEPS = 3  # 3 learn as well as 5 here and cost 40% less (ADR 0009)
# The quintic iteration x <- a x + b (x x^T) x + c (x x^T)^2 x of Muon: it drives every singular
# value into about [0.7, 1.2] in a few steps instead of converging slowly to exactly 1.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
EMBEDDING_WEIGHT_DECAY = 0.1


def orthogonalize(g: Tensor, steps: int = NS_STEPS) -> Tensor:
    """For each matrix ``g[i]`` of a batch ``[n, rows, cols]``, a matrix with the same singular
    vectors and every singular value near 1 (a zero matrix stays zero). Runs in bfloat16 on CUDA,
    where it is far faster and precise enough for a step direction; float32 elsewhere."""
    x = g.bfloat16() if g.is_cuda else g
    tall = x.size(-2) > x.size(-1)
    if tall:  # iterate on the wide form: the Gram matrix is then the small one
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = _NS_A * x + (_NS_B * gram + _NS_C * (gram @ gram)) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


def _by_shape(params: list[Tensor]) -> list[list[Tensor]]:
    """``params`` grouped by shape, in first-seen order (so the grouping is the same every step
    and in a resumed run)."""
    groups: dict[torch.Size, list[Tensor]] = {}
    for p in params:
        groups.setdefault(p.shape, []).append(p)
    return list(groups.values())


def _stacked_grads(params: list[Tensor]) -> Tensor:
    """The gradients of ``params`` (all one shape) as one tensor; a missing one counts as zero."""
    return torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])


class MuonAdamW(torch.optim.Optimizer):
    """Muon for ``matrices``, AdamW with weight decay for ``embeddings`` and without for ``gains``.

    Each parameter group has its own ``lr``; the trainer sets them every step from the group's
    ``peak_lr`` and the schedule. ``matrix_lr`` and ``other_lr`` are those peaks.
    """

    def __init__(
        self,
        matrices: Iterable[Tensor],
        embeddings: Iterable[Tensor],
        gains: Iterable[Tensor],
        *,
        matrix_lr: float,
        other_lr: float,
        momentum: float = MUON_MOMENTUM,
        ns_steps: int = NS_STEPS,
        betas: tuple[float, float] = ADAM_BETAS,
        eps: float = ADAM_EPS,
        weight_decay: float = EMBEDDING_WEIGHT_DECAY,
    ) -> None:
        matrices, embeddings, gains = list(matrices), list(embeddings), list(gains)
        for p in matrices:
            if p.dim() != 2:
                raise ValueError(f"Muon takes weight matrices, got a tensor of shape {tuple(p.shape)}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum!r}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be at least 1, got {ns_steps!r}")
        groups = [
            {"params": matrices, "kind": "muon", "peak_lr": matrix_lr, "weight_decay": 0.0},
            {
                "params": embeddings,
                "kind": "adam",
                "peak_lr": other_lr,
                "weight_decay": weight_decay,
            },
            {"params": gains, "kind": "adam", "peak_lr": other_lr, "weight_decay": 0.0},
        ]
        defaults = {"lr": 0.0, "momentum": momentum, "ns_steps": ns_steps, "betas": betas, "eps": eps}
        super().__init__([g for g in groups if g["params"]], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """One update of every group (``closure``, if given, is evaluated first, as
        :class:`torch.optim.Optimizer` allows)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            update = self._muon if group["kind"] == "muon" else self._adam
            for params in _by_shape(group["params"]):
                update(group, params)
        return loss

    def _muon(self, group: dict, params: list[Tensor]) -> None:
        lr, momentum = group["lr"], group["momentum"]
        grads = _stacked_grads(params)
        state = self.state[params[0]]
        buf = state.get("momentum")
        if buf is None:
            buf = state["momentum"] = torch.zeros_like(grads)
        buf.lerp_(grads, 1.0 - momentum)
        update = orthogonalize(grads.lerp_(buf, momentum), group["ns_steps"])  # Nesterov
        rows, cols = params[0].shape
        scale = max(1.0, rows / cols) ** 0.5  # the same change per output for tall matrices
        for p, u in zip(params, update.unbind(0), strict=True):
            p.add_(u, alpha=-lr * scale)

    def _adam(self, group: dict, params: list[Tensor]) -> None:
        lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
        grads = _stacked_grads(params)
        state = self.state[params[0]]
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grads)
            state["exp_avg_sq"] = torch.zeros_like(grads)
        state["step"] += 1
        t = state["step"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(grads, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1.0 - beta2)
        denom = (exp_avg_sq.sqrt() / (1.0 - beta2**t) ** 0.5).add_(eps)
        update = exp_avg / denom
        step_size = lr / (1.0 - beta1**t)
        decay = 1.0 - lr * group["weight_decay"]
        for p, u in zip(params, update.unbind(0), strict=True):
            if decay != 1.0:
                p.mul_(decay)  # decoupled weight decay, before the step as torch's AdamW does
            p.add_(u, alpha=-step_size)
