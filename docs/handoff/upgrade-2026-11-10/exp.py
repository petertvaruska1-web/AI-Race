"""Learning-efficiency experiments: one training run with a pluggable optimizer and schedule.

Uses the product's data (pools, sampler, held-out split) and model; only the optimizer, schedule
and a few switches vary. Reports held-out loss per dataset on a large fixed eval set, at a few
points of the run and at the end, plus pure training time.

    python exp.py --name X --shape 4,128,256 --tokens 12582912 [--opt muon ...]
Appends one JSON line to --results.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import fastmodel  # noqa: E402

from airace_ml.data.corpus import DATASET_IDS, Corpus  # noqa: E402
from airace_ml.data.prep import PrepConfig, heldout_docs  # noqa: E402
from airace_ml.data.sampler import DocPool, MixtureSampler, load_pools  # noqa: E402
from airace_ml.model.checkpoint import CheckpointMeta, save_checkpoint  # noqa: E402
from airace_ml.model.shape import ModelShape  # noqa: E402
from airace_ml.model.transformer import Transformer  # noqa: E402
from airace_ml.paths import corpus_dir, data_root  # noqa: E402

DEV = torch.device("cuda")


# -- optimizers -----------------------------------------------------------------------------------


def ns_orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched Newton-Schulz: the nearest semi-orthogonal matrix to each ``g[i]`` (approximately)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalized update for weight matrices, batched over same-shaped matrices."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))
        self._groups_by_shape = None

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            by_shape: dict[tuple, list] = {}
            for p in group["params"]:
                if p.grad is not None:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for shape, plist in by_shape.items():
                g = torch.stack([p.grad for p in plist])
                st = self.state[plist[0]]
                buf = st.get("buf")
                if buf is None:
                    buf = st["buf"] = torch.zeros_like(g)
                buf.lerp_(g, 1 - mom)
                u = g.lerp_(buf, mom) if group["nesterov"] else buf
                u = ns_orthogonalize(u, group["ns_steps"])
                scale = max(1.0, shape[0] / shape[1]) ** 0.5
                for p, ui in zip(plist, u, strict=True):
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(ui, alpha=-lr * scale)


def make_optimizers(model: Transformer, a) -> list[tuple[torch.optim.Optimizer, float]]:
    """[(optimizer, peak lr)]; the schedule multiplies every peak by the same factor."""
    params = list(model.parameters())
    mats = [p for p in params if p.dim() >= 2]
    gains = [p for p in params if p.dim() < 2]
    if a.opt == "adamw":
        opt = torch.optim.AdamW(
            [{"params": mats, "weight_decay": a.wd}, {"params": gains, "weight_decay": 0.0}],
            lr=0.0, betas=(a.beta1, a.beta2), eps=a.eps, fused=True,
        )
        return [(opt, a.lr)]
    if a.opt == "muon":
        emb = [model.tok_emb.weight]
        hidden = [p for p in mats if p is not model.tok_emb.weight]
        adam = torch.optim.AdamW(
            [{"params": emb, "weight_decay": a.emb_wd}, {"params": gains, "weight_decay": 0.0}],
            lr=0.0, betas=(a.beta1, a.beta2), eps=a.eps, fused=True,
        )
        muon = Muon(hidden, lr=0.0, momentum=a.momentum, weight_decay=a.wd, ns_steps=a.ns_steps)
        return [(muon, a.lr), (adam, a.emb_lr)]
    raise ValueError(a.opt)


def lr_factor(step: int, total: int, a) -> float:
    warm = max(1, int(a.warmup * total))
    up = min(1.0, (step + 1) / warm)
    if a.sched == "wsd":
        decay = max(1, int(a.decay * total))
        start = total - decay
        if step < start:
            down = 1.0
        else:
            prog = 1.0 if decay == 1 else min(1.0, (step - start) / (decay - 1))
            down = 1.0 - prog * (1.0 - a.min_ratio)
    elif a.sched == "linear":  # linear to min_ratio over everything after warmup
        prog = min(1.0, max(0.0, (step - warm) / max(1, total - warm - 1)))
        down = 1.0 - prog * (1.0 - a.min_ratio)
    elif a.sched == "cosine":
        prog = min(1.0, max(0.0, (step - warm) / max(1, total - warm - 1)))
        down = a.min_ratio + (1 - a.min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    else:
        raise ValueError(a.sched)
    return min(up, down)


# -- data -----------------------------------------------------------------------------------------


def eval_set(root: Path, seq_len: int, n_seqs: int):
    out = {}
    for name in DATASET_IDS:
        corpus = Corpus.open(corpus_dir(root) / name)
        pool = DocPool.from_corpus(corpus, heldout_docs(corpus), balanced=False)
        s = MixtureSampler({name: pool}, {name: 1.0}, seq_len, seed=0)
        x, y = s.next_batch(n_seqs)
        out[name] = (torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV))
    return out


@torch.inference_mode()
def evaluate(model, batches, rows_per_pass):
    model.eval()
    res = {}
    for name, (x, y) in batches.items():
        tot = 0.0
        for s in range(0, len(x), rows_per_pass):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x[s : s + rows_per_pass])
            tot += F.cross_entropy(logits.float().flatten(0, 1), y[s : s + rows_per_pass].flatten(),
                                   reduction="sum").item()
        res[name] = tot / y.numel()
    model.train()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    p.add_argument("--shape", default="4,128,256")
    p.add_argument("--tokens", type=int, default=12_582_912)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mix", default="even")
    p.add_argument("--cleaning", default="standard")
    p.add_argument("--opt", default="adamw")
    p.add_argument("--lr", type=float, default=2.683e-3)
    p.add_argument("--emb_lr", type=float, default=2.683e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--emb_wd", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--clip", type=float, default=1.5)
    p.add_argument("--sched", default="wsd")
    p.add_argument("--warmup", type=float, default=0.04)
    p.add_argument("--decay", type=float, default=0.2)
    p.add_argument("--min_ratio", type=float, default=0.1)
    p.add_argument("--fast", default="norm")
    p.add_argument("--casual", type=float, default=0.0, help="share of chat docs typed casually")
    p.add_argument("--init", default="", help="'+'-joined: zero (zero-init wo/w_down)")
    p.add_argument("--eval_seqs", type=int, default=128)
    p.add_argument("--evals", type=int, default=4)
    p.add_argument("--parent", default=None, help="checkpoint dir to continue from")
    p.add_argument("--save", default=None)
    a = p.parse_args()

    fastmodel.apply(a.fast)
    root = data_root()
    shape = ModelShape(*[int(v) for v in a.shape.split(",")])
    if a.mix == "even":
        mix = {n: 1.0 for n in DATASET_IDS}
    else:
        mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    prep = PrepConfig(cleaning=a.cleaning)
    pools = load_pools(root, mix, prep, {}, {})
    if a.casual > 0:
        import casual
        from airace_ml.paths import tokenizer_path
        from airace_ml.tokenizer import Tok

        casual.wrap_pools(pools, Tok.load(tokenizer_path(root)), a.casual)
    sampler = MixtureSampler(pools, mix, shape.ctx_len, a.seed)
    if a.parent:
        from airace_ml.model.checkpoint import load_checkpoint
        from airace_ml.model.growth import grow

        model, _ = load_checkpoint(Path(a.parent))
        if model.shape != shape:
            model = grow(model, shape, seed=a.seed)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(a.seed)
            model = Transformer(shape)
        if "zero" in a.init.split("+"):
            with torch.no_grad():
                for block in model.blocks:
                    block.attn.wo.weight.zero_()
                    block.mlp.w_down.weight.zero_()
    model = model.to(DEV).train()
    params = list(model.parameters())
    opts = make_optimizers(model, a)
    rows = max(1, a.batch // shape.ctx_len)
    n_tok = rows * shape.ctx_len
    steps = max(1, a.tokens // a.batch)
    micro_rows = max(1, 8192 // shape.ctx_len)
    batches = eval_set(root, shape.ctx_len, a.eval_seqs)
    eval_at = {max(1, round(steps * (i + 1) / a.evals)) for i in range(a.evals)}

    curve = []
    train_time = 0.0
    losses = []
    diverged = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        f = lr_factor(i, steps, a)
        for opt, peak in opts:
            for g in opt.param_groups:
                g["lr"] = peak * f
        x, y = sampler.next_batch(rows)
        x, y = torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)
        total = torch.zeros((), device=DEV)
        for s in range(0, rows, micro_rows):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x[s : s + micro_rows]).flatten(0, 1),
                                       y[s : s + micro_rows].flatten(), reduction="sum") / n_tok
            loss.backward()
            total += loss.detach()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        for opt, _ in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if (i + 1) % 10 == 0 or i + 1 == steps:
            v = total.item()
            losses.append(v)
            if not math.isfinite(v):
                diverged = True
                break
        if (i + 1) in eval_at:
            torch.cuda.synchronize()
            train_time += time.perf_counter() - t0
            ev = evaluate(model, batches, micro_rows)
            mean = float(np.mean(list(ev.values())))
            curve.append({"step": i + 1, "tokens": (i + 1) * n_tok, "train_s": round(train_time, 2),
                          "mean": round(mean, 4), **{k: round(v, 4) for k, v in ev.items()}})
            print(f"[{a.name}] step {i+1}/{steps} t={train_time:.1f}s train~{np.mean(losses[-10:]):.3f} "
                  f"heldout mean {mean:.4f} conv {ev['conversations']:.3f}", flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    rec = {
        "name": a.name, "args": vars(a), "params": shape.param_count(), "steps": steps,
        "diverged": diverged, "train_s": round(train_time, 2),
        "tok_per_s": round(steps * n_tok / train_time) if train_time else None,
        "final": curve[-1] if curve else None, "curve": curve,
    }
    with open(a.results, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if a.save and not diverged:
        meta = CheckpointMeta(tokenizer="tok-v1", shape=shape, lineage_id="exp", version_id=a.name,
                              parent_version_id=None, tokens_trained_total=steps * n_tok,
                              last_mixture=mix, runs=[])
        save_checkpoint(model, meta, Path(a.save))
    fin = curve[-1] if curve else {}
    print(f"RESULT {a.name}: mean {fin.get('mean')} conv {fin.get('conversations')} "
          f"creative {fin.get('creative')} facts {fin.get('facts')} web {fin.get('web')} "
          f"| {rec['tok_per_s']} tok/s {train_time:.1f}s diverged={diverged}", flush=True)


if __name__ == "__main__":
    main()
