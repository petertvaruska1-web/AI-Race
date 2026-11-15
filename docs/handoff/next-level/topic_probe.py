"""Which form of the on-topic lift separates best? Run from ml/.

Variants: the reply's first K tokens (or all), mean or sum, 8 or 32 contrast questions.
Then the verdicts on the stored small-talk replies of two models, for a few forms.
"""

import json
import math
import sys

import torch

from airace_ml.evals.judge import Judge
from airace_ml.experiments.chatprobes import SMALL_TALK
from airace_ml.experiments.scorecard import best_threshold, fit_pairs
from airace_ml.infer.lm import build_chat_prompt
from airace_ml.paths import data_root, judge_dir

dev = torch.device("cuda")
root = data_root()
judge = Judge.load(judge_dir(root), dev)
tok = judge.tok
ctx_len = judge.lm.ctx_len

pairs = fit_pairs(root, tok)
n = len(pairs)
questions = [h[0][1] for h, _ in pairs]
replies = [r for _, r in pairs]


def spread(k, prompt):
    out = []
    for i in range(k):
        at = i * n // k
        for step in range(n):
            q = questions[(at + step) % n]
            if q.strip().casefold() != prompt.strip().casefold():
                out.append(q)
                break
    return out


def prefix_scores(items, ks):
    """For each (question, reply): {K: (sum_logprob, n)} over the reply's first K tokens."""
    ctxs, conts, meta = [], [], []
    for idx, (q, r) in enumerate(items):
        full = [*tok.encode(r), tok.end_id][: ctx_len - 16]
        for k in ks:
            cont = full if k is None else full[:k]
            ctxs.append(build_chat_prompt(tok, [("user", q)], ctx_len, reserve=len(full)))
            conts.append(cont)
            meta.append((idx, k))
    scores = judge.lm.score_continuations(ctxs, conts)
    out = [dict() for _ in items]
    for (idx, k), s in zip(meta, scores, strict=True):
        out[idx][k] = (s.sum_logprob, s.n_tokens)
    return out


KS = (4, 8, 12, 16, 24, None)


def lifts_all(items, contrast_of):
    """items: (prompt, reply). Returns {(K, agg): [lift...]}."""
    flat, spans = [], []
    for prompt, reply in items:
        others = contrast_of(prompt)
        spans.append(len(others))
        flat += [(q, reply) for q in (prompt, *others)]
    sc = prefix_scores(flat, KS)
    out = {(k, agg): [] for k in KS for agg in ("mean", "sum")}
    at = 0
    for count in spans:
        own, rest = sc[at], sc[at + 1 : at + 1 + count]
        at += 1 + count
        for k in KS:
            s_own, n_tok = own[k]
            if n_tok <= 0 or math.isnan(s_own):
                out[(k, "mean")].append(-math.inf)
                out[(k, "sum")].append(-math.inf)
                continue
            mean_rest = sum(r[k][0] for r in rest) / len(rest)
            out[(k, "sum")].append(s_own - mean_rest)
            out[(k, "mean")].append((s_own - mean_rest) / n_tok)
    return out


real = list(zip(questions, replies, strict=True))
swapped = [
    (questions[i], replies[(i + n // 2) % n]) for i in range(n) if replies[(i + n // 2) % n] != replies[i]
]
near = [(questions[i], replies[(i + 1) % n]) for i in range(n) if replies[(i + 1) % n] != replies[i]]

thresholds = {}
for kc in (8, 32):
    got_real = lifts_all(real, lambda p, kc=kc: spread(kc, p))
    got_swap = lifts_all(swapped, lambda p, kc=kc: spread(kc, p))
    got_near = lifts_all(near, lambda p, kc=kc: spread(kc, p))
    for key in got_real:
        b = best_threshold(got_real[key], got_swap[key])
        t = b["threshold"]
        near_below = sum(v <= t for v in got_near[key]) / len(got_near[key])
        thresholds[(kc, *key)] = t
        print(
            f"contrasts {kc:2d} K {str(key[0]):>4} {key[1]:4}: thr {t:+.3f} bal.acc {b['balanced_accuracy']:.3f} "
            f"real above {b['positives_above']:.3f} swapped below {b['negatives_at_or_below']:.3f} "
            f"neighbour-swapped below {near_below:.3f}"
        )
sys.stdout.flush()

card = json.load(open("runs/scorecard-engine2/scorecard.json", encoding="utf-8"))
SHOW = [(8, None, "mean"), (8, 12, "sum"), (32, 12, "sum"), (32, 12, "mean"), (32, 8, "sum")]
for name in ("first-s1", "large"):
    talk = card["models"][name]["examples"]["small_talk"]
    items = [(e["prompt"], e["reply"]) for e in talk]
    res = {}
    for kc in (8, 32):
        got = lifts_all(items, lambda p, kc=kc: spread(kc, p))
        for key, vals in got.items():
            res[(kc, *key)] = vals
    sib = lifts_all(items, lambda p: [q for q in SMALL_TALK if q != p])
    print(f"== {name}: share on topic by form")
    for form in SHOW:
        t = thresholds[form]
        print(f"   {form}: {sum(v > t for v in res[form]) / len(items):.2f}")
    for form in ((12, "sum"), (None, "mean"), (12, "mean")):
        print(f"   siblings {form}: share with lift > 0: {sum(v > 0 for v in sib[form]) / len(items):.2f}")
    for i, (p, r) in enumerate(items[:60] if name == "first-s1" else []):
        marks = "".join("T" if res[f][i] > thresholds[f] else "-" for f in SHOW)
        s = sib[(12, "sum")][i]
        print(f"  {marks} sib12sum {s:+6.1f} {p!r} -> {r[:70]!r}")
