import copy
import math

import pytest
import torch

from airace_ml.infer.lm import ContinuationScore, Generation, TorchLM, build_chat_prompt
from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.tokenizer import encode_chat


def _naive_greedy(lm, ids, n):
    out = list(ids)
    for _ in range(n):
        logits = lm.model(torch.tensor([out]))[0, -1]
        out.append(int(logits.argmax()))
    return out[len(ids):]

def test_greedy_matches_naive_and_batching(tiny_lm, tiny_tok):
    p1, p2 = tiny_tok.encode("Once upon a time"), tiny_tok.encode("The cat")
    g = tiny_lm.generate([p1, p2], max_new_tokens=8, temperature=0, top_p=1, seed=0, stop_ids=[])
    assert g[0].tokens == _naive_greedy(tiny_lm, p1, 8) and g[1].tokens == _naive_greedy(tiny_lm, p2, 8)

def test_score_continuations_matches_manual(tiny_lm, tiny_tok):
    ctx, cont = tiny_tok.encode("The dog"), tiny_tok.encode(" runs fast")
    s = tiny_lm.score_continuations([ctx], [cont])[0]
    with torch.no_grad():
        logp = torch.log_softmax(tiny_lm.model(torch.tensor([ctx + cont]))[0], -1)
    manual = sum(float(logp[len(ctx) - 1 + i, t]) for i, t in enumerate(cont))
    assert s.n_tokens == len(cont) and math.isclose(s.sum_logprob, manual, abs_tol=1e-4)

def test_stops_at_end_token(tiny_lm, tiny_tok, monkeypatch):
    real = tiny_lm.model.forward
    def forced(*a, **k):
        out = real(*a, **k); out[..., tiny_tok.end_id] = 1e4; return out
    monkeypatch.setattr(tiny_lm.model, "forward", forced)
    g = tiny_lm.generate([tiny_tok.encode("hi")], max_new_tokens=5, temperature=0, top_p=1, seed=0)[0]
    assert g.stopped and g.tokens == []

def test_long_history_truncated(tiny_lm, tiny_tok):  # Review Focus 2
    hist = [("user" if i % 2 == 0 else "ai", f"message number {i} " * 5) for i in range(40)] + [("user", "last question?")]
    ids = build_chat_prompt(tiny_tok, hist, ctx_len=64, reserve=16)
    assert len(ids) <= 48 and ids[0] == tiny_tok.bos_id and ids[-1] == tiny_tok.ai_id
    assert tiny_tok.encode("last question?")[-1] in ids
    assert isinstance(tiny_lm.chat_reply(hist, max_new_tokens=16), str)

def test_seeded_sampling_and_probs(tiny_lm, tiny_tok):
    p = [tiny_tok.encode("Once")]
    a = tiny_lm.generate(p, max_new_tokens=10, temperature=1.0, top_p=0.95, seed=3, stop_ids=[])[0]
    b = tiny_lm.generate(p, max_new_tokens=10, temperature=1.0, top_p=0.95, seed=3, stop_ids=[])[0]
    assert a.tokens == b.tokens and len(a.token_probs) == len(a.tokens)
    assert all(0 < q <= 1 for q in a.token_probs + a.top1_probs)


# --- Beyond the brief: padding, caps, stops, chunking, scoring and prompt edge cases ---


def _gen(lm, prompts, **kw):
    kw = {"max_new_tokens": 8, "temperature": 0, "top_p": 1, "seed": 0, "stop_ids": [], **kw}
    return lm.generate(prompts, **kw)


def test_ragged_batch_greedy_matches_naive_for_every_row(tiny_lm, tiny_tok):
    prompts = [
        [tiny_tok.bos_id, *tiny_tok.encode("Once upon a time there was a small cat")],
        tiny_tok.encode("Hi"),
        [tiny_tok.bos_id, *tiny_tok.encode("The dog ran")],
    ]
    out = _gen(tiny_lm, prompts, max_new_tokens=10)
    assert [g.tokens for g in out] == [_naive_greedy(tiny_lm, p, 10) for p in prompts]


@pytest.fixture
def sharp_lm(tiny_tok):
    """Like ``tiny_lm`` but with large weights, so attention is peaked.

    An untrained model's attention is near-uniform, which hides padding and position bugs: every
    row's output barely depends on its context. Here it depends on every key and every position.
    """
    torch.manual_seed(1)
    model = Transformer(ModelShape(2, 64, 64), tiny_tok.vocab_size)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.mul_(20)
    return TorchLM(model, tiny_tok, torch.device("cpu"))


def _naive_greedy_with_top1(lm, ids, n):
    out, top1 = list(ids), []
    with torch.no_grad():
        for _ in range(n):
            probs = torch.softmax(lm.model(torch.tensor([out]))[0, -1], -1)
            out.append(int(probs.argmax()))
            top1.append(float(probs.max()))
    return out[len(ids):], top1


def test_ragged_batch_matches_naive_when_attention_is_sharp(sharp_lm, tiny_tok):
    # Different lengths force left padding; the padded slots must stay invisible and the
    # per-row positions must stay consistent between prefill and every decode step.
    prompts = [
        [tiny_tok.bos_id, *tiny_tok.encode("Once upon a time there was a small cat")],
        tiny_tok.encode("Hi"),
        [tiny_tok.bos_id, *tiny_tok.encode("The dog ran")],
    ]
    out = _gen(sharp_lm, prompts, max_new_tokens=10)
    for p, g in zip(prompts, out, strict=True):
        tokens, top1 = _naive_greedy_with_top1(sharp_lm, p, 10)
        assert g.tokens == tokens
        assert g.top1_probs == pytest.approx(top1, abs=1e-4)
    assert len({tuple(g.tokens) for g in out}) == len(out)  # the rows really differ


def test_batch_size_chunking_does_not_change_greedy_output(tiny_lm, tiny_tok):
    prompts = [tiny_tok.encode(t) for t in ("Once", "The cat sat", "A dog", "Hello there friend")]
    whole = _gen(tiny_lm, prompts)
    tiny_lm.batch_size = 1
    one_by_one = _gen(tiny_lm, prompts)
    tiny_lm.batch_size = 3
    chunked = _gen(tiny_lm, prompts)
    assert [g.tokens for g in whole] == [g.tokens for g in one_by_one] == [g.tokens for g in chunked]
    assert len(chunked) == len(prompts)


def test_stop_token_is_excluded_and_other_rows_continue(tiny_lm, tiny_tok):
    p1, p2 = tiny_tok.encode("Once upon a time"), tiny_tok.encode("The cat")
    free = _gen(tiny_lm, [p1, p2], max_new_tokens=12)
    # Stop on the first token of row 0 that row 1 never produces, so only row 0 stops.
    stop = next(t for t in free[0].tokens if t not in free[1].tokens)
    cut = free[0].tokens.index(stop)
    g = _gen(tiny_lm, [p1, p2], max_new_tokens=12, stop_ids=[stop])
    assert g[0].stopped and g[0].tokens == free[0].tokens[:cut]
    assert len(g[0].token_probs) == len(g[0].top1_probs) == cut
    assert not g[1].stopped and g[1].tokens == free[1].tokens


def test_default_stop_ids_are_end_and_bos(tiny_lm, tiny_tok, monkeypatch):
    real = tiny_lm.model.forward

    def forced_bos(*a, **k):
        out = real(*a, **k)
        out[..., tiny_tok.bos_id] = 1e4
        return out

    monkeypatch.setattr(tiny_lm.model, "forward", forced_bos)
    g = tiny_lm.generate([tiny_tok.encode("hi")], max_new_tokens=5, temperature=0, top_p=1, seed=0)
    assert g[0].stopped and g[0].tokens == []
    g = _gen(tiny_lm, [tiny_tok.encode("hi")], max_new_tokens=3)  # stop_ids=[] disables stopping
    assert not g[0].stopped and g[0].tokens == [tiny_tok.bos_id] * 3


def test_length_cap_and_no_position_overflow(tiny_lm, tiny_tok, monkeypatch):  # Review Focus 2
    ctx = tiny_lm.ctx_len
    seen = []
    real = tiny_lm.model.forward

    def spy(idx, **k):
        seen.append(int(k["positions"].max()))
        return real(idx, **k)

    monkeypatch.setattr(tiny_lm.model, "forward", spy)
    long_prompt = [tiny_tok.bos_id, *([tiny_tok.encode("x")[0]] * (ctx + 20))]  # longer than ctx
    near_full = [tiny_tok.bos_id, *([tiny_tok.encode("y")[0]] * (ctx - 3))]  # ctx - 2 tokens
    short = tiny_tok.encode("Hi")
    out = _gen(tiny_lm, [long_prompt, near_full, short], max_new_tokens=30)
    assert len(out[0].tokens) == 1  # truncated to ctx - 1 tokens, one slot left
    assert len(out[1].tokens) == 2  # total length reaches ctx_len
    assert len(out[2].tokens) == 30
    assert max(seen) <= ctx - 1


def test_overlong_prompt_keeps_bos_and_tail(tiny_lm, tiny_tok, monkeypatch):
    fed = []
    real = tiny_lm.model.forward

    def spy(idx, **k):
        fed.append(idx.clone())
        return real(idx, **k)

    monkeypatch.setattr(tiny_lm.model, "forward", spy)
    ctx = tiny_lm.ctx_len
    body = list(range(20, 20 + ctx + 30))
    _gen(tiny_lm, [[tiny_tok.bos_id, *body]], max_new_tokens=1)
    prefill = fed[0][0].tolist()
    assert len(prefill) == ctx - 1
    assert prefill[0] == tiny_tok.bos_id and prefill[1:] == body[-(ctx - 2):]
    fed.clear()
    _gen(tiny_lm, [body], max_new_tokens=1)  # no bos: plain tail
    assert fed[0][0].tolist() == body[-(ctx - 1):]


def test_probabilities_are_the_raw_distribution(tiny_lm, tiny_tok):
    p = [tiny_tok.encode("Once upon")]
    greedy = _gen(tiny_lm, p)[0]
    assert greedy.token_probs == greedy.top1_probs  # greedy picks the most likely token
    sampled = _gen(tiny_lm, p, temperature=2.0, top_p=0.9, seed=5, max_new_tokens=12)[0]
    assert all(t <= m + 1e-6 for t, m in zip(sampled.token_probs, sampled.top1_probs, strict=True))
    # top1 is measured on the raw softmax, independent of temperature and top-p.
    assert sampled.top1_probs[0] == pytest.approx(greedy.top1_probs[0], rel=1e-5)


def test_tiny_top_p_is_greedy_and_seeds_differ(tiny_lm, tiny_tok):
    p = [tiny_tok.encode("Once upon")]
    greedy = _gen(tiny_lm, p, max_new_tokens=10)[0].tokens
    nucleus = _gen(tiny_lm, p, temperature=1.0, top_p=1e-6, seed=9, max_new_tokens=10)[0].tokens
    assert nucleus == greedy
    runs = {
        tuple(_gen(tiny_lm, p, temperature=1.5, seed=s, max_new_tokens=10)[0].tokens)
        for s in range(4)
    }
    assert len(runs) > 1  # an untrained model is near-uniform, so different seeds diverge


def test_max_new_tokens_zero_and_empty_batch(tiny_lm, tiny_tok):
    g = _gen(tiny_lm, [tiny_tok.encode("Hi")], max_new_tokens=0)
    assert g == [Generation(tokens=[], token_probs=[], top1_probs=[], stopped=False)]
    assert _gen(tiny_lm, []) == []


@pytest.mark.parametrize(
    "kw", [{"temperature": -1.0}, {"top_p": 0.0}, {"top_p": 1.5}, {"max_new_tokens": -1}]
)
def test_generate_rejects_bad_arguments(tiny_lm, tiny_tok, kw):
    with pytest.raises(ValueError):
        _gen(tiny_lm, [tiny_tok.encode("Hi")], **kw)


def test_generate_rejects_empty_prompt(tiny_lm):
    with pytest.raises(ValueError, match="empty"):
        _gen(tiny_lm, [[]])


def test_score_ragged_batch_matches_one_by_one(tiny_lm, tiny_tok):
    ctxs = [
        tiny_tok.encode("The dog"),
        tiny_tok.encode("Once upon a time there was"),
        [tiny_tok.bos_id],
    ]
    conts = [tiny_tok.encode(" runs fast"), tiny_tok.encode(" a cat"), tiny_tok.encode("Hello")]
    batched = tiny_lm.score_continuations(ctxs, conts)
    assert all(isinstance(s, ContinuationScore) for s in batched)
    for c, k, s in zip(ctxs, conts, batched, strict=True):
        alone = tiny_lm.score_continuations([c], [k])[0]
        assert s.n_tokens == alone.n_tokens == len(k)
        assert s.sum_logprob == pytest.approx(alone.sum_logprob, abs=1e-4)
        assert s.sum_logprob < 0
    tiny_lm.batch_size = 2
    chunked = tiny_lm.score_continuations(ctxs, conts)
    assert [s.sum_logprob for s in chunked] == pytest.approx(
        [s.sum_logprob for s in batched], abs=1e-4
    )


def test_score_empty_continuation_and_bad_inputs(tiny_lm, tiny_tok):
    ctx = tiny_tok.encode("The dog")
    assert tiny_lm.score_continuations([ctx], [[]]) == [ContinuationScore(0.0, 0)]
    assert tiny_lm.score_continuations([], []) == []
    with pytest.raises(ValueError):
        tiny_lm.score_continuations([ctx], [])
    with pytest.raises(ValueError, match="empty"):
        tiny_lm.score_continuations([[]], [tiny_tok.encode("x")])


def test_score_overlong_inputs_are_left_truncated(tiny_lm, tiny_tok):
    ctx_len = tiny_lm.ctx_len
    cont = tiny_tok.encode(" runs fast")
    long_ctx = [tiny_tok.bos_id, *([tiny_tok.encode("z")[0]] * (ctx_len * 2))]
    kept = ctx_len - len(cont)  # context tokens that fit next to the continuation
    fit = [tiny_tok.bos_id, *long_ctx[-(kept - 1):]]
    got = tiny_lm.score_continuations([long_ctx], [cont])[0]
    want = tiny_lm.score_continuations([fit], [cont])[0]
    assert got.n_tokens == len(cont)
    assert got.sum_logprob == pytest.approx(want.sum_logprob, abs=1e-5)
    # A continuation that cannot fit is cut to what fits (one context token stays) and says so.
    huge = [7] * (ctx_len + 5)
    s = tiny_lm.score_continuations([tiny_tok.encode("a")], [huge])[0]
    assert s.n_tokens == ctx_len - 1 and math.isfinite(s.sum_logprob)


def test_build_chat_prompt_short_history_is_plain_chat_encoding(tiny_tok):
    hist = [("user", "Hello"), ("ai", "Hi there"), ("user", "Why is the sky blue?")]
    assert build_chat_prompt(tiny_tok, hist, ctx_len=256, reserve=32) == encode_chat(
        tiny_tok, hist, add_generation_prompt=True
    )
    assert build_chat_prompt(tiny_tok, [], ctx_len=64, reserve=8) == [
        tiny_tok.bos_id,
        tiny_tok.ai_id,
    ]


def test_build_chat_prompt_drops_oldest_turns_whole(tiny_tok):
    hist = [
        ("user", "first message here"),
        ("ai", "a reply"),
        ("user", "second"),
        ("ai", "ok"),
        ("user", "last"),
    ]
    full = encode_chat(tiny_tok, hist, add_generation_prompt=True)
    # A budget one token short of everything: exactly the oldest turn goes.
    ids = build_chat_prompt(tiny_tok, hist, ctx_len=len(full) + 7, reserve=8)
    assert ids == encode_chat(tiny_tok, hist[1:], add_generation_prompt=True)
    ids = build_chat_prompt(tiny_tok, hist, ctx_len=len(full) + 8, reserve=8)
    assert ids == full


def test_build_chat_prompt_single_huge_turn_keeps_tail_and_structure(tiny_tok):
    text = "alpha beta gamma " * 40 + "the very end?"
    ids = build_chat_prompt(tiny_tok, [("user", "hi"), ("user", text)], ctx_len=64, reserve=16)
    assert len(ids) == 48
    assert ids[:2] == [tiny_tok.bos_id, tiny_tok.user_id]
    assert ids[-2:] == [tiny_tok.end_id, tiny_tok.ai_id]
    assert ids[2:-2] == tiny_tok.encode(text)[-(48 - 4):]


def test_build_chat_prompt_rejects_unusable_budget(tiny_tok):
    with pytest.raises(ValueError):
        build_chat_prompt(tiny_tok, [("user", "hi")], ctx_len=64, reserve=61)
    with pytest.raises(ValueError):
        build_chat_prompt(tiny_tok, [("user", "hi")], ctx_len=64, reserve=-1)


def test_chat_reply_and_complete_are_seeded_text(tiny_lm, tiny_tok):
    hist = [("user", "Tell me a story")]
    a = tiny_lm.chat_reply(hist, max_new_tokens=12, seed=4)
    assert isinstance(a, str) and a == tiny_lm.chat_reply(hist, max_new_tokens=12, seed=4)
    c = tiny_lm.complete("Once upon", max_new_tokens=12, seed=4)
    assert isinstance(c, str) and c == tiny_lm.complete("Once upon", max_new_tokens=12, seed=4)
    g = tiny_lm.generate(
        [encode_chat(tiny_tok, hist, add_generation_prompt=True)],
        max_new_tokens=12, temperature=0.8, top_p=0.95, seed=4,
    )[0]
    assert a == tiny_tok.decode(g.tokens)  # chat_reply is generate() on the chat prompt


def test_chat_reply_survives_default_budget_on_a_tiny_context(tiny_lm):
    # The default max_new_tokens (96) exceeds this model's whole 64-token context.
    assert isinstance(tiny_lm.chat_reply([("user", "hello " * 80)]), str)


def test_lm_exposes_model_tok_device_ctx_len_and_leaves_grad_alone(tiny_lm, tiny_tok):
    assert tiny_lm.model.shape.ctx_len == tiny_lm.ctx_len == 64
    assert tiny_lm.tok is tiny_tok and tiny_lm.device == torch.device("cpu")
    assert not tiny_lm.model.training
    _gen(tiny_lm, [tiny_tok.encode("Hi")], max_new_tokens=2)
    assert torch.is_grad_enabled() and all(p.grad is None for p in tiny_lm.model.parameters())


def test_default_device_follows_pick_device_and_batch_size_is_32(tiny_tok, monkeypatch):
    monkeypatch.setenv("AIRACE_DEVICE", "cpu")
    model = Transformer(ModelShape(2, 64, 64), tiny_tok.vocab_size)
    lm = TorchLM(model, tiny_tok)
    assert lm.device == torch.device("cpu") and lm.batch_size == 32 and lm.model is model


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_bf16_path_tracks_the_cpu_path(sharp_lm, tiny_tok):
    gpu = TorchLM(copy.deepcopy(sharp_lm.model), tiny_tok, torch.device("cuda"))
    prompts = [tiny_tok.encode("Once upon a time"), tiny_tok.encode("The cat")]
    on_cpu = _gen(sharp_lm, prompts, max_new_tokens=10)
    on_gpu = _gen(gpu, prompts, max_new_tokens=10)
    assert [g.tokens for g in on_gpu] == [g.tokens for g in on_cpu]
    sampled = gpu.generate(prompts, max_new_tokens=10, temperature=0.9, top_p=0.9, seed=2)
    assert sampled == gpu.generate(prompts, max_new_tokens=10, temperature=0.9, top_p=0.9, seed=2)
    assert all(0 < q <= 1 for g in sampled for q in g.token_probs + g.top1_probs)
    ctxs = [tiny_tok.encode("The dog"), tiny_tok.encode("Once upon a time there was")]
    conts = [tiny_tok.encode(" runs fast"), tiny_tok.encode(" a cat")]
    want = [s.sum_logprob for s in sharp_lm.score_continuations(ctxs, conts)]
    got = [s.sum_logprob for s in gpu.score_continuations(ctxs, conts)]
    assert got == pytest.approx(want, rel=0.05)  # bf16 autocast, so not bit-identical
