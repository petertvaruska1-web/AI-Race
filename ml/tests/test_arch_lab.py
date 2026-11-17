"""The brain-design lab (docs/handoff/next-level/arch_lab.py), its model class: with every change
off it is the product's model bit for bit, each change does what it says and nothing else, the
product's optimizer takes every new parameter once, the KV cache gives the same reply as full
passes, and a lab folder is never read as a product checkpoint."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import CheckpointMeta, load_checkpoint, save_checkpoint
from airace_ml.model.shape import HEAD_DIM, ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.train import trainer
from airace_ml.train.config import learning_style

LAB_PATH = Path(__file__).resolve().parents[2] / "docs" / "handoff" / "next-level" / "arch_lab.py"
TINY = ModelShape(4, 128, 256)  # the scorecard's Tiny
SMALL = ModelShape(6, 192, 256)
MEDIUM = ModelShape(8, 256, 256)
VOCAB = 512
OUTPUT_PROJECTIONS = ("attn.wo.weight", "mlp.w_down.weight")


@pytest.fixture(scope="module")
def lab():
    spec = importlib.util.spec_from_file_location("arch_lab", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arch_lab"] = module  # dataclasses look their module up here
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("arch_lab", None)


def _lab_model(lab, options, shape=TINY, seed=0):
    torch.manual_seed(seed)
    return lab.LabTransformer(shape, VOCAB, options).eval()


def _product_model(shape=TINY, seed=0):
    torch.manual_seed(seed)
    return Transformer(shape, VOCAB).eval()


def _tokens(batch=3, length=40, seed=1):
    return torch.randint(0, VOCAB, (batch, length), generator=torch.Generator().manual_seed(seed))


def _meta(shape):
    return CheckpointMeta(
        tokenizer="tok-v1",
        shape=shape,
        lineage_id="L",
        version_id="v1",
        parent_version_id=None,
        tokens_trained_total=0,
        last_mixture={"web": 1.0},
        runs=[],
    )


def _same_as_base_except(model, base, changed=()):
    """Every parameter the model shares with the base model of the same seed has the base's
    starting value, except those whose names end with one of ``changed``."""
    base_params = dict(base.named_parameters())
    for name, p in model.named_parameters():
        if name in base_params and not name.endswith(changed):
            assert torch.equal(p, base_params[name]), name


def _first_block_input(model, x):
    """The hidden state the first block receives for the tokens ``x``."""
    seen = []
    hook = model.blocks[0].register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    try:
        model(x)
    finally:
        hook.remove()
    return seen[0]


def _cached_logits(model, x, prefill):
    """Logits of ``x`` from a prefill of ``prefill`` tokens and then one token at a time."""
    cache = model.new_kv_cache()
    parts = [model(x[:, :prefill], kv_cache=cache)]
    for t in range(prefill, x.shape[1]):
        parts.append(model(x[:, t : t + 1], kv_cache=cache))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def test_base_options_give_the_product_model(lab):
    product = _product_model()
    base = _lab_model(lab, lab.ArchOptions())
    ours, theirs = dict(base.named_parameters()), dict(product.named_parameters())
    assert list(ours) == list(theirs)
    for name, p in theirs.items():
        assert ours[name].shape == p.shape and torch.equal(ours[name], p), name
    assert list(base.state_dict()) == list(product.state_dict())
    assert base.lm_head.weight is base.tok_emb.weight
    assert base.num_params() == product.num_params() == TINY.param_count(VOCAB)
    x = _tokens()
    assert torch.equal(base(x), product(x))
    # The other two ways the game runs a model: explicit positions with left padding (a batch of
    # prompts of several lengths) and the KV cache.
    mask = torch.ones(x.shape, dtype=torch.bool)
    mask[1, :7] = False
    pos = (mask.long().cumsum(dim=1) - 1).clamp(min=0)
    padded = {"positions": pos, "key_padding_mask": mask}
    assert torch.equal(base(x, **padded), product(x, **padded))
    assert torch.equal(_cached_logits(base, x, 25), _cached_logits(product, x, 25))


def test_the_variants_and_their_names(lab):
    options = lab.ArchOptions
    assert list(lab.VARIANTS) == ["base", "zero", "qk", "emb", "relu2", "rope", "all"]
    assert lab.VARIANTS["base"] == options()
    assert lab.VARIANTS["zero"] == options(zero_out=True)
    assert lab.VARIANTS["qk"] == options(qk_norm=True)
    assert lab.VARIANTS["emb"] == options(emb_norm=True)
    assert lab.VARIANTS["relu2"] == options(relu2=True)
    assert lab.VARIANTS["rope"] == options(rope_base=500.0)
    assert lab.VARIANTS["all"] == options.all() == options(True, True, True, True, 500.0)
    for name, variant in lab.VARIANTS.items():
        assert variant.name == name
        assert options.from_dict(variant.to_dict()) == variant
    assert options(zero_out=True, qk_norm=True).name == "zero+qk"
    for bad in ({"zero_out": True}, {**options().to_dict(), "extra": 1}):
        with pytest.raises(lab.LabError):
            options.from_dict(bad)
    with pytest.raises(lab.LabError):
        options(rope_base=0.0)


@torch.no_grad()
def test_each_change_does_what_it_says(lab):
    options = lab.ArchOptions
    base = _lab_model(lab, options())
    x = _tokens()

    zero = _lab_model(lab, options(zero_out=True))
    for block in zero.blocks:
        assert not block.attn.wo.weight.any() and not block.mlp.w_down.weight.any()
    _same_as_base_except(zero, base, OUTPUT_PROJECTIONS)

    qk = _lab_model(lab, options(qk_norm=True))
    _same_as_base_except(qk, base)
    for block in qk.blocks:
        for norm in (block.attn.q_norm, block.attn.k_norm):
            assert torch.equal(norm.weight, torch.ones(HEAD_DIM))  # shape [HEAD_DIM], at 1
    before, plain_before = qk(x), base(x)
    for model in (qk, base):
        model.blocks[1].attn.wq.weight.mul_(3)
        model.blocks[1].attn.wk.weight.mul_(3)
    torch.testing.assert_close(qk(x), before, atol=1e-5, rtol=0)
    # Without the change the same scaling sharpens attention and moves the logits.
    assert (base(x) - plain_before).abs().max() > 1e-3
    base = _lab_model(lab, options())

    emb = _lab_model(lab, options(emb_norm=True))
    _same_as_base_except(emb, base)
    assert emb.emb_norm.weight.shape == (TINY.d_model,)
    assert torch.equal(emb.emb_norm.weight, torch.ones(TINY.d_model))
    rms = _first_block_input(emb, x).pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=0)
    bare = _first_block_input(base, x).pow(2).mean(dim=-1).sqrt()
    assert bare.max() < 0.1  # without the change the first block gets the small bare embedding

    relu2 = _lab_model(lab, options(relu2=True))
    assert [lab.relu2_hidden(s.ffn_hidden) for s in (TINY, SMALL, MEDIUM)] == [576, 768, 1056]
    h2, d = lab.relu2_hidden(TINY.ffn_hidden), TINY.d_model
    assert not any("w_gate" in name for name, _ in relu2.named_parameters())
    for block in relu2.blocks:
        assert not hasattr(block.mlp, "w_gate")
        assert block.mlp.w_up.weight.shape == (h2, d) and block.mlp.w_down.weight.shape == (d, h2)
    assert abs(relu2.num_params() - base.num_params()) <= 0.01 * base.num_params()
    mlp, h = relu2.blocks[0].mlp, torch.randn(2, 5, d)
    expected = (torch.relu(h @ mlp.w_up.weight.T) ** 2) @ mlp.w_down.weight.T
    torch.testing.assert_close(mlp(h), expected)

    rope = _lab_model(lab, options(rope_base=500.0))
    _same_as_base_except(rope, base)
    cos, _ = rope._default_rope(0, 2, torch.device("cpu"))
    half = 500.0 ** (-2 * torch.arange(HEAD_DIM // 2, dtype=torch.float64) / HEAD_DIM)
    want = torch.cos(1 * torch.cat([half, half])).float()
    torch.testing.assert_close(cos[0, 0, 1], want, atol=1e-6, rtol=0)
    # The base reaches every path: explicit positions give the default pass, and the model is a
    # different function from the base model with the very same weights.
    pos = torch.arange(x.shape[1]).expand(x.shape)
    torch.testing.assert_close(rope(x, positions=pos), rope(x), atol=1e-6, rtol=0)
    assert (rope(x) - base(x)).abs().max() > 1e-3
    assert (rope(x, positions=pos) - base(x, positions=pos)).abs().max() > 1e-3


def test_the_optimizer_takes_every_new_parameter_once(lab):
    model = _lab_model(lab, lab.ArchOptions.all()).train()
    optimizer = trainer._make_optimizer(model, learning_style(0.5))
    groups = optimizer.param_groups
    (muon,) = [g["params"] for g in groups if g["kind"] == "muon"]
    (embedding,) = [g["params"] for g in groups if g["kind"] == "adam" and g["weight_decay"] > 0]
    (gains,) = [g["params"] for g in groups if g["kind"] == "adam" and g["weight_decay"] == 0]
    named = dict(model.named_parameters())

    def ids(params):
        return {id(p) for p in params}

    matrices = {id(p) for name, p in named.items() if p.dim() == 2 and name != "tok_emb.weight"}
    assert ids(muon) == matrices and len(muon) == len(matrices)
    assert [id(p) for p in embedding] == [id(model.tok_emb.weight)]
    vectors = {id(p) for p in named.values() if p.dim() == 1}
    assert ids(gains) == vectors and len(gains) == len(vectors)
    new = [model.emb_norm.weight]
    new += [w for b in model.blocks for w in (b.attn.q_norm.weight, b.attn.k_norm.weight)]
    assert ids(new) <= ids(gains)
    every = [id(p) for p in muon + embedding + gains]
    assert len(every) == len(set(every)) and set(every) == ids(model.parameters())
    # One real step takes every group without complaint.
    for g in groups:
        g["lr"] = g["peak_lr"]
    model(_tokens()).logsumexp(dim=-1).mean().backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(b.attn.wo.weight.any() for b in model.blocks)  # zero at the start, learning now


@torch.no_grad()
def test_the_cache_gives_the_same_reply(lab, tiny_tok):
    torch.manual_seed(0)
    model = lab.LabTransformer(ModelShape(2, 64, 64), tiny_tok.vocab_size, lab.ArchOptions.all())
    # Zero output projections make every block a no-op at the start, which would leave the cache
    # nothing to get wrong, so they get random values. Then, as the product's inference tests do,
    # every matrix is made 20 times larger and the query and key gains 2 to 4: otherwise the
    # tied head makes a random model repeat its last token, and attention is near uniform, which
    # hides position and padding mistakes.
    for block in model.blocks:
        block.attn.wo.weight.normal_(0.0, 0.02)
        block.mlp.w_down.weight.normal_(0.0, 0.02)
        block.attn.q_norm.weight.uniform_(2.0, 4.0)
        block.attn.k_norm.weight.uniform_(2.0, 4.0)
    model.emb_norm.weight.uniform_(0.5, 1.5)
    for p in model.parameters():
        if p.dim() > 1:
            p.mul_(20)
    model.eval()

    x = torch.randint(0, tiny_tok.vocab_size, (2, 30))
    torch.testing.assert_close(_cached_logits(model, x, 12), model(x), atol=1e-4, rtol=1e-5)

    lm = TorchLM(model, tiny_tok, torch.device("cpu"))
    prompts = [tiny_tok.encode("Once upon a time"), tiny_tok.encode("The cat sat on the mat and")]
    assert len(prompts[0]) != len(prompts[1])  # a batch of both is left-padded

    def by_full_passes(ids):
        out = list(ids)
        for _ in range(20):
            out.append(int(model(torch.tensor([out]))[0, -1].argmax()))
        return out[len(ids) :]

    greedy = {"max_new_tokens": 20, "temperature": 0, "top_p": 1, "seed": 0, "stop_ids": []}
    expected = [by_full_passes(p) for p in prompts]
    assert all(len(set(reply)) > 1 for reply in expected)  # replies that say something
    assert [g.tokens for g in lm.generate(prompts[:1], **greedy)] == expected[:1]
    assert [g.tokens for g in lm.generate(prompts, **greedy)] == expected


def test_a_lab_folder_is_never_a_product_checkpoint(lab, tmp_path):
    options = lab.ArchOptions.all()
    model = _lab_model(lab, options)
    with torch.no_grad():
        for block in model.blocks:  # so that the saved model is not the trivial one
            block.attn.wo.weight.normal_(0.0, 0.02)
            block.mlp.w_down.weight.normal_(0.0, 0.02)
    folder = tmp_path / "lab-run"
    save_checkpoint(model, _meta(model.shape), folder)
    lab.save_lab_arch(folder, options)

    loaded = lab.load_lab_model(folder, "cpu").eval()
    assert isinstance(loaded, lab.LabTransformer) and loaded.options == options
    x = _tokens()
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))
    with pytest.raises(RuntimeError):
        load_checkpoint(folder)

    lab.save_lab_arch(folder, lab.ArchOptions())  # the file names another design than the weights
    with pytest.raises(RuntimeError):
        lab.load_lab_model(folder, "cpu")
    (folder / lab.LAB_ARCH_NAME).unlink()
    with pytest.raises(lab.LabError):
        lab.load_lab_model(folder, "cpu")
    product = tmp_path / "product"
    save_checkpoint(_product_model(), _meta(TINY), product)
    with pytest.raises(lab.LabError):
        lab.load_lab_model(product, "cpu")
