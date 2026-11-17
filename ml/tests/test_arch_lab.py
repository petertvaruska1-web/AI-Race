"""The brain-design lab (docs/handoff/next-level/arch_lab.py).

Its model class: with every change off it is the product's model bit for bit, each change does
what it says and nothing else, the product's optimizer takes every new parameter once, the KV cache
gives the same reply as full passes, and a lab folder is never read as a product checkpoint.

Its screen at equal time: the trainer builds the lab's model only inside the swap and is restored
whatever happens, speeds come from the trainer's progress events, budgets follow speed within
bounds, an unsteady baseline stops the screen, and the screen runs on the tiny fixture, measures
every model through the lab's loader and never reuses a run of one design for another."""

import contextlib
import importlib.util
import io
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from airace_ml.data.identity import data_identity
from airace_ml.experiments import scorecard as sc
from airace_ml.experiments.runs import TRAINING_SOURCES, source_digest
from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import CheckpointMeta, load_checkpoint, save_checkpoint
from airace_ml.model.shape import HEAD_DIM, ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.train import trainer
from airace_ml.train.config import TrainRunConfig, learning_style
from airace_ml.train.events import Done, HeldoutEval, Instability, Progress, Sample

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


# -- the screen at equal time -----------------------------------------------------------------


def test_the_design_swap_is_undone_whatever_happens(lab, tiny_tok):
    cfg = TrainRunConfig("swap", 3, ModelShape(2, 64, 64), 1024, {"web": 1.0}, batch_tokens=1024)
    lab.assert_restored()
    with lab.lab_design(lab.ArchOptions.all()) as built:
        model = trainer._start_model(cfg, tiny_tok, None)
        assert type(model) is lab.LabTransformer and model.options == lab.ArchOptions.all()
        assert built == [model]
        with pytest.raises(lab.LabError), lab.lab_design(lab.ArchOptions()):
            pass  # a swap inside a swap is refused before it changes anything
    lab.assert_restored()
    assert trainer.Transformer is Transformer
    # The base design built by the trainer is the product's model, seeded by the run's seed.
    with lab.lab_design(lab.ArchOptions()):
        ours = trainer._start_model(cfg, tiny_tok, None)
    theirs = trainer._start_model(cfg, tiny_tok, None)
    assert type(theirs) is Transformer
    pairs = zip(ours.state_dict().items(), theirs.state_dict().items(), strict=True)
    for (name, a), (_, b) in pairs:
        assert torch.equal(a, b), name
    with pytest.raises(RuntimeError, match="boom"), lab.lab_design(lab.ArchOptions.all()):
        raise RuntimeError("boom")
    lab.assert_restored()
    trainer.Transformer = lab.LabTransformer  # a trainer left patched is caught
    try:
        with pytest.raises(lab.LabError):
            lab.assert_restored()
    finally:
        trainer.Transformer = Transformer


def _progress(step, total=100, per_step=1000):
    return Progress(step, total, step * per_step, 2.0, 0.01)


def test_speed_is_taken_between_progress_events_after_the_first_steps(lab):
    """Steps 1 to 20 are slow (warming up); the rest train 1,000 tokens in 0.1 s, but an
    evaluation (held-out loss and samples) or a rollback between two progress events adds time
    that is not training: those intervals are left out."""
    events, t = [], 0.0
    for step in (1, *range(10, 101, 10)):
        t += 5.0 if step <= 20 else 1.0
        events.append((t, _progress(step)))
        if step in (30, 60):
            t += 4.0
            events.append((t, HeldoutEval(step, {"web": 3.0})))
            events.append((t, Sample(step, "Once upon a time", "...")))
        if step == 80:
            t += 2.0
            events.append((t, Instability(step, "rollback", 0.5)))
    events.append((t, Done("completed", {})))
    assert lab.progress_speed(events, skip=20) == pytest.approx(10_000.0)
    # Every interval counts when nothing happens between the progress events.
    plain = [(float(step), _progress(step)) for step in range(10, 101, 10)]
    assert lab.progress_speed(plain, skip=10) == pytest.approx(1000.0)
    with pytest.raises(lab.LabError):
        lab.progress_speed(events[:3], skip=20)  # nothing after step 20 to time


def test_budgets_follow_speed_and_stay_in_bounds(lab):
    batch = 64
    base_tokens = 1000 * batch
    same = lab.equal_time_budget(base_tokens, batch, 1000.0, 1000.0)
    assert (same.steps, same.tokens, same.clamped) == (1000, base_tokens, False)
    slower = lab.equal_time_budget(base_tokens, batch, 1000.0, 800.0)
    assert (slower.steps, slower.tokens, slower.clamped) == (800, 800 * batch, False)
    slowest = lab.equal_time_budget(base_tokens, batch, 1000.0, 300.0)
    assert (slowest.steps, slowest.clamped) == (500, True)
    fastest = lab.equal_time_budget(base_tokens, batch, 1000.0, 2000.0)
    assert (fastest.steps, fastest.clamped) == (1500, True)
    edge = lab.equal_time_budget(base_tokens, batch, 1000.0, 1500.0)
    assert (edge.steps, edge.clamped) == (1500, False)
    # The real Tiny: 1,536 steps of 8,192 tokens; 80% is 1,228.8 steps, rounded to 1,229.
    real = lab.equal_time_budget(sc.STEP_TOKENS, 8192, 1000.0, 800.0)
    assert (real.steps, real.tokens) == (1229, 1229 * 8192)
    with pytest.raises(lab.LabError):
        lab.equal_time_budget(base_tokens, batch, 1000.0, 0.0)


def _screen(lab, root, out):
    return lab.Screen(root, root, out, torch.device("cpu"), lab.TINY_SCALE, bench=[])


def test_unsteady_timing_stops_the_screen(lab, tiny_data_root, tmp_path, capsys):
    assert lab.steady_speed(1000.0, 1040.0) == pytest.approx(1020.0)
    assert lab.steady_speed(1000.0, 1050.0) == pytest.approx(1025.0)  # 5% is still steady
    for first, second in ((1000.0, 1060.0), (1060.0, 1000.0)):
        with pytest.raises(lab.LabError, match="6"):
            lab.steady_speed(first, second)

    # The timing phase times the baseline before and after the variants, and stops on an
    # unsteady baseline before it keeps any timing.
    out = tmp_path / "screen"
    screen = _screen(lab, tiny_data_root, out)
    screen.steady_tolerance = lab.STEADY_TOLERANCE  # the real card's rule
    asked = []

    def speed(size, variant, tag):
        asked.append((size, variant, tag))
        return {"warm-up": 1.0, "base": 1000.0, "all": 300.0, "base-again": 1060.0}[tag]

    screen.speed = speed
    with pytest.raises(lab.LabError, match="6"):
        screen.time_size("tiny", ["base", "all"])
    assert [tag for _, _, tag in asked] == ["warm-up", "base", "all", "base-again"]
    assert not (out / lab.TIMING_DIR / "tiny.json").exists()

    # Steady: kept, and the slow variant's budget is clamped to half the steps, with a warning.
    screen.speed = lambda size, variant, tag: {"base": 1000.0, "all": 300.0}.get(tag, 1020.0)
    timing = screen.time_size("tiny", ["base", "all"])
    assert timing["base"] == [1000.0, 1020.0] and timing["speeds"]["base"] == 1010.0
    base_steps = screen.config("tiny", 1, "probe").steps
    assert timing["budgets"]["base"]["steps"] == base_steps
    assert timing["budgets"]["all"] == {
        "steps": base_steps // 2,
        "tokens": base_steps // 2 * 1024,
        "share": pytest.approx(300.0 / 1010.0),
        "clamped": True,
    }
    assert "warning" in capsys.readouterr().out
    assert (out / lab.TIMING_DIR / "tiny.json").is_file()
    screen.speed = None  # a second call reuses the timing and measures nothing
    assert screen.time_size("tiny", ["all"])["budgets"]["all"]["steps"] == base_steps // 2


SCREEN_LINE = re.compile(
    r"^all together: better by ≥ 0\.04 held-out loss on every size: (yes|no); "
    r"by ≥ 3 bench points on every size: yes$",
    re.MULTILINE,
)


@pytest.fixture(scope="module")
def screened(lab, tiny_data_root, tmp_path_factory):
    """One screen of ``base`` and ``all`` on the tiny fixture, every size, one seed, with bench-v1
    replaced by a stand-in that records the model it was given."""
    out = tmp_path_factory.mktemp("arch-lab") / "screen"
    seen = []

    def run_benchmarks(lm, tok, **kwargs):
        seen.append((type(lm.model), lm.model.options))
        return SimpleNamespace(overall=20.0 if lm.model.options.relu2 else 10.0)

    def load_judge(root, device, identity):
        return SimpleNamespace(calibration=None)

    argv = ["--out", str(out), "--seeds", "1", "--variants", "base", "all"]
    argv += ["--bench", "base", "all", "--tiny-root", str(tiny_data_root), "--device", "cpu"]
    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(printed):
        mp.setattr(lab, "run_benchmarks", run_benchmarks)
        mp.setattr(lab, "build_suite", lambda: None)
        mp.setattr(lab, "_load_judge", load_judge)
        mp.setattr(lab, "_load_novelty", lambda root, identity: [])
        code = lab.main(argv)
    return SimpleNamespace(out=out, code=code, seen=seen, printed=printed.getvalue())


def test_the_screen_runs_on_the_tiny_fixture(lab, screened, tiny_data_root):
    assert screened.code == 0, screened.printed
    lab.assert_restored()
    assert trainer.Transformer is Transformer
    assert not hasattr(lab, "load_checkpoint")  # every measure loads through load_lab_model
    results = json.loads((screened.out / "results.json").read_text(encoding="utf-8"))
    assert results["sizes"] == ["tiny", "small", "medium"]
    assert results["variants"] == ["base", "all"]
    for size in results["sizes"]:
        timing = results["timing"][size]
        assert set(timing["speeds"]) == {"base", "all"} and all(s > 0 for s in timing["base"])
        for variant in ("base", "all"):
            entry = results["models"][size][variant]["1"]
            assert entry["status"] == "completed"
            assert entry["tokens"] == timing["budgets"][variant]["tokens"]
            assert entry["steps"] == timing["budgets"][variant]["steps"]
            for measure in ("heldout", "yardstick_loss", "seconds", "bench_overall"):
                assert isinstance(entry[measure], float), (size, variant, measure)
            folder = screened.out / lab.RUNS_DIR / f"{size}-{variant}-s1"
            loaded = lab.load_lab_model(folder)
            assert loaded.options == lab.VARIANTS[variant]
            assert loaded.shape == lab.TINY_SCALE.card.shapes[lab.SIZES[size][0]]
        assert results["models"][size]["base"]["1"]["steps"] == 12
        assert 6 <= results["models"][size]["all"]["1"]["steps"] <= 18
        diff = results["summary"][size]["all"]["bench_overall"]["minus_base"]
        assert diff == pytest.approx(10.0)
    # The timing runs are thrown away; only their speeds are kept.
    kept = sorted(p.name for p in (screened.out / lab.TIMING_DIR).iterdir())
    assert kept == ["medium.json", "small.json", "tiny.json"]
    # Bench-v1 was given the lab's models, of the right design, once per benched run.
    designs = [lab.VARIANTS[v] for _ in range(3) for v in ("base", "all")]
    assert screened.seen == [(lab.LabTransformer, options) for options in designs]
    # The yardstick was taken on the lab's model of the run.
    folder = screened.out / lab.RUNS_DIR / "tiny-all-s1"
    again = sc.yardstick_loss(lab.load_lab_model(folder), tiny_data_root)
    assert results["models"]["tiny"]["all"]["1"]["yardstick_loss"] == pytest.approx(again)
    with pytest.raises(RuntimeError):
        load_checkpoint(folder)  # the product cannot read it, and the lab never tried
    assert results["gate"]["all"]["bench"] is True
    assert SCREEN_LINE.search(screened.printed), screened.printed
    record = json.loads((folder / lab.RECORD_FILE).read_text(encoding="utf-8"))
    key = record["key"]
    assert key["options"] == lab.ArchOptions.all().to_dict()
    assert (key["size"], key["variant"], key["seed"]) == ("tiny", "all", 1)
    assert key["token_budget"] == results["timing"]["tiny"]["budgets"]["all"]["tokens"]
    assert key["data"] == data_identity(tiny_data_root)
    assert key["code"] == source_digest(TRAINING_SOURCES)


def test_a_run_of_one_design_is_never_reused_for_another(lab, screened, tiny_data_root):
    runs = screened.out / lab.RUNS_DIR
    # A finished run of another design where the qk run goes: it must be trained again.
    shutil.copytree(runs / "tiny-all-s1", runs / "tiny-qk-s1")
    argv = ["--out", str(screened.out), "--sizes", "tiny", "--seeds", "1"]
    argv += ["--variants", "base", "qk", "--tiny-root", str(tiny_data_root), "--device", "cpu"]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert lab.main(argv) == 0
    lab.assert_restored()
    said = printed.getvalue()
    assert "tiny-base-s1: finished earlier, reusing it" in said
    assert "tiny-qk-s1: training" in said and "tiny-all-s1" not in said
    assert lab.read_lab_arch(runs / "tiny-qk-s1") == lab.VARIANTS["qk"]
    assert lab.load_lab_model(runs / "tiny-qk-s1").options == lab.VARIANTS["qk"]
    record = json.loads((runs / "tiny-qk-s1" / lab.RECORD_FILE).read_text(encoding="utf-8"))
    assert record["key"]["options"] == lab.VARIANTS["qk"].to_dict()
    results = json.loads((screened.out / "results.json").read_text(encoding="utf-8"))
    assert set(results["models"]["tiny"]) == {"base", "qk"}
    # The first screen's all run is left as it was.
    assert lab.read_lab_arch(runs / "tiny-all-s1") == lab.ArchOptions.all()
