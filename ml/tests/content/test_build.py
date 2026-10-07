import ast
import difflib
import json
import re
import shutil
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import airace_content
import airace_content.build as build_module
import airace_content.cli as cli_module
from airace_content.assemble import (
    RECIPES,
    SCALES,
    Collected,
    Component,
    NoiseRates,
    _letter_runs,
    apply_noise,
    assemble_dataset,
    content_text,
    estimate_tokens,
    source_of_copies,
    split_count,
    validate_recipe,
)
from airace_content.build import (
    STRUCTURED_DATASETS,
    BuildError,
    BuildSummary,
    _simple_vocab,
    build_corpus,
    heldout_mask,
    purchase_ranks,
    stream,
    tokenizer_texts,
)
from airace_content.cli import main
from airace_content.noise import _PAGE_FOOTERS, _PAGE_HEADERS, inject_noise
from airace_content.sources import (
    FORMAT_CHECK_ROWS,
    PASSAGE_MIN_WORDS,
    SOURCES,
    SourceFormatError,
    chunk_passages,
    fixture_fetch,
    hf_fetch,
    is_childrens_book,
    row_to_content,
)
from airace_content.textproc import MOJIBAKE_CHARS, SPAM_MARKERS, build_vocab
from airace_ml.data.corpus import DATASET_IDS, NOISE_KINDS, TOPICS, Corpus
from airace_ml.data.prep import CLEANING_THRESHOLDS
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.skills.facts import FalseFactPlan, plan_false_facts
from airace_ml.skills.kb import load_kb
from airace_ml.tokenizer import Tok

FIX = Path(__file__).parents[1] / "fixtures" / "sources"


def test_adapters():
    assert row_to_content(
        "soda", {"dialogue": ["Hi", "Hey", "Bye"], "speakers": ["A", "B", "A"]}
    ) == [[("user", "Hi"), ("ai", "Hey"), ("user", "Bye")]]
    g = row_to_content("gsm8k", {"question": "Q?", "answer": "2+2 = <<2+2=4>>4\n#### 4"})[0]
    assert "<<" not in g and g.endswith("The answer is 4.")
    assert row_to_content("fineweb_edu", {"text": "x", "int_score": 2}) == []
    assert "\t" not in row_to_content("mbpp", {"text": "t", "code": "def f():\r\n\treturn 1"})[0]


def test_tiny_build_end_to_end(tmp_path):
    s = build_corpus("tiny", tmp_path, fixture_fetch(FIX), seed=0)
    assert s.tokenizer_trained and tokenizer_path(tmp_path).exists()
    for ds in DATASET_IDS:
        c = Corpus.open(corpus_dir(tmp_path) / ds)
        assert c.n_docs > 0 and len(c.tags.quality) == c.n_docs and c.tags.heldout.sum() >= 1
    man = json.loads((corpus_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert all(src["license"] for src in man["sources"])
    assert json.loads((corpus_dir(tmp_path) / "false_facts.json").read_text(encoding="utf-8"))
    assert Corpus.open(corpus_dir(tmp_path) / "web").tags.noise_kind.any()


def test_tokenizer_is_frozen_on_rebuild(tmp_path):
    build_corpus("tiny", tmp_path, fixture_fetch(FIX), seed=0)
    before = tokenizer_path(tmp_path).read_bytes()
    s2 = build_corpus("tiny", tmp_path, fixture_fetch(FIX), seed=0, datasets=["web"])
    assert not s2.tokenizer_trained and tokenizer_path(tmp_path).read_bytes() == before


# --- shared helpers ----------------------------------------------------------------------------

FETCH = fixture_fetch(FIX)
_NONE = NOISE_KINDS.index("none")
_KIND = {name: i for i, name in enumerate(NOISE_KINDS)}
CORPUS_FILES = ("tokens.bin", "offsets.npy", "info.json")
SHARED_FILES = ("manifest.json", "false_facts.json", "simple_vocab.txt", "known_vocab.txt")


@dataclass
class TinyBuild:
    root: Path
    summary: BuildSummary


@pytest.fixture(scope="module")
def tiny_build(tmp_path_factory) -> TinyBuild:
    """One tiny build from the fixtures (seed 0), shared by the tests that only read it."""
    root = tmp_path_factory.mktemp("tiny_build")
    return TinyBuild(root, build_corpus("tiny", root, FETCH, seed=0))


@pytest.fixture(scope="module")
def build_inputs():
    """What ``assemble_dataset`` needs, as a build of seed 0 makes it."""
    kb = load_kb()
    return kb, _simple_vocab("tiny", FETCH), plan_false_facts(kb, np.random.default_rng(0), 300)


def _copy_build(build: TinyBuild, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(build.root, root)
    return root


def _texts(root: Path, ds: str) -> tuple[Corpus, list[str]]:
    tok = Tok.load(tokenizer_path(root))
    corpus = Corpus.open(corpus_dir(root) / ds)
    return corpus, [tok.decode(corpus.doc(i)) for i in range(corpus.n_docs)]


def _assemble(ds, build_inputs, fetch=FETCH, scale="tiny", seed=0):
    kb, simple_vocab, plan = build_inputs
    stats: dict = {}
    rng = np.random.default_rng(seed)
    docs = assemble_dataset(ds, scale, fetch, kb, simple_vocab, plan, rng, stats=stats)
    return docs, {c["name"]: c for c in stats["components"]}


# --- sources and adapters ----------------------------------------------------------------------


def test_sources_are_the_designed_ones():
    table = {s.id: (s.hf_path, s.hf_config, s.split) for s in SOURCES.values()}
    assert table == {
        "tinystories": ("roneneldan/TinyStories", None, "train"),
        "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "train"),
        "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
        "cosmo_khan": ("HuggingFaceTB/cosmopedia", "khanacademy", "train"),
        "cosmo_wikihow": ("HuggingFaceTB/cosmopedia", "wikihow", "train"),
        "cosmo_openstax": ("HuggingFaceTB/cosmopedia", "openstax", "train"),
        "cosmo_stories": ("HuggingFaceTB/cosmopedia", "stories", "train"),
        "soda": ("allenai/soda", None, "train"),
        "everyday_conv": ("HuggingFaceTB/everyday-conversations-llama3.1-2k", None, "train_sft"),
        "gutenberg": ("sedthh/gutenberg_english", None, "train"),
        "gsm8k": ("openai/gsm8k", "main", "train"),
        "mbpp": ("google-research-datasets/mbpp", "full", "train"),
        "simplewiki": ("wikimedia/wikipedia", "20231101.simple", "train"),
    }
    licenses = {s.id: s.license for s in SOURCES.values()}
    assert licenses["tinystories"] == "cdla-sharing-1.0" and licenses["gutenberg"] == "mit"
    assert licenses["fineweb"] == licenses["fineweb_edu"] == "odc-by"
    assert licenses["soda"] == licenses["mbpp"] == "cc-by-4.0"
    assert licenses["simplewiki"] == "cc-by-sa-3.0" and licenses["gsm8k"] == "mit"
    assert all(
        s.homepage == f"https://huggingface.co/datasets/{s.hf_path}" for s in SOURCES.values()
    )


def test_hf_fetch_imports_datasets_only_when_it_runs(monkeypatch):
    calls = []

    def load_dataset(path, config, *, split, streaming):
        calls.append((path, config, split, streaming))
        return iter([{"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"}])

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)  # installed or not, this is what loads
    rows = hf_fetch(SOURCES["gsm8k"], 2)
    assert calls == []  # a generator: nothing runs until it is iterated
    assert list(rows) == [{"question": "Q1"}, {"question": "Q2"}]
    assert calls == [("openai/gsm8k", "main", "train", True)]


def _module_level_imports(tree: ast.Module):
    """Modules a file imports outside function bodies (so on import)."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""
        stack.extend(ast.iter_child_nodes(node))


def test_no_content_module_imports_datasets_on_import():
    package = Path(airace_content.__file__).parent
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set(_module_level_imports(tree))
        assert not {m for m in imported if m.split(".")[0] == "datasets"}, path.name


def test_fixtures_exercise_every_adapter_rule():
    rows = {sid: list(FETCH(spec, 100)) for sid, spec in SOURCES.items()}
    assert all(5 <= len(r) <= 20 for r in rows.values())
    books = rows["gutenberg"]
    assert any(not is_childrens_book(r["METADATA"]) for r in books)
    assert any("*** START OF" in r["TEXT"] for r in books)
    assert any("*** END OF" in r["TEXT"] for r in books)
    assert any(r["int_score"] < 3 for r in rows["fineweb_edu"])
    assert any(len(r["dialogue"]) < 2 for r in rows["soda"])
    assert any(m["role"] == "system" for r in rows["everyday_conv"] for m in r["messages"])
    assert any("<<" in r["answer"] and "####" in r["answer"] for r in rows["gsm8k"])
    assert any("\t" in r["code"] for r in rows["mbpp"])
    assert any("\r\n" in r["code"] for r in rows["mbpp"])
    assert any("\nRelated pages" in r["text"] for r in rows["simplewiki"])


def test_gutenberg_keeps_childrens_books_without_boilerplate():
    rows = list(FETCH(SOURCES["gutenberg"], 100))
    passages = [row_to_content("gutenberg", row) for row in rows]
    for row, kept in zip(rows, passages):
        assert bool(kept) == is_childrens_book(row["METADATA"])
        for p in kept:
            assert PASSAGE_MIN_WORDS <= len(p.split()) <= 400
            for marker in (
                "Project Gutenberg",
                "***",
                "Produced by",
                "SMALL PRINT",
                "gutenberg.org",
            ):
                assert marker not in p
    fox, _, animals, songs, economics, ships = passages
    assert len(fox) == 2 and fox[0].startswith("THE FOX AND THE LITTLE HEN")
    assert fox[-1].endswith("THE END")
    assert "\n" not in fox[0].split("\n\n")[1]  # hard-wrapped prose lines are joined
    assert "The cow says moo,\nThe sheep says baa," in animals[-1]  # verse keeps its lines
    assert songs[0].startswith("LITTLE SONGS")  # the "Produced by" credit is gone
    assert economics == ships == []


def test_childrens_subjects():
    assert is_childrens_book(json.dumps({"subjects": "Fairy tales -- Juvenile fiction"}))
    assert is_childrens_book(json.dumps({"subjects": ["Animals", "Fables"]}))
    assert is_childrens_book({"subjects": "Children's poetry"})
    assert not is_childrens_book(json.dumps({"subjects": "Economics; Trade"}))
    assert not is_childrens_book("not json")
    assert not is_childrens_book(None)


def test_chunk_passages_cuts_long_paragraphs():
    sentence = "The little dog ran to the park and played with a ball all day long."
    paragraph = " ".join([sentence] * 70)  # about 1,000 words
    text = paragraph + "\n\nA short ending paragraph with a few more words in it."
    passages = chunk_passages(text)
    assert all(len(p.split()) <= 400 for p in passages) and len(passages) == 3
    assert " ".join(passages).split() == text.split()
    run_on = " ".join(["word"] * 900)  # one sentence longer than a passage
    assert [len(p.split()) for p in chunk_passages(run_on)] == [400, 400, 100]
    assert chunk_passages("THE END") == []  # under PASSAGE_MIN_WORDS


def test_chat_adapters():
    three = {"dialogue": ["Hi", "Hello", "Hey all", "Bye"], "speakers": ["A", "B", "C", "A"]}
    assert row_to_content("soda", three) == [
        [("user", "Hi"), ("ai", "Hello"), ("ai", "Hey all"), ("user", "Bye")]
    ]
    assert row_to_content("soda", {"dialogue": ["Alone"], "speakers": ["A"]}) == []
    assert row_to_content("soda", {"dialogue": ["Hi", "  "], "speakers": ["A", "B"]}) == []
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert row_to_content("everyday_conv", {"messages": messages}) == [
        [("user", "Hi"), ("ai", "Hello!")]
    ]
    assert row_to_content("everyday_conv", {"messages": messages[:2]}) == []


def test_text_adapters():
    gsm = {"question": "Q?", "answer": "A is 1 + 2 = <<1+2=3>>3.\nB is 3 * 2 = <<3*2=6>>6.\n#### 6"}
    assert row_to_content("gsm8k", gsm) == [
        "Question: Q?\nAnswer: A is 1 + 2 = 3.\nB is 3 * 2 = 6. The answer is 6."
    ]
    assert row_to_content("gsm8k", {"question": "Q?", "answer": "no final answer"}) == []
    assert row_to_content("gsm8k", {"question": "Q?", "answer": "#### 5."}) == [
        "Question: Q?\nAnswer: The answer is 5."
    ]
    code = {"text": "Add one.", "code": "def f(x):\r\n\tif x:\r\n\t\treturn x + 1\r\n"}
    assert row_to_content("mbpp", code) == [
        "# Add one.\ndef f(x):\n    if x:\n        return x + 1"
    ]
    lead = " ".join(["Cats are small animals."] * 40)  # 160 words: cut at a sentence end
    [wiki] = row_to_content("simplewiki", {"title": "Cat", "text": f"{lead}\nMore text."})
    title, body = wiki.split("\n", 1)
    assert title == "Cat" and body.endswith("animals.") and len(body.split()) <= 150
    short = {"title": "Dog", "text": "Dogs bark.\nDogs run.\n" + " ".join(["word"] * 200)}
    assert row_to_content("simplewiki", short) == ["Dog\nDogs bark.\nDogs run."]
    # the lead section only: it ends at the first heading (a short line without a sentence end)
    stub = (
        "Bucy is a commune. It is in the Aisne department in northern France.\n\n"
        "Related pages \n Communes of the Aisne department\n\nReferences \n\n"
        "Other websites \n Official website"
    )
    assert row_to_content("simplewiki", {"title": "Bucy", "text": stub}) == [
        "Bucy\nBucy is a commune. It is in the Aisne department in northern France."
    ]
    wiki = {row["title"]: row for row in FETCH(SOURCES["simplewiki"], 100)}
    cat = (
        "Cat\nThe cat is a small animal that many people keep as a pet. Cats have soft fur, "
        "sharp claws and a long tail.\nCats like to sleep a lot. They can see well at night. "
        "A young cat is called a kitten."
    )
    assert row_to_content("simplewiki", wiki["Cat"]) == [cat]  # stops before "History"
    marville = (
        "Marville\nMarville is a small town in the north of the country. About two thousand "
        "people live there. It has a school, a market and an old stone bridge."
    )
    assert row_to_content("simplewiki", wiki["Marville"]) == [marville]
    quoted = {"title": "Ann", "text": 'Ann said "hello."\nShe left.\nLife\nMore.'}
    assert row_to_content("simplewiki", quoted) == ['Ann\nAnn said "hello."\nShe left.']
    assert row_to_content("simplewiki", {"title": "X", "text": "History\nIt began."}) == []
    assert row_to_content("fineweb_edu", {"text": "Kept.", "int_score": 3}) == ["Kept."]
    assert row_to_content("tinystories", {"text": "  "}) == []
    with pytest.raises(ValueError, match="unknown source"):
        row_to_content("nope", {})


# --- recipes and assembly ----------------------------------------------------------------------


def test_recipes_match_the_design():
    shares = {
        ds: [(c.kind, c.name, c.share, c.simplicity_min) for c in recipe.components]
        for ds, recipe in RECIPES.items()
    }
    src, gen = "source", "generator"
    assert shares == {
        "web": [(src, "fineweb", 1.0, 0.35)],
        "books": [(src, "gutenberg", 1.0, 0.30)],
        "educational": [
            (src, "cosmo_khan", 0.25, None),
            (src, "cosmo_wikihow", 0.25, None),
            (src, "cosmo_openstax", 0.20, None),
            (src, "fineweb_edu", 0.30, 0.35),
        ],
        "conversations": [
            (src, "soda", 0.55, None),
            (src, "everyday_conv", 0.15, None),
            (gen, "fact_chat", 0.15, None),
            (gen, "instructions", 0.15, None),
        ],
        "code": [(gen, "code", 0.80, None), (src, "mbpp", 0.20, None)],
        "reasoning": [
            (gen, "reasoning", 0.60, None),
            (gen, "patterns", 0.30, None),
            (src, "gsm8k", 0.10, None),
        ],
        "facts": [(gen, "fact_prose", 0.60, None), (src, "simplewiki", 0.40, 0.30)],
        "creative": [(src, "tinystories", 0.85, None), (src, "cosmo_stories", 0.15, None)],
    }
    noise = {ds: recipe.noise for ds, recipe in RECIPES.items()}
    assert noise == {
        "web": NoiseRates(
            typo=0.15, spam=0.06, boilerplate=0.10, garbled=0.05, duplicate=0.08, false_fact=0.06
        ),
        "books": NoiseRates(typo=0.02, garbled=0.01, duplicate=0.02),
        "educational": NoiseRates(duplicate=0.02, false_fact=0.01),
        "conversations": NoiseRates(typo=0.03),
        "code": NoiseRates(duplicate=0.03),
        "reasoning": NoiseRates(),
        "facts": NoiseRates(duplicate=0.02),
        "creative": NoiseRates(typo=0.02, duplicate=0.03),
    }
    assert SCALES["tiny"] == {"docs_per_dataset": 150}
    assert SCALES["full"]["target_tokens"] == {
        "web": 30e6, "books": 15e6, "educational": 20e6, "conversations": 20e6,
        "code": 10e6, "reasoning": 10e6, "facts": 12e6, "creative": 20e6,
    }  # fmt: skip
    for recipe in RECIPES.values():
        validate_recipe(recipe)


def test_validate_recipe_refuses_bad_recipes():
    code = RECIPES["code"]
    with pytest.raises(ValueError, match="simplicity"):
        validate_recipe(
            type(code)(
                "code", [type(code.components[0])("generator", "code", 1.0, 0.3)], code.noise
            )
        )
    with pytest.raises(ValueError, match="add up"):
        validate_recipe(type(code)("code", code.components[:1], code.noise))
    twice = [Component("generator", "code", 0.5), Component("generator", "code", 0.5)]
    with pytest.raises(ValueError, match="twice"):
        validate_recipe(type(code)("code", twice, code.noise))
    with pytest.raises(ValueError, match="unknown generator"):
        validate_recipe(
            type(code)("code", [type(code.components[0])("generator", "poems", 1.0)], code.noise)
        )


def test_split_count():
    assert split_count(150, [0.8, 0.2]) == [120, 30]
    assert split_count(150, [0.55, 0.15, 0.15, 0.15]) == [83, 23, 22, 22]
    assert split_count(150, [0.25, 0.25, 0.20, 0.30]) == [38, 37, 30, 45]
    assert sum(split_count(7, [1 / 3] * 3)) == 7


def test_tiny_scale_cycles_rows_to_the_quota(build_inputs):
    docs, comps = _assemble("books", build_inputs)
    gutenberg = comps["gutenberg"]
    assert gutenberg["docs"] == 150 and gutenberg["passes"] > 1  # 5 passages, cycled
    assert len({content_text(d.content) for d in docs if d.noise_kind == _NONE}) == 5


def test_cycling_stops_when_a_pass_accepts_nothing(build_inputs):
    calls: Counter = Counter()

    def fetch(spec, max_rows):
        calls[spec.id] += 1
        if spec.id == "fineweb_edu":  # every row is filtered out
            yield from [{"text": "A fine school text.", "int_score": 1}] * 5
        elif spec.id == "cosmo_openstax":  # an empty source
            yield from ()
        else:
            yield from FETCH(spec, max_rows)

    _, comps = _assemble("educational", build_inputs, fetch=fetch)
    assert comps["fineweb_edu"]["docs"] == 0 and calls["fineweb_edu"] == 1
    assert comps["cosmo_openstax"]["docs"] == 0 and calls["cosmo_openstax"] == 1
    assert comps["fineweb_edu"]["exhausted"] and comps["cosmo_openstax"]["exhausted"]
    # their 45 + 30 planned documents go to the others, in proportion to their shares
    khan, wikihow = comps["cosmo_khan"], comps["cosmo_wikihow"]
    assert (khan["planned_docs"], wikihow["planned_docs"]) == (38, 37)
    assert (khan["assigned_docs"], wikihow["assigned_docs"]) == (38 + 38, 37 + 37)
    assert (khan["docs"], wikihow["docs"]) == (76, 74)
    assert not khan["exhausted"] and not wikihow["exhausted"]


def test_full_scale_collects_to_the_token_target(build_inputs, monkeypatch):
    monkeypatch.setitem(SCALES["full"]["target_tokens"], "creative", 60_000)
    stories = [row["text"] for row in FETCH(SOURCES["tinystories"], 100)]

    def fetch(spec, max_rows):
        if spec.id == "tinystories":  # an endless supply of different stories
            for i in range(max_rows):
                yield {"text": f"Day {i}. {stories[i % len(stories)]}"}
        else:
            yield from FETCH(spec, max_rows)

    docs, comps = _assemble("creative", build_inputs, fetch=fetch, scale="full")
    stories, cosmo = comps["tinystories"], comps["cosmo_stories"]
    n_stories = stories["docs"]
    assert n_stories > 200  # past the estimate from the first 200 documents
    # cosmo_stories has 6 rows: full scale stops when a source is exhausted, never cycles,
    # and the tokens it could not give go to tinystories
    assert cosmo["docs"] == 6 and cosmo["passes"] == 1 and cosmo["exhausted"]
    given = sum(_tokens(d) for d in docs[n_stories : n_stories + 6])
    assert stories["planned_tokens"] == pytest.approx(51_000)
    assert stories["assigned_tokens"] == pytest.approx(51_000 + 9_000 - given, rel=0.01)
    tokens = sum(_tokens(d) for d in docs[:n_stories])
    assert 0.95 * stories["assigned_tokens"] <= tokens <= 1.05 * stories["assigned_tokens"]
    assert 0.95 * 60_000 <= tokens + given <= 1.05 * 60_000  # the dataset meets its target


def test_full_scale_generators_reach_their_targets(build_inputs, monkeypatch):
    monkeypatch.setitem(SCALES["full"]["target_tokens"], "reasoning", 40_000)
    docs, comps = _assemble("reasoning", build_inputs, scale="full")
    gsm8k = comps["gsm8k"]
    assert gsm8k["docs"] == 8 and gsm8k["exhausted"]  # 8 fixture rows, under its 4,000 tokens
    n_puzzles = comps["reasoning"]["docs"] + comps["patterns"]["docs"]
    short = 4_000 - sum(_tokens(d) for d in docs[n_puzzles:])
    start, total = 0, 0
    for name, share in (("reasoning", 0.6), ("patterns", 0.3)):
        comp = comps[name]
        assert not comp["exhausted"] and comp["planned_tokens"] == pytest.approx(share * 40_000)
        assert comp["assigned_tokens"] == pytest.approx(share * 40_000 + short * share / 0.9)
        tokens = sum(_tokens(d) for d in docs[start : start + comp["docs"]])
        assert 0.8 * comp["assigned_tokens"] <= tokens <= 1.2 * comp["assigned_tokens"], name
        start, total = start + comp["docs"], total + tokens
    assert 0.9 * 40_000 <= total + 4_000 - short <= 1.1 * 40_000


def _tokens(doc) -> int:
    return 1 + estimate_tokens(content_text(doc.content))


def test_full_scale_gives_an_exhausted_sources_quota_to_the_others(build_inputs, monkeypatch):
    monkeypatch.setitem(SCALES["full"]["target_tokens"], "code", 20_000)
    docs, comps = _assemble("code", build_inputs, scale="full")
    code, mbpp = comps["code"], comps["mbpp"]
    assert mbpp["exhausted"] and mbpp["docs"] == 8 and not code["exhausted"]
    assert code["assigned_tokens"] > code["planned_tokens"] == pytest.approx(16_000)
    total = sum(_tokens(d) for d in docs[: code["docs"] + mbpp["docs"]])
    assert 0.95 * 20_000 <= total <= 1.05 * 20_000


def test_reassignment_follows_the_shares_of_the_remaining_components(build_inputs, monkeypatch):
    monkeypatch.setitem(SCALES["full"]["target_tokens"], "educational", 30_000)

    def fetch(spec, max_rows):
        if spec.id == "cosmo_openstax":  # runs dry at once
            return
        rows = list(FETCH(spec, 100))
        for i in range(max_rows):  # the others never run dry
            row = dict(rows[i % len(rows)])
            row["text"] = f"Item {i}. {row['text']}"
            yield row

    _, comps = _assemble("educational", build_inputs, fetch=fetch, scale="full")
    assert comps["cosmo_openstax"]["exhausted"] and comps["cosmo_openstax"]["docs"] == 0
    for name, share in (("cosmo_khan", 0.25), ("cosmo_wikihow", 0.25), ("fineweb_edu", 0.30)):
        comp = comps[name]
        assert not comp["exhausted"]
        assert comp["assigned_tokens"] == pytest.approx(30_000 * share + 6_000 * share / 0.8)


def test_generator_topics_and_copies(build_inputs):
    docs, comps = _assemble("code", build_inputs)
    n_gen, n_mbpp = comps["code"]["docs"], comps["mbpp"]["docs"]
    assert (n_gen, n_mbpp) == (120, 30)
    technology = TOPICS.index("technology")
    assert all(d.topic == technology for d in docs[:n_gen])
    assert all(d.topic is None for d in docs[n_gen : n_gen + n_mbpp])
    assert all(d.component == "code" for d in docs[:n_gen])
    assert all(d.component == "mbpp" for d in docs[n_gen : n_gen + n_mbpp])
    copies = docs[n_gen + n_mbpp :]
    assert copies and all(d.noise_kind == _KIND["duplicate"] for d in copies)
    sources: dict[bytes, object] = {}
    for doc in docs[: n_gen + n_mbpp]:
        if doc.noise_kind == _NONE:
            sources.setdefault(doc.origin, doc)
    for copy in copies:
        source = sources[copy.origin]
        assert copy.topic == source.topic and copy.component == source.component
        assert difflib.SequenceMatcher(None, source.content, copy.content).ratio() > 0.95
    reasoning, comps = _assemble("reasoning", build_inputs)
    n_puzzles = comps["reasoning"]["docs"] + comps["patterns"]["docs"]
    assert all(d.topic == TOPICS.index("school") for d in reasoning[:n_puzzles])
    assert all(d.topic is None for d in reasoning[n_puzzles:])  # gsm8k: tagged by its text


def test_source_of_copies_finds_each_copy():
    rng = np.random.default_rng(0)
    vocabulary = ["the", "a", "cat", "dog", "sun", "tree", "red", "blue", "big", "small", "sad"]
    texts = [f"Note {i}: " + " ".join(rng.choice(vocabulary, 12)) + "." for i in range(200)]
    noisy = inject_noise(texts, NoiseRates(duplicate=0.3), np.random.default_rng(1))
    copies = [nd.text for nd in noisy[200:]]
    found = source_of_copies(texts, copies, list(range(200)))
    assert len(copies) > 50 and None not in found
    for copy, source in zip(copies, found):
        assert re.search(r"\d+", copy).group() == str(source)  # the number survives the typos
        assert sum(a != b for a, b in zip(_letter_runs(copy), _letter_runs(texts[source]))) <= 2
    assert source_of_copies(texts, ["Note 7: something else entirely."], list(range(200))) == [None]


def test_chats_get_typo_noise_only(build_inputs):
    kb, _, plan = build_inputs
    chat = [("user", "Hello there, how are you today?"), ("ai", "I am fine, thank you very much.")]
    docs = [Collected(list(chat)) for _ in range(40)]
    out = apply_noise(docs, NoiseRates(typo=0.25), np.random.default_rng(0))
    typos = [d for d in out if d.noise_kind == _KIND["typo"]]
    assert len(out) == 40 and len(typos) == 10
    assert all([role for role, _ in d.content] == ["user", "ai"] for d in out)
    assert all(text != old for d in typos for (_, text), (_, old) in zip(d.content, chat))
    assert all(d.content == chat for d in out if d.noise_kind == _NONE)
    for kind in ("spam", "boilerplate", "garbled", "duplicate", "false_fact"):
        with pytest.raises(ValueError, match="chat"):
            apply_noise(docs, NoiseRates(**{kind: 0.1}), np.random.default_rng(0), kb, plan)


def test_noise_keeps_each_documents_origin(build_inputs):
    kb, _, plan = build_inputs
    texts = [
        f"Note {i}: the cat sat on the red mat and the dog ran to the big tree." for i in range(100)
    ]
    clean = apply_noise([Collected(t) for t in texts], NoiseRates(), np.random.default_rng(0))
    assert len({d.origin for d in clean}) == 100
    rates = NoiseRates(
        typo=0.2, spam=0.1, boilerplate=0.1, garbled=0.1, false_fact=0.1, duplicate=0.2
    )
    noisy = apply_noise([Collected(t) for t in texts], rates, np.random.default_rng(1), kb, plan)
    assert [d.origin for d in noisy[:100]] == [d.origin for d in clean]  # noise keeps the origin
    assert {d.noise_kind for d in noisy[:100]} >= {_KIND[k] for k in ("typo", "spam", "garbled")}
    copies = noisy[100:]
    assert copies and all(d.origin in {c.origin for c in clean} for d in copies)
    for copy in copies:  # a copy shares the origin of the text it was copied from
        source = texts[[d.origin for d in clean].index(copy.origin)]
        assert re.search(r"\d+", copy.content).group() == re.search(r"\d+", source).group()


def test_a_recipe_with_text_noise_on_chats_is_refused(build_inputs, monkeypatch):
    recipe = RECIPES["conversations"]
    monkeypatch.setattr(recipe, "noise", NoiseRates(typo=0.03, spam=0.05))
    with pytest.raises(ValueError, match="'conversations' has chat documents"):
        _assemble("conversations", build_inputs)


def test_a_source_in_an_unexpected_format_fails_fast(build_inputs):
    read: Counter = Counter()

    def fetch(spec, max_rows):
        for _ in range(max_rows):
            read[spec.id] += 1
            yield {"TEXT": "Once upon a time.", "METADATA": "not json"}

    with pytest.raises(SourceFormatError, match="gutenberg"):
        _assemble("books", build_inputs, fetch=fetch, scale="full")
    assert read["gutenberg"] == FORMAT_CHECK_ROWS

    def few(spec, max_rows):  # fewer rows than the check needs, none of them right
        yield from [{"text": "A row without a score."}] * 3

    with pytest.raises(SourceFormatError, match="fineweb_edu"):
        _assemble("educational", build_inputs, fetch=few)


# --- the pipeline pieces -----------------------------------------------------------------------


def test_heldout_takes_whole_clusters():
    rng = np.random.default_rng(0)
    cluster = np.full(400, -1, np.int32)
    members = rng.permutation(400)[:120].reshape(30, 4)  # 30 clusters of 4
    for c, idx in enumerate(members):
        cluster[idx] = c
    for seed in range(20):
        held = heldout_mask(cluster, np.random.default_rng(seed))
        assert 8 <= held.sum() <= 8 + 3
        for idx in members:
            assert held[idx].all() or not held[idx].any()


def test_heldout_keeps_documents_of_one_origin_together():
    cluster = np.full(40, -1, np.int32)
    origins = [bytes([i]) for i in range(40)]
    origins[30] = origins[31] = origins[3]  # variants the clustering missed
    origins[32] = origins[7]
    for seed in range(30):
        held = heldout_mask(cluster, np.random.default_rng(seed), origins)
        assert held[3] == held[30] == held[31] and held[7] == held[32]
        assert 8 <= held.sum() <= 10
    cluster = np.array([0, 0, 1, 1] + [-1] * 36, np.int32)
    origins = [bytes([i]) for i in range(40)]
    origins[2] = origins[1]  # one origin across two clusters joins them
    for seed in range(30):
        held = heldout_mask(cluster, np.random.default_rng(seed), origins)
        assert len(set(held[:4].tolist())) == 1


def test_heldout_keeps_the_larger_part_for_training():
    assert not heldout_mask(np.zeros(20, np.int32), np.random.default_rng(0)).any()
    small = heldout_mask(np.full(10, -1, np.int32), np.random.default_rng(0))
    assert small.sum() == 5  # at most half, though the minimum is 8
    assert heldout_mask(np.zeros(0, np.int32), np.random.default_rng(0)).shape == (0,)


def test_purchase_ranks_are_a_permutation():
    for n in (0, 1, 1000):
        rank = purchase_ranks(n, np.random.default_rng(n))
        assert rank.dtype == np.float32 and ((rank >= 0) & (rank < 1)).all()
        assert (np.sort(np.round(rank.astype(np.float64) * n)) == np.arange(n)).all()


def test_tokenizer_texts_skip_heldout_and_share_the_budget():
    texts = {"a": [f"a{i} " * 10 for i in range(50)], "b": [f"b{i} " * 10 for i in range(50)]}
    heldout = {ds: np.arange(50) % 5 == 0 for ds in texts}
    sample = tokenizer_texts(texts, heldout, seed=0, budget_bytes=1000)
    for ds, ds_texts in texts.items():
        mine = [t for t in sample if t.startswith(ds)]
        assert mine and sum(len(t.encode("utf-8")) for t in mine) <= 500
        assert not set(mine) & {t for t, h in zip(ds_texts, heldout[ds]) if h}


# --- the tiny build ----------------------------------------------------------------------------


def test_tags_describe_the_documents(tiny_build):
    plan = FalseFactPlan.from_json(
        (corpus_dir(tiny_build.root) / "false_facts.json").read_text(encoding="utf-8")
    )
    furniture = _PAGE_HEADERS + _PAGE_FOOTERS
    for ds in DATASET_IDS:
        corpus, texts = _texts(tiny_build.root, ds)
        tags = corpus.tags
        clean = {t for t, k in zip(texts, tags.noise_kind) if k == _NONE}
        assert ((tags.quality >= 0) & (tags.quality <= 1)).all()
        assert (tags.false_fact == (tags.noise_kind == _KIND["false_fact"])).all()
        for text, kind in zip(texts, tags.noise_kind):
            name = NOISE_KINDS[kind]
            if name == "typo":
                assert text not in clean
            elif name == "spam":
                assert any(marker in text.lower() for marker in SPAM_MARKERS)
            elif name == "boilerplate":
                assert any(line in text for line in furniture)
            elif name == "garbled":
                assert any(c in MOJIBAKE_CHARS for c in text)
            elif name == "false_fact":
                assert any(s in text and wrong in text for (s, _), wrong in plan.mapping.items())
            elif name == "duplicate":
                assert any(
                    difflib.SequenceMatcher(None, text, other).ratio() > 0.95 for other in clean
                )
        for c in np.unique(tags.dup_cluster[tags.dup_cluster >= 0]):
            assert tags.dup_canonical[tags.dup_cluster == c].sum() == 1
        assert tags.dup_canonical[tags.dup_cluster < 0].all()
    web = Corpus.open(corpus_dir(tiny_build.root) / "web").tags.noise_kind
    assert set(np.unique(web)) == set(range(len(NOISE_KINDS)))  # every kind of noise in web


def test_code_and_reasoning_are_scored_as_structured_text(tiny_build):
    assert STRUCTURED_DATASETS == frozenset({"code", "reasoning"})
    thorough = np.float16(CLEANING_THRESHOLDS["thorough"])  # prep compares at float16
    for ds in sorted(STRUCTURED_DATASETS):
        tags = Corpus.open(corpus_dir(tiny_build.root) / ds).tags
        clean = tags.noise_kind == _NONE
        assert np.mean(tags.quality[clean] >= thorough) >= 0.98, ds


def test_web_quality_still_separates_noise_kinds(tiny_build):
    tags = Corpus.open(corpus_dir(tiny_build.root) / "web").tags
    quality = tags.quality.astype(np.float64)
    clean = quality[tags.noise_kind == _NONE].mean()
    for kind in ("typo", "boilerplate", "garbled"):
        assert quality[tags.noise_kind == _KIND[kind]].mean() < clean - 0.2, kind
    assert (quality[tags.noise_kind == _KIND["spam"]] < CLEANING_THRESHOLDS["light"]).all()


def test_chats_keep_their_structure(tiny_build):
    tok = Tok.load(tokenizer_path(tiny_build.root))
    corpus = Corpus.open(corpus_dir(tiny_build.root) / "conversations")
    assert set(np.unique(corpus.tags.noise_kind)) == {_NONE, _KIND["typo"]}
    roles = {tok.user_id, tok.ai_id}
    for i in range(corpus.n_docs):
        ids = corpus.doc(i).tolist()
        assert ids[0] == tok.bos_id and ids[1] in roles and ids[-1] == tok.end_id
        specials = [t for t in ids[1:] if t < 16]
        assert specials[0::2] == [t for t in specials if t in roles]  # role, end, role, end, ...
        assert specials[1::2] == [tok.end_id] * (len(specials) // 2) and len(specials) >= 4


def test_heldout_in_the_build(tiny_build):
    for ds in DATASET_IDS:
        tags = Corpus.open(corpus_dir(tiny_build.root) / ds).tags
        n = len(tags.heldout)
        assert 8 <= tags.heldout.sum() <= n // 2
        for c in np.unique(tags.dup_cluster[tags.heldout & (tags.dup_cluster >= 0)]):
            assert tags.heldout[tags.dup_cluster == c].all()
        rank = np.round(tags.purchase_rank.astype(np.float64) * n)
        assert (np.sort(rank) == np.arange(n)).all()


def test_no_heldout_text_has_a_variant_in_training(tiny_build):
    kb = load_kb()
    simple_vocab = _simple_vocab("tiny", FETCH)
    plan = plan_false_facts(kb, stream(0, "false_facts"), 300)
    for ds in DATASET_IDS:  # the build's documents again, with their origins
        rng = stream(0, f"assemble/{ds}")
        docs = assemble_dataset(ds, "tiny", FETCH, kb, simple_vocab, plan, rng)
        tags = Corpus.open(corpus_dir(tiny_build.root) / ds).tags
        assert [d.noise_kind for d in docs] == tags.noise_kind.tolist()
        held = {d.origin for d, h in zip(docs, tags.heldout) if h}
        trained = {d.origin for d, h in zip(docs, tags.heldout) if not h}
        assert held and not held & trained, ds


def test_manifest_and_shared_artifacts(tiny_build):
    root = corpus_dir(tiny_build.root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in manifest["sources"]] == list(SOURCES)
    for s in manifest["sources"]:
        assert s["license"] == SOURCES[s["id"]].license and s["homepage"]
    feeds = {s["id"]: s["feeds"] for s in manifest["sources"]}
    assert feeds["fineweb"] == ["web"] and feeds["cosmo_stories"] == ["creative"]
    assert list(manifest["datasets"]) == list(DATASET_IDS)
    assert manifest["datasets"] == tiny_build.summary.datasets
    for ds, entry in manifest["datasets"].items():
        corpus = Corpus.open(root / ds)
        assert (entry["docs"], entry["tokens"]) == (corpus.n_docs, corpus.n_tokens)
        assert entry["heldout_docs"] == corpus.tags.heldout.sum()
        assert sum(entry["noise_counts"].values()) == corpus.n_docs
        assert corpus.info["dataset"] == ds and corpus.info["seed"] == 0
        components = entry["components"]
        assert sum(c["tokens"] for c in components) == corpus.n_tokens
        for comp in components:
            assert not comp["exhausted"] and comp["docs"] == comp["assigned_docs"] > 0
            assert comp["planned_docs"] == comp["assigned_docs"]
    plan = FalseFactPlan.from_json((root / "false_facts.json").read_text(encoding="utf-8"))
    assert len(plan.mapping) == 300
    simple = (root / "simple_vocab.txt").read_text(encoding="utf-8").split("\n")[:-1]
    stories = [row["text"] for row in FETCH(SOURCES["tinystories"], 100)]
    assert simple == sorted(build_vocab(stories, 6000))
    known = (root / "known_vocab.txt").read_text(encoding="utf-8").split("\n")[:-1]
    assert known == sorted(known) and "the" in known and len(known) <= 30_000


def test_same_seed_same_corpus(tiny_build, tmp_path):
    again = build_corpus("tiny", tmp_path / "again", FETCH, seed=0)
    assert again.datasets == tiny_build.summary.datasets
    a, b = tiny_build.root, tmp_path / "again"
    assert tokenizer_path(a).read_bytes() == tokenizer_path(b).read_bytes()
    for name in SHARED_FILES:
        assert (corpus_dir(a) / name).read_bytes() == (corpus_dir(b) / name).read_bytes(), name
    for ds in DATASET_IDS:
        for name in CORPUS_FILES:
            assert (corpus_dir(a) / ds / name).read_bytes() == (
                corpus_dir(b) / ds / name
            ).read_bytes()
        # tags.npz is a zip whose entries carry their write time, so compare the arrays
        with (
            np.load(corpus_dir(a) / ds / "tags.npz") as x,
            np.load(corpus_dir(b) / ds / "tags.npz") as y,
        ):
            assert x.files == y.files and all((x[k] == y[k]).all() for k in x.files)
    other = build_corpus("tiny", tmp_path / "other", FETCH, seed=1)
    assert other.datasets["reasoning"]["tokens"] != again.datasets["reasoning"]["tokens"]


def test_partial_rebuild_reuses_shared_artifacts(tiny_build, tmp_path):
    root = _copy_build(tiny_build, tmp_path)
    cdir = corpus_dir(root)
    (cdir / "known_vocab.txt").write_text("zzz\n", encoding="utf-8")  # nothing in web is known
    before = {name: (cdir / name).read_bytes() for name in SHARED_FILES}
    books = (cdir / "books" / "tokens.bin").read_bytes()
    summary = build_corpus("tiny", root, FETCH, seed=1, datasets=["web"])
    assert list(summary.datasets) == ["web"] and not summary.tokenizer_trained
    assert Corpus.open(cdir / "web").tags.quality.max() == 0  # scored with the reused vocabulary
    for name in ("false_facts.json", "simple_vocab.txt", "known_vocab.txt"):
        assert (cdir / name).read_bytes() == before[name], name
    assert (cdir / "books" / "tokens.bin").read_bytes() == books
    manifest = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads(before["manifest.json"])
    assert list(manifest["datasets"]) == list(DATASET_IDS)
    assert manifest["datasets"]["web"]["seed"] == 1
    assert all(manifest["datasets"][ds] == old["datasets"][ds] for ds in DATASET_IDS if ds != "web")
    assert manifest["sources"] == old["sources"]


def test_partial_rebuild_without_artifacts_matches_a_full_build(tiny_build, tmp_path):
    root = _copy_build(tiny_build, tmp_path)
    cdir = corpus_dir(root)
    for name in ("false_facts.json", "simple_vocab.txt", "known_vocab.txt", "manifest.json"):
        (cdir / name).unlink()
    shutil.rmtree(cdir / "web")
    build_corpus("tiny", root, FETCH, seed=0, datasets=["web"])
    original = corpus_dir(tiny_build.root)
    for name in ("false_facts.json", "simple_vocab.txt", "known_vocab.txt"):
        assert (cdir / name).read_bytes() == (original / name).read_bytes(), name
    for name in CORPUS_FILES:
        assert (cdir / "web" / name).read_bytes() == (original / "web" / name).read_bytes()
    manifest = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["datasets"]) == ["web"] and [s["id"] for s in manifest["sources"]] == [
        "fineweb"
    ]


def test_builds_that_cannot_run(tmp_path):
    with pytest.raises(BuildError, match="all datasets"):  # no tokenizer yet
        build_corpus("tiny", tmp_path, FETCH, datasets=["web"])
    tokenizer_path(tmp_path).parent.mkdir(parents=True)
    tokenizer_path(tmp_path).write_text("{}", encoding="utf-8")
    with pytest.raises(BuildError, match="retraining"):
        build_corpus("tiny", tmp_path, FETCH, datasets=["web"], retrain_tokenizer=True)
    with pytest.raises(BuildError, match="unknown dataset"):
        build_corpus("tiny", tmp_path, FETCH, datasets=["web", "poems"])
    with pytest.raises(BuildError, match="unknown scale"):
        build_corpus("huge", tmp_path, FETCH)
    with pytest.raises(BuildError, match="seed"):
        build_corpus("tiny", tmp_path, FETCH, seed=-1)


def test_a_failed_build_leaves_no_new_tokenizer(tiny_build, tmp_path, monkeypatch):
    write = build_module.write_corpus

    def write_until_books(out_dir, *args):
        if Path(out_dir).name == "books":
            raise OSError("disk full")
        write(out_dir, *args)

    monkeypatch.setattr(build_module, "write_corpus", write_until_books)
    fresh = tmp_path / "fresh"
    with pytest.raises(OSError, match="disk full"):
        build_corpus("tiny", fresh, FETCH, seed=0)
    assert not list(tokenizer_path(fresh).parent.iterdir())
    root = _copy_build(tiny_build, tmp_path)
    before = tokenizer_path(root).read_bytes()
    with pytest.raises(OSError, match="disk full"):
        build_corpus("tiny", root, FETCH, seed=1, retrain_tokenizer=True)
    assert [p.name for p in tokenizer_path(root).parent.iterdir()] == ["tokenizer.json"]
    assert tokenizer_path(root).read_bytes() == before
    monkeypatch.setattr(build_module, "write_corpus", write)
    build_corpus("tiny", root, FETCH, seed=1, retrain_tokenizer=True)
    assert tokenizer_path(root).read_bytes() != before  # so the failed retrain did change nothing


def test_full_scale_build_from_fixtures(tmp_path, monkeypatch):
    for ds in DATASET_IDS:
        monkeypatch.setitem(SCALES["full"]["target_tokens"], ds, 30_000)
    summary = build_corpus("full", tmp_path / "ok", FETCH, seed=0)
    assert summary.tokenizer_trained
    assert Tok.load(tokenizer_path(tmp_path / "ok")).vocab_size == 4096
    assert all(e["target_tokens"] == 30_000 and e["tokens"] > 0 for e in summary.datasets.values())
    # too little text for 4096 entries: refused, and no tokenizer is left to be reused as frozen
    monkeypatch.setattr(build_module, "TOKENIZER_SAMPLE_BYTES", 4_000)
    with pytest.raises(RuntimeError, match="4096"):
        build_corpus("full", tmp_path / "short", FETCH, seed=0)
    assert not list(tokenizer_path(tmp_path / "short").parent.iterdir())


# --- the command line --------------------------------------------------------------------------


def test_cli_rebuilds_a_dataset(tiny_build, tmp_path, capsys):
    root = _copy_build(tiny_build, tmp_path)
    argv = ["build", "--scale", "tiny", "--datasets", "web", "--fixtures", str(FIX)]
    assert main([*argv, "--out", str(root)]) == 0
    out = capsys.readouterr().out
    assert "web" in out and "books" not in out and "frozen" in out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["frobnicate"],
        ["build"],
        ["build", "--scale", "huge"],
        ["build", "--scale", "tiny", "--datasets", "web,poems", "--fixtures", "{fix}"],
        ["build", "--scale", "tiny", "--datasets", ",", "--fixtures", "{fix}"],
        ["build", "--scale", "full", "--fixtures", "{fix}"],
        ["build", "--scale", "tiny", "--fixtures", "{missing}"],
        ["build", "--scale", "tiny", "--colour"],
    ],
)
def test_cli_rejects_bad_arguments(argv, tmp_path, capsys):
    argv = [a.format(fix=FIX, missing=tmp_path / "missing") for a in argv]
    assert main([*argv, "--out", str(tmp_path)] if argv else argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.splitlines()) == 1
    assert not corpus_dir(tmp_path).exists()


def test_cli_reports_a_build_that_cannot_run(tmp_path, capsys):
    argv = ["build", "--scale", "tiny", "--datasets", "web", "--fixtures", str(FIX)]
    assert main([*argv, "--out", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("error: the tokenizer is trained on")


def test_cli_warns_about_exhausted_sources(tiny_build, tmp_path, capsys):
    root = _copy_build(tiny_build, tmp_path)
    fixtures = tmp_path / "fixtures"
    shutil.copytree(FIX, fixtures)
    (fixtures / "cosmo_openstax.jsonl").write_text("", encoding="utf-8")  # runs dry at once
    argv = ["build", "--scale", "tiny", "--datasets", "educational", "--fixtures", str(fixtures)]
    assert main([*argv, "--out", str(root)]) == 0
    warnings = capsys.readouterr().err.splitlines()
    assert len(warnings) == 1
    assert warnings[0].startswith("warning: educational: cosmo_openstax ran out at 0 of 30 ")


def test_cli_reports_a_source_in_an_unexpected_format(tiny_build, tmp_path, capsys):
    root = _copy_build(tiny_build, tmp_path)
    fixtures = tmp_path / "fixtures"
    shutil.copytree(FIX, fixtures)
    (fixtures / "gutenberg.jsonl").write_text('{"TEXT": "x", "METADATA": "?"}\n', encoding="utf-8")
    argv = ["build", "--scale", "tiny", "--datasets", "books", "--fixtures", str(fixtures)]
    assert main([*argv, "--out", str(root)]) == 1
    assert capsys.readouterr().err.startswith("error: source 'gutenberg'")


def test_cli_reports_a_missing_datasets_package(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "datasets", None)  # importing it fails, installed or not
    assert main(["build", "--scale", "tiny", "--out", str(tmp_path)]) == 1
    assert "content extra" in capsys.readouterr().err


def test_cli_lets_other_import_errors_through(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ImportError("a bug, not a missing package")

    monkeypatch.setattr(cli_module, "build_corpus", broken)
    with pytest.raises(ImportError, match="a bug"):
        main(["build", "--scale", "tiny", "--fixtures", str(FIX), "--out", str(tmp_path)])


def test_cli_help(capsys):
    assert main(["build", "--help"]) == 0
    assert "--retrain-tokenizer" in capsys.readouterr().out
