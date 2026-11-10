import numpy as np
import pytest

from airace_ml.data.casual import CASUAL_SHARE, CasualTyping, casual_doc, casual_text
from airace_ml.data.corpus import Corpus
from airace_ml.data.prep import PrepConfig, eligible_docs
from airace_ml.data.sampler import DocPool, MixtureSampler, load_pools
from airace_ml.paths import corpus_dir
from airace_ml.tokenizer import encode_chat, encode_doc

Q = "What is the capital of France?"


@pytest.mark.parametrize(
    ("style", "typed"),
    [
        (0.0, "What is the capital of france?"),
        (0.39, "What is the capital of france?"),
        (0.40, "What is the capital of france"),
        (0.69, "What is the capital of france"),
        (0.70, "What is the capital of France"),
        (0.84, "What is the capital of France"),
        (0.85, "what is the capital of france?"),
        (0.92, "what is the capital of france?"),
        (0.925, "what is the capital of france"),
        (0.999, "what is the capital of france"),
    ],
)
def test_casual_text_types_a_turn_one_of_five_ways(style, typed):
    assert casual_text(Q, style) == typed


@pytest.mark.parametrize("style", [0.0, 0.5, 0.75, 0.9, 0.99])
def test_casual_text_never_empties_a_turn(style):
    assert casual_text("?", style) == "?"
    assert casual_text("...", style) == "..."
    assert casual_text("", style) == ""
    assert casual_text("Really?!", style).rstrip("?!").lower() == "really"
    assert casual_text("I am 5.", style).lower().startswith("i am 5")


def _turns(tok, ids):
    """[(role id, text)] of a chat document's turns."""
    out, i = [], 0
    while i < len(ids):
        if ids[i] in (tok.user_id, tok.ai_id):
            end = i + 1
            while end < len(ids) and ids[end] >= 16:
                end += 1
            out.append((ids[i], tok.decode(ids[i + 1 : end])))
            i = end
        else:
            i += 1
    return out


def test_casual_doc_retypes_user_turns_and_leaves_replies_alone(tiny_tok):
    chat = [
        ("user", "Hello there! How are you?"),
        ("ai", "I am fine, thank you. And You?"),
        ("user", "Tell me about Paris."),
        ("ai", "Paris is the capital of France."),
    ]
    doc = np.asarray(encode_chat(tiny_tok, chat), dtype=np.uint16)
    for style in (0.1, 0.5, 0.75, 0.9, 0.99):
        typed = casual_doc(doc, tiny_tok, style)
        assert typed.dtype == doc.dtype
        got = _turns(tiny_tok, typed.tolist())
        assert [role for role, _ in got] == [
            tiny_tok.user_id,
            tiny_tok.ai_id,
            tiny_tok.user_id,
            tiny_tok.ai_id,
        ]
        assert got[0][1] == casual_text(chat[0][1], style)
        assert got[2][1] == casual_text(chat[2][1], style)
        assert got[1][1] == chat[1][1] and got[3][1] == chat[3][1]  # the AI's words: untouched
        # the framing tokens are where they were: bos first, an end after every turn
        specials = [t for t in typed.tolist() if t < 16]
        assert specials == [t for t in doc.tolist() if t < 16]


def test_a_document_without_user_turns_is_handed_back_as_it_is(tiny_tok):
    doc = np.asarray(encode_doc(tiny_tok, "Once upon a time. The End."), dtype=np.uint16)
    assert casual_doc(doc, tiny_tok, 0.5) is doc


def test_casual_typing_draws_once_per_document_and_keeps_its_share(tiny_tok):
    chat = np.asarray(encode_chat(tiny_tok, [("user", Q), ("ai", "Paris.")]), dtype=np.uint16)
    story = np.asarray(encode_doc(tiny_tok, "Once upon a time."), dtype=np.uint16)
    typing = CasualTyping(tiny_tok, 0.25)
    rng = np.random.default_rng(7)
    changed = sum(typing(chat, rng) is not chat for _ in range(4000))
    assert 0.21 < changed / 4000 < 0.29
    # one draw per document, with or without user turns: the stream is the same either way
    a, b, plain = (np.random.default_rng(3) for _ in range(3))
    typing(chat, a), typing(story, b), plain.random()
    assert a.random() == b.random() == plain.random()
    never, always = CasualTyping(tiny_tok, 0.0), CasualTyping(tiny_tok, 1.0)
    assert all(never(chat, rng) is chat for _ in range(50))
    assert all(always(chat, rng) is not chat for _ in range(50))
    assert always(story, rng) is story
    assert 0.0 < CASUAL_SHARE < 1.0
    with pytest.raises(ValueError):
        CasualTyping(tiny_tok, 1.5)


def test_a_pool_hands_out_documents_through_its_transform(tiny_data_root, tiny_tok):
    corpus = Corpus.open(corpus_dir(tiny_data_root) / "conversations")
    ids = eligible_docs(corpus, PrepConfig())
    seen = []

    def mark(doc, rng):
        seen.append(len(doc))
        return doc[:1]

    pool = DocPool.from_corpus(corpus, ids, balanced=False, transform=mark)
    rng = np.random.default_rng(0)
    assert [len(pool.sample_doc(rng)) for _ in range(5)] == [1] * 5 and len(seen) == 5
    plain = DocPool.from_corpus(corpus, ids, balanced=False)
    assert len(plain.sample_doc(rng)) > 1


def test_load_pools_types_base_datasets_casually_but_never_the_players_own_text(
    tiny_data_root, tiny_tok
):
    lesson = encode_chat(tiny_tok, [("user", "Who Are You?"), ("ai", "I am Zula.")])
    weights = {"conversations": 1.0, "coaching": 1.0}
    custom = {"coaching": [lesson], "notebook": []}
    always = CasualTyping(tiny_tok, 1.0)
    pools = load_pools(tiny_data_root, weights, PrepConfig(), {}, custom, transform=always)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert pools["coaching"].sample_doc(rng).tolist() == lesson  # as the player wrote it
    plain = load_pools(tiny_data_root, weights, PrepConfig(), {}, custom)
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    typed = [pools["conversations"].sample_doc(a) for _ in range(20)]
    b_docs = []
    for _ in range(20):
        b_docs.append(plain["conversations"].sample_doc(b))
        b.random()  # the transform's one draw
    assert any(t.tolist() != p.tolist() for t, p in zip(typed, b_docs, strict=True))
    for t, p in zip(typed, b_docs, strict=True):  # the same documents, the replies word for word
        ai = [text for role, text in _turns(tiny_tok, t.tolist()) if role == tiny_tok.ai_id]
        assert ai == [text for role, text in _turns(tiny_tok, p.tolist()) if role == tiny_tok.ai_id]


def test_rows_are_still_full_and_resumable_with_casual_typing(tiny_data_root, tiny_tok):
    weights = {"conversations": 1.0, "creative": 1.0}
    typing = CasualTyping(tiny_tok, 0.5)

    def sampler(seed):
        pools = load_pools(tiny_data_root, weights, PrepConfig(), {}, {}, transform=typing)
        return MixtureSampler(pools, weights, 64, seed)

    s = sampler(1)
    x, y = s.next_batch(8)
    assert x.shape == y.shape == (8, 64) and (x[:, 1:] == y[:, :-1]).all()
    state = s.rng_state()
    want = s.next_batch(8)
    again = sampler(1)
    again.set_rng_state(state)
    got = again.next_batch(8)
    assert (got[0] == want[0]).all() and (got[1] == want[1]).all()
