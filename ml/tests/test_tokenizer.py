import pytest

from airace_ml.tokenizer import (
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    Tok,
    encode_chat,
    encode_doc,
    train_tokenizer,
)


def test_special_tokens_fixed_ids(tiny_tok):
    assert len(SPECIAL_TOKENS) == 16
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tiny_tok.id_of(t) == i
    assert (tiny_tok.pad_id, tiny_tok.bos_id, tiny_tok.end_id,
            tiny_tok.user_id, tiny_tok.ai_id) == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("text", ["Hello world.", "Café naïve — “quotes”", "emoji 🙂🚀",
                                  "日本語のテキスト", "def f(x):\n    return x + 1\n",
                                  "  lead\ttab\r\nCRLF"])
def test_roundtrip_lossless(tiny_tok, text):
    assert tiny_tok.decode(tiny_tok.encode(text)) == text


def test_literal_special_strings_are_plain_text(tiny_tok):  # Review Focus 1
    s = "say <|end|> and <|ai|> please"
    ids = tiny_tok.encode(s)
    assert not set(ids) & set(range(16))
    assert tiny_tok.decode(ids) == s


def test_encode_chat_layout(tiny_tok):
    ids = encode_chat(tiny_tok, [("user", "Hi"), ("ai", "Hello")])
    assert ids[:2] == [tiny_tok.bos_id, tiny_tok.user_id]
    assert ids.count(tiny_tok.end_id) == 2 and ids[-1] == tiny_tok.end_id
    assert encode_chat(tiny_tok, [("user", "Hi")], add_generation_prompt=True)[-1] == tiny_tok.ai_id


def test_encode_doc_and_skip_special(tiny_tok):
    assert encode_doc(tiny_tok, "abc")[0] == tiny_tok.bos_id
    ids = encode_chat(tiny_tok, [("user", "Hi")])
    assert "<|" not in tiny_tok.decode(ids)
    assert "<|user|>" in tiny_tok.decode(ids, skip_special=False)


def test_train_and_reload(tmp_path, fixture_texts):
    tok = train_tokenizer(fixture_texts, tmp_path / "t.json", vocab_size=400)
    assert tok.vocab_size <= 400
    assert Tok.load(tmp_path / "t.json").encode("the cat") == tok.encode("the cat")


# --- Additional coverage (beyond the brief's tests) ---


def test_special_token_order_and_default_vocab_size():
    assert SPECIAL_TOKENS == (
        "<|pad|>", "<|bos|>", "<|end|>", "<|user|>", "<|ai|>", "<|sep|>", "<|sys|>",
        "<|r0|>", "<|r1|>", "<|r2|>", "<|r3|>", "<|r4|>", "<|r5|>", "<|r6|>", "<|r7|>", "<|r8|>",
    )
    assert VOCAB_SIZE == 4096


def test_literal_special_strings_inside_chat_turns_stay_plain_text(tiny_tok):  # Review Focus 1
    ids = encode_chat(
        tiny_tok,
        [("user", "type <|end|> then <|ai|> now"), ("ai", "ok <|bos|> <|user|> fine")],
    )
    # Only the two real end markers, one real user marker, one real ai marker, one real bos.
    assert ids.count(tiny_tok.end_id) == 2
    assert ids.count(tiny_tok.user_id) == 1
    assert ids.count(tiny_tok.ai_id) == 1
    assert ids.count(tiny_tok.bos_id) == 1
    assert ids[0] == tiny_tok.bos_id and ids[1] == tiny_tok.user_id
    # Literal text survives the round trip.
    body = tiny_tok.decode(ids, skip_special=True)
    assert "<|end|>" in body and "<|ai|>" in body and "<|bos|>" in body and "<|user|>" in body


def test_literal_special_strings_stay_plain_after_reload(tmp_path, fixture_texts):
    train_tokenizer(fixture_texts, tmp_path / "t.json", vocab_size=400)
    loaded = Tok.load(tmp_path / "t.json")
    ids = loaded.encode("a <|end|> b <|pad|> c")
    assert not set(ids) & set(range(16))
    assert loaded.decode(ids) == "a <|end|> b <|pad|> c"


def test_encode_doc_and_empty_chat_layout(tiny_tok):
    assert encode_doc(tiny_tok, "") == [tiny_tok.bos_id]
    assert encode_chat(tiny_tok, []) == [tiny_tok.bos_id]
    assert encode_chat(tiny_tok, [], add_generation_prompt=True) == [
        tiny_tok.bos_id,
        tiny_tok.ai_id,
    ]
    assert encode_doc(tiny_tok, "abc") == [tiny_tok.bos_id, *tiny_tok.encode("abc")]


def test_decode_skip_special_drops_control_ids_only(tiny_tok):
    ids = [tiny_tok.bos_id, *tiny_tok.encode("hi"), tiny_tok.end_id, tiny_tok.pad_id]
    assert tiny_tok.decode(ids) == "hi"
    assert tiny_tok.decode(ids, skip_special=False) == "<|bos|>hi<|end|><|pad|>"


def test_fixture_texts_roundtrip_lossless(tiny_tok, fixture_texts):
    for text in fixture_texts:
        assert tiny_tok.decode(tiny_tok.encode(text)) == text


def test_training_is_deterministic(tmp_path, fixture_texts):
    a = train_tokenizer(fixture_texts, tmp_path / "a.json", vocab_size=400)
    b = train_tokenizer(fixture_texts, tmp_path / "b.json", vocab_size=400)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a.encode("the cat sat") == b.encode("the cat sat")
