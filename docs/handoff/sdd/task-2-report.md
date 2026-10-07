# Task 2 report: Tokenizer

**Status:** DONE
**Commit:** `195e43f feat(ml): byte-level BPE tokenizer with frozen special tokens and chat encoding`
**Branch:** `m1-it-learns`

## What was implemented

- `ml/src/airace_ml/tokenizer.py`
  - `SPECIAL_TOKENS` (16, exact order `<|pad|> <|bos|> <|end|> <|user|> <|ai|> <|sep|> <|sys|> <|r0|>..<|r8|>`), `VOCAB_SIZE = 4096`, `Role = Literal["user", "ai"]`.
  - `train_tokenizer(texts, out_path, vocab_size=VOCAB_SIZE) -> Tok`:
    - HF `tokenizers` with `models.BPE()`, `pre_tokenizers.ByteLevel(add_prefix_space=False)`, `decoders.ByteLevel()`.
    - `BpeTrainer(vocab_size, special_tokens=list(SPECIAL_TOKENS), initial_alphabet=ByteLevel.alphabet(), show_progress=False)`.
    - No normalizer. Creates the parent dir, saves to `out_path`.
  - `class Tok`:
    - `load` classmethod, `encode`, `decode(ids, skip_special=True)`, `id_of`.
    - Attributes `vocab_size, pad_id=0, bos_id=1, end_id=2, user_id=3, ai_id=4`.
  - `encode_doc` returns `[bos] + encode(text)`.
  - `encode_chat` returns `[bos]`, then per turn `[user_id|ai_id] + encode(text) + [end]`, then `[ai_id]` if `add_generation_prompt`.
- Fixture texts in `ml/tests/fixtures/text/`, all hand-authored (33,877 bytes total, within 25-40 KB):

  | File | Bytes | Content |
  |---|---|---|
  | `stories.txt` | 10,142 | 17 short, distinct children's stories |
  | `chats.txt` | 7,063 | 26 `User:`/`AI:` exchanges |
  | `code.txt` | 3,876 | small Python programs |
  | `facts.txt` | 8,189 | 37 factual paragraphs |
  | `unicode.txt` | 4,607 | accents, symbols, emoji, CJK, Korean, Cyrillic, Greek, Arabic, Hebrew, tabs, extra spaces |

  Paragraphs are separated by blank lines. The fixtures are not gitignored (`!ml/tests/fixtures/**` in `.gitignore`).
- `ml/tests/conftest.py`
  - Session fixture `fixture_texts() -> list[str]`: reads the five files (explicit utf-8) and returns **one string per blank-line-separated paragraph** (150 items). The brief only says `list[str]`. I chose paragraphs because the fixtures are meant to be reused as corpora, so each item is document-sized.
  - Session fixture `tiny_tok() -> Tok`: trained with `vocab_size=512` into a `tmp_path_factory` dir.
- `ml/tests/test_tokenizer.py`: the brief's 6 tests verbatim, plus 6 extra tests (listed below).

## Literal special tokens (`encode_special_tokens`)

`tokenizers.Tokenizer.encode_special_tokens` exists in the installed version (tokenizers 0.23.2), so it is used as the brief specifies.

- It is a runtime property and is **not** persisted in `tokenizer.json`. A freshly loaded `Tokenizer.from_file(...)` has it `False`.
- `Tok.__init__` therefore sets it to `True` on every construction, so both `train_tokenizer(...)` and `Tok.load(...)` get the safe behavior.
- Mutation check: with the flag off, `encode("say <|end|> now")` yields id `2` (the control token). With it on, the string becomes plain byte-level pieces `[98, 323, 236, 43, 107, 84, 279, 107, 45, 312, 318]`.
- `Tok.__init__` also verifies that the loaded tokenizer has all 16 special tokens at IDs 0-15, and raises `ValueError` otherwise. This enforces the frozen contract cheaply.

## Tests and results

Full suite: `python -m uv run pytest` gives 22 passed (4 from Task 1, 18 here), 3.85 s, no warnings.
`python -m uv run ruff check .` gives "All checks passed!".

Extra tests beyond the brief:
- Exact `SPECIAL_TOKENS` order and `VOCAB_SIZE == 4096`.
- Literal `<|end|>`, `<|ai|>`, `<|bos|>` and `<|user|>` **inside chat turns** do not add control ids. `end_id` count is exactly 2, and user, ai and bos counts are exactly 1.
- Literal special strings stay plain text after `Tok.load` (guards the "flag is not persisted" pitfall).
- Empty doc and empty chat layouts, with and without a generation prompt.
- `decode(skip_special=True/False)` on a hand-built ID list.
- Lossless round-trip over every fixture paragraph.
- Training determinism: two trainings produce byte-identical `tokenizer.json`.

### TDD evidence
RED: `cd ml && python -m uv run pytest tests/test_tokenizer.py -v`
```
ImportError while loading conftest 'C:\Users\petko\Desktop\AI Race\ml\tests\conftest.py'.
tests\conftest.py:6: in <module>
    from airace_ml.tokenizer import Tok, train_tokenizer
E   ModuleNotFoundError: No module named 'airace_ml.tokenizer'
```
Expected: the module does not exist yet.

GREEN: same command after implementing:
```
collected 18 items
... (18 PASSED lines)
============================= 18 passed in 0.21s ==============================
```

## Files changed
- Created: `ml/src/airace_ml/tokenizer.py`, `ml/tests/test_tokenizer.py`, `ml/tests/fixtures/text/{stories,chats,code,facts,unicode}.txt`
- Modified: `ml/tests/conftest.py` (was empty)

## Self-review
- The brief's test code is untouched. It passes `ruff check`, so I did not apply `ruff format`. `ruff format --check` would reformat `test_tokenizer.py`, and `tests/test_paths_device.py` from Task 1 already fails it too. The project's lint command is only `ruff check`.
- Only the specified interface is implemented: no extra public API.
- One addition: `encode_chat` raises `ValueError` for an unknown role instead of silently treating it as `ai`. `Role` is only a static type, and a silent default would hide caller bugs.
- `decode` of invalid byte sequences (e.g. a model emitting a partial UTF-8 byte sequence) yields U+FFFD, which is standard for the HF ByteLevel decoder.

## Concerns (minor, none blocking)
- `conftest.py` imports `airace_ml.tokenizer` at module level, so an import error in the tokenizer fails collection of every test.
- Vocab size and token ids for the real `tok-v1` come from training on the real corpus (a later task). Requesting 4096 on the 34 KB fixtures also reaches 4096, so a small corpus does not necessarily under-fill the vocab. A future build should still assert `tok.vocab_size == VOCAB_SIZE` after training.
- Windows cp1252 console: printing decoded text with U+FFFD or emoji raised `UnicodeEncodeError` in my ad-hoc check script. This is not a tokenizer issue. It is relevant to Review Focus 5 for the CLI (Task 19).
