### Task 12: Code, instruction and grammar generators, and checkers

**Files:**
- Create: `ml/src/airace_ml/skills/code.py`, `instructions.py`, `grammar.py`, `checkers.py`
- Test: `ml/tests/test_code_instr_grammar.py`

**Interfaces:**
- Consumes: MiniPy (Task 9), KB (Task 10), `skills/types.py`.
- Produces:
  - `CHECKERS: dict[str, Callable[[str, dict], bool]]` with these keys:

    | Key | Passes when |
    |---|---|
    | `one_word` | exactly 1 word after stripping punctuation |
    | `yes_no` | reply is "yes" or "no", case-insensitive, punctuation stripped |
    | `list_n` | `args["n"]` items separated by commas, "and" or newlines |
    | `starts_with` | reply starts with `args["prefix"]` |
    | `all_caps` | letters present and all uppercase |
    | `repeat_word` | reply equals `args["word"]`, case-insensitive, stripped |
    | `contains_any` | reply contains one of `args["words"]` |
    | `minipy_function_tests` | uses `args["prompt"]`, `args["name"]` and `args["tests"]` (`[[args, expected], …]`) to build the body (below) and every test passes via `call_function` |

    For `minipy_function_tests`, the body is the reply cut at the first line that is non-empty and starts without whitespace; the source is prompt + body.
  - `code_train_docs(rng, n) -> list[TextDoc]`:
    - 50% `"Program:\n{code}\nOutput: {stdout}"`, from random straight-line, loop and if programs over small ints and strings that print exactly one line
    - 50% function definitions with a docstring and 2 usage examples
  - `code_bench_items(rng, n_output: int = 100, n_func: int = 50) -> list[ExactItem | CheckItem]`, category `"coding"`:
    - output prediction: an `ExactItem` with prompt `"Program:\n{code}\nOutput:"`
    - function completion: a `CheckItem` with `check="minipy_function_tests"`, `chat=False`, prompt `"def {name}({args}):\n    \"\"\"{doc}\"\"\"\n"`, and `reference` = the correct body

    Function families: `add, double, square, max2, is_even, first, last, count_of, sum_list, maybe_negate`.
  - `instruction_train_docs(kb, rng, n) -> list[TextDoc]`: chats whose AI replies satisfy their checker.
  - `instruction_bench_items(kb, rng, n: int = 120) -> list[CheckItem]`: category `"instruction"`, chat format, each carrying a passing `reference`.
  - `grammar_pairs(rng, n: int = 200) -> list[PairItem]`: category `"language"`. Families, each tagged `fam:<name>`: `agreement`, `article`, `word_order`, `tense`, `plural`.
  - Partition by canonical key, as in Task 11.

- [ ] **Step 1: Write the failing tests:**

```python
from airace_ml.minipy.interpreter import run_program
from airace_ml.skills.checkers import CHECKERS
from airace_ml.skills.code import code_bench_items, code_train_docs
from airace_ml.skills.grammar import grammar_pairs
from airace_ml.skills.instructions import instruction_bench_items, instruction_train_docs
from airace_ml.skills.kb import load_kb
from airace_ml.skills.types import CheckItem, ExactItem, skill_rng

def test_output_items_match_interpreter():
    items = code_bench_items(skill_rng("coding", "bench"))
    assert sum(isinstance(i, ExactItem) for i in items) == 100 and sum(isinstance(i, CheckItem) for i in items) == 50
    for it in items:
        if isinstance(it, ExactItem):
            code = it.prompt.removeprefix("Program:\n").removesuffix("\nOutput:")
            r = run_program(code); assert r.error is None and r.stdout.strip() == it.answers[0]

def test_function_reference_passes_and_wrong_fails():
    for it in [i for i in code_bench_items(skill_rng("coding", "bench")) if isinstance(i, CheckItem)]:
        assert CHECKERS[it.check](it.reference, it.check_args)
        assert not CHECKERS[it.check]("    return None\n", it.check_args)

def test_checkers_basic():
    assert CHECKERS["one_word"]("Blue.", {}) and not CHECKERS["one_word"]("light blue", {})
    assert CHECKERS["yes_no"]("Yes!", {}) and not CHECKERS["yes_no"]("maybe", {})
    assert CHECKERS["list_n"]("cat, dog and cow", {"n": 3}) and not CHECKERS["list_n"]("cat", {"n": 3})
    assert CHECKERS["starts_with"]("Hello there", {"prefix": "Hello"}) and CHECKERS["all_caps"]("I AM HERE")
    assert CHECKERS["repeat_word"](" banana ", {"word": "banana"}) and not CHECKERS["all_caps"]("123")

def test_instruction_items_and_train_self_consistent():
    kb = load_kb()
    for it in instruction_bench_items(kb, skill_rng("instruction", "bench")):
        assert it.chat and CHECKERS[it.check](it.reference, it.check_args)
    assert all(d.kind == "chat" and d.turns[-1][0] == "ai" for d in instruction_train_docs(kb, skill_rng("instr", "train"), 200))

def test_grammar_pairs():
    ps = grammar_pairs(skill_rng("language", "bench"))
    assert len(ps) == 200 and all(p.good != p.bad and p.good and p.bad for p in ps)
    assert {t for p in ps for t in p.tags} >= {f"fam:{f}" for f in ("agreement", "article", "word_order", "tense", "plural")}

def test_code_train_docs_execute():
    for d in code_train_docs(skill_rng("code", "train"), 100):
        if d.text.startswith("Program:"):
            code, out = d.text.removeprefix("Program:\n").split("\nOutput: ")
            assert run_program(code).stdout.strip() == out.strip()
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/test_code_instr_grammar.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement to the interfaces above.**

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/test_code_instr_grammar.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(ml): code, instruction, grammar generators and checkers"`.

---

