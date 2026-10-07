import keyword
import re
from collections import Counter

import numpy as np
import pytest

from airace_ml.minipy.interpreter import run_program
from airace_ml.skills import code, grammar, instructions, types
from airace_ml.skills.checkers import CHECKERS, function_body
from airace_ml.skills.code import code_bench_items, code_train_docs
from airace_ml.skills.grammar import grammar_pairs
from airace_ml.skills.instructions import instruction_bench_items, instruction_train_docs
from airace_ml.skills.kb import load_kb
from airace_ml.skills.types import (
    CheckItem,
    ExactItem,
    PairItem,
    fair_quota,
    reserved_for_bench,
    skill_rng,
)


def test_shared_helpers_live_in_types():
    # one interleave, one rng pick and one capitalize for the Task 12 generators
    assert types.in_turn([[1, 2, 3], [4], [5, 6]]) == [1, 4, 5, 2, 6, 3] and types.in_turn([]) == []
    assert types.capitalized("the dog") == "The dog" and types.capitalized("") == ""
    rng, same = np.random.default_rng(7), np.random.default_rng(7)
    assert types.pick(rng, 5) == int(same.integers(5))
    assert types.choice(rng, ("a", "b", "c")) == ("a", "b", "c")[int(same.integers(3))]
    for module in (code, instructions, grammar):
        for name in ("_pick", "_choice", "_round_robin", "_in_turn", "_cap", "_capitalized"):
            assert not hasattr(module, name), (module.__name__, name)


TOPICS = {
    "animals", "food", "nature", "family", "school", "science", "sports", "technology",
    "places", "feelings", "other",
}  # fmt: skip


def test_output_items_match_interpreter():
    items = code_bench_items(skill_rng("coding", "bench"))
    assert (
        sum(isinstance(i, ExactItem) for i in items) == 100
        and sum(isinstance(i, CheckItem) for i in items) == 50
    )
    for it in items:
        if isinstance(it, ExactItem):
            code = it.prompt.removeprefix("Program:\n").removesuffix("\nOutput:")
            r = run_program(code)
            assert r.error is None and r.stdout.strip() == it.answers[0]


def test_function_reference_passes_and_wrong_fails():
    for it in [
        i for i in code_bench_items(skill_rng("coding", "bench")) if isinstance(i, CheckItem)
    ]:
        assert CHECKERS[it.check](it.reference, it.check_args)
        assert not CHECKERS[it.check]("    return None\n", it.check_args)


def test_checkers_basic():
    assert CHECKERS["one_word"]("Blue.", {}) and not CHECKERS["one_word"]("light blue", {})
    assert CHECKERS["yes_no"]("Yes!", {}) and not CHECKERS["yes_no"]("maybe", {})
    assert CHECKERS["list_n"]("cat, dog and cow", {"n": 3}) and not CHECKERS["list_n"](
        "cat", {"n": 3}
    )
    assert CHECKERS["starts_with"]("Hello there", {"prefix": "Hello"}) and CHECKERS["all_caps"](
        "I AM HERE"
    )
    assert CHECKERS["repeat_word"](" banana ", {"word": "banana"}) and not CHECKERS["all_caps"](
        "123"
    )


def test_instruction_items_and_train_self_consistent():
    kb = load_kb()
    for it in instruction_bench_items(kb, skill_rng("instruction", "bench")):
        assert it.chat and CHECKERS[it.check](it.reference, it.check_args)
    assert all(
        d.kind == "chat" and d.turns[-1][0] == "ai"
        for d in instruction_train_docs(kb, skill_rng("instr", "train"), 200)
    )


def test_grammar_pairs():
    ps = grammar_pairs(skill_rng("language", "bench"))
    assert len(ps) == 200 and all(p.good != p.bad and p.good and p.bad for p in ps)
    assert {t for p in ps for t in p.tags} >= {
        f"fam:{f}" for f in ("agreement", "article", "word_order", "tense", "plural")
    }


def test_code_train_docs_execute():
    for d in code_train_docs(skill_rng("code", "train"), 100):
        if d.text.startswith("Program:"):
            code, out = d.text.removeprefix("Program:\n").split("\nOutput: ")
            assert run_program(code).stdout.strip() == out.strip()


# ---------------------------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------------------------

REPLIES_THAT_SAY_NOTHING = ["", " ", "\n\n", "...", "!?", "\t"]
ARGS_FOR_ALL_CHECKERS = {"n": 2, "prefix": "Sure", "word": "dog", "words": ["dog"]}


@pytest.mark.parametrize("name", [n for n in CHECKERS if n != "minipy_function_tests"])
def test_no_checker_accepts_an_empty_reply(name):
    for reply in REPLIES_THAT_SAY_NOTHING:
        assert not CHECKERS[name](reply, ARGS_FOR_ALL_CHECKERS), (name, reply)


def test_one_word_and_yes_no_forgive_form_not_content():
    one = CHECKERS["one_word"]
    assert one("Paris") and one(" paris. ") and one("PARIS!") and one("well-known") and one("don't")
    assert not one("New York") and not one("Paris\nFrance") and not one("It is Paris.")
    yes_no = CHECKERS["yes_no"]
    for reply in ("yes", "Yes.", "YES!", " no ", "No", "no."):
        assert yes_no(reply), reply
    for reply in ("yes yes", "Yes, it is.", "maybe", "yes or no", "nope", "yess"):
        assert not yes_no(reply), reply


def test_one_word_can_refuse_yes_and_no():
    refuse = {"not": ["yes", "no"]}
    assert CHECKERS["one_word"]("Paris.", refuse) and CHECKERS["one_word"]("Yes", {})
    assert not CHECKERS["one_word"]("Yes.", refuse) and not CHECKERS["one_word"]("NO!", refuse)


def test_list_n_counts_items_separated_by_commas_and_or_lines():
    n3 = {"n": 3}
    for reply in (
        "cat, dog and cow",
        "cat, dog, cow",
        "cat, dog, and cow",
        "cat\ndog\ncow",
        "Cat, Dog and Cow.",
        "1. cat\n2. dog\n3. cow",
        "- cat\n- dog\n- cow",
        "ice cream, polar bear and sea lion",
    ):
        assert CHECKERS["list_n"](reply, n3), reply
    for reply in (
        "cat",
        "cat, dog",
        "cat, dog, cow, pig",
        "cat, cat, cat",  # the same item over and over is not three items
        "The cat is nice, the dog is big, and the cow is fine.",  # sentences are not items
        "cat, dog, and",
        "Here are three animals: cat, dog and cow",
    ):
        assert not CHECKERS["list_n"](reply, n3), reply
    assert CHECKERS["list_n"]("cat and dog", {"n": 2})
    assert not CHECKERS["list_n"]("cat and dog", {"n": 3})


def test_starts_with_wants_the_word_at_the_start():
    sw = CHECKERS["starts_with"]
    arg = {"prefix": "Sure"}
    for reply in ("Sure! The sky is blue.", "sure, it is", "  SURE.", "Sure"):
        assert sw(reply, arg), reply
    for reply in ("Surely!", "Oh sure", "The sky is blue. Sure", "Sur"):
        assert not sw(reply, arg), reply
    assert not sw("Sure " * 30, arg)  # nothing but the prefix over and over
    assert sw("Once upon a time there was a cat.", {"prefix": "Once upon a time"})


def test_all_caps_needs_letters_and_no_lowercase():
    caps = CHECKERS["all_caps"]
    for reply in ("I AM HERE", "YES!", "THE DOG IS A MAMMAL.", "A1 B2"):
        assert caps(reply), reply
    for reply in ("I am here", "Yes", "THE dog", "123", "!!!"):
        assert not caps(reply), reply
    assert not caps("AA " * 30)
    sentence = {"min_words": 3}  # a sentence asked for in capitals is not one word
    assert caps("THE DOG RUNS.", sentence) and caps("YES", {}) and caps("YES", {"min_words": 1})
    assert not caps("YES", sentence) and not caps("YES SIR", sentence)


def test_repeat_word_is_the_word_alone():
    rw = CHECKERS["repeat_word"]
    arg = {"word": "banana"}
    for reply in ("banana", " Banana ", "banana.", '"banana"', "BANANA!"):
        assert rw(reply, arg), reply
    for reply in ("banana banana", "a banana", "bananas", "I say banana", "banan"):
        assert not rw(reply, arg), reply


def test_contains_any_finds_a_whole_word():
    ca = CHECKERS["contains_any"]
    arg = {"words": ["dog", "dogs", "cat", "cats"]}
    for reply in ("The dog is big.", "I like CATS.", "Dogs are fun.", "a cat", "cat"):
        assert ca(reply, arg), reply
    for reply in ("The doge is big.", "A category.", "I like cows.", "Hmm hmm hmm."):
        assert not ca(reply, arg), reply
    assert not ca("dog " * 30, arg)


def test_degenerate_replies_fail_every_checker():
    junk = "the " * 40
    args = {"n": 3, "prefix": "the", "word": "the", "words": ["the"]}
    for name in CHECKERS:
        if name != "minipy_function_tests":
            assert not CHECKERS[name](junk, args), name


def test_starts_with_can_ask_for_an_answer_after_the_word():
    sw = CHECKERS["starts_with"]
    arg = {"prefix": "Sure", "min_words": 2}  # the word, then an answer
    for reply in ("Sure! Kigali.", "sure, it is Kigali", "SURE. THE SKY IS BLUE."):
        assert sw(reply, arg), reply
    for reply in ("Sure", "Sure!", " sure. ", "Kigali. Sure!", "Sure Sure"):
        assert not sw(reply, arg), reply
    assert sw("Sure", {"prefix": "Sure"})  # without min_words the word alone is enough


def test_contains_any_can_ask_for_a_sentence():
    ca = CHECKERS["contains_any"]
    arg = {"words": ["otter", "otters"], "min_words": 3}
    for reply in ("The otter swims.", "Otters are cute!", "I SAW AN OTTER."):
        assert ca(reply, arg), reply
    for reply in ("otter", "Otter!", "the otter", "an otter."):
        assert not ca(reply, arg), reply


def test_contains_any_takes_exactly_the_listed_forms():
    # a plural counts when it is listed, as it is spelt: no rule that makes "cherrys" a word
    ca = CHECKERS["contains_any"]
    cherry = {"words": ["cherry", "cherries"], "min_words": 3}
    for reply in ("Cherries are red.", "The cherry is red.", "I like the cherry's color."):
        assert ca(reply, cherry), reply
    for reply in ("Cherrys are red.", "Cherryes are red.", "A cherrytree is tall."):
        assert not ca(reply, cherry), reply
    assert not ca("Otters are cute.", {"words": ["otter"], "min_words": 3})


INSTRUCTION_CASES = {  # checker -> (instruction, arguments, a reply that follows it)
    "one_word": ("What is the capital of Peru? Answer with one word.", {"not": ["yes"]}, "Lima."),
    "yes_no": ("Is 3 bigger than 2? Answer yes or no.", {}, "Yes."),
    "list_n": ("Name three animals that eat grass.", {"n": 3}, "cow, goat and sheep"),
    "starts_with": (
        "Start your reply with the word Sure. What is the capital of Peru?",
        {"prefix": "Sure", "min_words": 2},
        "Sure! The capital of Peru is Lima.",
    ),
    "all_caps": (
        "What is the capital of Peru? Answer in a sentence, in all capital letters.",
        {"min_words": 3},
        "THE CAPITAL OF PERU IS LIMA.",
    ),
    "repeat_word": ("Say the word otter.", {"word": "otter"}, "otter"),
    "contains_any": (
        "Write one sentence that uses the word otter.",
        {"words": ["otter"], "min_words": 3},
        "The otter swims fast.",
    ),
}


def test_a_copy_of_the_instruction_is_not_a_reply():
    # a model that repeats the user's message has not followed it, whatever words that has in it
    assert set(INSTRUCTION_CASES) == set(CHECKERS) - {"minipy_function_tests"}
    for name, (instruction, args, good) in INSTRUCTION_CASES.items():
        args = {**args, "instruction": instruction}
        check = CHECKERS[name]
        assert check(good, args), name
        copies = [
            instruction,
            instruction.upper(),
            instruction.lower().rstrip("."),
            "  " + instruction,
        ]
        copies += [f"{instruction}\n{instruction}", f'"{instruction}"']
        if name != "one_word":  # "Sure!" alone is a one-word answer
            copies.append(f"Sure! {instruction}")
        for copy in copies:
            assert not check(copy, args), (name, copy)
        # what follows a copy is judged on its own
        assert check(f"{instruction}\n{good}", args), name
        assert check(f"{instruction} {good}", args), name


FUNCTION_PROMPT = 'def add(a, b):\n    """Return the sum of a and b."""\n'
ADD_ARGS = {
    "prompt": FUNCTION_PROMPT,
    "name": "add",
    "tests": [[[1, 2], 3], [[5, 7], 12], [[0, 9], 9]],
}


def test_function_body_stops_at_the_first_unindented_line():
    assert function_body("    return a + b\n") == "    return a + b\n"
    assert function_body("    return a + b\nprint(add(1, 2))\n") == "    return a + b\n"
    assert function_body("    x = 1\n\n    return x\n\ndef other():\n    pass\n") == (
        "    x = 1\n\n    return x\n\n"
    )
    assert function_body("return a + b") == ""
    assert function_body("") == ""
    assert function_body("    return a + b") == "    return a + b"
    assert function_body("   \n\t\n    return 1\n# comment\n    return 2\n") == (
        "   \n\t\n    return 1\n"
    )


def test_function_checker_runs_the_reply_against_every_test():
    check = CHECKERS["minipy_function_tests"]
    assert check("    return a + b\n", ADD_ARGS)
    assert check("    return b + a\n\nprint(add(1, 2))\n", ADD_ARGS)
    assert check("    total = a + b\n    return total\n\n\ndef next_one():\n    pass\n", ADD_ARGS)
    for body in (
        "",
        "    return None\n",
        "    return 0\n",
        "    return a\n",
        "    return b\n",
        "    return a - b\n",
        "    return a * b\n",
        "    pass\n",
        "return a + b\n",  # not indented: not a body
        "    return a +\n",  # a syntax error
        "    return a + c\n",  # a name that does not exist
        "    while True:\n        pass\n",  # never ends: MiniPy stops it
        "    return a ** b\n",  # not in MiniPy
        "    return str(a + b)\n",  # "3" is not 3
        "    return True\n",
    ):
        assert not check(body, ADD_ARGS), body
    # type matters: 1 is not True, and a list must match element by element
    even = {"prompt": 'def f(n):\n    """d"""\n', "name": "f", "tests": [[[4], True], [[7], False]]}
    assert check("    return n % 2 == 0\n", even)
    assert not check("    return 1 - n % 2\n", even)
    firsts = {"prompt": 'def f(x):\n    """d"""\n', "name": "f", "tests": [[[[1, 2]], [1, 2]]]}
    assert check("    return [x[0], x[1]]\n", firsts)
    assert not check("    return [x[0], x[1], 3]\n", firsts)
    assert not check("    return x[0]\n", firsts)


# ---------------------------------------------------------------------------------------------
# Coding
# ---------------------------------------------------------------------------------------------


def split_items(items):
    return (
        [i for i in items if isinstance(i, ExactItem)],
        [i for i in items if isinstance(i, CheckItem)],
    )


def family_tag(item) -> str:
    (tag,) = [t for t in item.tags if t.startswith("fam:")]
    return tag.removeprefix("fam:")


@pytest.fixture(scope="module")
def coding_bench():
    return code_bench_items(skill_rng("coding", "bench"))


@pytest.fixture(scope="module")
def code_train():
    return code_train_docs(skill_rng("code", "train"), 6000)


def program_of(item: ExactItem) -> str:
    return item.prompt.removeprefix("Program:\n").removesuffix("\nOutput:")


def train_programs(docs) -> list[tuple[str, str]]:
    found = []
    for doc in docs:
        if doc.text.startswith("Program:"):
            source, out = doc.text.removeprefix("Program:\n").split("\nOutput: ")
            found.append((source, out))
    return found


def test_coding_bench_shape(coding_bench):
    outputs, functions = split_items(coding_bench)
    assert [i.id for i in coding_bench] == [f"coding-{k:04d}" for k in range(150)]
    assert len(outputs) == 100 and len(functions) == 50
    for it in coding_bench:
        assert it.category == "coding" and not it.chat
        # an output item's world key is its prompt; a function item's is its def line
        key = it.prompt if isinstance(it, ExactItem) else def_line(it.prompt)
        assert it.prompt.isascii() and reserved_for_bench(key)
    for it in outputs:
        assert it.extract == "first_line" and it.max_new_tokens == code.OUTPUT_TOKENS
        assert it.prompt.startswith("Program:\n") and it.prompt.endswith("\nOutput:")
        assert len(it.answers) == 1 and it.answers[0] == it.answers[0].strip() != ""
        assert "fmt:output" in it.tags and family_tag(it) in code.OUTPUT_FAMILIES
    for it in functions:
        assert it.check == "minipy_function_tests" and it.max_new_tokens == code.FUNCTION_TOKENS
        assert re.fullmatch(r'def \w+\(\w+(, \w+)?\):\n    """[^"\n]+"""\n', it.prompt)
        assert it.check_args["prompt"] == it.prompt
        assert it.check_args["name"] == re.match(r"def (\w+)", it.prompt)[1]
        assert "fmt:function" in it.tags and family_tag(it) in code.FUNCTION_FAMILIES
    assert len({i.prompt for i in coding_bench}) == 150
    # both kinds are spread over the list, so a prefix has both
    assert 8 <= sum(isinstance(i, CheckItem) for i in coding_bench[:30]) <= 12


def test_coding_bench_families_are_balanced(coding_bench):
    outputs, functions = split_items(coding_bench)
    out_counts = Counter(family_tag(i) for i in outputs)
    assert set(out_counts) == set(code.OUTPUT_FAMILIES)
    assert sorted(out_counts.values()) == [16, 16, 17, 17, 17, 17]
    assert Counter(family_tag(i) for i in functions) == {f: 5 for f in code.FUNCTION_FAMILIES}
    # families take turns, so the first 12 outputs show every family twice
    first = [family_tag(i) for i in outputs[:12]]
    assert Counter(first) == {f: 2 for f in code.OUTPUT_FAMILIES}


def test_output_programs_print_exactly_one_line_and_are_tiny(coding_bench):
    outputs, _ = split_items(coding_bench)
    for it in outputs:
        source = program_of(it)
        result = run_program(source)
        assert result.error is None and result.stdout == it.answers[0] + "\n", source
        assert result.steps < 400 and len(source.split("\n")) <= 12, source
        assert ".upper()" not in source and ".lower()" not in source  # answers are case-blind


def int_and_str_literals(source: str) -> tuple[list[int], list[str]]:
    strings = re.findall(r'"([^"]*)"', source)
    bare = re.sub(r'"[^"]*"', '""', source)
    return [int(x) for x in re.findall(r"(?<![A-Za-z_])\d+", bare)], strings


def shortcut_scores(programs: list[tuple[str, str]]) -> dict[str, float]:
    """The share of programs a rule using only the program text answers, one rule at a time.

    A rule that names several candidates counts as the chance of picking the right one.
    """
    totals: Counter[str] = Counter()
    for source, answer in programs:
        ints, strs = int_and_str_literals(source)
        literals = [*(str(i) for i in ints), *strs]
        rules = {
            "last int literal": [str(ints[-1])] if ints else [],
            "first int literal": [str(ints[0])] if ints else [],
            "largest int literal": [str(max(ints))] if ints else [],
            "smallest int literal": [str(min(ints))] if ints else [],
            "any int literal": [str(i) for i in ints],
            "any literal": literals,
            "any literal, in any case": [lit.casefold() for lit in literals],
            "last string literal": strs[-1:],
            "first string literal": strs[:1],
            "sum of the int literals": [str(sum(ints))] if ints else [],
            "sum of the last two int literals": [str(sum(ints[-2:]))] if ints else [],
            "product of the last two int literals": (
                [str(ints[-1] * ints[-2])] if len(ints) > 1 else []
            ),
            "all string literals joined": ["".join(strs)] if strs else [],
            "number of lines": [str(len(source.split("\n")))],
        }
        for name, candidates in rules.items():
            target = answer.casefold() if "any case" in name else answer
            if candidates:
                totals[name] += candidates.count(target) / len(candidates)
    return {name: total / len(programs) for name, total in totals.items()}


def test_no_program_gives_its_answer_away(coding_bench, code_train):
    outputs, _ = split_items(coding_bench)
    bench = [(program_of(i), i.answers[0]) for i in outputs]
    train = train_programs(code_train)
    assert len(train) > 2500
    for name, programs in (("bench", bench), ("train", train)):
        scores = shortcut_scores(programs)
        assert max(scores.values()) <= 0.2, (name, scores)
        for source, answer in programs:  # the line is worked out, never read off the program
            ints, strs = int_and_str_literals(source)
            assert answer not in map(str, ints), source
            assert answer.casefold() not in (s.casefold() for s in strs), source
            assert len(answer) < 3 or answer.casefold() not in source.casefold(), source
    assert max(shortcut_scores(bench).values()) <= 0.1


def test_the_most_common_answer_is_not_a_good_guess(coding_bench, code_train):
    outputs, _ = split_items(coding_bench)
    prior = Counter(answer for _, answer in train_programs(code_train))
    top, count = prior.most_common(1)[0]
    assert count / sum(prior.values()) < 0.06
    assert sum(1 for i in outputs if i.answers[0] == top) / len(outputs) <= 0.2
    answers = Counter(i.answers[0] for i in outputs)
    assert len(answers) >= 50 and max(answers.values()) <= 10
    kinds = Counter("int" if re.fullmatch(r"-?\d+", i.answers[0]) else "text" for i in outputs)
    assert kinds["text"] >= 15 and kinds["int"] >= 60


def canonical_code(source: str) -> str:
    """The program with its variable names replaced by v0, v1... in order of first appearance."""
    keep = set(keyword.kwlist) | {
        "print", "len", "range", "max", "min", "sum", "abs", "str", "int", "list", "sorted",
        "append", "upper", "lower",
    }  # fmt: skip
    names: dict[str, str] = {}

    def rename(match: re.Match[str]) -> str:
        word = match.group(0)
        return word if word in keep else names.setdefault(word, f"v{len(names)}")

    parts = re.split(r'("[^"]*")', source)
    return "".join(
        part if part.startswith('"') else re.sub(r"(?<![\w.])[A-Za-z_]\w*", rename, part)
        for part in parts
    )


def test_training_never_contains_a_benchmark_program_under_any_names(coding_bench, code_train):
    outputs, functions = split_items(coding_bench)
    bench_worlds = {canonical_code(program_of(i)) for i in outputs}
    assert len(bench_worlds) == 100  # no two items are the same computation
    train = train_programs(code_train)
    seen = {canonical_code(source) for source, _ in train}
    assert len(seen) > 2000 and not seen & bench_worlds
    all_text = "\n".join(d.text for d in code_train)
    assert not any(it.prompt in all_text for it in coding_bench)
    # training uses no reserved prompt either, and no reserved function header
    assert not any(reserved_for_bench(f"Program:\n{source}\nOutput:") for source, _ in train)
    headers = {d.text[: d.text.index('"""\n') + 4] for d in code_train if d.text.startswith("def ")}
    assert len(headers) > 400
    assert not headers & {i.prompt for i in functions}
    assert not any(reserved_for_bench(h) for h in headers)


def test_every_shape_of_program_is_in_training_and_the_benchmark(code_train, coding_bench):
    shapes: Counter[str] = Counter()
    for source, _ in train_programs(code_train):
        for shape, found in {
            "case": ".upper()" in source or ".lower()" in source,
            "while": "while" in source,
            "elif": "elif" in source,
            "def": "def " in source,
            "append": "append" in source,
            "for": re.search(r"\bfor\b", source) is not None,
            "string": '"' in source,
        }.items():
            shapes[shape] += found
    assert min(shapes.values()) > 50, shapes
    outputs, _ = split_items(coding_bench)
    bench_text = "\n".join(program_of(i) for i in outputs)
    for part in ("while", "elif", "def ", "append", "for ", '"'):
        assert part in bench_text, part


def test_every_program_template_has_enough_worlds_to_draw_from():
    # a template with a tiny space cannot fill a benchmark (some once had 3-9 worlds)
    rng = np.random.default_rng(0)
    for family in code.OUTPUT_FAMILIES:
        for control in range(code.N_VARIANTS[family]):
            worlds = {code._TEMPLATES[family](rng, control).world for _ in range(3000)}
            assert len(worlds) >= 90, (family, control, len(worlds))
            assert sum(reserved_for_bench(w) for w in worlds) >= 5, (family, control)


# --- functions ---------------------------------------------------------------------------------


def header_parts(prompt: str) -> tuple[str, list[str]]:
    name, params = re.match(r"def (\w+)\(([^)]*)\)", prompt).groups()
    return name, params.split(", ")


TRIVIAL_BODIES = (
    "",
    "    pass\n",
    "    return None\n",
    "    return 0\n",
    "    return 1\n",
    "    return -1\n",
    "    return 2\n",
    "    return True\n",
    "    return False\n",
    '    return ""\n',
    "    return []\n",
)


def trivial_bodies_for(params: list[str]) -> list[str]:
    bodies = list(TRIVIAL_BODIES)
    for p in params:
        bodies += [f"    return {p}\n", f"    return -{p}\n", f"    return {p} + 1\n"]
        bodies += [f"    return len({p})\n", f"    return {p}[0]\n", f"    return {p} % 2\n"]
    if len(params) == 2:
        a, b = params
        bodies += [f"    return {a} + {b}\n", f"    return {a} * {b}\n", f"    return {a} - {b}\n"]
        bodies += [f"    return min({a}, {b})\n", f"    return max({a}, {b})\n"]
        bodies += [f"    return {a} == {b}\n", f"    return {a} > {b}\n"]
    return bodies


def test_no_trivial_body_passes_any_function_item(coding_bench):
    _, functions = split_items(coding_bench)
    for it in functions:
        _, params = header_parts(it.prompt)
        for body in trivial_bodies_for(params):
            if CHECKERS[it.check](body, it.check_args):
                # only a body that is just what the item asks for may pass (max, sum, ...)
                assert body.strip() == it.reference.strip(), (it.prompt, body)


def test_every_family_has_tests_that_tell_it_from_every_other_family():
    # the body of any other family, written over this function's parameters, never passes
    for fam in code._FUNCTION_FAMILIES:
        variant = code._Variant(fam.name, fam.names[0], fam.params[0], 0)
        args = {"prompt": variant.prompt, "name": variant.name, "tests": fam.test_cases}
        for other in code._FUNCTION_FAMILIES:
            if other.name == fam.name or len(other.params[0]) > len(fam.params[0]):
                continue
            for style in range(len(other.bodies)):
                fields = {f"p{i}": p for i, p in enumerate(fam.params[0])}
                fields |= {"l0": "count", "l1": "item"}
                body = other.bodies[style].format(**fields)
                assert not CHECKERS["minipy_function_tests"](body, args), (fam.name, other.name)


def test_every_body_style_of_every_header_is_right():
    # the bodies the training text uses, over every name, parameter set and local variable name
    rng = np.random.default_rng(0)
    families = {f.name: f for f in code._FUNCTION_FAMILIES}
    seen = 0
    for k, pools in enumerate(code._variants()):
        for family, variants in pools.items():
            for variant in variants[:: 1 if k == 0 else 3]:
                fam = families[family]
                args = {"prompt": variant.prompt, "name": variant.name, "tests": fam.test_cases}
                for style in range(len(fam.bodies)):
                    for body_rng in (None, rng):
                        body = variant.body(style, body_rng)
                        assert CHECKERS["minipy_function_tests"](body, args), (variant, style)
                        seen += 1
    assert seen > 3500


def test_function_tests_come_from_what_the_function_does():
    for fam in code._FUNCTION_FAMILIES:
        assert len(fam.tests) >= 5 and len(set(map(repr, fam.test_cases))) == len(fam.tests)
        outputs = {repr(expected) for _, expected in fam.test_cases}
        # a constant never passes: a yes/no function has both answers, the others at least four
        assert len(outputs) >= (2 if fam.name == "is_even" else 4), fam.name
        assert not any(expected is None for _, expected in fam.test_cases)
        for call_args, expected in fam.test_cases:
            assert fam.ref(*call_args) == expected


def test_function_tests_fit_every_docstring_of_their_family():
    # the hidden tests check what every wording promises: a function described as returning
    # "the first number in the list" is never called on a list of words
    for fam in code._FUNCTION_FAMILIES:
        if any("number" in doc for doc in fam.docs):
            for call_args, _ in fam.test_cases:
                for value in call_args:
                    items = value if isinstance(value, list) else [value]
                    assert all(type(v) is int for v in items), (fam.name, call_args)


def def_line(prompt: str) -> str:
    """``def name(params):``, the first line of a function header."""
    return prompt.split("\n", 1)[0]


def test_no_benchmark_function_name_and_parameters_occur_in_training(coding_bench, code_train):
    # a function's world is its def line, whatever its docstring: the benchmark asks for bodies of
    # functions training never shows under that name and those parameters (it teaches every family
    # under other names)
    _, functions = split_items(coding_bench)
    lines = {def_line(i.prompt) for i in functions}
    assert len(lines) == len(functions) == 50  # every item a def line of its own
    assert all(reserved_for_bench(line) for line in lines)
    text = "\n".join(d.text for d in code_train)
    assert sum(d.text.startswith("def ") for d in code_train) > 2500
    assert not [line for line in lines if line in text]
    trained = {def_line(d.text) for d in code_train if d.text.startswith("def ")}
    assert not any(reserved_for_bench(line) for line in trained)
    by_name = {n: f.name for f in code._FUNCTION_FAMILIES for n in f.names}
    taught = {by_name[re.match(r"def (\w+)", line)[1]] for line in trained}
    assert taught == set(code.FUNCTION_FAMILIES)  # every family is still taught


def test_every_function_family_has_enough_benchmark_def_lines():
    bench, _ = code._variants()
    for family in code.FUNCTION_FAMILIES:
        assert len({def_line(v.prompt) for v in bench[family]}) >= 5, family


def test_docstrings_ask_only_for_what_minipy_can_do():
    # "x squared" invites x ** 2 and "divided by" invites /, which MiniPy does not have
    unsupported = re.compile(r"squared|square of|power|divided|divide\b|/|\*\*", re.IGNORECASE)
    for fam in code._FUNCTION_FAMILIES:
        for doc in fam.docs:
            assert not unsupported.search(doc), (fam.name, doc)


def test_function_names_headers_and_docs_are_unambiguous():
    names = [n for fam in code._FUNCTION_FAMILIES for n in fam.names]
    assert len(names) == len(set(names)) >= 80  # a name says which family it is
    for fam in code._FUNCTION_FAMILIES:
        assert len(fam.names) >= 8 and len(fam.docs) >= 6 and len(fam.params) == 8
        arity = len(fam.params[0])
        for params in fam.params:
            assert len(params) == arity and len(set(params)) == arity
            assert not set(params) & set(fam.names)
            assert not set(params) & {"sum", "max", "min", "len", "list", "str", "int", "print"}
        for doc in fam.docs:
            assert doc.endswith(".") and '"' not in doc and "\n" not in doc
            assert all(f"{{p{i}}}" in doc for i in range(arity))
    bench, train = code._variants()
    for family in code.FUNCTION_FAMILIES:
        assert len(bench[family]) >= 30 and len(train[family]) >= 250
        assert not {v.prompt for v in bench[family]} & {v.prompt for v in train[family]}
        assert not {v.def_line for v in bench[family]} & {v.def_line for v in train[family]}


def test_function_training_docs_run_and_are_right(code_train):
    by_name = {n: f for f in code._FUNCTION_FAMILIES for n in f.names}
    functions = [d for d in code_train if d.text.startswith("def ")]
    assert len(functions) > 2500
    families: Counter[str] = Counter()
    for doc in functions:
        prompt_end = doc.text.index('"""\n', doc.text.index('"""') + 3) + 4
        prompt, rest = doc.text[:prompt_end], doc.text[prompt_end:]
        body, examples = rest.split("\n\n")
        name, _ = header_parts(prompt)
        fam = by_name[name]
        families[fam.name] += 1
        args = {"prompt": prompt, "name": name, "tests": fam.test_cases}
        assert CHECKERS["minipy_function_tests"](body, args), doc.text
        # two usage examples; running the whole text prints what the comments say
        calls = examples.split("\n")
        assert len(calls) == 2 and len(set(calls)) == 2
        printed = run_program(doc.text)
        assert printed.error is None, doc.text
        assert printed.stdout.split("\n")[:-1] == [c.split("  # ")[1] for c in calls], doc.text
        assert doc.kind == "plain" and doc.topic == "technology"
        if fam.name == "is_even":  # the two examples show both answers
            assert {c.split("  # ")[1] for c in calls} == {"True", "False"}
    assert set(families) == set(code.FUNCTION_FAMILIES)
    assert min(families.values()) > 150


def test_training_has_programs_and_functions_in_equal_parts(code_train):
    programs = sum(d.text.startswith("Program:") for d in code_train)
    functions = sum(d.text.startswith("def ") for d in code_train)
    assert programs + functions == len(code_train) == 6000
    assert 0.45 < programs / len(code_train) < 0.55
    assert all(d.kind == "plain" and d.topic == "technology" for d in code_train)


def test_code_generators_are_deterministic_and_stream_dependent():
    a = [d.text for d in code_train_docs(skill_rng("code", "train"), 40)]
    assert a == [d.text for d in code_train_docs(skill_rng("code", "train"), 40)]
    assert a != [d.text for d in code_train_docs(skill_rng("code", "train", "v2"), 40)]
    assert code_train_docs(skill_rng("code", "train"), 0) == []
    b = code_bench_items(skill_rng("coding", "bench"), 12, 6)
    assert [i.prompt for i in b] == [
        i.prompt for i in code_bench_items(skill_rng("coding", "bench"), 12, 6)
    ]
    assert len(b) == 18 and [i.id for i in b] == [f"coding-{k:04d}" for k in range(18)]
    assert code_bench_items(skill_rng("coding", "bench"), 0, 0) == []
    only_outputs = code_bench_items(skill_rng("coding", "bench"), 7, 0)
    assert len(only_outputs) == 7 and all(isinstance(i, ExactItem) for i in only_outputs)
    only_functions = code_bench_items(skill_rng("coding", "bench"), 0, 10)
    assert Counter(family_tag(i) for i in only_functions) == {f: 1 for f in code.FUNCTION_FAMILIES}
    with pytest.raises(ValueError, match="cannot take"):
        code_bench_items(skill_rng("coding", "bench"), 0, 100_000)


def test_code_gives_up_when_nothing_can_be_accepted(monkeypatch):
    calls = []
    monkeypatch.setattr(code, "reserved_for_bench", lambda key: calls.append(key) or False)
    with pytest.raises(RuntimeError, match="code"):
        code_bench_items(skill_rng("coding", "bench"), 3, 0)
    assert len(calls) == 50 * 3  # the budget is 50 world draws per requested program
    monkeypatch.setattr(code, "reserved_for_bench", lambda key: True)
    with pytest.raises(RuntimeError, match="code"):
        code_train_docs(skill_rng("code", "train"), 40)


# ---------------------------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------------------------


NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def word_parts(text: str) -> set[str]:
    """The words of ``text`` in lower case, whole ("one-word") and in pieces ("one", "word")."""
    whole = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text)
    return {w.casefold() for w in [*whole, *re.findall(r"[^\W_]+", text)]}


def expected_check(user: str, reply: str = "") -> tuple[str, dict]:
    """What a person reading the instruction would say is asked: the check and its arguments.

    Every check is given the instruction, so a copy of it is not a reply. A one-word answer is not
    "yes", "no" or a word of the question (other than the answer, ``reply``).
    """
    check, args = expected_check_of(user)
    if check == "one_word":
        args["not"] = sorted(({"yes", "no"} | word_parts(user)) - word_parts(reply))
    return check, {**args, "instruction": user}


def expected_check_of(user: str) -> tuple[str, dict]:
    if re.search(r"\byes or no\b", user, re.IGNORECASE):
        return "yes_no", {}
    if re.search(r"\bone[- ]word\b", user, re.IGNORECASE):
        return "one_word", {}
    if re.search(r"capital letters|ALL CAPS", user):
        return "all_caps", {"min_words": 3}
    start = re.search(
        r"\b(?:start|begin) (?:your (?:answer|reply) )?with the word (\w+)", user, re.IGNORECASE
    )
    if start:  # the word, then an answer to the question
        return "starts_with", {"prefix": start[1], "min_words": 2}
    repeat = re.fullmatch(
        r"(?:Say the word (\w+)\.|Repeat the word (\w+)\.|Say (\w+) and nothing else\.|"
        r"Just say (\w+)\.|Reply with the word (\w+) only\.|Repeat after me: (\w+)|"
        r"Please repeat the word (\w+)\.|Write the word (\w+) and nothing else\.)",
        user,
    )
    if repeat:
        return "repeat_word", {"word": next(g for g in repeat.groups() if g)}
    listing = re.match(
        r"(?:List|Name|Give me|Write down|Tell me|Can you list|Think of|I need|Please list|"
        r"Name any|Write a list of|Could you name) "
        r"(two|three|four|five|six) ",
        user,
    )
    if listing:
        return "list_n", {"n": NUMBER_WORDS[listing[1]]}
    contains = re.search(r"the word (\w+)", user)
    if contains and re.search(r"sentence|Say something", user):  # a sentence: 3 words or more
        return "contains_any", {"words": contains_forms(contains[1]), "min_words": 3}
    raise AssertionError(f"cannot tell what is asked: {user!r}")


@pytest.fixture(scope="module")
def kb():
    return load_kb()


@pytest.fixture(scope="module")
def instruction_bench(kb):
    return instruction_bench_items(kb, skill_rng("instruction", "bench"))


@pytest.fixture(scope="module")
def instruction_train(kb):
    return instruction_train_docs(kb, skill_rng("instr", "train"), 3000)


def chat_pairs(docs):
    for doc in docs:
        turns = doc.turns
        assert [role for role, _ in turns] == ["user", "ai"] * (len(turns) // 2)
        for k in range(0, len(turns), 2):
            yield turns[k][1], turns[k + 1][1]


def test_instruction_bench_shape(instruction_bench):
    items = instruction_bench
    assert [i.id for i in items] == [f"instruction-{k:04d}" for k in range(120)]
    kinds = Counter(family_tag(i) for i in items)
    assert set(kinds) == set(instructions.KINDS) and sorted(kinds.values()) == [17] * 6 + [18]
    for it in items:
        assert isinstance(it, CheckItem) and it.category == "instruction" and it.chat
        assert it.check == family_tag(it) and it.check in CHECKERS
        assert [t.split(":")[0] for t in it.tags] == ["fam", "topic"]
        assert it.tags[1].removeprefix("topic:") in TOPICS
        assert it.prompt.strip() == it.prompt and it.prompt and it.reference
        assert reserved_for_bench(it.prompt)
        assert it.max_new_tokens == instructions.MAX_NEW_TOKENS
    assert len({i.prompt for i in items}) == 120
    # the kinds take turns, so any prefix is as balanced as it can be
    assert {family_tag(i) for i in items[:7]} == set(instructions.KINDS)
    assert max(Counter(family_tag(i) for i in items[:35]).values()) == 5


def test_instruction_text_says_what_the_checker_checks(instruction_bench, instruction_train):
    for it in instruction_bench:
        assert expected_check(it.prompt, it.reference) == (it.check, it.check_args), it.prompt
    seen = Counter()
    for user, _ in chat_pairs(instruction_train):
        seen[expected_check(user)[0]] += 1
    assert set(seen) == set(instructions.KINDS) and min(seen.values()) > 300


def test_instruction_references_pass_and_wrong_forms_fail(instruction_bench):
    for it in instruction_bench:
        check = CHECKERS[it.check]
        assert check(it.reference, it.check_args), it.prompt
        for reply in ("", "  ", "...", "the " * 40):
            assert not check(reply, it.check_args), (it.check, reply)
        match it.check:
            case "one_word":
                assert not check(it.reference + " " + it.reference, it.check_args)
                assert not check("Yes.", it.check_args) and not check("No", it.check_args)
            case "yes_no":
                assert not check("Maybe.", it.check_args) and not check("Yes, it is.", {})
            case "list_n":
                items = re.split(r"\n|, and |, | and ", it.reference)
                assert len(items) == it.check_args["n"]
                assert not check(", ".join(items[1:]), it.check_args)
                assert not check(", ".join([*items, "extra"]), it.check_args)
            case "starts_with":
                assert not check("Hmm. The sky is blue.", it.check_args)
            case "all_caps":
                assert not check(it.reference.lower(), it.check_args)
                assert not check(it.reference.title(), it.check_args)
                assert not check("YES", it.check_args) and not check("PARIS.", it.check_args)
            case "repeat_word":
                assert not check(it.reference.rstrip(".") * 2, it.check_args)
                assert not check("I will say " + it.reference, it.check_args)
            case "contains_any":
                assert not check("Hmm hmm hmm.", it.check_args)
                assert check(it.reference.upper(), it.check_args)


def natural_variants(item) -> list[str]:
    """Other ways of writing the reference that a person would count as the same reply."""
    ref = item.reference
    variants = [ref.lower(), ref.upper(), ref.rstrip("."), f"  {ref}  ", f'"{ref}"']
    if item.check == "list_n":
        items = re.split(r"\n|, and |, | and ", ref)
        variants = [
            ", ".join(items),
            ", ".join(items[:-1]) + " and " + items[-1],
            "\n".join(items),
            "\n".join(f"{k}. {x}" for k, x in enumerate(items, 1)),
            "\n".join(f"- {x}" for x in items),
            ", ".join(x.title() for x in items) + ".",
        ]
    elif item.check == "starts_with":
        variants += [ref.replace("!", ",", 1), ref.replace("!", ".", 1)]
    elif item.check == "all_caps":
        variants = [ref.rstrip("."), ref.rstrip(".") + "!", f"  {ref}"]
    elif item.check in ("one_word", "yes_no", "repeat_word"):
        variants += [ref.strip(".") + "!", ref.strip(".").title()]
    return variants


def test_natural_variants_of_a_right_reply_pass(instruction_bench):
    for it in instruction_bench:
        for reply in natural_variants(it):
            assert CHECKERS[it.check](reply, it.check_args), (it.prompt, reply)


def test_no_constant_reply_passes_many_instructions(instruction_bench):
    constants = (
        "Yes", "No.", "YES", "DOG", "Paris", "Sure! The dog is a mammal.", "cat, dog and cow",
        "I do not know.", "the the the " * 30, "<|end|>", "",
    )  # fmt: skip
    for reply in constants:
        passed = sum(CHECKERS[i.check](reply, i.check_args) for i in instruction_bench)
        assert passed <= 0.2 * len(instruction_bench), (reply, passed)
    assert sum(CHECKERS[i.check]("", i.check_args) for i in instruction_bench) == 0
    # even the best constant (a one-word answer, or "yes") passes one kind, not two
    for reply in ("Yes", "DOG", "Paris", "No"):
        kinds = {i.check for i in instruction_bench if CHECKERS[i.check](reply, i.check_args)}
        assert len(kinds) <= 1, (reply, kinds)


def replies_made_from(prompt: str) -> dict[str, str]:
    """Replies a model could make by copying from the prompt instead of following it."""
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", prompt)
    named = re.search(r"the word (\w+)", prompt)
    word = named[1] if named else words[-1]
    return {
        "the prompt": prompt,
        "the prompt in capitals": prompt.upper(),
        "the prompt twice": f"{prompt}\n{prompt}",
        "Sure! and the prompt": f"Sure! {prompt}",
        "its first word": words[0],
        "its last word": words[-1],
        "its last three words": " ".join(words[-3:]),
        "its longest word": max(words, key=len),
        "the word it names": word,
        "the word it names, with !": f"{word}!",
        "Sure! and the word it names": f"Sure! {word}",
    }


def prompt_copy_scores(items) -> dict[str, float]:
    """The share of items each reply made from the prompt passes, not counting repeat_word
    items (copying a word is what those ask for), with the kinds it passes."""
    scores = {}
    for k, name in enumerate(replies_made_from("x")):
        passed = [
            i.check
            for i in items
            if i.check != "repeat_word"
            and CHECKERS[i.check](list(replies_made_from(i.prompt).values())[k], i.check_args)
        ]
        scores[name] = (len(passed) / len(items), set(passed))
    return scores


def test_no_reply_made_from_the_prompt_passes_many_instructions(instruction_bench):
    for name, (share, kinds) in prompt_copy_scores(instruction_bench).items():
        # never more than one kind, which is what a constant reply passes too (one kind: 0.14)
        assert len(kinds) <= 1 and share <= 0.15, (name, share, kinds)


def test_a_one_word_answer_is_not_a_word_of_the_question(instruction_bench, instruction_train):
    pairs = [
        (i.prompt, i.reference, i.check_args) for i in instruction_bench if i.check == "one_word"
    ]
    pairs += [
        (user, reply, expected_check(user, reply)[1])
        for user, reply in chat_pairs(instruction_train)
        if expected_check(user)[0] == "one_word"
    ]
    for prompt, reference, args in pairs:
        answer = reference.strip(".").casefold()
        assert CHECKERS["one_word"](reference.upper(), args)
        for word in set(re.findall(r"[a-z]+", prompt.casefold())) - {answer}:
            assert not CHECKERS["one_word"](word, args), (prompt, word)


def test_letter_lists_use_simple_everyday_words(kb):
    hard = {f.subject for r in ("element_symbol", "shape_sides") for f in kb.facts_for(r)}
    hard |= {f.subject for f in kb.facts_for("animal_group") if f.subject.endswith("s")}
    letters = [g for g in instructions._list_groups(kb) if g[0].startswith(("starts:", "ends:"))]
    words = {w for *_, found in letters for w in found}
    assert len(words) > 300
    assert not words & hard  # no "molybdenum", "lions" or "heptagon"
    assert all(3 <= len(w) <= 9 and w.isalpha() and w.islower() for w in words)


def test_list_sizes_take_turns_in_the_benchmark(instruction_bench):
    sizes = Counter(i.check_args["n"] for i in instruction_bench if i.check == "list_n")
    assert set(sizes) == set(instructions.LIST_SIZES)
    assert max(sizes.values()) - min(sizes.values()) <= 1


def test_instruction_wordings_read_well(instruction_bench, instruction_train):
    prompts = [i.prompt for i in instruction_bench] + [u for u, _ in chat_pairs(instruction_train)]
    for prompt in prompts:
        assert not re.search(r"\babout [a-z]+\.", prompt), prompt  # not "a sentence about leopard."
    natural = {
        "Sure",
        "Okay",
        "Hello",
        "Hi",
        "Great",
        "Alright",
        "Certainly",
        "Absolutely",
        "Of course",
    }
    for user, reply in chat_pairs(instruction_train):
        if expected_check(user)[0] == "starts_with":
            first, rest = reply.split("! ", 1)
            assert first in natural and rest[0].isupper(), reply


SAME_IN_THE_PLURAL = {"sheep", "deer", "moose", "bison", "salmon", "trout", "cod", "carp", "shrimp"}
SAME_IN_THE_PLURAL |= {"squid"}
IRREGULAR_S_FORMS = {  # beyond the grammar tests' IRREGULAR_PLURALS
    "goose": "geese", "wolf": "wolves", "potato": "potatoes", "tomato": "tomatoes",
    "mosquito": "mosquitoes", "mango": "mangoes",
}  # fmt: skip


def contains_forms(word: str) -> list[str]:
    """The forms of an asked word a sentence may use: the word, and its plural (or its he/she
    form, for a verb) where it has one."""
    relations = {f.relation for f in load_kb().facts_about(word)}
    if word[0].isupper():  # a name: only days have a plural ("Mondays")
        return [word, word + "s"] if "day_after" in relations else [word]
    if relations & {"element_symbol", "animal_group"}:  # elements; words already plural
        return [word]
    if word in SAME_IN_THE_PLURAL or word.endswith("fish"):
        return [word]
    return [word, IRREGULAR_S_FORMS.get(word) or plural_by_rule(word)]


def test_every_word_a_sentence_can_be_asked_to_use_takes_its_plural(kb):
    worlds = [w for pool in instructions._pools(kb) for w in pool["contains_any"]]
    assert len(worlds) > 700
    for world in worlds:
        (word,) = world.data
        forms = instructions._word_forms(kb, word)
        assert forms == contains_forms(word), (word, forms)
        args = {"words": forms, "min_words": 3}
        for form in forms:
            assert CHECKERS["contains_any"](f"I saw {form} today.", args), (word, form)
    known = {
        "cherry": "cherries", "canary": "canaries", "wolf": "wolves", "mouse": "mice",
        "goose": "geese", "child": "children", "potato": "potatoes", "fox": "foxes",
        "bus": "buses", "otter": "otters", "leaf": "leaves", "kangaroo": "kangaroos",
        "accept": "accepts", "buy": "buys", "freeze": "freezes",
    }  # fmt: skip
    for word, plural in known.items():
        assert instructions._word_forms(kb, word) == [word, plural], word
    for word in ("sheep", "Cuba", "sodium", "elephants", "goldfish"):
        assert instructions._word_forms(kb, word) == [word], word
    assert instructions._word_forms(kb, "Sunday") == ["Sunday", "Sundays"]


def facts_index(kb):
    """The statements and questions the knowledge base can write, with the facts they are about."""
    statements: dict[str, list] = {}
    questions: dict[str, list] = {}
    for fact in kb.facts:
        relation = kb.relations[fact.relation]
        for template in relation.train_templates:
            statements.setdefault(template.format(s=fact.subject, o=fact.obj), []).append(fact)
        for user, _ in relation.chat_templates:
            questions.setdefault(user.format(s=fact.subject), []).append(fact)
    return statements, questions


def is_false_claim(kb, statement: str) -> bool:
    """Whether the statement is one of the knowledge base's sentences with a wrong object."""
    for fact in kb.facts:
        if fact.subject not in statement:
            continue
        for wrong in kb.wrong_objects(fact):
            for template in kb.relations[fact.relation].train_templates:
                if template.format(s=fact.subject, o=wrong) == statement:
                    return True
    return False


YES_NO_FRAMES = (
    r"Is this true\? (?P<st>.*) Answer yes or no\.",
    r"(?P<st>.*) Is that right\? Answer yes or no\.",
    r"Answer yes or no\. Is this statement true\? (?P<st>.*)",
    r"Yes or no: is this true\? (?P<st>.*)",
)
COMPARE_FRAMES = (
    r"Is (?P<a>\d+) (?:bigger|larger) than (?P<b>\d+)\?.*",
    r"Answer yes or no: is (?P<a>\d+) greater than (?P<b>\d+)\?",
    r"Yes or no: is (?P<a>\d+) more than (?P<b>\d+)\?",
)


def listed_things(kb) -> dict[str, tuple[str, str]]:
    """What a list can ask for, as the words of the instruction, with its (relation, object)."""
    things = {phrase: key for key, phrase in instructions.LIST_THINGS.items()}
    things |= {f"countries in {o}": ("continent_of", o) for o in kb.objects_for("continent_of")}
    things |= {f"things that are {o}": ("color_of", o) for o in kb.objects_for("color_of")}
    return things


def check_list_reply(kb, user: str, reply: str) -> None:
    items = [i for i in re.split(r"\n|, and |, | and ", reply) if i]
    known = {*kb.subjects, *(f.obj for f in kb.facts)}
    assert len(set(items)) == len(items) and set(items) <= known, (user, reply)
    things = listed_things(kb)
    asked = max((t for t in things if t in user), key=len, default=None)
    if asked is not None:
        relation, obj = things[asked]
        assert set(items) <= {f.subject for f in kb.facts_for(relation) if f.obj == obj}, user
    else:
        letter = re.search(r"the letter (\w)", user)[1].lower()
        assert all((i[-1] if "end with" in user else i[0]) == letter for i in items), user


def test_training_replies_are_right_as_well_as_well_formed(kb, instruction_train):
    statements, questions = facts_index(kb)
    checked = Counter()
    for user, reply in chat_pairs(instruction_train):
        check, args = expected_check(user, reply)
        assert CHECKERS[check](reply, args), (user, reply)
        checked[check] += 1
        asked = [f for q, fs in questions.items() if q in user for f in fs]
        match check:
            case "yes_no":
                frame = next((m for f in YES_NO_FRAMES if (m := re.fullmatch(f, user))), None)
                if frame:  # a statement about a fact: yes if the knowledge base says so
                    truth = frame["st"] in statements
                    assert truth or is_false_claim(kb, frame["st"]), user
                else:  # a comparison of numbers
                    m = next(m for f in COMPARE_FRAMES if (m := re.fullmatch(f, user)))
                    truth = int(m["a"]) > int(m["b"])
                assert reply.strip(".").lower() == ("yes" if truth else "no"), (user, reply)
            case "one_word":
                assert any(reply.strip(".").lower() == f.obj.lower() for f in asked), (user, reply)
            case "starts_with" | "all_caps" | "contains_any":
                if check == "starts_with":
                    statement = reply.split("! ", 1)[1]
                    known = statements
                elif check == "all_caps":
                    statement = reply
                    known = {s.upper(): fs for s, fs in statements.items()}
                else:
                    statement = reply
                    known = statements
                assert statement in known, (user, reply)  # a true sentence of the knowledge base
                if check != "contains_any":  # and the question is about the same fact
                    assert set(asked) & set(known[statement]), (user, reply)
            case "list_n":
                check_list_reply(kb, user, reply)
            case "repeat_word":
                assert reply.strip(".") == args["word"]
    assert set(checked) == set(instructions.KINDS)


def test_yes_no_training_answers_are_balanced(instruction_train):
    answers = Counter()
    for user, reply in chat_pairs(instruction_train):
        if expected_check(user)[0] == "yes_no":
            answers[reply.strip(".")] += 1
    assert set(answers) == {"Yes", "No"}
    assert abs(answers["Yes"] - answers["No"]) < 0.1 * sum(answers.values())


def test_instruction_training_never_contains_a_benchmark_instruction(
    kb, instruction_bench, instruction_train
):
    bench_prompts = {i.prompt for i in instruction_bench}
    train_prompts = {user for user, _ in chat_pairs(instruction_train)}
    assert len(train_prompts) > 4000 and not train_prompts & bench_prompts
    # worlds and wordings: nothing used in training is reserved, nothing in the benchmark is not
    chats = instructions._train_chats(kb, skill_rng("instr", "train", "v2"), 1500)
    bench_worlds = {w.key for pool in instructions._pools(kb)[0].values() for w in pool}
    assert len(bench_worlds) > 600
    for chat in chats:
        assert len({e.world for e in chat}) == len(chat)  # no world twice in a chat
        for e in chat:
            assert not reserved_for_bench(e.world) and not reserved_for_bench(e.user)
            assert e.world not in bench_worlds and e.user not in bench_prompts
    assert all(
        reserved_for_bench(w.key) for pool in instructions._pools(kb)[0].values() for w in pool
    )
    assert not any(
        reserved_for_bench(w.key) for pool in instructions._pools(kb)[1].values() for w in pool
    )


def test_the_same_instruction_about_the_same_thing_is_never_in_both(
    kb, instruction_bench, instruction_train
):
    _, questions = facts_index(kb)
    things = listed_things(kb)
    claims = [  # (a statement template as a pattern, its relation)
        (re.compile(re.escape(t).replace(r"\{s\}", "(?P<s>.+)").replace(r"\{o\}", "(?P<o>.+)")), r)
        for r, relation in kb.relations.items()
        for t in relation.train_templates
    ]

    def claimed(statement: str) -> set:
        """The (subject, relation) a yes-or-no statement makes a claim about."""
        found = set()
        for pattern, relation in claims:
            m = pattern.fullmatch(statement)
            if m and any(f.subject == m["s"] for f in kb.facts_for(relation)):
                found.add((m["s"], relation))
        return found

    def about(kind: str, user: str) -> set:
        args = expected_check(user)[1]
        if kind == "repeat_word":
            return {args["word"]}
        if kind == "contains_any":
            return {args["words"][0]}
        if kind == "list_n":  # the group of things and how many
            letter = re.search(r"words that (start|end) with the letter (\w)", user)
            group = letter.groups() if letter else max((t for t in things if t in user), key=len)
            return {(group, args["n"])}
        if kind == "yes_no":
            number = next((m for f in COMPARE_FRAMES if (m := re.fullmatch(f, user))), None)
            if number:
                return {(int(number["a"]), int(number["b"]))}
            frame = next(m for f in YES_NO_FRAMES if (m := re.fullmatch(f, user)))
            found = claimed(frame["st"])
            assert found, user
            return found
        return {(f.subject, f.relation) for q, fs in questions.items() if q in user for f in fs}

    for kind in instructions.KINDS:
        in_bench = set().union(
            *(about(kind, i.prompt) for i in instruction_bench if i.check == kind)
        )
        in_train = set()
        for user, _ in chat_pairs(instruction_train):
            if expected_check(user)[0] == kind:
                in_train |= about(kind, user)
        assert len(in_bench) >= 15 and len(in_train) > 200 and not in_bench & in_train, kind


def test_instruction_generators_are_deterministic_and_validated(kb):
    def prompts(rng, n):
        return [i.prompt for i in instruction_bench_items(kb, rng, n)]

    a = prompts(skill_rng("instruction", "bench"), 30)
    assert a == prompts(skill_rng("instruction", "bench"), 30)
    assert a != prompts(skill_rng("instruction", "bench", "v2"), 30)
    docs = instruction_train_docs(kb, skill_rng("instr", "train"), 25)
    assert [d.turns for d in docs] == [
        d.turns for d in instruction_train_docs(kb, skill_rng("instr", "train"), 25)
    ]
    assert instruction_train_docs(kb, skill_rng("instr", "train"), 0) == []
    assert prompts(skill_rng("instruction", "bench"), 0) == []
    assert len(prompts(skill_rng("instruction", "bench"), 7)) == 7
    with pytest.raises(ValueError, match="cannot take"):
        prompts(skill_rng("instruction", "bench"), 100_000)


def test_instruction_chats_have_one_to_three_instructions(instruction_train):
    sizes = Counter(len(d.turns) // 2 for d in instruction_train)
    assert set(sizes) == {1, 2, 3}
    assert sizes[1] > sizes[2] > sizes[3] > 0.1 * len(instruction_train)
    for d in instruction_train:
        assert d.kind == "chat" and d.text == "" and d.turns[-1][0] == "ai"
        assert d.topic in TOPICS
        assert all(text.strip() == text and text for _, text in d.turns)


def test_instruction_gives_up_when_nothing_can_be_accepted(kb, monkeypatch):
    instructions._pools.cache_clear()
    pools = instructions._pools(kb)  # built with the real partition
    monkeypatch.setattr(instructions, "reserved_for_bench", lambda key: False)
    with pytest.raises(RuntimeError, match="instructions"):
        instruction_bench_items(kb, skill_rng("instruction", "bench"), 20)
    monkeypatch.setattr(instructions, "reserved_for_bench", lambda key: True)
    with pytest.raises(RuntimeError, match="instructions"):
        instruction_train_docs(kb, skill_rng("instr", "train"), 3)
    assert instructions._pools(kb) is pools


def test_list_groups_are_written_properly(kb):
    groups = instructions._list_groups(kb)
    assert len({g[0] for g in groups}) == len(groups) > 60
    for _, things, topic, found in groups:
        assert things == things.strip() and things.isascii() and found and topic in TOPICS
        assert len(set(found)) == len(found)
    letters = [g for g in groups if g[0].startswith(("starts:", "ends:"))]
    assert len(letters) > 20
    for group, _, _, found in letters:
        kind, letter = group.split(":")
        assert all((w[0] if kind == "starts" else w[-1]) == letter for w in found)
    assert instructions.LIST_THINGS[("animal_class", "fish")] == "fish"  # no "fishs"
    assert "mammals" in instructions.LIST_THINGS.values()
    assert "animals that live in a nest" in instructions.LIST_THINGS.values()


# ---------------------------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------------------------

IRREGULAR_PLURALS = {
    "man": "men",
    "woman": "women",
    "child": "children",
    "mouse": "mice",
    "leaf": "leaves",
}
IRREGULAR_PAST = {  # past, participle
    "run": ("ran", "run"), "swim": ("swam", "swum"), "eat": ("ate", "eaten"),
    "drink": ("drank", "drunk"), "sit": ("sat", "sat"), "go": ("went", "gone"),
    "fall": ("fell", "fallen"), "sing": ("sang", "sung"), "see": ("saw", "seen"),
    "write": ("wrote", "written"), "take": ("took", "taken"), "give": ("gave", "given"),
    "ride": ("rode", "ridden"), "break": ("broke", "broken"), "speak": ("spoke", "spoken"),
    "draw": ("drew", "drawn"), "throw": ("threw", "thrown"), "wear": ("wore", "worn"),
    "choose": ("chose", "chosen"), "grow": ("grew", "grown"), "sleep": ("slept", "slept"),
}  # fmt: skip


def plural_by_rule(noun: str) -> str:
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def third_by_rule(verb: str) -> str:
    if verb.endswith(("s", "x", "z", "ch", "sh", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def regular_past(verb: str) -> str:
    return verb + ("d" if verb.endswith("e") else "ed")


def test_plurals_and_verb_forms_are_spelled_right():
    assert len(grammar.PLURALS) == len(set(grammar.PLURALS.values())) >= 40
    for noun, plural in grammar.PLURALS.items():
        assert noun.isascii() and noun.islower() and plural == plural_by_rule(noun), noun
    assert not set(grammar.PEOPLE) & set(grammar.ANIMALS) & set(grammar.THINGS)
    for verb in grammar.VERBS:
        assert verb.third == third_by_rule(verb.base), verb
        if verb.base in IRREGULAR_PAST:
            assert (verb.past, verb.participle) == IRREGULAR_PAST[verb.base], verb
        else:
            assert verb.past == verb.participle == regular_past(verb.base), verb
        assert verb.who in ("p", "b") and len(verb.complements) >= 3
        assert len({verb.base, verb.third, verb.past}) == 3
    bases = [v.base for v in grammar.VERBS]
    assert len(set(bases)) == len(bases) >= 25
    assert len(grammar.PERFECT_VERBS) >= 15
    assert all(v.past != v.participle for v in grammar.PERFECT_VERBS)
    assert all(v.participle != v.base for v in grammar.PARTICIPLE_VERBS)
    assert len(set(grammar.NAMES)) == len(grammar.NAMES) >= 10


def starts_with_vowel_sound(word: str) -> bool:
    if word in {"hour", "honest", "heir", "honor"}:
        return True
    if word in {
        "unicorn", "university", "uniform", "union", "useful", "usual", "used", "unique", "one",
    }:  # fmt: skip
        return False
    return word[0] in "aeiou"


def test_article_words_are_sorted_by_sound():
    assert all(starts_with_vowel_sound(noun) for noun in grammar.VOWEL_NOUNS)
    assert not any(starts_with_vowel_sound(noun) for noun in grammar.CONSONANT_NOUNS)
    assert all(starts_with_vowel_sound(a) for a in grammar.VOWEL_ADJECTIVES)
    assert not any(starts_with_vowel_sound(a) for a in grammar.CONSONANT_ADJECTIVES)
    assert starts_with_vowel_sound("honest") and not any(
        starts_with_vowel_sound(w)
        for w in (*grammar.YOU_SOUND_WORDS, *grammar.YOU_SOUND_ADJECTIVES)
    )


@pytest.fixture(scope="module")
def language_bench():
    return grammar_pairs(skill_rng("language", "bench"))


@pytest.fixture(scope="module")
def big_language_bench():
    # a larger sample for the audits: 400 pairs a family
    return grammar_pairs(skill_rng("language", "bench"), 2000)


def words_of(sentence: str) -> list[str]:
    return sentence.rstrip(".?").split()


def test_language_bench_shape(language_bench):
    assert [p.id for p in language_bench] == [f"language-{k:04d}" for k in range(200)]
    for p in language_bench:
        assert isinstance(p, PairItem) and p.category == "language"
        assert len(p.tags) == 1 and p.tags[0].startswith("fam:")
        assert reserved_for_bench(p.good)
        for sentence in (p.good, p.bad):
            assert sentence.isascii() and sentence[0].isupper() and sentence[-1] in ".?"
            assert sentence == " ".join(sentence.split())
            assert sum(sentence.count(c) for c in ".?") == 1
        assert p.good[-1] == p.bad[-1]
    families = Counter(family_tag(p) for p in language_bench)
    assert families == {f: 40 for f in grammar.FAMILIES}
    assert len({p.good for p in language_bench}) == 200 == len({p.bad for p in language_bench})
    assert not {p.good for p in language_bench} & {p.bad for p in language_bench}
    # families take turns, so any prefix is as balanced as it can be
    assert [family_tag(p) for p in language_bench[:5]] == list(grammar.FAMILIES)
    assert max(Counter(family_tag(p) for p in language_bench[:50]).values()) == 10


def test_pairs_differ_in_one_word_or_in_order_and_nothing_else(language_bench):
    for p in language_bench:
        good, bad = words_of(p.good), words_of(p.bad)
        assert len(good) == len(bad)  # the same number of words
        if family_tag(p) == "word_order":
            assert sorted(good) == sorted(bad) and good != bad
        else:
            differing = [(g, b) for g, b in zip(good, bad) if g != b]
            assert len(differing) == 1, (p.good, p.bad)
        assert abs(len(p.good) - len(p.bad)) <= 4, (p.good, p.bad)  # about as long in letters
    gaps = [abs(len(p.good) - len(p.bad)) for p in language_bench]
    assert sum(gaps) / len(gaps) <= 2.0


def judge_agreement(sentence: str) -> bool:
    tokens = words_of(sentence)
    first = tokens[0]
    if first in ("He", "She", "It"):
        number, rest = "singular", tokens[1:]
    elif first in ("We", "They", "You", "I"):  # "I walk", "I have"
        number, rest = "plural", tokens[1:]
    elif first == "The":
        assert tokens[1] in grammar.PLURALS.values() or tokens[1] in grammar.PLURALS
        number = "plural" if tokens[1] in grammar.PLURALS.values() else "singular"
        rest = tokens[2:]
    else:
        assert first in grammar.NAMES
        number, rest = "singular", tokens[1:]
    if number == "singular":
        allowed = {"is", "was", "has"} | {v.third for v in grammar.VERBS}
    else:
        allowed = {"are", "were", "have"} | {v.base for v in grammar.VERBS}
    return rest[0] in allowed


def judge_plural(sentence: str) -> bool:
    tokens = words_of(sentence)
    (quantifier,) = [
        t for t in tokens if t in grammar.SINGULAR_QUANTIFIERS + grammar.PLURAL_QUANTIFIERS
    ]
    noun = tokens[tokens.index(quantifier) + 1]
    if quantifier in grammar.PLURAL_QUANTIFIERS:
        return noun in grammar.PLURALS.values()
    return noun in grammar.PLURALS


def judge_article(sentence: str) -> bool:
    tokens = words_of(sentence)
    (index,) = [k for k, t in enumerate(tokens) if t in ("a", "an")]
    return (tokens[index] == "an") == starts_with_vowel_sound(tokens[index + 1])


ALL_PAST = {v.past for v in grammar.VERBS}
ALL_BASE = {v.base for v in grammar.VERBS}
ALL_PARTICIPLE = {v.participle for v in grammar.VERBS}
ALL_VERB_FORMS = ALL_PAST | ALL_BASE | ALL_PARTICIPLE | {v.third for v in grammar.VERBS}


def judge_tense(sentence: str) -> bool:
    tokens = words_of(sentence)
    index = next(k for k, t in enumerate(tokens) if k >= 1 and t in ALL_VERB_FORMS)
    verb = tokens[index]
    if tokens[0] == "Did":
        return verb in ALL_BASE
    if tokens[0] in ("Yesterday", "Last"):
        return verb in ALL_PAST
    if tokens[0] in ("Tomorrow", "Next"):
        return tokens[index - 1] == "will" and verb in ALL_BASE
    return tokens[index - 1] in ("has", "have") and verb in ALL_PARTICIPLE


def test_the_good_sentence_is_right_and_the_bad_one_wrong(big_language_bench):
    judges = {
        "agreement": judge_agreement,
        "plural": judge_plural,
        "article": judge_article,
        "tense": judge_tense,
    }
    judged = Counter()
    for p in big_language_bench:
        judge = judges.get(family_tag(p))
        if judge:
            assert judge(p.good) and not judge(p.bad), (p.good, p.bad)
            judged[family_tag(p)] += 1
    assert set(judged) == set(judges) and min(judged.values()) == 400


def test_word_order_pairs_move_words_the_way_each_kind_does():
    adjectives = "|".join(grammar.ORDER_WORDS)
    verbs = "|".join(grammar.TRANSITIVE_PAST)
    prepositions = "|".join(sorted({prep for _, prep in grammar.PLACE_PHRASES}))
    helpers = "|".join(grammar.HELPERS)
    # kind -> (the good sentence, the bad one written from what the good one matched)
    kinds = {
        "adjective": (  # in the subject, where an adjective after its noun is never English
            rf"The ({adjectives}) (\w+) (is|was|sleeps|ran) (.+)\.",
            lambda m: f"The {m[2]} {m[1]} {m[3]} {m[4]}.",
        ),
        "object": (
            rf"The (\w+) ({verbs}) the (\w+)\.",
            lambda m: f"The {m[1]} the {m[3]} {m[2]}.",
        ),
        "preposition": (
            rf"The (\w+) is ({prepositions}) the (\w+)\.",
            lambda m: f"The {m[1]} is the {m[3]} {m[2]}.",
        ),
        "determiner": (
            r"(\w+) (\w+) the (\w+) (.+)\.",
            lambda m: f"{m[1]} {m[2]} {m[3]} the {m[4]}.",
        ),
        "helper": (
            rf"(.+?) ({helpers}) (\w+) (.+)\.",
            lambda m: f"{m[1]} {m[3]} {m[2]} {m[4]}.",
        ),
    }
    for kind, (pattern, bad_from) in kinds.items():
        pairs = [p for p in grammar._pairs() if (p.family, p.subtype) == ("word_order", kind)]
        assert len(pairs) > 500
        for p in pairs[:: max(1, len(pairs) // 300)]:
            found = re.fullmatch(pattern, p.good)
            assert found, (kind, p.good)
            assert p.bad == bad_from(found), (kind, p.good, p.bad)


def diff_features(sentence: str, other: str) -> set[str]:
    """The words of ``sentence`` that ``other`` lacks, as the word, its last letter and last two."""
    spare = words_of(other)
    features = set()
    for word in words_of(sentence):
        if word in spare:
            spare.remove(word)
        else:
            features |= {f"w:{word.lower()}", f"e1:{word[-1:].lower()}", f"e2:{word[-2:].lower()}"}
    return features


def predictor_accuracy(pairs) -> float:
    """Train on half the pairs and test on the other half, then swap: how often a model that
    knows only which words (and word endings) occur in the good and in the bad sentences of its
    training pairs says which sentence of a new pair is the good one."""
    score = 0.0
    for fold in (0, 1):
        counts: Counter[str] = Counter()
        for k, p in enumerate(pairs):
            if k % 2 != fold:
                counts.update({f: 1 for f in diff_features(p.good, p.bad)})
                counts.update({f: -1 for f in diff_features(p.bad, p.good)})
        for k, p in enumerate(pairs):
            if k % 2 == fold:
                good = sum(counts[f] for f in diff_features(p.good, p.bad))
                bad = sum(counts[f] for f in diff_features(p.bad, p.good))
                score += 1.0 if good > bad else 0.5 if good == bad else 0.0
    return score / len(pairs)


def test_no_word_ending_or_length_tells_the_bad_sentence(big_language_bench):
    for family in grammar.FAMILIES:
        pairs = [p for p in big_language_bench if family_tag(p) == family]
        assert len(pairs) == 400
        shorter = sum(
            1.0 if len(p.good) < len(p.bad) else 0.5 if len(p.good) == len(p.bad) else 0.0
            for p in pairs
        ) / len(pairs)
        assert 0.4 <= shorter <= 0.6, (family, "good is shorter", shorter)
        assert predictor_accuracy(pairs) <= 0.6, (family, predictor_accuracy(pairs))


def test_no_single_word_marks_the_bad_sentence(big_language_bench):
    for family in grammar.FAMILIES:
        in_bad: Counter[str] = Counter()
        in_good: Counter[str] = Counter()
        for p in (q for q in big_language_bench if family_tag(q) == family):
            in_bad.update(f for f in diff_features(p.bad, p.good) if f.startswith("w:"))
            in_good.update(f for f in diff_features(p.good, p.bad) if f.startswith("w:"))
        for word in {*in_bad, *in_good}:
            total = in_bad[word] + in_good[word]
            if total >= 20:  # a word that matters in the family
                assert 0.25 <= in_bad[word] / total <= 0.75, (family, word, in_bad[word], total)


def test_the_mistake_goes_both_ways(big_language_bench):
    pairs = {f: [p for p in big_language_bench if family_tag(p) == f] for f in grammar.FAMILIES}

    def share(family: str, test) -> float:
        return sum(test(p) for p in pairs[family]) / len(pairs[family])

    def only_in(sentence: str, other: str) -> set[str]:
        return set(words_of(sentence)) - set(words_of(other))

    # article: "an" is the good word as often as "a"
    assert 0.45 <= share("article", lambda p: " an " in f" {p.good} ") <= 0.55
    # agreement: a singular verb form is the good one as often as the bad one
    singular = {"is", "was", "has"} | {v.third for v in grammar.VERBS}
    assert 0.45 <= share("agreement", lambda p: bool(only_in(p.good, p.bad) & singular)) <= 0.55
    # plural: the plural noun is the good one in half of the pairs
    plurals = set(grammar.PLURALS.values())
    assert 0.45 <= share("plural", lambda p: bool(only_in(p.good, p.bad) & plurals)) <= 0.55
    # tense: the past form is as often the good one as the bad one; so are the plain form and the
    # participle (a form that only occurs in the bad sentence would give the answer away)
    for name, forms in (("past", ALL_PAST), ("plain", ALL_BASE), ("participle", ALL_PARTICIPLE)):
        good_has = share("tense", lambda p, f=forms: bool(only_in(p.good, p.bad) & f))
        bad_has = share("tense", lambda p, f=forms: bool(only_in(p.bad, p.good) & f))
        assert abs(good_has - bad_has) <= 0.06, (name, good_has, bad_has)


def test_tense_forms_are_balanced_by_the_weights_not_by_luck():
    # over every benchmark-reserved pair, weighted as the benchmark draws them: a word that is a
    # past form (or a plain form, or a participle) is as often only in the good sentence as only
    # in the bad one, so no sample of the benchmark leans on it
    pools = grammar._bench_pools()
    weights = grammar.SUBTYPE_WEIGHTS["tense"]

    def only_in(sentence: str, other: str) -> set[str]:
        return set(words_of(sentence)) - set(words_of(other))

    for name, forms in (("past", ALL_PAST), ("plain", ALL_BASE), ("participle", ALL_PARTICIPLE)):
        gap = 0.0
        for subtype, weight in weights.items():
            pool = pools[("tense", subtype)]
            good = sum(bool(only_in(p.good, p.bad) & forms) for p in pool) / len(pool)
            bad = sum(bool(only_in(p.bad, p.good) & forms) for p in pool) / len(pool)
            gap += weight * (good - bad)
        assert abs(gap) <= 0.01, (name, gap)


def test_large_language_benches_can_be_drawn_with_any_seed():
    # a good sentence two families share must not leave a family short of pairs it was given
    for k in range(4):
        pairs = grammar_pairs(skill_rng("language", "bench", f"seed{k}"), 2000)
        assert len(pairs) == 2000 and len({p.good for p in pairs}) == 2000


def test_article_pairs_cover_vowels_consonants_adjectives_and_odd_spellings(language_bench):
    pairs = [p for p in language_bench if family_tag(p) == "article"]
    odd = [
        p
        for p in pairs
        if re.search(r"\ban? (?:honest|hour|honor|useful|used|unique|uni\w+)", p.good)
    ]
    assert len(odd) >= 0.15 * len(pairs)
    nouns = {*grammar.VOWEL_NOUNS, *grammar.CONSONANT_NOUNS}
    adjective_pairs = mismatch = 0
    for p in pairs:
        tokens = words_of(p.good)
        index = next(k for k, t in enumerate(tokens) if t in ("a", "an"))
        following = tokens[index + 1 : index + 3]
        if len(following) == 2 and following[1] in nouns:  # "an old dog", "a big apple"
            adjective_pairs += 1
            mismatch += starts_with_vowel_sound(following[0]) != starts_with_vowel_sound(
                following[1]
            )
    assert adjective_pairs >= 0.5 * len(pairs)
    # the noun's sound does not decide: the adjective does, in most of those pairs
    assert mismatch >= 0.8 * adjective_pairs


# Independent of the module's tables: adjectives that only living things can be, adjectives that
# only things can be, and the nouns that are animals.
LIVING_ONLY = {"angry", "unhappy", "excited", "happy", "kind", "hungry", "sleepy", "brave", "young"}
THINGS_ONLY = {"empty", "open", "extra", "round", "new", "blue", "icy", "fresh", "heavy", "used"}
ANIMAL_NOUNS = {
    *grammar.ANIMALS, "owl", "ant", "insect", "otter", "octopus", "elephant", "eagle", "ostrich",
    "animal", "bird", "fish", "unicorn",
}  # fmt: skip
PEOPLE_THINGS = {"a book", "a pen", "a kite", "a bike", "a hat", "a boat", "a cake", "a bag"}
PEOPLE_THINGS |= {"a car", "a new coat", "a big house", "three books", "two cats", "two dogs"}


def adjective_phrases():
    """(adjective, noun) of every article and adjective-order pair, from the good sentence."""
    for p in grammar._pairs():
        tokens = words_of(p.good)
        if p.family == "article":
            k = next(i for i, t in enumerate(tokens) if t in ("a", "an"))
            if len(tokens) > k + 2 and tokens[k + 1] not in ("honest",):
                yield tokens[k + 1], tokens[k + 2], p.good
        elif (p.family, p.subtype) == ("word_order", "adjective"):
            yield tokens[1], tokens[2], p.good


def test_adjectives_go_with_nouns_they_can_describe():
    seen = 0
    for adjective, noun, sentence in adjective_phrases():
        if noun not in grammar.PLURALS and noun not in {
            *grammar.VOWEL_NOUNS,
            *grammar.CONSONANT_NOUNS,
        }:
            continue  # "a useful tool": a noun of the odd-spelling phrases
        seen += 1
        if adjective in LIVING_ONLY:
            assert noun in ANIMAL_NOUNS, sentence  # not "an unhappy cup"
        if adjective in THINGS_ONLY:
            assert noun not in ANIMAL_NOUNS, sentence  # not "an empty dog", "a new owl"
    assert seen > 5000
    text = " ".join(p.good for p in grammar._pairs())
    assert not re.search(r"\ba usual\b", text)  # "a usual cat" is not English anyone says


def test_people_and_animals_have_and_sit_where_they_can():
    small_places = re.compile(r"\b(?:in the (?:box|bag)|(?:on|under) the (?:table|bed|chair))\b")
    big = {"horse", "cow", "pig", "bear", "lion", "goat", "man", "woman", "teacher", "farmer"}
    big |= {"doctor", "friend"}
    for p in grammar._pairs():
        tokens = words_of(p.good)
        if p.family == "plural" and tokens[:2] == ["She", "has"]:
            noun = tokens[3]
            assert noun not in grammar.PEOPLE and noun not in grammar.PEOPLE.values(), p.good
        if p.subtype in ("have_singular", "have_plural"):
            subject = tokens[1] if tokens[0] == "The" else None
            if subject in grammar.ANIMALS or subject in grammar.ANIMALS.values():
                obj = " ".join(tokens[3:])
                assert obj not in PEOPLE_THINGS, p.good  # not "The pig has a kite."
        if (p.family, p.subtype) == ("word_order", "preposition") and small_places.search(p.good):
            assert tokens[1] not in big, p.good  # not "The horse is in the bag."
        if (p.family, p.subtype) == ("word_order", "determiner") and tokens[:2] == ["She", "has"]:
            assert tokens[3] not in grammar.PEOPLE, p.good  # not "She has the teacher at school."


def test_tense_sentences_say_one_thing_about_time():
    # "Yesterday Tom walked every day." and "Last night the boy slept all day." are not things
    # anyone says: a one-time past or future frame never takes a habit or a whole day
    habit = re.compile(r"\b(?:every day|every night|all day)\b")
    grows = re.compile(r"\b(?:grow|grew|grown)\b")
    frames = 0
    for p in grammar._pairs():
        if p.family == "tense" and re.match(r"(?:Yesterday|Last|Tomorrow|Next)\b", p.good):
            frames += 1
            assert not habit.search(p.good), p.good
            if grows.search(p.good):  # growing takes a year, not a week
                assert re.match(r"(?:Last|Next) (?:year|summer)\b", p.good), p.good
    assert frames > 50_000


ANIMAL_WORDS = set(grammar.ANIMALS) | set(grammar.ANIMALS.values())
WILD = {"lion", "lions", "bear", "bears", "fox", "foxes", "monkey", "monkeys", "frog", "frogs"}
PEOPLE_ONLY = {"tea", "juice", "lunch", "dinner", "breakfast", "school"}  # for an animal subject
PET_THINGS = {"bed", "ball", "toy", "name"}  # not for a wild animal


def test_good_sentences_are_things_people_say():
    nouns = set(grammar.PLURALS) | set(grammar.PLURALS.values())
    checked = 0
    for p in grammar._pairs():
        tokens = words_of(p.good)
        assert not re.search(r"\bthe friends?\b", p.good, re.IGNORECASE), p.good  # whose friend?
        if p.family == "plural":  # not "I see every car at home.", "Tom drew each man at school."
            assert not {"each", "every"} & set(tokens), p.good
        if p.family in ("agreement", "tense") or p.subtype == "helper":
            found = [k for k, t in enumerate(tokens) if t in nouns]
            if not found:
                continue  # a name or a pronoun
            subject, rest = tokens[found[0]], tokens[found[0] + 1 :]
            checked += 1
            assert subject not in rest, p.good  # not "The dog sees the dog."
            if subject in ANIMAL_WORDS:
                assert not PEOPLE_ONLY & set(rest), p.good  # not "Last week the dog drank tea."
            if subject in WILD:
                assert not PET_THINGS & set(rest), p.good  # not "The lion has a bed."
    assert checked > 30_000


def test_language_pairs_are_simple_and_spelt_right(big_language_bench):
    text = " ".join(p.good + " " + p.bad for p in big_language_bench)
    misspelt = re.compile(
        r"\b(?:childs|mans|womans|mouses|foots|babys|boxs|buss|dishs|peachs|leafs|knifes)\b",
        re.IGNORECASE,
    )
    assert not misspelt.search(text)
    vocabulary = {w.lower() for w in re.findall(r"[A-Za-z]+", text)}
    assert len(vocabulary) < 700  # common words only
    assert max(len(w) for w in vocabulary) <= 11


def test_language_pairs_are_deterministic_and_validated():
    a = [(p.good, p.bad) for p in grammar_pairs(skill_rng("language", "bench"), 40)]
    assert a == [(p.good, p.bad) for p in grammar_pairs(skill_rng("language", "bench"), 40)]
    assert a != [(p.good, p.bad) for p in grammar_pairs(skill_rng("language", "bench", "v2"), 40)]
    assert grammar_pairs(skill_rng("language", "bench"), 0) == []
    small = grammar_pairs(skill_rng("language", "bench"), 12)
    counts = Counter(family_tag(p) for p in small)
    assert sum(counts.values()) == 12 and max(counts.values()) - min(counts.values()) <= 1
    assert len({p.good for p in small}) == 12
    with pytest.raises(ValueError, match="cannot take"):
        grammar_pairs(skill_rng("language", "bench"), 1_000_000)


def test_every_pair_of_the_space_is_a_valid_pair():
    space: dict[tuple[str, str], list] = {}
    for p in grammar._pairs():
        space.setdefault((p.family, p.subtype), []).append(p)
    assert set(grammar.SUBTYPE_WEIGHTS) == set(grammar.FAMILIES)
    for family, weights in grammar.SUBTYPE_WEIGHTS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-3, family
        assert {sub for fam, sub in space if fam == family} == set(weights)
    goods, bads = set(), set()
    for (family, subtype), pairs in space.items():
        for p in pairs:
            assert p.family == family and p.subtype == subtype and p.good != p.bad
            goods.add(p.good)
            bads.add(p.bad)
    assert not goods & bads  # no sentence is good in one pair and bad in another
    assert len(goods) > 50_000


def test_allocate_splits_in_proportion_and_respects_capacity():
    allocate = grammar._allocate
    assert allocate(10, {"a": 0.5, "b": 0.5}, {"a": 10, "b": 10}) == {"a": 5, "b": 5}
    assert allocate(10, {"a": 0.5, "b": 0.5}, {"a": 2, "b": 10}) == {"a": 2, "b": 8}
    shares = {"a": 1 / 3, "b": 0.1, "c": 0.5667}
    assert sum(allocate(40, shares, dict.fromkeys("abc", 99)).values()) == 40
    assert allocate(1, {"a": 0.2, "b": 0.8}, {"a": 5, "b": 5}) == {"a": 0, "b": 1}
    assert allocate(0, {"a": 1.0}, {"a": 5}) == {"a": 0}
    with pytest.raises(ValueError, match="cannot take"):
        allocate(11, {"a": 0.5, "b": 0.5}, {"a": 5, "b": 5})
    # the families share items the way fair_quota says
    assert fair_quota({"a": 3, "b": 40}, 20) == {"a": 3, "b": 17}
