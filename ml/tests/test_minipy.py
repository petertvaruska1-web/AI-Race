import ast
import random
from pathlib import Path

import pytest

import airace_ml.minipy.interpreter as interpreter_module
from airace_ml.minipy.interpreter import call_function, run_program


@pytest.mark.parametrize("src,out", [
    ("x = 3\ny = x + 4\nprint(y)", "7\n"),
    ("t = 0\nfor i in range(5):\n    t += i\nprint(t)", "10\n"),
    ("def sq(n):\n    return n * n\nprint(sq(6))", "36\n"),
    ("xs = [1, 2]\nxs.append(5)\nprint(xs[2], len(xs))", "5 3\n"),
    ("s = 'ab'\nprint(s.upper() + 'c')", "ABc\n"),
    ("n = 7\nif n % 2 == 0:\n    print('even')\nelif n > 5:\n    print('big')\nelse:\n    print('odd')", "big\n"),
    ("i = 0\nwhile True:\n    i += 1\n    if i == 3:\n        break\nprint(i)", "3\n"),
    ("print(max([3, 9, 2]), min(4, 1), sum([1, 2, 3]), abs(-2), str(5) + '!')", "9 1 6 2 5!\n"),
])
def test_programs(src, out):
    r = run_program(src); assert r.error is None and r.stdout == out

@pytest.mark.parametrize("src,err", [
    ("while True:\n    pass", "StepLimit"), ("import os", "Unsupported: Import"),
    ("x = 1\nx.__class__", "Unsupported: Attribute"), ("open('f')", "NameError: open"),
    ("exec('1')", "NameError: exec"), ("def f(n):\n    return f(n)\nf(1)", "RecursionLimit"),
    ("x = 2 ** 10", "Unsupported: Pow"), ("x = 999999 * 999999 * 999999", "Overflow"),
    ("print(1 // 0)", "ZeroDivisionError"), ("def (:", "SyntaxError"),
    ("for i in range(100000):\n    print('spam spam spam')", "OutputLimit"),
])
def test_sandbox_errors(src, err):
    assert run_program(src, max_steps=200_000).error == err

def test_call_function():
    assert call_function("def add(a, b):\n    return a + b", "add", [2, 3]) == (5, None)
    assert call_function("def f(x):\n    return x[5]", "f", [[1]])[1] == "IndexError"

def test_partial_stdout_kept_on_error():
    r = run_program("print('a')\nprint(1 // 0)"); assert r.stdout == "a\n" and r.error == "ZeroDivisionError"


# ---- controller rulings and edge behaviour -------------------------------------------------

@pytest.mark.parametrize("src,err", [
    ("x = 7 / 2", "Unsupported: Div"),
    ("x = 1 << 3", "Unsupported: LShift"),
    ("x = 1 | 2", "Unsupported: BitOr"),
    ("x = 1\nx **= 2", "Unsupported: Pow"),
    ("x = +1", "Unsupported: UAdd"),
    ("x = ~1", "Unsupported: Invert"),
    ("x = 1 is None", "Unsupported: Is"),
    ("x = 1 not in [2]", "Unsupported: NotIn"),
    ("import os", "Unsupported: Import"),
    ("from os import path", "Unsupported: ImportFrom"),
    ("f = lambda: 1", "Unsupported: Lambda"),
    ("x = [i for i in range(3)]", "Unsupported: ListComp"),
    ("x = (1, 2)", "Unsupported: Tuple"),
    ("a, b = 1, 2", "Unsupported: Tuple"),
    ("x = {}", "Unsupported: Dict"),
    ("x = 1.5", "Unsupported: Constant"),
    ("x = f'a'", "Unsupported: JoinedStr"),
    ("class A:\n    pass", "Unsupported: ClassDef"),
    ("try:\n    pass\nexcept Exception:\n    pass", "Unsupported: Try"),
    ("del x", "Unsupported: Delete"),
    ("assert True", "Unsupported: Assert"),
    ("with x:\n    pass", "Unsupported: With"),
    ("global x", "Unsupported: Global"),
    ("x = [1][::2]", "Unsupported: Slice"),
    ("def f(a=1):\n    pass", "Unsupported: FunctionDef"),
    ("def f(*a):\n    pass", "Unsupported: FunctionDef"),
    ("def f():\n    def g():\n        pass", "Unsupported: FunctionDef"),
    ("print('a', end='')", "Unsupported: keyword"),
])
def test_unsupported_constructs(src, err):
    # "f()" is appended where a def needs to actually run its body
    program = src + "\nf()" if src.startswith("def f():") else src
    assert run_program(program).error == err

@pytest.mark.parametrize("src", [
    "xs = [1]\nxs.append",
    "x = 'a'.split",
    "x = 'a'\nx.split(',')",
    "xs = [1]\nxs.pop()",
    "xs = [1]\nxs.upper()",
    "s = 'a'\ns.append('b')",
    "x = 1\nx.__class__()",
    "x = 1\nx.real = 2",
])
def test_attribute_access_is_only_the_three_methods(src):
    assert run_program(src).error == "Unsupported: Attribute"

@pytest.mark.parametrize("src,err", [
    ("x = 1 + 'a'", "TypeError"),
    ("x = 'a' - 'b'", "TypeError"),
    ("x = 'a' % 1", "TypeError"),
    ("x = '%s' % 1", "TypeError"),
    ("x = [1] + 'ab'", "TypeError"),
    ("x = None + 1", "TypeError"),
    ("x = -'a'", "TypeError"),
    ("x = 1 < 'a'", "TypeError"),
    ("x = len(5)", "TypeError"),
    ("x = len()", "TypeError"),
    ("x = abs('a')", "TypeError"),
    ("x = max([])", "TypeError"),
    ("x = max(5)", "TypeError"),
    ("x = sum(['a'])", "TypeError"),
    ("x = int('abc')", "TypeError"),
    ("x = int(None)", "TypeError"),
    ("x = range(0, 5, 0)", "TypeError"),
    ("x = range('a')", "TypeError"),
    ("x = sorted([1, 'a'])", "TypeError"),
    ("x = 5\nx(1)", "TypeError"),
    ("x = 5\nx[0]", "TypeError"),
    ("x = 'abc'\nx[0] = 'z'", "TypeError"),
    ("x = [1]\nx['a']", "TypeError"),
    ("x = 1 in 5", "TypeError"),
    ("x = 1 in 'abc'", "TypeError"),
    ("def f(a):\n    return a\nf()", "TypeError"),
    ("def f(a):\n    return a\nf(1, 2)", "TypeError"),
    ("x = [1].append(1, 2)", "TypeError"),
    ("x = 'a'.upper(1)", "TypeError"),
    ("x = range(3)\nx[0]", "TypeError"),
    ("x = range(3) + range(3)", "TypeError"),
    ("return 1", "SyntaxError"),
    ("break", "SyntaxError"),
    ("continue", "SyntaxError"),
    ("def f():\n    break\nwhile True:\n    f()", "SyntaxError"),
])
def test_type_errors(src, err):
    assert run_program(src).error == err

@pytest.mark.parametrize("src,err", [
    ("x = [1, 2]\nx[2]", "IndexError"),
    ("x = [1, 2]\nx[-3]", "IndexError"),
    ("x = 'ab'\nx[5]", "IndexError"),
    ("x = [1]\nx[1] = 3", "IndexError"),
    ("x = [1]\nx[1] += 3", "IndexError"),
    ("x = 5 % 0", "ZeroDivisionError"),
    ("x = 5\nx //= 0", "ZeroDivisionError"),
    ("print(undefined_name)", "NameError: undefined_name"),
    ("x += 1", "NameError: x"),
    ("def f():\n    return y\nf()", "NameError: y"),
])
def test_runtime_errors(src, err):
    assert run_program(src).error == err

def test_negative_index_slices_and_strings():
    r = run_program("xs = [1, 2, 3, 4]\ns = 'hello'\n"
                    "print(xs[-1], s[1], xs[1:3], s[:2], s[-3:], xs[9:], xs[:-1])")
    assert r.error is None
    assert r.stdout == "4 e [2, 3] he llo [] [1, 2, 3]\n"

def test_print_formatting_matches_python_str():
    r = run_program("print()\nprint(None, True, False, 'x', [1, 'a', [True, None]])\nprint('')")
    assert r.error is None
    assert r.stdout == "\nNone True False x [1, 'a', [True, None]]\n\n"

def test_print_of_self_containing_list_terminates():
    r = run_program("xs = [1]\nxs.append(xs)\nprint(xs)")
    assert r.error is None and r.stdout == "[1, [...]]\n"

def test_bools_behave_like_python_ints():
    r = run_program("print(True + True, True * 5, -True, abs(False), sum([True, True]), True == 1)")
    assert r.error is None and r.stdout == "2 5 -1 0 2 True\n"

def test_comparisons_and_boolean_operators():
    r = run_program("a = 1\nprint(1 < a + 1 < 3, 1 < 2 < 2, [1, 2] < [1, 3], 'a' < 'b', 2 in [1, 2], 'b' in 'abc')\n"
                    "print(0 or 'x', 5 and 6, None or None, not [], not 'a', 1 if a == 2 else 9)")
    assert r.error is None
    assert r.stdout == "True False True True True True\nx 6 None True False 9\n"

def test_short_circuit_does_not_evaluate_the_dead_branch():
    r = run_program("x = 1 or boom\ny = 0 and boom\nz = 5 if True else boom\nprint(x, y, z)")
    assert r.error is None and r.stdout == "1 0 5\n"

def test_builtins_conversions_and_collections():
    r = run_program("print(int('42') + 1, int(True), int(), str(), str(None), str([1]), list('ab'), list(range(3)))\n"
                    "print(sorted([3, 1, 2]), sorted('ba'), len('abc'), len(range(10)), sum(range(5)), max('abc'))\n"
                    "print(list(), max(2, 8, 5), min([4, 2, 9]), abs(-7), 'abc'.lower(), 'aBc'.upper())")
    assert r.error is None
    assert r.stdout == ("43 1 0  None [1] ['a', 'b'] [0, 1, 2]\n"
                        "[1, 2, 3] ['a', 'b'] 3 10 10 c\n"
                        "[] 8 2 7 abc ABC\n")

def test_loops_else_continue_and_live_list_iteration():
    r = run_program(
        "t = 0\nfor i in range(10):\n    if i % 2 == 1:\n        continue\n    if i > 6:\n        break\n    t += i\n"
        "else:\n    t = -1\nprint(t)\n"
        "n = 3\nwhile n > 0:\n    n -= 1\nelse:\n    print('done', n)\n"
        "for c in 'abc':\n    print(c)\n"
        "xs = [1]\nfor x in xs:\n    if len(xs) < 4:\n        xs.append(x + 1)\nprint(xs)")
    assert r.error is None
    assert r.stdout == "12\ndone 0\na\nb\nc\n[1, 2, 3, 4]\n"

def test_functions_scope_recursion_and_shadowing():
    src = (
        "g = 10\n"
        "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n"
        "def read_global():\n    return g\n"
        "def shadow():\n    g = 1\n    return g\n"
        "def noreturn():\n    pass\n"
        "def early(xs):\n    for x in xs:\n        if x > 1:\n            return x\n    return -1\n"
        "print(fact(10), read_global(), shadow(), g, noreturn(), early([1, 2, 3]), early([]))\n"
        "len = 5\nprint(len)"
    )
    r = run_program(src)
    assert r.error is None and r.stdout == "3628800 10 1 10 None 2 -1\n5\n"

def test_reading_a_name_that_is_local_later_does_not_leak_the_global():
    r = run_program("x = 1\ndef f():\n    y = x\n    x = 2\n    return y\nf()")
    assert r.error == "NameError: x"

def test_augmented_list_assignment_mutates_in_place_like_python():
    r = run_program("a = [1]\nb = a\nb += [2]\nb *= 2\nxs = [0, 0]\nxs[1] += 5\nxs[0] = 9\nprint(a, xs)")
    assert r.error is None and r.stdout == "[1, 2, 1, 2] [9, 5]\n"

def test_function_values_are_first_class_but_not_data():
    r = run_program("def f():\n    return 1\nfs = [f]\nprint(fs[0]())\nx = print")
    assert r.error is None and r.stdout == "1\n"
    assert call_function("def f():\n    return f", "f", []) == (None, "TypeError")

def test_steps_are_counted_and_capped():
    r = run_program("x = 1")
    assert r.error is None and r.steps == 2        # Assign + Constant
    r = run_program("while True:\n    pass", max_steps=50)
    assert r.error == "StepLimit" and r.steps == 50
    assert run_program("for i in range(10):\n    pass", max_steps=1000).steps > 10

def test_huge_ranges_are_lazy_and_costed():
    assert run_program("for i in range(999999999999):\n    break").error is None
    assert run_program("x = sum(range(999999999999))").error == "StepLimit"
    assert run_program("x = max(range(999999999999))").error == "StepLimit"
    assert run_program("x = list(range(999999999999))").error == "Overflow"
    assert run_program("x = sorted(range(2000))").error == "Overflow"
    assert run_program("x = 5 in range(999999999999)").error is None

def test_integer_overflow_boundaries():
    assert run_program("print(999999999999 + 1)").stdout == "1000000000000\n"
    assert run_program("x = 999999999999 + 2").error == "Overflow"
    assert run_program("x = -999999999999 - 2").error == "Overflow"
    assert run_program("x = 1000000000000 * 1000000000000").error == "Overflow"
    assert run_program("x = 99999999999999999999").error == "Overflow"
    assert run_program("x = 1000000000001").error == "Overflow"
    assert run_program("x = int('1' * 40)").error == "Overflow"
    assert run_program("x = sum([1000000000000, 1])").error == "Overflow"

def test_sequence_size_limits():
    assert run_program("x = 'a' * 1000\nprint(len(x))").stdout == "1000\n"
    assert run_program("x = 'a' * 1001").error == "Overflow"
    assert run_program("x = 1001 * 'a'").error == "Overflow"
    assert run_program("x = 'a' * 1000 + 'b'").error == "Overflow"
    assert run_program("x = [0] * 1000\nprint(len(x))").stdout == "1000\n"
    assert run_program("x = [0] * 1001").error == "Overflow"
    assert run_program("x = [0] * 500 + [1] * 501").error == "Overflow"
    assert run_program("x = [0] * 1000\nx.append(1)").error == "Overflow"
    assert run_program("x = [0] * 999\nx.append(1)\nprint(len(x))").stdout == "1000\n"
    assert run_program("x = 'a' * 1000000000000").error == "Overflow"
    assert run_program("x = '' * 1000000000000\nprint(len(x))").stdout == "0\n"
    assert run_program("x = [0] * 600\nx += [0] * 600").error == "Overflow"
    assert run_program("x = 'ab' * 500\ny = str(x)\nprint(len(y))").stdout == "1000\n"
    assert run_program("x = [1] * 1000\ny = str(x)").error == "Overflow"
    assert run_program("x = [0] * 1000\nx *= 2").error == "Overflow"
    assert run_program("x = '" + "a" * 1001 + "'").error == "Overflow"
    assert run_program("x = [" + ",".join(["1"] * 1001) + "]").error == "Overflow"

def test_doubling_a_list_does_not_blow_up_memory_or_output():
    src = "x = [0]\nfor i in range(40):\n    x = [x, x]\nprint(x)"
    assert run_program(src, max_steps=100_000).error == "OutputLimit"
    assert run_program("x = [0]\nfor i in range(40):\n    x = [x, x]\ny = str(x)", max_steps=100_000).error == "Overflow"

def test_comparing_huge_shared_structures_is_charged_to_the_step_budget():
    build = "a = [0]\nb = [0]\nfor i in range(40):\n    a = [a, a]\n    b = [b, b]\n"
    assert run_program(build + "print(a == b)").error == "StepLimit"
    assert run_program(build + "print(a < b)").error == "StepLimit"
    assert run_program(build + "print(a in [b])").error == "StepLimit"
    assert run_program(build + "print(max([a, b]))").error == "StepLimit"
    assert run_program(build + "print(a == a)").stdout == "True\n"
    cyclic = "x = [1]\nx.append(x)\ny = [1]\ny.append(y)\nprint(x == y)"
    assert run_program(cyclic).error in {"StepLimit", "RecursionLimit"}

def test_comparing_values_across_types():
    r = run_program("print(1 == 'a', [1] == 1, None == None, [[1], 'a'] == [[1], 'a'], [1, 2] != [1, 2])\n"
                    "print(sorted([[2, 'a'], [1], [1, 5]]), max(['b', 'a']), min([[3], [2, 9]]), True < 2)")
    assert r.error is None
    assert r.stdout == "False False True True False\n[[1], [1, 5], [2, 'a']] b [2, 9] True\n"
    assert run_program("x = [1, 'a'] < [1, 2]").error == "TypeError"
    assert run_program("x = None < None").error == "TypeError"
    assert run_program("x = print < print").error == "TypeError"

def test_output_limit_keeps_the_stdout_produced_before_the_offending_print():
    r = run_program("print('abc')\nprint('defg')\nprint('hi')", max_output_chars=9)
    assert r.error == "OutputLimit" and r.stdout == "abc\ndefg\n"
    r = run_program("print('abc')\nprint('defg')", max_output_chars=9)
    assert r.error is None and r.stdout == "abc\ndefg\n"
    r = run_program("for i in range(100000):\n    print('spam spam spam')", max_steps=200_000)
    assert r.error == "OutputLimit" and 0 < len(r.stdout) <= 2_000
    assert run_program("print(1)", max_output_chars=0).error == "OutputLimit"
    assert run_program("print('a', 'b', 'c')", max_output_chars=5).error == "OutputLimit"

def test_recursion_limit_is_fifty_calls():
    deep = "def f(n):\n    if n == 0:\n        return 0\n    return 1 + f(n - 1)\n"
    assert call_function(deep, "f", [49]) == (49, None)
    assert call_function(deep, "f", [50]) == (None, "RecursionLimit")
    assert run_program(deep + "print(f(40))").stdout == "40\n"

def test_mutual_and_complex_recursion_fits_in_the_python_stack():
    src = (
        "def walk(n):\n    t = 0\n    for i in range(2):\n        while t < 1:\n            if n > 0:\n"
        "                t += walk(n - 1)\n            else:\n                t += 1\n    return t\n"
    )
    assert call_function(src, "walk", [45], max_steps=1_000_000)[1] is None

def test_hostile_nesting_never_raises():
    deep_sum = "x = " + " + ".join(["1"] * 3000)
    assert run_program(deep_sum, max_steps=10 ** 9).error in {"SyntaxError", "RecursionLimit"}
    assert run_program("x = " + "(" * 500 + "1" + ")" * 500).error == "SyntaxError"
    assert run_program("x = [" * 500).error == "SyntaxError"
    assert run_program("x = " + "-" * 5000 + "1", max_steps=10 ** 9).error in {"SyntaxError", "RecursionLimit"}
    medium = "x = " + " + ".join(["1"] * 400)
    assert run_program(medium + "\nprint(x)", max_steps=10 ** 9).stdout == "400\n"

@pytest.mark.parametrize("src", [
    "def (:", "x = (", "if True print(1)", "\x00", "x = 1\x00", "print('a'", "1 +", "   x = 1", "else:\n    pass",
    "x = " + "9" * 5000, "x = '\\", "\ufeffx = 1\n$",
])
def test_unparseable_source_is_syntax_error(src):
    r = run_program(src)
    assert r.error == "SyntaxError" and r.stdout == "" and r.steps == 0

def test_empty_and_comment_only_programs_are_fine():
    for src in ["", "\n", "# nothing", "x = 1  # trailing"]:
        r = run_program(src)
        assert r.error is None and r.stdout == ""

def test_no_python_warnings_leak_out_of_parsing(recwarn):
    r = run_program("print('\\d')")
    assert r.error is None and r.stdout == "\\d\n"
    assert len(recwarn) == 0

def test_non_string_inputs_do_not_raise():
    assert run_program(None).error is not None  # type: ignore[arg-type]
    assert run_program(5).error is not None  # type: ignore[arg-type]
    assert call_function(5, "f", [])[1] is not None  # type: ignore[arg-type]
    assert call_function("def f():\n    return 1", "f", None)[1] is not None  # type: ignore[arg-type]

def test_unicode_text_round_trips():
    r = run_program("print('héllo', '日本語', 'café'.upper())")
    assert r.error is None and r.stdout == "héllo 日本語 CAFÉ\n"
    assert run_program("x = 'ß' * 600\ny = x.upper()").error == "Overflow"

@pytest.mark.parametrize("name,args,expected", [
    ("echo", [[1, "a", [True, None]]], ([1, "a", [True, None]], None)),
    ("echo", [None], (None, None)),
    ("echo", ["héllo"], ("héllo", None)),
    ("echo", [True], (True, None)),
    ("echo", [2.5], (None, "TypeError")),
    ("echo", [(1, 2)], (None, "TypeError")),
    ("echo", [{"a": 1}], (None, "TypeError")),
    ("echo", [[1, 2.5]], (None, "TypeError")),
    ("echo", [10 ** 13], (None, "Overflow")),
    ("echo", ["a" * 1001], (None, "Overflow")),
    ("echo", [], (None, "TypeError")),
    ("missing", [1], (None, "NameError: missing")),
    ("len", ["a"], (None, "NameError: len")),
    ("value", [], (None, "TypeError")),
])
def test_call_function_values_and_errors(name, args, expected):
    src = "value = 3\ndef echo(x):\n    return x\n"
    assert call_function(src, name, args) == expected

def test_call_function_does_not_alias_the_callers_list():
    data = [1, 2]
    value, err = call_function("def f(xs):\n    xs.append(3)\n    return xs", "f", [data])
    assert err is None and value == [1, 2, 3] and data == [1, 2]

def test_call_function_reports_module_body_errors_and_limits():
    assert call_function("def f():\n    return 1\nboom", "f", []) == (None, "NameError: boom")
    assert call_function("def (:", "f", []) == (None, "SyntaxError")
    assert call_function("def f():\n    while True:\n        pass", "f", [], max_steps=500) == (None, "StepLimit")
    assert call_function("def f(n):\n    return f(n)", "f", [1]) == (None, "RecursionLimit")
    # module-level code runs first and its prints do not leak or break the call
    assert call_function("print('hi')\ndef f():\n    return 7", "f", []) == (7, None)
    spam = "def f():\n    for i in range(1000):\n        print('spam spam spam')\n    return 1"
    assert call_function(spam, "f", [], max_steps=100_000) == (None, "OutputLimit")


_FUZZ_SEEDS = [
    "x = 3\ny = x + 4\nprint(y)",
    "t = 0\nfor i in range(5):\n    t += i\nprint(t)",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nprint(fib(9))",
    "xs = [1, 2]\nxs.append(5)\nprint(xs[2], len(xs), xs[0:2], xs[-1], sorted(xs), 2 in xs)",
    "s = 'ab'\nprint(s.upper() + 'c', s * 3, s[1], s[:1], max(s), int('7'), list(s))",
    "i = 0\nwhile True:\n    i += 1\n    if i == 3:\n        break\nprint(i, not i, i or 0)",
]
_FUZZ_PIECES = [
    "(", ")", "[", "]", ":", ",", "+", "-", "*", "//", "%", "==", "<", "=", "+=", "x", "xs", "0", "1",
    "999999", "'a'", "None", "True", "not", "and", "in", "if", "else", "for", "while", "def", "return",
    "break", "continue", "range", "print", "len", "\n", "\n    ", " ", "lambda", ".", "**", "/", "import",
    "'", "\\", "\x00", "[0]*1000", ";", "{", "}",
]
_FIXED_ERRORS = {"SyntaxError", "StepLimit", "OutputLimit", "RecursionLimit", "Overflow",
                 "ZeroDivisionError", "TypeError", "IndexError"}

def test_mutated_programs_never_raise_and_only_report_fixed_errors():
    rng = random.Random(1234)
    for _ in range(4000):
        chars = list(rng.choice(_FUZZ_SEEDS))
        for _ in range(rng.randint(1, 4)):
            pos = rng.randint(0, len(chars))
            if chars and rng.random() < 0.4:
                del chars[min(pos, len(chars) - 1)]
            else:
                chars[pos:pos] = list(rng.choice(_FUZZ_PIECES))
        src = "".join(chars)
        r = run_program(src, max_steps=3000, max_output_chars=300)
        assert r.error is None or r.error in _FIXED_ERRORS or r.error.startswith(
            ("NameError: ", "Unsupported: ")), (src, r.error)
        assert len(r.stdout) <= 300 and 0 <= r.steps <= 3000, src
        value, error = call_function(src, "fib", [3], max_steps=3000)
        assert value is None or error is None, src


# ---- the interpreter file must stay a closed box ---------------------------------------------

def test_interpreter_never_uses_python_execution_or_io():
    tree = ast.parse(Path(interpreter_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    assert imported <= {"__future__", "ast", "collections", "dataclasses", "functools", "warnings"}
    assert not called & {"exec", "eval", "compile", "open", "input", "__import__", "getattr", "setattr",
                         "globals", "locals", "vars", "system", "popen"}
    assert "sys" not in imported and "os" not in imported

def test_hostile_escape_attempts_all_fail_safely():
    attempts = [
        "print(().__class__.__bases__)",
        "print(__import__('os').system('echo hi'))",
        "x = [].__class__",
        "print(open('C:/Windows/win.ini').read())",
        "print(globals())",
        "print(getattr(1, 'real'))",
        "print(eval('1 + 1'))",
        "print(compile('1', 'f', 'eval'))",
        "print(__builtins__)",
        "print(print.__self__)",
        "exec('print(1)')",
        "print(type(1))",
        "print(chr(65))",
        "print(input())",
        "print('{0.__class__}'.format(1))",
        "print('%s' % 1)",
        "x = 1\nprint(x.__class__.__mro__)",
    ]
    for src in attempts:
        r = run_program(src)
        assert r.error is not None, src
        assert r.stdout == "", src
