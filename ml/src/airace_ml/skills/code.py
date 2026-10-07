"""Code as text: MiniPy programs for training and held-out benchmark items.

Two kinds of text, both written in MiniPy (the safe Python subset of ``airace_ml.minipy``):

- *Output prediction*: ``Program:\\n<code>\\nOutput: <line>``. The program prints exactly one line.
  Six families of tiny programs over small ints and short words (each tagged ``fam:<name>`` in
  benchmarks): ``straight`` (assignments and arithmetic), ``loop`` (for and while), ``if`` (if,
  elif, else), ``string`` (joining, repeating, reversing, indexing), ``list`` (append, sum, sorted,
  loops over lists) and ``call`` (a ``def`` and a call). The answer is whatever the interpreter
  prints, so it is right by construction.
- *Function completion*: ``def name(args):\\n    \\"\\"\\"doc\\"\\"\\"\\n``, to be continued with a
  body. Ten families (``add double square max2 is_even first last count_of sum_list
  maybe_negate``), each written in many surface forms (function name, parameter names, wording
  of the docstring and the style of the body). A benchmark item is judged by running the reply
  against hidden tests (``checkers.minipy_function_tests``); a training document is the function,
  its body and two usage examples.

Measurement. The printed line must be evaluated, never read off the program: a program whose
output is one of its own literals (compared without case, as the benchmark compares answers) or
text that sits inside the program is not used, so copying or guessing a literal never works.
The ``.upper()`` and ``.lower()`` programs only occur in training, because answers are compared
without case. Every function family has tests that no trivial body (``return None``, a constant, a
parameter) passes.

Partition. A program belongs to a *world*: its computation, whatever its variable names (the code
with its names replaced by ``v0, v1, ...``). Benchmark programs use only reserved worlds whose
prompt is also reserved (:func:`airace_ml.skills.types.reserved_for_bench`); training text uses
neither. So no benchmark program, nor the same computation under other names, occurs in training
text. A function belongs to the world of its ``def`` line (its name and parameters), whatever its
docstring: a benchmark function's def line is reserved and occurs nowhere in training text, which
teaches every family under other names and parameters (and never shows a reserved header either).

Program worlds are drawn at random with a pooled budget of 50 draws per requested program; function
headers are listed up front (64-80 def lines per family, at least 5 of them reserved, times 6
docstrings). Nothing here is a model output; these are data generators.
"""

import ast
import operator
from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import cache

import numpy as np

from airace_ml.minipy.interpreter import run_program
from airace_ml.skills.types import (
    CheckItem,
    ExactItem,
    TextDoc,
    choice,
    fair_quota,
    in_turn,
    pick,
    reserved_for_bench,
)

TOPIC = "technology"
OUTPUT_FAMILIES = ("straight", "loop", "if", "string", "list", "call")
FUNCTION_CHECK = "minipy_function_tests"
PROGRAM_SHARE = 0.5  # share of training documents that are output-prediction programs
MAX_ATTEMPTS_PER_ITEM = 50  # world draws
RENDER_TRIES = 40  # variable namings tried for an accepted world
MAX_ANSWER = 150  # the largest number a program prints
MAX_ANSWER_CHARS = 24
OUTPUT_TOKENS = 16  # new tokens a benchmark gives the model for a printed line
FUNCTION_TOKENS = 64  # ... and for a function body
N_EXAMPLES = 2  # usage examples after a training function

NAME_POOLS = {
    "int": (
        *"abcdemnpqrstuvwxyz",
        "total",
        "count",
        "result",
    ),
    "loop": ("i", "j", "k", "n"),
    "str": ("s", "t", "w", "a", "b", "x", "y", "word", "text", "name", "msg"),
    "char": ("c", "ch", "letter", "x"),
    "list": ("xs", "ys", "nums", "items", "data", "values", "lst", "a", "b"),
    "func": ("f", "g", "h", "calc", "go", "fn"),
}
WORDS = (
    "cat", "dog", "sun", "hat", "bus", "fox", "pen", "cup", "bed", "pig", "hen", "bee",
    "fish", "bird", "tree", "moon", "star", "book", "ball", "frog", "duck", "milk",
    "rain", "snow", "wind", "lake", "cake", "door", "shoe", "ship", "corn", "rice",
    "bean", "nut", "egg", "jam", "pie", "tea", "ham", "fan", "map", "net", "box", "key",
    "bag", "toy", "kid", "boat", "road", "farm", "hill", "rock", "sand", "leaf", "rose",
    "seed", "lamp", "desk", "sock", "coat",
)  # fmt: skip
_OPS: dict[str, Callable[[int, int], int]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
}
_OP_CHOICES = ("+", "+", "-", "-", "*", "*", "//", "%")


def _flip(rng: np.random.Generator) -> int:
    return int(rng.integers(2))


def _lit(rng: np.random.Generator, low: int, high: int) -> int:
    """A random int in ``[low, high]``."""
    return int(rng.integers(low, high + 1))


def _plus(rng: np.random.Generator, high: int) -> str:
    """`` + d`` for a random ``d`` in ``1..high``, or nothing (about 1 time in ``high + 1``)."""
    d = _lit(rng, 0, high)
    return f" + {d}" if d else ""


# --- output prediction: program templates --------------------------------------------------------


@dataclass(frozen=True)
class _Draft:
    """A program with its variable names left open: ``{0}``, ``{1}``... in the lines.

    ``kinds[i]`` names the pool (``NAME_POOLS``) that placeholder ``i`` is named from.
    """

    lines: tuple[str, ...]
    kinds: tuple[str, ...]

    def render(self, names: list[str]) -> str:
        return "\n".join(self.lines).format(*names)

    @property
    def world(self) -> str:
        """The code under canonical names ``v0, v1, ...``: what the program computes."""
        return self.render([f"v{i}" for i in range(len(self.kinds))])

    def name(self, rng: np.random.Generator) -> list[str]:
        """A random naming; no two variables share a name."""
        used: set[str] = set()
        names: list[str] = []
        for kind in self.kinds:
            options = [n for n in NAME_POOLS[kind] if n not in used]
            names.append(choice(rng, options))
            used.add(names[-1])
        return names


class _Builder:
    """Collects the lines of a draft; ``var(kind)`` returns a new placeholder to write into them."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.kinds: list[str] = []

    def var(self, kind: str) -> str:
        self.kinds.append(kind)
        return f"{{{len(self.kinds) - 1}}}"

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def draft(self) -> _Draft:
        return _Draft(tuple(self.lines), tuple(self.kinds))


Template = Callable[[np.random.Generator, int], _Draft]


def _straight_arith(rng: np.random.Generator, control: int) -> _Draft:
    """2-3 ints, 1-2 new variables from them, then print one."""
    b = _Builder()
    values: list[tuple[str, int]] = []
    for _ in range(2 + _flip(rng)):
        placeholder, value = b.var("int"), _lit(rng, 2, 12)
        b.add(f"{placeholder} = {value}")
        values.append((placeholder, value))

    def combine() -> tuple[str, int]:
        for _ in range(30):
            op = choice(rng, _OP_CHOICES)
            left, x = choice(rng, values)
            if _flip(rng):
                right, y = choice(rng, values)
            else:
                y = _lit(rng, 2, 5)
                right = str(y)
            if y == 0 and op in ("//", "%"):
                continue
            value = _OPS[op](x, y)
            if 0 <= value <= MAX_ANSWER:
                return f"{left} {op} {right}", value
        left, x = values[0]
        return f"{left} + 2", x + 2

    for _ in range(1 + _flip(rng)):
        expression, value = combine()
        placeholder = b.var("int")
        b.add(f"{placeholder} = {expression}")
        values.append((placeholder, value))
    shown = values[-1][0]
    if _flip(rng):
        shown, _ = combine()
    b.add(f"print({shown})")
    return b.draft()


def _loop(rng: np.random.Generator, control: int) -> _Draft:
    b = _Builder()
    total, i = b.var("int"), b.var("loop")
    match control:
        case 0:  # add up a range
            low = _lit(rng, 0, 3)
            added = (i, i, f"{i} + {_lit(rng, 1, 3)}")[pick(rng, 3)]
            b.add(f"{total} = {_lit(rng, 0, 9)}")
            b.add(f"for {i} in range({low}, {low + _lit(rng, 3, 8)}):")
            b.add(f"    {total} = {total} + {added}")
        case 1:  # add up a range with a step
            low = _lit(rng, 0, 3)
            b.add(f"{total} = {_lit(rng, 0, 9)}")
            b.add(f"for {i} in range({low}, {low + _lit(rng, 7, 14)}, {_lit(rng, 2, 4)}):")
            b.add(f"    {total} += {i}")
        case 2:  # a multiple of each number
            low = _lit(rng, 1, 3)
            b.add(f"{total} = {_lit(rng, 0, 5)}")
            b.add(f"for {i} in range({low}, {low + _lit(rng, 3, 6)}):")
            b.add(f"    {total} = {total} + {i} * {_lit(rng, 2, 5)}")
        case 3:  # multiply by each number, or by a fixed one
            low = _lit(rng, 1, 3)
            factor = (i, i, str(_lit(rng, 2, 3)))[pick(rng, 3)]
            b.add(f"{total} = {_lit(rng, 1, 4)}")
            b.add(f"for {i} in range({low}, {low + _lit(rng, 2, 4)}):")
            b.add(f"    {total} = {total} * {factor}{_plus(rng, 2)}")
        case 4:  # doubling and more
            b.add(f"{total} = {_lit(rng, 1, 5)}", f"for {i} in range({_lit(rng, 2, 6)}):")
            b.add(f"    {total} = {total} * {_lit(rng, 2, 3)}{_plus(rng, 2)}")
        case 5:  # a while loop that grows
            b.add(f"{total} = {_lit(rng, 1, 5)}", f"while {total} < {_lit(rng, 20, 99)}:")
            b.add(f"    {total} = {total} * {_lit(rng, 2, 3)}")
        case 6:  # a while loop that steps up
            b.add(f"{total} = {_lit(rng, 0, 5)}", f"while {total} < {_lit(rng, 10, 60)}:")
            b.add(f"    {total} = {total} + {_lit(rng, 3, 8)}")
        case 7:  # how many halvings (the counter is printed)
            by = _lit(rng, 2, 3)
            b.add(f"{total} = {_lit(rng, 8, 500)}", f"{i} = 0", f"while {total} > 1:")
            b.add(f"    {total} = {total} // {by}", f"    {i} = {i} + 1")
            total = i
        case 8:  # add up the numbers that pass a test
            mod = _lit(rng, 2, 4)
            op = ("==", "!=")[_flip(rng)]
            added = (i, "1")[_flip(rng)]
            b.add(f"{total} = 0", f"for {i} in range({_lit(rng, 6, 15)}):")
            b.add(f"    if {i} % {mod} {op} {_lit(rng, 0, mod - 1)}:")
            b.add(f"        {total} = {total} + {added}")
        case _:  # nested loops
            j = b.var("loop")
            added = (f"{i} + {j}", f"{i} * {j}", "1")[pick(rng, 3)]
            b.add(f"{total} = {_lit(rng, 0, 5)}", f"for {i} in range({_lit(rng, 2, 5)}):")
            b.add(f"    for {j} in range({_lit(rng, 2, 5)}):")
            b.add(f"        {total} = {total} + {added}")
    b.add(f"print({total})")
    return b.draft()


def _if(rng: np.random.Generator, control: int) -> _Draft:
    b = _Builder()
    x, y = b.var("int"), b.var("int")
    match control:
        case 0:  # two branches
            b.add(f"{x} = {_lit(rng, 2, 14)}", f"if {x} > {_lit(rng, 4, 10)}:")
            b.add(
                f"    {y} = {x} * {_lit(rng, 2, 3)}", "else:", f"    {y} = {x} + {_lit(rng, 2, 6)}"
            )
        case 1:  # even or odd
            b.add(f"{x} = {_lit(rng, 3, 99)}", f"if {x} % 2 == 0:", f"    {y} = {x} // 2")
            b.add("else:", f"    {y} = {x} * {_lit(rng, 2, 3)}{_plus(rng, 2)}")
        case 2:  # a chain
            b.add(f"{x} = {_lit(rng, 2, 18)}", f"if {x} < {_lit(rng, 5, 8)}:")
            b.add(f"    {y} = {x} + {_lit(rng, 2, 9)}", f"elif {x} < {_lit(rng, 10, 14)}:")
            b.add(
                f"    {y} = {x} * {_lit(rng, 2, 3)}", "else:", f"    {y} = {x} - {_lit(rng, 2, 5)}"
            )
        case 3:  # the difference of two numbers
            z = b.var("int")
            b.add(f"{x} = {_lit(rng, 2, 20)}", f"{z} = {_lit(rng, 2, 25)}", f"if {x} > {z}:")
            b.add(f"    {y} = {x} - {z}", "else:", f"    {y} = {z} - {x}")
        case _:  # an if inside an if
            b.add(f"{x} = {_lit(rng, 2, 16)}", f"if {x} > {_lit(rng, 4, 8)}:")
            b.add(f"    if {x} > {_lit(rng, 9, 12)}:", f"        {y} = {x} + {_lit(rng, 2, 4)}")
            b.add("    else:", f"        {y} = {x} + {_lit(rng, 5, 9)}", "else:")
            b.add(f"    {y} = {x} - {_lit(rng, 2, 3)}")
    b.add(f"print({y})")
    return b.draft()


def _letters(rng: np.random.Generator, length: int) -> str:
    """Letters that read like a word: consonants and vowels in turn."""
    consonants, vowels = "bdfgklmnprstvz", "aeiou"
    start = _flip(rng)
    return "".join(
        choice(rng, tuple(vowels if (k + start) % 2 else consonants)) for k in range(length)
    )


def _string(rng: np.random.Generator, control: int) -> _Draft:
    b = _Builder()
    word, other = choice(rng, WORDS), choice(rng, WORDS)
    s, t = b.var("str"), b.var("str")
    # ``.upper()`` and ``.lower()`` programs are for training only: answers are compared without
    # case, so a benchmark could not tell them from copying
    match control:
        case 0:  # join two words
            b.add(f'{s} = "{word}"', f'{t} = "{other}"', f"print({s} + {t})")
        case 1:  # join with a space
            b.add(f'{s} = "{word}"', f'{t} = "{other}"', f'print({s} + " " + {t})')
        case 2:  # repeat
            if _flip(rng):
                b.add(f'{s} = "{word[:3]}"', f"print({s} * {_lit(rng, 2, 4)})")
            else:
                b.add(f'{s} = "{word[:3]}"', f'{t} = "{other[:3]}"')
                b.add(f"print({s} + {t} * {_lit(rng, 2, 3)})")
        case 3:  # reverse by a loop
            c = b.var("char")
            b.add(f'{s} = ""', f'for {c} in "{_letters(rng, _lit(rng, 3, 5))}":')
            b.add(f"    {s} = {c} + {s}", f"print({s})")
        case 4:  # lengths
            if _flip(rng):
                b.add(f'{s} = "{word}"', f'{t} = "{other}"', f"print(len({s}) + len({t}))")
            else:
                b.add(f'{s} = "{word}"', f"print(len({s}) * {_lit(rng, 2, 4)})")
        case 5:  # letters by position
            end = choice(rng, ("1", "2", "3", "-1"))
            b.add(f'{s} = "{word}"', f"print({s}[{_lit(rng, 0, 2)}] + {s}[{end}])")
        case _:  # training only: case
            method = ("upper", "lower")[_flip(rng)]
            b.add(f'{s} = "{word}"', f'{t} = "{other}"', f"print({s}.{method}() + {t})")
    return b.draft()


def _list(rng: np.random.Generator, control: int) -> _Draft:
    b = _Builder()
    xs = b.var("list")
    items = ", ".join(str(_lit(rng, 1, 9)) for _ in range(_lit(rng, 3, 5)))
    if control < 5:
        b.add(f"{xs} = [{items}]")
    match control:
        case 0:  # append, then add up
            b.add(f"{xs}.append({_lit(rng, 1, 9)})", f"print(sum({xs}))")
        case 1:  # the spread
            b.add(f"print(max({xs}) - min({xs}))")
        case 2:  # the ends
            b.add(f"print({xs}[0] + {xs}[-1])")
        case 3:  # sorted
            b.add(f"print(sorted({xs}))")
        case 4:  # count the large ones
            n, v = b.var("int"), b.var("loop")
            b.add(f"{n} = 0", f"for {v} in {xs}:", f"    if {v} > {_lit(rng, 3, 6)}:")
            b.add(f"        {n} = {n} + 1", f"print({n})")
        case _:  # build a list in a loop
            i = b.var("loop")
            low = _lit(rng, 0, 2)
            b.add(f"{xs} = []", f"for {i} in range({low}, {low + _lit(rng, 3, 5)}):")
            b.add(f"    {xs}.append({i} * {_lit(rng, 2, 5)}{_plus(rng, 3)})", f"print({xs})")
    return b.draft()


def _call(rng: np.random.Generator, control: int) -> _Draft:
    b = _Builder()
    f, x = b.var("func"), b.var("int")
    match control:
        case 0:  # a formula
            b.add(f"def {f}({x}):", f"    return {x} * {_lit(rng, 2, 5)}{_plus(rng, 9)}")
            b.add(f"print({f}({_lit(rng, 2, 15)}))")
        case 1:  # two parameters
            y = b.var("int")
            formula = (f"{x} * {y} - {x}", f"{x} * {y} + {y}", f"{x} + {y} * 2", f"{x} * 2 + {y}")
            b.add(f"def {f}({x}, {y}):", f"    return {choice(rng, formula)}")
            b.add(f"print({f}({_lit(rng, 2, 12)}, {_lit(rng, 2, 9)}))")
        case 2:  # an if inside
            b.add(f"def {f}({x}):", f"    if {x} > {_lit(rng, 4, 8)}:")
            b.add(f"        return {x} - {_lit(rng, 2, 4)}", f"    return {x} + {_lit(rng, 3, 6)}")
            b.add(f"print({f}({_lit(rng, 2, 14)}))")
        case 3:  # a loop inside
            t, i = b.var("int"), b.var("loop")
            added = (i, f"{i} * 2", "1")[pick(rng, 3)]
            b.add(f"def {f}({x}):", f"    {t} = {_lit(rng, 0, 4)}", f"    for {i} in range({x}):")
            b.add(
                f"        {t} = {t} + {added}", f"    return {t}", f"print({f}({_lit(rng, 3, 12)}))"
            )
        case _:  # two calls
            b.add(f"def {f}({x}):", f"    return {x} * {_lit(rng, 2, 4)}")
            join = ("+", "-")[_flip(rng)]
            b.add(f"print({f}({_lit(rng, 3, 12)}) {join} {f}({_lit(rng, 2, 9)}))")
    return b.draft()


_TEMPLATES: dict[str, Template] = {
    "straight": _straight_arith,
    "loop": _loop,
    "if": _if,
    "string": _string,
    "list": _list,
    "call": _call,
}


@dataclass(frozen=True)
class _Program:
    family: str
    world: str
    code: str
    answer: str

    @property
    def prompt(self) -> str:
        return f"Program:\n{self.code}\nOutput:"


def _literals(code: str) -> list[object]:
    """The int and string literals of a program."""
    return [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and type(node.value) in (int, str)
    ]


def _copies(answer: str, code: str) -> bool:
    """Whether the printed line can be read off the program: it is a literal or sits inside it.

    Answers are compared without case, so this is too. A short line (a number) is compared to
    the literals only, since a digit is inside many numbers.
    """
    line = answer.casefold()
    if any(line == str(literal).casefold() for literal in _literals(code)):
        return True
    return len(line) >= 3 and line in code.casefold()


def _acceptable(code: str, stdout: str) -> str | None:
    """The answer line if the run printed exactly one fitting line that is not in the program."""
    lines = stdout.split("\n")
    if len(lines) != 2 or lines[1] != "" or not lines[0].strip():
        return None
    answer = lines[0].strip()
    if len(answer) > MAX_ANSWER_CHARS or _copies(answer, code):
        return None
    if answer.lstrip("-").isdigit() and not 0 <= int(answer) <= MAX_ANSWER:
        return None
    return answer


class _ProgramDrawer:
    """Draws programs of one split until one is accepted, from a pooled budget of world draws."""

    def __init__(self, rng: np.random.Generator, n_items: int, *, bench: bool) -> None:
        self.rng = rng
        self.bench = bench
        self.left = MAX_ATTEMPTS_PER_ITEM * n_items

    def draw(self, family: str, control: int, avoid: Collection[str] = ()) -> _Program:
        """A program of ``family`` (``control`` picks its template) in this drawer's split.

        Its world is not in ``avoid``. Raises ``RuntimeError`` when the budget runs out.
        """
        while self.left > 0:
            self.left -= 1
            draft = _TEMPLATES[family](self.rng, control)
            world = draft.world
            if reserved_for_bench(world) != self.bench or world in avoid:
                continue
            for _ in range(RENDER_TRIES):
                code = draft.render(draft.name(self.rng))
                if reserved_for_bench(f"Program:\n{code}\nOutput:") == self.bench:
                    break
            else:
                continue
            result = run_program(code)
            answer = _acceptable(code, result.stdout) if result.error is None else None
            if answer is not None:
                return _Program(family, world, code, answer)
        raise RuntimeError(f"code: could not draw enough {family} programs within the budget")


# --- function completion: families -------------------------------------------------------------

LOCAL_NAMES = (  # names a body may give its own variables, in order of preference
    ("count", "total", "n", "c", "result", "t"),
    ("x", "item", "e", "y", "value", "v"),
)


@dataclass(frozen=True)
class _Family:
    """One function to write, in every surface form it is shown in.

    ``docs`` and ``bodies`` are templates over ``{p0}``, ``{p1}`` (the parameters) and, in bodies,
    ``{l0}``, ``{l1}`` (variables of the body's own). ``bodies[0]`` is the reference. ``ref``
    defines what the function does; ``tests`` are the hidden calls, with the values ``ref``
    gives; ``example`` draws a call for a usage example, and ``contrast`` says whether two
    examples together are worth showing (a yes/no function shows both answers).
    """

    name: str
    names: tuple[str, ...]
    params: tuple[tuple[str, ...], ...]
    docs: tuple[str, ...]
    bodies: tuple[str, ...]
    ref: Callable[..., object]
    tests: tuple[tuple[object, ...], ...]
    example: Callable[[np.random.Generator], tuple[object, ...]]
    # whether two usage examples together show the function well (both answers of a yes/no)
    contrast: Callable[[tuple, tuple], bool] = lambda a, b: True

    @property
    def test_cases(self) -> list[list[object]]:
        return [[list(args), self.ref(*args)] for args in self.tests]


def _ints(rng: np.random.Generator, low: int, high: int, size: int) -> list[int]:
    return [_lit(rng, low, high) for _ in range(size)]


def _two_numbers(rng: np.random.Generator) -> tuple[object, ...]:
    return (_lit(rng, 0, 20), _lit(rng, 0, 20))


def _one_number(rng: np.random.Generator) -> tuple[object, ...]:
    return (_lit(rng, 0, 20),)


def _a_list(rng: np.random.Generator) -> tuple[object, ...]:
    return (_ints(rng, 0, 20, _lit(rng, 2, 5)),)


def _list_and_item(rng: np.random.Generator) -> tuple[object, ...]:
    return (_ints(rng, 1, 5, _lit(rng, 4, 7)), _lit(rng, 1, 5))


def _number_and_flag(rng: np.random.Generator) -> tuple[object, ...]:
    return (_lit(rng, 1, 20), bool(_flip(rng)))


_PAIR_PARAMS = (
    ("a", "b"), ("x", "y"), ("m", "n"), ("p", "q"), ("num1", "num2"), ("u", "v"), ("c", "d"),
    ("first", "second"),
)  # fmt: skip
_ONE_PARAMS = (("x",), ("n",), ("num",), ("v",), ("value",), ("k",), ("a",), ("number",))
_LIST_PARAMS = (
    ("xs",),
    ("items",),
    ("values",),
    ("lst",),
    ("nums",),
    ("seq",),
    ("arr",),
    ("data",),
)
_FUNCTION_FAMILIES = (
    _Family(
        "add",
        ("add", "add_numbers", "plus", "total", "sum_two", "add_two", "add_up", "sum_of_two"),
        _PAIR_PARAMS,
        (
            "Return the sum of {p0} and {p1}.",
            "Add {p0} and {p1} and return the result.",
            "Return {p0} plus {p1}.",
            "Return the total of {p0} and {p1}.",
            "Add the two numbers {p0} and {p1}.",
            "Add {p0} to {p1} and return the answer.",
        ),
        (
            "    return {p0} + {p1}\n",
            "    return {p1} + {p0}\n",
            "    {l0} = {p0} + {p1}\n    return {l0}\n",
        ),
        lambda a, b: a + b,
        ((1, 2), (5, 7), (0, 9), (8, 0), (-3, 8), (6, 11)),
        _two_numbers,
    ),
    _Family(
        "double",
        (
            "double",
            "twice",
            "double_it",
            "times_two",
            "doubled",
            "make_double",
            "dbl",
            "two_times",
            "make_twice",
            "double_value",
        ),
        _ONE_PARAMS,
        (
            "Return {p0} times two.",
            "Return twice {p0}.",
            "Return {p0} multiplied by 2.",
            "Double {p0} and return it.",
            "Return double the value of {p0}.",
            "Return {p0} doubled.",
        ),
        ("    return {p0} * 2\n", "    return 2 * {p0}\n", "    return {p0} + {p0}\n"),
        lambda x: x * 2,
        ((3,), (5,), (0,), (-4,), (10,), (7,)),
        _one_number,
    ),
    _Family(
        "square",
        (
            "square",
            "squared",
            "sq",
            "square_of",
            "make_square",
            "get_square",
            "square_it",
            "self_times",
        ),
        _ONE_PARAMS,
        (
            "Return {p0} times itself.",
            "Return {p0} times {p0}.",
            "Multiply {p0} by itself and return the result.",
            "Return {p0} multiplied by {p0}.",
            "Return {p0} multiplied by itself.",
            "Return the result of {p0} times {p0}.",
        ),
        ("    return {p0} * {p0}\n",),
        lambda x: x * x,
        ((3,), (4,), (0,), (-5,), (10,), (6,)),
        _one_number,
    ),
    _Family(
        "max2",
        ("max2", "bigger", "larger", "greater", "biggest", "max_of_two", "max_two", "higher"),
        _PAIR_PARAMS,
        (
            "Return the larger of {p0} and {p1}.",
            "Return the bigger number, {p0} or {p1}.",
            "Return whichever of {p0} and {p1} is greater.",
            "Return the maximum of {p0} and {p1}.",
            "Return the greater of the two numbers {p0} and {p1}.",
            "Return the biggest of {p0} and {p1}.",
        ),
        (
            "    return max({p0}, {p1})\n",
            "    if {p0} > {p1}:\n        return {p0}\n    return {p1}\n",
            "    if {p0} >= {p1}:\n        return {p0}\n    else:\n        return {p1}\n",
        ),
        lambda a, b: max(a, b),
        ((3, 5), (9, 2), (4, 4), (-1, -6), (0, 7), (12, 8)),
        _two_numbers,
        lambda a, b: (a[0] > a[1]) != (b[0] > b[1]),
    ),
    _Family(
        "is_even",
        (
            "is_even",
            "even",
            "check_even",
            "is_even_number",
            "even_number",
            "is_it_even",
            "even_check",
            "test_even",
            "is_even_num",
            "check_if_even",
        ),
        _ONE_PARAMS,
        (
            "Return True if {p0} is even, otherwise False.",
            "Check whether {p0} is an even number.",
            "Return True if {p0} is divisible by 2.",
            "Return whether the number {p0} is even.",
            "Return True when {p0} is an even number.",
            "Return True if {p0} is a multiple of 2, else False.",
        ),
        (
            "    return {p0} % 2 == 0\n",
            "    if {p0} % 2 == 0:\n        return True\n    return False\n",
        ),
        lambda n: n % 2 == 0,
        ((4,), (7,), (0,), (-3,), (10,), (1,)),
        _one_number,
        lambda a, b: a[0] % 2 != b[0] % 2,
    ),
    _Family(
        "first",
        (
            "first",
            "first_item",
            "head",
            "get_first",
            "first_of",
            "first_value",
            "front",
            "first_one",
        ),
        _LIST_PARAMS,
        (
            "Return the first item of the list {p0}.",
            "Return the first element in {p0}.",
            "Get the first value of {p0}.",
            "Return the item at the start of {p0}.",
            "Return the first number in the list {p0}.",
            "Return the first thing in the list {p0}.",
        ),
        ("    return {p0}[0]\n",),
        lambda xs: xs[0],
        (([4, 8, 1],), ([7],), ([2, 9, 5, 6],), ([-1, 3],), ([0, 5],), ([6, 1, 1],)),
        _a_list,
    ),
    _Family(
        "last",
        ("last", "last_item", "tail", "get_last", "last_of", "last_value", "last_one", "end_item"),
        _LIST_PARAMS,
        (
            "Return the last item of the list {p0}.",
            "Return the last element in {p0}.",
            "Get the last value of {p0}.",
            "Return the item at the end of {p0}.",
            "Return the last number in the list {p0}.",
            "Return the last thing in the list {p0}.",
        ),
        ("    return {p0}[-1]\n", "    return {p0}[len({p0}) - 1]\n"),
        lambda xs: xs[-1],
        (([4, 8, 1],), ([7],), ([2, 9, 5, 6],), ([3, -1],), ([5, 0],), ([2, 2, 8],)),
        _a_list,
    ),
    _Family(
        "count_of",
        (
            "count_of",
            "count",
            "how_many",
            "count_items",
            "times_in",
            "count_equal",
            "occurrences",
            "count_matches",
            "how_often",
            "count_value",
        ),
        (
            ("xs", "v"),
            ("items", "target"),
            ("values", "x"),
            ("lst", "item"),
            ("nums", "n"),
            ("seq", "w"),
            ("arr", "val"),
            ("data", "key"),
        ),
        (
            "Return how many times {p1} appears in the list {p0}.",
            "Count how many items in {p0} are equal to {p1}.",
            "Return the number of times {p1} is in {p0}.",
            "Count the occurrences of {p1} in {p0}.",
            "Return how many items of {p0} are {p1}.",
            "Return how often {p1} occurs in {p0}.",
        ),
        (
            (
                "    {l0} = 0\n    for {l1} in {p0}:\n        if {l1} == {p1}:\n"
                "            {l0} += 1\n    return {l0}\n"
            ),
            (
                "    {l0} = 0\n    for {l1} in {p0}:\n        if {l1} == {p1}:\n"
                "            {l0} = {l0} + 1\n    return {l0}\n"
            ),
        ),
        lambda xs, v: xs.count(v),
        (
            ([1, 2, 1, 3, 1], 1),
            ([4, 4, 4], 4),
            ([5, 6, 7], 9),
            ([], 2),
            ([2, 3, 2], 2),
            ([6, 1], 1),
        ),
        _list_and_item,
    ),
    _Family(
        "sum_list",
        (
            "sum_list",
            "total_of",
            "sum_all",
            "add_all",
            "list_sum",
            "sum_items",
            "add_list",
            "list_total",
            "add_up_all",
            "sum_of_list",
        ),
        _LIST_PARAMS,
        (
            "Return the sum of all the numbers in {p0}.",
            "Add up all the numbers in the list {p0} and return the total.",
            "Return the total of the items in {p0}.",
            "Return the sum of the list {p0}.",
            "Add the numbers in {p0} together and return the result.",
            "Return the total when all the numbers in {p0} are added.",
        ),
        (
            "    return sum({p0})\n",
            "    {l0} = 0\n    for {l1} in {p0}:\n        {l0} = {l0} + {l1}\n    return {l0}\n",
            "    {l0} = 0\n    for {l1} in {p0}:\n        {l0} += {l1}\n    return {l0}\n",
        ),
        lambda xs: sum(xs),
        (([1, 2, 3],), ([10],), ([4, 5, 6, 7],), ([],), ([-2, 5],), ([9, 0, 3],)),
        _a_list,
    ),
    _Family(
        "maybe_negate",
        (
            "maybe_negate",
            "flip_sign",
            "negate_if",
            "sign_flip",
            "negate_when",
            "flip_if",
            "negate_maybe",
            "maybe_flip",
        ),
        (
            ("x", "flag"),
            ("n", "neg"),
            ("num", "on"),
            ("v", "flip"),
            ("value", "negate"),
            ("k", "switch"),
            ("a", "sign"),
            ("number", "negative"),
        ),
        (
            "Return -{p0} if {p1} is True, otherwise return {p0}.",
            "If {p1} is True return {p0} negated, otherwise return {p0} unchanged.",
            "Negate {p0} when {p1} is True, and keep it the same when {p1} is False.",
            "Return the opposite of {p0} if {p1} is True. Return {p0} if {p1} is False.",
            "Return {p0} with its sign flipped when {p1} is True, else return {p0}.",
            "Flip the sign of {p0} if {p1} is True; otherwise return {p0}.",
        ),
        (
            "    if {p1}:\n        return -{p0}\n    return {p0}\n",
            "    if {p1}:\n        return 0 - {p0}\n    else:\n        return {p0}\n",
            "    return -{p0} if {p1} else {p0}\n",
        ),
        lambda x, flag: -x if flag else x,
        ((5, True), (5, False), (-3, True), (0, True), (7, False), (-2, False), (9, True)),
        _number_and_flag,
        lambda a, b: a[1] != b[1],
    ),
)
FUNCTION_FAMILIES = tuple(f.name for f in _FUNCTION_FAMILIES)
_FAMILY_BY_NAME = {f.name: f for f in _FUNCTION_FAMILIES}


@dataclass(frozen=True)
class _Variant:
    """One surface form of a function: the header the model is shown. Its def line is its world."""

    family: str
    name: str
    params: tuple[str, ...]
    doc: int  # which docstring of the family

    @property
    def def_line(self) -> str:
        return f"def {self.name}({', '.join(self.params)}):"

    @property
    def prompt(self) -> str:
        fam = _FAMILY_BY_NAME[self.family]
        doc = fam.docs[self.doc].format(**{f"p{i}": p for i, p in enumerate(self.params)})
        return f'{self.def_line}\n    """{doc}"""\n'

    def body(self, style: int, rng: np.random.Generator | None = None) -> str:
        """The body in the family's ``style``-th form; ``rng`` picks the body's variable names."""
        taken = {self.name, *self.params}
        fields = {f"p{i}": p for i, p in enumerate(self.params)}
        for slot, candidates in enumerate(LOCAL_NAMES):
            options = [n for n in candidates if n not in taken]
            fields[f"l{slot}"] = options[0] if rng is None else choice(rng, options)
            taken.add(fields[f"l{slot}"])
        return _FAMILY_BY_NAME[self.family].bodies[style].format(**fields)


@cache
def _variants() -> tuple[dict[str, tuple[_Variant, ...]], dict[str, tuple[_Variant, ...]]]:
    """Every header of every family, split into the benchmark's and training's by its def line.

    The benchmark's have a reserved def line (with any docstring); training's have neither a
    reserved def line nor a reserved header.
    """
    bench: dict[str, tuple[_Variant, ...]] = {}
    train: dict[str, tuple[_Variant, ...]] = {}
    for fam in _FUNCTION_FAMILIES:
        every = [
            _Variant(fam.name, name, params, doc)
            for name in fam.names
            for params in fam.params
            for doc in range(len(fam.docs))
        ]
        bench[fam.name] = tuple(v for v in every if reserved_for_bench(v.def_line))
        train[fam.name] = tuple(
            v
            for v in every
            if not reserved_for_bench(v.def_line) and not reserved_for_bench(v.prompt)
        )
    return bench, train


def _show(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    return repr(value)


def _function_doc(rng: np.random.Generator) -> TextDoc:
    """A function with its docstring and body, then 2 calls that print what it returns."""
    family = _FUNCTION_FAMILIES[pick(rng, len(_FUNCTION_FAMILIES))]
    pool = _variants()[1][family.name]
    variant = pool[pick(rng, len(pool))]
    text = variant.prompt + variant.body(pick(rng, len(family.bodies)), rng) + "\n"
    shown: list[tuple[object, ...]] = []
    while len(shown) < N_EXAMPLES:
        args = family.example(rng)
        if args not in shown and (not shown or family.contrast(shown[0], args)):
            shown.append(args)
    calls = [
        f"print({variant.name}({', '.join(_show(a) for a in args)}))  # {_show(family.ref(*args))}"
        for args in shown
    ]
    return TextDoc("plain", text + "\n".join(calls), topic=TOPIC)


# --- public generators --------------------------------------------------------------------------

N_VARIANTS = {"straight": 1, "loop": 10, "if": 5, "string": 7, "list": 6, "call": 5}
BENCH_VARIANTS = {**N_VARIANTS, "string": 6}  # without the training-only case programs


def code_train_docs(rng: np.random.Generator, n: int) -> list[TextDoc]:
    """``n`` documents: half output-prediction programs, half functions with usage examples.

    A program is ``Program:\\n<code>\\nOutput: <line>`` (a random family and template, over
    random names and numbers). A function is its header, a docstring, a body, and 2 calls
    that print what it returns (``print(add(2, 3))  # 5``). None of it is a benchmark program or
    function header, nor a benchmark computation under other names.
    """
    drawer = _ProgramDrawer(rng, n, bench=False)
    docs: list[TextDoc] = []
    for _ in range(n):
        if rng.random() < PROGRAM_SHARE:
            family = OUTPUT_FAMILIES[pick(rng, len(OUTPUT_FAMILIES))]
            program = drawer.draw(family, pick(rng, N_VARIANTS[family]))
            docs.append(TextDoc("plain", f"{program.prompt} {program.answer}", topic=TOPIC))
        else:
            docs.append(_function_doc(rng))
    return docs


def _bench_outputs(rng: np.random.Generator, n: int) -> list[ExactItem]:
    drawer = _ProgramDrawer(rng, n, bench=True)
    seen: set[str] = set()
    programs: list[_Program] = []
    for i in range(n):
        family = OUTPUT_FAMILIES[i % len(OUTPUT_FAMILIES)]
        control = (i // len(OUTPUT_FAMILIES)) % BENCH_VARIANTS[family]
        programs.append(drawer.draw(family, control, avoid=seen))
        seen.add(programs[-1].world)
    return [
        ExactItem(
            "",
            "coding",
            p.prompt,
            [p.answer],
            (f"fam:{p.family}", "fmt:output"),
            max_new_tokens=OUTPUT_TOKENS,
            extract="first_line",
        )
        for p in programs
    ]


def _bench_functions(rng: np.random.Generator, n: int) -> list[CheckItem]:
    """``n`` function items, each on a def line of its own, with one of its docstrings."""
    lines: dict[str, dict[str, list[_Variant]]] = {}  # family -> def line -> its headers
    for family, pool in _variants()[0].items():
        for variant in pool:
            lines.setdefault(family, {}).setdefault(variant.def_line, []).append(variant)
    quota = fair_quota({family: len(by_line) for family, by_line in lines.items()}, n)
    chosen: list[list[_Variant]] = []
    for family in FUNCTION_FAMILIES:
        headers = list(lines[family].values())
        picks = rng.choice(len(headers), size=quota[family], replace=False)
        chosen.append([choice(rng, headers[int(i)]) for i in picks])
    items = []
    for v in in_turn(chosen):
        fam = _FAMILY_BY_NAME[v.family]
        args = {"prompt": v.prompt, "name": v.name, "tests": fam.test_cases}
        items.append(
            CheckItem(
                "",
                "coding",
                v.prompt,
                FUNCTION_CHECK,
                args,
                v.body(0),
                (f"fam:{v.family}", "fmt:function"),
                chat=False,
                max_new_tokens=FUNCTION_TOKENS,
            )
        )
    return items


def code_bench_items(
    rng: np.random.Generator, n_output: int = 100, n_func: int = 50
) -> list[ExactItem | CheckItem]:
    """``n_output`` output-prediction and ``n_func`` function-completion items (``coding-0000``...).

    Output items: ``Program:\\n<code>\\nOutput:`` and the one printed line as the answer
    (``extract="first_line"``). The families take turns, and within a family so do its templates,
    so any prefix is as balanced as it can be. No world repeats.

    Function items: a ``def`` header with a docstring (``chat=False``), judged by
    ``minipy_function_tests`` against hidden tests; ``reference`` is a correct body. The 10
    families share the items evenly, each item on a def line of its own that is reserved for the
    benchmark (and so in no training text); ``ValueError`` if ``n_func`` is more than there are.

    Both kinds are spread over the list in proportion, so a prefix has both. Every output prompt
    and every function's def line is reserved for the benchmark (:func:`reserved_for_bench`).
    """
    outputs, functions = _bench_outputs(rng, n_output), _bench_functions(rng, n_func)
    total = n_output + n_func
    items: list[ExactItem | CheckItem] = []
    o = f = 0
    for k in range(total):
        take_function = (k + 1) * n_func // total > k * n_func // total
        item = functions[f] if take_function else outputs[o]
        f, o = f + take_function, o + (not take_function)
        item.id = f"coding-{k:04d}"
        items.append(item)
    return items
