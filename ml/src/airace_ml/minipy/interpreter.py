"""MiniPy: a tree-walking interpreter for a tiny Python subset.

Benchmarks judge code written by players' models. That code never reaches the real Python
runtime: it is parsed with ``ast.parse`` (the only Python facility used) and the resulting tree is
walked by this file. Nothing here executes, evaluates or compiles model output, and nothing here
touches the file system, the network or the process.

Language: int/str/bool/None/list values; ``+ - * // %``, ``- not``, ``and or``, comparisons
(``== != < <= > >= in``), indexing and simple slices, ``if/elif/else``, ``while``, ``for``,
top-level ``def`` with positional parameters, ``return/break/continue/pass``; the builtins
``print len range max min sum abs str int list sorted`` and the methods ``list.append``,
``str.upper``, ``str.lower``. Everything else reports ``Unsupported: <NodeName>``.

Limits. Each one ends the run with a fixed error string, and none depends on the machine, the
caller or the wall clock (except the last, a safety net that legitimate programs never reach):

- ``StepLimit``: every statement, expression node, assignment target, rendered value node, copied
  value node and element walked by a builtin costs one step, so the steps spent bound the host
  work done. Per-node work never grows with the size of the source.
- ``RecursionLimit``: more than 50 nested user calls, or more than ``MAX_NESTING`` nested
  evaluation levels (statement blocks, expressions, and recursion through nested data when
  comparing, sorting or printing). This is a counter, not Python's own recursion limit.
- ``Overflow``: integers beyond 10**12, strings or lists beyond 1000 elements, more than
  ``MAX_CREATED_ELEMENTS`` elements created over a whole run (every place that builds a string or
  list charges it, which bounds memory regardless of steps), or source longer than
  ``MAX_SOURCE_CHARS``.
- ``OutputLimit``: stdout beyond ``max_output_chars``.
- ``TimeLimit``: more than ``max_seconds`` of wall-clock time, read every 256 steps.

Host stack headroom: the evaluator recurses on the Python stack, using about two frames per
nesting level plus one or two per user call. Programs that saturate ``MAX_NESTING = 250`` at call
depth 50 were measured at about 610 frames in total (for example a recursive function whose body is
a ``for`` inside a ``while`` inside an ``if``, five levels per call). A caller therefore needs
about 610 free frames below Python's default limit of 1000, so it is safe from up to roughly 350
frames deep; the tests also run from 300 frames deep. ``RecursionError`` is still caught as a last
resort and reported as ``RecursionLimit``. ``sys.setrecursionlimit`` is never touched.
"""
from __future__ import annotations

import ast
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cmp_to_key
from time import perf_counter

MAX_INT = 10**12
MAX_SEQ = 1000
MAX_DEPTH = 50
MAX_NESTING = 250
MAX_CREATED_ELEMENTS = 200_000
MAX_SOURCE_CHARS = 20_000
DEFAULT_MAX_STEPS = 10_000
DEFAULT_MAX_OUTPUT_CHARS = 2_000
DEFAULT_MAX_SECONDS = 2.0
_CLOCK_INTERVAL = 256  # steps between wall-clock reads
_PARSE_LOCK = threading.Lock()  # warnings.catch_warnings swaps process-global filters


@dataclass
class RunResult:
    stdout: str
    error: str | None
    steps: int


# ---- internal signals --------------------------------------------------------------------------

class _Err(Exception):
    """A MiniPy error; ``message`` is one of the fixed error strings."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class _Control(Exception):
    """Base for break/continue/return signals."""


class _Break(_Control):
    pass


class _Continue(_Control):
    pass


class _Return(_Control):
    def __init__(self, value: object):
        super().__init__()
        self.value = value


class _TooLong(Exception):
    """Rendering a value produced more text than the caller allows."""


def _unsupported(name: str) -> _Err:
    return _Err(f"Unsupported: {name}")


# ---- values ------------------------------------------------------------------------------------

class _Func:
    __slots__ = ("body", "local_names", "name", "params")

    def __init__(self, name: str, params: list[str], body: list[ast.stmt], local_names: frozenset):
        self.name, self.params, self.body, self.local_names = name, params, body, local_names


class _Builtin:
    __slots__ = ("fn", "name")

    def __init__(self, name: str, fn: Callable[[list], object]):
        self.name, self.fn = name, fn


class _Scope:
    """Variables of one call frame. ``local_names`` is None for the module scope."""

    __slots__ = ("local_names", "vars")

    def __init__(self, variables: dict, local_names: frozenset | None):
        self.vars, self.local_names = variables, local_names


def _is_int(value: object) -> bool:
    return type(value) is int or type(value) is bool


def _int(value: int) -> int:
    if value > MAX_INT or value < -MAX_INT:
        raise _Err("Overflow")
    return value


def _cap(length: int) -> None:
    if length > MAX_SEQ:
        raise _Err("Overflow")


def _text(value: str) -> str:
    _cap(len(value))
    return value


def _arity(args: list, low: int, high: int) -> None:
    if not low <= len(args) <= high:
        raise _Err("TypeError")


def _assigned_names(body: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if type(node) is ast.Name and type(node.ctx) is ast.Store:
                names.add(node.id)
    return names


def _parse(source: object) -> ast.Module:
    if not isinstance(source, str):
        raise _Err("TypeError")
    if len(source) > MAX_SOURCE_CHARS:
        raise _Err("Overflow")
    try:
        # catch_warnings swaps the process-wide filter list, so only one parse may run at a time.
        with _PARSE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. invalid escape sequences in model text
            return ast.parse(source, filename="<minipy>", mode="exec")
    except Exception:  # noqa: BLE001  SyntaxError, ValueError (null bytes), RecursionError, ...
        raise _Err("SyntaxError") from None


_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
_COMPARES = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In)
_METHOD_RECEIVERS = {"append": list, "upper": str, "lower": str}


# ---- the interpreter ---------------------------------------------------------------------------

class _Interpreter:
    def __init__(self, max_steps: int, max_output_chars: int, max_seconds: float):
        self.max_steps = max_steps
        self.max_output = max_output_chars
        self.steps = 0
        self.created = 0  # strings/list elements built so far, for MAX_CREATED_ELEMENTS
        self.nesting = 0  # evaluation levels currently open, for MAX_NESTING
        self.depth = 0  # user calls currently open, for MAX_DEPTH
        self.out: list[str] = []
        self.out_len = 0
        self._deadline = perf_counter() + max_seconds
        self._next_clock_check = _CLOCK_INTERVAL
        self._definitions: dict[ast.FunctionDef, tuple[list[str], frozenset[str]]] = {}
        self.module = _Scope({}, None)
        self.scope = self.module
        self.builtins: dict[str, _Builtin] = {
            "print": _Builtin("print", self._b_print), "len": _Builtin("len", self._b_len),
            "range": _Builtin("range", self._b_range), "max": _Builtin("max", self._b_max),
            "min": _Builtin("min", self._b_min), "sum": _Builtin("sum", self._b_sum),
            "abs": _Builtin("abs", self._b_abs), "str": _Builtin("str", self._b_str),
            "int": _Builtin("int", self._b_int), "list": _Builtin("list", self._b_list),
            "sorted": _Builtin("sorted", self._b_sorted),
        }
        self._statements: dict[type, Callable] = {
            ast.Expr: self._s_expr, ast.Assign: self._s_assign, ast.AugAssign: self._s_augassign,
            ast.If: self._s_if, ast.While: self._s_while, ast.For: self._s_for,
            ast.FunctionDef: self._s_functiondef, ast.Return: self._s_return,
            ast.Pass: self._s_pass, ast.Break: self._s_break, ast.Continue: self._s_continue,
        }
        self._expressions: dict[type, Callable] = {
            ast.Constant: self._e_constant, ast.Name: self._e_name, ast.BinOp: self._e_binop,
            ast.UnaryOp: self._e_unaryop, ast.BoolOp: self._e_boolop, ast.Compare: self._e_compare,
            ast.Call: self._e_call, ast.List: self._e_list, ast.Subscript: self._e_subscript,
            ast.IfExp: self._e_ifexp,
        }

    # -- entry points --

    def run(self, source: object) -> None:
        self._block(_parse(source).body)

    def call(self, source: object, name: object, args: object) -> object:
        self.run(source)
        if name not in self.module.vars:
            raise _Err(f"NameError: {name}")
        function = self.module.vars[name]
        if type(function) is not _Func or type(args) is not list:
            raise _Err("TypeError")
        values = [self._plain_copy(arg) for arg in args]
        return self._plain_copy(self._call_user(function, values))

    # -- accounting --

    def _tick(self, count: int = 1) -> None:
        self.steps += count
        if self.steps > self.max_steps:
            self.steps = self.max_steps
            raise _Err("StepLimit")
        if self.steps >= self._next_clock_check:
            self._next_clock_check = self.steps + _CLOCK_INTERVAL
            if perf_counter() > self._deadline:
                raise _Err("TimeLimit")

    def _create(self, count: int) -> None:
        self.created += count
        if self.created > MAX_CREATED_ELEMENTS:
            raise _Err("Overflow")

    def _made(self, sequence: str | list) -> str | list:
        """A string or list that was just built: enforce the size cap and charge its elements."""
        _cap(len(sequence))
        self._create(len(sequence))
        return sequence

    def _enter(self) -> None:
        # Callers decrement ``nesting`` in a ``finally``; an error ends the run, so a raise here
        # does not need to undo its own increment.
        self.nesting += 1
        if self.nesting > MAX_NESTING:
            raise _Err("RecursionLimit")

    # -- statements --

    def _block(self, body: list[ast.stmt]) -> None:
        self._enter()
        try:
            for statement in body:
                self._tick()
                handler = self._statements.get(type(statement))
                if handler is None:
                    raise _unsupported(type(statement).__name__)
                handler(statement)
        finally:
            self.nesting -= 1

    def _s_expr(self, node: ast.Expr) -> None:
        self._eval(node.value)

    def _s_assign(self, node: ast.Assign) -> None:
        value = self._eval(node.value)
        for target in node.targets:
            self._tick()  # one step per target, so `a = a = ... = 1` is charged by its length
            self._store(target, value)

    def _s_augassign(self, node: ast.AugAssign) -> None:
        operator = type(node.op)
        if operator not in _ARITH:
            raise _unsupported(operator.__name__)
        target = node.target
        if type(target) is ast.Name:
            current = self._load(target.id)
            result = self._in_place(current, self._arith(operator, current, self._eval(node.value)))
            self.scope.vars[target.id] = result
        elif type(target) is ast.Subscript:
            container, index = self._item_target(target)
            current = self._get_item(container, index)
            result = self._in_place(current, self._arith(operator, current, self._eval(node.value)))
            self._set_item(container, index, result)
        else:
            raise _unsupported(type(target).__name__)

    def _in_place(self, current: object, result: object) -> object:
        # ``xs += ys`` and ``xs *= n`` change the list itself, so aliases see it, as in Python.
        if type(current) is list and type(result) is list:
            current[:] = result
            return current
        return result

    def _s_if(self, node: ast.If) -> None:
        self._block(node.body if self._eval(node.test) else node.orelse)

    def _s_while(self, node: ast.While) -> None:
        broke = False
        while self._eval(node.test):
            try:
                self._block(node.body)
            except _Continue:
                continue
            except _Break:
                broke = True
                break
        if not broke:
            self._block(node.orelse)

    def _s_for(self, node: ast.For) -> None:
        if type(node.target) is not ast.Name:
            raise _unsupported(type(node.target).__name__)
        name = node.target.id
        sequence = self._eval(node.iter)
        if type(sequence) not in (list, str, range):
            raise _Err("TypeError")
        broke = False
        for item in sequence:  # lists are iterated live, like Python
            self.scope.vars[name] = item
            try:
                self._block(node.body)
            except _Continue:
                continue
            except _Break:
                broke = True
                break
        if not broke:
            self._block(node.orelse)

    def _s_functiondef(self, node: ast.FunctionDef) -> None:
        if self.scope is not self.module:
            raise _unsupported("FunctionDef")
        # A def can run again and again inside a loop, each time for one step, so everything
        # derived from its source is worked out once per node and remembered.
        definition = self._definitions.get(node)
        if definition is None:
            arguments = node.args
            if (arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg
                    or arguments.kwarg or arguments.defaults or node.decorator_list
                    or node.type_params):
                raise _unsupported("FunctionDef")
            params = [arg.arg for arg in arguments.args]
            if len(set(params)) != len(params):
                raise _Err("SyntaxError")
            definition = (params, frozenset(params) | _assigned_names(node.body))
            self._definitions[node] = definition
        self.scope.vars[node.name] = _Func(node.name, definition[0], node.body, definition[1])

    def _s_return(self, node: ast.Return) -> None:
        raise _Return(None if node.value is None else self._eval(node.value))

    def _s_pass(self, node: ast.Pass) -> None:
        pass

    def _s_break(self, node: ast.Break) -> None:
        raise _Break

    def _s_continue(self, node: ast.Continue) -> None:
        raise _Continue

    # -- assignment targets --

    def _store(self, target: ast.expr, value: object) -> None:
        if type(target) is ast.Name:
            self.scope.vars[target.id] = value
        elif type(target) is ast.Subscript:
            container, index = self._item_target(target)
            self._set_item(container, index, value)
        else:
            raise _unsupported(type(target).__name__)

    def _item_target(self, node: ast.Subscript) -> tuple[object, object]:
        if type(node.slice) is ast.Slice:
            raise _unsupported("Slice")
        return self._eval(node.value), self._eval(node.slice)

    def _get_item(self, container: object, index: object) -> object:
        if type(container) not in (list, str) or not _is_int(index):
            raise _Err("TypeError")
        if not -len(container) <= index < len(container):
            raise _Err("IndexError")
        return container[index]

    def _set_item(self, container: object, index: object, value: object) -> None:
        if type(container) is not list or not _is_int(index):
            raise _Err("TypeError")
        if not -len(container) <= index < len(container):
            raise _Err("IndexError")
        container[index] = value

    # -- expressions --

    def _eval(self, node: ast.expr) -> object:
        self._tick()
        handler = self._expressions.get(type(node))
        if handler is None:
            raise _unsupported(type(node).__name__)
        self._enter()
        try:
            return handler(node)
        finally:
            self.nesting -= 1

    def _e_constant(self, node: ast.Constant) -> object:
        value = node.value
        kind = type(value)
        if kind is bool or value is None:
            return value
        if kind is int:
            return _int(value)
        if kind is str:
            return _text(value)
        raise _unsupported("Constant")  # floats, bytes, complex, Ellipsis

    def _e_name(self, node: ast.Name) -> object:
        return self._load(node.id)

    def _load(self, name: str) -> object:
        scope = self.scope
        if name in scope.vars:
            return scope.vars[name]
        if scope.local_names is not None:
            if name in scope.local_names:  # assigned later in this function: still unbound
                raise _Err(f"NameError: {name}")
            if name in self.module.vars:
                return self.module.vars[name]
        if name in self.builtins:
            return self.builtins[name]
        raise _Err(f"NameError: {name}")

    def _e_binop(self, node: ast.BinOp) -> object:
        operator = type(node.op)
        if operator not in _ARITH:
            raise _unsupported(operator.__name__)
        left = self._eval(node.left)
        return self._arith(operator, left, self._eval(node.right))

    def _arith(self, operator: type, a: object, b: object) -> object:
        a_int, b_int = _is_int(a), _is_int(b)
        if operator is ast.Add:
            if a_int and b_int:
                return _int(a + b)
            if type(a) is str and type(b) is str:
                return self._made(a + b)
            if type(a) is list and type(b) is list:
                return self._made(a + b)
        elif operator is ast.Sub:
            if a_int and b_int:
                return _int(a - b)
        elif operator is ast.Mult:
            if a_int and b_int:
                return _int(a * b)
            if type(a) in (str, list) and b_int:
                return self._repeat(a, b)
            if a_int and type(b) in (str, list):
                return self._repeat(b, a)
        elif a_int and b_int:  # FloorDiv, Mod
            if b == 0:
                raise _Err("ZeroDivisionError")
            return _int(a // b if operator is ast.FloorDiv else a % b)
        raise _Err("TypeError")

    def _repeat(self, sequence: str | list, count: int) -> str | list:
        if count <= 0:
            return sequence[:0]
        _cap(len(sequence) * count)  # refuse before building anything
        return self._made(sequence * count)

    def _e_unaryop(self, node: ast.UnaryOp) -> object:
        operator = type(node.op)
        if operator is ast.Not:
            return not self._eval(node.operand)
        if operator is not ast.USub:
            raise _unsupported(operator.__name__)
        operand = self._eval(node.operand)
        if not _is_int(operand):
            raise _Err("TypeError")
        return _int(-operand)

    def _e_boolop(self, node: ast.BoolOp) -> object:
        is_and = type(node.op) is ast.And
        result: object = None
        for operand in node.values:
            result = self._eval(operand)
            if bool(result) != is_and:  # "and" stops at a falsy value, "or" at a truthy one
                return result
        return result

    def _e_compare(self, node: ast.Compare) -> object:
        left = self._eval(node.left)
        for operator, comparator in zip(node.ops, node.comparators):
            kind = type(operator)
            if kind not in _COMPARES:
                raise _unsupported(kind.__name__)
            right = self._eval(comparator)
            if not self._compare(kind, left, right):
                return False
            left = right
        return True

    def _compare(self, operator: type, a: object, b: object) -> bool:
        if operator is ast.Eq:
            return self._equal(a, b)
        if operator is ast.NotEq:
            return not self._equal(a, b)
        if operator is ast.In:
            return self._contains(b, a)
        order = self._order(a, b)
        if operator is ast.Lt:
            return order < 0
        if operator is ast.LtE:
            return order <= 0
        if operator is ast.Gt:
            return order > 0
        return order >= 0

    def _equal(self, a: object, b: object) -> bool:
        # Lists are compared here, not by Python, so that comparing two huge shared structures
        # is charged to the step budget (and to the nesting counter) instead of running away.
        if a is b:
            return True
        if type(a) is list and type(b) is list:
            if len(a) != len(b):
                return False
            self._enter()
            try:
                for x, y in zip(a, b):
                    self._tick()
                    if not self._equal(x, y):
                        return False
                return True
            finally:
                self.nesting -= 1
        if type(a) is list or type(b) is list:
            return False
        return a == b

    def _order(self, a: object, b: object) -> int:
        """Three-way comparison (-1, 0, 1) for numbers, strings and lists of those."""
        if (_is_int(a) and _is_int(b)) or (type(a) is str and type(b) is str):
            return (a > b) - (a < b)
        if type(a) is list and type(b) is list:
            self._enter()
            try:
                for x, y in zip(a, b):
                    self._tick()
                    if not self._equal(x, y):
                        return self._order(x, y)
                return (len(a) > len(b)) - (len(a) < len(b))
            finally:
                self.nesting -= 1
        raise _Err("TypeError")

    def _contains(self, container: object, item: object) -> bool:
        kind = type(container)
        if kind is str:
            if type(item) is not str:
                raise _Err("TypeError")
            self._tick(len(container))  # a substring search reads the whole container
            return item in container
        if kind is range:
            return _is_int(item) and item in container
        if kind is list:
            self._tick(len(container))
            return any(self._equal(element, item) for element in container)
        raise _Err("TypeError")

    def _e_ifexp(self, node: ast.IfExp) -> object:
        return self._eval(node.body if self._eval(node.test) else node.orelse)

    def _e_list(self, node: ast.List) -> object:
        _cap(len(node.elts))
        self._create(len(node.elts))
        return [self._eval(element) for element in node.elts]

    def _e_subscript(self, node: ast.Subscript) -> object:
        bounds = node.slice
        if type(bounds) is not ast.Slice:
            container = self._eval(node.value)
            return self._get_item(container, self._eval(bounds))
        if bounds.step is not None:
            raise _unsupported("Slice")
        container = self._eval(node.value)
        lower = None if bounds.lower is None else self._eval(bounds.lower)
        upper = None if bounds.upper is None else self._eval(bounds.upper)
        if type(container) not in (list, str):
            raise _Err("TypeError")
        for bound in (lower, upper):
            if bound is not None and not _is_int(bound):
                raise _Err("TypeError")
        return self._made(container[lower:upper])

    # -- calls --

    def _e_call(self, node: ast.Call) -> object:
        if node.keywords:
            raise _unsupported("keyword")
        if type(node.func) is ast.Attribute:
            return self._method_call(node.func, node.args)
        callee = self._eval(node.func)
        return self._invoke(callee, [self._eval(arg) for arg in node.args])

    def _method_call(self, func: ast.Attribute, arg_nodes: list[ast.expr]) -> object:
        self._tick()  # the Attribute node itself
        receiver_type = _METHOD_RECEIVERS.get(func.attr)
        if receiver_type is None:
            raise _unsupported("Attribute")
        self._enter()  # this frame sits between two evaluation levels, so it counts as one too
        try:
            receiver = self._eval(func.value)
            if type(receiver) is not receiver_type:
                raise _unsupported("Attribute")
            args = [self._eval(arg) for arg in arg_nodes]
        finally:
            self.nesting -= 1
        if func.attr == "append":
            _arity(args, 1, 1)
            _cap(len(receiver) + 1)
            self._create(1)
            receiver.append(args[0])
            return None
        _arity(args, 0, 0)
        return self._made(receiver.upper() if func.attr == "upper" else receiver.lower())

    def _invoke(self, callee: object, args: list) -> object:
        if type(callee) is _Func:
            return self._call_user(callee, args)
        if type(callee) is _Builtin:
            return callee.fn(args)
        raise _Err("TypeError")

    def _call_user(self, function: _Func, args: list) -> object:
        if len(args) != len(function.params):
            raise _Err("TypeError")
        if self.depth >= MAX_DEPTH:
            raise _Err("RecursionLimit")
        saved = self.scope
        self.scope = _Scope(dict(zip(function.params, args)), function.local_names)
        self.depth += 1
        try:
            self._block(function.body)
        except _Return as signal:
            return signal.value
        except (_Break, _Continue):  # break/continue outside a loop is a compile-time error
            raise _Err("SyntaxError") from None
        finally:
            self.scope = saved
            self.depth -= 1
        return None

    # -- values crossing in and out --

    def _plain_copy(self, value: object) -> object:
        """Copy plain data (int/str/bool/None/list) across the call_function boundary."""
        self._tick()
        kind = type(value)
        if kind is bool or value is None:
            return value
        if kind is int:
            return _int(value)
        if kind is str:
            return _text(value)
        if kind is list:
            _cap(len(value))
            self._create(len(value))
            self._enter()
            try:
                return [self._plain_copy(item) for item in value]
            finally:
                self.nesting -= 1
        raise _Err("TypeError")

    def _render(self, value: object, limit: int) -> str:
        """``str(value)`` as Python prints it; raises _TooLong past ``limit`` characters.

        One step is charged per value node, so rendering cost is bounded by the step budget.
        """
        parts: list[str] = []
        used = 0
        open_lists: set[int] = set()

        def emit(text: str) -> None:
            nonlocal used
            used += len(text)
            if used > limit:
                raise _TooLong
            parts.append(text)

        def walk(item: object, nested: bool) -> None:
            self._tick()
            self._enter()
            try:
                kind = type(item)
                if kind is str:
                    emit(repr(item) if nested else item)
                elif kind is int or kind is bool:
                    emit(str(item))
                elif item is None:
                    emit("None")
                elif kind is list:
                    if id(item) in open_lists:
                        emit("[...]")
                        return
                    open_lists.add(id(item))
                    emit("[")
                    for index, element in enumerate(item):
                        if index:
                            emit(", ")
                        walk(element, True)
                    emit("]")
                    open_lists.discard(id(item))
                elif kind is range:
                    emit(repr(item))
                elif kind is _Func:
                    emit(f"<function {item.name}>")
                elif kind is _Builtin:
                    emit(f"<built-in function {item.name}>")
                else:
                    raise _Err("TypeError")
            finally:
                self.nesting -= 1

        walk(value, False)
        return "".join(parts)

    # -- builtins --

    def _sequence(self, value: object) -> list | str | range:
        if type(value) not in (list, str, range):
            raise _Err("TypeError")
        return value

    def _consume(self, value: object) -> list | str | range:
        """A sequence about to be walked element by element: the walk is charged as steps."""
        sequence = self._sequence(value)
        self._tick(len(sequence))
        return sequence

    def _b_print(self, args: list) -> None:
        budget = self.max_output - self.out_len - 1  # keep one character for the newline
        parts: list[str] = []
        try:
            for index, arg in enumerate(args):
                if index:
                    budget -= 1  # the separating space
                text = self._render(arg, budget)  # all arguments draw on one shared budget
                budget -= len(text)
                parts.append(text)
        except _TooLong:
            raise _Err("OutputLimit") from None
        if budget < 0:
            raise _Err("OutputLimit")
        line = " ".join(parts) + "\n"
        self.out.append(line)
        self.out_len += len(line)

    def _b_len(self, args: list) -> int:
        _arity(args, 1, 1)
        return _int(len(self._sequence(args[0])))  # a range can be wider than the integer cap

    def _b_range(self, args: list) -> range:
        _arity(args, 1, 3)
        if not all(_is_int(arg) for arg in args) or (len(args) == 3 and args[2] == 0):
            raise _Err("TypeError")
        return range(*args)

    def _extreme(self, args: list, sign: int) -> object:
        _arity(args, 1, 10**6)
        items = list(self._consume(args[0])) if len(args) == 1 else args
        if not items:
            raise _Err("TypeError")
        best = items[0]
        for item in items[1:]:
            if self._order(item, best) * sign > 0:
                best = item
        return best

    def _b_max(self, args: list) -> object:
        return self._extreme(args, 1)

    def _b_min(self, args: list) -> object:
        return self._extreme(args, -1)

    def _b_sum(self, args: list) -> int:
        _arity(args, 1, 1)
        total = 0
        for item in self._consume(args[0]):
            if not _is_int(item):
                raise _Err("TypeError")
            total += item
        return _int(total)

    def _b_abs(self, args: list) -> int:
        _arity(args, 1, 1)
        if not _is_int(args[0]):
            raise _Err("TypeError")
        return _int(abs(args[0]))

    def _b_str(self, args: list) -> str:
        _arity(args, 0, 1)
        if not args:
            return ""
        try:
            text = self._render(args[0], MAX_SEQ)
        except _TooLong:
            raise _Err("Overflow") from None
        self._create(len(text))
        return text

    def _b_int(self, args: list) -> int:
        _arity(args, 0, 1)
        if not args:
            return 0
        value = args[0]
        if _is_int(value):
            return _int(int(value))
        if type(value) is str:
            try:
                return _int(int(value))
            except ValueError:
                raise _Err("TypeError") from None
        raise _Err("TypeError")

    def _b_list(self, args: list) -> list:
        _arity(args, 0, 1)
        if not args:
            return []
        _cap(len(self._sequence(args[0])))  # refuse a wide range before walking it
        return self._made(list(self._consume(args[0])))

    def _b_sorted(self, args: list) -> list:
        _arity(args, 1, 1)
        _cap(len(self._sequence(args[0])))
        return self._made(sorted(self._consume(args[0]), key=cmp_to_key(self._order)))


# ---- public API --------------------------------------------------------------------------------

def _guarded(action: Callable[[], object]) -> tuple[object | None, str | None]:
    """Run ``action``; every failure, expected or not, becomes a fixed error string."""
    try:
        return action(), None
    except _Err as error:
        return None, error.message
    except _Control:  # return/break/continue where Python would have refused to compile
        return None, "SyntaxError"
    except RecursionError:  # last resort: the nesting counter normally stops a run first
        return None, "RecursionLimit"
    except MemoryError:
        return None, "Overflow"
    except Exception:  # noqa: BLE001  an interpreter bug must not surface as a traceback
        return None, "TypeError"


def run_program(
    source: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> RunResult:
    """Run a MiniPy program. stdout produced before an error is kept."""
    interpreter = _Interpreter(max_steps, max_output_chars, max_seconds)
    _, error = _guarded(lambda: interpreter.run(source))
    return RunResult(stdout="".join(interpreter.out), error=error, steps=interpreter.steps)


def call_function(
    source: str,
    name: str,
    args: list[object],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> tuple[object | None, str | None]:
    """Run the module body of ``source``, then call its function ``name`` with plain-data ``args``.

    Returns ``(value, None)`` or ``(None, error)``. Values crossing the boundary are copied, and
    the copy is charged to the steps and element budgets like any other work.
    """
    interpreter = _Interpreter(max_steps, DEFAULT_MAX_OUTPUT_CHARS, max_seconds)
    return _guarded(lambda: interpreter.call(source, name, args))
