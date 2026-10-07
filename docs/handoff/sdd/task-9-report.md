# Task 9 report: MiniPy sandboxed interpreter

Status: DONE_WITH_CONCERNS (concerns are design notes for the reviewer, none blocking)
Commit: `76a9e7c feat(ml): MiniPy sandboxed interpreter for coding tasks`

## What was implemented
- `ml/src/airace_ml/minipy/interpreter.py` (single file, 790 lines, of which 530 are the `_Interpreter` class): tree-walking interpreter over `ast.parse(source, mode="exec")`, which is the only Python facility used. Imports are limited to `ast`, `warnings`, `dataclasses`, `functools.cmp_to_key` and `collections.abc.Callable`. There is no `exec`/`eval`/`compile`/`getattr`/I/O, and node dispatch is a `type(node) -> handler` dict, so model text never reaches Python attribute lookup.
- `ml/src/airace_ml/minipy/__init__.py`: re-exports `RunResult`, `run_program`, `call_function`.
- `ml/tests/test_minipy.py`: the brief's tests verbatim (unchanged, no assertion weakened) plus 160 parametrized/extra cases total for the controller rulings and edge behaviour.

Interfaces are exactly as in the brief: `RunResult(stdout, error, steps)` (dataclass), `run_program(source, *, max_steps=10_000, max_output_chars=2_000)`, `call_function(source, name, args, *, max_steps=10_000) -> (value, error)`.

### Controller rulings, as implemented
- Unsupported operator nodes report the operator class (`Pow`, `Div`, `LShift`, `BitOr`, `UAdd`, `Invert`, `Is`, `NotIn`, ...); other nodes report the node class (`Import`, `Lambda`, `ListComp`, `Tuple`, `JoinedStr`, `Constant` for float/bytes/Ellipsis, `keyword` for keyword args, `Slice` for stepped slices or slice assignment, `FunctionDef` for unsupported def forms).
- Attribute: only as callee of a Call, only `append` on a list and `upper`/`lower` on a str. Anything else (bare `x.__class__`, `xs.append` uncalled, `s.split()`, `xs.upper()`, wrong receiver type) is `Unsupported: Attribute`. Method names are compared to a 3-entry table, never looked up on the object.
- Parse failures (`SyntaxError`, null bytes, `RecursionError` from deep nesting, 4300-digit literals, non-printable chars) give `SyntaxError`. `warnings.catch_warnings()` wraps the parse so `'\d'` style SyntaxWarnings do not leak. Non-`str` source gives `TypeError`.
- Nothing escapes: a final guard maps `_Err` to its fixed string, stray break/continue/return to `SyntaxError` (Python refuses those at compile time), `RecursionError` to `RecursionLimit`, `MemoryError` to `Overflow`, any other exception to `TypeError`.
- Fixed strings for wrong arg count/bad operand/non-sequence indexing (`TypeError`), out-of-range index including `xs[i] = v` (`IndexError`), unknown name (`NameError: <name>`), `//` and `%` by zero (`ZeroDivisionError`).
- `print` joins with spaces and ends with a newline, using Python `str()` formatting (strings inside lists use `repr`, self-containing lists render `[...]`).
- Limits: each statement and each evaluated expression node costs one step (clamped so `RunResult.steps <= max_steps`); depth over 50 gives `RecursionLimit`; `|int| > 10**12`, str/list over 1000 give `Overflow`; stdout over the cap gives `OutputLimit`; stdout produced before an error is kept.
- `call_function`: runs the module body, then calls the user function; `(value, None)` or `(None, error)`; missing name gives `NameError: <name>`; non-function gives `TypeError`. Args and the result are copied as plain data (int/str/bool/None/nested list), so the callee cannot mutate the caller's list. A returned function or `range` gives `TypeError`.
- Bools behave as Python ints in arithmetic.

## Tests and results
- `python -m uv run pytest tests/test_minipy.py -q`: `160 passed in 1.42s`.
- Full suite `python -m uv run pytest`: `448 passed, 3 deselected in 34.10s` (no warnings).
- `python -m uv run ruff check .`: `All checks passed!`

Extra coverage beyond the brief: unsupported-construct table (28), attribute whitelist (8), type errors (34), runtime errors (10), slices/negative indexes, print formatting and cyclic lists, bool arithmetic, comparison chains and short-circuit, builtins and conversions, for/while `else`/`continue`/live list iteration, scoping and first-class functions, local-before-assign `NameError`, in-place `+=`/`*=` on aliased lists, step accounting, lazy huge ranges (`sum(range(999999999999))` gives `StepLimit`, `list(range(...))` gives `Overflow`), int/size boundary values, doubling-list memory/output bombs, output-limit boundaries, recursion boundary (49 ok, 50 gives `RecursionLimit` at the call_function level), hostile nesting, unparseable sources, non-string inputs, unicode, call_function value/error table, no-aliasing, a static test that parses the interpreter's own source and asserts no exec/eval/compile/getattr/open/I-O calls and an import whitelist, 17 sandbox-escape attempts, and a seeded 4000-program mutation fuzz asserting no exception escapes and only fixed error strings appear.

Additionally (not committed) I ran about 330k mutated programs through the internal entry point and counted any non-`_Err` exception as an internal bug: zero found.

## TDD evidence
- RED: `cd ml && python -m uv run pytest tests/test_minipy.py -q` before the module existed:
  ```
  tests\test_minipy.py:6: in <module>
      import airace_ml.minipy.interpreter as interpreter_module
  E   ModuleNotFoundError: No module named 'airace_ml.minipy'
  ERROR tests/test_minipy.py
  !!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!
  1 error in 0.23s
  ```
  Expected: the interpreter did not exist yet.
- GREEN: same command after implementation: `160 passed in 1.42s`.

## Files changed (all new)
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\minipy\__init__.py`
- `C:\Users\petko\Desktop\AI Race\ml\src\airace_ml\minipy\interpreter.py`
- `C:\Users\petko\Desktop\AI Race\ml\tests\test_minipy.py`

I staged these three files only instead of the brief's `git add ml`. Files are LF (repo uses `eol=lf`).

## Self-review findings
- Ruff 0.16 here has a larger default rule set than expected. I applied the autofix for RUF023 (sorted `__slots__`) and added two `# noqa: BLE001` on the two intentional catch-alls (parse failure and the final guard), each with a reason. `ruff format` was not applied: the repo is not format-clean (9 existing files would change) and the brief's test lines are intentionally compact.
- Brief test code needed no ruff changes.
- Comparisons are implemented inside the interpreter (`_equal`, `_order`, `functools.cmp_to_key` for `sorted`), not via Python's `==`/`<`. Reason: two structurally equal list DAGs built with `a = [a, a]` 40 times would otherwise compare in 2**40 native steps with no step charge. Now each element visited costs a step (tested: gives `StepLimit` instantly).
- Builtins that walk a sequence (`sum`, `max`, `min`, `sorted`, `list`, `in` on lists) charge `len` steps up front, so `range(10**12)` is lazy in `for` and costs a `StepLimit` anywhere else.

## Concerns and judgement calls (please sanity-check)
1. **Python stack coupling.** Evaluation is plain recursion on the host stack. Measured cost is about 6 Python frames per MiniPy call for simple bodies and about 12 for a function with `for`+`while`+`if` nesting. Complex recursion at depth ~49 passes with up to ~300 frames already on the caller's stack and fails (reported as `RecursionLimit`) at ~400. A flat chain `1+1+...` works to about 495 terms, beyond that gives `RecursionLimit`. This is consistent with the brief but means a very deep calling harness could see spurious `RecursionLimit` on near-limit recursion. I did not touch `sys.setrecursionlimit` (process-global). Say if you want that.
2. **ValueError-like conditions** use `"TypeError"` because the brief's error list has no ValueError: `int('abc')`, `max([])`, `range(0, 5, 0)`.
3. **Size caps apply wherever a value is created**, not just after `+`, `*` and `append`: string constants over 1000 chars, list displays over 1000 elements, `str(list)` over 1000 chars, `upper()`/`lower()` growth (e.g. `'ß'`), `sorted`/`list` of long ranges. Cap is `> 1000` (exactly 1000 allowed).
4. **Output limit:** a `print` that would push stdout past `max_output_chars` is dropped whole (no partial line), so `len(stdout) <= max_output_chars` always holds. `call_function` has no `max_output_chars` parameter, so it uses the 2,000 default and prints in the module body or the function count against it (`OutputLimit`); its stdout is discarded.
5. **Language corners decided by me:** nested `def` gives `Unsupported: FunctionDef` (no closures, rather than silently wrong scoping); decorators, defaults, `*args`, keyword-only/positional-only params and generics give the same. Annotations on parameters/returns are accepted and ignored (never evaluated). `while`/`for` `else` is supported. Assigning a name anywhere in a function makes it local for the whole function (reading it earlier gives `NameError`, like Python's UnboundLocalError). Lists are iterated live in `for` (like Python). `xs += ys` and `xs *= n` mutate in place so aliases see it; `list += str` is a `TypeError` (Python would extend with characters). `not in`, `is`, `is not`, tuples, dicts, floats, f-strings, comprehensions, lambdas, `try` are all unsupported per the brief's allowed list.
6. **Thread-safety:** `warnings.catch_warnings()` mutates global warning filters during the parse. It is fine for a single-threaded worker; not for concurrent threads.
7. **Memory/time bounds:** each node costs a step but a single node can create a 1000-element list, so peak memory is roughly `max_steps x 1000` elements in a hostile program (about 80 MB at the default 10,000 steps). No cap on source length is applied (the model's context length bounds it in practice).
8. `docs/progress.md` was not updated (not part of the brief; left for the controller).


---

# Fix report, round 1 (security review: resource limits)

Commit: `21f3f17 fix(ml): MiniPy resource limits hold: memoised defs, per-target and per-node step charges, element budget, deterministic nesting cap, source cap, wall-clock net`
Files: `ml/src/airace_ml/minipy/interpreter.py`, `ml/tests/test_minipy.py` (the brief's original tests are untouched; `__init__.py` unchanged).

## What changed, per item

1. **Uncharged host work (critical).**
   - `def` analysis (unsupported-feature check, parameter list, duplicate check, `_assigned_names` walk) is computed once per `FunctionDef` node and memoised in `_Interpreter._definitions` (keyed by the node, which keeps it alive). Re-executing a `def` in a loop is now O(1) host work for one step.
   - `_s_assign` ticks one step per target.
   - Also audited every other per-node path for source-size-proportional work and found none left: list displays, calls, bool/compare chains all tick per element or operand.
2. **Per-step host work.**
   - `_render` is now an interpreter method that ticks one step per value node rendered (and counts nesting), so `str(list)` costs about its node count.
   - `print` draws all its arguments from one shared remaining-output budget (the newline and separators included), so it stops after about 7 of 50 large arguments instead of rendering all of them.
   - `str in str` ticks `len(container)` steps.
3. **Memory.** `MAX_CREATED_ELEMENTS = 200_000`, charged cumulatively per run by `_create`/`_made` at every creation site: list display, `+`, `*` (str and list), `append` (1 element), `sorted`, `list()`, slices (str and list), `upper`/`lower`, `str()`, and the `call_function` copy-in/copy-out. Exceeding it gives `"Overflow"`. It is a plain counter, so deterministic.
4. **Deterministic RecursionLimit.** `MAX_NESTING = 250` is a counter (`self.nesting`) incremented in `_block`, `_eval`, `_method_call`, and in the recursive helpers (`_equal`, `_order`, `_render`, `_plain_copy`), always decremented in `finally`. Exceeding it gives `"RecursionLimit"`. `RecursionError` is still caught as a fallback; `sys.setrecursionlimit` is not touched. Headroom is documented in the module docstring (see below).
5. **Hardening.**
   - (a) `len()` is wrapped in `_int(...)`: `len(range(-10**12, 10**12))` gives `Overflow`.
   - (b) the whole `catch_warnings` + `ast.parse` is under a module-level `threading.Lock` (`_PARSE_LOCK`), `filename="<minipy>"`.
   - (c) `MAX_SOURCE_CHARS = 20_000`, else `"Overflow"` (checked before parsing, `steps == 0`).
   - (d) `run_program` and `call_function` take `max_seconds: float = 2.0`; `perf_counter()` is read every 256 steps; exceeding gives the new fixed string `"TimeLimit"`.
   - (e) the `call_function` boundary copy ticks one step per node copied (the old `_DATA_NODE_LIMIT` is gone; the step and element budgets bound it).

## Results (before vs after, default 10,000 steps unless noted)

| Hostile program | Before | After |
|---|---|---|
| `def` with 50 assignments re-defined in `while True` | 1200 ms | 7 ms |
| `def` with 2500 params in `while True` | 1338 ms | 12 ms |
| 10 KB `a = a = ... = 1` in `while True` | 2040 ms | 7 ms |
| `str(x)` of a 900-char list in a loop | 283 ms | 7 ms |
| `x = [x] * 1000` loop, 200k steps | StepLimit, peak 268.5 MB | Overflow after 1,403 steps, peak 1.6 MB |

The 33 hostile shapes in `_HOSTILE` (loops, 20 KB parse bombs, nested/shared structures) all finish in at most about 22 ms; the test bound is 1.0 s.

## Tests (new in `tests/test_minipy.py`)
- `test_hostile_programs_finish_quickly_at_default_steps[...]`: 33 parametrized hostile programs, each under 1.0 s at `max_steps=10_000`.
- Def memoisation (also at 100k steps, bound 2 s, and a re-executed def still behaves), per-target step (`a = b = c = 1` costs 5), render and print shared budget, string containment charge, default constants.
- 14 creation-site cases (one per site, with the element budget monkeypatched to 5,000), ordinary programs unaffected, tracemalloc peak under 30 MB at 200k steps, call_function boundary (StepLimit, Overflow, DAG bomb).
- 17 nesting cases (recursion 49/50, recursion through six `if`s, flat sums, 90 nested ifs, method calls, builtin calls, compare/render/sort of nested lists) each run at host depth 0 and 300 and required to give the same result; a near-full-stack test proves the `RecursionError` fallback still reports `RecursionLimit`; `test_mutual_and_complex_recursion...` now also runs 300 frames deep.
- `len(range)` cap, concurrent parsing leaves `warnings.filters` unchanged (8 threads, switch interval 1e-6; I checked that without the lock it corrupts the filters in 5 of 5 attempts), source cap, wall-clock net.
- Existing tests updated for the changed numbers only: `x = 1` now costs 3 steps (one per target), the flat-sum case uses 200 terms (the cap is 250 levels), and the static test allows `threading`/`time` imports and forbids a `sleep` call.

Commands and output:
- RED (new tests against the old implementation): `python -m uv run pytest tests/test_minipy.py -q` gave `35 failed, 196 passed, 1 warning` (hostile timings over 1 s, missing budgets, host-dependent nesting, filter corruption, no `max_seconds`).
- GREEN: `python -m uv run pytest tests/test_minipy.py -q` gave `237 passed in 2.38s` (run three times, stable).
- Full suite: `python -m uv run pytest` gave `525 passed, 3 deselected in 37.64s`, no warnings. `python -m uv run ruff check .` gave `All checks passed!`.
- About 200k more mutated programs through the internal entry point found no non-`_Err` exception.

## Host stack headroom (documented in the module docstring)
Measured with programs that saturate the cap: about 2.4 frames per nesting level; the worst case I found (a depth-48 recursion whose body is `for` in `while` in `if`, five levels per call) needs about 610 frames. A caller therefore needs about 610 free frames below Python's limit of 1000, so it is safe from roughly 350 frames deep; pytest itself sits at about 33 frames and the tests run from 300 deep, which leaves about 100 spare in the heaviest test.

## Concerns / judgement calls
1. **Cap size vs headroom.** 250 levels gives about five levels per call at depth 50: a recursive function of the usual shapes (an `if`, a `return 1 + f(n-1)`, or `xs.append(f(n-1))`) fits with room to spare, but a call buried in six nested `if`s stops at depth 26, and `xs.append(abs(-f(n - 1)))` stops before 50 (7 levels per call). A cap of 300 would need about 730 frames and break the "run from 300 frames deep" requirement, so I kept 250; one constant if you want to trade differently.
2. **Quadratic string building now overflows sooner.** The element budget counts every created element, so building a string by repeated `s = s + c` reaches 200,000 created characters at about 630 characters. That is far beyond what a 10,000-step program needs, but it is a visible behaviour.
3. **`call_function` can fail with `StepLimit` on a correct result** if the function used nearly all its steps and the returned list is large, because the copy-out is charged (as requested).
4. **Parse depth is still host-dependent in one narrow band.** `ast.parse` itself recurses against Python's limit: a flat chain such as `1+1+...` of roughly 700 to 1000 terms parses on a shallow stack and gives `RecursionLimit` from the nesting cap, but fails to parse (`SyntaxError`) on a deeper stack. Pass/fail never differs (both are errors), only the error string for such programs. Shorter chains (up to about 250 terms) are deterministic.
5. **`TimeLimit` is wall-clock**, so by nature it is machine-dependent; with the accounting fixes the whole hostile battery is about 22 ms at the default steps against a 2 s default, so legitimate programs do not get near it. `perf_counter` is the only clock use.
6. **Lock scope.** `_PARSE_LOCK` serialises MiniPy's own parses (a few milliseconds each at most) and protects the global warning filters from MiniPy's concurrent callers; a non-MiniPy thread that emits a warning during that window could still see it suppressed.
