# Task 12 dispatch notes (controller context for the implementer and reviewer)

These are the clarifications the controller gave the Task 12 implementer, in addition to the brief (`task-brief PLAN 12`). Reuse them when resuming Task 12.

## Context from earlier tasks
- **MiniPy** (`airace_ml/minipy/interpreter.py`):
  - Entry points: `run_program(source, *, max_steps=10_000, max_output_chars=2_000, max_seconds=2.0) -> RunResult(stdout, error, steps)` and `call_function(source, name, args, *, max_steps=10_000, max_seconds=2.0) -> (value, error)`.
  - Fixed error strings: SyntaxError, StepLimit, OutputLimit, RecursionLimit, Overflow, TimeLimit, ZeroDivisionError, "NameError: x", TypeError, IndexError, "Unsupported: X".
  - Limits:
    - 200,000-element cumulative creation budget
    - nesting 250
    - |int| ≤ 10**12
    - strings and lists ≤ 1,000 elements
    - source ≤ 20,000 chars
  - Supported: ints, strs, bools, None, lists; `+ - * // %`; comparisons, and/or/not, `in`; if/elif/else, while, for, def/return, break/continue/pass; builtins `print len range max min sum abs str int list sorted`; methods `list.append`, `str.upper`, `str.lower`. No `**` and no `/`.
  - **Every generated program must run without error and stay tiny.**
- **Shared types** (`skills/types.py`): `skill_rng`, `reserved_for_bench`, `fair_quota`, `TextDoc`, `ExactItem(..., answers: list[str], extract="first_line")`, `CheckItem(id, category, prompt, check, check_args, reference, tags, chat=True, max_new_tokens=48)`, `PairItem`.
- **KB** (`skills/kb.py`): `load_kb()` is immutable and provides `relations` (with `mc_safe`/`exact_safe`), `true_object` and `accepted_answers(fact)`.
- **House style:** Task 11's `reasoning.py`/`patterns.py` show it: canonical keys, `reserved_for_bench`, a world-level partition, and a **shortcut-audit test**.

## Measurement-validity requirements
- **Ids and categories:**
  - `coding-{i:04d}`, category `coding`. Output items are `ExactItem(extract="first_line", answers=[stdout_line])`.
  - `instruction-{i:04d}`, category `instruction`.
  - `language-{i:04d}`, category `language`.
- **Code output items:**
  - Programs print exactly one line, and the answer must require evaluation.
  - Shortcut audit: "answer is the last integer literal", "answer equals any literal in the program" and "answer is the largest literal" must each be at or below 0.2.
- **Function items:**
  - Checking: the checker cuts the reply at the first non-empty line that starts without whitespace, then runs `call_function(prompt + body, name, args)` per test.
  - Tests must be discriminative. Trivial bodies (`return None`, `return 0`, `return x`, returning the first arg) must fail for every family. Add negative tests.
- **Instruction items:**
  - Chat format with a passing `reference`.
  - An empty or degenerate reply fails; natural variants pass (punctuation, case, "Yes." vs "yes").
  - Training chats are self-consistent.
  - Partition by canonical key (instruction + entity).
- **Grammar pairs:**
  - Minimal pairs scored by summed log-prob. Good and bad have **the same word count** and are as close as possible in tokens.
  - The bad member is genuinely ungrammatical; the good one is natural.
  - Families are balanced, with an audit that no single surface feature separates good from bad: the bad member doesn't always contain the same word, and good isn't always the shorter or longer one.
  - `article` uses correct a/an by sound.
- **Simple language** throughout, with correct spelling and plurals (explicit plural maps).

## State at handoff (2026-10-07)
- The implementation was committed at `f4eaa83` (5 files, +3684 lines). The full suite passes (699) and ruff is clean.
- The implementer was **stopped by the controller mid-refinement**. It was making further instruction-generator edits ("list groups, sizes, wordings") that were never committed, and **no `task-12-report.md` was written**.
- **Next:** dispatch a fresh implementer with the brief, this file, and the commit. It should:
  1. Audit `f4eaa83` against the brief and these requirements.
  2. Finish any refinements needed, including the instruction-generator list groups, sizes and wordings it was working on.
  3. Write `task-12-report.md` with test evidence and the shortcut-audit numbers.
  4. Commit.

  Then run the standard task review (base `f1a092b`).
