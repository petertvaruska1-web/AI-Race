# Task 1 Report: Python project scaffold

**Status:** DONE (one minor deviation from the brief, see Concerns)
**Commit:** `747d806 feat(ml): scaffold Python ML project with uv, paths, device selection`

## What was implemented

- Installed uv 0.12.23 with `python -m pip install --user uv==0.12.23`. `uv.exe` lands in `C:\Users\petko\AppData\Roaming\Python\Python314\Scripts`, which is **not on PATH** in either Git Bash or PowerShell. Per the brief, every uv command was run as `python -m uv`, and the Commands section of `CLAUDE.md` was updated to match (added a note, a `sync` line, and `python -m uv` on the test and lint lines).
- `ml/pyproject.toml` and `ml/.python-version` (`3.13`) written verbatim from the brief.
- `uv sync` downloaded managed CPython 3.13.16 and installed 32 packages, including `torch==2.14.1+cu130` from the `pytorch-cu130` explicit index (see lock check below).
- `ml/src/airace_ml/paths.py`:
  - `REPO_ROOT = Path(__file__).resolve().parents[3]`
  - `TOKENIZER_VERSION = "tok-v1"`, `CORPUS_VERSION = "v1"`
  - `data_root()` honours `AIRACE_DATA`, otherwise `REPO_ROOT / "data"`
  - `tokenizer_path`, `corpus_dir`, `novelty_path` (`novelty/v1/index.npy`) and `judge_dir` (`judge/v1`) each take an optional `root`
- `ml/src/airace_ml/device.py`: `pick_device(prefer=None)` returns CPU if `prefer` or `AIRACE_DEVICE` is `cpu`, otherwise CUDA when available, otherwise CPU.
- Empty files: `airace_ml/__init__.py`, `airace_content/__init__.py`, `tests/__init__.py`, `tests/content/__init__.py`, `tests/conftest.py`.
- `ml/tests/test_paths_device.py` is the brief's test code, with one blank line added (see Concerns).

## Torch / CUDA verification

```
$ cd ml && python -m uv run python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
2.14.1+cu130 True
```

`uv.lock` pins `torch 2.14.1+cu130` from `https://download.pytorch.org/whl/cu130` for `sys_platform == 'linux' or 'win32'`, and plain PyPI `2.14.1` for other platforms (the marker in the brief's `[tool.uv.sources]` works as intended). The `content` extra (`datasets>=5.1`) also resolved and is locked.

## TDD evidence

RED (implementation absent, test file present):
```
$ cd ml && python -m uv run pytest tests/test_paths_device.py -v
collected 0 items / 1 error
tests\test_paths_device.py:1: in <module>
    from airace_ml import paths
E   ImportError: cannot import name 'paths' from 'airace_ml' (...\ml\src\airace_ml\__init__.py)
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.22s ===============================
```

GREEN (after implementing `paths.py` and `device.py`):
```
$ cd ml && python -m uv run pytest -v
tests/test_paths_device.py::test_data_root_env_override PASSED           [ 25%]
tests/test_paths_device.py::test_data_root_default PASSED                [ 50%]
tests/test_paths_device.py::test_artifact_paths PASSED                   [ 75%]
tests/test_paths_device.py::test_pick_device_prefer_and_env PASSED       [100%]
============================== 4 passed in 3.52s ==============================

$ python -m uv run ruff check .
All checks passed!
```

Final re-run after the lint fix and just before reporting: `4 passed in 3.54s`, ruff `All checks passed!`. No warnings in the pytest output.

## Files changed (12 in the commit)

- `CLAUDE.md` (Commands section)
- `ml/.python-version`, `ml/pyproject.toml`, `ml/uv.lock` (1084 lines)
- `ml/src/airace_ml/{__init__,paths,device}.py`
- `ml/src/airace_content/__init__.py`
- `ml/tests/{__init__,conftest}.py`, `ml/tests/content/__init__.py`, `ml/tests/test_paths_device.py`

`ml/.venv`, `.pytest_cache` and `.ruff_cache` are gitignored (confirmed with `git check-ignore`) and not committed. Working tree is clean after the commit.

## Self-review

- Completeness: every interface in the brief exists with the specified signature and values. Test, lint and commit steps were all executed.
- YAGNI: `pick_device` implements only what the brief says. It does not add handling for other explicit device strings such as `cuda:1`.
- Quality: `paths.py` uses a single `_`-free, flat API. `REPO_ROOT` depends on the package being installed editable (the uv default for the project itself), where `__file__` points into `ml/src/`. It would be wrong for a non-editable wheel install, which is not a use case here.
- Commit message includes the required Co-Authored-By trailer.

## Concerns

1. **Lint vs. verbatim test code.** The brief's test code produced ruff `I001` (import block needs two blank lines before the first `def`), and the brief requires "no lint errors". I applied `ruff check --fix`, which only inserted one blank line after the imports in `tests/test_paths_device.py`. Test logic is unchanged. Functions in that file still have a single blank line between them (E302 is not in ruff's default rule set), which a later `ruff format` pass would normalise.
2. **`uv` is not on PATH.** All uv commands must use `python -m uv`. `CLAUDE.md` was updated accordingly. Later task briefs that say `uv run ...` need the `python -m uv` form.
3. `uv.lock` was committed as the brief required. It was generated on Windows but contains the cross-platform resolution, with the Linux and Windows CUDA markers shown above.
