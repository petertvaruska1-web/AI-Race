# Lab scripts of the next-level work (plans A and B)

These scripts are the record of the measurements in ADR `docs/decisions/0010-next-level.md`. Each states its own command in its docstring, written **before the release** of training text 2 (2026-11-24), when:

- `data/` held text 1 (and was the measuring stick), and
- `data2/` held text 2 (the second game).

**Since the release** (ADR 0010 §8):

- `data/` holds text 2 (the main game's data),
- `data/yardstick/` holds text 1 with its judge and novelty index (the measuring stick, unchanged),
- `data2/` keeps only the retired second game's worlds and its old rival pack.

So, in an old command, read `AIRACE_DATA=../data2` as no `AIRACE_DATA` at all (`../data`), and a `../data` yardstick (`--yardstick-root ../data`) as `../data/yardstick`. `airace-ml scorecard`, `arch_lab.py`, `decode_trial.py` and `mix_lab.py` already default to `data/yardstick` when it is there (`airace_ml.paths.yardstick_root`). Scripts that take the yardstick loss on the data root itself (`growth_lab.py`, `lessons_lab.py`) measured text 1 at the time; on today's `data/` they measure text 2's held-out conversations.
