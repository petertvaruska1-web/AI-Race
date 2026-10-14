### Task 19: CLI

**Files:**
- Create: `ml/src/airace_ml/cli.py`
- Test: `ml/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int`. At startup it calls `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (and the same for stderr). Subcommands (all accept `--data-root PATH` and `--device {cpu,cuda}`):
  - `train --config FILE --out DIR [--parent DIR]`: prints `step {s}/{n}  loss {l:.3f}  lr {lr:.2e}`, sample lines and a final summary; writes `result.json`; exits 0. A config error prints `error: <message>` and exits 2.
  - `chat --model DIR [--temperature F] [--max-tokens N]`: REPL with `/reset`, `/raw` (toggles completion mode) and `/quit`. Reads lines from stdin.
  - `bench --model DIR [--out FILE] [--no-creativity] [--max-items N]`: prints a category table and writes JSON.
  - `fingerprint --model DIR`: prints traits.
  - `build-novelty-index`, and `build-judge [--budget-tokens N]`.
  - `gate --out DIR [--seeds 1,2,3] [--cpu-speed] [--quick]`: calls `experiments.gate.run_gate` (Task 20). Until Task 20 lands it exits 2 with "gate not available".

- [ ] **Step 1: Write the failing tests:**

```python
import io, json, sys
from airace_ml.cli import main
from airace_ml.model.shape import ModelShape
from airace_ml.train.config import TrainRunConfig

def _cfg(tmp_path):
    c = TrainRunConfig(run_id="cli1", seed=1, shape=ModelShape(2, 64, 64), token_budget=1024 * 8,
                       mixture={"creative": 1.0}, batch_tokens=1024)
    p = tmp_path / "c.json"; p.write_text(c.to_json(), encoding="utf-8"); return p

def test_train_bench_chat(tiny_data_root, tmp_path, monkeypatch, capsys):
    out = tmp_path / "m"
    assert main(["train", "--config", str(_cfg(tmp_path)), "--out", str(out), "--data-root", str(tiny_data_root), "--device", "cpu"]) == 0
    assert (out / "result.json").exists()
    assert main(["bench", "--model", str(out), "--no-creativity", "--max-items", "3", "--out", str(tmp_path / "b.json"),
                 "--data-root", str(tiny_data_root), "--device", "cpu"]) == 0
    assert "scores" in json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("héllo 🙂 <|end|>\n/raw\nOnce upon\n/quit\n"))  # Review Focus 1 & 5
    assert main(["chat", "--model", str(out), "--max-tokens", "8", "--data-root", str(tiny_data_root), "--device", "cpu"]) == 0

def test_bad_config_exit_2(tmp_path, capsys):
    p = tmp_path / "bad.json"; p.write_text('{"run_id": "x"}', encoding="utf-8")
    assert main(["train", "--config", str(p), "--out", str(tmp_path / "o")]) == 2
    assert "error:" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `uv run pytest tests/test_cli.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `cli.py` with argparse.** The model directory's tokenizer comes from `tokenizer_path(data_root)`.

- [ ] **Step 4: Run the tests and the whole suite.** Run `uv run pytest -v && uv run ruff check .`. Expected: all PASS, under 3 minutes.

- [ ] **Step 5: Commit.** Run `git add ml && git commit -m "feat(ml): airace-ml CLI for train, chat, bench, fingerprint, judge and index"`.

---

