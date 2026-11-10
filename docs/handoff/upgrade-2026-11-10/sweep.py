"""Run a list of exp.py configurations one after another (skipping names already done)."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = Path(spec.get("results", HERE / "results.jsonl"))
done = set()
if results.exists():
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["name"])
common = spec.get("common", [])
for name, args in spec["runs"]:
    if name in done:
        print(f"skip {name}", flush=True)
        continue
    cmd = [sys.executable, str(HERE / "exp.py"), "--name", name, "--results", str(results), *common, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("RESULT") or "Error" in ln]
    print("\n".join(lines[-3:]) if lines else f"{name}: no result (exit {proc.returncode})\n{proc.stderr[-800:]}", flush=True)
print("SWEEP DONE", flush=True)
