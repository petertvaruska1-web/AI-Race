"""Copy every completed agent result of a final-review workflow run into the repo.

Usage: python3 export-results.py JOURNAL_JSONL OUT_JSON

JOURNAL_JSONL is <session transcripts>/subagents/workflows/<run id>/journal.jsonl.
The output maps each agent's label (find:core/correctness, critic, dedup,
verify:3:intent, synthesize, ...) to its structured return value, so a fresh
session can replay finished work with final-review.js (args.cacheFile) instead
of re-running it. Only completed results are exported; failed agents are listed
under "_failed" (labels whose latest event is a failure).
"""

import json
import sys

journal, out = sys.argv[1], sys.argv[2]
labels, results, status = {}, {}, {}
with open(journal, encoding="utf-8") as fh:
    for line in fh:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = ev.get("key")
        if ev.get("type") == "started" and key:
            labels[key] = ev.get("label") or key
        elif ev.get("type") == "result" and key:
            results[key] = ev.get("result")
            status[key] = "done"
        elif ev.get("type") == "failed" and key:
            status.setdefault(key, "failed")
            if status[key] != "done":
                status[key] = "failed"

exported = {labels.get(k, k): v for k, v in results.items()}
exported["_failed"] = sorted(labels.get(k, k) for k, s in status.items() if s == "failed")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(exported, fh, ensure_ascii=False, indent=1, sort_keys=True)
print(f"exported {len(results)} results, {len(exported['_failed'])} still failed -> {out}")
