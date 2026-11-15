"""A blind reading of small-talk replies: which of two (or more) models talks more sensibly?

The scorecard stores every small-talk reply it drew. This script puts the replies of the chosen
models on one sheet, shuffled and without saying which model wrote which, so that a reader who
does not know the models can grade them; then it adds up the grades per model.

    python blind_read.py sheet --out DIR LABEL=CARD.json:MODEL [LABEL=CARD.json:MODEL ...]
    python blind_read.py score --dir DIR

``sheet`` writes ``DIR/sheet.md`` (the rubric and the numbered items; a reply two models gave to
the same prompt appears once) and ``DIR/key.json`` (which models gave each item; the reader must
not see it). The reader writes ``DIR/verdicts.json``: a list of ``{"n": 1, "score": 0|1|2,
"person": true|false}``, one per item. ``score`` prints, per label: the share of sensible replies
(score 2), of nonsense (score 0), the mean score, and the share of replies that claim a person's
life; then, for every pair of labels, the difference in the share of sensible replies with a 90%
interval from resampling the prompts (replies to one prompt hang together, so prompts are the
unit).

Standard library only; run it from anywhere.
"""

import argparse
import json
import random
from pathlib import Path

RUBRIC = """# Reading sheet

Each item shows what a person typed to a small chat assistant, and the assistant's reply. The
assistants are very small language models: judge only whether the reply makes sense as an answer,
not whether it is clever. Grade every item on its own.

**score**
- **2, sensible:** the reply responds to what was typed, in understandable English, as an
  assistant or a friendly chat partner might. Short is fine ("Yes." to a yes-or-no question). A
  story after "Tell me a story." is sensible even when the story is simple or stops in the middle.
  A reply that says it does not know, or that it is an AI and so has no favourite, is sensible.
- **1, partly:** the reply is related to what was typed but is odd, garbled in part, contradicts
  itself or drifts away.
- **0, not sensible:** the reply has nothing to do with what was typed (a fact about something
  else, a greeting in answer to "Goodbye.", a story nobody asked for), or is mostly garbled.

**person**: true when the reply speaks as a human being with a human life (a body, a family, a
job, an age, a home, having eaten or slept, or calling the person by a made-up name as if they had
met), false otherwise. Saying "I'm doing well" or "I like helping" is not a person's life.

Write your verdicts to `verdicts.json` in this folder as one JSON list with an entry for every
item, in order: `[{"n": 1, "score": 2, "person": false}, ...]`. Do not open `key.json`.

---
"""


def _load(spec: str) -> tuple[str, list[dict]]:
    label, _, rest = spec.partition("=")
    path, _, model = rest.rpartition(":")
    if not label or not path or not model:
        raise SystemExit(f"expected LABEL=CARD.json:MODEL, got {spec!r}")
    card = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        talk = card["models"][model]["examples"]["small_talk"]
    except KeyError as e:
        raise SystemExit(f"{path} holds no small-talk replies for {model!r}: missing {e}") from e
    return label, talk


def sheet(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items: dict[tuple[str, str], dict[str, int]] = {}
    for spec in args.models:
        label, talk = _load(spec)
        for entry in talk:
            who = items.setdefault((entry["prompt"], entry["reply"]), {})
            who[label] = who.get(label, 0) + 1
    order = list(items)
    random.Random(args.seed).shuffle(order)
    lines = [RUBRIC]
    key = []
    for n, (prompt, reply) in enumerate(order, start=1):
        lines.append(f"**{n}.** Typed: {json.dumps(prompt, ensure_ascii=False)}\n")
        lines.append(f"Reply: {json.dumps(reply, ensure_ascii=False)}\n")
        key.append({"n": n, "prompt": prompt, "models": items[(prompt, reply)]})
    (out / "sheet.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(order)} items written to {out / 'sheet.md'}")


def _rates(rows: list[tuple[str, int, bool]]) -> dict[str, float]:
    n = len(rows)
    return {
        "replies": n,
        "sensible": sum(s == 2 for _, s, _ in rows) / n,
        "nonsense": sum(s == 0 for _, s, _ in rows) / n,
        "mean_score": sum(s for _, s, _ in rows) / n,
        "person": sum(p for _, _, p in rows) / n,
    }


def score(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    verdicts = {v["n"]: v for v in json.loads((folder / "verdicts.json").read_text("utf-8"))}
    missing = [k["n"] for k in key if k["n"] not in verdicts]
    if missing:
        raise SystemExit(f"no verdict for items {missing[:10]} ({len(missing)} in all)")
    by_label: dict[str, list[tuple[str, int, bool]]] = {}
    for k in key:
        v = verdicts[k["n"]]
        if v["score"] not in (0, 1, 2):
            raise SystemExit(f"item {k['n']}: score must be 0, 1 or 2, got {v['score']!r}")
        for label, count in k["models"].items():
            by_label.setdefault(label, []).extend(
                [(k["prompt"], int(v["score"]), bool(v["person"]))] * count
            )
    result = {label: _rates(rows) for label, rows in by_label.items()}
    for label, r in result.items():
        print(
            f"{label}: sensible {r['sensible']:.0%}, nonsense {r['nonsense']:.0%}, mean score "
            f"{r['mean_score']:.2f}, person's life {r['person']:.0%} ({r['replies']} replies)"
        )
    labels = list(by_label)
    rng = random.Random(0)
    prompts = sorted({p for rows in by_label.values() for p, _, _ in rows})
    pairs = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            per = {
                label: {p: [s == 2 for q, s, _ in by_label[label] if q == p] for p in prompts}
                for label in (a, b)
            }

            def diff(sample: list[str], per=per, a=a, b=b) -> float:
                share = {}
                for label in (a, b):
                    got = [x for p in sample for x in per[label][p]]
                    share[label] = sum(got) / len(got) if got else 0.0
                return share[b] - share[a]

            draws = sorted(diff(rng.choices(prompts, k=len(prompts))) for _ in range(2000))
            low, high = draws[100], draws[1899]
            print(
                f"{b} minus {a}: sensible {diff(prompts):+.0%} "
                f"(90% interval {low:+.0%} to {high:+.0%})"
            )
            pairs[f"{b}-{a}"] = {"difference": diff(prompts), "low": low, "high": high}
    (folder / "result.json").write_text(
        json.dumps({"labels": result, "pairs": pairs}, indent=1), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("sheet")
    make.add_argument("--out", required=True)
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("models", nargs="+", metavar="LABEL=CARD.json:MODEL")
    make.set_defaults(run=sheet)
    count = commands.add_parser("score")
    count.add_argument("--dir", required=True)
    count.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
