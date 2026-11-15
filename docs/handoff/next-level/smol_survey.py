"""What does smol-smoltalk give the text-2 build? Run from ml/ (needs the network):

    python -m uv run --extra content python ../docs/handoff/next-level/smol_survey.py [--rows N]

Streams the pinned dataset through the build's own adapter and prints, per part of the dataset:
rows, conversations kept, those that also pass the recipe's simplicity rule, their tokens, how many
carry lists, bold text, rule text or a preamble, and a random sample of kept conversations.
"""

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from airace_content.assemble import content_text
from airace_content.sources import SOURCES, hf_fetch, row_to_content
from airace_content.textproc import simplicity
from airace_ml.paths import corpus_dir, tokenizer_path
from airace_ml.tokenizer import Tok, encode_chat

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=1_000_000)
p.add_argument("--sample", type=int, default=25)
p.add_argument("--simplicity", type=float, default=0.55)
a = p.parse_args()

root = Path("../data")
tok = Tok.load(tokenizer_path(root))
simple_vocab = set((corpus_dir(root, "v1") / "simple_vocab.txt").read_text("utf-8").split())

MARKS = {
    "list": re.compile(r"(?m)^\s*(?:[*\-•]|\d+[.)])\s"),
    "bold": re.compile(r"\*\*|__|(?m:^#+\s)"),
    "rules": re.compile(
        r"(?i)your (?:response|answer|reply|entire response) (?:should|must)|at least \d+ "
        r"(?:sentences|words|bullet|paragraphs)|P\.S\.|\[\w+\]|keywords?"
    ),
    "preamble": re.compile(r"(?i)^you are (?:an?|the) "),
}

rows_by_part: Counter = Counter()
kept: Counter = Counter()
simple: Counter = Counter()
tokens: Counter = Counter()
simple_tokens: Counter = Counter()
marks: dict[str, Counter] = defaultdict(Counter)
turns: dict[str, Counter] = defaultdict(Counter)
samples: dict[str, list] = defaultdict(list)
rng = random.Random(0)
n = 0
for row in hf_fetch(SOURCES["smol_smoltalk"], a.rows):
    n += 1
    part = str(row.get("source"))
    rows_by_part[part] += 1
    for chat in row_to_content("smol_smoltalk", row):
        kept[part] += 1
        count = len(encode_chat(tok, chat))
        tokens[part] += count
        text = content_text(chat)
        is_simple = simplicity(text, simple_vocab) >= a.simplicity
        users = "\n".join(t for r, t in chat if r == "user")
        ais = "\n".join(t for r, t in chat if r == "ai")
        flags = {
            "list": bool(MARKS["list"].search(ais)),
            "bold": bool(MARKS["bold"].search(ais)),
            "rules": bool(MARKS["rules"].search(users)),
            "preamble": bool(MARKS["preamble"].search(chat[0][1])),
        }
        if is_simple:
            simple[part] += 1
            simple_tokens[part] += count
            turns[part][len(chat)] += 1
            for name, hit in flags.items():
                marks[part][name] += hit
            marks[part]["plain"] += not any(flags.values())
            seen = simple[part]
            item = (flags, chat)
            if len(samples[part]) < a.sample:
                samples[part].append(item)
            elif rng.randrange(seen) < a.sample:
                samples[part][rng.randrange(a.sample)] = item
    if n % 50_000 == 0:
        print(f"... {n:,} rows", flush=True)

print(f"rows read: {n:,}")
for part, count in rows_by_part.most_common():
    print(
        f"{part}: rows {count:,}, kept {kept[part]:,} ({tokens[part]:,} tokens), simple enough "
        f"{simple[part]:,} ({simple_tokens[part]:,} tokens)"
    )
    if simple[part]:
        print(f"   of the simple ones: {dict(marks[part])}; turns {dict(sorted(turns[part].items()))}")
for part, items in samples.items():
    print(f"\n===== {part}: {len(items)} of the simple conversations")
    for flags, chat in items:
        on = ",".join(k for k, v in flags.items() if v) or "plain"
        print(f"--- [{on}]")
        for role, text in chat:
            print(f"  {role.upper()}: {text}")
