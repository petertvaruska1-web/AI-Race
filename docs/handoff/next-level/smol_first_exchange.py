"""How much plain question-and-answer text does smol-smoltalk's main part hold?

    python -m uv run --extra content python ../docs/handoff/next-level/smol_first_exchange.py

Run from ml/ (needs the network). Reads the pinned dataset's `smol-magpie-ultra-short` rows and
counts the conversations whose opening exchanges are plain: no preamble that tells the AI who it
is, no role-play, no code, no lists or bold text, a short question and a short answer. Prints the
count and tokens for a few length limits, and a sample.
"""

import random
import re
from collections import Counter
from pathlib import Path

from airace_content.sources import SOURCES, hf_fetch
from airace_ml.paths import tokenizer_path
from airace_ml.tokenizer import Tok, encode_chat

tok = Tok.load(tokenizer_path(Path("../data")))
NOT_PLAIN = re.compile(r"```|\\\(|\$\$|\d{4}|\*\*|__|(?m:^#+\s)|(?m:^\s*(?:[*\-•]|\d+[.)])\s)")
PREAMBLE = re.compile(r"(?i)^\s*(?:you are|you're|act as|imagine you|assume the role|i want you to)\b")
STAGE = re.compile(r"\(\s*[a-z][^)]{2,40}\)|\*[a-z][^*]{2,40}\*")  # (smiling), *laughs*
ROLES = {"user": "user", "assistant": "ai"}

LIMITS = [(200, 300), (300, 400), (300, 600)]  # (user chars, AI chars)
count: Counter = Counter()
tokens: Counter = Counter()
samples: list = []
rng = random.Random(0)
rows = 0
for row in hf_fetch(SOURCES["smol_smoltalk"], 1_000_000):
    if row.get("source") != "smol-magpie-ultra-short":
        continue
    rows += 1
    turns = [
        (ROLES[m["role"]], str(m.get("content") or "").strip())
        for m in row.get("messages") or []
        if m.get("role") in ROLES and str(m.get("content") or "").strip()
    ]
    if len(turns) < 2 or turns[0][0] != "user" or PREAMBLE.search(turns[0][1]):
        count["preamble or no opening"] += 1
        continue
    for user_max, ai_max in LIMITS:
        kept = []
        for i in range(0, min(len(turns) - 1, 6), 2):
            (ru, u), (ra, a) = turns[i], turns[i + 1]
            if ru != "user" or ra != "ai" or len(u) > user_max or len(a) > ai_max:
                break
            if NOT_PLAIN.search(u) or NOT_PLAIN.search(a) or STAGE.search(a) or "\n" in u:
                break
            kept += [turns[i], turns[i + 1]]
        if kept:
            key = (user_max, ai_max)
            count[key] += 1
            count[(*key, len(kept) // 2)] += 1
            tokens[key] += len(encode_chat(tok, kept))
            if key == LIMITS[1]:
                seen = count[key]
                if len(samples) < 30:
                    samples.append(kept)
                elif rng.randrange(seen) < 30:
                    samples[rng.randrange(30)] = kept

print(f"rows of the part: {rows:,}; with a preamble or no opening question: {count['preamble or no opening']:,}")
for key in LIMITS:
    by_len = {n: count[(*key, n)] for n in (1, 2, 3)}
    print(f"user <= {key[0]} chars, AI <= {key[1]} chars: {count[key]:,} conversations, {tokens[key]:,} tokens; exchanges kept {by_len}")
print("\n===== sample (user <= 300, AI <= 400)")
for chat in samples:
    print("---")
    for role, text in chat:
        print(f"  {role.upper()}: {text}")
