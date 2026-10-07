# Task 16 controller notes (binding alongside the brief)

## Interfaces from earlier tasks
- **Model interface** (`airace_ml.infer.lm`):
  - `LanguageModel` protocol:
    - attribute `ctx_len`
    - `score_continuations(contexts: list[list[int]], continuations: list[list[int]]) -> list[ContinuationScore(sum_logprob, n_tokens)]`
    - `generate(prompts, *, max_new_tokens, temperature, top_p, seed, stop_ids=None) -> list[Generation(tokens, token_probs, top1_probs, stopped)]`. `stop_ids=None` means stop on `(end, bos)`. `temperature=0` is greedy; read `TorchLM.generate` to confirm how greedy is requested.
  - `TorchLM(model, tok, device=None, batch_size=32)` batches internally (`self.batch_size`). Scoring logits are `[B, W, V]` (about 512 MB at B32 × ctx1024 × V4096 in fp32), so leave the default or lower it. Never raise it.
- **Tokenizer** (`airace_ml.tokenizer`):
  - `encode_doc(tok, text)` → `[bos, ...]`
  - `encode_chat(tok, [("user", prompt)], add_generation_prompt=True)` → `bos <|user|> ... <|end|> <|ai|>`
  - `tok.encode(text)`, `tok.decode(ids)`, `tok.end_id`, `tok.bos_id`
  - Literal special-token strings in text encode as plain text.
- **Item types** (`airace_ml.skills.types`):
  - `MCItem(id, category, prompt, options, answer_index, tags, chat=False, group=None)`
  - `ExactItem(id, category, prompt, answers: list[str], tags, chat=False, max_new_tokens=8, extract="first_line"|"first_item")`
  - `CheckItem(id, category, prompt, check, check_args, reference, tags, chat=True, max_new_tokens=48)`
  - `PairItem(id, category, good, bad, tags)`
- **Item builders**, each called with `skill_rng(<category>, "bench")`:
  - language: `grammar.grammar_pairs(rng, n=200)`
  - reasoning: `reasoning.reasoning_bench_items(rng, n=200)` (MC)
  - pattern: `patterns.pattern_bench_items(rng, n=150)` (Exact)
  - knowledge: `facts.knowledge_bench_items(kb, rng)` (150 MC + 50 Exact = 200)
  - coding: `code.code_bench_items(rng, n_output=100, n_func=50)` (100 Exact + 50 Check, `chat=False`)
  - consistency: `facts.consistency_groups(kb, rng, n_groups=40)` (120 MC with `group` set, 3 per group)
  - instruction: `instructions.instruction_bench_items(kb, rng, n=120)` (Check, `chat=True`)

  Check each builder's actual signature in its module.
- **Checkers:** `airace_ml.skills.checkers.CHECKERS[check](text, check_args) -> bool`. Pass `check_args` through unchanged: some carry extra keys (`instruction`, `not`, `min_words`) that the checkers use.

## Rulings
1. **Exact matching.**
   - Normalize both the extracted reply and every accepted answer the same way:
     - strip surrounding whitespace
     - strip surrounding quote characters (`"'` and U+2018/U+2019/U+201C/U+201D)
     - strip trailing sentence punctuation (`.!?;:,`)
     - collapse internal whitespace runs to one space
     - casefold
   - The item scores 1 if the result equals any normalized entry of `answers`.
   - Do NOT strip other punctuation. A leading minus sign, list brackets and internal commas are content ("-3" ≠ "3"; "[1, 2]" is a coding answer).
   - Extraction happens before normalization: `first_item` cuts at the first "," or "\n", and `first_line` at the first "\n".
2. **Prompts.**
   - Non-chat items: generation and MC contexts use `encode_doc(prompt)`.
   - Chat items: they use `encode_chat([("user", prompt)], True)`.
   - MC continuations: `tok.encode(" " + option)` after a plain context, and `tok.encode(option)` (no leading space) after a chat context. AI turns in training chats start without a leading space. Today no MC item is chat, but the rule must hold.
3. **Generation for Exact and Check items.**
   - Greedy, seeded with the `seed` argument, `max_new_tokens` from the item.
   - Stop ids are the default `(end, bos)`.
   - The reply text is `tok.decode(generation.tokens)`. For a CheckItem the full decoded reply goes to the checker; the function-item checker cuts the body itself.
4. **`tag_breakdown()` reports only tags with at least `MIN_TAG_ITEMS = 10` items** (a named constant). Tiny tags such as `fam:double` must not reach players as discoveries. Each value is `(mean item score × 100, count)`.
5. **Consistency.**
   - A group's item score is its agreement: the fraction of equal pairs among its members' chosen option *texts*.
   - Category `raw` = mean agreement over groups. Category `score` = `normalize(raw, chance)`, where chance = the mean over groups of `1/len(options)`. That is the probability that two independent uniform choices agree, which is what the plan specifies.
   - `n` = number of groups.
   - ItemResults are one per group, with the group id as `item_id`, unioned member tags, and the agreement as `score`.
   - `max_items_per_category` counts items, rounded down to whole groups, with at least one group.
6. **`max_items_per_category`** takes the first N items in suite order (deterministic).
7. **Degenerate outputs** (Review Focus 4): empty generations, only special tokens, endless repetition. Every score must be finite. An empty category (0 items after limits) contributes nothing and is not NaN. `overall` = mean of present category scores, or 0.0 if there are none.
8. **`build_suite()`** may cache its result (e.g. `functools.cache`) because it is deterministic and is called repeatedly in tests. If cached, it must return a fresh `Suite` whose item lists callers cannot mutate into the cache: copy the lists. Item order is the builders' order.
9. **Batching.** Collect all scoring calls of a category into one `score_continuations` call, and all generations of a category (per prompt kind) into one `generate` call. `TorchLM` chunks internally.
10. **`tests/fakes.py`.**
    - `ScriptedLM` decodes prompt ids with `tok.decode` to decide replies. `answer_key_lm(tok, suite)` maps each item's encoded prompt/context (and continuation) to its answer.
    - Scores: 0.0 for the correct option or the good sentence's tokens, −10 otherwise.
    - Reply: `item.answers[0]` for Exact items, `item.reference` for Check items.
    - Key by exact token tuples, not by substring search, so similar prompts can't collide.
11. **Creativity** is not available until Task 17. With `judge`/`novelty` None, "creativity" goes into `missing`. Keep the hook (`judge`, `novelty` params) in place for Task 17 to wire `evals.creativity.score_creativity`. Do not create `creativity.py` now.
12. **Timing.** `BenchReport.seconds` = wall-clock time of `run_benchmarks` (`time.perf_counter`).
