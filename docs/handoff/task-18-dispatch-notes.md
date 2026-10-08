# Task 18 controller notes (binding alongside the brief)

## Interfaces from earlier tasks
- **Model interface** (`airace_ml.infer.lm`):
  - `LanguageModel.generate(prompts, *, max_new_tokens, temperature, top_p, seed, stop_ids=None) -> list[Generation(tokens, token_probs, top1_probs, stopped)]`
  - `top1_probs` are the model's own temperature-1 confidences.
  - One call samples every row independently from one seeded generator, so identical prompts in one batch give different samples.
- **Tokenizer:**
  - `encode_chat(tok, [("user", text)], add_generation_prompt=True)`
  - `tok.decode(ids)`
- **KB** (`airace_ml.skills.kb`):
  - `load_kb()`
  - `kb.relations[name]`, a `Relation` with `question_templates`, `chat_templates`, `mc_safe`, `exact_safe` and `topic`
  - `kb.facts_for(relation)`
  - `kb.accepted_answers(fact)`: the true object first, then other accepted forms
  - `kb.wrong_objects(fact)`: the relation's clearly wrong objects, excluding confusable or accepted ones
- **Fakes:** `tests/fakes.py` `ScriptedLM(tok, ctx_len=256, reply=None, score=None)`. `generate` returns `tok.encode(reply(prompt_ids))` with probs of 0.5.

## Rulings
1. **Factual probes.** Author them once from the KB into `probes.json`, frozen.
   - Use only relations with `mc_safe` and `exact_safe` whose objects are distinctive words or names, such as capitals, languages, countries and animal homes. Avoid relations whose objects are numbers or colours, because those words occur incidentally in ordinary replies and would inflate `slip_rate`.
   - `answers` = `kb.accepted_answers(fact)`.
   - `wrong_answers` = `kb.wrong_objects(fact)`, not every other object: an accepted or confusable form must never count as a slip.
   - Probe text is a plain question in simple English.
   - Add a test that every factual probe still agrees with the current KB: the true object, accepted forms, and wrong answers ⊆ `wrong_objects`.
2. **Other probes** (open 20, creative 10, help 10, opinion 5) are short, simple-English chat prompts. They are varied, contain no ML jargon, are fit for all ages, and are not copied from any dataset.
3. **Sampling.** All probes × k samples go in ONE `lm.generate` call:
   - prompts `encode_chat(tok, [("user", p.text)], True)`
   - the given temperature, top_p, max_new_tokens and `seed`
   - Reply text is `tok.decode(generation.tokens)`, stripped.
   - `Fingerprint.samples` records one dict per reply: `{"probe": id, "kind": kind, "sample": j, "text": reply}`. Keep it JSON-safe.
4. **Words.** Words are lowercase alphabetic word tokens, extracted the same way everywhere in this module. Answer matching is whole-word or whole-phrase and case-insensitive; multi-word answers match as a phrase with word boundaries.
5. **Trait edge cases.** Every trait is finite, and 0 for empty input.
   - `confidence` averages per-reply mean `top1_probs` over replies that have at least one token; 0 when no reply has tokens.
   - `steadiness` with k = 1 has no pairs, so it is 0.0. `measure_fingerprint` validates `k >= 1`. Document this.
   - `repetitiveness` of one reply is `1 - unique_trigrams / total_trigrams`, and 0 for a reply with no trigrams.
   - `register` and `warmth` follow the brief's formulas over all replies pooled.
6. **`describe`.** For each trait in `TRAITS` order, compare the value with `np.percentile(population values, low_pct)` and `np.percentile(..., high_pct)` using strict `<` and `>`, and emit the low or high phrase. It returns `[]` when `len(population) < 5`. A population whose trait is constant yields no phrase for that trait unless the value differs from it.
7. **`fingerprint_distance`.** It uses the population std (`ddof=0`) per trait, with a floor of 1e-6. The result is symmetric, and 0 for identical fingerprints.
8. **Lexicons** must contain at least the test words: formal "therefore", "however", "additionally"; casual "yeah", "lol", "gonna", "hey". Each has 30–60 lowercase single words. `POSITIVE_WORDS` are warm and affective words ("love", "happy", "great", "kind", ...).
9. **`TRAIT_PHRASES`** are plain-language words a player understands (R7), e.g. verbosity → ("terse", "talkative"), confidence → ("hesitant", "self-assured"), precision → ("vague on facts", "careful with facts"), slip_rate → ("rarely slips", "often slips"). Every trait has one pair.
10. **The package directory gets an `__init__.py`.** `probes.json` must ship with the package. Hatch includes package data under `src/airace_ml` by default; verify `load_probes()` reads it via `importlib.resources` or a path relative to `__file__`, with explicit utf-8.
