# Task 12 report: code, instruction and grammar generators, and checkers

**Status:** DONE_WITH_CONCERNS (the concerns are minor and listed at the end)
**Commits:** `f4eaa83` (previous implementer, the implementation) + `983fc1b` fix(ml): finish code, instruction and grammar generators (this session)
**Base for review:** `f1a092b`
**Environment:** cloud Linux, `cd ml && uv run --no-sync ...`

## 1. What the task delivers (f4eaa83 + 983fc1b)

Files: `ml/src/airace_ml/skills/{checkers,code,instructions,grammar}.py`, `ml/tests/test_code_instr_grammar.py` (80 tests, including the brief's 6 tests unchanged).

- **`checkers.py`**: `CHECKERS` with the 8 keys from the brief (`one_word yes_no list_n starts_with all_caps repeat_word contains_any minipy_function_tests`).
  - Common rules for the text checkers:
    - An empty or degenerate reply fails.
    - Case, punctuation and surrounding spaces never decide the result.
    - New in 983fc1b: a copy of the instruction never counts (`args["instruction"]`).
  - `minipy_function_tests` cuts the reply at the first non-empty line that starts without whitespace. It runs `call_function(prompt + body, name, args)` for every test and compares value and type (`True` is not `1`).
  - Model text only ever runs in MiniPy.
- **`code.py`**:
  - `code_train_docs`: half output programs, half function documents.
    - Programs are `Program:\n{code}\nOutput: {line}`. There are 6 families (straight, loop, if, string, list, call) and 34 templates. Each prints exactly one line.
    - Function documents have a docstring and 2 usage examples.
  - `code_bench_items(rng, 100, 50)` makes ids `coding-0000`… in category `coding`.
    - Output items are `ExactItem(extract="first_line")`.
    - Function items are `CheckItem(check="minipy_function_tests", chat=False)` with `reference` = the correct body. They cover the 10 families from the brief, 5 items each.
  - A program whose printed line equals one of its own literals (compared without case) or sits inside its text is rejected. So the answer always needs evaluation.
  - Partition:
    - A program's world is its code with variables renamed to `v0, v1…`.
    - A function's world is its header (the `def` line plus the docstring).
    - The benchmark uses reserved worlds and reserved prompts only. Training uses neither.
- **`instructions.py`**:
  - There are 7 kinds of instruction, built from the knowledge base and judged by the checker of the same name.
  - `instruction_train_docs`: 1–3 exchanges per chat. Every reply passes its checker and is factually right per the KB.
  - `instruction_bench_items(kb, rng, 120)` makes ids `instruction-0000`… in category `instruction`. They are chat items, each with a passing `reference`. The tags are `fam:<kind>` and `topic:<topic>`.
  - Partition by canonical key = instruction (its kind, plus n for a list) + entity (the fact, the word, the number pair or the group). The benchmark uses reserved keys and reserved prompts only.
- **`grammar.py`**:
  - `grammar_pairs(rng, 200)` makes ids `language-0000`… in category `language`. There are 5 families × 40 pairs, tagged `fam:<family>`.
  - About 154k pairs are listed up front. The partition key is the good sentence (reserved only).
  - The two members always have the same word count, and differ in one word or in word order.

## 2. Audit findings at f4eaa83 and what I changed

The brief's tests and most dispatch requirements were already met. These are the gaps I found, all fixed in 983fc1b.

### Instruction generator and checkers

| # | Finding at f4eaa83 | Measured | Fix |
|---|---|---|---|
| I1 | A reply copied from the prompt passed several kinds, so a parroting model would score. | "Last word of the prompt" passed **55%** of the bench (one_word, yes_no, starts_with, contains_any, repeat_word). "The word after *the word*" passed **36%** (starts_with, contains_any, repeat_word). Echoing the prompt passed every contains_any item. The prompt in capitals passed all_caps + contains_any (**29%**). | (1) `args["instruction"]` on every item: copies of the instruction (any case or punctuation) are removed before the check. (2) `one_word` refuses the words of the question other than its accepted answers (`args["not"]`). (3) `starts_with` takes `min_words` = prefix + 1, so the reply must go on with an answer. (4) `contains_any` takes `min_words` = 3, so a sentence is needed. (5) Two of the four yes/no wordings no longer end in "no". (6) A word said 2+ times and nothing else ("Sure Sure") is degenerate. |
| I2 | Letter lists ("words that start with the letter M") used every KB subject. | Lists included molybdenum, zirconium, beryllium, plurals like "ants"/"bats", and heptagon/nonagon. | The letter pool is now the everyday words of 7 relations: animals, baby animals, foods, colors, vehicles, jobs/places, opposites. That gives 3–9 letters and 300+ words. |
| I3 | The wording "Write a sentence about {w}. Use the word {w}." | Ungrammatical with count nouns ("about leopard"), 67 of 3000 training chats. | Replaced with "Write a sentence and put the word {w} in it." |
| I4 | The prefix "Well!" read awkwardly ("Well! The buttermilk is…"). | — | Replaced with "Absolutely". |
| I5 | List sizes were unbalanced in the bench. | Size counts {3:6, 6:4, 2:3, 4:3, 5:1}. | Sizes now share the list items by `fair_quota` and take turns: {2:4, 3:4, 4:3, 5:3, 6:3}. This exposed a real constraint: a list world has few wordings, so only about 57% of its reserved worlds have a reserved prompt. Only those count as capacity now. I also added 4 natural list wordings (12 in total). |
| I6 | KB facts whose object is "yes"/"no" became one-word worlds that could never render. | — | Filtered out. |
| I7 | The partition test checked 5 of the 7 kinds. | — | Extended to list_n (group + n) and yes_no (claimed fact or number pair). It passed: no overlap. |

### Grammar

| # | Finding at f4eaa83 | Fix |
|---|---|---|
| G1 | **The bad member of an adjective-order pair was often grammatical English.** "Tom found the dog big", "He likes the cake happy", "She has the box empty" are object + adjective constructions (cf. "Tom found the box empty"). | The adjective now sits in the subject, where a postnominal adjective is never English: "The big dog is here." / "The dog big is here." |
| G2 | Unnatural good sentences: 816 of 5457 article adjective pairs were clearly odd ("an empty dog", "an unhappy cup", "a round eagle", "an extra horse"). | Nouns carry kinds (animal, food, object, vehicle, plant, building, furniture, place, sky, plus own/container/opens). `ADJECTIVE_FITS` says what each adjective can describe. The article and order pairs use only fitting pairs. Article frames grew from 6 to 9 (3 only for things a person can have) to keep capacity. |
| G3 | Unnatural sound pairs: "a usual cat", "an honest baby", "a union". | `usual`→`used` with per-adjective targets ("a used car", "a unique gift"). Dropped "baby" and "union". Added "an honor", "an honest mistake/answer" and two more "an hour" frames. |
| G4 | "The pig has a kite", "She has these men", "She has every duck", "The horse is in the bag", "She has the teacher at school". | People and animals have their own have-objects. "She has" takes only animals/things and numbers/"many". A box or bag holds only small things. Nothing as big as a horse or an adult goes on or under a table, bed or chair. |
| G5 | **The tense weights balanced canonical forms but not the overlapping form sets** ("walked" is both a past and a participle). "A participle-looking word marks the good sentence" had an expected edge of 0.358 vs 0.316 (gap 0.039; plain-form gap 0.017). The 0.06 sample tolerance held only by luck. | Exact balance is impossible with the 7 subtypes (the least-change solve needs negative weights). I added one natural subtype, `perfect_base` ("Tom has eaten lunch." / "Tom has eat lunch."), and solved new weights. The expected gaps over the reserved pools are now past −0.005, plain +0.003, participle +0.001. There is a new test on the expectation, which doesn't depend on the sample. |
| G6 | **Latent crash.** `grammar_pairs(…, 2000)` raised "could not draw enough word_order preposition pairs" for other seeds. Agreement and preposition pairs can share a good sentence ("The dog is in the park."), and the preposition pool was fully allocated. | `_bench_pools` gives a shared good sentence to the first family only. |

### Code

| # | Finding at f4eaa83 | Fix |
|---|---|---|
| C1 | `first`/`last` had a hidden test on `["x", "y"]`, but 2 of their 6 docstrings say "the first/last **number** in the list". | Numeric tests (`[6, 1, 1]`, `[2, 2, 8]`). The families remain discriminative: no trivial body passes. |

Everything else in the audit checked out:
- ids and categories
- one-line, evaluation-only outputs
- literal audits
- trivial bodies fail every family (11 constant bodies + 6 per parameter + 7 two-parameter bodies)
- negative checker tests
- the cut rule
- self-consistent training chats
- simple language, with explicit plural tables

## 3. Measured shortcut audits (final, at 983fc1b)

**Code: literal shortcuts.** Share of programs answered by a rule that reads only the program text. Bench has 100 output items; train has 3001 programs from 6000 documents.

| Rule | Bench | Train |
|---|---|---|
| answer = last int literal | **0.000** | 0.000 |
| answer = any literal (also case-blind) | **0.000** | 0.000 |
| answer = largest int literal | **0.000** | 0.000 |
| first/smallest int, first/last string literal | 0.000 | 0.000 |
| sum of last two ints / product of last two | 0.050 / 0.050 | 0.042 / 0.024 |
| sum of all ints | 0.040 | 0.044 |
| all strings joined | 0.030 | 0.040 |
| number of lines | 0.010 | 0.017 |

- The most common training answer ("8") is 4.2% of training answers and 2% of bench answers.
- No trivial body passes any function item: 0 passes.

**Instruction.** Share of the 120 bench items passed, with the kinds passed. Copy rules don't count repeat_word, where copying the word is the skill.
- Copied from the prompt:
  - the prompt / in capitals / twice / longest word / "Sure! "+named word: **0.000**
  - "Sure! "+prompt: 0.142 (one_word: "Sure" is one word)
  - its first word: 0.033 (yes_no)
  - its last word: 0.075 (yes_no)
  - the word it names (± "!"): 0.075 (yes_no)
  - its last three words: 0.108 (contains_any: partial echo "the word otter")
- Constants:
  - "Yes"/"No."/"YES": 0.142 (yes_no)
  - "DOG"/"Paris"/"<|end|>": 0.142 (one_word)
  - "cat, dog and cow": 0.033
  - "Sure! The dog is a mammal.": 0.025
  - "", "I do not know.", "the the…": 0.000
- No rule or constant passes more than one kind. One kind is about 0.14; that is inherent, because the brief's checkers judge form, so any one word passes one_word and "no" passes yes_no.
- Before the fix: last word 0.550 (5 kinds), named word 0.358 (3 kinds), prompt in capitals 0.292 (2 kinds).

**Grammar: surface features.** Measured on the 2000-pair sample, 400 per family.

| Family | Word/ending predictor (train half, test half) | Good is shorter | Differing word's share in bad (words seen ≥ 20×) |
|---|---|---|---|
| agreement | 0.469 | 0.520 | 0.39 (has) – 0.61 (have) |
| article | 0.455 | 0.515 | 0.48 (a) – 0.52 (an) |
| word_order | 0.500 | 0.500 | — (same words) |
| tense | 0.506 | 0.507 | 0.30 (run) – 0.70 (ran) |
| plural | 0.484 | 0.504 | — (no word ≥ 20×) |

Direction balance:
- "an" is good in 0.485 of article pairs.
- A singular verb is good in 0.517 of agreement pairs.
- The plural noun is good in 0.500 of plural pairs.
- Tense gaps between good-only and bad-only:
  - past: 0.005 in the sample, −0.005 expected
  - plain: 0.003 in the sample, +0.003 expected
  - participle: 0.045 in the sample, +0.001 expected (pool expectation 0.006). The sample figure is noise: across 30 seeds the mean is 0.010, SD 0.023 and max 0.05.

## 4. Test evidence

TDD RED → GREEN for every fix:

1. **Checkers** (`test_starts_with_can_ask_for_an_answer_after_the_word`, `test_contains_any_can_ask_for_a_sentence`, `test_a_copy_of_the_instruction_is_not_a_reply`)
   - Command: `uv run --no-sync pytest tests/test_code_instr_grammar.py -q -k "answer_after_the_word or ask_for_a_sentence or copy_of_the_instruction"`
   - RED: `AssertionError: Sure` (prefix alone passed), `AssertionError: otter` (the word alone passed), `AssertionError: one_word` (copy + answer was not judged on the answer). 3 failed. This was expected: no `min_words` and no copy removal yet.
   - GREEN after the fix: file 69 passed.
2. **Instruction generator** (`test_no_reply_made_from_the_prompt_passes_many_instructions`, `test_a_one_word_answer_is_not_a_word_of_the_question`, `test_letter_lists_use_simple_everyday_words`, `test_list_sizes_take_turns_in_the_benchmark`, `test_instruction_wordings_read_well`)
   - RED, 5 failed:
     - `('the prompt in capitals', {'all_caps', 'contains_any'})`
     - `one_word('give', {'not': ['yes', 'no']})` → True
     - letter words ∩ {aluminum, antimony, ants, argon, arsenic, barium, …}
     - sizes `Counter({3: 6, 6: 4, 4: 3, 2: 3, 5: 1})`
     - `'Write a sentence about sodium. Use the word sodium.'`
   - GREEN after the fix: file 74 passed.
3. **Grammar** (`test_word_order_pairs_move_words_the_way_each_kind_does` with the new adjective pattern, `test_adjectives_go_with_nouns_they_can_describe`, `test_people_and_animals_have_and_sit_where_they_can`, `test_tense_forms_are_balanced_by_the_weights_not_by_luck`, `test_large_language_benches_can_be_drawn_with_any_seed`)
   - Run against f4eaa83's `grammar.py` (restored temporarily from `git show HEAD:`).
   - RED, 5 failed:
     - `('adjective', 'I see the big dog.')`
     - `AssertionError: I see an empty dog.`
     - `AssertionError: The dog has a hat.`
     - `('plain', 0.01699) <= 0.01` (participle gap 0.039 too)
     - `RuntimeError: grammar: could not draw enough word_order preposition pairs`
   - GREEN with the new module: file 78 passed.
4. **Code** (`test_function_tests_fit_every_docstring_of_their_family`)
   - RED: `AssertionError: ('first', [['x', 'y']])`.
   - GREEN: file 79 passed.
5. **Coverage, green from the start:** `test_natural_variants_of_a_right_reply_pass` checks that lower/upper case, no period, quotes, "Sure," for "Sure!", and every list layout (commas, "and", Oxford comma, lines, numbered, bulleted, title case) pass for every bench item. The partition test is extended to list_n and yes_no.

Results:
- **Task file:** `uv run --no-sync pytest tests/test_code_instr_grammar.py -q` gives **80 passed** in about 20 s.
- **Full suite:** `uv run --no-sync pytest -q` gives **713 passed, 4 deselected in 51.7 s**. The baseline was 699 passed in 49.7 s. No warnings.
- **Lint:** `uv run --no-sync ruff check .` gives **All checks passed!** The Task 12 files are `ruff format`-clean. Nine files from other tasks were already unformatted at HEAD and are untouched.
- **Brief tests:** byte-identical to f4eaa83, which had only ruff formatting applied.

Changed test helpers (stricter or corrected expectations, not weakenings):
- `expected_check` now also expects `instruction`, `min_words` and the one-word refusals.
- The yes/no frames match the new wordings.
- `check_list_reply` accepts KB objects as letter-list words, since the pool now includes "hospital" and "pull". The letter or group check is still applied.
- `starts_with_vowel_sound` and the odd-spelling regex know "used" and "honor".

## 5. Files changed (983fc1b)

- `ml/src/airace_ml/skills/checkers.py`:
  - copy removal (`_own`)
  - public `words`
  - `min_words` for starts_with/contains_any
  - short-repetition degenerate rule
  - one_word refusal docs
- `ml/src/airace_ml/skills/instructions.py`:
  - letter-word pool
  - wordings and prefix
  - one_word refusals
  - instruction/min_words args
  - list-size quota over wordable worlds (`_wordable`, `_take`, `_in_turn`)
- `ml/src/airace_ml/skills/grammar.py`:
  - noun kinds and `ADJECTIVE_FITS`
  - new frames and sound phrases
  - size-aware places and have-objects
  - subject-position adjective order
  - `perfect_base` and solved tense weights
  - cross-family pool ownership
- `ml/src/airace_ml/skills/code.py`: numeric first/last tests.
- `ml/tests/test_code_instr_grammar.py`: 14 new tests and updated helpers.

## 6. Self-review

- **Interfaces:**
  - Unchanged from the brief.
  - New checker arguments are optional, and the brief tests call without them.
  - Task 16 only passes `check_args` through, so it needs no change.
  - `check_args` stays JSON-serializable: lists, strings and ints.
- **Determinism:**
  - Everything flows from `skill_rng` and the blake2b partition.
  - The bench contents changed versus f4eaa83, which is fine because bench-v1 isn't frozen yet:
    - the instruction prompts and arguments
    - the grammar pairs
    - the first/last hidden tests
- **YAGNI:** no new modules. The noun-kind table is the smallest structure that removes the unnatural combinations.
- **MiniPy:** code programs are unchanged. Every generated program still runs error-free and tiny: fewer than 400 steps and at most 12 lines (tested).

## 7. Concerns (for the controller)

1. **Ruling to confirm: the list key includes n.** "List three mammals" (bench) and "List four mammals" (train) can coexist. The canonical key is instruction (list n) + entity (mammals). The score is form only (exactly n items), so seeing other counts for the same group gives no advantage. Keying by group alone would leave about 10 reserved groups for 17 list items, which unbalances the kinds.
2. **KB chat templates (Task 10) put "the" before mass nouns** in instruction prompts: "which food group the pork belongs in", "the rye", "the buttermilk". Not changed here (KB files are Task 10's). This is a small simple-language blemish in prompts.
3. **Residuals inherent to form-only checkers (the brief's design):**
   - Any one word passes one_word (0.142 of the bench).
   - "no" passes every yes_no item (0.142).
   - A partial echo of the prompt's last 3 words passes 0.108, contains_any only.
   - Removing partial copies would also strip legitimate answers that reuse the question's phrase ("THE CAPITAL OF PERU IS LIMA"), so I didn't.
4. **The tense participle gap in the default 2000-pair audit sample is 0.045** against a 0.06 tolerance. The expectation is balanced (0.006, guarded by a new exact test), so this is sample noise (SD 0.023).
5. **2-word sentences fail sentence checks.** `all_caps` and `contains_any` count a sentence as 3+ words, so a valid 2-word sentence ("OTTERS SWIM.") fails. I kept this for consistency with f4eaa83's all_caps choice. It's rare in practice, and the cost is a slight under-count.
