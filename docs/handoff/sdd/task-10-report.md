# Task 10 report: knowledge base, fact rendering, knowledge and consistency items, false-fact plans

Status: DONE_WITH_CONCERNS (concerns are design notes for later tasks, nothing blocks).
Commit: `0e2f221 feat(ml): knowledge base, fact rendering, knowledge/consistency items, false-fact plans`

## What I implemented

New package `ml/src/airace_ml/skills/`:

- `__init__.py`: docstring only.
- `types.py` (the shared contract for Tasks 11, 12, 14, 16): `Split`, `skill_rng`, `reserved_for_bench`, and the dataclasses `TextDoc`, `MCItem`, `ExactItem`, `CheckItem` (with `reference`), `PairItem`, with the exact fields and defaults from the brief (`ExactItem.extract` is `"first_item" | "first_line"`, default `"first_line"`, `max_new_tokens=8`; `CheckItem.chat=True`, `max_new_tokens=48`).
- `kb.py`: `Fact`, `Relation`, `KB` (`facts`, `relations`, `objects_for`, `true_object`, `distractors`, plus helpers `facts_for`, `facts_about`, `subjects`) and cached `load_kb()`. Loading validates the data and raises `ValueError` for: template placeholders that are wrong, a (subject, relation) with two objects, a relation with fewer than 4 distinct objects, a duplicate relation name. Data is read via `Path(__file__).parent / "kb_data"`.
- `facts.py`: `fact_prose_docs`, `fact_chat_docs`, `knowledge_bench_items`, `consistency_groups`, `FalseFactPlan` (`to_json`, `from_json`), `plan_false_facts`, `false_fact_sentence`.
- `kb_data/{countries,animals,things,foods,space,elements,words,jobs}.json`: hand-written, ASCII, **1478 facts over 22 relations** (needed: >= 1200 and >= 12).

Relations (facts): capital_of 148, continent_of 171, country_language 95, landmark_country 36 (countries.json); animal_class 157, animal_sound 32, animal_legs 137, animal_home 64, animal_food 75, baby_animal 39, animal_group 22 (animals.json); color_of 113, vehicle_travel 25, shape_sides 14 (things.json); food_group 103 (foods.json); planet_order 8 (space.json); element_symbol 56 (elements.json); opposite_of 81, day_after 7, month_after 12 (words.json); job_tool 43, job_place 40 (jobs.json). All 15 suggested relations are present, plus 7 extra.

Each relation has 5-6 `train_templates`, 4 `bench_templates`, 4 `question_templates` and 4 `chat_templates`. Numbers are words ("four", "first"), symbols are strings ("Au"), and travel mode is "by road" / "by water" / "by rail" / "by air".

How the pieces behave:

- **Prose docs**: one random subject, 3-6 sentences from `train_templates`. Every fact about the subject is stated once before any is restated with a different template. Doc topic is the topic of the first sentence's relation.
- **Chat docs**: one subject, 1-3 exchanges, one fact and one `chat_templates` pair per exchange (user then ai, `text=""`).
- **Knowledge bench**: exactly 150 `MCItem` (4 options, shuffled, `answer_index` of the true object) and 50 `ExactItem` (`answers=[obj]`, `extract="first_line"`, `max_new_tokens=8`). Ids `knowledge-0000` to `knowledge-0199` (MC first, then exact), category `"knowledge"`, tags `("rel:<r>", "topic:<t>")`. Facts are spread evenly over all 22 relations by water-filling (small relations are used fully, no fact repeats). About 40% of items use a bench template as a fill-in-the-blank, the rest use a question template. Prompt for both kinds is `Question: <text>\nAnswer:`; a blank item reads `Question: Fill in the blank. The ___ group of foods includes the milk.\nAnswer:`.
- **Consistency**: 40 groups x 3. Distinct facts, spread over relations. Each group uses 3 different question templates; the distractors are shared by the group and the option order is shuffled independently for each paraphrase. Ids `consistency-GGG-k`, group `consistency-GGG`.
- **False facts**: `plan_false_facts` spreads picks evenly over the *falsifiable* relations; the wrong object is another object of the same relation, never the true one and never the subject. `false_fact_sentence` renders a random planned fact with a random train template. `FalseFactPlan.to_json()` returns a JSON string (list of `[subject, relation, wrong]`); `from_json` accepts that string or the already-parsed list.

## Tests and results

`ml/tests/test_kb_facts.py`: the 6 brief tests unchanged in substance, plus 35 more (41 total). The additions cover: item/doc defaults and `skill_rng` / `reserved_for_bench` stability; KB data ships inside the package (`importlib.resources`); required relations; topics are in `TOPICS`; lookups; distractors never the truth, deterministic and bounded; template placeholders; `train_templates[0]` has `{s}` before `{o}`; training templates never use `Question:` / `Answer:` / newline; every rendered train, bench, chat-user, chat-ai and question sentence starts uppercase, is ASCII, has no braces; bench statements are disjoint from all training statements and question bodies from chat user turns (fully rendered, over all facts); every possible benchmark prompt maps to exactly one fact (no ambiguous question); prose is made of KB statements about one subject, 3-6 sentences; chat roles alternate and replies contain the true object; knowledge bench composition, ids, tags, answers equal the true object, relation spread, no repeated prompt, determinism, both phrasings used, answer positions shuffled; consistency shape, same fact and options in a group, independent shuffles, determinism; false-fact plan determinism, spread, only clear relations, JSON round trip, sentence = a train template of the planned fact, empty plan errors.

Results:
- Focused: `python -m uv run pytest tests/test_kb_facts.py -q` -> `41 passed in 1.47s`
- Full suite: `python -m uv run pytest -q` -> `566 passed, 3 deselected in 35.18s` (no warnings)
- `python -m uv run ruff check .` -> `All checks passed!`

Mutation checks (temporary edits, restored afterwards): letting `distractors` include the true object failed 4 tests; building the blank item from a train template failed `test_knowledge_answers_are_the_true_objects`; giving all consistency paraphrases one template failed `test_consistency_groups`.

## TDD evidence

RED (tests written first, no implementation yet):
```
$ cd ml && python -m uv run pytest tests/test_kb_facts.py -q -x
E   ModuleNotFoundError: No module named 'airace_ml.skills.facts'
ERROR tests/test_kb_facts.py
1 error in 0.24s
```
Expected: the module did not exist.

GREEN (after `types.py`, `kb.py`, `facts.py` and the JSON data):
```
$ python -m uv run pytest tests/test_kb_facts.py -q
.........................................                                [100%]
41 passed in 1.47s
```

## Files changed (all new)

- `ml/src/airace_ml/skills/__init__.py`, `types.py`, `kb.py`, `facts.py`
- `ml/src/airace_ml/skills/kb_data/countries.json`, `animals.json`, `things.json`, `foods.json`, `space.json`, `elements.json`, `words.json`, `jobs.json`
- `ml/tests/test_kb_facts.py`

Packaging check: `python -m uv build --wheel -o <scratch>` produced a wheel that contains all 8 `airace_ml/skills/kb_data/*.json` files (hatchling includes them by default). The build output went to the scratchpad, not the repo. The `kb_data` path is also covered by `test_kb_data_ships_inside_the_package`.

Formatting note: I ran `ruff format` on my new files (it re-wrapped the brief's semicolon-joined test lines); no assertion was changed. Twelve older files in `ml/` (e.g. `minipy/`, `test_corpus.py`) are already not `ruff format` clean; I left them alone.

## Self-review

- Accuracy pass over every fact. Removed anything with an arguable answer: countries with multiple or disputed capitals (South Africa, Bolivia, Israel, Sri Lanka, Tanzania, Ivory Coast, Libya, Benin, Indonesia, Kazakhstan, Myanmar, Burundi, Yemen, Chad, and others) from `capital_of`; transcontinental countries (Russia, Turkey, Egypt, Australia) from `continent_of`; multilingual countries (Switzerland, Belgium, Canada, India, South Africa, Nigeria) from `country_language`.
- Distractor ambiguity: kept answers mutually exclusive where I could (for example no "hay" next to "grass", no "chirp" next to "tweet", no both "bark" and "woof"). Some relations are still fuzzy (what an animal eats, where it lives, which tool a job uses); those are flagged `falsifiable: false` so they never become "false facts".
- Templates avoid a/an problems (`the {s}`, `every {s}`, objects that all start with a consonant where "a {o}" is used). Subjects such as "the United States" never start a sentence.
- Single-fact subjects (a planet, a weekday) make a paragraph of 3-5 rephrasings of the same fact. That follows "3-6 sentences about one subject"; noted below.
- The KB generator scripts I used to author the JSON live only in the scratchpad and are not committed; the JSON files are the source of truth.

## Concerns and notes for later tasks

1. **`falsifiable` field (small addition).** `Relation` has one extra trailing field, `falsifiable: bool = True`, defaulted so constructing it with the six brief fields still works. It is set from a `"falsifiable"` key in the JSON. Ten relations are excluded from false-fact plans because swapping their object is not clearly false (`animal_food`, `animal_home`, `animal_sound`, `animal_group`, `color_of`, `food_group`, `opposite_of`, `country_language`, `job_tool`, `job_place`). 12 relations and 810 facts remain eligible, enough for Task 14's `plan_false_facts(n=300)`.
2. **`skill_rng` seed.** The brief says "first 8 bytes of `blake2b(...)`", so I used the default 64-byte digest and took the first 8 bytes, big-endian. `reserved_for_bench` uses `digest_size=8`, as written. If a later task meant `digest_size=8` for `skill_rng` too, it is a one-line change, and no test pins the seed values.
3. **Use of `bench_templates`.** The brief says the bench uses both bench and question templates but fixes the prompt as `Question: ...\nAnswer:`. I use bench templates as a fill-in-the-blank inside that frame (`Question: Fill in the blank. <statement with ___>\nAnswer:`), 40% of knowledge items. The consistency groups use question templates only (the brief says so).
4. **Benchmark frame is new to the model.** Fact training text never contains `Question:` / `Answer:` (required by the disjointness rule). A tiny model has to carry what it learned from fact prose into that frame. Whether it can will depend on other datasets teaching the frame, which matters for Task 14 (dataset assembly) and Task 16 (scoring) if knowledge scores sit at chance.
5. **Contract for Task 18 (`false_fact_rate`).** Every relation's `train_templates[0]` has `{s}` before `{o}`, so "render up to `{o}`" gives a usable context; this is tested. For some relations `{o}` is not the last word of the template (for example `The {s} has {o} legs.`), so the context ends at "has" and the object is scored as `" four"`. This matches the plan text.
6. **Exact items match only the true object**, as briefed. A model that writes `Paris.` with a period will not match unless Task 16's matching strips punctuation.
7. `docs/progress.md` is not updated by me (left for the controller at the review handoff).


---

# Fix round 1 (reviewer read all facts; bench-v1 measurement validity)

Status: DONE. Commit: `132a3f9 fix(ml): knowledge base measures what it claims: unsafe relations never benchmarked, real disjointness tests, retired fuzzy facts, immutable KB`.

## What changed

**1. `country_language`** now keeps only countries where the listed language is what most people speak. Deleted (as listed): Senegal, Mali, Niger, Burkina Faso, Guinea, Togo (French), Mozambique, Ethiopia, Belize. I judged the rest and also removed: Jamaica, Guyana, Trinidad and Tobago, Barbados, the Bahamas (English-based creoles), Angola and Suriname (Portuguese/Dutch spoken by a part of the population), Qatar, Kuwait, Bahrain, the United Arab Emirates, Oman (Arabic, but expatriates are about half or more), Nepal, Malaysia, Laos (listed language spoken natively by about half). 70 facts remain (95 before). The relation is also `mc_safe: false`.

**2. `mc_safe` (and one more flag).**
- `Relation.mc_safe` (JSON key `mc_safe`, required). `false` for: animal_food, animal_home, animal_sound, animal_group, color_of, food_group, opposite_of, country_language, job_tool, job_place. The other 12 relations are safe: capital_of, continent_of, landmark_country, animal_class, animal_legs, baby_animal, vehicle_travel, shape_sides, planet_order, element_symbol, day_after, month_after.
- `Relation.exact_safe` (JSON key `exact_safe`, required). `false` only for `vehicle_travel`: its objects are phrases ("by road", "by rail"), which a free answer would match too rarely and too arbitrarily ("on the road"). Its MC items stay.
- `knowledge_bench_items` draws the 50 exact items from `mc_safe and exact_safe` relations first, then the 150 MC items from `mc_safe` relations (never reusing a fact). `consistency_groups` draws from the same `mc_safe` pool. Each pool is split with `_fair_quota` over its own relations. Unsafe relations still feed `fact_prose_docs` / `fact_chat_docs`, and stay in false-fact plans where `falsifiable`.
- Counts are kept: 150 MC, 50 exact, 40 consistency groups (120 items). The four reviewer examples (alligator meat/fish, coconut brown/white, hedgehog insects/worms, ash gray/black) cannot occur any more: all four relations are unsafe, and coconut and the color facts were also edited.

**3. Disjointness.**
- Replaced the vacuous `test_bench_prompts_never_in_training_text` with a real check: for every knowledge and consistency item (the seeded bench), the question, the question with the answer filled in, and every text stretch of 30+ characters around a blank must not occur in 3000 prose + 2000 chat training documents (case and punctuation ignored).
- New `test_no_benchmark_phrasing_contains_or_is_contained_in_a_training_phrasing`: for every fact, every bench/question rendering against every train/chat-user/chat-ai rendering, both directions, case- and punctuation-insensitive, with the object both filled and blanked.
- New `test_no_benchmark_template_is_a_reordering_of_a_training_template`: no bench template has the same word multiset as a train or chat-AI template of its relation; no question template matches a chat user template.
- Reworded 9 templates that failed these tests: the four substring leaks the reviewer named (capital_of question vs chat user; animal_sound bench wrapping a train template; color_of bench containing `The {s} is {o}.`; day_after bench starting with the chat reply `Tomorrow is {o}.`, and the same pattern in month_after), the four reorderings the reviewer named (capital_of, landmark_country, planet_order, job_place), and one more the new test found (animal_legs: question vs chat user).

**4. Bench-validity minors.**
- Retired facts: rabbit=carrots (animal_food), dog=kennel (animal_home), sun=yellow, coconut=brown, orange=orange (color_of; also rice, teddy bear, wood, which are not one color), butter / cream / sour cream / ice cream as dairy, sweet=sour, old=young, king=queen, uncle=aunt, boy=girl, fill=empty (opposite_of), frog / toad = tadpole (baby_animal), plus deer (fawn vs calf), mouse and rat (pup vs kitten) and rabbit (bunny / kit / kitten) from baby_animal, meerkat from animal_legs (stands on two legs, has four), Panama, Iceland and Trinidad and Tobago from continent_of, taxi and tram from vehicle_travel, `jet` renamed `jet plane`. horse=barn became horse=stable. A test (`RETIRED_FACTS`) keeps them out.
- Polysemy: `kiwi` is now `kiwi bird` (animal relations) and `kiwi fruit` (food_group); the protein foods are `chicken meat`, `turkey meat`, `fish fillet`, `salmon fillet`, `tuna steak`, `cod fillet`, `trout fillet`; `orange` is no longer a color subject. Test: no subject is both an animal-relation subject and a food_group subject.
- Facts whose subject equals or contains the answer (or the reverse), case-insensitive, are kept in training text but never asked: Mexico, Luxembourg, Guatemala, Panama, Kuwait, Andorra, Tunisia (capital); South Africa and the Central African Republic (Africa); bluebird, bluebell, blueberry; fruit bat; every fish whose name ends in "fish"; hummingbird and kiwi bird (bird); pig / piglet, chicken / chick, duck / duckling; and every element whose symbol is a substring of its name (H, He, C, N, O, ...), which leaves the 23 elements whose symbol is not readable from the name (Fe, Au, Na, Ag, ...). In total 52 facts of the safe relations are never asked. Test: no bench item has an answer inside its subject or the reverse.
- `baby_animal` is now `falsifiable: false`. Also removed from false-fact reach: taxi, tram, jet (water taxi, tram on the road, jet ski). 11 relations and 767 facts stay falsifiable.
- `load_kb()` returns an immutable `KB`: `facts` and `subjects` are tuples, `relations` is a read-only mapping, lookups return copies. `load_kb_from(directory)` builds a KB from any data folder; every malformed input raises `KBDataError` (a `ValueError`) naming the file, the relation and the key (missing key, wrong type, two objects for one subject, bad template placeholder, invalid JSON, missing `relations`).

## New per-relation counts (seeded bench: `skill_rng("knowledge", "bench")`, `skill_rng("consistency", "bench")`)

| relation | facts | MC | exact | consistency groups |
|---|---|---|---|---|
| animal_class | 157 | 16 | 5 | 4 |
| animal_legs | 136 | 16 | 5 | 4 |
| baby_animal | 35 | 15 | 5 | 3 |
| capital_of | 148 | 16 | 5 | 4 |
| continent_of | 169 | 16 | 5 | 4 |
| day_after | 7 | 3 | 4 | 3 |
| element_symbol | 56 | 15 | 4 | 3 |
| landmark_country | 36 | 16 | 5 | 3 |
| month_after | 12 | 8 | 4 | 3 |
| planet_order | 8 | 4 | 4 | 3 |
| shape_sides | 14 | 10 | 4 | 3 |
| vehicle_travel | 24 | 15 | 0 | 3 |
| **total (12 relations)** | | **150** | **50** | **40** |

The ten unsafe relations (country_language 70, animal_sound 32, animal_home 63, animal_food 74, animal_group 22, color_of 107, food_group 99, opposite_of 75, job_tool 43, job_place 40) have no bench items. Knowledge base now: 1427 facts over 22 relations (was 1478).

## Tests

New or replaced tests in `ml/tests/test_kb_facts.py` (62 tests now): real train/bench containment check; exhaustive phrasing containment (both directions, filled and blanked); word-multiset reordering; unsafe flags exactly as listed and `exact_safe` only false for vehicle_travel; known-confusable table over 8 seeded benches (no item from an unsafe relation, no confusable pair in any option set); exact items only from safe relations; retired facts stay retired; unsafe relations still produce training text (at least 5 facts each in 3000 prose docs); no bench item gives its answer away; the leaking facts are still in the KB; no polysemous subject; KB is immutable; data loads from another directory; each of the 8 relation keys reported when missing (with file and relation); duplicate subject, bad placeholder, bad flag type, invalid JSON, missing `relations`. The bench-selection and consistency tests were updated to the 12 safe relations (each used, 7 to 25 items per relation).

RED evidence (new tests against the old data/templates):
```
$ python -m uv run pytest tests/test_kb_facts.py -q
FAILED test_bench_prompts_never_in_training_text      leak: ('knowledge-0018', 'Which city is the capital of Montenegro?')
FAILED test_no_benchmark_phrasing_contains_or_is_contained_in_a_training_phrasing   (animal_sound, 'The word for the sound of the {s} is ...', 'The sound of the {s} is ...') and 3 more
FAILED test_no_benchmark_template_is_a_reordering_of_a_training_template   ('capital_of', 'The city of {o} is the capital of {s}.')
```
A template scan over all facts listed exactly the 9 templates above; after rewording it is empty.

GREEN:
```
$ python -m uv run pytest tests/test_kb_facts.py -q   ->  62 passed in 6.06s
$ python -m uv run pytest -q                           ->  587 passed, 3 deselected in 39.93s (no warnings)
$ python -m uv run ruff check .                        ->  All checks passed!
```

Mutation checks (temporary, restored): making `_gives_away_answer` return False fails `test_benchmark_items_never_give_the_answer_away`; flipping one `mc_safe` flag to true fails `test_overlapping_relations_are_flagged_unsafe_for_benchmarks` and the safe-relation test.

## Notes

- bench-v1 now holds 12 relations only (no color, food, home, job or sound questions). If more topic coverage is wanted later, add relations whose answers cannot overlap rather than loosening the flags.
- Element symbols: asking only the 23 non-derivable ones removes a letter-matching shortcut, but makes the element items harder than the rest. The planet items are all 8 planets once each (4 MC + 4 exact).
- The word-multiset and containment checks are strict on purpose; a later template edit that trips them is a real leak, not a test problem.
- The data files were regenerated from scratch scripts kept in the scratchpad (not committed); the JSON files remain the source of truth.


---

# Fix round 2 (confusable objects in safe relations)

Status: DONE. Commit: `ae6fbce fix(ml): confusable objects in safe relations never become wrong options; free answers accept listed forms; never-false pairs`.

## What changed

- **(a) Confusable mechanism.** Three new required JSON keys per relation, loaded into `Relation` and checked at load time (unknown objects, groups of fewer than 2, empty lists, wrong types raise `KBDataError` with file and relation):
  - `confusable`: groups of objects that can be mistaken for each other. `KB.distractors` never offers an object that is in a group with the true object, and never two objects of the same group together.
  - `also_accepted`: extra forms a free answer may use for an object, per true object (for example `puppy` also accepts `pup`, but `pup` does not accept `puppy`: that word is only for dogs). The reviewer's "synonyms in the same group" had to be directed: accepting every member of a group would accept "duckling" for a chick, or "puppy" for a seal.
  - `never_false`: (subject, object) pairs that are not clearly false.
- New `KB.wrong_objects(fact)` is the single definition of "clearly wrong": not the true object, not anything confusable with it, not an accepted form, not a `never_false` pair for the subject. `distractors` and `plan_false_facts` both draw from it. `KB.accepted_answers(fact)` returns the true object followed by its accepted forms.
- **(b) Populated.**
  - `baby_animal` confusable groups: `puppy/pup`, `cub/pup`, and `chick/duckling/gosling/cygnet` (chick is also said for the young of ducks, geese and swans). `also_accepted`: `puppy` -> `pup`. There is no `kit` object, so no kitten/kit group.
  - Scan of the other 11 safe relations: no other pair of *objects* is interchangeable (continents, capitals, countries, class names, number words, shape/planet/day/month names and symbols are distinct answers). I found only extra *forms* of one answer and added them as `also_accepted`: digits for animal_legs (`0` and `none` for zero, `2`, `4`, `6`, `8`), shape_sides (`3` to `10`) and `1st` to `8th` for planet_order; `Kiev`, `Ulan Bator`, `Berne`, and `Washington D.C.` / `Washington, DC` / `Washington, D.C.` / `Washington` for capital_of; `Turkiye` and `Russian Federation` for landmark_country.
- **(d) Exact items** now carry `kb.accepted_answers(fact)` (for example the dog item accepts `puppy` and `pup`; a legs item accepts `four` and `4`). The seeded bench has 13 such items among its 50.
- **(e)** `RETIRED_FACTS` now also lists Niger, Burkina Faso, Guinea and Togo (country_language).
- **(f)** `never_false` pairs: `lizard` / `zero` (animal_legs; legless lizards exist), and for landmark_country `Hagia Sophia` / `Greece` (a church of that name stands in Thessaloniki) and `Mount Olympus` / `Turkey` (Bithynian Olympus). Such a pair is never offered as a wrong option and never used as a false fact, so MC items about a lizard can no longer offer "zero" either.

The coordinator's reproduction now reads: `np.random.default_rng(10)` -> knowledge-0069 "What name is given to the young of the dog?" options `puppy, joey, calf, cygnet`. Over 300 seeds no knowledge item offers `puppy` and `pup` together (0 hits, was 37).

## Tests (72 in `ml/tests/test_kb_facts.py`, 10 new or changed)

- `test_no_option_set_offers_two_confusable_objects_across_300_seeds`: for 300 seeds of the knowledge bench and of the consistency groups, no MC option set contains two objects of one confusable group (the groups in the JSON plus the independent `CONFUSABLE_OBJECTS` table, which now also lists the `baby_animal` pairs, a safe relation).
- `test_a_wrong_option_is_never_confusable_with_the_answer`: for dog, seal, bear, duck, goose, chicken and lizard, 300 seeded draws of distractors.
- `test_distractors_never_offer_two_mutually_confusable_objects`: every group of every relation.
- `test_free_answers_accept_the_listed_forms` (including that a seal's `pup` does not accept `puppy`); `test_accepted_forms_are_never_offered_as_wrong_options_or_swaps`.
- `test_never_false_pairs_are_never_offered_or_swapped`: distractors over 100 seeds and false-fact plans of 400 facts over 100 seeds; every swapped object is in `wrong_objects`.
- `test_mistakes_in_the_confusable_data_are_reported`; the missing-key test now also covers `confusable`, `also_accepted`, `never_false`.
- `RETIRED_FACTS` extended (4 country_language rows, and baby_animal deer/mouse/rat/rabbit from round 1 were already there).

RED: with the `confusable` data emptied for baby_animal (restoring the old behaviour) the new tests fail:
```
FAILED test_benchmark_items_only_use_safe_relations_and_never_offer_a_confusable_pair
FAILED test_no_option_set_offers_two_confusable_objects_across_300_seeds
FAILED test_a_wrong_option_is_never_confusable_with_the_answer
3 failed, 69 passed
```
GREEN:
```
$ python -m uv run pytest tests/test_kb_facts.py -q   ->  72 passed in 8.78s
$ python -m uv run pytest -q                           ->  597 passed, 3 deselected in 44.55s (no warnings)
$ python -m uv run ruff check .                        ->  All checks passed!
```

## Per-relation counts of the seeded bench (unchanged by this round)

| relation | facts | MC | exact | consistency groups |
|---|---|---|---|---|
| animal_class | 157 | 16 | 5 | 4 |
| animal_legs | 136 | 16 | 5 | 4 |
| baby_animal | 35 | 15 | 5 | 3 |
| capital_of | 148 | 16 | 5 | 4 |
| continent_of | 169 | 16 | 5 | 4 |
| day_after | 7 | 3 | 4 | 3 |
| element_symbol | 56 | 15 | 4 | 3 |
| landmark_country | 36 | 16 | 5 | 3 |
| month_after | 12 | 8 | 4 | 3 |
| planet_order | 8 | 4 | 4 | 3 |
| shape_sides | 14 | 10 | 4 | 3 |
| vehicle_travel | 24 | 15 | 0 | 3 |
| **total** | 1427 | **150** | **50** | **40** |

## Notes

- Exact answers now accept digit and spelling variants, so a downstream matcher should compare the generated first line (stripped, case-insensitive) against every entry of `ExactItem.answers`, not only the first.
- A wrong option set is now chosen by a greedy pass over a seeded permutation, so the option sets of the committed seeded bench differ from round 1 (same relations, same counts).
- The `kb_data` JSON files stay the source of truth; the authoring scripts remain outside the repo.
