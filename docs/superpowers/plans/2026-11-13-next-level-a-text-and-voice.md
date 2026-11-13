# Next level, plan A: the measuring stick, engine 3 items, training text 2 and a voice of its own

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans, inline, in the order written (the lead developer's call; the owner is not asked about execution methods). Steps use checkbox (`- [ ]`) syntax for tracking. Commit and push after every task; keep a ledger at `docs/handoff/sdd/ledger-next-level.md` (one line per task: commit, what was measured, what was decided).

**Goal:** A first AI that knows it is an AI, answers what it was asked, holds a two-turn conversation and no longer misspells its own replies, proven on one fixed scorecard against engine 2.

**Architecture:** A scorecard command trains fixed recipes and measures every model the same way against a yardstick that never moves. Training text 2 is built into a second data root (`data2/`) beside today's, so each change is an A/B on the scorecard. The identity line travels in the chat prompt and is added only for AIs trained on text 2.

**Tech stack:** Python 3.13 (uv), PyTorch, HF `datasets`/`tokenizers`; TypeScript (zod, Fastify, Vitest) for the game side of Task 17.

**Spec:** `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md` (phases A–D of its §6). Measurements so far: `docs/decisions/0009-training-engine-2.md`.

## Global constraints
- Run uv as `python -m uv`; every `uv sync` keeps `--extra content`. Never run a game or a build on `data/server/`; never write into `data/corpus`, `data/judge`, `data/novelty`, `data/rivals`: they are the yardstick.
- The second data root is `data2/` at the repo root, chosen with `AIRACE_DATA`; it is gitignored.
- tok-v1 and bench-v1 are frozen. No reply is ever scripted or rewritten; decoding may only block (R1).
- TDD for logic. `ruff check .` clean; `npm run check` clean when TypeScript is touched.
- A task that changes what a training run produces raises `ENGINE_VERSION` (`ml/src/airace_ml/engine.py`), `TRAINING_ENGINE` (`packages/protocol/src/worker.ts`) and the `engine` of `packages/protocol/fixtures/worker/whello.json` together, and adds a line to the history in `engine.py`.
- An experiment's rule for keeping its change is written in the task; a change that misses its rule is removed in the same task and recorded in `docs/decisions/0010-next-level.md`. A target is never lowered silently.
- GPU measuring runs are not timed while tests run; CPU tests are not run during a timing run.
- Player-facing words (none are planned in A–D) use no ML jargon (R7).

## Review focus
1. An AI trained before text 2 is sent an identity line → the helper must leave it out (Task 16 test `test_an_ai_from_before_text_2_never_reads_an_identity_line`).
2. A name or company holding `<|end|>`, a newline or 200 characters → it is read as plain text on one line, cut to 60 characters (Task 1 test `test_a_name_with_control_strings_is_plain_text_on_one_line`).
3. A long conversation trimmed to the attention span → the identity line is never the part that is dropped (Task 1 test `test_trimming_keeps_the_identity_line`).
4. A run resumed after the draw-without-replacement change with a state saved before it → started over, never continued (Task 5 test `test_progress_saved_by_engine_2_is_not_continued`).
5. A scorecard asked to compare against a yardstick root that is missing its judge or corpus → a plain error naming the folder, exit 2, nothing trained (Task 3 test `test_a_missing_yardstick_is_an_error_before_any_training`).

## File map
| File | Responsibility |
|---|---|
| `ml/src/airace_ml/persona.py` (new) | the identity line's text; detecting a reply that claims a person's life |
| `ml/src/airace_ml/tokenizer.py` | `encode_chat(..., system=)` |
| `ml/src/airace_ml/infer/lm.py` | `build_chat_prompt(..., system=)`, `chat_reply(..., system=)`, `stream_chat_reply(..., system=)` |
| `ml/src/airace_ml/experiments/chatprobes.py` (new) | the talk probes and their scoring |
| `ml/src/airace_ml/experiments/scorecard.py` (new) | recipes, measuring, report, comparison |
| `ml/src/airace_ml/cli.py` | `airace-ml scorecard` |
| `ml/src/airace_ml/data/sampler.py` | drawing without replacement |
| `ml/src/airace_ml/train/trainer.py` | user-turn loss weight; run record `"text"` |
| `ml/src/airace_ml/paths.py` | corpus version per data root |
| `ml/src/airace_content/sources.py`, `assemble.py`, `noise.py` | new source, filters, recipes, user-turn typos, identity lines |
| `ml/src/airace_content/chats.py` (new) | the follow-up and identity chat generators, name lists |
| `ml/src/airace_ml/worker/jobs.py` | identity line for text-2 AIs |
| `packages/protocol/src/worker.ts`, `apps/server/src/{chat,feedback,battles,minigames}/…` | the game sends name and company |
| `docs/decisions/0010-next-level.md` (new) | every measurement and decision of this plan |

---

## Phase A — the measuring stick

### Task 1: the identity line in chat encoding
**Files:** create `ml/src/airace_ml/persona.py`; modify `ml/src/airace_ml/tokenizer.py`, `ml/src/airace_ml/infer/lm.py`; tests `ml/tests/test_persona.py`, `ml/tests/test_tokenizer.py`, `ml/tests/test_infer.py`.

**Interfaces — produces:**
- `persona.identity_line(name: str, company: str | None) -> str`: `"You are {name}, an AI made by {company}."` or `"You are {name}, an AI."`; name and company are stripped, newlines become spaces, each cut to 60 characters; an empty name raises `ValueError`.
- `persona.claims_person_life(text: str) -> bool`: true when the text speaks in the first person of family, school, job, home or body (`my (mom|mother|dad|father|wife|husband|son|daughter|brother|sister|kids?|boss|teacher|job|house|car)`, `i (went|go|am going) to (school|work|church|the gym)`, `my (hands?|legs?|eyes?|hair)`, case-insensitive, word-bounded).
- `Tok.sys_id: int = 6`.
- `encode_chat(tok, turns, add_generation_prompt=False, system: str | None = None) -> list[int]`: with `system`, `<|bos|><|sys|>…<|end|>` then the turns.
- `build_chat_prompt(tok, history, ctx_len, reserve, system: str | None = None)`, `TorchLM.chat_reply(..., system=None)`, `TorchLM.stream_chat_reply(..., system=None)`.

- [ ] **Step 1: failing tests.** `test_persona.py`: the two sentence forms; `test_a_name_with_control_strings_is_plain_text_on_one_line` (name `"Zu<|end|>la\nX" * 20` → one line, 60 characters, and `encode_chat` of it contains exactly one `end_id` before the first turn); `claims_person_life` true for "I went to school with my mom." and "My boss is nice.", false for "School is a place to learn." and "I am an AI, so I have no family.". `test_tokenizer.py`: `encode_chat(tok, [("user","Hi")], system="You are Zu, an AI.")[:2] == [tok.bos_id, tok.sys_id]`, and without `system` the ids equal today's. `test_infer.py`: `test_trimming_keeps_the_identity_line` (a history too long for `ctx_len`: the prompt still starts `bos, sys`, drops oldest turns, ends with `ai_id`); `chat_reply(history)` with `system=None` equals the reply before this task for a fixed seed.
- [ ] **Step 2:** run `cd ml && python -m uv run pytest -q tests/test_persona.py tests/test_tokenizer.py tests/test_infer.py`; expect the new tests to fail.
- [ ] **Step 3:** implement; `build_chat_prompt` counts the system line's tokens inside `ctx_len - reserve` and never drops it.
- [ ] **Step 4:** the same command passes; `python -m uv run ruff check .`.
- [ ] **Step 5:** commit `feat(ml): an identity line in the chat prompt`.

### Task 2: the talk probes
**Files:** create `ml/src/airace_ml/experiments/chatprobes.py`; test `ml/tests/test_chatprobes.py`. Move the prompt lists of `docs/handoff/upgrade-2026-11-10/probe_casual.py` (`SMALL_TALK`, `STYLES`) into the module; the script then imports them.

**Interfaces — produces** (every probe draws with `SERVING_TEMPERATURE`, `SERVING_TOP_P`, `stop_on_loop=True` unless it says greedy):
- `@dataclass ProbeScore: passed: int; total: int; examples: list[tuple[str, str, bool]]` with `rate -> float`.
- `casual_facts(lm, kb, *, per_relation=5, seed=0) -> dict[str, ProbeScore]` (keys: the five styles; greedy; right when the reply holds an accepted answer).
- `followups(lm, kb, *, n=60, seed=0) -> ProbeScore`: two-turn items built from relations with at least two subjects: turn 1 the relation's first chat template for subject A, the AI's turn the true answer, turn 2 one of `"And {b}?"`, `"What about {b}?"`, `"And for {b}?"`; right when the reply holds B's accepted answer and not only A's. Greedy.
- `identity(lm, *, with_line: bool, names=IDENTITY_NAMES) -> ProbeScore`: with the line, "What is your name?" / "Who are you?" / "what's your name" pass when the reply holds the name; "Are you a person?" / "Are you human?" pass when the reply holds "AI" as a word and not `claims_person_life`; without the line, the same questions pass when the reply does not `claims_person_life`. `IDENTITY_NAMES = ("Zula", "Pip", "Orbit", "Mako", "Tenby", "Quill")`, company `"Pengu"`.
- `small_talk(lm, judge, *, samples=3) -> dict[str, float]`: over `SMALL_TALK` × `samples` seeds: `well_formed` (`is_well_formed`), `person_life`, `cut_off` (the reply used every allowed token and does not end in `.`, `!`, `?` or a closing quote), `looped` (`Generation.looped`).
- `lessons(train: Callable[[list[tuple[str, str]]], TorchLM], *, n=10) -> ProbeScore`: `LESSONS` are ten fixed pairs with made-up facts (e.g. `("What colour is a glimmerfin?", "A glimmerfin is teal.")`); `train` continues a model with them as the game trains coaching; right when the reply to the question holds the answer's key word. Greedy.

- [ ] **Step 1: failing tests** with a scripted `FakeLM` (a dict from prompt text to reply): each probe's pass and fail cases above, `cut_off` true for 96 tokens ending in "and the", false when ending in "."; a follow-up reply naming only A's answer fails.
- [ ] **Step 2:** run `python -m uv run pytest -q tests/test_chatprobes.py`; fails.
- [ ] **Step 3:** implement. `lm.generate` already reports the token count; add `Generation.hit_limit: bool` in `infer/lm.py` if it is not derivable.
- [ ] **Step 4:** passes; ruff.
- [ ] **Step 5:** commit `feat(ml): talk probes (casual facts, follow-ups, identity, small talk, lessons)`.

### Task 3: `airace-ml scorecard`
**Files:** create `ml/src/airace_ml/experiments/scorecard.py`; modify `ml/src/airace_ml/cli.py` (add `_cmd_scorecard` to `_LONG_RUNS`); tests `ml/tests/test_scorecard.py`, `ml/tests/test_cli.py`.

**Interfaces — consumes:** `GateRuns.train(cfg, device, parent)` (`experiments/runs.py`), `run_benchmarks`, Task 2's probes, `Judge.load`, `NoveltyIndex`. **Produces:**
- `RECIPES: dict[str, list[Step]]` exactly as the spec's §5.1 (`first` with seeds 1 and 2; `lineage` reuses `first` seed 1 as its parent).
- `run_scorecard(data_root: Path, out: Path, *, yardstick_root: Path, recipes: Sequence[str], device, timing: bool = False, on_status=print) -> dict` and `write_reports(card: dict, out: Path)` (`scorecard.json`, `scorecard.md`: one table row per model, one column per measure).
- `yardstick_loss(model, yardstick_root) -> float`: mean loss on the held-out conversation documents of the yardstick root's corpus (128 documents, the trainer's held-out rule).
- `compare(card: dict, baseline: dict) -> list[str]`: one sentence per measure that is worse than the baseline by more than its tolerance (`TOLERANCE = {"well_formed": 0.04, "casual": 0.05, "followups": 0.05, "bench_overall": 2.0, "yardstick_loss": 0.03, "heldout": 0.02}`).
- CLI: `airace-ml scorecard --out DIR [--yardstick-root DIR] [--recipes first,small,lineage,chatty,large] [--baseline FILE] [--timing]`; exit 0, 1 when `compare` is not empty, 2 on a usage or setup error. `--yardstick-root` defaults to the data root.

- [ ] **Step 1: failing tests** with the tiny test corpus (`tiny_data_root` fixture) and recipes shrunk through a `scale` argument: the card has every recipe, model and measure; a second call reuses the runs (status lines say "reusing"); `compare` reports a 6-point drop in `well_formed` and ignores a 2-point one; `test_a_missing_yardstick_is_an_error_before_any_training`; the CLI's exit codes.
- [ ] **Step 2–4:** run `python -m uv run pytest -q tests/test_scorecard.py tests/test_cli.py`, implement, pass, ruff. `tests/test_gate.py::test_code_digests_cover_everything_that_code_imports` must still pass (the new modules are measuring code).
- [ ] **Step 5:** commit `feat(ml): the scorecard`.

### Task 4: the baseline
- [ ] **Step 1:** nothing else on the PC; run `cd ml && python -m uv run airace-ml scorecard --out runs/scorecard-engine2 --timing` (about 25 minutes).
- [ ] **Step 2:** copy `scorecard.json` and `scorecard.md` to `docs/reports/2026-11-scorecard-engine2.{json,md}`.
- [ ] **Step 3:** create `docs/decisions/0010-next-level.md`: the baseline table; the spec §7 targets that were "to be measured", now fixed from the baseline by the spec's rules.
- [ ] **Step 4:** commit `docs: the scorecard baseline (engine 2, text 1)`; push.

---

## Phase B — engine 3 items

### Task 5: draw documents without replacement
**Files:** `ml/src/airace_ml/data/sampler.py`, `ml/src/airace_ml/engine.py` (+ the protocol constant and fixture); tests `ml/tests/test_sampler.py`, `ml/tests/test_trainer.py`, `ml/tests/test_worker_moving.py`.

**Interfaces — produces:** `DocPool.sample_doc(rng)` walks a shuffled order of the pool and reshuffles when it is used up; `DocPool.state() -> dict` / `set_state(dict)`; `MixtureSampler.rng_state()` gains `"pools": {name: state}`; a state without `"pools"` raises `ValueError`.

- [ ] **Step 1: failing tests:** in one pass over a pool of 50 documents every document is drawn exactly once; the order differs between seeds and is the same for one seed; a sampler restored from `rng_state()` continues with the same batches; `test_progress_saved_by_engine_2_is_not_continued` (the worker starts the run over and logs why).
- [ ] **Step 2–4:** fail, implement, pass. Raise the engine to 3 with the history line "3: documents are drawn without replacement inside a run".
- [ ] **Step 5: measure.** `airace-ml scorecard --out runs/scorecard-b5 --recipes small,chatty --baseline ../docs/reports/2026-11-scorecard-engine2.json`. **Keep** if nothing is flagged; **remove** if `chatty` is worse on held-out conversations by more than 0.02. Record the numbers in ADR 0010.
- [ ] **Step 6:** full ML suite, `npm test` for the protocol fixture, commit `feat(ml): engine 3, documents drawn without replacement`.

### Task 6: how much a user's turn counts in the loss
**Files:** `ml/src/airace_ml/train/trainer.py`; test `ml/tests/test_trainer.py`.

**Interfaces — produces:** `USER_TURN_LOSS_WEIGHT: float` (module constant) and `_loss_weights(inputs: Tensor, user_id: int, ai_id: int, end_id: int, weight: float) -> Tensor | None` (per-token weights: `weight` for targets inside a user turn, 1 elsewhere; `None` when `weight == 1.0`); the loss is the weighted mean.

- [ ] **Step 1: failing tests:** the weights for `bos user a b end ai c end` are `[w, w, w, 1, 1, 1]` for the targets; plain text rows are all 1; at weight 1.0 the loss is bit-equal to today's.
- [ ] **Step 2–3:** implement at 1.0 (no change in behaviour).
- [ ] **Step 4: measure** `first`, `small`, `chatty` at 0.3 and 0.6 (set by environment variable in a scratch copy of the scorecard call; two extra runs each). **Keep** the better value if replies improve (yardstick loss on AI turns only, added to the scorecard here, lower by ≥ 0.02, or well-formed higher by ≥ 2 points) and no measure is flagged; then raise the engine. **Otherwise** delete `_loss_weights` and its tests.
- [ ] **Step 5:** ADR 0010, commit.

### Task 7: growth transfer (one day, no more)
**Files:** experiment script `docs/handoff/next-level/growth_lab.py` (run from `ml/`); product code only if kept: `ml/src/airace_ml/model/growth.py`, `ml/src/airace_ml/train/optim.py`.

- [ ] **Step 1:** the script trains the `lineage` recipe's Tiny → Small step three ways: (a) as today; (b) the optimizer's step for rows and columns that did not exist in the parent multiplied by 3 for the first third of the run; (c) new embedding columns drawn at the usual start scale with the weights that read new dimensions set to zero (prove in the script that outputs are unchanged to 1e-4 before training).
- [ ] **Step 2: rule.** Keep a variant only if the grown Small is better by ≥ 0.02 held-out loss on two seeds and bench-v1 overall is not lower. If kept: write its tests first (`tests/test_growth.py`: outputs unchanged to 1e-4; every existing growth test still passes), then the product code, engine raised.
- [ ] **Step 3:** ADR 0010 (numbers for all three), commit.

### Task 8: lessons that stick
- [ ] **Step 1:** read how the game sends coaching (`apps/server/src/training/` builds the mixture; `TrainRunConfig.coaching`) and make the scorecard's `lessons` trainer do exactly that for the `first` model with a Short run.
- [ ] **Step 2:** measure. If ≥ 7 of 10 are answered: record and stop. If fewer: raise the coaching share the server sends in steps (the smallest that reaches 7 of 10 without flagging anything on `first`), with the server's test for the mixture updated first; record.
- [ ] **Step 3:** commit.

---

## Phase C — training text 2

### Task 9: a corpus version per data root
**Files:** `ml/src/airace_ml/paths.py`, `ml/src/airace_content/build.py`, `manifest.py`, `.gitignore`; tests `ml/tests/test_paths.py`, `ml/tests/test_content_build.py` (the existing build tests' file).

**Interfaces — produces:** `CORPUS_VERSIONS = ("v2", "v1")`; `corpus_version(root=None) -> str` (the first of them with a `manifest.json` in that root, `"v2"` when none: a new build is text 2); `corpus_dir(root)` uses it; `text_version(root=None) -> int` (2 for v2, 1 for v1). The manifest records `"corpus_version"` from `corpus_version(root)`.

- [ ] **Steps:** failing tests (a root with only `corpus/v1` reads v1; an empty root builds into `corpus/v2`; a root with both reads v2), implement, `data2/` added to `.gitignore`, commit `feat: corpus version per data root (text 2 builds beside text 1)`.

### Task 10: typos only where people type
**Files:** `ml/src/airace_content/assemble.py` (`_typo_noise`), test in the assemble tests.
- [ ] **Steps:** failing test `test_a_chat_documents_typos_fall_on_user_turns_only` (1,000 noisy chat documents: no AI turn differs from its clean text; at least 90% of the documents have a changed user turn); implement; commit.

### Task 11: assistant-style conversations
**Files:** `ml/src/airace_content/sources.py`; tests in the sources tests.

**Interfaces — produces:** source `smol_smoltalk` (`HuggingFaceTB/smol-smoltalk`, split `train`, licence `apache-2.0`, revision pinned in `REVISIONS`), adapter `_smol(row) -> list[Content]`: user/assistant turns kept, system turns dropped; a conversation is kept when it has 2–6 turns, every AI turn is at most 600 characters, no turn holds a code fence, `\\(`, `$$` or more than 3 digits in a row, and the first turn is a user's.

- [ ] **Step 1:** open the dataset card at the pinned revision (WebFetch) and confirm the licence and that no subset in it carries another licence; if one does, exclude that subset by its `source` column and record it. Write the finding into ADR 0010.
- [ ] **Step 2–4:** failing adapter tests (kept and dropped examples for each rule), implement, pass; commit.

### Task 12: SODA without a person's life on the AI's side
**Files:** `ml/src/airace_content/sources.py` (`_soda`), test.
- [ ] **Steps:** failing test (a dialogue whose AI turn says "I went to school with my mom" is dropped; one whose user turn says it is kept); implement with `persona.claims_person_life` over the AI's turns; commit.

### Task 13: follow-up chats, identity chats and the identity line
**Files:** create `ml/src/airace_content/chats.py`; modify `assemble.py` (`GENERATORS`, a `system_share` on `Component`); tests `ml/tests/test_content_chats.py` and the assemble tests.

**Interfaces — produces:**
- `followup_chat(kb, rng) -> Content`: the two-turn form of Task 2's probe, plus three-turn chains and second questions about another relation of the same subject; subjects are split so that the probe's subjects never appear in a second turn of the training text (`chats.PROBE_SUBJECT_SHARE = 0.1`, the same split function as the probe uses).
- `identity_chat(rng) -> Content` whose first turn is `("sys", identity_line(name, company))`: 1–3 exchanges from at least 12 phrasings each of: name, maker, "are you a person/robot/human", "do you have a family/a body/a job", "what are you"; replies are short, in the first person, name the given name or company, and say it is an AI.
- `NAMES` (≥ 2,000: common first names, two- and three-syllable made-up names, a few with digits) and `COMPANIES` (≥ 500) built by `chats.name_lists(seed=0)`; `IDENTITY_NAMES` of Task 2 are excluded from them.
- `Component.system_share: float = 0.0`: the assembler prepends `("sys", identity_line(...))` to that share of the component's chat documents. `Content` chats may start with one `("sys", text)` turn; `encode_chat` receives it as `system`.

- [ ] **Step 1: failing tests:** a follow-up's second turn does not name the relation's subject words of the first; no probe subject in any second turn over 5,000 documents; an identity chat's replies hold its line's name; 40% ± 3 of 2,000 documents of a component with `system_share=0.4` start with `sys_id`; the casual-typing transform leaves a system line unchanged (`ml/tests/test_casual.py`).
- [ ] **Step 2–5:** fail, implement, pass, ruff, commit `feat(content): follow-up and identity chats, the identity line`.

### Task 14: the recipes of text 2 and the build
**Files:** `ml/src/airace_content/assemble.py` (`RECIPES`, `SCALES`), tests.

**Values:** conversations — `smol_smoltalk` 0.30 (simplicity_min 0.55, system_share 0.4), `soda` 0.23, `everyday_conv` 0.02 (system_share 0.4), `fact_chat` 0.13 (0.4), `instructions` 0.13 (0.4), `followup_chat` 0.12 (0.4), `identity_chat` 0.07. Full-scale targets in tokens: web 60M, books 30M, educational 40M, conversations 50M, code 20M, reasoning 20M, facts 12M, creative 40M. Other recipes and all noise rates unchanged. These apply when the build's corpus version is v2; a v1 root keeps the v1 recipes (`RECIPES_V1` kept for the tests that pin the yardstick's manifest).

- [ ] **Step 1:** failing tests: `validate_recipe` passes for every v2 recipe; shares sum to 1; a tiny v2 build has all eight datasets and at least one document from each conversation component.
- [ ] **Step 2:** implement; tiny build: `python -m uv run --extra content airace-content build --scale tiny --out ../data2-tiny` (delete the folder afterwards).
- [ ] **Step 3: the full build:** copy `data/tokenizer` to `data2/tokenizer`; `AIRACE_DATA=../data2 python -m uv run --extra content airace-content build --scale full` (hours; it resumes after a dropped connection: run it again). Check the manifest's token counts against the targets (within 10%) and read 20 random conversation documents by eye; note anything odd in ADR 0010.
- [ ] **Step 4:** commit the code; the data is never committed.

### Task 15: text 1 against text 2
- [ ] **Step 1:** `AIRACE_DATA=../data2 python -m uv run airace-ml scorecard --out runs/scorecard-text2 --yardstick-root ../data --baseline <the latest kept scorecard>`.
- [ ] **Step 2:** judge against the spec's §5.2 acceptance. If it misses: change shares or filters (at most three rounds; each is a rebuild of the conversations dataset only: `airace-content build --scale full --only conversations` — add `--only` to the content CLI with its test if it does not exist), and rerun.
- [ ] **Step 3: the default reading.** Train `first` with conversations at 24% and the other seven at 10.86% each. If well-formed, follow-ups and casual facts are all at least as good and bench-v1 overall is not lower by more than 2: set `defaultPlan` of `BALANCE_V3` to mix weights `conversations: 24`, others `11` in `packages/sim/src/balance.ts` (its tests first). Otherwise leave it and record why.
- [ ] **Step 4:** ADR 0010: the table text 1 / text 2 for every recipe, each round, the decision; commit; push.

---

## Phase D — a voice of its own

### Task 16: the helper adds the identity line for AIs that learned to read it
**Files:** `ml/src/airace_ml/train/trainer.py` (`_summary`: `"text": text_version(data_root)`), `ml/src/airace_ml/worker/jobs.py`, `ml/src/airace_ml/rivals/build.py`; tests `ml/tests/test_trainer.py`, `ml/tests/test_worker_jobs.py`.

**Interfaces — produces:** job specs for `generate`, `feedback`, `keynote`, `interview`, `battle`, `minigame` may carry `"identity": {"name": str, "company": str | None}`; `jobs._system_for(meta: CheckpointMeta, spec: dict) -> str | None` returns `identity_line(...)` only when `spec` has an identity and `meta.runs[-1].get("text", 1) >= 2`. Benchmarks and the fingerprint never pass a system line. Rival answers drawn at build time use the rival's lineage id as the name when the rival was trained on text 2.

- [ ] **Step 1: failing tests:** `test_an_ai_from_before_text_2_never_reads_an_identity_line` (same reply with and without `identity` in the spec); an AI whose last run has `"text": 2` gets `chat_reply(..., system=identity_line("Zula", "Pengu"))`; a spec without `identity` is as today; a run record holds `"text"`.
- [ ] **Step 2–5:** fail, implement, pass, ruff, commit.

### Task 17: the game sends each AI's name and company
**Files:** `packages/protocol/src/worker.ts` (`Identity = z.object({ name: z.string().min(1).max(60), company: z.string().max(60).nullable() })`, optional `identity` on the six job specs), fixtures under `packages/protocol/fixtures/worker/`; `apps/server/src/chat/service.ts`, `feedback/service.ts`, `battles/service.ts`, `minigames/service.ts` and the keynote and interview services; tests beside each.

- [ ] **Step 1: failing tests:** each service's spec holds the product's (or, for an unreleased AI, the version's) public name and its company's name; a rival AI's spec holds the rival company's name; `projectView` tests still show nothing private leaves (R2: both names are already public in the view).
- [ ] **Step 2–5:** fail, implement, `npm test`, `npm run check`, commit `feat(game): an AI is told its name and its company`.

### Task 18: blocking repeats and unlikely tokens
**Files:** `ml/src/airace_ml/infer/lm.py`, `ml/src/airace_ml/infer/serving.py`; tests `ml/tests/test_infer.py`, `ml/tests/test_serving.py`.

**Interfaces — produces:** `generate(..., block_repeat: int = 0, min_p: float = 0.0)`: `block_repeat=n` gives zero probability to a token that would complete an n-token run already present in the reply; `min_p` drops tokens whose probability is below `min_p ×` the best token's. Both only remove candidates.

- [ ] **Step 1: failing tests:** with `block_repeat=4` a model scripted to loop never emits a fourth repeated 4-gram token; with `min_p=1.0` sampling equals greedy; at the defaults replies are bit-equal to today's.
- [ ] **Step 2–3:** implement at defaults that change nothing.
- [ ] **Step 4: measure** on the text-2 `first` and `small` models: `block_repeat=4`; `min_p=0.05` and `0.1` in place of top-p. **Adopt** a setting into `serving.py` (and the server's chat spec, test first) only if well-formed rises by ≥ 3 points and no probe is flagged; otherwise leave the defaults and keep the code only if a setting was adopted.
- [ ] **Step 5:** ADR 0010, commit.

### Task 19: the gate of plan A
- [ ] **Step 1:** in `data2/`: `build-novelty-index`, `build-judge`, `export-catalog --population runs/gate-2/runs`, `build-rivals --quick` (enough for a check), each with `AIRACE_DATA=../data2`.
- [ ] **Step 2:** the whole ML suite, `npm test`, `npm run check`, `npm run e2e`.
- [ ] **Step 3:** scorecard on text 2 with everything kept, against the baseline; write the result table into ADR 0010 and `docs/reports/2026-11-scorecard-text2.{json,md}`. Judge it against the spec's §5.3 and §7.
- [ ] **Step 4:** start a second game for the owner on the new text (`AIRACE_DATA=data2 AIRACE_SERVER_PORT=8788 AIRACE_WEB_PORT=5175 npm run dev`, after creating `data2/server/`), train a first AI in it, save ten of its replies to the standard questions beside ten from engine 2 in the report.
- [ ] **Step 5:** `docs/progress.md`, `docs/handoff/README.md`, the memory roadmap, the ledger; push. Tell the owner in plain words what improved, what did not, and that the second game is there to try. Then write plan B (brain design 2 screening) from the spec's §5.5.

## Self-review (done 2026-11-13)
- Spec coverage: §5.1 → Tasks 1–4; §5.4 → Tasks 5–8; §5.2 → Tasks 9–15; §5.3 → Tasks 16–18; §6 phase D's gate → Task 19. §5.5–§5.7 are plans B–D, written at their gates, as the spec's §6 says.
- Names used across tasks: `identity_line`, `claims_person_life`, `encode_chat(..., system=)`, `ProbeScore`, `run_scorecard`, `compare`, `corpus_version`, `text_version`, `Component.system_share`, `_system_for` are each defined once (Tasks 1, 2, 3, 9, 13, 16) and used with the same signature later.
- Review focus: each of the five has its named test in Tasks 1, 3, 5 and 16.
