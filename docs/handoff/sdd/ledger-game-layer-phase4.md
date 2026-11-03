# SDD ledger — plan: docs/superpowers/plans/2026-10-29-game-layer-phase4.md

Spec: docs/superpowers/specs/2026-10-27-game-layer-design.md (§7, §10, §11, §13–§15). Owner's go: 2026-11-03 ("continue, start phase 4 also").
Executor: inline (executing-plans). Branch m2-two-friends. Base: f9f20ef. The playtest pack took sim-5, so this plan's "sim-5" is **sim-6** and its "sim-4 world" a **sim-5 world** (note at the top of the plan).

Pre-flight: shared interfaces — T1 `Discovery`/`EvalSignals`/`Breakdown` (protocol worker.ts) → T4 (signals → milestones), T5 (discoveryFound), T9 (server spec/result); T2 interview job + catalog `interviewCards` → T10 (service), T12 (client chips); T3 sim-6 + notebook state → T11 (service), T13 (Data step); T6 interviewScored → T10; T7 Goal/GoalSpec → T11 (world creation), T14 (picker). No conflicts found.

Task 1: Ruling: the plan names `TRAIT_WORDS`; the fingerprint module calls it `TRAIT_PHRASES` (low, high) — used as is — cost if wrong: none.
Task 1: Ruling: tag words cover every bench tag in plain words (a table for all `fam:` and `rel:` tags, not only relations), and `topic:` tags read by category ("facts about animals", "instructions about animals", "animals questions asked different ways"), because the same topic in three categories measures three things — cost if wrong: wording.
Task 1: Ruling: a grammar pair has no reply to quote, so a grammar finding carries no evidence (null) — cost if wrong: grammar discoveries show no example.
Task 1: Ruling: trait changes need a peer spread above 0 and a non-zero move (with balance-test's minZ 0 every trait would otherwise "change") — cost if wrong: none.
Task 1: Ruling: `signals` is nullable with a null default and `breakdown`/`discoveries` default empty in `EvalJobResult`, so a result from an older worker still parses — cost if wrong: none.
Task 1: complete (commit eaf2bf2; tests: ml test_discoveries 10/10, worker eval 2/2; protocol 79/79). `find_discoveries` in `ml/src/airace_ml/discoveries.py`; `eval_signals` counts well-formed fingerprint replies (judge only; null without), free-answer facts right and written functions passing.
Task 2: complete (commit eaf2bf2; tests: ml test_interviews 4/4, worker interview 2/2, catalog 1 new; protocol catalog 5/5). 24 cards (12 checked: 3 each knowledge/pattern/instruction/coding on the "interviews" stream, never a bench prompt; 12 open, the plan's copy); `score_answer`; job `interview` (priority 1, worker slot 1, older workers 0).
Task 2: Ruling: the catalog version goes to catalog-v3 so the dev launcher rewrites an installed v2 catalog with the cards (otherwise the owner's game would never show any); `interviewCards` still defaults to [] — cost if wrong: one catalog rewrite on the next `npm run dev`.
Task 2: Ruling: checked cards keep their generator prompt verbatim ("Question: …\nAnswer:"), as customers ask them, so the card's check scores exactly what the AI was asked — cost if wrong: chips read a little formal.
Side fix (found by the e2e, before Phase 4 work): `Tok.decode` turns byte 0 into U+FFFD — PostgreSQL text/jsonb can't store NUL, so a tiny model's reply holding one lost its whole feedback batch ("unsupported Unicode escape sequence") — commit cc05a2b. The e2e's Games pill click now targets the label (Playwright can't click a radio its label covers); e2e passed (2.3 min).
Task 3: Ruling: the notebook reaches the world as the system event `notebookSaved {companyId, hash, words}` instead of a player command — players can't send system events at all, so no gateway exception is needed — cost if wrong: none.
Task 3: Ruling: `TrainingPlan.mix` is a strict object of the eight datasets plus `notebook` (default 0), so plans logged before sim-6 replay unchanged — cost if wrong: none.
Task 4: Ruling: firstProfit, gradeA and beatArch are read daily from the kept report cards and battles (not at the moment of the month close or battle), so an upgraded world catches up on them the next day, as the plan's Review Focus 5 asks for state-based milestones — cost if wrong: a milestone dated the day after.
Task 4: Ruling: numberOne's detail is the skill in plain words (SKILL_WORDS) — cost if wrong: none.
