# SDD ledger — plan: docs/superpowers/plans/2026-11-04-coaching.md

Executed inline (superpowers:executing-plans), committing and pushing after every task; one self-review at the end (the owner prefers no helper agents). Spec: docs/superpowers/specs/2026-11-04-remaining-features-design.md §3 (C45).

Pre-flight:
- Task 1 → Task 2: Task 1 produces the system event `lessonsCounted {companyId, count}` and `plan.coaching`; Task 2 submits the event after storing or deleting and reads `plan.coaching` in `toTrainRunConfig`. Consistent.
- Task 1 → Task 3: Task 1 adds `me.company.lessons` and `previewPlan(..., lessons)`; Task 3's card shows the count and locks the switch with the same refusal copy. Consistent.
- Task 2 → Task 3: Task 2 adds client messages `lesson` and `lessonDelete` and query `lessons`; Task 3 sends them. Consistent.
- Stage 2 ended at sim-7, so this is sim-8 (as the plan's Global Constraints say).
Task 1: Ruling: `LESSONS_MIN = 5` and `LESSONS_MAX = 200` are sim constants (exported) rather than balance values, since the spec fixes them for every balance; coaching is free, so the price doesn't change — cost if wrong: a balance field later.
Task 1: complete (tests: sim coaching 4/4; npm test 747/747; check clean). sim-8: `CompanyState.lessons` (0 for new and upgraded companies), system event `lessonsCounted {companyId, count}` (players' companies only, whole 0..200, else "That isn't valid."), `TrainingPlan.coaching` (default false; upgraded runs' plans false), `previewPlan(b, parent, plan, lent, loved, lessons)` refuses coaching under 5 ("Your AI needs at least 5 lessons first. You have {n}."), `me.company.lessons` in the view only; `SIM_VERSION = "sim-8"`, `UPGRADABLE` gains sim-7.
