# SDD ledger — plan: docs/superpowers/plans/2026-11-03-playtest-pack.md

Spec: docs/superpowers/specs/2026-11-03-playtest-pack-design.md (owner interview 2026-11-03, C31–C35).
Executor: inline (executing-plans). Branch m2-two-friends. Base: 40a7dd6.

Pre-flight: shared interfaces — T3 GameItem → T4 play(); T4 worker schemas → T5/T12 (job types minigame/keynote/appReview); T5 commands → T6–T10 (sim) and T12–T15; T6 sim-5 state fields → T7–T10 (each adds its own default in upgrade.ts); T10 plan extras → T11 (ML expert prep) and T13 (server config). No conflicts found; each later task reads the earlier task's Interfaces block.

Task 1: complete (fair battles; tests: ml test_battles 10/10 + worker battle 2/2, npm test 508/508, check clean but 1 old warning). Diagnosis: battles asked items as chats at temperature 0.8 with 96 tokens, while the benchmark asks them in their own format, greedily, at the item's length; pass/fail then tied two wrong answers. Fix: `battle_prompt` asks like the benchmark; `score_challenge` = 1 on a pass, else 0.5 × closeness (number distance, word-overlap F1 with the answer or reference, share of a function's tests passed); `expected` travels to the Battle window ("Right answer: …", Right/Close/Miss per reply, "(no answer)" for an empty one); drawMargin 0.25 → 0.1.
Task 1: Ruling: `expected` is optional in sim BattleRound (absent in sim-4 logs) and nullable in the view — old logged battleResolved inputs replay unchanged — cost if wrong: none.
Task 2: complete (posts pause; tests: broker 12/12, gateway 10/10, playground 6/6, npm test 511/511). Client sends `inPlayground {worldId}` on entering the Playground and every 15 s; the gateway passes it on only for a player watching that world; the broker leases no `feedback` job (list `HELD_IN_PLAYGROUND`, appReview joins in T13) for a world signed within 30 s; the 5 s lease sweep resumes them. The Playground says "Customer posts are paused while you're here…" and, after 6 messages, suggests a new conversation.
Task 2: Ruling: the client message is `inPlayground`, not `presence` — the sim already has a `presence` system event (online/offline) — cost if wrong: a rename.
