# SDD ledger — plan: docs/superpowers/plans/2026-10-29-game-layer-phase3.md

Spec: docs/superpowers/specs/2026-10-27-game-layer-design.md (§7-§10, §12-§15) and docs/superpowers/specs/2026-10-28-immersion-design.md.
Ruling: executed inline without a review stop - the owner's go ("lets go ahead with building phase 3") - cost if wrong: rework if the owner disagrees with a plan choice.
Pre-flight:
- T1 -> T3: battle_challenges(skill, n, seed), score_challenge(item, reply, tok, judge, novelty): consistent.
- T2 -> T4/T10: RivalPack/RivalReply zod in packages/protocol/src/rivals.ts: consistent.
- T3 -> T7/T12: BATTLE_SKILLS in protocol world.ts (added in T3); battle progress a/b -> mine/theirs in T12: consistent.
- T4 -> T5-T8: RivalState (counters, focus, lastTrashTalkDay), startCash, packKey: consistent.
- T6 -> T7: world.league.points filled by T7: consistent.
- T10 -> T11/T12: RivalService.modelUrl(packKey): consistent.
Task 1: Ruling: battle challenges also skip any prompt the benchmark, customer pools or story prompts ask (the generators can repeat a prompt across streams) - the plan's 'fresh, never from the scored suite' - cost if wrong: a slightly smaller pool.
Task 1: complete (commits 2f4bd9e..941919b, tests: ml pytest -> 1623 passed, 5 skipped; ruff clean)
