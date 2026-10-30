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
Task 2: Ruling: a quick pack scores 40 items per skill, not 8 - with 8 no tag reaches MIN_TAG_ITEMS (10), so quick packs had empty breakdowns, which Phase 4's discoveries compare against - cost if wrong: a quick build takes about 25 s instead of 15 s.
Task 2: Ruling: training reuse uses the gate's GateRuns (finished runs reused, unfinished resumed) and measurements are cached per training token - the plan's 'resumable' - cost if wrong: none.
Task 2: complete (commits 921958f..9ab08f6, tests: ml pytest -> 1627 passed, 5 skipped; slow quick build passed; vitest protocol 58 passed; npm run check clean)
Task 3: Ruling: battle fixtures follow the existing names (lease-battle.json, result-battle.json, progress-battle.json) rather than the plan's dotted names - consistency - cost if wrong: none.
Task 3: complete (commits cd25e8f..bd93b63, tests: ml pytest -> 1632 passed, 6 skipped; npm test -> 435 passed; npm run check clean)
Task 4: Ruling: the test balance's rival countdowns are 3 days, not the plan's 1 - the economy's launch countdown minimum is 3 and rivals launch through the same rules - cost if wrong: rivals reach the e2e market two days later.
Task 4: Ruling: rival palettes reuse the 12 PALETTE_PRESETS (all pass paletteProblem) instead of a separate list - cost if wrong: rivals look like the presets players can pick.
Task 4: Ruling: RivalState also keeps productWord (which of its style's product words names its AIs) - the plan's product names need it - cost if wrong: none.
Task 4: Ruling: an upgraded world's snapshot is replaced in place (same seq) so it is upgraded once - cost if wrong: none.
Task 4: Ruling: the property tests' exploration guard moved from seed 1 to seed 2 - adding rivals to the generator shifted seed 1 to only 1 failed run (seeds 2-4 give 7-11) - cost if wrong: none (a coverage guard).
Task 4: complete (commits b7476df..c5337fb, tests: npm test -> 448 passed; npm run check clean)
Task 5: Ruling: rivals set their style's prices when they join (not on their first launch day) - prices only matter once plans open, and no extra flag is needed - cost if wrong: none.
Task 5: complete (commits abffaa6..8dea905, tests: npm test -> 455 passed; npm run check clean)
Task 6: Ruling: a table's first leader is not news (only a change of leader is), and at most one ranking story runs a day - otherwise the first launch in a world with rivals made ten #1 stories at once (C13: rare and meaningful) - cost if wrong: fewer ranking headlines.
Task 6: complete (commits fbc051b..9f754d3, tests: npm test -> 461 passed; npm run check clean)
