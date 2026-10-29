# SDD ledger — plan: docs/superpowers/plans/2026-10-28-game-layer-phase2.md

Spec: docs/superpowers/specs/2026-10-27-game-layer-design.md (§5, §8, §13-§15) and docs/superpowers/specs/2026-10-28-immersion-design.md.
Ruling: executed inline without an owner review stop - the owner's explicit go ("after you are done, just jump straight into phase 2") and CLAUDE.md (decide and proceed) - cost if wrong: rework if the owner disagrees with a plan choice.
Ruling: the final review is a self-review - the owner asked for no helper agents this milestone - cost if wrong: a weaker second look.
Pre-flight:
- T1 -> T2: REGION_IDS for revenueByRegion: consistent.
- T2 -> T7: finance(company, "loans"|"investors"|"repaid", amount): consistent.
- T4 -> T5: ownedPflops(company, day) and compute.trainingShare; tellNews exported in T4: consistent.
- T4 -> T7: NEWS_TOPICS gains rescue in T4: consistent.
- T5 -> T11: previewPlan(b, parent, plan, ownedPflopsPerDay = 0): consistent.
- T6 -> T7: month-close order pinned in T7 (ledger close, report card, loan payment, offer): consistent.
- T1/T3/T9 -> T10: map marks and campaign commands: consistent.
Task 1: complete (commits 1b7f282..743eb15, tests: npm test →    Duration  54.09s (tests 81%, import 8%, setup 5%, environment 5%, transform 1%))
Task 2: complete (commits 743eb15..fd36af0, tests: npm test →    Duration  54.33s (tests 81%, import 8%, setup 5%, environment 5%, transform 1%))
Task 3: complete (commits fd36af0..f3dd2ef, tests: npm test →    Duration  53.59s (tests 81%, import 8%, setup 5%, environment 5%, transform 1%))
Task 4: Ruling: the overload headline rounds the answered share to the nearest percent (plan said n%) - floor showed 39% for 0.4 by float error - cost if wrong: a headline one percent off
Task 4: complete (commits f3dd2ef..8c7c03c, tests: npm test →    Duration  64.41s (tests 79%, import 10%, setup 6%, environment 5%, transform 1%))
Task 5: Ruling: each data center stores its capacity (pflopsPerDay) when built, beyond the plan's fields - ownedPflops(company, day) then needs no balance - cost if wrong: none (one more number per center).
Task 5: Ruling: REGION_PLACES (protocol) names a region after 'in' (the Pacific, New England) for headlines - cost if wrong: wording only.
Task 5: Ruling: a NaN share is refused by the command schema before the rules see it; the rules refuse -0.1 and 1.5 in plain words - cost if wrong: none.
Task 5: complete (commits 8c7c03c..c914563, tests: npm test →    Duration  59.98s (tests 81%, import 8%, setup 5%, environment 4%, transform 1%))
Task 6: Ruling: the card's satisfaction field is named mood - a second 'satisfaction' key in the owner's view would blunt the privacy property's key count - cost if wrong: a field name.
Task 6: Ruling: gradeFor(...) is exported and tested directly for each letter, beside the integration tests - cost if wrong: none.
Task 6: complete (commits c914563..f9f7f99, tests: npm test →    Duration  57.42s (tests 81%, import 8%, setup 5%, environment 5%, transform 1%))
Task 7: Ruling: a rescue restores max(0, cushion - cash); when cash is already above the cushion the offer just closes - cost if wrong: none (the offer only opens below zero).
Task 7: complete (commits f9f7f99..f895d69, tests: npm test →    Duration  56.31s (tests 81%, import 8%, setup 5%, environment 4%, transform 1%))
Task 8: Ruling: added a seventh target (ads take 60+ days to pay back) - at the plan's starting price one blitz made 33k in 110 days, a dominant strategy - cost if wrong: a slower-growing ad game.
Task 8: Ruling: the spendthrift target uses 14-day campaigns (plan: five $10,000-a-day campaigns, length unstated) - with 7 days the ads earned the cash back before the deadline and the offer rightly lapsed - cost if wrong: none.
Task 8: Ruling: the spike target reads day 61 (the spike lands after day 60's figures) - cost if wrong: none.
Task 8: complete (commits f895d69..9bcdd56, tests: npm test →    Duration  58.27s (tests 81%, import 8%, setup 5%, environment 4%, transform 1%))
Task 9: complete (commits 9bcdd56..5ef1dfb, tests: npm test →    Duration  57.19s (tests 81%, import 8%, setup 5%, environment 4%, transform 1%))
Task 10: Ruling: fixed a timing race in an older server test (aborted upload cleanup checked after a fixed 200 ms) that failed once under the full parallel run - it now waits up to 4 s for the partial file to go - cost if wrong: none (same guarantee).
Task 10: complete (commits 5ef1dfb..10005fa, tests: npm test →              at least ~3.41s faster with isolate: false — reuses workers across files instead of one per file)
Task 11: Ruling: financing shows in its own 'Loans and investors' panel (only when there is any) rather than under Going out - loans are money in - cost if wrong: placement only.
Task 11: complete (commits 10005fa..05958eb, tests: npm test →              at least ~3.53s faster with isolate: false — reuses workers across files instead of one per file)
Task 12: complete (commits 05958eb..664af6f, tests: npm test →              at least ~3.75s faster with isolate: false — reuses workers across files instead of one per file)
Task 13: complete (commits 664af6f..72f9e08, tests: npm test →              at least ~3.86s faster with isolate: false — reuses workers across files instead of one per file)
Task 14: complete (commits 72f9e08..9c73b79, tests: npm test →    Duration  55.95s (tests 80%, import 8%, setup 6%, environment 5%, transform 1%))
Task 14: Ruling: e2e found that the live connection gave no ids to startCampaign and buildDataCenter (every one refused 'Missing ... id'); fixed in the gateway with a server test RED->GREEN (a2ede6b) - cost if wrong: none.
Task 14: Ruling: after playing it, the region map is capped in width and its tint spreads from the weakest region to the strongest (shares are close, so best-only scaling looked flat) (9c73b79) - cost if wrong: looks only.
Final review: self-review (no subagent tool by the owner's wish). Critical/Important: the gateway id bug above (fixed). e2e passed twice; npm test 421/421; check clean.
Final: minor (deferred): report cards have no rank changes against rivals yet (Phase 3, when rivals exist).
Final: minor (deferred): ads pay back in about 3 months and then keep paying (a $350k blitz is +$690k by day 230); revisit when rivals compete for awareness.
Final: minor (deferred): fast early growth can briefly overload the cloud, so the overload warning may flash for a typical launch; designed, but watch how it feels.
Final: minor (deferred): a sim-2 world (the owner's Phase 1 game) no longer opens; it lists as made with an older version (C29).
