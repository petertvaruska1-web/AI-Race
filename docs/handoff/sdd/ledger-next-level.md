# SDD ledger — plan: docs/superpowers/plans/2026-11-13-next-level-a-text-and-voice.md

Spec: docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md. Branch m2-two-friends (the project's work branch).

Ruling: no separate git worktree — the GPU work needs the generated `data/` and the uv environment of this checkout, and every earlier plan ran here — a mistaken commit is on the work branch, never on main.
Ruling: implementers and reviewers are dispatched one at a time; the lead developer runs measurements and decides keep/remove — owner's instruction 2026-11-13.
Models: sonnet for bounded implementation and reviews; opus for integration-heavy tasks (3, 13); the lead developer (session model) for measurements, data design decisions and Tasks 4, 15, 19.

## Pre-flight scan (2026-11-13)
| Tasks | Produces / consumes | Found | Ruling |
|---|---|---|---|
| 1 and 13 | `encode_chat` system line / training documents with a line | Task 13 hands documents with a first ("sys", text) turn; Task 1 only had `system=` | Task 1 accepts a first ("sys", text) turn too (plan edited) |
| 1 and 2, 16 | `chat_reply(system=)` | agree | - |
| 2 and 3 | probes / scorecard | `small_talk` took a Judge, untestable with a fake | takes `well_formed: Callable[[str], bool]` (plan edited) |
| 2, 3 and 8 | `lessons` probe needs the game's coaching recipe | Task 3 could not know it | the card gains `lessons` in Task 8 (plan edited) |
| 3 and 4, 5 | baseline before any training change | Task 5 changes the sampler | order kept: 4 before 5; nothing before Task 5 changes a run |
| 9 and every test fixture | corpus version of a new build | "v2 when none" would move every tiny fixture to text 2 | `--text {1,2}` on the build, default 1 until the release (plan edited); costs one flag at the release |
| 10 and the yardstick | typo noise | would change a text 1 rebuild | text 2 builds only (plan edited) |
| 9 and 14, 16 | `text_version`, recipes by text | agree after the edit | - |
| 13 and 14 | `Component.system_share`, generators | agree | - |
| 5, 6, 7 and the engine constant | each raises the engine if kept | agree with the global constraint | - |
| 16 and 17 | `identity` in job specs | agree; job names to be checked against `packages/protocol/src/worker.ts` in Task 17 | - |

Each task's own text: 1-19 read against their tests and files; no contradiction found beyond the rows above.

## Tasks
Task 1: implemented fe88dd4 (sonnet, 1814 tests pass). Review (sonnet): spec compliant, no Critical or Important, 4 Minor.
Ruling: an identity line longer than half the prompt budget is left out instead of raising a ValueError - a long name must never make a chat fail - costs: that AI answers without its name. Fix round 1 sent to the implementer (also: test rename, long lines).
Task 2: implemented 3fa11f3 (sonnet, 1834 tests pass), DONE_WITH_CONCERNS: added TorchLM.chat_generation and Generation.hit_limit (observations, not scope doubts). In review.
ADR 0010 started at 31abc15 (licence decision: only smol-magpie-ultra-short and smol-contraints rows of smol-smoltalk).
Task 1: fix round 1/5 commit c913c9d (line left out when longer than half the budget, or when it would leave under 4 tokens; test rename; long lines). Scoped re-review dispatched.
Task 2: review (sonnet): Needs fixes - Important: lessons key word and identity name matched as substrings; spec: accepted answers matched as substrings. Ruling: every 'holds' check is one whole-word helper - false passes would steer data decisions - costs nothing. Ruling: follow-ups must never loop for ever (reviewer: minor; raised because a measuring command must not hang).
Task 2: minor (deferred): follow-up test should match the relation as well as the subject; test_module_imports_without_torch_heavy_names does not test a light import; test_casual_facts_accepts_other_spellings asserts any(); a blank reply passes the no-line identity probe (as the brief says; watch for empty replies in the baseline).
Task 2: fix round 1/5 sent to the implementer.
Task 1: fix round 1/5 (3 addressed, 0 open; commits fe88dd4..c913c9d). Re-review: all addressed, no new breakage.
Task 1: complete (commits 6ce98b9..c913c9d, review clean)
Task 1: minor (deferred): the extra 'fewer than 4 tokens left' rule has no direct test; task-1-report.md's first section is stale about the ValueError.
Task 2: fix round 1 committed 43d9c59; scoped re-review dispatched.
Task 2: fix round 1/5 (4 addressed, 0 open; commits 3fa11f3..43d9c59). Re-review: all addressed, no new breakage.
Task 2: complete (commits 31abc15..43d9c59, review clean)
Task 2: minor (deferred): whole-word matching fails an inflected reply ("hums" for "hum"): intended strictness; the follow-up answer-overlap exclusion is still a substring check (only over-excludes).
Task 3: dispatched to opus (base 43d9c59).
Task 3: implemented b943c12 (opus, 1864 tests pass), DONE_WITH_CONCERNS. Ruling: yardstick loss = the trainer's held-out rule over 128 packed rows (the brief said '128 documents') - same rule as every other held-out number, less noise - costs nothing. Review dispatched (opus: the measuring stick is high risk).
Task 4: baseline started (runs/scorecard-engine2, background) on b943c12; a measurement bug found in review only costs re-measuring, the trained runs are reused.
Ruling: Tasks 9-13 (data-building code, no GPU, no training source touched) run before Tasks 5-8 while the baseline occupies the GPU - Tasks 5-8 need the finished baseline and the lead's measurements - costs nothing: nothing in 9-13 depends on 5-8. Tasks 9+10 are dispatched as one batch (two commits, one review), likewise 11+12.
Task 3: review (opus): Needs fixes - Important: compare drops a measure that became None and a model that stopped early; compare does not check that baseline and card used the same yardstick, judge and scale.
Ruling: a baseline with another yardstick identity, judge, novelty index, row length or scale is refused (exit 2); another engine is comparable; with another data identity the own-root held-out measures are skipped with a note - otherwise a text-2 card would be compared on different text - costs: the baseline is re-measured once.
Ruling: compare, TOLERANCE and the report move to experiments/scorecard_report.py outside the measurement digest - report wording must not re-measure seven models - costs one more module than the plan's file map.
Ruling: identity(with_line=False) passes only when the reply holds 'AI' as a word and claims no person's life - the first baseline showed the old rule passes 100% trivially and measures nothing - changes Task 2's brief.
Task 3: fix round 1/5 sent to the implementer (opus).
Task 4: first baseline run finished on b943c12 (53.0 s first AI; first AI well-formed 77-80%, follow-ups 10-12%, identity 0%); to be re-measured after the Task 3 fix. Seen: lineage-small well-formed 60% against small 90% (60 replies each) - to watch in Task 7.
Tasks 9+10: implementer (sonnet) resumed after the usage limit.
Tasks 9+10: implemented 8edb055, 6270d0f (sonnet, 1871 tests pass), DONE_WITH_CONCERNS (observations). Review dispatched (sonnet).
Task 3: fix round 1 committed 2ee1582 (opus); scoped re-review dispatched (sonnet). Baseline re-measure started (runs reused).
Ruling: the SODA filter is a second source id `soda_ai`; `soda` and its adapter stay untouched - a text-1 build must stay byte-for-byte - costs one more source spec. Plan Tasks 12 and 14 edited.
Tasks 11+12: dispatched (sonnet), base 2ee1582.
Task 3: fix round 1/5 (5 addressed, 0 open; commits b943c12..2ee1582). Re-review (sonnet): all addressed, no new Critical/Important.
Task 3: complete (commits 43d9c59..2ee1582, review clean)
Task 3: minor (deferred): an unlike baseline is only detected after measuring (except a baseline without measured_with); the instrument label comparison is stricter than needed; the novelty identity is the index's length, not a content hash.
Tasks 9+10: review (sonnet): approved, no Critical or Important.
Task 9: complete (commit 8edb055, review clean)
Task 10: complete (commit 6270d0f, review clean)
Tasks 9+10: minor (deferred): the refusal rule is written twice (cli.py and build.py); a text-2 chat document with no user turn has no test; docs/handoff/upgrade-2026-11-10/tok_probe.py hard-codes corpus/v1; for the release plan: check whether the server or tools read corpus/v1 directly.
Task 4: complete (baseline re-measured on 2ee1582: docs/reports/2026-11-scorecard-engine2.{json,md}; ADR 0010 section 3 with the targets fixed from it).
