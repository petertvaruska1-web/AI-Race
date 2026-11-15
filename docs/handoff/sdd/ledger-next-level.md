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
Finding (lead, reading the baseline's replies with docs/handoff/next-level/talk_show.py): the gate's well-formed rule fails good short replies ("You're welcome.", "I'm good.", "Yes.") because of its minimum length and because it never sees the question; the grown Small brain's 60% is partly that and partly real nonsense ("It's Oceania." to "Tell me about cats.").
Ruling: the card gains `fits` - the judge's loss of the reply GIVEN the question, under the 90th percentile of real held-out replies - and stores the small-talk transcript; well_formed stays for continuity with G2 - a target on well-formed alone would reward longer replies, not better ones - costs a third measuring of the baseline. Sent to the Task 3 implementer (opus) as an extension (Task 3b).
Caveat recorded: the yardstick judge was trained on text 1, so it may under-rate the assistant style of text 2; Task 15 also reads transcripts and judges with the new root's own judge.
Tasks 11+12: implemented 1917510, dd76f41 (sonnet, 1891 tests pass); smol_smoltalk pinned at f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc. Review (sonnet): approved, no Critical or Important.
Task 11: complete (commit 1917510, review clean)
Task 12: complete (commit dd76f41, review clean)
Tasks 11+12: minor (deferred): _smol keeps conversations whose roles do not alternate or that end on a user turn (to be tightened with the text-2 recipes, Task 14); _smol/_everyday share copied turn extraction; m.get on a non-dict message would raise.
Task 3b: implemented 24b78df (opus, focused tests 401 pass). Review dispatched (opus). Baseline re-measure 3 started.
Task 13: dispatched (opus), base 24b78df.
Task 3b: review (opus): approved, no Critical or Important.
Task 3b: complete (commit 24b78df, review clean)
Task 3b: minor (deferred): infinite losses are dropped before the percentile; the threshold test restates the implementation; the threshold cache key has no device (exact float compare in unlike); fits is stored as None without a rule.
Task 4: baseline re-measured with `fits` (threshold 2.524 over 1000 pairs): first AI 77-78%, small 90%, lineage-small 77%, lineage-medium 90%, chatty 90%, large 97%. Reports and ADR 0010 updated.
Finding (lead, reading the stored transcripts): `fits` passes fluent off-topic replies and fails a bare "Yes." - it measures fluency given the question more than relevance. Ruling: add `on_topic` (the reply's loss after its own question against its mean loss after K other held-out questions; threshold and error rate calibrated on real vs shuffled held-out pairs) as Task 3c after Task 13 lands - open small talk has no exact answer and this is the best automatic stand-in; decisions also read transcripts - costs one more baseline re-measure.
STOPPED 2026-11-13 (usage limit). State: Tasks 1, 2, 3 (+3b), 4, 9, 10, 11, 12 complete and reviewed. Task 13 (follow-up and identity chats) was with an opus implementer (agent af256b61b3530e0b5, base 24b78df) and had not committed; on resume check `git status` and `git log` for its work, resume or re-dispatch it from task-13-brief.md, then review it. Next after that: Task 3c (on-topic measure, to the scorecard implementer a15874185e5b4fb21 or a fresh opus agent), re-measure the baseline, Task 14 (text-2 recipes; also make _smol keep only alternating conversations ending on an AI turn) and start the full text-2 build into data2/, then Tasks 5-8 on the GPU while it builds.
RESUMED 2026-11-14. Task 13: implemented e2e732e (opus, 1920 tests pass), DONE_WITH_CONCERNS (stiff KB answer templates; identity lines added after the token quota, under 1% over; names filtered by curated lists). Lead read the samples: natural follow-ups, honest identity replies; accepted for review. Review dispatched (opus: a leak in the probe split would inflate the follow-up measure).
Task 3c: on-topic measure sent to the scorecard implementer (opus) with scale changes (small-talk samples 5, follow-ups 120, identity 3 seeds a question; small-talk tolerances 0.06).
Task 13: review (opus): Needs fixes - Critical: rude made-up names ("Puta", "Putor") in NAMES; Important: the pronoun and repeat moves and other facts' answers bring probe-held subject strings into later AI turns (0 measured pairs leak, 160 strings in 20,000 chats); minors.
Ruling: the split rule stands literally (no held subject string in any later turn, user or AI) - the follow-up measure steers the programme - costs a little variety in follow-up chats. Ruling: `_followup` gets a try limit (a build must not hang).
Task 13: fix round 1/5 sent to the implementer (opus).
Task 13: minor (deferred): chats.py builds its name lists at import and holds three things; a few replies are loose ("What's your favorite food?" -> "Nothing. I'm an AI, so I don't eat."); identity lines are added after the token quota (under 1% over); generic company names may match real firms.
Task 3c: implemented 4fafe1d (opus). Baseline re-measure 4 started. Review to dispatch.
RESUMED 2026-11-14 (second time, after the 5-hour limit). Task 13: fix round 1 committed 5b45c7d (opus): rude-stem list over twelve languages applied to names and companies, pronoun and repeat moves draw from askable subjects only and every rendered follow-up is checked against all held subjects, `_followup` try limit, Leo/Cleo, whole-name filter. Scoped re-review dispatched (opus, reads every name).
Task 3c: review (opus): approved, no Critical or Important; identity seeds k*samples.. accepted. Minors: tolerance comments overstate precision; NaN under another question gives +inf; calibration test restates the implementation.
Baseline re-measure 4 (on 4fafe1d): on-topic calibration balanced accuracy 87% (real above 95%, swapped at or below 80%); first AI on topic 83-86%, fits and on topic 61-65%; large 87%.
Finding (lead, reading the stored replies and running docs/handoff/next-level/topic_probe.py): the mean-per-token lift fails every story (a long reply dilutes it) and rejects only 57% of hard negatives (the neighbouring pair's reply). A total over the reply's first 24 tokens with 32 other questions: balanced accuracy 0.939, hard negatives rejected 0.81. Neither form catches a fact sentence after "What is your name?" or "Hello! How can I help you today?" after "Goodbye.": the judge itself (trained on text 1) finds those likely; the identity probes cover the first.
Ruling: the lift becomes the total over the first 24 tokens with 32 other questions; the calibration also records the hard-negative share and the share of real replies passing both rules (the ceiling for "fits and on topic") - the stick must be as sharp as it can cheaply be before decisions rest on it - costs a fifth baseline re-measure. Sent as Task 3d to the scorecard implementer (opus). After it the stick is frozen.
Ruling: the small-talk decision of Task 15 also rests on a blind reading: the two first AIs' stored replies, shuffled and unlabelled, are graded sensible or not by a separate reviewer agent against a written rubric - the judge model cannot see what a reader sees - costs a few thousand tokens per comparison.
RESUMED 2026-11-15 (third time, after the 5-hour limit): Task 3d implementer (opus), Task 13 re-reviewer (opus) and Task 14 implementer (sonnet, dispatched 2026-11-14 with the _smol alternation rule and measured bytes-per-token entries; `--datasets` already rebuilds one dataset, so no `--only`) all resumed.
Blind reading built (docs/handoff/next-level/blind_read.py) and checked on the baseline replies (A first-s1, C first-s2, B large; 258 items, reader: sonnet, did not see the key): sensible 34% / 27% / 55%, nonsense 43% / 43% / 20%; large minus first +21 and +28 points (90% intervals exclude zero), the two first AIs 7 points apart (interval includes zero). The reader marks a person's life in 9-10% of first-AI replies and 23% of the large one's, mostly the person being called by a made-up name; the word list saw 0-2%. Stored in docs/reports/2026-11-blind-reading-engine2/.
Finding: a reader finds about 30% of a first AI's small-talk replies sensible; the judge-based "fits and on topic" said 61-65%. The judge rules are lenient guards; the blind reading is the measure of talk.
Ruling: Task 15's keep rule for small talk is the blind reading (text 2's first AI against text 1's, same prompts and seeds): kept when sensible is higher with a 90% interval that excludes zero - the instrument tells a large brain from a first AI and puts two seeds of one recipe together - costs one reader agent per comparison.
Ruling: soda_ai also drops a dialogue when an AI turn holds either speaker's name (to be added by the Task 14 implementer after its commit) - the made-up-name habit comes from SODA's named speakers and a reader sees it in 9-23% of replies - costs SODA variety, of which there is plenty (1.19M dialogues for about 140k needed).
Task 13: scoped re-review of 5b45c7d (opus, read all 2,100 names and 699 companies): Needs fixes - Important: names that still read as rude in Japanese, Greek, Afrikaans and English (Kuson, Makuso, Gamo, Fokio, Jiziso); Important (by the lead's rule): 12.5% of identity chats name a held subject in a later turn (the trailing fact question, and companies named Lemon or Hedgehog); minors: doubtful names, real brands (Tripo, Brita, Teva, Pinecone), the rude-stem and held-subject tests copy the code they check, the favourite-food reply. The follow-up chats' split, the try limit and text 1 are correct; the 60 chats read are natural.
Ruling: every finding is fixed (brief: task-13-fix2-brief.md): the split rule has no exception; the fix goes to a fresh sonnet agent after Task 14 hands back (mechanical edits; same package as Task 14) and the lead reads the names that are new after it - a block list cannot be proven complete, and these names live in the training text only (players name their own AI), so a third full read is not bought - costs a small residual risk recorded here.
Task 3d: implemented a717d2e (opus, focused tests 415 pass): `Judge.reply_totals`, the lift as a total over the first 24 tokens against 32 other questions, the hard-negative share and the share of real replies passing both rules in the calibration and under the report's table, the reviewer's minors.
Ruling: no separate review of a717d2e - on the real data its calibration (balanced accuracy 94%, real above 94%, swapped at or below 94%, hard swapped 81%) equals, share for share, what the lead's own script (topic_probe.py, written before the change) measured for this form; two independent implementations agreeing on 1,000 real pairs checks more than a reading of the diff - costs nothing. Real replies that fit and are on topic: 86% (the ceiling of that rate). The stick is frozen here.
Baseline re-measure 5 started on a717d2e.
Task 14: implemented 7da7d88 (sonnet, 1950 tests pass): RECIPES_V2 and the text-2 targets beside unchanged text 1, sources_needed by text, the _smol alternation rule, measured bytes-per-token entries, soda_ai drops dialogues whose AI side uses a speaker name (57% of rows survive). DONE_WITH_CONCERNS: smol_smoltalk reads noisy. Not reviewed yet: the recipe and the smol source change again in Tasks 12b and 13b; one review covers the three commits.
Finding (lead, docs/handoff/next-level/smol_survey.py over all 460,341 rows): smol-magpie-ultra-short keeps 2,685 of 270,838 rows (1.4M tokens; preambles, role-play, adult-level explanation); smol-contraints keeps 21,088, 99% rule text. 15M tokens were planned. A kinder rule gives at most 1.4M tokens of the same kind. TinyChat (1M simple chats): no licence on its card, second speaker talks as a person.
Ruling: smol_smoltalk is removed from the code and the recipe (Task 12b) - 1% usable and what is usable teaches role-play and adult text - costs Task 11's work.
Ruling: the game writes its own assistant conversations: generator `assistant_chat` (Task 13b, opus; brief task-13b-brief.md) - no simple, permissively licensed assistant conversations exist at scale - costs templated variety, measured by the blind reading on 20 held-out lines (airace_content/held.py, fixed before the generator was written, commit 6ccdcf3).
Ruling: soda_ai keeps only dialogues whose AI side listens (no first-person word outside a list of listener phrases; Task 12b, sonnet; brief task-12b-brief.md) - SODA's second speaker is a person with a life, and the word list of life claims catches 3% - costs most of SODA, of which there is plenty.
Ruling: text-2 conversations recipe: soda_ai 0.30, assistant_chat 0.20, fact_chat 0.14, instructions 0.13, followup_chat 0.12, identity_chat 0.10, everyday_conv 0.01 - first guess, tuned in Task 15's rounds by rebuilding conversations alone (`--datasets conversations`).
Baseline re-measure 5 (final, on a717d2e): first AI fits and on topic 53-55% (real replies 86%), on topic 65-67%; large 83%. Reports copied; ADR 0010 sections 2 and 3 rewritten (talk measures, blind reading baseline, targets; the "90% fits and on topic" target replaced by the blind reading, with the reason).
Ruling: the engine becomes 3 once, for every Phase B item that is kept (not one number per item) - nothing is released between them - costs nothing.
Dispatched 2026-11-15: Task 13 fix round 2 (sonnet), Task 12b (sonnet), Task 13b (opus), in parallel on disjoint files.
