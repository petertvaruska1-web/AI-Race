# SDD ledger — plan: docs/superpowers/plans/2026-10-27-game-layer-phase1.md
Spec: docs/superpowers/specs/2026-10-27-game-layer-design.md (read). Executor: inline (executing-plans). Branch: m2-two-friends. Merge base for the final review: 442bd43.
Pre-flight:
- T1 -> T2..T9: GROUP_IDS, PLAN_IDS, REGION_IDS and Balance.economy fields; every consumer names the same fields. Clean.
- T2 -> T3/T5/T13: ProductState (incl. feedbackKeys), PlansState, launchProduct{versionId, name, plans, countdownDays}. Clean.
- T3 -> T4: customers[company][region][group][plan] feeds ledgerDay. Clean.
- T5 -> T8: feedbackScored{productId, key, results[{group, questionId, question, reply, score}]} equals FeedbackJobResult.results plus key. Clean.
- T8 -> T9: FeedbackJobSpec {productId, versionId, modelTarUrl, groups, perGroup, seed}. Clean.
- T6 -> T10..T14: view shapes; VersionView.traits (T6) feeds orbParams (T11). evalCompleted gains traits in T6 (sim + training service). Clean.
Task 1: Ruling: adoptionShare lives in packages/sim/src/balance.ts (the plan lists it under Task 3's market module) — Task 1's test needs it and Task 3 imports it from there — cost if wrong: one move.
Task 1: Ruling: added BENCH_CATEGORIES (the eight bench categories) to packages/protocol/src/world.ts, unnamed in the plan — Task 1's test and later group weights need one canonical list — cost if wrong: none.
Task 1: complete (commits 442bd43..dcec57c, tests: npm test →    Duration  31.96s (tests 85%, import 9%, environment 4%, setup 1%, transform 1%))
Task 2: Ruling: ProductState gains launchPlans (the plans it powers from launch day), unnamed in the plan — the launch command lists plans but they apply only when the countdown ends, so they must be stored — cost if wrong: none.
Task 2: Ruling: CompanyState.fame {reputation, followers} is created in Task 2 (the plan defines it in Task 5) — hype during a countdown reads reputation and followers; Task 5 adds their dynamics — cost if wrong: none.
Task 2: Ruling: the zod countdownDays range is 0-365 and the sim enforces 3-14 with a plain reason — zod errors surface as "That request isn't valid", the sim's reason names the range — cost if wrong: none.
Task 2: complete (commits dcec57c..07ecd1c, tests: npm test →    Duration  31.85s (tests 85%, import 9%, environment 4%, setup 1%, transform 1%))
Task 3: Ruling: the free plan carries a utility penalty (economy.utility.freePenalty = 1.5, unnamed in the plan) — with no price, free would always beat paid and nobody would pay; the penalty stands for the free plan's limits — cost if wrong: one balance number (Task 7 tunes it).
Task 3: Ruling: business-plan customers are accounts (target = people x share / seatsPerAccount), so business revenue in Task 4 is accounts x seats x price — matches the plan's "accounts x seatsPerAccount" — cost if wrong: none.
Task 3: Ruling: spotlights are drawn from the world rng at createWorld (rng advances), so world seeds now also decide spotlights — deterministic and replay-safe — cost if wrong: none.
Task 3: complete (commits 07ecd1c..a4d433e, tests: npm test →    Duration  32.10s (tests 86%, import 8%, environment 4%, setup 1%, transform 1%))
Task 4: Ruling: INCOME_CATEGORIES and COST_CATEGORIES live in protocol/world.ts beside GROUP_IDS — Task 6's view schemas need them and protocol cannot import sim — cost if wrong: none
Task 4: Ruling: property (d) drops "cash never negative" — serving costs can take cash below zero (spec §8: rescue at a month end is Phase 2; a company cannot otherwise die); training is still refused without the cash — cost if wrong: one assertion
Task 4: Ruling: property (d) runs over the current arbitraries; Task 6 extends them with products, and the same identity then covers income and serving — cost if wrong: none
Task 4: Ruling: serving counts each business seat (cost follows use); a closed plan's leaving customers neither pay nor use compute — cost if wrong: balance tuning
Task 4: Ruling: the ledger is made in createCompany (newLedger(s.day)), not in newCompanyBusiness — avoids a products<->ledger import cycle — cost if wrong: none
Task 4: Ruling: closeMonths runs first in the tick, so customersEnd is the count at the end of the month's last day and the new day books into the new month; a company's open month always equals the month of the current day — cost if wrong: none
Task 4: Ruling: rounding is per booking (each group-and-plan revenue, one serving amount per company per day); amounts under 50 cents a day book nothing — negligible at game scale — cost if wrong: a small early-game understatement
Task 4: Ruling: the conservation identity over ledger months holds while at most 24 months have closed (older ones drop off); tests stay within that — cost if wrong: add carried totals to the ledger
Task 4: Ruling: customersEnd counts a business account once (the market's unit), rounded at close — cost if wrong: one multiplication
Task 4: complete (commits a4d433e..54cdbd7, tests: npm test →    Duration  31.57s (tests 84%, import 9%, environment 4%, setup 1%, transform 1%))
Task 5: Ruling: posts carry productName (unnamed in the plan) — a lineup has up to 3 products and the post's frame names the one that answered; names never change after launch — cost if wrong: none
Task 5: Ruling: NEWS_TOPICS and POST_TONES live in protocol/world.ts — Task 6's view schema needs them — cost if wrong: none
Task 5: Ruling: FameState gains newsSentiment (the news mood, a moving average) with fame.newsAlpha 0.3, and good-news sentiments (launch 0.5, customers 0.6, first paying 0.8) join the balance — the plan names an EMA of news sentiment without its rate — cost if wrong: balance tuning
Task 5: Ruling: CompanyState.announced {customers, firstPaying} remembers milestones already told, since the feed drops old items — cost if wrong: none
Task 5: Ruling: WorldState.feedSeq numbers feed ids (f1, f2, ...) — sim-made items need world-unique ids — cost if wrong: none
Task 5: Ruling: satisfaction moves once per group per batch, toward that group's mean score — a launch burst with several questions per group moves it no more than a weekly batch — cost if wrong: one line
Task 5: Ruling: at most one viral story per batch, from the more extreme post (a roast wins a tie) — two extreme posts would otherwise make two stories at once — cost if wrong: one line
Task 5: Ruling: a jump past several customer marks in one day tells only the highest — news stays rare — cost if wrong: none
Task 5: Ruling: product ids are unique across a world (launchProduct checks every company; test added) — feedback finds a product by id alone and satisfaction is keyed by product id — cost if wrong: none
Task 5: Ruling: feedback for a product that is not live is refused ("This product is not on sale.") — cost if wrong: a late batch for a just-retired product is dropped
Task 5: Ruling: launch headline is "<Company> launches <Product>, scoring <n> in <best skill>" — no ranks in Phase 1 — cost if wrong: copy
Task 5: note for the final review: findRun/findVersion look ids up without Object.hasOwn (an id like "constructor" in a system event would resolve to a prototype member); ids come from the server, so the risk is low; findProduct uses Object.hasOwn
Task 5: complete (commits 54cdbd7..a0a5e5e, tests: npm test →    Duration  30.83s (tests 86%, import 9%, environment 4%, setup 1%, transform 1%))
Task 6: Ruling: PublicProduct.version is null during the countdown (the brief lists a version for countdown products too) — spec R2: never another company's unreleased versions — cost if wrong: rivals see a product's scores only from launch day
Task 6: Ruling: a version behind a live product is public through that product; the privacy property treats live-product versions like shared ones — cost if wrong: none
Task 6: Ruling: public prices show a plan only while a live product powers it (free as a yes/no, paid as the price or null) — cost if wrong: none
Task 6: Ruling: public customers are whole numbers below 1,000 and three significant figures above — cost if wrong: none
Task 6: Ruling: the owner's satisfaction view covers every own product and all six groups, with the starting value where no feedback has arrived yet — cost if wrong: none
Task 6: Ruling: evalCompleted.traits is optional (missing means null), so dev worlds logged before this change replay unchanged; the server passes stored fingerprints as traits only when they parse as numbers — cost if wrong: none
Task 6: Ruling: added property (e): customers are never negative and never more than the people willing (spec §14 lists it) — cost if wrong: none
Task 6: Ruling: a third of the generated worlds start with a scripted launch (train, score, launch, feedback), so launches and posts are common enough for the properties to mean something — cost if wrong: none
Task 6: Ruling: the privacy property counts private keys (cash, ledger, satisfaction, plans, fame once in the viewer's own company; hype once per own product; awareness never) instead of searching for values, which would false-alarm on 0 or 0.5 — the strict schemas catch private values under public names — cost if wrong: none
Task 6: Ruling: the property test file has a 30 s limit — under the parallel full suite the 200-world properties took over 5 s (1-2 s alone) — cost if wrong: none
Task 6: complete (commits a0a5e5e..dff7058, tests: npm test →    Duration  32.68s (tests 86%, import 8%, environment 3%, setup 1%, transform 1%))
Task 7: Ruling: runBot's days count from launch day; the bot launches on all three plans at default prices and sends weekly feedback scored from the model's group-weighted skill — the plan says "launch with default plans" without naming them — cost if wrong: none
Task 7: Ruling: the price target compares each group's paying customers (personal plus business) as a share drop — with a free plan, a price rise moves students to free rather than away — cost if wrong: none
Task 7: Ruling: added utility.businessFit (1, for the groups that buy business, on the business plan) to the appeal formula, with a market test — without it the business plan cost three times as much per seat for nothing and businesses never chose it, against spec §6 — cost if wrong: one balance number
Task 7: Ruling: balance-v2 retuned: noAi 6 (was 1), score 2 (4), freePenalty 0.5 (1.5), wordOfMouth 0.5 (0.02), launchBoost 0.1 (0.01), serving 0.01 per million parameters a day (0.0004) — the old numbers gave a typical first AI 44% of the market and $533M in its first year, and made serving free — cost if wrong: balance tuning
Task 7: Ruling: a typical company still ends its first year with about $8M; Phase 2's data centers, campaigns and loans are the money sinks, and its harness revisits this — cost if wrong: money feels plentiful late in year one until Phase 2
Task 7: complete (commits dff7058..81c69d6, tests: npm test →    Duration  33.37s (tests 86%, import 8%, environment 3%, setup 1%, transform 1%))
Task 8: Ruling: a product's launch batch is asked whenever the service first sees it live (not only on its launch day), deduped by feedback_requests — a server that was down on launch day still asks once — cost if wrong: none
Task 8: Ruling: WorkerCapabilities.slots.feedback defaults to 0, so a worker from before feedback jobs (today's Python worker until Task 9) gets none — cost if wrong: none
Task 8: Ruling: outdated worlds are recognised by a typed OutdatedWorldError from WorldActor.load; their list entry takes its name from the worlds table — cost if wrong: none
Task 8: Ruling: a batch's seed is a hash of product id and key, so a retried job asks the same questions — cost if wrong: none
Task 8: Ruling: the request row is written before its job is queued; a crash between the two skips that batch (as a final failure does) — cost if wrong: one batch of posts lost on a crash
Task 8: Ruling: the brief's six tests are four: lease, result and posts in one flow; dedupe and launch-once in one restart test; every bullet is asserted — cost if wrong: none
Task 8: complete (commits 81c69d6..12e27f3, tests: npm test →    Duration  36.13s (tests 87%, import 8%, environment 3%, setup 1%, transform 1%))
Task 9: Ruling: the reasoning generator makes only multiple-choice items, which a chat cannot ask, so students' questions come from knowledge and researchers' from patterns (GROUP_CATEGORIES still lists reasoning, per the plan) — cost if wrong: feedback does not test reasoning until a chat reasoning item exists
Task 9: Ruling: customer items are renamed "<category>-customer-NNNN" — the generators number items as the benchmark does, so ids would collide — cost if wrong: none
Task 9: Ruling: questions are asked exactly as the benchmark frames them (for example "Question: ...\nAnswer:") and posts show what was asked — cost if wrong: posts read a little formal; the client can tidy the frame for display
Task 9: Ruling: a reply's score is its check's pass or fail (1 or 0), so satisfaction moves in coarse steps — cost if wrong: a partial-credit score later
Task 9: Ruling: the shared hello sample gains a feedback slot, as workers now announce one — cost if wrong: none
Task 9: Ruling: the "reply equals chat_reply" test lives in test_worker_jobs.py beside the trained-model fixture; the pool tests are in test_feedback.py — cost if wrong: none
Task 9: note: the full Python suite takes 2 min 45 s, close to the 3-minute limit
Task 9: complete (commits 12e27f3..789326c, tests: bash -c 'cd ml && python -m uv run --no-sync pytest tests/test_feedback.py tests/test_worker_jobs.py tests/test_worker_protocol.py -q' → 33 passed, 4 skipped in 7.59s)
Task 10: Ruling: the restyle moves every screen from Card to Panel and deletes Card — one glass surface everywhere — cost if wrong: none
Task 10: Ruling: Market and Money show "This screen opens soon." until Tasks 13-14 build them in this phase — cost if wrong: none
Task 10: Ruling: CountUp shows its first value at once and rolls only between values — the top bar's cash would otherwise roll up from $0 on every screen change — cost if wrong: less flourish on first view
Task 10: Ruling: CountUp jumps wherever reduced motion can't be checked (no matchMedia) — cost if wrong: none
Task 10: complete (commits 789326c..8e3cff7, tests: npm test →    Duration  36.72s (tests 86%, import 8%, environment 3%, transform 1%, setup 1%))
Task 11: Ruling: rings are a smooth count — each skill's ring fades in between scores 15 and 25 (half at 20) and reading adds log10 of millions of words, unrounded — spec §4 says every orb parameter is continuous, against the brief's count-and-floor; at scores away from 15-25 and whole powers of ten it equals the brief's formula — cost if wrong: one function
Task 11: Ruling: intake carries shares beside colors ({rate, colors, shares}) so streams follow the data mix — cost if wrong: none
Task 11: Ruling: an AI with no version yet (0 words read) is the faint spark; an unscored version uses its words read for rings and neutral motion — cost if wrong: none
Task 11: Ruling: during training the orb shows the AI the run grows from (its parent, or nothing for a first AI) and brightens as the revealed loss falls — cost if wrong: none
Task 11: complete (commits 8e3cff7..59013cb, tests: npm test →    Duration  36.04s (tests 86%, import 8%, environment 4%, setup 1%, transform 1%))
Task 12: Ruling: the customers figure shows its change "this month" (against the last month-end count in the ledger) instead of "this week" — the view has no daily history, and this survives a reload — cost if wrong: a weekly change needs a short customer history in the view
Task 12: Ruling: the Company screen's Launch button opens Task 13's dialog; until Task 13 it only records the intent — cost if wrong: none at the end of the phase
Task 12: Ruling: the feed shows newest first and scrolls inside its panel; an empty reply shows "(nothing at all)" as the game's note — cost if wrong: none
Task 12: Ruling: cash and profit are stacked in the Money panel — the browser check showed them colliding side by side in narrow columns — cost if wrong: none
Task 12: complete (commits 59013cb..2d01350, tests: npm test →    Duration  36.48s (tests 85%, import 8%, environment 4%, setup 1%, transform 1%))
Task 13: Ruling: the launch dialog opens from the Company screen, the Money screen ("Launch an AI") and each scored version in the Lab ("Launch this AI"); all hold with the same reason until there is a scored AI — cost if wrong: none
Task 13: Ruling: a price is set with an explicit Set button, whole dollars within the balance's bounds (shown beside it); nothing is sent while typing — cost if wrong: none
Task 13: Ruling: a refused plan, price or retire command shows the server's reason in one alert at the top of the Money screen — cost if wrong: none
Task 13: Ruling: profit by month is drawn as simple bars above and below a zero line (no chart library) — cost if wrong: none
Task 13: complete (commits 2d01350..8b29a34, tests: npm test →              at least ~3.68s faster with isolate: false — reuses workers across files instead of one per file)
Task 14: Ruling: a group's satisfaction is its customers' average over the plans it uses (the plain average of the products' values while it has no customers) — the view has satisfaction per product, customers per plan — cost if wrong: one function
Task 14: Ruling: "what each group cares about" lists its top three skills from the economy's weights, plus price and speed words from its price and speed numbers — cost if wrong: copy
Task 14: note: under the usage limit, the Market tests were written first but run only after the screen existed (no separate red run) — the final review should weigh them
Stopped after Task 14 (usage limit). Next: Task 15 (E2E, dev run with a real GPU-trained AI, docs), then the final whole-branch review and the report to the owner.
Task 14: complete (commits 8b29a34..3d0a1f9, tests: npm test →    Duration  38.62s (tests 83%, import 9%, environment 5%, setup 2%, transform 1%))
Task 15: Ruling: the playthrough passed on its first run (1.1 min) and again (51 s); the launch, customers, news, posts and money features already existed, so it found no bug to fix — cost if wrong: none
Task 15: Ruling: the end-to-end test reads two hooks (data-testid "customers" on the Company screen's customer total, "income-personal" on the Money screen) — cost if wrong: none
Task 15: dev run on the RTX 3050 (balance-v2, 4x): first AI trained in about 80 s and scored in 15 s (overall 43, "inventive, timid"); launched as Wave; 5,555 customers two days after launch; launch, first-paying and 1,000-customer news; posts quote the AI's real replies ("Phiated.", "BYetch"); feedback batches take 1-2 s each, once a game week; a chat during the run started and finished within a second
Task 15: Ruling: the docs (progress entry, handoff resume point, ledger sync) are committed after the final review, so they record its outcome; the end-to-end test is committed now — cost if wrong: none
Final review: self-review (no reviewer agent: the owner asked for no helper agents this milestone) over 442bd43..ff095e4, in passes by area (sim, server, ML, client), plus the GPU dev run and the browser checks
Final: fixed ids named like built-in object parts (__proto__, constructor, ...) passing the id check, so retireProduct "__proto__" set a field on Object.prototype and launchProduct/startTraining could use Object.prototype as an AI (Critical; it also covered chat versionId and telemetry runId on the server) — zId refuses them; "refuses names of built-in object parts..." and "refuses ids named like built-in object parts, changing nothing anywhere" RED->GREEN, suite 319/319, e2e passed
Final: fixed the Money screen showing a rise in costs as a green arrow (Important: green reads as good news) — Stat downIsGood; "colors a change by whether it is good news" and "colors this month's costs against last month's..." RED->GREEN, suite 319/319
Final: minor (deferred): instruction checks score the format only, so a wrong answer in the right format is "Loved it" (the dev run's orangutan answered "Africa" in one word)
Final: minor (deferred): posts show the benchmark's frame around a question ("Question: ...\nAnswer:"); the display could tidy it
Final: minor (deferred): the Company screen holds Launch while the newest AI is unscored even if an older one is scored (the Lab's "Launch this AI" works)
Final: minor (deferred): a world loaded only to list it on Home also asks for its week's feedback batch (some GPU time when the player merely opens Home)
Final: minor (deferred): a feedback job holds its AI for the whole batch (1-2 s on the GPU), so a chat arriving then waits that long
Final: minor (deferred): the Money screen's price box keeps what was typed instead of following a price changed elsewhere
Final: minor (deferred): in multiplayer a post doesn't say which company's product it is about (Phase 1 is single player)
Task 15: complete (commits 3d0a1f9..690268b, tests: npm test →              at least ~1.89s faster with isolate: false — reuses workers across files instead of one per file)
