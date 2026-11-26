# SDD ledger — plan: docs/superpowers/plans/2026-11-26-look-redesign-1-bench-and-company.md
Spec: docs/superpowers/specs/2026-11-26-look-redesign-design.md (owner-approved 2026-11-26). Branch m2-two-friends (the project's working branch, per the task); merge base with m1-it-learns 1ab0a27; plan start eaec660.
Execution: inline (executing-plans), chosen by Claude (owner non-technical; CLAUDE.md: decide technical matters).
Pre-flight:
- T1→T2: useLook, glassFor produced in theme/glass.ts, consumed by GlassBackground — match.
- T2→T5/T7: framePolicy, due, createFrameMeter, useGraphics — match.
- T1→T6: readableOn(color, glass) for chat names — match.
- T3/T4→T5/T6/T9: Module, Display, Readout, Lamp, PushButton, ToggleSwitch, KeyRow, RotarySelector — match.
- T7→T8: Stage3D {scene, camera, update(t,dt):boolean, pointer?, dispose}, View3D props, gearRing, materials — match.
- T8→T9: Core(props same as Orb) — match.
Task 1: complete (commits eaec660..9c8c88f, tests: npm test →    Duration  163.82s (tests 83%, import 8%, setup 4%, environment 4%, transform 1%))
Task 1: note: npm run check passes; its 2 warnings are pre-existing (apps/server/test/discoveries.test.ts, packages/sim/src/balance.ts unused imports), not from this plan.
Task 2: Ruling: three/support.ts (webglAvailable) is created here, not in Task 7, with a version parameter (1 for the glass, 2 for three.js) — the glass needs it first; three.js r187 needs WebGL 2 — cost if wrong: none, Task 7 consumes the same function.
Task 2: complete (commits 9c8c88f..6dd04b9, tests: npm test →    Duration  163.58s (tests 83%, import 8%, setup 4%, environment 3%, transform 1%))
Task 3: Ruling: bench.css sits in Tailwind's components layer — so a screen's utility classes can still adjust a part (unlayered CSS would beat every utility) — cost if wrong: a part's own padding loses to a screen's utility, fixable per part. Note: Python edits on Windows wrote CRLF; normalized to LF (write with newline='\n' from now on).
Task 3: complete (commits 6dd04b9..311fd61, tests: npm test →    Duration  164.92s (tests 83%, import 8%, setup 4%, environment 4%, transform 1%))
Task 4: Ruling: the glass draws at min(1.25, devicePixelRatio) pixels per CSS pixel, not half (Task 2) — at half resolution the grout lines blurred into a graph-paper grid; the shader is a few dozen operations a pixel — cost if wrong: a little more GPU time; Lighter graphics still stills it.
Task 4: complete (commits 311fd61..38bc214, tests: npm test →    Duration  162.14s (tests 83%, import 8%, setup 4%, environment 4%, transform 1%))
Task 5: Ruling: the bench slides in only (no exit slide) — two benches sharing the frame would overlap windows and focus; cost: a little less motion.
Task 5: Ruling: the bench area scrolls inside itself at every size for now — stations not yet rebuilt (plan 4) are taller than the window and must stay reachable; the page itself never scrolls; cost: none once stations fit.
Task 5: committed (client tests 233/233, npm run check passed) but NOT yet recorded with task-done (usage limit). RESUME HERE: run task-done for Task 5 (base 38bc214), look at the frame in the check game, then Task 6.
Task 5: complete (commits 38bc214..b5de950, tests: npm test →    Duration  161.75s (tests 83%, import 9%, setup 4%, environment 4%, transform 1%))
Task 5 follow-up: Ruling: the readout class is .dotface, not .dot — .dot is the Market crowd's pop-and-bob animation (tokens.css) and faded every display number; found in the game, guarded by 'keeps display numbers clear of the crowd's bobbing dots' (added after the fix: the bug was visual) — cost if wrong: none.
Task 6: Ruling: notifications print at the top under the strip (design §7.3), so lab.test's 'notifications sit above the bottom bar' now checks they sit at the top; the chat still reserves the bottom bar — cost if wrong: none. Ruling: chat names are made readable on the dark display they sit in (readableOn(color, "dark")), not on the viewer's glass — the display is always dark — cost if wrong: none.
Task 6: complete (commits 70bbeb2..def76b7, tests: npm test →    Duration  169.95s (tests 84%, import 8%, setup 4%, environment 3%, transform 1%))
Task 7: Ruling: View3D takes an onReady(handle) callback (stage + markDirty) so an object can be changed in place (new measurements) instead of rebuilt — the plan's interface had no way to redraw a still object after a change — cost if wrong: none (optional prop).
Task 7: complete (commits def76b7..fa95f4f, tests: npm test →    Duration  179.80s (tests 83%, import 8%, setup 4%, environment 3%, transform 2%))
Task 8: Ruling: the shared orb test inputs live in test/orbInputs.ts (not views.ts) — views.ts already exports a different SCORES; one small file avoids a name clash — cost if wrong: none. Ruling: CoreStage adds setStill so lighter graphics and reduced motion still the core in place — cost: none.
Task 8: complete (commits fa95f4f..1d71774, tests: npm test →    Duration  184.92s (tests 82%, import 8%, setup 4%, environment 4%, transform 1%))
Task 8 follow-up: tuned in the game: rings at crystal radius x (1.42 + 0.17 i), the camera frames the outermost ring; steel darker (0x8f959e, roughness 0.46, env 0.75). A trained AI shows about 11 rings (R5: one per skill past its threshold plus the words ring).
Task 9: Ruling: the trophy shelf stays outside the drawer (right column, under the feed) — surprises.test expects it on the Company screen, and the owner should see trophies at a glance; the drawer holds Team, Licenses, Partner apps and Goal (when the world has one) — cost if wrong: none. Ruling: displays power on with no fill (lit at both ends) — a browser pane that pauses animations showed them dark — cost: none.
Task 9: complete (commits 962ede9..aa78cb8, tests: npm test →    Duration  173.02s (tests 83%, import 8%, setup 4%, environment 4%, transform 1%))
Task 10: Ruling: the Lighter graphics switch sits at the foot of the station column, not in the strip — at 1280 wide the strip wrapped its labels and cut the company name — cost if wrong: none (same switch, same name).
Task 10: Ruling: the goal's progress sits beside the trophy shelf, out of the drawer — npm run e2e (solo step 29) found it hidden behind the drawer's Team key; a player should always see how close they are to winning — covered by goal.test 'stays in view on the Company station, outside the drawer' (RED then GREEN) — cost if wrong: none.
Final: fixed #1 the WebGL glass never started under StrictMode (cleanup lost the context) — glassStrict.test 'keeps drawing with WebGL after React mounts it twice' RED→GREEN. Re-judged the Task 4 ruling on the real shader: full-resolution tiles look right; it stands.
Final: fixed #2 dragging the core opened the AI window — dragGuard.test (3 cases) RED→GREEN; View3D swallows the click that ends a drag (capture phase).
Final: fixed #7 e2e solo step 32 looked for the Licenses region in the closed drawer — the test now opens the drawer's Licenses key first (locator only, behaviour unchanged); goal-progress fixed earlier (53b5dff).
Final: fixed #3 every measurement change rebuilt every ring — core.test 'changes in place when only the day's new customers change' RED→GREEN (rings rebuild only when count, detail or size change).
Final: re-graded #13 (camera framed on ceil(rings)) from Minor to Important: a stronger AI looked smaller, against R5's size-from-score — fixed with a still camera for the largest possible core; core.test 'keeps the camera still as the AI grows, so a stronger AI looks bigger' RED→GREEN.
Final: fixed #6 the slow-frame meter tripped on healthy 50/72/75/100 Hz screens and saved the game's own switch — graphics.test 'leaves a healthy 72 Hz screen alone at 30 frames a second' and 'does not remember lighter graphics the game switched on by itself' RED→GREEN.
Final: fixed #4 the strip overflowed with long values and the overload chip — rescue.test 'sits on a warning plate under the strip' and shell.test 'keeps the computer chip on one line' RED→GREEN; measured in the check game at 1280x720 with a 32-char name, $1,234,567,890 and -$12,345,678: strip scrollWidth 1244 = clientWidth, emblem 40 px, all speed keys inside.
Final: fixed #5 the column and bench sat in the auto row — CSS only (no unit-testable layout in happy-dom): measured before 84/576/0 rows and an early column end; after 94.8/0/593 rows and the column ends at 702 on a 720 window.
Final: spec §7.1/§7.3/§11 updated: Lighter graphics at the foot of the station column; warnings (rescue, overload) on hazard plates.
Final: minor (deferred): #8 glass GPU cost: the aurora is computed twice a pixel and the still glass's frame loop keeps waking; render the aurora into a tile-resolution texture (plan 6 performance pass).
Final: minor (deferred): #9 no webglcontextlost/restored handling for the glass (GPU reset under CUDA load).
Final: minor (deferred): #10 View3D reads the first IntersectionObserver entry; should read the last.
Final: minor (deferred): #11 focus outlines use --lamp, too light on light glass for deep colors (about 2:1); use --accent outside displays.
Final: minor (deferred): #12 the training lamp repeats 'Training' to screen readers; the auto-lighter note's live region may not announce.
Final: note: Task 9's trophies-outside-the-drawer and the goal beside them override owner assumption A2 in part — tell the owner.
Final: npm test 894/894, npm run check clean, npm run e2e 3/3 (9.5 min) after the fix pass. Final review: fresh reviewer (opus); Critical/Important fixed with RED→GREEN tests; minors deferred above.
