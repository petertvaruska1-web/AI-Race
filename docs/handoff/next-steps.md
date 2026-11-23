# Next steps: read this first (written 2026-11-23)

This file holds what a new session needs and must not depend on any earlier chat. Branch: `m2-two-friends` (the working line; never force-push it, never merge it into `m1-it-learns` unless the owner asks). Project rules: `CLAUDE.md`. The owner is not technical: no technical questions or option menus; decide, record, proceed.

## The order the owner set
1. **The release (plan C).** The owner starts it **personally in another session**. Do not begin it on your own initiative.
2. **The ultimate game document**, right after the release (the owner's decision of 2026-11-23). Brief below.
3. **Every computer (phase G, spec section 5.7).** Not yet scheduled by the owner; the spec places it after the release. If it is done after the document, update the document.

## Where the training upgrade stands
| Phase (spec `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`, section 6) | State |
|---|---|
| A measuring stick | done |
| B engine 3 | done; **already what the game runs** (`ENGINE_VERSION` and `TRAINING_ENGINE` are 3) |
| C training text 2 | built in `data2/`; **not in the main game** |
| D a voice of its own | built; only works for AIs trained on text 2. **The owner's own try of the second game is still open** |
| E brain design 2 | screened 2026-11-22 and **dropped** (ADR 0010 section 7); today's design stays |
| F the release | **next; its plan is not written** |
| G every computer | not started; plan written after F |

**The main game today** (`npm run dev`, `data/`): engine 3, text 1, `balance-v3`, rival packs v1 and v2. A new AI there does not yet show the talking gains. **The second game** (launch entry `text2`, or `AIRACE_DATA=data2 AIRACE_SERVER_PORT=8788 AIRACE_WEB_PORT=5175 npm run dev`, then http://localhost:5175) shows them.

**Friends:** a helper from before engine 3 gets no work; each friend must run the helper's setup file again before the next game together.

**Measured result to carry into the release** (ADR 0010 section 5.4, first AI, new against today's): small talk sensible in blind reading 55-59% against 18-27%; nonsense 10% against 38-46%; speaks as a person 0% against 7-12%; follow-ups 73-78% against 0-2%; says it is an AI 89-98% against 0%; says its name 54-74% against 2-3%; lowercase facts 87% against 63-72%; bench 52.6-53.6 against 47.5-49.7.

**Known gaps (they do not block the release; report them, never lower a target):** small brains often get their own name wrong; third-generation bench 60.2 against target 64; a played session with Large and Huge brains is outstanding; gate G3 (the facts specialist's lead) and G7 (cleaning) failed at engine 2 and must be measured again.

## What the release must do (write plan C in `docs/superpowers/plans/` from spec section 5.6, then build)
- First, offer the owner the try of a first AI in the second game: it is phase D's open gate.
- Text 2 and engine 3 become the defaults. New worlds get a **new balance version** whose default reading for a new AI is **conversations 30%, the other seven 10% each**. Old worlds keep their rules; their AIs chat and battle as before; their next runs read text 2.
- Text 2's rival pack gets **its own version (rivals-v3)**: the pack in `data2/rivals/v2` has keys that still read "-e2-". The server keeps earlier packs loaded for rivals already seated in old worlds, whatever data they came from (today a pack from other data is refused, which would strand them), and seats new worlds only from a pack built on the current data. Never delete or build into `data/rivals/v1` or `v2`.
- In `data2/`: novelty index, judge (with the released engine), catalog, rival pack 3.
- Reword the Conversations card: "People talking and answering each other." fits text 1, not text 2.
- Checks: the feasibility gate again, `npm run e2e`, the scorecard against the baseline (ADR 0010 section 3; measuring stick frozen at a717d2e).
- Swap the data folders; the old ones become `data/yardstick/`. **Back up and protect `data/server/` (the owner's real worlds), and check every path that reads the yardstick before moving anything.**
- Update `docs/progress.md`, ADR 0010, `docs/handoff/sdd/ledger-next-level.md`, this file, and any `CLAUDE.md` commands that change.

## Working rules that matter here
- **The owner uses the graphics card for other projects.** Ask before any long GPU run and say roughly how long. A competing load would also invalidate speed measurements.
- Never run a test game on `data/server/`; never write into `data/corpus`, `data/judge`, `data/novelty` or `data/rivals` outside the release's planned swap.
- Blind reading (`docs/handoff/next-level/blind_read.py`) is the measure of talk; judge-based rules are guards within one text only.
- Git-ignored, so only on this PC: `data/`, `data2/`, `ml/runs/`. Their conclusions are in ADR 0010.
- The owner likes multiple-choice questions for creative decisions, and plain answers with honest limits.

## The ultimate game document (start after the release)
**The owner's request of 2026-11-04, word for word:** "I want you to create an extensive text document that would contain the the entirety of the game: The core idea, how it works, describing the technical systems behind the game, describing the layout and mechanics, explain in detail how the game looks like externally, internally and so on. So that someone reading this document will fully grasp our entire work here. By someone I mean other AI system's so do not even worry about making it easy to read for people. Save this for yourself too just in case. Give it to me in pdf, include screenshots too if you can. I want to create a "marketing, financial, technical and most importantly design agents" but first they need to understand our work front to back. Do not change anything yet just document it"

**Restated by the owner on 2026-11-23:** "the ultimate game document describing every aspect of the game from the code, layout, mechanics, system, design and so on".

- **Readers:** other AI systems, which will become the owner's marketing, financial, technical and, above all, design agents.
- **Deliver:** a PDF with real screenshots, plus a text copy kept in the repository for future sessions.
- **Scope:** idea, design, every screen and layout, mechanics, economy and balance, code structure, systems, training, multiplayer, data, decisions and their reasons.
- **Source of truth:** the code as it runs after the release, checked against the design PDFs, specs and ADRs. Mark what is built, set aside (AI Hub) or planned (tutorial, phase G); never describe a plan as a feature.
- **Documentation only:** it changes nothing in the game. Take screenshots from a separate data root, never from the owner's `data/server/`.
