# Next steps: read this first (written 2026-11-23, updated 2026-11-24 after the release)

This file holds what a new session needs and must not depend on any earlier chat. Branch: `m2-two-friends` (the working line; never force-push it, never merge it into `m1-it-learns` unless the owner asks). Project rules: `CLAUDE.md`. The owner is not technical: no technical questions or option menus; decide, record, proceed.

## The order the owner set
1. ~~**The release (plan C).**~~ **Done 2026-11-24** (plan `docs/superpowers/plans/2026-11-24-next-level-c-release.md`, ADR 0010 section 8, owner's notes `docs/reports/2026-11-24-release-notes.md`).
2. **The look redesign comes first** (the owner's decision of 2026-11-26): the game gets its new look before the document, so the document's screenshots show it. Design `docs/superpowers/specs/2026-11-26-look-redesign-design.md` (approved), ADR `docs/decisions/0011-look-redesign.md`, plans `docs/superpowers/plans/2026-11-26-look-redesign-*.md` (plans 1 and 2 done 2026-11-26: the bench, the glass, the 3D core and Company; the emblem maker, the lab door and founding; plans 3–6 are written one at a time, just before each starts; next is plan 3, the 3D map and the Market station). Done means: the owner approves the new look in the running game, npm test / check / e2e pass, pushed, progress log updated.
3. **The ultimate game document**: after the redesign. Brief below.
4. **Every computer (phase G, spec section 5.7).** Not yet scheduled by the owner; the spec places it after the release. If it is done after the document, update the document.

## Where the training upgrade stands
| Phase (spec `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`, section 6) | State |
|---|---|
| A measuring stick | done; the stick (text 1) lives in `data/yardstick/` since the release |
| B engine 3 | done |
| C training text 2 | done; **the main game's data since the release** |
| D a voice of its own | done; the owner's try passed 2026-11-24 |
| E brain design 2 | screened 2026-11-22 and **dropped** (ADR 0010 section 7) |
| F the release | **done 2026-11-24** (ADR 0010 section 8) |
| G every computer | not started; plan to be written when the owner schedules it |

**The main game today** (`npm run dev`, `data/`): engine 3, training text 2, new worlds `balance-v4` (first reading: conversations 30%, the other seven 10% each), rival packs v1 and v2 (old worlds) and v3 (new worlds; text 2). The owner's seven worlds are `balance-v2` and keep their rules and rivals; their next runs read text 2. Backup of the worlds before the swap: `data/server-backup-2026-11-24`. The second game (`data2/`) is retired; its test worlds stay on disk.

**Friends:** each friend runs the helper's setup file once more before the next game together (the game's data changed). Their download is about 750 MB; helpers no longer download rival packs.

**Measured result** (ADR 0010 section 5.4, first AI, new against before): small talk sensible in blind reading 55-59% against 18-27%; nonsense 10% against 38-46%; speaks as a person 0% against 7-12%; follow-ups 73-78% against 0-2%; says it is an AI 89-98% against 0%; says its name 54-74% against 2-3%; lowercase facts 87% against 63-72%; bench 52.6-53.6 against 47.5-49.7. Gate on text 2 (gate 3): 5 of 7, as before.

**Known gaps (report them, never lower a target):** small brains often get their own name wrong; third-generation bench 60.2 against target 64; a played session with Large and Huge brains is outstanding; gate G3 (the facts specialist's knowledge lead, +3.3 against 8.7 needed: text 2's conversations teach facts too) and G7 (cleaning) fail, as at gate 2.

## What the release did (for reference)
- New worlds: `balance-v4`; old worlds unchanged. A build into an empty root makes text 2 (`--text 1` for the stick's text).
- Rival pack 3 (`data/rivals/v3`, keys `-r3-`); the server keeps every pack loaded (earlier ones whatever their data) and seats new worlds only from the current data's pack.
- The Conversations card: "Short chats between people and a friendly AI: questions, requests and small talk."
- The swap: `docs/handoff/release/swap.sh`; old worlds checked with `docs/handoff/release/check_worlds.ts` (run it on a copy, never on `data/server`).

## Working rules that matter here
- **The owner uses the graphics card for other projects.** Ask before any long GPU run and say roughly how long. A competing load would also invalidate speed measurements.
- Never run a test game on `data/server/` (use the `check` game, `data-check/`, whose folders link to `data/`); never write into `data/corpus`, `data/judge`, `data/novelty`, `data/rivals` or `data/yardstick` except by a planned rebuild; never build into or delete `data/rivals/v1` or `v2`.
- Blind reading (`docs/handoff/next-level/blind_read.py`) is the measure of talk; judge-based rules are guards within one text only.
- Git-ignored, so only on this PC: `data/`, `data2/`, `ml/runs/`. Their conclusions are in ADR 0010.
- The owner likes multiple-choice questions for creative decisions, and plain answers with honest limits.

## The ultimate game document (next)
**The owner's request of 2026-11-04, word for word:** "I want you to create an extensive text document that would contain the the entirety of the game: The core idea, how it works, describing the technical systems behind the game, describing the layout and mechanics, explain in detail how the game looks like externally, internally and so on. So that someone reading this document will fully grasp our entire work here. By someone I mean other AI system's so do not even worry about making it easy to read for people. Save this for yourself too just in case. Give it to me in pdf, include screenshots too if you can. I want to create a "marketing, financial, technical and most importantly design agents" but first they need to understand our work front to back. Do not change anything yet just document it"

**Restated by the owner on 2026-11-23:** "the ultimate game document describing every aspect of the game from the code, layout, mechanics, system, design and so on".

- **Readers:** other AI systems, which will become the owner's marketing, financial, technical and, above all, design agents.
- **Deliver:** a PDF with real screenshots, plus a text copy kept in the repository for future sessions.
- **Scope:** idea, design, every screen and layout, mechanics, economy and balance, code structure, systems, training, multiplayer, data, decisions and their reasons.
- **Source of truth:** the code as it runs after the release, checked against the design PDFs, specs and ADRs. Mark what is built, set aside (AI Hub) or planned (tutorial, phase G); never describe a plan as a feature.
- **Documentation only:** it changes nothing in the game. Take screenshots from a separate data root, never from the owner's `data/server/`.
