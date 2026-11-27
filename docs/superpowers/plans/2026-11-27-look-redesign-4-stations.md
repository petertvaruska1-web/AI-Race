# Look redesign, plan 4: every other station as a bench

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Money, the AI Lab (all five steps), the Playground, Benchmarks, Rivals, the Arena and the AI Directory each become a full-window bench of instruments (spec §10), with no page scrolling at 1440 × 900, as the Company and Market stations already are.

**Architecture:** Each station is rebuilt in its own file on the bench parts of plans 1–3 (`apps/client/src/bench/`), plus five new parts made in Task 1: ledger rows (the restyled `Bars`, so every window that lists money takes the look too), a vertical fader with a level meter, a needle gauge, a step dial and a stamped card. Station logic, commands and texts stay as they are; only arrangement and controls change. Long lists scroll inside their own displays.

**Tech Stack:** React 19, TypeScript, Tailwind 4 (`bench/bench.css`, `@layer components`), motion, zustand, Vitest + happy-dom, Playwright (Edge), three.js 0.187 (the shared engine and the 3D core of plan 1).

**Spec:** `docs/superpowers/specs/2026-11-26-look-redesign-design.md` (§5 controls and materials, §7.1 sizes, §10 every station) and `docs/decisions/0011-look-redesign.md`. Visual reference: look B in `docs/superpowers/specs/2026-11-26-look-redesign-mockups/directions.html`.

## The plans of this redesign (updated 2026-11-27)

Plans 1–3 are done (the bench, glass, 3D core and Company; the emblem, lab door and founding; the 3D map and the Market). Plan 1 listed "every other station and window" as one plan; it is two, so each stays a size one session can finish and check:

4. **This plan:** Money, AI Lab, Playground (chat terminal and interview; the games keep their current boards until plan 6), Benchmarks, Rivals, Arena, AI Directory.
5. **The windows and the 3D objects:** all 30 windows arranged into modules and displays, the six bought orb looks around the core, trophies, the data center and graphics card models, the Victory ceremony.
6. **Challenges and pictures:** the pacing director, the challenge chamber and its five 3D boards and dice, duels, battles and showcases at a watchable pace; the pictures of C63 replacing every emoji.
7. **The generic-design hunt and the finish:** spec §12, the deferred minors of every plan, the final checks and the owner's approval.

## Global Constraints

- R1: the AI's words are shown as written; nothing here edits, cuts or reorders them.
- R2: visuals of another company use only what its public view already shows.
- R3: new 3D views draw only while visible and changing, honor lighter graphics (`useGraphics`) and reduced motion (`motion.ts`).
- R5: the AI core's look comes only from `orbParams`.
- R7: plain words on every label; hardware engraving in capitals only on plate labels and keycaps (A1).
- Every control keeps its role and accessible name, or its tests change in the same task. The browser tests' names stay exactly: the station links; `1 Data` … `5 Versions` (buttons); `Start training`; `Send`; `Your message`; `Message`; `Launch` (exact); `Save`; `Write`; `Notebook text`; `Product name`; `Resume`; `Pause`; `Start a battle`; `Start the battle`; `Against`; `License it`; `👍 Good answer`; the `Playground mode` group; the `Report cards` region; the test ids `chat-log`, `overall`, `score-*`, `version-*`, `dataset-*`, `plan-cost`, `fund`, `income-*`, `cost-*`; the launch dialog's slider.
- No page scroll at 1440 × 900 on any station (check with the frame's scroll box: `scrollHeight === clientHeight`); at 1280 × 720 the bench may scroll inside its frame; below 1280 wide modules stack. Long content scrolls inside its own `Display scroll`.
- Text on the glass outside displays uses the glass's ink; other companies' colors outside displays use `useReadable()` (§5.3); lamp colors only inside dark displays.
- Old worlds keep working: no protocol, sim or server change in this plan.
- Browser checks only in the `check` game (`data-check/`, ports 8788/5175), never on `data/server/`.
- Each task ends with `npm test` and `npm run check` passing at the repository root and a commit on `m2-two-friends` ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push at the end of the plan (and whenever the owner should see work).

## Review Focus

- **A brand-new company** (no AI, no product, no report, no customer, no version): every station shows its empty state as a bench, with no collapsed or overlapping modules. Each station task has a render test of the empty company.
- **A busy company** (3 products with 40-character names, 14 report cards, 20 versions, a running loan, every optional cost): long lists scroll inside their displays and never push the bench past the window. Tests: Task 2 (report rack and ledgers are scroll displays), Task 5 (the versions rack).
- **The keyboard:** every new control (fader, step dial, price knob, table selector) moves with the arrow keys and shows its focus ring. Tests: Task 1 (fader and step dial keys), Task 2 (price knob keys).
- **Light glass:** text outside displays and rival names stay readable on light glass. Tests: Task 7 (rival plates, as today's rivals test), Task 8 (scoreboard names inside the dark display keep the lamp colors; names outside use `useReadable`).
- **The browser tests' names:** the e2e solo and friends runs find the Lab steps, training, chat, battles and licensing by name. Test: `npm run e2e` in Task 9, and the unit tests named in each task.

---

### Task 1: Bench parts for the stations

**Files:**
- Modify: `apps/client/src/components/Bars.tsx` (ledger rows on the bench; same props)
- Create: `apps/client/src/bench/Fader.tsx`, `apps/client/src/bench/NeedleGauge.tsx`, `apps/client/src/bench/StepDial.tsx`, `apps/client/src/bench/StampCard.tsx`
- Modify: `apps/client/src/bench/bench.css`
- Test: `apps/client/test/bench.test.tsx`

**Interfaces:**
- Produces: `Bars` keeps its current props (rows with label, value, color, testId, onOpen); each row with `onOpen` stays a button named `Open <label>`, as today; values in the dot face, bars as level strips in a dark display.
- Produces: `Fader({ label, value, min, max, step = 1, onChange, level, readout, testId? })`: a vertical slider (native `input type="range"`, role slider, named `label`), a level meter beside it showing `level` (0–1), `readout` (string) in a small display above, arrow keys step it.
- Produces: `NeedleGauge({ label, value: number | null, max = 100, change?: number | null, testId? })`: an SVG arc gauge with ticks and a needle at `value`, the value in a small display ("Not measured" when null), the change as a lamp-colored sign (good up, danger down); `role="meter"` named `label` with `aria-valuenow` when measured.
- Produces: `StepDial<T extends string>({ legend, steps: { value: T; label: string }[], value: T, onChange })`: a knob whose pointer turns to the active step, with one key (button, `aria-current="step"` when active) per step around it, named by the step's label; ArrowLeft/ArrowRight from a focused key moves to the neighbor step.
- Produces: `StampCard({ grade, caption, color, onClick, label })`: a stamped card (a plate with the grade stamped in its color and a caption), a button named `label`.

- [ ] **Step 1: Write the failing tests** in `bench.test.tsx`: "a ledger row opens its window and shows its amount in the display" (render `Bars` with one row with `onOpen`, click the button named `Open <label>`, assert `onOpen` called, the amount inside a `.display`); "a fader moves with the arrow keys and shows its level" (`getByRole("slider", { name })`, keyDown ArrowUp → `onChange(value + step)`; the meter element's height style equals `level` × 100%); "a needle gauge says when nothing is measured, and points at a score" (null → text "Not measured" and no `aria-valuenow`; 62 → `aria-valuenow="62"`, the needle's rotation for 62 differs from 0's); "a step dial marks the current step and moves with the arrows" (`aria-current="step"` on the active key; ArrowRight on it → `onChange(next)`); "a stamped card opens with its name" (button named `label`, grade text present).
- [ ] **Step 2: Run them** — `npx vitest run apps/client/test/bench.test.tsx`; expected: the five new tests fail (missing modules, `Bars` without a display).
- [ ] **Step 3: Build the five parts** and their styles (look B: dark displays with inner shadow, Doto in displays, knurled knob from `Knob`'s styles, lamp-colored needle and level).
- [ ] **Step 4: Run** `npx vitest run apps/client/test/bench.test.tsx` and the money, reports and windows tests — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): bench parts for the stations: ledger rows, fader, needle gauge, step dial, stamped card`.

### Task 2: The Money bench

**Files:**
- Modify: `apps/client/src/screens/Money.tsx`, `apps/client/src/spending/Spending.tsx` (`FundPanel` as a module), `apps/client/src/components/Computers.tsx` (module contents), `apps/client/src/bench/bench.css`
- Test: `apps/client/test/money.test.tsx`, `apps/client/test/computers.test.tsx`, `apps/client/test/reports.test.tsx`

**Interfaces:**
- Consumes: `Bars` (ledger rows), `StampCard`, `Readout`, `Trace`, `Module`, `Display`, `PushButton`, `Knob`, `KeyRow` (Task 1 and earlier plans).
- Produces: `Money({ view })` as before; the module names stay as region names: `Coming in`, `Going out`, `Loans and investors`, `Investment fund`, `Report cards`, `Computers`, `Profit by month`, `Your AIs on sale`, `Plans and prices`.

Arrangement at 1440 × 900 (three columns above, three below): Coming in (this month's readout and change, then one ledger display with By plan, Other income, By customer group and By region, scrolling inside) · Going out (readout, ledger display, the serving-costs note on a plate, Loans and investors under it when shown) · Profit by month (the month bars in a display, `Open money` push button named as today, then Report cards as a rack of stamped cards in a scroll display); below: Your AIs on sale (one plate per product: name in the dot face, its AI and state, `Retire` key; `Launch an AI` push button) · Plans and prices (per plan: the AI as a `KeyRow` of Closed and the live products, legend `AI for the <plan> plan`; the price as a `Knob` within the plan's bounds named `<Plan> price` with its value in a display and a `Set the <plan> price` push button) · Computers and Investment fund in one column.

- [ ] **Step 1: Update the tests first:** money.test "sets a price within its bounds and refuses one outside them" becomes "sets a price with its knob, within its bounds" (ArrowUp on the slider named `Personal price` then click `Set the personal price` → `setPrice` with the stepped value; End key → the bound's max, never past it); "opens a month from its bar, a category from its row and the money window" keeps its names; add "shows a new company's Money bench with every module and its empty states" (no products: "Nothing on sale yet…", no reports: "Your first report card comes…"); add "keeps a long report-card rack inside its own display" (14 reports → the rack's display has the scroll class and 14 cards named `<Month>: <grade>`); the plan's AI select becomes radios named by product in the group `AI for the personal plan` (update "shows the server's reason when an AI that powers a plan is retired" to use them).
- [ ] **Step 2: Run** `npx vitest run apps/client/test/money.test.tsx` — expected: the new and changed tests fail.
- [ ] **Step 3: Rebuild the Money bench** as arranged above.
- [ ] **Step 4: Run** the money, computers, reports, spending and rescue tests — expected: pass.
- [ ] **Step 5: Look in the check game** at 1440 × 900 (fits without page scroll) and 1280 × 720, dark and light glass; fix what is off.
- [ ] **Step 6: Commit** — `feat(client): the Money station as a bench`.

### Task 3: The AI Lab's frame and the mixing desk

**Files:**
- Modify: `apps/client/src/screens/lab/LabLayout.tsx` (step dial; the summary plate), `apps/client/src/screens/lab/DataStep.tsx` (mixing desk), `apps/client/src/surprises/Notebook.tsx` and `Lessons.tsx` (their cards as modules beside the desk), `apps/client/src/bench/bench.css`
- Test: `apps/client/test/lab.test.tsx`

**Interfaces:**
- Consumes: `StepDial`, `Fader` (Task 1), `usePlan().setWeight`, `useBottomBar`.
- Produces: the Lab's steps as `StepDial` keys named `1 Data` … `5 Versions`; the summary plate keeps `plan-cost`, `Start training` and the bottom-bar reservation; each dataset channel keeps `data-testid="dataset-<id>"` and its slider named `<title> share`.

The desk: one channel per dataset side by side, each with the share in a small display at the top, the fader with its level meter, the title engraved under it and a `See examples` key; the examples of the chosen channel show in one display under the desk (as text, R1). The notebook and lessons modules stand at the desk's right.

- [ ] **Step 1: Write the failing tests:** "mixes the reading on a desk of faders, one per dataset" (each `<title> share` slider is vertical (`aria-orientation="vertical"`) inside `dataset-<id>`; ArrowUp raises the weight by 1; each channel's display shows its share of the total); "shows a channel's examples as text under the desk" (click `See examples` in the Books channel → its samples appear in the examples display, markup as text); "turns the step dial to the step on screen" (`5 Versions` gets `aria-current="step"` after a click; the dial keeps the five names).
- [ ] **Step 2: Run** `npx vitest run apps/client/test/lab.test.tsx` — expected: the three new tests fail.
- [ ] **Step 3: Build the step dial frame, the summary plate and the mixing desk.**
- [ ] **Step 4: Run** the lab tests (all, including "keeps plain words on every control" and the bottom bar's reservation) — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): the AI Lab's step dial, summary plate and mixing desk`.

### Task 4: The AI Lab's Prepare and Design steps

**Files:**
- Modify: `apps/client/src/screens/lab/PrepareStep.tsx`, `apps/client/src/screens/lab/DesignStep.tsx`, `apps/client/src/bench/bench.css`
- Test: `apps/client/test/lab.test.tsx`

**Interfaces:**
- Consumes: `ToggleSwitch`, `RotarySelector`, `KeyRow`, `Knob`, `Module`, `Display`.
- Produces: the same plan changes as today (`usePlan` setters), the same names on every control (radio groups keep their legends, toggles their labels).

Prepare: the cleaning level as a `RotarySelector` (legend `Cleaning`), each cleaning option as a `ToggleSwitch` with its lamp, and the before-and-after examples in displays (`What this removes` stays a list). Design: Grow from, Brain size, Attention span, Training length and Learning style as `RotarySelector`s (more than five options) or `KeyRow`s (up to five), each with its plain explanation on a plate; "Review what it knows" as a toggle switch; the test question as a display you type into.

- [ ] **Step 1: Write the failing tests:** "sets the cleaning on a rotary selector and its options on switches" (the `Cleaning` group's radios; each option is a switch with its lamp); "designs the brain on selectors with their explanations" (each of the five groups by legend; changing Brain size reprices as the existing pricing test expects).
- [ ] **Step 2: Run** — expected: fail.
- [ ] **Step 3: Rebuild both steps.**
- [ ] **Step 4: Run** the lab tests — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): the AI Lab's Prepare and Design steps on the bench`.

### Task 5: The training chamber and the versions rack

**Files:**
- Modify: `apps/client/src/screens/lab/TrainStep.tsx`, `apps/client/src/screens/lab/VersionsStep.tsx`, `apps/client/src/components/LineChart.tsx` (an oscilloscope display), `apps/client/src/bench/bench.css`
- Test: `apps/client/test/train.test.tsx`, `apps/client/test/lab.test.tsx`

**Interfaces:**
- Consumes: `Core` (plan 1; its `training` input already streams the mix's colors), `Lamp`, `Readout`, `Display`, `NeedleGauge` or a progress ring.
- Produces: the run's stage names, `Game time`, the two traces (`How wrong its guesses are`, `What it's learning`) with their expand behavior, `What it says` as text; each version a cartridge with `data-testid="version-<label>"` that opens its AI window, with a small still core, its label and overall score in displays.

Train: the chamber module (heavy) with the live core and its streams, the run status lamp and stage name, a progress gauge and game time; the two traces as oscilloscope displays; `What it says` as a scrolling display. Versions: a rack of cartridges (scrolls inside its display past one row), `Train further` push button.

- [ ] **Step 1: Write the failing tests:** "shows a run in the training chamber with its status lamp and progress" (status lamp state `on` while training, `good` when done; the progress readout); "racks every version as a cartridge that opens its AI" (20 versions → 20 `version-*` cartridges inside one scroll display; click opens the `ai` window).
- [ ] **Step 2: Run** — expected: fail.
- [ ] **Step 3: Build the chamber, the oscilloscope displays and the rack** (small cores still, rendered once, so twenty of them cost nothing while idle).
- [ ] **Step 4: Run** the train and lab tests — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): the training chamber and the versions rack`.

### Task 6: The Playground

**Files:**
- Modify: `apps/client/src/screens/Playground.tsx`, `apps/client/src/components/ChatLog.tsx`, `apps/client/src/surprises/Interview.tsx` (its mode on the bench), `apps/client/src/bench/bench.css`
- Test: `apps/client/test/playground.test.tsx`, `apps/client/test/coaching.test.tsx`

**Interfaces:**
- Consumes: `KeyRow` (the mode keys, group `Playground mode`), `Display`, `PushButton`.
- Produces: the chat as a terminal display (`data-testid="chat-log"`; your lines and the AI's replies as text), coaching keys under each reply named as today (`👍 Good answer` …), `Your message` and `Send` kept; the AI picker stays a native select (the friends e2e picks an option by text) dressed as a display with step keys beside it (`Previous AI`, `Next AI`).

The bench: the AI picker and mode keys on a plate across the top; the terminal display filling the middle; the typing display and `Send` on a plate at the bottom; the games keep today's boards inside a module until plan 6's chamber; the interview mode on the same bench.

- [ ] **Step 1: Write the failing tests:** "talks in a terminal display" (chat-log is a display; a sent message and the streamed reply appear in it as text); "steps through the AIs with keys beside the picker" (`Next AI` selects the next option and the chat switches to it).
- [ ] **Step 2: Run** — expected: fail.
- [ ] **Step 3: Rebuild the Playground bench.**
- [ ] **Step 4: Run** the playground, coaching, games and surprises tests — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): the Playground as a bench with a chat terminal`.

### Task 7: Benchmarks and Rivals

**Files:**
- Modify: `apps/client/src/screens/Benchmarks.tsx`, `apps/client/src/screens/Rivals.tsx`, `apps/client/src/bench/bench.css`
- Test: `apps/client/test/benchmarks.test.tsx`, `apps/client/test/rivals.test.tsx`

**Interfaces:**
- Consumes: `NeedleGauge`, `Readout`, `Module`, `EmblemBadge` (plan 2, 3D from 64 px), `useReadable`.
- Produces: Benchmarks keeps `overall` and `score-<id>`; the version picker becomes step keys around a display (`Previous version`, `Next version`, the label in the display). Rivals keeps every name the rivals tests use (`Try it`, `Battle`, `License it`, the rival window).

Benchmarks: the overall score in a large display with its change, the personality words on a plate, eight needle gauges in a 4 × 2 rack, each with its plain description. Rivals: the arch-rival's plate wide at the top (3D badge, name, type, AI, public scores, customers), the other rivals' plates in a row under it.

- [ ] **Step 1: Write the failing tests:** "reads each category on a needle gauge" (eight `meter`s named by category; an unmeasured one says "Not measured"); "steps between versions with keys" (`Previous version` shows the parent's scores); "stands each rival on a plate with its badge" (the arch-rival plate first and wide; names readable on light glass, as today).
- [ ] **Step 2: Run** — expected: fail.
- [ ] **Step 3: Rebuild both benches.**
- [ ] **Step 4: Run** the benchmarks, rivals and rivalsEverywhere tests — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): Benchmarks on needle gauges and Rivals on plates`.

### Task 8: The Arena and the AI Directory

**Files:**
- Modify: `apps/client/src/screens/Arena.tsx`, `apps/client/src/screens/Directory.tsx`, `apps/client/src/bench/bench.css`
- Test: `apps/client/test/arena.test.tsx`, `apps/client/test/directory.test.tsx`, `apps/client/test/licensing.test.tsx`

**Interfaces:**
- Consumes: `RotarySelector` (the tables), `Display`, `PushButton`, `Lamp`.
- Produces: the league tables as a scoreboard display (rank, company, value in the dot face) with the table chosen on a rotary selector (legend `Tables`); Battles as a list display with `Start a battle`; the Directory's four catalogs (`Shared by players`, `License offers`, `Ask a player`, `Rivals' AIs on sale`) as modules with catalog displays and push buttons, every name as today.

- [ ] **Step 1: Write the failing tests:** "shows a league table on a scoreboard chosen by a rotary selector" (the `Tables` radios; the rows inside a display; other companies' revenue stays hidden, as today's test); "lists each catalog in its own display" (the four modules by name; a friend's shared AI with `Try it`).
- [ ] **Step 2: Run** — expected: fail.
- [ ] **Step 3: Rebuild both benches.**
- [ ] **Step 4: Run** the arena, battles, directory and licensing tests — expected: pass.
- [ ] **Step 5: Commit** — `feat(client): the Arena's scoreboard and the AI Directory's catalogs`.

### Task 9: The plan in the running game

- [ ] **Step 1:** In the check game, a world with a trained and launched AI (training takes about a minute on the RTX 3050; the owner's game must be closed) and a fresh world: every station at 1440 × 900 (no page scroll) and 1280 × 720, on dark and light glass, empty and busy; fix what is off, each fix with its test where a test can see it.
- [ ] **Step 2:** `npm test`, `npm run check`, `npm run e2e` (background, about 11 minutes) pass.
- [ ] **Step 3:** A fresh whole-plan review (superpowers:requesting-code-review, most capable model, this plan's Review Focus); fix Critical and Important findings with tests; ledger the minors.
- [ ] **Step 4:** Progress entry, next-steps note, the ledger copied to `docs/handoff/sdd/ledger-look-redesign-4.md`; commit and push.
- [ ] **Step 5:** Show the owner the stations in their game.
