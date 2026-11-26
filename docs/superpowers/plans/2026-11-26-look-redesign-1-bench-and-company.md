# Look redesign, plan 1: the bench, the glass, the 3D engine and the Company station

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every screen of a world sits in the new "Instrument bench" frame over the pixelated glass, with the bench's shared surfaces and controls, a working 3D engine and the 3D AI core, and the Company station finished as the first complete bench to show the owner.

**Architecture:** The shared components every screen already uses (`Panel`, `Button`, `Toggle`, `Field`, `Stat`) are rebuilt on new bench primitives in `apps/client/src/bench/`, so all screens take the look at once; the frame (`AppShell`, `TopBar`, `SideNav`, windows, toasts, chat) becomes the station frame; the glass is one full-window WebGL shader; one shared three.js renderer draws 3D objects into their own canvases; the Company screen is rebuilt as a bench around the 3D core.

**Tech Stack:** React 19, TypeScript, Tailwind 4, motion, zustand, Vitest + happy-dom, three.js 0.187, @fontsource (Barlow Semi Condensed, Doto).

**Spec:** `docs/superpowers/specs/2026-11-26-look-redesign-design.md` (owner decisions C57–C63) and `docs/decisions/0011-look-redesign.md`. The visual reference is look B in the approved mockups (copied into the repository by Task 1): `docs/superpowers/specs/2026-11-26-look-redesign-mockups/directions.html` (pick "B · Instrument bench") and `.../founding.html`. Take colors, gradients, screw and display styling, the glass shader and the knob and lamp styling from them.

## The plans of this redesign

Written one at a time, each just before it starts; each leaves the game working.

1. **This plan:** the bench primitives, the glass, the station frame, windows and overlays, the 3D engine, the AI core, the Company station. Ends with the owner seeing the Company station.
2. **The emblem, the lab door and founding:** the emblem type and its checks (protocol, sim, server), the flat and 3D emblem with the logo motions, the palette rule of C57, knobs, sliders, the lever and the keypad, the ID badge sign-in, the worlds rack, the new-world bench, the join keypad, the founding bench, the opening switch, the glass power-up.
3. **The 3D map:** the states from `us-atlas`, the nine regions, leader glow and heights, markers, controls and readouts; the Market station; the map in Company and in the rival and region windows.
4. **Every other station and window:** Money, AI Lab (with the mixing desk), Playground (chat terminal, interview), Benchmarks gauges, Rivals, Arena, AI Directory, all 30 windows, the orb looks in 3D, trophies, the data center and graphics card models, the Victory ceremony.
5. **Challenges and pictures:** the pacing director, the challenge chamber, the five 3D boards and dice, duels, battles and showcases at a watchable pace; the pictures of C63 replacing every emoji.
6. **The generic-design hunt and the finish:** screenshots of everything on both glasses, every review tool, fixes, the full test run, the docs, the owner's approval.

## Global Constraints

- R1: the AI's words are shown as written; nothing here may edit, cut or reorder them.
- R2: visuals of another company use only what its public view already shows.
- R3 (shared card): draw only what is on screen and changing; ambient motion at most 30 frames a second, 60 while the player drags; nothing while the tab is hidden; a "Lighter graphics" switch.
- R5: the AI core's look comes only from `orbParams` (`apps/client/src/orb/params.ts`); no tiers or unlocks.
- R7: plain words in every label.
- Every control keeps its role and accessible name (unit tests and `tests/e2e/*.spec.ts` find controls by role and name); if a name must change, the tests change in the same task.
- Reduced motion (`apps/client/src/motion.ts`) is honored by every new motion, the glass and the 3D.
- No network at play time: fonts are bundled; nothing loads from a font or map service.
- Old worlds keep working: nothing in this plan changes the protocol, the sim or the server.
- The glass rule (C57): light glass when `contrastRatio(main, "#ffffff") >= contrastRatio(main, "#000000")`, dark glass otherwise.
- Browser checks only in the `check` game (launch entry `check`: `data-check/`, ports 8788/5175), never on `data/server/`.
- Each task ends with `npm test` and `npm run check` passing at the repository root, a commit on `m2-two-friends` ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, and a push.

## Review Focus

- **No WebGL** (a blocked or missing graphics driver, the headless Edge of the e2e tests): the glass shows its CSS stand-in and every 3D object its flat stand-in, and nothing throws. Tests: Task 2 (glass fallback), Task 7 (`View3D` fallback), Task 8 (`Core` falls back to the 2D orb).
- **Long names and big numbers in fixed displays:** a 32-character company name, a world name of 40 characters, cash of $1,234,567,890 and a negative profit must not overflow the strip. Test: Task 5.
- **Reduced motion:** no bench slide, a still glass, a still core. Tests: Task 2 (frame policy), Task 5 (no slide), Task 8 (one frame).
- **Fast station switching:** switching stations twice quickly leaves exactly one bench showing the last station. Test: Task 5.
- **A hidden tab:** the glass and the 3D stop drawing. Tests: Task 2 and Task 7 (frame policy and due views with `hidden`).

---

### Task 1: Bundled fonts, the glass rule and the bench theme

**Files:**
- Modify: `apps/client/package.json` (add `@fontsource/barlow-semi-condensed@^5.3.0`, `@fontsource/doto@^5.3.0`)
- Modify: `apps/client/index.html` (remove the two `preconnect` links and the Google Fonts stylesheet)
- Modify: `apps/client/src/main.tsx` (import `@fontsource/barlow-semi-condensed/{400,500,600,700}.css` and `@fontsource/doto/{700,800}.css`)
- Create: `apps/client/src/theme/glass.ts`
- Modify: `apps/client/src/theme/tokens.css`
- Modify: `apps/client/src/screens/World.tsx` (call `applyBenchTheme` instead of `applyCompanyTheme`)
- Create: `docs/superpowers/specs/2026-11-26-look-redesign-mockups/directions.html` and `founding.html` (copied from `.superpowers/brainstorm/754-1795680674/content/`)
- Test: `apps/client/test/glass.test.ts`

**Interfaces:**
- Produces (in `theme/glass.ts`):
  - `type Glass = "light" | "dark"`
  - `GLASS_BASE: Record<Glass, string>` = `{ dark: "#0b0e15", light: "#f2f3f6" }` (the glass colors that text sits on, for contrast checks)
  - `glassFor(primary: string): Glass` (the C57 rule)
  - `lampColor(p: Palette): string`: the main color on dark glass; `mix("#ffffff", p.primary, 0.5)` on light glass (`mix` from `theme/palette.ts`)
  - `readableOn(color: string, glass: Glass, min = 3): string`: `color` if its contrast with `GLASS_BASE[glass]` is at least `min`, otherwise the first of `mix(target, color, k)` for k = 0.1, 0.2, … 1 that reaches it, where target is `#ffffff` on dark glass and `#000000` on light
  - `benchVars(p: Palette | null): Record<string, string>`: everything `companyThemeVars` sets, plus `--lamp`, `--lamp-rgb`, `--p-rgb`, `--s-rgb` (space-separated `r g b`) and `--accent` (`p.primary`, readable on its own glass by C57)
  - `applyBenchTheme(p: Palette | null, root = document.documentElement): void`: sets the variables and `root.dataset.glass`, and stores the palette in `useLook` (below); `null` means the neutral palette and dark glass
  - `useLook`: a zustand store `{ palette: Palette | null }` read by the glass (Task 2)

- [ ] **Step 1: Write the failing tests** in `test/glass.test.ts`

```ts
it("gives deep main colors light glass and bright ones dark glass", () => {
  expect(glassFor("#7f1d3a")).toBe("light");
  expect(glassFor("#1e3a8a")).toBe("light");
  expect(glassFor("#3b82f6")).toBe("dark");
  for (const { palette } of PALETTE_PRESETS) expect(glassFor(palette.primary)).toBe("dark");
});
it("lifts a deep lamp color so it glows in a dark display", () => {
  expect(contrastRatio(lampColor({ primary: "#7f1d3a", secondary: "#f59e0b" }), "#05070a")).toBeGreaterThanOrEqual(4.5);
  expect(lampColor({ primary: "#3b82f6", secondary: "#2dd4bf" })).toBe("#3b82f6");
});
it("makes any color readable on either glass and leaves readable ones alone", () => {
  for (let r = 0; r < 256; r += 32) for (let g = 0; g < 256; g += 32) for (let b = 0; b < 256; b += 32) {
    const c = `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
    for (const glass of ["dark", "light"] as const)
      expect(contrastRatio(readableOn(c, glass), GLASS_BASE[glass])).toBeGreaterThanOrEqual(3);
  }
  expect(readableOn("#3b82f6", "dark")).toBe("#3b82f6");
});
it("marks the root with its glass and sets the lamp", () => {
  const root = document.createElement("div");
  applyBenchTheme({ primary: "#7f1d3a", secondary: "#f59e0b" }, root);
  expect(root.dataset.glass).toBe("light");
  expect(root.style.getPropertyValue("--lamp")).toBe(lampColor({ primary: "#7f1d3a", secondary: "#f59e0b" }));
  applyBenchTheme(null, root);
  expect(root.dataset.glass).toBe("dark");
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/glass.test.ts`
Expected: FAIL, `theme/glass.ts` does not exist.

- [ ] **Step 3: Install the fonts and write `theme/glass.ts`**

Run `npm install -w @airace/client @fontsource/barlow-semi-condensed@^5.3.0 @fontsource/doto@^5.3.0` (the owner allowed these downloads on 2026-11-26), import the font files in `main.tsx`, remove the Google Fonts links, write `glass.ts`, and switch `World.tsx` to `applyBenchTheme`.

- [ ] **Step 4: Rewrite `tokens.css` for the bench**

Keep the Tailwind `@theme inline` names (`--color-bg`, `--color-surface`, `--color-surface-2`, `--color-text`, `--color-muted`, `--color-border`, `--color-danger`, `--color-brand`, `--color-brand-2`, `--color-on-brand`, `--color-brand-soft`, `--color-good`) so every existing class keeps working, now mapped to the bench: `bg` = the glass base, `surface` and `surface-2` = display blacks, `text` = ink, `muted` = ink 2, `border` = hairline. Define `:root[data-glass="dark"]` and `:root[data-glass="light"]` blocks with `--ink`, `--ink-2`, `--ink-3`, `--hair`, `--metal`, `--brush`, `--screw`, `--screw-lo`, `--engrave`, `--engrave-a`, `--engrave-shadow`, `--good`, `--warn`, `--danger`, with the values of `directions.html`'s `:root` and `.glass-light` blocks (`--warn: #facc15`). Fonts: `--font-sans` and `--font-display` = `"Barlow Semi Condensed"`, `--font-dot` = `"Doto"`. Body background transparent (the glass of Task 2 paints behind). Copy the two mockups into `docs/superpowers/specs/2026-11-26-look-redesign-mockups/`.

- [ ] **Step 5: Run the tests**

Run: `npx vitest run apps/client/test/glass.test.ts apps/client/test/palette.test.ts`
Expected: PASS.

- [ ] **Step 6: Full checks and commit**

Run `npm test` and `npm run check` (both pass), then:

```bash
git add apps/client/package.json package-lock.json apps/client/index.html apps/client/src/main.tsx apps/client/src/theme apps/client/src/screens/World.tsx apps/client/test/glass.test.ts docs/superpowers/specs/2026-11-26-look-redesign-mockups
git commit -m "feat(client): bundled bench fonts, the glass rule (C57) and the bench theme"
git push
```

### Task 2: The glass and the aurora

**Files:**
- Create: `apps/client/src/graphics/frames.ts`
- Create: `apps/client/src/store/graphics.ts`
- Create: `apps/client/src/glass/shader.ts`
- Create: `apps/client/src/glass/GlassBackground.tsx`
- Modify: `apps/client/src/app.tsx` (render `<GlassBackground />` once, before the routed screen)
- Test: `apps/client/test/graphics.test.tsx`

**Interfaces:**
- Consumes: `useLook`, `glassFor` (Task 1); `reducedMotion()` (`motion.ts`).
- Produces:
  - `graphics/frames.ts`: `type FramePolicy = { fps: number; still: boolean }`; `framePolicy(o: { reducedMotion: boolean; lighter: boolean; hidden: boolean; interacting: boolean }): FramePolicy` returning `{fps: 0, still: true}` when hidden, reduced motion or lighter; `{fps: 60, still: false}` when interacting; else `{fps: 30, still: false}`; `due(nowMs: number, lastMs: number, fps: number): boolean` (`fps > 0 && nowMs - lastMs >= 1000 / fps - 1`); `createFrameMeter(): { record(frameMs: number): boolean }`, which keeps the last 90 frame times and returns true once, the first time 90 are kept and their median is above 40 ms (the glass loop and, in Task 7, the 3D loop call `useGraphics.getState().setLighter(true, true)` when it does).
  - `store/graphics.ts`: `useGraphics` (zustand) `{ lighter: boolean; auto: boolean; setLighter(lighter: boolean, auto?: boolean): void }`, kept in `localStorage["airace.lighterGraphics"]` as `"1"` or `"0"`.
  - `glass/GlassBackground.tsx`: `GlassBackground(): JSX.Element`, a fixed, full-window, `aria-hidden` layer behind everything (`z-index: -1`, `pointer-events: none`) reading `useLook` and `useGraphics`.

- [ ] **Step 1: Write the failing tests**

```ts
it("stops drawing when hidden, stills for reduced motion and lighter graphics", () => {
  expect(framePolicy({ reducedMotion: false, lighter: false, hidden: true, interacting: true })).toEqual({ fps: 0, still: true });
  expect(framePolicy({ reducedMotion: true, lighter: false, hidden: false, interacting: false }).still).toBe(true);
  expect(framePolicy({ reducedMotion: false, lighter: true, hidden: false, interacting: false }).still).toBe(true);
  expect(framePolicy({ reducedMotion: false, lighter: false, hidden: false, interacting: false }).fps).toBe(30);
  expect(framePolicy({ reducedMotion: false, lighter: false, hidden: false, interacting: true }).fps).toBe(60);
});
it("is due once a frame's time has passed", () => {
  expect(due(1033, 1000, 30)).toBe(true);
  expect(due(1010, 1000, 30)).toBe(false);
  expect(due(5000, 0, 0)).toBe(false);
});
it("asks for lighter graphics once frames stay slow", () => {
  const meter = createFrameMeter();
  const answers = Array.from({ length: 120 }, () => meter.record(55));
  expect(answers.slice(0, 89).some(Boolean)).toBe(false);
  expect(answers.filter(Boolean)).toHaveLength(1);
  const fine = createFrameMeter();
  expect(Array.from({ length: 120 }, () => fine.record(16)).some(Boolean)).toBe(false);
});
it("remembers lighter graphics", () => {
  useGraphics.getState().setLighter(true);
  expect(localStorage.getItem("airace.lighterGraphics")).toBe("1");
});
it("shows the CSS glass in the company's glass when WebGL is missing", () => {
  useLook.setState({ palette: { primary: "#7f1d3a", secondary: "#f59e0b" } });
  const { container } = render(<GlassBackground />);
  const fallback = container.querySelector("[data-glass-fallback]");
  expect(fallback?.getAttribute("data-glass")).toBe("light");
  expect(container.querySelector("canvas")).toBeNull();
  expect(container.firstElementChild?.getAttribute("aria-hidden")).toBe("true");
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/graphics.test.tsx`
Expected: FAIL, modules missing.

- [ ] **Step 3: Write `frames.ts`, `graphics.ts`, `shader.ts` and `GlassBackground.tsx`**

The shader is the mockups' glass shader (`founding.html`, uniforms `uRes`, `uTime`, `uP`, `uS`, `uLight`, `uCell`, `uMouse`, `uPower`), with the cell size 15 CSS pixels and the light-glass aurora of `directions.html`. Draw into a canvas at half the device pixel ratio (the tiles hide it). One `requestAnimationFrame` loop asks `framePolicy` each frame (`hidden` from `document.visibilityState`); a still policy draws one frame whenever the palette, the glass or the size changes. The pointer lamp follows `pointermove` on `window`. `uLight` eases toward the target glass over about a second. Without WebGL (`canvas.getContext("webgl")` is null, or throws), render `<div data-glass-fallback data-glass={glass}>` with a CSS rendering of the same colors: a radial wash of the second color over the main color's glass tint, with a 15-pixel tile grid drawn by two repeating linear gradients.

- [ ] **Step 4: Mount it in `app.tsx`**

`App` returns `<><GlassBackground />{screen}</>` for every route, including the loading text.

- [ ] **Step 5: Run the tests, full checks, and look at it**

Run: `npx vitest run apps/client/test/graphics.test.tsx`, then `npm test` and `npm run check`: PASS. Start the `check` preview, open a world and confirm the glass and aurora move behind the old panels and that the tab uses little GPU (the glass stops when the tab is hidden).

- [ ] **Step 6: Commit**

```bash
git add apps/client/src/graphics apps/client/src/store/graphics.ts apps/client/src/glass apps/client/src/app.tsx apps/client/test/graphics.test.tsx
git commit -m "feat(client): the pixelated glass and aurora behind every screen"
git push
```

### Task 3: Bench surfaces: module, display, readout, lamp

**Files:**
- Create: `apps/client/src/bench/bench.css` (imported at the end of `theme/tokens.css`)
- Create: `apps/client/src/bench/Module.tsx`, `Display.tsx`, `Readout.tsx`, `Lamp.tsx`
- Modify: `apps/client/src/components/Panel.tsx` (render a `Module`, same props)
- Modify: `apps/client/src/components/Stat.tsx` (engraved label, value in a display in the dot face, same props and `data-testid`)
- Test: `apps/client/test/bench.test.tsx`

**Interfaces:**
- Produces:
  - `Module(props: { label?: ReactNode; hint?: ReactNode; actions?: ReactNode; weight?: "plate" | "heavy" | "thin"; className?: string; children?: ReactNode } & Omit<HTMLAttributes<HTMLElement>, "title">)`: a `<section>` plate with four screws, the label as an engraved `<h2>`, the hint under it, actions at the right of the label row.
  - `Display(props: { scroll?: boolean; label?: string; className?: string; children: ReactNode })`: a recessed dark display; with `scroll`, `role="region"`, `aria-label={label}`, `tabIndex={0}` and inner scrolling.
  - `Readout(props: { label: ReactNode; value: ReactNode; tone?: "lamp" | "good" | "danger" | "plain"; size?: "sm" | "md" | "lg"; testId?: string })`: engraved label over a display holding the value in the dot face, tabular figures.
  - `Lamp(props: { state: "on" | "ready" | "wait" | "problem" | "off"; label: string; blink?: boolean })`: `<span role="img" aria-label={label} data-state={state}>`.
  - CSS classes for later tasks: `.module`, `.module--heavy`, `.module--thin`, `.engrave`, `.display`, `.display--scroll`, `.dot`, `.lamp`, `.keycap`.

- [ ] **Step 1: Write the failing tests**

```ts
it("engraves a module's label as its heading", () => {
  render(<Module label="Customers">x</Module>);
  expect(screen.getByRole("heading", { name: "Customers" }).className).toContain("engrave");
});
it("keeps Panel's title, hint and actions", () => {
  render(<Panel title="Money" hint="This month" actions={<button type="button">Open</button>}>y</Panel>);
  expect(screen.getByRole("heading", { level: 2, name: "Money" })).toBeTruthy();
  expect(screen.getByText("This month")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Open" })).toBeTruthy();
});
it("lets a scrolling display be reached and named", () => {
  render(<Display scroll label="What people are saying">z</Display>);
  const region = screen.getByRole("region", { name: "What people are saying" });
  expect(region.tabIndex).toBe(0);
});
it("names a lamp by its state", () => {
  render(<Lamp state="ready" label="Ready to found" />);
  expect(screen.getByRole("img", { name: "Ready to found" }).dataset.state).toBe("ready");
});
it("shows a stat's value in a display, keeping its test id", () => {
  render(<Stat label="Cash" value={1200} format={(n) => `$${n}`} testId="cash" />);
  expect(screen.getByTestId("cash").closest(".display")).not.toBeNull();
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/bench.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Write the components and `bench.css`**

Module plates, screws, brushed metal and shadows as `.dir-b .inst` in `directions.html`; displays as `.dir-b .screen`; the dot face as `.dir-b .big` (glow in `--lamp`); lamps as `.lamp`/`.led`. Inside a display, text uses light ink on both glasses (displays are always dark). One engraved label per module (spec §2 note): capitals and wide spacing only on `.engrave` and on key caps.

- [ ] **Step 4: Rebuild `Panel` and `Stat` on them**

- [ ] **Step 5: Run the tests**

Run: `npx vitest run apps/client/test`, then `npm test` and `npm run check`. Expected: PASS (fix any test that asserted an old class name, keeping its intent).

- [ ] **Step 6: Commit**

```bash
git add apps/client/src/bench apps/client/src/theme/tokens.css apps/client/src/components/Panel.tsx apps/client/src/components/Stat.tsx apps/client/test
git commit -m "feat(client): bench modules, displays, readouts and lamps"
git push
```

### Task 4: Bench controls: push buttons, switches, key rows, rotary selectors

**Files:**
- Create: `apps/client/src/bench/PushButton.tsx`, `ToggleSwitch.tsx`, `KeyRow.tsx`, `RotarySelector.tsx`
- Modify: `apps/client/src/components/Button.tsx` (delegate to `PushButton`: primary → primary, secondary → secondary, ghost → quiet)
- Modify: `apps/client/src/components/Toggle.tsx` (delegate to `KeyRow`)
- Modify: `apps/client/src/components/Field.tsx` (`INPUT` becomes the display-style input class, `"display-input"`, defined in `bench.css`)
- Test: `apps/client/test/bench.test.tsx`

**Interfaces:**
- Produces:
  - `PushButton(props: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "quiet" })`: a native `<button type="button">`; primary has the lit ring in `--lamp`; pressed travel on `:active`; disabled keeps `title`.
  - `ToggleSwitch(props: { label: ReactNode; checked: boolean; onChange: (checked: boolean) => void; disabled?: boolean; hint?: ReactNode })`: a native `<input type="checkbox" role="switch">` under a lever switch with a lamp, labelled by `label`.
  - `KeyRow<T extends string>(props: { legend: string; options: { label: string; value: T; disabled?: boolean; title?: string }[]; value: T; onChange: (v: T) => void })`: a `<fieldset>` with an `sr-only` legend and native radios under illuminated key caps.
  - `RotarySelector<T extends string>(props: { legend: string; options: { label: string; value: T }[]; value: T; onChange: (v: T) => void; size?: number })`: a `role="radiogroup"` named `legend`; a knob that turns to the chosen option, options engraved around it as native radios; arrow keys step and wrap.

- [ ] **Step 1: Write the failing tests**

```ts
it("presses a primary button and keeps a disabled reason", async () => {
  const onClick = vi.fn();
  render(<><PushButton onClick={onClick}>Train</PushButton><PushButton disabled title="Train it first">Launch</PushButton></>);
  await userEvent.click(screen.getByRole("button", { name: "Train" }));
  expect(onClick).toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Launch" }).getAttribute("title")).toBe("Train it first");
});
it("flips a switch", async () => {
  const onChange = vi.fn();
  render(<ToggleSwitch label="Fair mode" checked={false} onChange={onChange} />);
  await userEvent.click(screen.getByRole("switch", { name: "Fair mode" }));
  expect(onChange).toHaveBeenCalledWith(true);
});
it("picks a key in a key row", async () => {
  const onChange = vi.fn();
  render(<KeyRow legend="Difficulty" options={[{ label: "Easy", value: "easy" }, { label: "Hard", value: "hard" }]} value="easy" onChange={onChange} />);
  await userEvent.click(screen.getByRole("radio", { name: "Hard" }));
  expect(onChange).toHaveBeenCalledWith("hard");
});
it("turns a rotary selector with the arrow keys", async () => {
  const onChange = vi.fn();
  render(<RotarySelector legend="Step" options={[{ label: "Data", value: "data" }, { label: "Prepare", value: "prepare" }]} value="data" onChange={onChange} />);
  screen.getByRole("radio", { name: "Data" }).focus();
  await userEvent.keyboard("{ArrowRight}");
  expect(onChange).toHaveBeenCalledWith("prepare");
  expect(screen.getByRole("radiogroup", { name: "Step" })).toBeTruthy();
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/bench.test.tsx`
Expected: FAIL for the four new tests.

- [ ] **Step 3: Write the controls and rebuild `Button`, `Toggle` and `INPUT` on them**

Styling from `directions.html` (`.dir-b .btn`, `.btn.primary`, `.speed button`) and `founding.html` (`.switch`, `.knob`). Push buttons show their text as written (no forced capitals in the DOM; capitals by CSS on key caps only), so accessible names do not change.

- [ ] **Step 4: Run every client test, full checks**

Run: `npm test` and `npm run check`. Expected: PASS; update tests that asserted old pill classes (for example `bg-brand`) to assert the control's state through its role and `aria-checked`/`checked` instead.

- [ ] **Step 5: Commit**

```bash
git add apps/client/src/bench apps/client/src/components/Button.tsx apps/client/src/components/Toggle.tsx apps/client/src/components/Field.tsx apps/client/test
git commit -m "feat(client): bench push buttons, switches, key rows and rotary selectors"
git push
```

### Task 5: The station frame

**Files:**
- Modify: `apps/client/src/components/AppShell.tsx` (the frame grid, the bench slide, the rescue plate)
- Modify: `apps/client/src/components/TopBar.tsx` (the top strip; `monthProfit` stays exported here)
- Modify: `apps/client/src/components/SideNav.tsx` (the station knob)
- Modify: `apps/client/src/components/bottomBar.ts` (offsets for the new frame)
- Test: `apps/client/test/shell.test.tsx`

**Interfaces:**
- Consumes: `Module`, `Display`, `Readout`, `Lamp` (Task 3), `PushButton`, `ToggleSwitch` (Task 4), `useGraphics` (Task 2).
- Produces:
  - `AppShell` (same props): root `<div data-frame="station">` filling the window (`h-dvh`, no document scroll); grid: the strip across the top, the station column at the left (the knob, the nine links, the world chat intercom under them), the bench at the right. The bench is keyed by `tab` inside `AnimatePresence mode="popLayout"`: it slides in from the right by 6% with opacity 0 → 1 over 280 ms, and out to the left; under reduced motion it changes instantly. Below 1280 × 720 the bench area scrolls inside the frame (`overflow-y: auto`); above, it never scrolls.
  - `TopBar` (same props and accessible names): one long plate: the emblem (today's `Logo` with `LogoMotion`), the company name (`player-text text-brand`, truncated with an ellipsis and its full text in `title`), the world name, the money button holding Cash and Profit this month as `Readout`s, the overload chip, `CardsChip`, the clock (date readout, host speed keys named as today: "Pause"/"Resume", "1×", "2×", "4×"), a `ToggleSwitch` named "Lighter graphics" bound to `useGraphics`, and the day bar.
  - `SideNav` (same props, still `nav` named "Screens" with nine links, `aria-current="page"` on the open one): a large decorative knob (`aria-hidden`) turned to `-120deg + index × 30deg` (transition 700 ms, `cubic-bezier(.3,1.45,.45,1)`, instant under reduced motion), then the links, each with a `Lamp` lit on the open station.

- [ ] **Step 1: Write the failing tests** (add to `shell.test.tsx`; keep the existing ones)

```ts
it("fills the window as a station frame without page scrolling", () => {
  const { container } = render(<AppShell view={viewOf(runningWorld())} tab="company"><p>Screen</p></AppShell>);
  expect(container.querySelector('[data-frame="station"]')).not.toBeNull();
});
it("keeps long names and big numbers inside the strip", () => {
  const base = viewOf(runningWorld());
  const long = "A".repeat(32);
  const view = {
    ...base,
    companies: base.companies.map((c) => (c.id === base.me.company?.id ? { ...c, name: long } : c)),
    me: { ...base.me, company: base.me.company && { ...base.me.company, cash: 1_234_567_890 } },
  };
  render(<AppShell view={view} tab="company"><p>Screen</p></AppShell>);
  const name = screen.getByText(long);
  expect(name.className).toContain("truncate");
  expect(name.getAttribute("title")).toBe(long);
  expect(screen.getAllByText(formatMoney(1_234_567_890)).length).toBeGreaterThan(0);
});
it("says why lighter graphics came on by itself", () => {
  useGraphics.getState().setLighter(true, true);
  render(<AppShell view={viewOf(runningWorld())} tab="company"><p>Screen</p></AppShell>);
  expect(screen.getByText("Switched on to keep the game smooth")).toBeTruthy();
});
it("switches lighter graphics from the strip", async () => {
  useGraphics.getState().setLighter(false);
  render(<AppShell view={viewOf(runningWorld())} tab="company"><p>Screen</p></AppShell>);
  await userEvent.click(screen.getByRole("switch", { name: "Lighter graphics" }));
  expect(useGraphics.getState().lighter).toBe(true);
});
it("shows one bench after two quick station changes", () => {
  const view = viewOf(runningWorld());
  const { rerender } = render(<AppShell view={view} tab="company"><p>Company bench</p></AppShell>);
  rerender(<AppShell view={view} tab="market"><p>Market bench</p></AppShell>);
  rerender(<AppShell view={view} tab="money"><p>Money bench</p></AppShell>);
  expect(screen.getAllByText(/bench$/).map((e) => e.textContent)).toEqual(["Money bench"]);
});
```

(The suite already stubs reduced motion with `motion(true)`; add `it("changes stations instantly under reduced motion")` asserting the new bench is present synchronously after `rerender` with `motion(true)`.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/shell.test.tsx`
Expected: FAIL for the new tests.

- [ ] **Step 3: Rebuild the frame, the strip and the knob**

Layout and styling from `directions.html` look B (`.strip`, `.stations`, `.knob`, `.led`). The glass shows between plates; the strip and the station column are plates. The one orchestrated moment of a station change: after the bench slides in, its displays power on (a 160 ms scan from top to bottom, 40 ms apart, none under reduced motion). Next to the "Lighter graphics" switch, when `useGraphics().auto` is true, a small polite live text: "Switched on to keep the game smooth". The rescue banner becomes a hazard plate under the strip (diagonal stripes in `--danger`, a blinking `Lamp` named "Rescue needed", the same "Choose a rescue" button and text).

- [ ] **Step 4: Run every test, full checks, and look at it**

Run: `npm test`, `npm run check`: PASS. In the `check` game at 1440 × 900 and 1280 × 720, every station shows inside the frame and the page never scrolls.

- [ ] **Step 5: Commit**

```bash
git add apps/client/src/components apps/client/test/shell.test.tsx
git commit -m "feat(client): the station frame: top strip, station knob and sliding benches"
git push
```

### Task 6: Windows, toasts and the world chat on the bench

**Files:**
- Modify: `apps/client/src/windows/WindowFrame.tsx` (the heavy panel and its title plate)
- Modify: `apps/client/src/windows/WindowHost.tsx` (the dimmed-glass backdrop)
- Modify: `apps/client/src/components/Toasts.tsx` (printed slips under the strip)
- Modify: `apps/client/src/components/WorldChat.tsx` (the intercom)
- Modify: `apps/client/src/screens/LaunchDialog.tsx` (the same panel look; keep its `overflow-y-auto` and `max-h-[calc(100dvh-2rem)]` classes that `spending.test.tsx` checks)
- Test: `apps/client/test/windows.test.tsx`, `apps/client/test/worldchat.test.tsx`

**Interfaces:**
- Consumes: `Module`, `Display`, `PushButton`, `Lamp` (Tasks 3–4).
- Produces: `WindowFrame` (same props): `role="dialog"`, `aria-modal`, `aria-labelledby` as today; a heavy plate (`.module--heavy`) that still grows out of what was clicked; the title in an engraved title plate (`<h2 class="title-plate …">`); "Back" and "Close" as small push buttons with those names. `WindowHost`: the backdrop button "Close all windows" dims and blurs the glass. `Toasts`: each toast a slip that prints out of a slot under the strip (slides down 120 ms, holds, slides up), with a lamp in its kind's color; same text and roles. `WorldChat`: same names, buttons and behaviour; the closed state is an intercom plate in the station column with a speaker grille and the unread count in a small display; the open state a module with the messages in a scrolling display (`Display scroll label="World chat"`), the typing display and the "Send" push button; names keep their company colors through `readableOn(color, glass)`.

- [ ] **Step 1: Write the failing tests**

```ts
// windows.test.tsx, with its `openFrom` helper:
it("puts a window's title on an engraved plate", async () => {
  await openFrom({ kind: "customers" });
  expect(screen.getByRole("dialog", { name: "Customers" }).querySelector("h2.title-plate")).not.toBeNull();
});

// worldchat.test.tsx, with its `world` and `push` helpers (the sim refuses deep colors until plan 2,
// so the deep color is put into the view directly):
it("shows other players' names readable on the viewer's glass", async () => {
  const base = viewOf(world());
  const view = {
    ...base,
    companies: base.companies.map((c) => (c.id === "c2" ? { ...c, palette: { primary: "#1e3a8a", secondary: "#ffb020" } } : c)),
  };
  render(<WorldChat view={view} />);
  fireEvent.click(screen.getByRole("button", { name: /World chat/ }));
  push(view.world.id, { name: "Ben", companyId: "c2", text: "hi" });
  const name = await screen.findByText("Ben");
  expect(name.style.color).toBe(hexToRgbString(readableOn("#1e3a8a", "dark")));
});
```

(`hexToRgbString("#rrggbb")` is a three-line test helper giving `rgb(r, g, b)`, the form happy-dom reports.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/windows.test.tsx apps/client/test/worldchat.test.tsx`
Expected: FAIL for the new tests.

- [ ] **Step 3: Restyle the window frame, host, toasts, chat and launch dialog**

- [ ] **Step 4: Run every test and full checks**

Run: `npm test`, `npm run check`. Expected: PASS, with every existing window, chat and rescue test unchanged.

- [ ] **Step 5: Commit**

```bash
git add apps/client/src/windows apps/client/src/components/Toasts.tsx apps/client/src/components/WorldChat.tsx apps/client/src/screens/LaunchDialog.tsx apps/client/test
git commit -m "feat(client): windows as bench panels, printed toasts and the chat intercom"
git push
```

### Task 7: The 3D engine

**Files:**
- Modify: `apps/client/package.json` (add `three@^0.187.0`; dev: `@types/three@^0.187.0`)
- Create: `apps/client/src/three/support.ts`, `engine.ts`, `View3D.tsx`, `materials.ts`, `gear.ts`
- Test: `apps/client/test/three.test.tsx`

**Interfaces:**
- Consumes: `framePolicy`, `due` (Task 2); `useGraphics` (Task 2); `reducedMotion()`.
- Produces:
  - `support.ts`: `webglAvailable(): boolean`, cached; false when `getContext("webgl2")` returns null or throws (always false under happy-dom).
  - `engine.ts`:
    - `interface Stage3D { scene: THREE.Scene; camera: THREE.PerspectiveCamera; update(t: number, dt: number): boolean; pointer?(e: { type: "down" | "move" | "up"; dx: number; dy: number }): void; dispose(): void }` (`update` returns whether it is still moving; a still stage is drawn only when marked dirty)
    - `type ViewState = { id: number; visible: boolean; dirty: boolean; moving: boolean; last: number }`
    - `dueViews(views: ViewState[], now: number, policy: FramePolicy): number[]`: the ids to draw now: visible ones that are dirty, plus visible moving ones when `due(now, last, policy.fps)`; none when `policy.fps === 0` except dirty ones when `policy.still` and not hidden
    - `attachView(canvas: HTMLCanvasElement, stage: Stage3D): { markDirty(): void; setVisible(v: boolean): void; detach(): void }`: one shared `WebGLRenderer` (alpha, antialias, ACES tone mapping, sRGB output, pixel ratio ≤ 1.5, 1 when lighter) renders each due view at its size and copies it into the view's own 2D canvas with `drawImage`; one `requestAnimationFrame` loop for all views; stops when the tab is hidden
    - `markInteracting(ms = 1200): void` (the policy uses 60 fps until then)
    - the loop records each frame's time in a `createFrameMeter()` (Task 2) and switches on lighter graphics automatically when it says so
    - `sharedEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture` (`RoomEnvironment` through PMREM, made once)
  - `View3D.tsx`: `View3D(props: { make: () => Stage3D; label: string; fallback: ReactNode; className?: string; style?: CSSProperties })`: the fallback when `!webglAvailable()`; otherwise `<canvas role="img" aria-label={label}>` attached while mounted, visibility from an `IntersectionObserver`, size from a `ResizeObserver` (marks dirty), pointer events forwarded to `stage.pointer` (with `markInteracting`).
  - `materials.ts`: `metal(): MeshStandardMaterial` (`color #aeb4bd, metalness 1, roughness 0.38`), `darkMetal()` (`#3a3f47`, metalness 1, roughness 0.45), `enamel(color: string)` (`MeshPhysicalMaterial`, clearcoat 1, roughness 0.35), `glowing(color: string, intensity = 2.5)` (dark base, emissive); cached by arguments.
  - `gear.ts`: `gearRing(o: { radius: number; width: number; depth: number; teeth: number; toothDepth: number }): THREE.BufferGeometry` (an `ExtrudeGeometry` of a ring `Shape` whose outer edge carries `teeth` rectangular teeth, with a small bevel).

- [ ] **Step 1: Write the failing tests**

```ts
it("has no WebGL under the test DOM and shows the stand-in", () => {
  expect(webglAvailable()).toBe(false);
  render(<View3D make={() => { throw new Error("not called"); }} label="Your AI" fallback={<p>flat</p>} />);
  expect(screen.getByText("flat")).toBeTruthy();
});
it("draws only visible views that changed or move on time", () => {
  const v = (id: number, o: Partial<ViewState>): ViewState => ({ id, visible: true, dirty: false, moving: false, last: 0, ...o });
  const live = { fps: 30, still: false };
  expect(dueViews([v(1, { dirty: true }), v(2, { moving: true }), v(3, { visible: false, dirty: true })], 40, live)).toEqual([1, 2]);
  expect(dueViews([v(2, { moving: true, last: 30 })], 40, live)).toEqual([]);
  expect(dueViews([v(1, { dirty: true }), v(2, { moving: true })], 40, { fps: 0, still: true })).toEqual([1]);
});
it("builds a toothed gear ring", () => {
  const few = gearRing({ radius: 1, width: 0.08, depth: 0.05, teeth: 12, toothDepth: 0.03 });
  const many = gearRing({ radius: 1, width: 0.08, depth: 0.05, teeth: 48, toothDepth: 0.03 });
  expect(many.getAttribute("position").count).toBeGreaterThan(few.getAttribute("position").count);
});
```

(A hidden tab: `dueViews` is never asked then, because the loop stops on `visibilitychange`; the "fps 0, not still" policy returns `[]`; add that case to the second test.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/three.test.tsx`
Expected: FAIL, modules missing.

- [ ] **Step 3: Install three and write the engine**

Run `npm install -w @airace/client three@^0.187.0` and `npm install -w @airace/client -D @types/three@^0.187.0` (allowed by the owner on 2026-11-26), then write the five files.

- [ ] **Step 4: Run the tests and full checks**

Run: `npx vitest run apps/client/test/three.test.tsx`, `npm test`, `npm run check`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/client/package.json package-lock.json apps/client/src/three apps/client/test/three.test.tsx
git commit -m "feat(client): one shared 3D engine with flat stand-ins"
git push
```

### Task 8: The AI core in 3D

**Files:**
- Create: `apps/client/src/core/coreStage.ts`, `apps/client/src/core/Core.tsx`
- Modify: `apps/client/src/screens/Company.tsx`, `apps/client/src/screens/lab/TrainStep.tsx`, `apps/client/src/windows/AiWindow.tsx`, `apps/client/src/windows/ProductWindow.tsx` (use `Core` where `Orb` was)
- Test: `apps/client/test/core.test.tsx`

**Interfaces:**
- Consumes: `orbParams`, `OrbInput`, `OrbParams` (`orb/params.ts`); `Stage3D`, `View3D`, `gearRing`, `metal`, `darkMetal`, `glowing`, `sharedEnvironment` (Task 7); `Orb` (the flat stand-in).
- Produces:
  - `makeCoreStage(params: OrbParams, o: { still: boolean }): Stage3D & { setParams(p: OrbParams): void }`: a scene holding a mesh named `"crystal"` (icosahedron, flat facets, a shader material whose heart glows by gathering light along the refracted view ray: the `CORE_FS` look of `directions.html`, heart brightness from `brightness`, breathing from `smoothness`, flicker from `flicker`), `Math.ceil(params.rings)` gear rings named `"ring-0"`, `"ring-1"`, … (metal, `teeth = 24 + round(36 × ringDetail)`, the last one's opacity the fractional part of `rings`, spin from `swirl`, tilt wobble from `turbulence`, alternating directions), a halo sprite named `"halo"` (additive, size from `glowSoftness`), sparks (`Points`, rate from `sparkRate`) and, when `intake` or `arrivals` are set, streams of points flowing in from the edge in their colors and shares. Before the first AI (`rings === 0`): a small crystal in the empty cradle (two rings at zero opacity are not built). `update` returns false when `still`.
  - `Core(props: { input: OrbInput; size?: number; label?: string })` (same props as `Orb`; default size 280): `<View3D>` with `makeCoreStage(orbParams(input), { still: reducedMotion() || lighter })`, drag to turn (pointer `dx`, `dy` turn it; it eases back to a slow turn), and `<Orb input={input} size={size} label={label} />` as the fallback.

- [ ] **Step 1: Write the failing tests**

```ts
it("falls back to the drawn orb without WebGL", () => {
  render(<Core input={SCORED} label="Your AI v1" />);
  expect(screen.getByRole("img", { name: "Your AI v1" }).tagName).toBe("CANVAS");
});
it("builds one gear ring per measured ring and a crystal", () => {
  const stage = makeCoreStage(orbParams(SCORED), { still: true });
  const names = stage.scene.children.flatMap((c) => [c.name, ...c.children.map((k) => k.name)]);
  expect(names).toContain("crystal");
  const rings = Math.ceil(orbParams(SCORED).rings);
  for (let i = 0; i < rings; i++) expect(names).toContain(`ring-${i}`);
  expect(names).not.toContain(`ring-${rings}`);
});
it("is a bare crystal before the first AI", () => {
  const stage = makeCoreStage(orbParams(NONE), { still: true });
  expect(stage.scene.getObjectByName("ring-0")).toBeUndefined();
  expect(stage.update(0, 0.016)).toBe(false);
});
```

(`SCORED` and `NONE` as in `test/orb.test.ts`; move them to `test/views.ts` and import them in both.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/core.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Write `coreStage.ts` and `Core.tsx`; replace `Orb` in the four screens and windows**

- [ ] **Step 4: Run every test and full checks, and look at it**

Run: `npm test`, `npm run check`: PASS. In the `check` game, open "Release check": the core turns, drags, and shows its rings and glow; with "Lighter graphics" on it stands still and the GPU load drops.

- [ ] **Step 5: Commit**

```bash
git add apps/client/src/core apps/client/src/screens/Company.tsx apps/client/src/screens/lab/TrainStep.tsx apps/client/src/windows/AiWindow.tsx apps/client/src/windows/ProductWindow.tsx apps/client/test
git commit -m "feat(client): the AI core in 3D: a crystal with a glowing heart in geared rings"
git push
```

### Task 9: The Company bench

**Files:**
- Create: `apps/client/src/bench/Trace.tsx`
- Modify: `apps/client/src/screens/Company.tsx`
- Test: `apps/client/test/company.test.tsx`, `apps/client/test/bench.test.tsx`

**Interfaces:**
- Consumes: `Module`, `Display`, `Readout`, `Lamp` (Task 3), `PushButton`, `KeyRow` (Task 4), `Core` (Task 8), `RegionMap` (unchanged until plan 3), `Feed`, `TrophyShelf`, `TeamPanel`, `LicensesPanel`, `AppsPanel`, `GoalProgress` (unchanged inside).
- Produces:
  - `Trace(props: { values: number[]; label: string; zeroLine?: boolean })`: an `<svg role="img" aria-label={label}>` polyline in `--lamp` with a soft glow and a dashed zero line, drawn in on first show (the existing `chart-draw` class).
  - The Company bench (design §10): a three-column bench at 1440 × 900. Center, the heaviest module: the core cradle (`Core` size 300, still opening the AI window when the AI exists, as today), the AI's label in a dot-face readout, "Overall score N" and the personality words, the Train / Chat / Launch push buttons (same names and rules), the launch reason, "Customize your look", the lineup as plates, the arch-rival plate. Left: Customers (the count `Stat` with its test id `customers`, the group meter and group keys as today's `GroupBar`, the small `RegionMap` inside a display), Money (Cash and Profit this month readouts, `Trace` of profit by month labelled "Profit by month", the ownership chip, the latest report card as a stamped plate). Right: "What people are saying" (`Feed` inside `Display scroll label="What people are saying"`). Bottom: the drawer bay, a module whose `KeyRow` legend is "Company drawer" with keys Team, Licenses, Partner apps, Goal, Trophies (Goal only when the world has a goal, Licenses only where `LicensesPanel` renders today), showing one panel at a time; Team first.

- [ ] **Step 1: Write the failing tests**

```ts
it("draws profit by month as a trace", () => {
  render(<Trace values={[-2, 4, 9]} label="Profit by month" zeroLine />);
  expect(screen.getByRole("img", { name: "Profit by month" }).querySelector("polyline")).not.toBeNull();
});
it("keeps the team, licenses and partner apps in one drawer", async () => {
  render(<Company view={viewOf(launchedWorld())} />);
  const drawer = screen.getByRole("group", { name: "Company drawer" });
  expect(within(drawer).getByRole("radio", { name: "Team" })).toHaveProperty("checked", true);
  await userEvent.click(within(drawer).getByRole("radio", { name: "Partner apps" }));
  expect(screen.getByRole("heading", { name: "Partner apps" })).toBeTruthy();
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run apps/client/test/company.test.tsx apps/client/test/bench.test.tsx`
Expected: FAIL for the two new tests.

- [ ] **Step 3: Write `Trace` and rebuild the Company screen**

Every existing button, window opener and text stays (the existing `company.test.tsx` cases are the guard).

- [ ] **Step 4: Run every test and full checks**

Run: `npm test`, `npm run check`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/client/src/bench/Trace.tsx apps/client/src/screens/Company.tsx apps/client/test
git commit -m "feat(client): the Company station as a bench around the 3D core"
git push
```

### Task 10: The slice in the running game, and the owner's look

**Files:**
- Modify: `tests/e2e/solo.spec.ts`, `tests/e2e/friends.spec.ts` (only where a control's place or name changed)
- Modify: `docs/progress.md`
- Modify: `docs/handoff/next-steps.md`

- [ ] **Step 1: Look at every station in the `check` game**

Start the `check` preview; at 1440 × 900 and 1280 × 720, open each of the nine stations, three windows (customers, money, an AI), the chat, a toast and the rescue plate if reachable; confirm nothing is clipped, the page never scrolls, and the glass and core run. Screenshot the Company station.

- [ ] **Step 2: Run the end-to-end tests**

Run: `npm run e2e` (about 5 minutes). Expected: all pass; where a test fails because a control moved or was renamed, fix the test's locator to the new role and name, never the behaviour.

- [ ] **Step 3: Write the progress entry and the next-steps note**

`docs/progress.md`: a dated entry "Look redesign, plan 1" (what changed, the checks run). `docs/handoff/next-steps.md`: at the top, "The look redesign (spec `docs/superpowers/specs/2026-11-26-look-redesign-design.md`, plans `docs/superpowers/plans/2026-11-26-look-redesign-*.md`) comes before the ultimate game document; plan 1 done on <date>."

- [ ] **Step 4: Full checks and commit**

Run: `npm test`, `npm run check`: PASS.

```bash
git add tests/e2e docs/progress.md docs/handoff/next-steps.md
git commit -m "docs: look redesign plan 1 (bench, glass, 3D core, Company) checked in the game"
git push
```

- [ ] **Step 5: Show the owner the Company station**

Open the Company station of "Release check" in the browser pane on 1440 × 900 and ask the owner, as one multiple-choice question, whether the bench feels right before the other stations follow.
