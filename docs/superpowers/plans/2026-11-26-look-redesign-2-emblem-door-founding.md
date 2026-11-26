# Look redesign, plan 2: the emblem maker, the lab door and founding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A company's logo becomes an emblem built with dials (C58), shown flat everywhere and as a 3D badge where it is large; any main color can found a company because the glass follows it (C57); the start of the game is the lab door (C62) and founding happens on the bench of the approved mockup (C61).

**Architecture:** The protocol's `Logo` gains an optional `emblem`; old logos (and rival packs') map to emblems that look like them, so every saved world keeps working and the simulation's rules only relax (no new simulation version). One flat SVG renderer draws every emblem; a 3D badge (the plan 1 engine) shows the same art on an extruded, beveled frame. New bench controls (knob, slider, hue dial, cartridge, lever, keypad) build the founding bench and the lab door.

**Tech Stack:** as plan 1 (React 19, TypeScript, Tailwind 4, motion, zustand, zod, Vitest + happy-dom, three.js 0.187).

**Spec:** `docs/superpowers/specs/2026-11-26-look-redesign-design.md` §5.4, §6, §8.2, §9 (owner decisions C57, C58, C61, C62); ADR `docs/decisions/0011-look-redesign.md` §7. Visual reference: `docs/superpowers/specs/2026-11-26-look-redesign-mockups/founding.html` (founding exactly as shown, C61). Plan 1 and its ledger: `docs/superpowers/plans/2026-11-26-look-redesign-1-bench-and-company.md`, `docs/handoff/sdd/ledger-look-redesign-1.md`.

## Global Constraints

- Everything in plan 1's Global Constraints (R1, R2, R3, R5, R7, roles and names kept, reduced motion, no network at play time, browser checks only in the `check` game, commit and push per task).
- The e2e tests' hooks stay: the sign-in has one textbox and a "Continue" button; Home has a "World name" field, goal choices clickable by their text (e.g. "Reach 1,000,000 customers"), a "Goal target" field, a "Fair mode" checkbox (`getByLabel("Fair mode").check()`), a "Start" button; founding has a name field with the placeholder "Acme Intelligence", quick-color buttons named by their preset label (e.g. /Ember/), a "Found company" button; the lobby has `data-testid="invite-code"`, "Copy link" and "Open for business"; the join page has "Join" (exact) and the Fair mode notice text.
- Old logos keep rendering as themselves; a saved world loads and replays unchanged (the reducer only accepts more than before).
- Emblem fields and limits (spec §9.3): frame `circle | corners | shield | diamond | rosette`; corners 3–12; turn 0–355 in steps of 5; roundness 0–100; rings 0–3; ring weight 1–6; glow 0–100; symbol one of the sixteen `LOGO_SYMBOLS` or `null` for letters; letters 1–3 of `A–Z0–9`; symbol size 60–130; pattern `none | rays | grid | dots | waves`; finish `metal | enamel | glass | light`.
- Invite codes use `ABCDEFGHJKMNPQRSTUVWXYZ23456789` (`apps/server/src/worlds/inviteCode.ts`).

## Review Focus

- **Old and rival logos:** every saved logo (`shape`, `symbol`, `initials`, no `emblem`) and every rival pack's logo (`initials: ""`) renders as a recognisable emblem, never blank. Test: Task 1 (mapping) and Task 2 (render).
- **Letters with no letters:** a company name with no letters or digits ("!!!") still gets valid emblem letters. Test: Task 5.
- **A hostile company name or letters:** shown as text, never markup. Test: Task 2 and Task 5.
- **The lever by keyboard:** "Found company" works with Enter and Space and stays disabled until the name is long enough. Test: Task 3 and Task 5.
- **A pasted invite code** in lower case or with spaces still joins. Test: Task 6.

---

### Task 1: The emblem type, its mapping and the palette rule (C57)

**Files:**
- Modify: `packages/protocol/src/world.ts` (`Emblem`, the enums, `Logo.emblem`, `emblemOf`)
- Modify: `packages/sim/src/project.ts` (project `emblem` when present)
- Modify: `packages/sim/src/palette.ts` (`paletteProblem` keeps the hex and "two different colors" rules only)
- Test: `packages/protocol/test/schemas.test.ts`, `packages/sim/test/world.test.ts`

**Interfaces:**
- Produces (protocol): `EMBLEM_FRAMES`, `EMBLEM_PATTERNS`, `EMBLEM_FINISHES` (as consts), `Emblem` (zod strict object with the fields above), `type Emblem`, `Logo` with `emblem: Emblem.optional()`, and `emblemOf(logo: Logo): Emblem` (the logo's emblem, or the one an old logo maps to: circle → circle; rounded-square → corners 4, turn 45, roundness 40; hexagon → corners 6, turn 0; shield → shield; diamond → diamond; symbol kept, `null` → letters = initials or "A"; rings 0, ring weight 3, glow 0, symbol size 100, pattern none, finish enamel; corners 6 and roundness 0 where the frame does not use them).

- [ ] **Step 1: Write the failing tests**

```ts
// schemas.test.ts
it("accepts an emblem and refuses one out of range", () => {
  const emblem = { frame: "corners", corners: 6, turn: 0, roundness: 0, rings: 2, ringWeight: 3, glow: 40, symbol: "spark", letters: "LL", symbolSize: 100, pattern: "rays", finish: "metal" };
  expect(Logo.safeParse({ shape: "circle", symbol: "spark", initials: "", emblem }).success).toBe(true);
  expect(Logo.safeParse({ shape: "circle", symbol: "spark", initials: "", emblem: { ...emblem, corners: 13 } }).success).toBe(false);
  expect(Logo.safeParse({ shape: "circle", symbol: "spark", initials: "", emblem: { ...emblem, letters: "<b>" } }).success).toBe(false);
  expect(Logo.safeParse({ shape: "circle", symbol: "spark", initials: "" }).success).toBe(true);
});
it("maps old logos to emblems that look like them", () => {
  expect(emblemOf({ shape: "rounded-square", symbol: "bolt", initials: "" })).toMatchObject({ frame: "corners", corners: 4, turn: 45, roundness: 40, symbol: "bolt" });
  expect(emblemOf({ shape: "hexagon", symbol: null, initials: "AL" })).toMatchObject({ frame: "corners", corners: 6, symbol: null, letters: "AL" });
  expect(emblemOf({ shape: "shield", symbol: null, initials: "" }).letters).toBe("A");
});
// world.test.ts (replaces the "too dark" cases)
it("founds a company in any main color: the glass follows it (C57)", () => {
  expect(paletteProblem({ primary: "#1a1a1a", secondary: "#4f8cff" })).toBeNull();
  expect(paletteProblem({ primary: "#4f8cff", secondary: "#4F8CFF" })).toMatch(/different/i);
});
```

The existing world.test case that founds with a dark palette now expects success, and the projection test expects `emblem` to pass through when present.

- [ ] **Step 2: Run them to see them fail** — `npx vitest run packages/protocol/test/schemas.test.ts packages/sim/test/world.test.ts` → FAIL.
- [ ] **Step 3: Implement** the schema, `emblemOf`, the projection and the palette rule.
- [ ] **Step 4: Run every test** — `npm test` and `npm run check` → PASS (fix any test that asserted the old "too dark" refusal to assert the new rule).
- [ ] **Step 5: Commit** `feat: the emblem in the logo, old logos mapped, any main color founds (C57)`.

### Task 2: The flat emblem

**Files:**
- Create: `apps/client/src/emblem/art.ts` (the frame path and the pieces of an emblem as SVG markup strings, shared by the flat emblem and the 3D badge's face)
- Modify: `apps/client/src/components/Logo.tsx` (draws `emblemOf(logo)`; same props; keeps `data-shape`, `data-symbol`; adds `data-frame`)
- Test: `apps/client/test/emblem.test.tsx`

**Interfaces:**
- Produces: `framePath(e: Emblem): string` (a closed SVG path in a 100 × 100 box: circle; `corners` as a regular polygon from the top, turned by `turn`, corners rounded by `roundness`; the shield of the old logo; a diamond; a rosette of `corners` scallops); `emblemSvg(e: Emblem, palette: Palette, o?: { idPrefix?: string }): string` (the whole emblem as an `<svg>` string: face in the finish, pattern clipped to the frame, rings, symbol or letters as SVG `<text>`, glow); letters escaped as text.
- `Logo` renders `<svg role="img">` from the same parts as React elements (not `dangerouslySetInnerHTML`), so letters stay text.

- [ ] **Step 1: Write the failing tests**

```ts
it("draws a rival's old logo as an emblem with its symbol", () => {
  render(<Logo logo={{ shape: "hexagon", symbol: "rocket", initials: "" }} palette={PALETTE} title="Rival" />);
  const svg = screen.getByRole("img", { name: "Rival" });
  expect(svg.getAttribute("data-symbol")).toBe("rocket");
  expect(svg.getAttribute("data-frame")).toBe("corners");
  expect(svg.innerHTML).toContain(PALETTE.primary);
});
it("draws letters as text, never markup", () => {
  const { container } = render(<Logo logo={{ shape: "circle", symbol: null, initials: "AB", emblem: { ...EMBLEM, symbol: null, letters: "XYZ" } }} palette={PALETTE} />);
  expect(container.querySelector("text")?.textContent).toBe("XYZ");
});
it("closes every frame's path", () => {
  for (const frame of EMBLEM_FRAMES) expect(framePath({ ...EMBLEM, frame })).toMatch(/Z\s*$/);
});
```

- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement** `art.ts` and the new `Logo` (frames and patterns as in `founding.html`'s `framePath` and `patternSvg`; finishes as its `emblemSvg`; the sixteen symbols as in today's `Logo.tsx`).
- [ ] **Step 4: Run every test, check** → PASS.
- [ ] **Step 5: Commit** `feat(client): emblems drawn flat everywhere, old logos as themselves`.

### Task 3: Bench controls for setting things: knob, slider, hue dial, cartridge, lever, keypad

**Files:**
- Create: `apps/client/src/bench/Knob.tsx`, `Slider.tsx`, `HueDial.tsx`, `Cartridge.tsx`, `Lever.tsx`, `Keypad.tsx` (styles in `bench.css`)
- Test: `apps/client/test/bench.test.tsx`

**Interfaces:**
- `Knob(props: { label: string; value: number; min: number; max: number; step?: number; onChange(v: number): void; format?(v: number): string; size?: number })`: `role="slider"` with `aria-valuemin/max/now/text` and `aria-label={label}`, focusable; drag up or down, the wheel, arrows, Page Up/Down, Home/End; the value in a small display under it in the dot face.
- `Slider(props: { label: string; value: number; min: number; max: number; step?: number; onChange(v: number): void })`: a native `<input type="range" aria-label={label}>` under a machined cap and lit track.
- `HueDial(props: { label: string; hue: number; color: string; onChange(hue: number): void })`: a `Knob` over 0–359 with the hue ring and a jewel of `color`.
- `Cartridge(props: { label: string; primary: string; secondary: string; pressed: boolean; onClick(): void })`: a `<button aria-pressed>` named `label` showing the two colors.
- `Lever(props: { label: string; disabled?: boolean; onPull(): void; status?: ReactNode })`: a `<button>` named `label`; pulling (drag down past three quarters, a click, Enter or Space) animates the handle down and calls `onPull` once; disabled keeps the handle up and grey.
- `Keypad(props: { label: string; alphabet: string; value: string; maxLength: number; onChange(v: string): void })`: a display input named `label` (typing works) and one key per character of `alphabet` plus "Delete"; keys append; lower case and spaces typed or pasted become upper case without spaces.

- [ ] **Step 1: Write the failing tests** (one per control: roles and names; Knob ArrowUp steps and Home goes to min; Slider change; Cartridge `aria-pressed`; Lever calls `onPull` once on Enter and never when disabled; Keypad keys append and " ab c" becomes "ABC").
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement** the six controls (styling from `founding.html`: `.knob`, `.knob.hue`, `.slider`, `.cart`, `.slot`/`.handle`, the key styling of `.keycap`).
- [ ] **Step 4: Run every test, check** → PASS.
- [ ] **Step 5: Commit** `feat(client): knobs, sliders, hue dials, cartridges, the lever and the keypad`.

### Task 4: The 3D emblem badge

**Files:**
- Create: `apps/client/src/emblem/badgeStage.ts`, `apps/client/src/emblem/EmblemBadge.tsx`
- Test: `apps/client/test/emblem.test.tsx`

**Interfaces:**
- Consumes: `framePath`, `emblemSvg` (Task 2); `Stage3D`, `View3D` (plan 1); `metal`, `darkMetal`.
- Produces: `makeBadgeStage(emblem: Emblem, palette: Palette, o: { still: boolean; motion: string }): Stage3D & { setLook(e: Emblem, p: Palette): void }`: a group named `"badge"` holding the frame extruded from `framePath` (depth 0.18 of its size, beveled; sides in metal, or dark metal for the light finish) and a face named `"badge-face"` textured with `emblemSvg` (drawn into a canvas texture once the SVG image loads; the stage marks itself changed then); a slow sway, drag to turn; the logo motions: `logo:pulse` a heartbeat scale, `logo:shimmer` a light sweeping across, `logo:orbit` a spark circling the frame, `logo:still` nothing.
- `EmblemBadge(props: { logo: Logo; palette: Palette; size: number; motion?: string; label: string })`: `View3D` with the flat `Logo` as its fallback.

- [ ] **Step 1: Write the failing tests** — `EmblemBadge` shows the flat logo without WebGL; `makeBadgeStage` builds `"badge"` and `"badge-face"`; `setLook` with another frame replaces the frame geometry.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every test, check, and look at a badge in the `check` game** → PASS.
- [ ] **Step 5: Commit** `feat(client): the 3D emblem badge`.

### Task 5: The founding bench (C61) and opening for business

**Files:**
- Modify: `apps/client/src/screens/Lobby.tsx` (the founding bench, the open-for-business bench)
- Delete: `apps/client/src/components/LogoBuilder.tsx`, `apps/client/src/components/PalettePicker.tsx` (replaced)
- Modify: `apps/client/src/theme/glass.ts` (`powerUpGlass(): void`, a pulse the glass reads as `useLook.power`), `apps/client/src/glass/GlassBackground.tsx` (brightens the aurora for 2.5 s on a pulse)
- Test: `apps/client/test/lobby.test.tsx` (rewritten for the new bench, same behaviours)

**Interfaces:**
- The founding bench (as `founding.html`): Name plate (`input` placeholder "Acme Intelligence", dot face), Colors (two `HueDial`s named "Main color" and "Second color", a brightness `Slider` under each, twelve `Cartridge`s named by preset label, the glass line "Your lab glass is light/dark: it follows your main color."), the emblem press (`EmblemBadge` size 260 with `data-testid="preview"` around it, live), the deck (`Knob`s Frame, Corners, Round, Turn, Rings, Glow; symbol keys: one `<button aria-pressed>` per symbol named by the symbol and one named "Letters"; `Slider`s Ring weight and Symbol size; `KeyRow`s Pattern and Finish), Players here, Invite friends (`data-testid="invite-code"`, "Copy link"), and the `Lever` "Found company" (disabled until the name has `TEXT_LIMITS.companyName[0]` characters; pulling sends `createCompany` with `logo: { shape, symbol, initials, emblem }` where `shape`, `symbol` and `initials` are the closest old fields for older clients: frame circle → circle, corners 4 → rounded-square, corners 6 → hexagon, shield → shield, diamond → diamond, else circle; initials = the first two letters). Founding calls `powerUpGlass()` and the strip stamps the badge.
- Letters default to the company's initials (first letter of up to three words, `A–Z0–9` only), "A" when there are none; while Letters is chosen, typing the name updates them.
- The open-for-business bench: the `EmblemBadge` size 220, the name, the founding note, and for the host a big push button "Open for business"; others see "Waiting for the host to open the world…" with a waiting lamp.

- [ ] **Step 1: Rewrite the lobby tests for the bench** (same five behaviours as today: the preview shows the chosen palette and symbol; a deep main color founds without a warning and the glass line says light; exactly the chosen name, logo (with emblem) and palette are sent, from the keyboard (Enter on the lever) too; a hostile name stays text; the server's reason shows) plus: "!!!" gets letters "A"; open for business and inviting friends as today.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement** the bench, the power-up and the open-for-business bench.
- [ ] **Step 4: Run every test, check; found a company in the `check` game on dark glass (Ember) and on light glass (a deep main color)** → PASS.
- [ ] **Step 5: Commit** `feat(client): the founding bench and the lab powering up (C61)`.

### Task 6: The lab door (C62): ID badge, worlds rack, new-world bench, keypad

**Files:**
- Modify: `apps/client/src/screens/Home.tsx` (`Welcome`, `Home`)
- Modify: `apps/client/src/screens/Join.tsx`
- Modify: `apps/client/src/surprises/Goals.tsx` (`GoalPicker` as a rotary selector; the "Goal target" field and the deadline keep their names)
- Test: `apps/client/test/home.test.tsx`, `apps/client/test/join.test.tsx`

**Interfaces:**
- `Welcome`: an ID badge printer: "What should we call you?" over a display input; "Continue" prints the badge (the name in the dot face slides out of a slot, 700 ms, none under reduced motion) and then signs in.
- `Home`: a wall rack of world panels (lit for a running world, amber for not started, dim for an older version with its note; each with its name, day or "Not started yet", players, and "Open"); the new-world bench (name plate "World name" with placeholder "Friday night", the goal selector, "Goal target" and "Goal deadline", the `ToggleSwitch` "Fair mode" with its notice plate, the push button "Start"); the join keypad (`Keypad` "Invite code" over the invite alphabet, push button "Join").
- `Join`: the same keypad bench with the code filled in, the world's name and players in displays, the Fair mode notice plate, "Join" and "Back".

- [ ] **Step 1: Write the failing tests** — the existing home and join tests keep their assertions; add: a pasted " ab c23 " in the keypad joins `/join/ABC23`; the worlds rack shows a not-started world's panel with "Not started yet".
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every test, check, look at the door in the `check` game** → PASS.
- [ ] **Step 5: Commit** `feat(client): the lab door: ID badge, worlds rack, new-world bench and keypad (C62)`.

### Task 7: The plan in the running game

- [ ] **Step 1:** In the `check` game: sign in as a new guest, start a world with a goal and Fair mode, found one company on dark glass and one (in another world) on light glass, open for business; old worlds' and rivals' logos still show.
- [ ] **Step 2:** `npm test`, `npm run check`, `npm run e2e` → all pass (fix locators only where a control moved).
- [ ] **Step 3:** Progress entry "Look redesign, plan 2"; next-steps note updated; ledger copied to `docs/handoff/sdd/ledger-look-redesign-2.md`; commit and push.
- [ ] **Step 4:** A fresh whole-branch review; fix Critical and Important findings with tests; ledger the minors.
- [ ] **Step 5:** Show the owner the lab door and the founding bench.
