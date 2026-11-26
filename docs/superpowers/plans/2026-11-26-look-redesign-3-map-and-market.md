# Look redesign, plan 3: the 3D map and the Market station

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The US map becomes a real 3D tabletop model (the 50 states and DC grouped into the game's nine regions, each region's top glowing in its leader's color and raised by the leader's share, the same for everybody), shown on the Market, Company, rival and region benches; and the Market station is rebuilt as a bench around it.

**Architecture:** Public-domain state outlines (`us-atlas`, already projected) are merged per region with `topojson-client` into pure geometry (`map/geometry.ts`, tested without a screen). A three.js stage (`map/mapStage.ts`) on the shared engine extrudes them into glass prisms with lit tops, marks and a turnable camera; `RegionMap` shows it through `View3D`, with the current tile map as its flat stand-in, real region keys and the hover readout. Who leads a region is a pure function of public data (`regions.ts`), so every player sees the same map.

**Tech Stack:** React 19, TypeScript, three.js 0.187 (shared engine of plan 1), `us-atlas` 3.0.1 (`states-albers-10m.json`), `topojson-client` 3.1.0 with `@types/topojson-client`, Tailwind 4 + `bench.css`, Vitest + happy-dom.

**Spec:** `docs/superpowers/specs/2026-11-26-look-redesign-design.md` (§8.0, §8.3, §7.1, §10 Market, §11); ADR `docs/decisions/0011-look-redesign.md` §1, §4. Plans 1 and 2: `docs/superpowers/plans/2026-11-26-look-redesign-1-bench-and-company.md`, `-2-emblem-door-founding.md`.

## Global Constraints

- Keep every behaviour, control and R7 plain-language text; C5 and C30 stand (windows grow out of what was clicked).
- R2: the map's marks (campaigns, data centers, sponsorships) come only from `view.me.company`; tops and heights only from public data (`view.companies`, `view.world.market`).
- Same for everybody (spec §8.3, B5): a region's top is its leader's color and its height the leader's share of the region's willing people; the result must not depend on who is looking.
- "A region nobody leads yet is plain glass."
- One 3D engine (§8.0): objects draw only while visible and changing; ambient motion at most 30 frames a second, 60 while dragging; nothing while the tab is hidden; a flat stand-in without WebGL and in unit tests.
- Lighter graphics and reduced motion (§11): a still map (no pulses, heights jump), simpler 3D (no state borders, no bevel).
- The page never scrolls (§7.1); designed at 1440×900, holds from 1280×720; below that modules stack and the bench scrolls inside the frame.
- Downloads: only `us-atlas`, `topojson-client`, `@types/topojson-client` (approved by the owner on 2026-11-26). Attribution goes in the client's licenses note (ADR 0011 §4).
- Verify only in the `check` game (`data-check/`, ports 8788/5175), never on `data/server/`.

## Review Focus

- **No customers anywhere yet** (a new world): every region is plain glass, with no NaN heights or errors. Test: Task 2 ("nobody leads…") and Task 3 ("builds plain glass…").
- **A leader with a very dark color** (any main color founds since plan 2): its region's top still shows in the dark map chamber. Test: Task 2 ("a very dark leader…").
- **The wheel over the map while the bench scrolls** (small windows): the page scrolls; the map zooms only once picked. Test: Task 4 ("zooms only a picked map").
- **A drag or right-drag that ends over a region** opens nothing, and the right button shows no browser menu. Test: Task 4 ("turning the map opens nothing").
- **Two maps at once** (a region window over the Market): both draw, and closing the window frees its 3D objects. Test: Task 3 ("frees everything it made").

---

### Task 1: The map's shapes

**Files:**
- Modify: `apps/client/package.json` (dependencies `us-atlas` ^3.0.1, `topojson-client` ^3.1.0; devDependency `@types/topojson-client` ^3.1.5), `package-lock.json`
- Create: `apps/client/src/map/geometry.ts`
- Create: `apps/client/LICENSES.md`
- Test: `apps/client/test/mapGeometry.test.ts`

**Interfaces:**
- Produces:
  - `MAP_FRAME = { width: 975, height: 610 }` (us-atlas's projected frame, y down).
  - `type Ring = [number, number][]`, `type Polygon = Ring[]` (outer ring first, then holes).
  - `FIPS_TO_STATE: Record<string, string>` (the 51 FIPS ids of `states-albers-10m.json` → postal codes, e.g. `"06" → "CA"`, `"11" → "DC"`).
  - `type RegionShape = { region: RegionId; polygons: Polygon[]; borders: Ring[]; anchor: [number, number] }` — `borders` are the state lines inside the region; `anchor` is where its marks stand: the centroid of its largest mainland polygon (Alaska and Hawaii left out, as `REGION_CENTERS` does).
  - `regionShapes(): RegionShape[]` — nine, in `REGION_IDS` order, built once and cached; rings simplified (`SIMPLIFY = 0.6` frame units) and islands under `MIN_AREA = 6` square units dropped.
  - `simplifyRing(ring: Ring, tolerance: number): Ring` (Douglas–Peucker; a closed ring keeps at least 4 points; tolerance 0 keeps every point).
  - `ringArea(ring: Ring): number` (unsigned).
- Region of a state comes from `STATE_TILES` in `regions.ts` (unchanged).

- [ ] **Step 1: Install the approved packages**

Run: `npm install -w @airace/client us-atlas@^3.0.1 topojson-client@^3.1.0 && npm install -w @airace/client -D @types/topojson-client@^3.1.5`
Expected: added to `apps/client/package.json`; nothing else downloaded.

- [ ] **Step 2: Write the failing tests** in `test/mapGeometry.test.ts`:

```ts
it("places every state of the map in one of the nine regions", () => {
  const ids = (topology.objects.states as GeometryCollection).geometries.map((g) => String(g.id));
  expect(ids).toHaveLength(51);
  for (const id of ids) expect(STATE_TILES.some((t) => t.state === FIPS_TO_STATE[id])).toBe(true);
});
it("gives each region its shapes, Alaska and Hawaii in the Pacific's", () => {
  const shapes = regionShapes();
  expect(shapes.map((s) => s.region)).toEqual([...REGION_IDS]);
  for (const s of shapes) expect(s.polygons.length).toBeGreaterThan(0);
  const pacific = shapes.find((s) => s.region === "pacific");
  // The insets sit in the frame's lower left, under the mainland's west coast.
  const lowerLeft = pacific?.polygons.filter(([outer]) => outer.every(([x, y]) => x < 330 && y > 420));
  expect(lowerLeft?.length).toBeGreaterThanOrEqual(2);
});
it("stands each region's marks inside the region", () => {
  for (const s of regionShapes()) expect(s.polygons.some((p) => inside(s.anchor, p))).toBe(true);
});
it("keeps state lines inside a region, not between regions", () => {
  expect(regionShapes().find((s) => s.region === "southAtlantic")?.borders.length).toBeGreaterThan(0);
});
it("simplifies a straight run to its ends and keeps closed rings closed", () => {
  expect(simplifyRing([[0, 0], [1, 0.01], [2, 0], [3, 0]], 0.1)).toEqual([[0, 0], [3, 0]]);
  const square: Ring = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  expect(simplifyRing(square, 50).length).toBeGreaterThanOrEqual(4);
  expect(simplifyRing(square, 0)).toEqual(square);
});
it("stays light enough to draw on a laptop card", () => {
  const points = regionShapes().flatMap((s) => s.polygons.flat()).reduce((n, r) => n + r.length, 0);
  expect(points).toBeLessThan(14_000);
});
```
(`inside` is a small even-odd point-in-polygon helper in the test file.)

- [ ] **Step 3: Run them**: `cd apps/client && npx vitest run test/mapGeometry.test.ts` → FAIL (no `map/geometry.ts`).

- [ ] **Step 4: Implement `map/geometry.ts`**: import the topology (`us-atlas/states-albers-10m.json`, typed as `Topology` from `topojson-specification`); per region `merge(topology, its geometries)` for the polygons and `mesh(topology, states, (a, b) => a !== b && regionOf(a) === r && regionOf(b) === r)` for the borders; the anchor from `merge` of the region's states without AK and HI. Simplify every ring, drop small islands.

- [ ] **Step 5: Write `apps/client/LICENSES.md`**: three.js (MIT), Barlow Semi Condensed and Doto (OFL-1.1, via @fontsource), us-atlas (ISC; boundaries from the US Census Bureau's cartographic boundary files, public domain), topojson-client (ISC).

- [ ] **Step 6: Run the tests, then `npm run check`** → PASS.

- [ ] **Step 7: Commit** `feat(client): the map's real state shapes, merged into the nine regions`.

---

### Task 2: Who leads each region

**Files:**
- Modify: `apps/client/src/regions.ts` (add `RegionTop`, `RegionMarks`, `leaderTops`, `companyTops`), `apps/client/src/screens/Market.tsx` (drop its private `leaders()`), `apps/client/src/components/RegionMap.tsx` (import `RegionMarks` from `regions.ts`)
- Test: `apps/client/test/leaders.test.ts`

**Interfaces:**
- Produces (in `regions.ts`):
  - `type RegionTop = { color: string | null; height: number; leaderId: string | null }` — `color: null` is plain glass; `height` in 0..1.
  - `type RegionMarks = Partial<Record<RegionId, { campaign?: boolean; dataCenter?: "building" | "open"; sponsor?: string }>>` (moved from `RegionMap.tsx`, unchanged).
  - `leaderTops(view: WorldView): Record<RegionId, RegionTop>` — the leader is the company with the most customers in the region (ties to the lower id); nobody leads where the most is 0 or nobody is willing. Led: `height = 0.25 + 0.75 × share / most` (share = leader's customers ÷ the region's `adoptable`, most = the largest share among led regions), `color = readableOn(leader.palette.primary, "dark")` (the map always sits in a dark display, whatever the viewer's glass). Unled: `{ color: null, height: 0, leaderId: null }`.
  - `companyTops(view: WorldView, companyId: string): Record<RegionId, RegionTop>` — the same spread for one company's own customers (the rival window); regions without its customers are plain glass.

- [ ] **Step 1: Write the failing tests** in `test/leaders.test.ts`:

```ts
it("leaves every region plain glass before anyone has customers", () => {
  const tops = leaderTops(viewOf(runningWorld()));
  for (const r of REGION_IDS) expect(tops[r]).toEqual({ color: null, height: 0, leaderId: null });
});
it("tops each region with its leader, the same for every player", () => {
  const s = ticks(rivalWorld(), 30);
  const tops = leaderTops(viewOf(s, "u1"));
  expect(tops).toEqual(leaderTops(viewOf(s, "u2")));
  for (const r of REGION_IDS) {
    const lead = [...viewOf(s).companies].sort(
      (a, b) => b.customersByRegion[r] - a.customersByRegion[r] || a.id.localeCompare(b.id),
    )[0];
    expect(tops[r]?.leaderId).toBe(lead?.id);
  }
  const heights = REGION_IDS.map((r) => tops[r].height);
  expect(Math.max(...heights)).toBe(1);
  expect(Math.min(...heights)).toBeGreaterThanOrEqual(0.25);
});
it("lifts a very dark leader's color so its top shows in the dark map", () => {
  // a world whose only company with customers founded in #101010
  expect(tops.pacific.color).toBe(readableOn("#101010", "dark"));
});
it("shows one company's own regions for its window", () => {
  // companyTops: regions where it has no customers are plain glass, the rest 0.25..1
});
```
(`ticks`, `rivalWorld`, `runningWorld`, `viewOf` exist in the test helpers; a second member `u2` joins with `memberJoined` as in `worldchat.test.tsx`.)

- [ ] **Step 2: Run** `cd apps/client && npx vitest run test/leaders.test.ts` → FAIL.

- [ ] **Step 3: Implement** `leaderTops`, `companyTops`, move `RegionMarks`; Market's map reads `leaderTops` (its private `leaders()` goes).

- [ ] **Step 4: Run every client test, then `npm run check`** → PASS.

- [ ] **Step 5: Commit** `feat(client): who leads each region, the same for every player`.

---

### Task 3: The 3D map

**Files:**
- Modify: `apps/client/src/three/engine.ts` (`pointer` events carry `button?: number`), `apps/client/src/three/View3D.tsx` (passes `e.button` on down)
- Create: `apps/client/src/map/mapStage.ts`
- Test: `apps/client/test/mapStage.test.ts`, `apps/client/test/view3d.test.tsx` (new: the button reaches the stage; `vi.mock` of `../src/three/support.ts` with `webglAvailable: () => true` and of `../src/three/engine.ts` with `attachView` returning `{ markDirty() {}, setVisible() {}, detach() {} }` and a no-op `markInteracting`, in a file of its own because `vi.mock` holds for the whole file)

**Interfaces:**
- Consumes: `regionShapes()`, `MAP_FRAME` (Task 1); `RegionTop`, `RegionMarks` (Task 2); `Stage3D`, shared materials (plan 1).
- Produces (in `map/mapStage.ts`):
  - `type MapLook = { tops: Record<RegionId, RegionTop>; marks: RegionMarks; focus: RegionId | null; markColor: string }`
  - `MAP_VIEW = { yaw: 0, tilt: 0.95, zoom: 1 }` (radians; the same opening view for everybody) and `MAP_LIMITS = { yaw: 0.9, tiltMin: 0.5, tiltMax: 1.35, zoomMin: 0.75, zoomMax: 1.8 }`.
  - `type MapStage = Stage3D & { setLook(look: MapLook): void; setHover(region: RegionId | null): void; pick(x: number, y: number): RegionId | null; zoom(by: -1 | 1): void; resetView(): void; view(): { yaw: number; tilt: number; zoom: number }; anchor(region: RegionId): THREE.Vector3 }` — `pick` takes normalized device coordinates (−1..1, y up); `anchor` is the world point on top of the region where its marks stand.
  - `makeMapStage(look: MapLook, o: { still: boolean; lighter: boolean }): MapStage`
- The model (the builder's call within this): the frame scaled to 2 world units wide on a dark metal plinth; per region a group `region:<id>` with a glass prism (extruded outline, height `0.04 + 0.32 × top.height`) and, when `top.color` is set, a lit top `top:<id>` in that color with the state lines `borders:<id>` on it (left out with `lighter`); marks `dc:<id>` (open: a lit building; building: a lower block in scaffolding), `campaign:<id>` (a flat ring in `markColor`, pulsing unless still), `sponsor:<id>` (a pin with a flag showing the mark). Heights ease to new values in about 0.6 s unless still. Hover brightens the region's top; `focus` dims the other regions. A left drag turns (yaw), a right drag tilts; both stay within `MAP_LIMITS`.

- [ ] **Step 1: Write the failing tests** in `test/mapStage.test.ts`:

```ts
it("builds plain glass for a map nobody leads yet", () => {
  const stage = makeMapStage(look(PLAIN), { still: true, lighter: false });
  for (const r of REGION_IDS) {
    expect(stage.scene.getObjectByName(`region:${r}`)).toBeDefined();
    expect(stage.scene.getObjectByName(`top:${r}`)).toBeUndefined();
  }
  stage.scene.traverse((o) => expect(Number.isFinite(o.position.y)).toBe(true));
});
it("raises a region by its leader's share and lights its top in the leader's color", () => {
  const stage = makeMapStage(look({ ...PLAIN, pacific: { color: "#ff9f1c", height: 1, leaderId: "c1" } }), STILL);
  const top = stage.scene.getObjectByName("top:pacific") as THREE.Mesh;
  expect((top.material as THREE.MeshStandardMaterial).emissive.getHexString()).toBe("ff9f1c");
  expect(stage.anchor("pacific").y).toBeGreaterThan(stage.anchor("plains").y);
});
it("finds the region under the pointer", () => {
  const stage = makeMapStage(look(PLAIN), STILL);
  for (const r of ["pacific", "newEngland", "southCentral"] as const) {
    const p = stage.anchor(r).clone().project(stage.camera);
    expect(stage.pick(p.x, p.y)).toBe(r);
  }
  expect(stage.pick(0.99, 0.99)).toBeNull();
});
it("stands the player's marks on their regions", () => {
  const marks: RegionMarks = {
    pacific: { campaign: true, dataCenter: "open" },
    plains: { dataCenter: "building", sponsor: "🎪" },
  };
  const stage = makeMapStage({ ...look(PLAIN), marks }, STILL);
  for (const name of ["dc:pacific", "campaign:pacific", "dc:plains", "sponsor:plains"])
    expect(stage.scene.getObjectByName(name), name).toBeDefined();
  expect(stage.scene.getObjectByName("campaign:plains")).toBeUndefined();
  expect(stage.scene.getObjectByName("dc:mountain")).toBeUndefined();
});
it("turns, tilts and zooms within limits, and resets to the shared view", () => {
  const stage = makeMapStage(look(PLAIN), STILL);
  stage.pointer?.({ type: "down", dx: 0, dy: 0, button: 0 });
  stage.pointer?.({ type: "move", dx: 10_000, dy: 0 });
  stage.pointer?.({ type: "up", dx: 0, dy: 0 });
  expect(stage.view().yaw).toBe(MAP_LIMITS.yaw);
  stage.pointer?.({ type: "down", dx: 0, dy: 0, button: 2 });
  stage.pointer?.({ type: "move", dx: 0, dy: -10_000 });
  stage.pointer?.({ type: "up", dx: 0, dy: 0 });
  expect(stage.view().tilt).toBe(MAP_LIMITS.tiltMax);
  expect(stage.view().yaw).toBe(MAP_LIMITS.yaw);
  for (let i = 0; i < 40; i++) stage.zoom(1);
  expect(stage.view().zoom).toBe(MAP_LIMITS.zoomMax);
  stage.resetView();
  expect(stage.view()).toEqual(MAP_VIEW);
});
it("keeps still with lighter graphics, and pulses a campaign otherwise", () => {
  const marks: RegionMarks = { pacific: { campaign: true } };
  expect(makeMapStage({ ...look(PLAIN), marks }, { still: true, lighter: true }).update(1, 0.03)).toBe(false);
  expect(makeMapStage({ ...look(PLAIN), marks }, { still: false, lighter: false }).update(1, 0.03)).toBe(true);
  expect(makeMapStage(look(PLAIN), { still: false, lighter: false }).update(5, 0.03)).toBe(false);
});
it("frees everything it made", () => {
  const geometries = vi.spyOn(THREE.BufferGeometry.prototype, "dispose");
  const materials = vi.spyOn(THREE.Material.prototype, "dispose");
  const stage = makeMapStage({ ...look(LED), marks: { plains: { sponsor: "🎪" } } }, STILL);
  stage.dispose();
  expect(geometries.mock.calls.length).toBeGreaterThanOrEqual(REGION_IDS.length);
  expect(materials).toHaveBeenCalled();
});
```
(`PLAIN` is every region `{ color: null, height: 0, leaderId: null }`, `LED` every region lit, `look(tops)` a `MapLook` with no marks, no focus and `markColor: "#4f8cff"`, `STILL = { still: true, lighter: false }`.)
And in `view3d.test.tsx`: `it("passes which button started a drag", …)` — render `<View3D make={() => stage} label="Map" fallback={null} />` with a stage whose `pointer` is `vi.fn()`; `fireEvent.pointerDown(canvas, { button: 2, clientX: 5, clientY: 5 })`; expect `stage.pointer` called with `{ type: "down", dx: 0, dy: 0, button: 2 }`.

- [ ] **Step 2: Run** `cd apps/client && npx vitest run test/mapStage.test.ts test/three.test.tsx` → FAIL.

- [ ] **Step 3: Implement** the engine's `button`, View3D's, and `map/mapStage.ts`.

- [ ] **Step 4: Run every client test, then `npm run check`** → PASS.

- [ ] **Step 5: Commit** `feat(client): the US map in 3D: glass regions, lit leaders, marks, a turnable table`.

---

### Task 4: The map on the benches

**Files:**
- Create: `apps/client/src/map/TileMap.tsx` (today's SVG tile map, moved out of `RegionMap` as the flat stand-in), `apps/client/src/bench/wheel.ts`, `apps/client/src/components/RegionReadout.tsx`
- Modify: `apps/client/src/components/RegionMap.tsx`, `apps/client/src/bench/Knob.tsx` (uses `wheel.ts`), `apps/client/src/screens/Company.tsx`, `apps/client/src/screens/Market.tsx`, `apps/client/src/windows/RivalWindow.tsx`, `apps/client/src/windows/RegionWindow.tsx`, `apps/client/src/bench/bench.css`
- Test: `apps/client/test/regions.test.tsx`, `apps/client/test/rivalsEverywhere.test.tsx`, `apps/client/test/campaigns.test.tsx`, `apps/client/test/bench.test.tsx`, `apps/client/test/regionMap3d.test.tsx` (new; the 3D path with the stand-ins of `view3d.test.tsx` plus `vi.mock("../src/map/mapStage.ts")` returning a stage whose `pick` answers `"pacific"` and whose `zoom`, `setHover`, `resetView` are `vi.fn()`), `tests/e2e/solo.spec.ts` (locators where the map's name changed)

**Interfaces:**
- Consumes: `makeMapStage`, `MapStage`, `MapLook` (Task 3); `leaderTops`, `companyTops`, `RegionTop`, `RegionMarks` (Task 2).
- Produces:
  - `RegionMap({ label, tops, marks = {}, focus = null, onOpen, readout, small = false, markColor }: { label: string; tops: Record<RegionId, RegionTop>; marks?: RegionMarks; focus?: RegionId | null; onOpen?: (region: RegionId, el: Element) => void; readout?: (region: RegionId) => ReactNode; small?: boolean; markColor: string })` — a `fieldset` named `label`; the map in a dark chamber display (3D through `View3D`, `TileMap` without WebGL); the hover readout (at the pointer in 3D, over the region in the tile map); a "Reset view" push button (3D only); the nine region keys, each a real button named `Open <region name>` (hovering or focusing a key lights its region).
  - `createWheel(): (e: WheelEvent) => -1 | 0 | 1` in `bench/wheel.ts` — one notch per mouse-wheel event, a trackpad's small steps (|deltaY| < 50, pixel mode) summed to 50 per notch; `+1` is toward the user (zoom in, value up). Knob uses it (its tests unchanged).
- In 3D: a pointer move picks the region under it (hover); a click that did not end a drag picks and opens it (`onOpen(region, the map's view element)`); the right button tilts and its context menu is suppressed; the wheel zooms only while the map's view holds focus (`tabIndex={-1}`, so a click picks it without adding a tab stop), and then stops the page's scroll.
- Where the map goes: Company (small, `label="Who leads each region"`, `leaderTops`), Market (large, `label="Who leads each region"`, `leaderTops`; the "Color the map by" switch goes: the map always shows the leaders), RivalWindow (`label={`Where ${rival.name}'s customers are`}`, `companyTops`), RegionWindow (small, `leaderTops`, `focus={region}`, `label={`${REGION_NAMES[region]} on the map`}`). `markColor` is the player's lamp color (`lampColor(pub.palette)`).
- Readouts (spec §8.3): `RegionReadout({ view, region })` in `components/RegionReadout.tsx`, shared by the Market, Company and region-window maps: the region's name, its leader ("Led by <name>" in its map color, or "Nobody leads it yet"), your customers there and your share of its willing people. The rival window's readout: the rival's customers there.

- [ ] **Step 1: Write the failing tests** (flat stand-in, as unit tests run without WebGL):

```ts
// regions.test.tsx
it("draws each region in its top's color and height, plain glass where nobody leads", () => {
  const tops = { ...PLAIN, pacific: { color: "#ff9f1c", height: 1, leaderId: "c1" } };
  const { container } = render(<RegionMap label="Who leads each region" tops={tops} markColor="#ff9f1c" />);
  expect(container.querySelector('[data-state="CA"]')?.getAttribute("fill")).toBe("#ff9f1c");
  expect(container.querySelector('[data-state="NY"]')?.getAttribute("data-plain")).toBe("true");
  expect(screen.getAllByRole("button", { name: /^Open / })).toHaveLength(9);
});
it("lights a region and shows its readout while its key has focus", async () => {
  render(<RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" readout={(r) => <p>{`readout ${r}`}</p>} />);
  screen.getByRole("button", { name: "Open Pacific" }).focus();
  expect(await screen.findByText("readout pacific")).toBeTruthy();
});
it("says who leads a region and what you have there", () => {
  const view = viewOf(launchedWorld());
  render(<RegionReadout view={view} region="pacific" />);
  const lead = view.companies.find((c) => c.id === leaderTops(view).pacific.leaderId);
  expect(screen.getByText(lead ? `Led by ${lead.name}` : "Nobody leads it yet")).toBeTruthy();
  expect(screen.getByText(formatCount(regionCustomers(view, "pacific")))).toBeTruthy();
});
// rivalsEverywhere.test.tsx: the Market's map shows the leaders without a switch:
//   no radio named "Leaders"; each region's first tile is filled with readableOn(leader's primary, "dark")
// bench.test.tsx
it("counts one notch per mouse-wheel event and sums a trackpad's small steps", () => {
  const notch = createWheel();
  expect(notch(new WheelEvent("wheel", { deltaY: -100 }))).toBe(1);
  expect(notch(new WheelEvent("wheel", { deltaY: 20 }))).toBe(0);
  expect(notch(new WheelEvent("wheel", { deltaY: 40 }))).toBe(-1);
});
```
And the map's interaction rules, tested on `RegionMap` with a stand-in stage injected through `View3D` (a test-only `makeStage` prop is not added; instead stub `webglAvailable` to true and `HTMLCanvasElement.prototype.getContext` as `three.test.tsx` does, and mock `makeMapStage` with `vi.mock("../src/map/mapStage.ts")` returning a stage whose `pick` returns `"pacific"`):

```ts
it("zooms only a picked map, and then keeps the page still", () => {
  render(<RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" />);
  const view = screen.getByTestId("map-view");
  const wheel = () => {
    const e = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
    view.dispatchEvent(e);
    return e;
  };
  expect(wheel().defaultPrevented).toBe(false);
  expect(stage.zoom).not.toHaveBeenCalled();
  view.focus();
  expect(wheel().defaultPrevented).toBe(true);
  expect(stage.zoom).toHaveBeenCalledWith(1);
});
it("turning the map opens nothing, and the right button shows no menu", () => {
  const onOpen = vi.fn();
  render(<RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" onOpen={onOpen} />);
  const canvas = screen.getByRole("img", { name: "Who leads each region" });
  fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10 });
  fireEvent.pointerMove(canvas, { clientX: 60, clientY: 10 });
  fireEvent.pointerUp(canvas, { clientX: 60, clientY: 10 });
  fireEvent.click(canvas, { clientX: 60, clientY: 10 });
  expect(onOpen).not.toHaveBeenCalled();
  expect(fireEvent.contextMenu(canvas)).toBe(false); // default prevented
});
it("opens the region under a plain click", () => {
  const onOpen = vi.fn();
  render(<RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" onOpen={onOpen} />);
  fireEvent.click(screen.getByRole("img", { name: "Who leads each region" }), { clientX: 30, clientY: 30 });
  expect(onOpen).toHaveBeenCalledWith("pacific", screen.getByTestId("map-view"));
});
```
(`data-testid="map-view"` is on the element that holds the canvas, takes focus and receives the wheel.)

- [ ] **Step 2: Run them** → FAIL.

- [ ] **Step 3: Implement** `wheel.ts` (and Knob on it), `TileMap.tsx`, the new `RegionMap`, the four callers, and the chamber styles in `bench.css` (the readout as a small display, the reset key in the chamber's corner, the keys as a row of keycaps).

- [ ] **Step 4: Update the locators where the map's name changed**: `"Your customers by region"` → `"Who leads each region"` (`regions.test.tsx`, `campaigns.test.tsx`, `tests/e2e/solo.spec.ts`), `"Customers by region"` → `"Who leads each region"` (Company in `regions.test.tsx`).

- [ ] **Step 5: Run every test, `npm run check`; look at the map in the `check` game** (Market, Company, a region window over the Market, a rival window; dark and light glass; turn, tilt, zoom, reset, hover, click) → PASS.

- [ ] **Step 6: Commit** `feat(client): the 3D map on the Market, Company, rival and region benches`.

---

### Task 5: The Market station as a bench

**Files:**
- Modify: `apps/client/src/screens/Market.tsx`, `apps/client/src/spending/Spending.tsx` (`SponsorPanel` as a module), `apps/client/src/bench/bench.css`
- Test: `apps/client/test/market.test.tsx`, `apps/client/test/campaigns.test.tsx`

**Interfaces:**
- Consumes: `RegionMap` (Task 4); bench parts of plan 1 (`Module`, `Display`, `Readout`, `PushButton`, `Lamp`).
- The bench (spec §10 Market), at 1440×900 without scrolling:
  - a strip of three readout modules: "People willing to use an AI" (the number and "N% of everyone, and growing"), "Your customers" (the number and the growth trace; opens the customers window, button "Open customers" as today), "Your share" ("of everyone willing");
  - the map module (heavy) with the large map and the line "Height shows how many of a region's willing people use its leader's AI.";
  - beside it, "Ad campaigns" (the running campaigns in a display that scrolls, each with its days left, spend, progress and its "Stop the … campaign" key; the push button "New ad campaign") and "Sponsorships" (the running ones in a display; "Sponsor an event");
  - a row of the six groups, each a module named by its group (a `region`, as today) with its count, the mood face and percentage, its crowd as a dot-matrix display, its spotlight customer's name and bio, and the "Open <group>" button; above the row, the line "Each dot is N people".
- The crowd as a dot-matrix display: `crowdDots(people: number, step: number): { lit: number; total: number }` — `total = CROWD_MOST` (120, a 20 × 6 grid), `lit` as today's dot count; lit dots carry `data-dot` (as the test counts), dark ones `data-off`.

- [ ] **Step 1: Write the failing test** in `market.test.tsx`:

```ts
it("lights one dot of a crowd display per step of people, the rest of its 120 stay dark", () => {
  const view = viewOf(launchedWorld());
  render(<Market view={view} />);
  const step = crowdStep(Math.max(...GROUP_IDS.map((g) => groupCustomers(view, g))));
  const card = screen.getByRole("region", { name: GROUPS.students.many });
  const lit = card.querySelectorAll("[data-dot]").length;
  expect(lit + card.querySelectorAll("[data-off]").length).toBe(120);
});
```
The existing Market and campaign tests stay as they are (their controls keep their names).

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** the bench layout (a grid in `Market.tsx` and `bench.css`; the six groups stack into three columns below 1280 wide).

- [ ] **Step 4: Run every test, `npm run check`; look at the Market in the `check` game at 1440×900 and 1280×720, dark and light glass** → PASS.

- [ ] **Step 5: Commit** `feat(client): the Market station as a bench around the 3D map`.

---

### Task 6: The plan in the running game

- [ ] **Step 1:** In the `check` game: a world with rivals after a few game weeks (leaders in several colors, heights differ), a fresh world (plain glass everywhere), a data center being built and one open, a campaign and a sponsorship on the map; turn, tilt, zoom and reset; hover and click every region; the Company map; a rival window and a region window over the Market; dark and light glass; lighter graphics on.
- [ ] **Step 2:** `npm test`, `npm run check`, `npm run e2e` → all pass (fix locators only where a control moved).
- [ ] **Step 3:** Progress entry "Look redesign, plan 3"; next-steps note updated; the ledger copied to `docs/handoff/sdd/ledger-look-redesign-3.md`; commit and push.
- [ ] **Step 4:** A fresh whole-plan review; fix Critical and Important findings with tests; ledger the minors.
- [ ] **Step 5:** Show the owner the Market and the map.
