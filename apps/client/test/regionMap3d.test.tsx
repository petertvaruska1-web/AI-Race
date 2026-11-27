import { REGION_IDS, type RegionId } from "@airace/protocol";
import { act, fireEvent, render, screen } from "@testing-library/react";
import * as THREE from "three";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RegionMap } from "../src/components/RegionMap.tsx";
import type { RegionTop } from "../src/regions.ts";
import { useGraphics } from "../src/store/graphics.ts";

// The map's 3D path without a screen: WebGL "available", the shared engine and the map's stage
// stood in (the stage's own behaviour is tested in mapStage.test.ts).
const stage = vi.hoisted(() => ({
  pick: vi.fn((): string | null => "pacific"),
  zoom: vi.fn(),
  setHover: vi.fn(),
  setLook: vi.fn(),
  resetView: vi.fn(),
  pointer: vi.fn(),
  setMode: vi.fn(),
}));
vi.mock("../src/three/support.ts", () => ({ webglAvailable: () => true }));
vi.mock("../src/three/engine.ts", () => ({
  attachView: () => ({ markDirty() {}, setVisible() {}, detach() {} }),
  markInteracting() {},
}));
vi.mock("../src/map/mapStage.ts", () => ({
  MAP_VIEW: { yaw: 0, tilt: 0.95, zoom: 1 },
  makeMapStage: () => ({
    ...stage,
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(),
    update: () => false,
    view: () => ({ yaw: 0, tilt: 0.95, zoom: 1 }),
    anchor: () => new THREE.Vector3(),
    dispose() {},
  }),
}));

const PLAIN = Object.fromEntries(
  REGION_IDS.map((r) => [r, { color: null, height: 0, leaderId: null }]),
) as Record<RegionId, RegionTop>;

beforeEach(() => {
  for (const f of Object.values(stage)) f.mockClear();
  useGraphics.setState({ lighter: false, auto: false });
});

const rect = (left: number, top: number, width: number, height: number) =>
  ({
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    x: left,
    y: top,
    toJSON() {},
  }) as DOMRect;

describe("the 3D map on a bench", () => {
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
    render(
      <RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" onOpen={onOpen} />,
    );
    const canvas = screen.getByRole("img", { name: "Who leads each region" });
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10 });
    fireEvent.pointerMove(canvas, { clientX: 60, clientY: 10 });
    fireEvent.pointerUp(canvas, { clientX: 60, clientY: 10 });
    fireEvent.click(canvas, { clientX: 60, clientY: 10 });
    expect(onOpen).not.toHaveBeenCalled();
    expect(fireEvent.contextMenu(canvas)).toBe(false);
  });

  it("opens the region under a plain click", () => {
    const onOpen = vi.fn();
    render(
      <RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" onOpen={onOpen} />,
    );
    fireEvent.click(screen.getByRole("img", { name: "Who leads each region" }), {
      clientX: 30,
      clientY: 30,
    });
    expect(onOpen).toHaveBeenCalledWith("pacific", screen.getByTestId("map-view"));
  });

  it("points at the region under the pointer on the drawn map, not on its frame", () => {
    const onOpen = vi.fn();
    render(
      <RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" onOpen={onOpen} />,
    );
    const canvas = screen.getByRole("img", { name: "Who leads each region" });
    vi.spyOn(screen.getByTestId("map-view"), "getBoundingClientRect").mockReturnValue(
      rect(0, 0, 224, 120),
    );
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue(rect(12, 10, 200, 100));
    fireEvent.click(canvas, { clientX: 12, clientY: 60 });
    const [x, y] = stage.pick.mock.calls.at(-1) as unknown as [number, number];
    expect(x).toBeCloseTo(-1, 5);
    expect(y).toBeCloseTo(0, 5);
  });

  it("passes lighter graphics switched on while the map shows to the 3D map", () => {
    render(<RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" />);
    act(() => useGraphics.getState().setLighter(true, true));
    expect(stage.setMode).toHaveBeenLastCalledWith({ still: true, lighter: true });
    act(() => useGraphics.getState().setLighter(false, true));
    expect(stage.setMode).toHaveBeenLastCalledWith({ still: false, lighter: false });
  });

  it("puts the map back to the shared view from its reset key", () => {
    render(<RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" />);
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    expect(stage.resetView).toHaveBeenCalled();
  });
});
