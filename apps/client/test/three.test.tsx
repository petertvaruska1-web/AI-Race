import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { dueViews, type ViewState } from "../src/three/engine.ts";
import { gearRing } from "../src/three/gear.ts";
import { webglAvailable } from "../src/three/support.ts";
import { View3D } from "../src/three/View3D.tsx";

const v = (id: number, o: Partial<ViewState>): ViewState => ({
  id,
  visible: true,
  dirty: false,
  moving: false,
  last: 0,
  ...o,
});

describe("the 3D engine", () => {
  it("has no WebGL under the test DOM and shows the stand-in", () => {
    expect(webglAvailable()).toBe(false);
    render(
      <View3D
        make={() => {
          throw new Error("not called");
        }}
        label="Your AI"
        fallback={<p>flat</p>}
      />,
    );
    expect(screen.getByText("flat")).toBeTruthy();
  });

  it("draws only visible views that changed or move on time", () => {
    const live = { fps: 30, still: false };
    expect(
      dueViews(
        [v(1, { dirty: true }), v(2, { moving: true }), v(3, { visible: false, dirty: true })],
        40,
        live,
      ),
    ).toEqual([1, 2]);
    expect(dueViews([v(2, { moving: true, last: 30 })], 40, live)).toEqual([]);
    expect(
      dueViews([v(1, { dirty: true }), v(2, { moving: true })], 40, { fps: 0, still: true }),
    ).toEqual([1]);
    expect(
      dueViews([v(1, { dirty: true }), v(2, { moving: true })], 40, { fps: 0, still: false }),
    ).toEqual([]);
  });

  it("builds a toothed gear ring", () => {
    const few = gearRing({ radius: 1, width: 0.08, depth: 0.05, teeth: 12, toothDepth: 0.03 });
    const many = gearRing({ radius: 1, width: 0.08, depth: 0.05, teeth: 48, toothDepth: 0.03 });
    expect(many.getAttribute("position").count).toBeGreaterThan(few.getAttribute("position").count);
  });
});
