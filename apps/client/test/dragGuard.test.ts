import { describe, expect, it } from "vitest";
import { createDragGuard } from "../src/three/dragGuard.ts";

describe("turning a 3D object by dragging", () => {
  it("lets a plain click through", () => {
    const guard = createDragGuard();
    guard.down(100, 100);
    guard.move(102, 101);
    guard.up();
    expect(guard.takeClick()).toBe(false);
  });

  it("swallows the one click that ends a drag", () => {
    const guard = createDragGuard();
    guard.down(100, 100);
    guard.move(155, 100);
    guard.up();
    expect(guard.takeClick()).toBe(true);
    expect(guard.takeClick()).toBe(false);
  });

  it("starts fresh with each press", () => {
    const guard = createDragGuard();
    guard.down(0, 0);
    guard.move(40, 0);
    guard.up();
    guard.down(0, 0);
    guard.up();
    expect(guard.takeClick()).toBe(false);
  });
});
