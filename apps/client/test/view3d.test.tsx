import { fireEvent, render, screen } from "@testing-library/react";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { Stage3D } from "../src/three/engine.ts";
import { View3D } from "../src/three/View3D.tsx";

// A working 3D view without a screen: WebGL "available", the shared engine stood in.
vi.mock("../src/three/support.ts", () => ({ webglAvailable: () => true }));
vi.mock("../src/three/engine.ts", () => ({
  attachView: () => ({ markDirty() {}, setVisible() {}, detach() {} }),
  markInteracting() {},
}));

describe("a 3D view", () => {
  it("passes which button started a drag", () => {
    const stage: Stage3D = {
      scene: new THREE.Scene(),
      camera: new THREE.PerspectiveCamera(),
      update: () => false,
      pointer: vi.fn(),
      dispose() {},
    };
    render(<View3D make={() => stage} label="Map" fallback={null} />);
    fireEvent.pointerDown(screen.getByRole("img", { name: "Map" }), {
      button: 2,
      clientX: 5,
      clientY: 5,
    });
    expect(stage.pointer).toHaveBeenCalledWith({ type: "down", dx: 0, dy: 0, button: 2 });
  });
});
