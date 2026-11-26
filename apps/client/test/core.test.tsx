import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Core } from "../src/core/Core.tsx";
import { makeCoreStage } from "../src/core/coreStage.ts";
import { orbParams } from "../src/orb/params.ts";
import { NONE, SCORED } from "./orbInputs.ts";

describe("the AI core in 3D", () => {
  it("falls back to the drawn orb without WebGL", () => {
    render(<Core input={SCORED} label="Your AI v1" />);
    expect(screen.getByRole("img", { name: "Your AI v1" }).tagName).toBe("CANVAS");
  });

  it("builds one gear ring per measured ring and a crystal", () => {
    const stage = makeCoreStage(orbParams(SCORED), { still: true });
    expect(stage.scene.getObjectByName("crystal")).toBeDefined();
    const rings = Math.ceil(orbParams(SCORED).rings);
    expect(rings).toBeGreaterThan(0);
    for (let i = 0; i < rings; i++) expect(stage.scene.getObjectByName(`ring-${i}`)).toBeDefined();
    expect(stage.scene.getObjectByName(`ring-${rings}`)).toBeUndefined();
  });

  it("is a bare crystal before the first AI", () => {
    const stage = makeCoreStage(orbParams(NONE), { still: true });
    expect(stage.scene.getObjectByName("crystal")).toBeDefined();
    expect(stage.scene.getObjectByName("ring-0")).toBeUndefined();
    expect(stage.update(0, 0.016)).toBe(false);
  });

  it("changes its rings in place when the measurements change", () => {
    const stage = makeCoreStage(orbParams(NONE), { still: true });
    stage.setParams(orbParams(SCORED));
    expect(stage.scene.getObjectByName("ring-0")).toBeDefined();
  });
});
