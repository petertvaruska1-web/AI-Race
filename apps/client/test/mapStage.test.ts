import { REGION_IDS, type RegionId } from "@airace/protocol";
import * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MAP_LIMITS, MAP_VIEW, type MapLook, makeMapStage } from "../src/map/mapStage.ts";
import type { RegionMarks, RegionTop } from "../src/regions.ts";

const tops = (top: RegionTop) =>
  Object.fromEntries(REGION_IDS.map((r) => [r, { ...top }])) as Record<RegionId, RegionTop>;
const PLAIN = tops({ color: null, height: 0, leaderId: null });
const LED = tops({ color: "#4f8cff", height: 0.5, leaderId: "c1" });
const look = (t: Record<RegionId, RegionTop>): MapLook => ({
  tops: t,
  marks: {},
  focus: null,
  markColor: "#4f8cff",
});
const STILL = { still: true, lighter: false };

afterEach(() => vi.restoreAllMocks());

describe("the 3D map (design §8.3)", () => {
  it("builds plain glass for a map nobody leads yet", () => {
    const stage = makeMapStage(look(PLAIN), STILL);
    for (const r of REGION_IDS) {
      expect(stage.scene.getObjectByName(`region:${r}`), r).toBeDefined();
      expect(stage.scene.getObjectByName(`top:${r}`), r).toBeUndefined();
    }
    stage.scene.traverse((o) => expect(Number.isFinite(o.position.y)).toBe(true));
  });

  it("raises a region by its leader's share and lights its top in the leader's color", () => {
    const stage = makeMapStage(
      look({ ...PLAIN, pacific: { color: "#ff9f1c", height: 1, leaderId: "c1" } }),
      STILL,
    );
    const top = stage.scene.getObjectByName("top:pacific") as THREE.Mesh;
    expect((top.material as THREE.MeshStandardMaterial).emissive.getHexString()).toBe("ff9f1c");
    expect(stage.anchor("pacific").y).toBeGreaterThan(stage.anchor("plains").y);
  });

  it("finds the region under the pointer", () => {
    const stage = makeMapStage(look(PLAIN), STILL);
    for (const r of ["pacific", "newEngland", "southCentral"] as const) {
      const p = stage.anchor(r).clone().project(stage.camera);
      expect(stage.pick(p.x, p.y), r).toBe(r);
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
    expect(
      makeMapStage({ ...look(PLAIN), marks }, { still: true, lighter: true }).update(1, 0.03),
    ).toBe(false);
    expect(
      makeMapStage({ ...look(PLAIN), marks }, { still: false, lighter: false }).update(1, 0.03),
    ).toBe(true);
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
});
