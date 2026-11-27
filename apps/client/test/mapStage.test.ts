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
    stage.pointer?.({ type: "move", dx: -10_000, dy: 0 });
    stage.pointer?.({ type: "up", dx: 0, dy: 0 });
    expect(stage.view().yaw).toBe(MAP_LIMITS.yaw);
    stage.pointer?.({ type: "down", dx: 0, dy: 0, button: 2 });
    stage.pointer?.({ type: "move", dx: 0, dy: 10_000 });
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
    const marks: RegionMarks = {
      pacific: { campaign: true, dataCenter: "open" },
      plains: { dataCenter: "building", sponsor: "🎪" },
    };
    const stage = makeMapStage({ ...look(LED), marks }, { still: false, lighter: false });
    const made = new Set<{ dispose(): void }>();
    stage.scene.traverse((o) => {
      const thing = o as THREE.Object3D & {
        geometry?: THREE.BufferGeometry;
        material?: THREE.Material | THREE.Material[];
      };
      if (thing.geometry) made.add(thing.geometry);
      for (const m of [thing.material ?? []].flat()) {
        made.add(m);
        const map = (m as THREE.MeshBasicMaterial).map;
        if (map) made.add(map);
      }
    });
    // Three.js's one shared sprite quad is not the map's to free.
    const spies = [...made]
      .filter((thing) => !(thing instanceof THREE.BufferGeometry && isSpriteQuad(stage, thing)))
      .map((thing) => vi.spyOn(thing, "dispose"));
    expect(spies.length).toBeGreaterThan(REGION_IDS.length * 3);
    stage.dispose();
    for (const spy of spies) expect(spy).toHaveBeenCalled();
  });

  it("frames the whole map, plinth and all, at every turn, tilt and canvas shape", () => {
    const stage = makeMapStage(look(tops({ color: "#4f8cff", height: 1, leaderId: "c1" })), STILL);
    const box = new THREE.Box3().setFromObject(stage.scene);
    const corners = [box.min.x, box.max.x].flatMap((x) =>
      [box.min.y, box.max.y].flatMap((y) =>
        [box.min.z, box.max.z].map((z) => new THREE.Vector3(x, y, z)),
      ),
    );
    const drag = (button: number, dx: number, dy: number) => {
      stage.pointer?.({ type: "down", dx: 0, dy: 0, button });
      stage.pointer?.({ type: "move", dx, dy });
      stage.pointer?.({ type: "up", dx: 0, dy: 0 });
    };
    for (const aspect of [0.8, 1.6, 2.9, 6.5])
      for (const turn of [-10_000, 0, 10_000])
        for (const tip of [-10_000, 0, 10_000]) {
          stage.resetView();
          drag(0, turn, 0);
          drag(2, 0, tip);
          stage.camera.aspect = aspect;
          stage.camera.updateProjectionMatrix();
          stage.update(0, 0);
          const seen = corners.map((c) => c.clone().project(stage.camera));
          const where = `aspect ${aspect}, view ${JSON.stringify(stage.view())}`;
          for (const p of seen) {
            expect(Math.abs(p.x), where).toBeLessThanOrEqual(1);
            expect(Math.abs(p.y), where).toBeLessThanOrEqual(1);
          }
          // ...and fills the frame on its tighter side rather than floating small in it.
          expect(
            Math.max(...seen.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)))),
            where,
          ).toBeGreaterThan(0.85);
        }
  });

  it("turns the map the way the pointer drags, and tips it toward overhead on a drag down", () => {
    const stage = makeMapStage(look(PLAIN), STILL);
    stage.camera.aspect = 2;
    stage.camera.updateProjectionMatrix();
    stage.update(0, 0);
    const box = new THREE.Box3().setFromObject(stage.scene);
    const near = new THREE.Vector3(0, 0, box.max.z);
    const before = near.clone().project(stage.camera).x;
    stage.pointer?.({ type: "down", dx: 0, dy: 0, button: 0 });
    stage.pointer?.({ type: "move", dx: 40, dy: 0 });
    stage.pointer?.({ type: "up", dx: 0, dy: 0 });
    stage.update(0, 0);
    expect(near.clone().project(stage.camera).x).toBeGreaterThan(before);
    const tilt = stage.view().tilt;
    stage.pointer?.({ type: "down", dx: 0, dy: 0, button: 2 });
    stage.pointer?.({ type: "move", dx: 0, dy: 40 });
    stage.pointer?.({ type: "up", dx: 0, dy: 0 });
    expect(stage.view().tilt).toBeGreaterThan(tilt);
  });

  it("eases a region to its new height, and jumps there when still", () => {
    const moving = makeMapStage(look(PLAIN), { still: false, lighter: false });
    const low = moving.anchor("pacific").y;
    moving.setLook(look({ ...PLAIN, pacific: { color: "#ff9f1c", height: 1, leaderId: "c1" } }));
    expect(moving.update(1, 0.3)).toBe(true);
    const half = moving.anchor("pacific").y;
    expect(half).toBeGreaterThan(low);
    moving.update(1.3, 0.3);
    moving.update(1.6, 0.3);
    const high = moving.anchor("pacific").y;
    expect(high).toBeGreaterThan(half);
    expect(moving.update(1.9, 0.3)).toBe(false);

    const still = makeMapStage(look(PLAIN), STILL);
    still.setLook(look({ ...PLAIN, pacific: { color: "#ff9f1c", height: 1, leaderId: "c1" } }));
    still.update(1, 0.03);
    expect(still.anchor("pacific").y).toBeCloseTo(high, 6);
  });

  it("takes away marks that are gone, and keeps the ones that stay as they are", () => {
    const stage = makeMapStage(
      { ...look(PLAIN), marks: { pacific: { campaign: true }, plains: { dataCenter: "open" } } },
      STILL,
    );
    const ring = stage.scene.getObjectByName("campaign:pacific");
    stage.setLook({
      ...look(LED),
      marks: { pacific: { campaign: true }, plains: { dataCenter: "open" } },
    });
    expect(stage.scene.getObjectByName("campaign:pacific")).toBe(ring);
    stage.setLook({ ...look(LED), marks: { plains: { dataCenter: "open" } } });
    expect(stage.scene.getObjectByName("campaign:pacific")).toBeUndefined();
    expect(stage.scene.getObjectByName("dc:plains")).toBeDefined();
  });

  it("follows lighter graphics and stillness switched while the map is showing", () => {
    const stage = makeMapStage(
      { ...look(PLAIN), marks: { pacific: { campaign: true } } },
      { still: false, lighter: false },
    );
    const lines = () => stage.scene.getObjectByName("borders:pacific");
    expect(lines()?.visible).toBe(true);
    stage.setMode({ still: true, lighter: true });
    expect(lines()?.visible ?? false).toBe(false);
    expect(stage.update(1, 0.03)).toBe(false);
    const ring = stage.scene.getObjectByName("campaign:pacific") as THREE.Mesh;
    const size = ring.scale.x;
    stage.update(1.37, 0);
    expect(ring.scale.x).toBe(size);
    stage.setMode({ still: false, lighter: false });
    expect(lines()?.visible).toBe(true);
    expect(stage.update(2, 0.03)).toBe(true);
  });
});

/** Three.js draws every sprite on one shared quad, which no sprite owns. */
function isSpriteQuad(stage: ReturnType<typeof makeMapStage>, g: THREE.BufferGeometry): boolean {
  let shared = false;
  stage.scene.traverse((o) => {
    if (o instanceof THREE.Sprite && o.geometry === g) shared = true;
  });
  return shared;
}
