import { REGION_IDS } from "@airace/protocol";
import type { GeometryCollection, Topology } from "topojson-specification";
import topologyJson from "us-atlas/states-albers-10m.json";
import { describe, expect, it } from "vitest";
import {
  FIPS_TO_STATE,
  type Polygon,
  type Ring,
  regionShapes,
  simplifyRing,
} from "../src/map/geometry.ts";
import { STATE_TILES } from "../src/regions.ts";

const topology = topologyJson as unknown as Topology;

/** Even-odd point in polygon (holes count as outside). */
function inside([x, y]: [number, number], polygon: Polygon): boolean {
  let hit = false;
  for (const ring of polygon) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i] as [number, number];
      const [xj, yj] = ring[j] as [number, number];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
  }
  return hit;
}

describe("the map's shapes (public-domain state outlines, ADR 0011 §4)", () => {
  it("places every state of the map in one of the nine regions", () => {
    const ids = (topology.objects.states as GeometryCollection).geometries.map((g) => String(g.id));
    expect(ids).toHaveLength(51);
    for (const id of ids)
      expect(
        STATE_TILES.some((t) => t.state === FIPS_TO_STATE[id]),
        id,
      ).toBe(true);
  });

  it("gives each region its shapes, Alaska and Hawaii in the Pacific's", () => {
    const shapes = regionShapes();
    expect(shapes.map((s) => s.region)).toEqual([...REGION_IDS]);
    for (const s of shapes) expect(s.polygons.length, s.region).toBeGreaterThan(0);
    const pacific = shapes.find((s) => s.region === "pacific");
    // The insets sit in the frame's lower left, under the mainland's west coast.
    const lowerLeft = pacific?.polygons.filter(([outer]) =>
      (outer ?? []).every(([x, y]) => x < 330 && y > 420),
    );
    expect(lowerLeft?.length).toBeGreaterThanOrEqual(2);
  });

  it("stands each region's marks inside the region", () => {
    for (const s of regionShapes())
      expect(
        s.polygons.some((p) => inside(s.anchor, p)),
        s.region,
      ).toBe(true);
  });

  it("keeps state lines inside a region, not between regions", () => {
    expect(
      regionShapes().find((s) => s.region === "southAtlantic")?.borders.length,
    ).toBeGreaterThan(0);
  });

  it("simplifies a straight run to its ends and keeps closed rings closed", () => {
    expect(
      simplifyRing(
        [
          [0, 0],
          [1, 0.01],
          [2, 0],
          [3, 0],
        ],
        0.1,
      ),
    ).toEqual([
      [0, 0],
      [3, 0],
    ]);
    const square: Ring = [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [0, 0],
    ];
    expect(simplifyRing(square, 50).length).toBeGreaterThanOrEqual(4);
    expect(simplifyRing(square, 0)).toEqual(square);
  });

  it("stays light enough to draw on a laptop card", () => {
    const points = regionShapes()
      .flatMap((s) => s.polygons.flat())
      .reduce((n, r) => n + r.length, 0);
    expect(points).toBeLessThan(14_000);
  });
});
