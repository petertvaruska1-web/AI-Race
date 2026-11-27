import { REGION_IDS, type RegionId } from "@airace/protocol";
import { merge, mesh } from "topojson-client";
import type {
  GeometryCollection,
  GeometryObject,
  MultiPolygon,
  Topology,
  Polygon as TopoPolygon,
} from "topojson-specification";
import topologyJson from "us-atlas/states-albers-10m.json";
import { STATE_TILES } from "../regions.ts";

// The map's shapes (design §8.3, ADR 0011 §4): the 50 states and DC from us-atlas (US Census
// Bureau cartographic boundaries, public domain), already projected (Albers USA, with Alaska and
// Hawaii as insets), merged into the game's nine regions. Pure, so tests need no screen.

/** us-atlas's projected frame (y runs down). */
export const MAP_FRAME = { width: 975, height: 610 } as const;

export type Ring = [number, number][];
/** An outer ring first, then its holes. */
export type Polygon = Ring[];

export type RegionShape = {
  region: RegionId;
  polygons: Polygon[];
  /** The state lines inside the region. */
  borders: Ring[];
  /** Where the region's marks stand: inside its largest mainland polygon. */
  anchor: [number, number];
};

/** How far (frame units) a simplified outline may stray from the real one. */
const SIMPLIFY = 0.6;
/** Islands smaller than this (square frame units) are left out. */
const MIN_AREA = 6;

/** The FIPS ids of us-atlas's states, as postal codes. */
export const FIPS_TO_STATE: Record<string, string> = {
  "01": "AL",
  "02": "AK",
  "04": "AZ",
  "05": "AR",
  "06": "CA",
  "08": "CO",
  "09": "CT",
  "10": "DE",
  "11": "DC",
  "12": "FL",
  "13": "GA",
  "15": "HI",
  "16": "ID",
  "17": "IL",
  "18": "IN",
  "19": "IA",
  "20": "KS",
  "21": "KY",
  "22": "LA",
  "23": "ME",
  "24": "MD",
  "25": "MA",
  "26": "MI",
  "27": "MN",
  "28": "MS",
  "29": "MO",
  "30": "MT",
  "31": "NE",
  "32": "NV",
  "33": "NH",
  "34": "NJ",
  "35": "NM",
  "36": "NY",
  "37": "NC",
  "38": "ND",
  "39": "OH",
  "40": "OK",
  "41": "OR",
  "42": "PA",
  "44": "RI",
  "45": "SC",
  "46": "SD",
  "47": "TN",
  "48": "TX",
  "49": "UT",
  "50": "VT",
  "51": "VA",
  "53": "WA",
  "54": "WV",
  "55": "WI",
  "56": "WY",
};

const topology = topologyJson as unknown as Topology;
const states = topology.objects.states as GeometryCollection;
const regionOfState = new Map(STATE_TILES.map((t) => [t.state, t.region]));

function regionOf(g: GeometryObject): RegionId | undefined {
  return regionOfState.get(FIPS_TO_STATE[String(g.id)] ?? "");
}

const same = (a: [number, number], b: [number, number]) => a[0] === b[0] && a[1] === b[1];

/** The distance from `p` to the segment `a`–`b`. */
function toSegment(p: [number, number], a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = dx * dx + dy * dy;
  const t =
    len === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/**
 * The ring or line with the points that stray less than `tolerance` from it left out
 * (Douglas–Peucker). A closed ring keeps at least 4 points; tolerance 0 keeps every point.
 */
export function simplifyRing(ring: Ring, tolerance: number): Ring {
  if (tolerance <= 0 || ring.length <= 2) return ring.slice();
  const last = ring.length - 1;
  const keep = new Uint8Array(ring.length);
  keep[0] = 1;
  keep[last] = 1;
  const spans: [number, number][] = [[0, last]];
  while (spans.length > 0) {
    const [from, to] = spans.pop() as [number, number];
    let far = -1;
    let most = tolerance;
    for (let i = from + 1; i < to; i++) {
      const d = toSegment(
        ring[i] as [number, number],
        ring[from] as [number, number],
        ring[to] as [number, number],
      );
      if (d > most) {
        most = d;
        far = i;
      }
    }
    if (far >= 0) {
      keep[far] = 1;
      spans.push([from, far], [far, to]);
    }
  }
  const out = ring.filter((_, i) => keep[i] === 1);
  const closed = same(ring[0] as [number, number], ring[last] as [number, number]);
  return closed && out.length < 4 ? ring.slice() : out;
}

/** A ring's area (unsigned). */
export function ringArea(ring: Ring): number {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i] as [number, number];
    const [xj, yj] = ring[j] as [number, number];
    sum += xj * yi - xi * yj;
  }
  return Math.abs(sum) / 2;
}

/** A ring's centroid (area-weighted). */
function centroid(ring: Ring): [number, number] {
  let a = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i] as [number, number];
    const [xj, yj] = ring[j] as [number, number];
    const f = xj * yi - xi * yj;
    a += f;
    cx += (xj + xi) * f;
    cy += (yj + yi) * f;
  }
  return a === 0 ? (ring[0] ?? [0, 0]) : [cx / (3 * a), cy / (3 * a)];
}

/** Whether `p` is inside the polygon (even-odd, so holes count as outside). */
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

/**
 * A point inside the polygon near `near`: `near` itself when it is inside, otherwise the middle
 * of the inside stretch of its horizontal line that lies closest to it (a bay or a lake can hold
 * an outline's centroid).
 */
function interiorPoint(polygon: Polygon, near: [number, number]): [number, number] {
  if (inside(near, polygon)) return near;
  const y = near[1];
  const xs: number[] = [];
  for (const ring of polygon) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i] as [number, number];
      const [xj, yj] = ring[j] as [number, number];
      if (yi > y !== yj > y) xs.push(((xj - xi) * (y - yi)) / (yj - yi) + xi);
    }
  }
  xs.sort((a, b) => a - b);
  let best: [number, number] = (polygon[0]?.[0] as [number, number]) ?? near;
  let gap = Number.POSITIVE_INFINITY;
  for (let i = 0; i + 1 < xs.length; i += 2) {
    const mid = ((xs[i] as number) + (xs[i + 1] as number)) / 2;
    const d = Math.abs(mid - near[0]);
    if (d < gap) {
      gap = d;
      best = [mid, y];
    }
  }
  return best;
}

/** GeoJSON polygons as simplified rings, small islands and holes left out. */
function polygonsOf(coordinates: number[][][][]): Polygon[] {
  const out: Polygon[] = [];
  for (const rings of coordinates) {
    const [outer, ...holes] = rings.map((r) => simplifyRing(r as Ring, SIMPLIFY));
    if (!outer || ringArea(outer) < MIN_AREA) continue;
    out.push([outer, ...holes.filter((h) => ringArea(h) >= MIN_AREA)]);
  }
  return out;
}

function build(): RegionShape[] {
  return REGION_IDS.map((region) => {
    const mine = states.geometries.filter((g) => regionOf(g) === region) as (
      | TopoPolygon
      | MultiPolygon
    )[];
    const polygons = polygonsOf(merge(topology, mine).coordinates as number[][][][]);
    const borders = mesh(
      topology,
      states,
      (a, b) => a !== b && regionOf(a) === region && regionOf(b) === region,
    ).coordinates.map((line) => simplifyRing(line as Ring, SIMPLIFY));
    // Marks stand on the mainland: Alaska and Hawaii would pull the Pacific's middle out to sea.
    const mainland = mine.filter((g) => g.id !== "02" && g.id !== "15");
    const land = polygonsOf(merge(topology, mainland).coordinates as number[][][][]);
    const largest = land.reduce<Polygon | null>(
      (big, p) => (!big || ringArea(p[0] as Ring) > ringArea(big[0] as Ring) ? p : big),
      null,
    );
    const anchor = largest ? interiorPoint(largest, centroid(largest[0] as Ring)) : [0, 0];
    return { region, polygons, borders, anchor: anchor as [number, number] };
  });
}

let shapes: RegionShape[] | null = null;

/** The nine regions' shapes, in `REGION_IDS` order (built once). */
export function regionShapes(): RegionShape[] {
  shapes ??= build();
  return shapes;
}
