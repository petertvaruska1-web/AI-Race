import { REGION_IDS, type RegionId } from "@airace/protocol";
import * as THREE from "three";
import type { RegionMarks, RegionTop } from "../regions.ts";
import type { Stage3D } from "../three/engine.ts";
import { type Ring, regionShapes } from "./geometry.ts";

// The US map as a 3D tabletop model (design §8.3): the nine regions as glass prisms on a dark
// metal plinth, each led region raised by its leader's share and lit in its color, with the
// player's own marks standing on top. Built from the shared shapes; drawn by the shared engine.

export type MapLook = {
  tops: Record<RegionId, RegionTop>;
  marks: RegionMarks;
  focus: RegionId | null;
  /** The player's own color for their marks. */
  markColor: string;
};

/** The view every player opens the map at (radians; zoom 1 frames the whole map). */
export const MAP_VIEW = { yaw: 0, tilt: 0.95, zoom: 1 } as const;
/** How far the map turns, tilts and zooms. */
export const MAP_LIMITS = {
  yaw: 0.9,
  tiltMin: 0.5,
  tiltMax: 1.35,
  zoomMin: 0.75,
  zoomMax: 1.8,
} as const;

export type MapStage = Stage3D & {
  setLook(look: MapLook): void;
  /** Reduced motion and lighter graphics, switched while the map is showing. */
  setMode(mode: { still: boolean; lighter: boolean }): void;
  setHover(region: RegionId | null): void;
  /** The region under a point of the canvas, in normalized device coordinates (y up). */
  pick(x: number, y: number): RegionId | null;
  zoom(by: -1 | 1): void;
  resetView(): void;
  view(): { yaw: number; tilt: number; zoom: number };
  /** Where a region's marks stand: the world point on top of it. */
  anchor(region: RegionId): THREE.Vector3;
};

/** The map's width in world units; the heights of plain glass and of the most-led region. */
const WIDTH = 2;
const BASE = 0.03;
const RISE = 0.13;
/** Seconds a region takes to rise or sink to a new height. */
const EASE = 0.6;
/** Seconds of one campaign pulse. */
const PULSE = 1.8;
const FOV = 30;
const GLASS = 0x8fa3bd;
const TOP_GLOW = 0.9;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** A ring without its closing point (three.js closes shapes itself). */
function open(ring: Ring): Ring {
  const a = ring[0];
  const b = ring[ring.length - 1];
  return a && b && a[0] === b[0] && a[1] === b[1] ? ring.slice(0, -1) : ring;
}

type Region = {
  id: RegionId;
  group: THREE.Group;
  prism: THREE.Mesh;
  glass: THREE.MeshPhysicalMaterial;
  topGeometry: THREE.ShapeGeometry;
  top: THREE.Mesh | null;
  topMaterial: THREE.MeshStandardMaterial | null;
  borders: THREE.LineSegments | null;
  /** Map-space point of its marks. */
  anchor: [number, number];
  height: number;
  target: number;
};

type Mark = { region: RegionId; object: THREE.Object3D; offset: THREE.Vector3; pulse: boolean };

export function makeMapStage(first: MapLook, o: { still: boolean; lighter: boolean }): MapStage {
  const shapes = regionShapes();
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const s of shapes)
    for (const p of s.polygons)
      for (const [x, y] of p[0] ?? []) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const scale = WIDTH / (maxX - minX);
  const deep = (maxY - minY) * scale;
  let still = o.still;
  let lighter = o.lighter;
  const toMap = ([x, y]: [number, number]) => new THREE.Vector2(x - cx, -(y - cy));
  /** A height (0..1) as the prism's depth in map units. */
  const depthOf = (h: number) => (BASE + RISE * h) / scale;

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];
  const own = <G extends THREE.BufferGeometry>(g: G): G => {
    geometries.push(g);
    return g;
  };
  const ownMaterial = <M extends THREE.Material>(m: M): M => {
    materials.push(m);
    return m;
  };

  const scene = new THREE.Scene();
  // Glass mirrors the shared room light at a little over half its strength.
  scene.environmentIntensity = 0.55;
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 40);
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(-1.5, 3, 2.2);
  const rim = new THREE.DirectionalLight(0x9fb8ff, 0.7);
  rim.position.set(1.2, 1.2, -2.5);
  scene.add(key, rim, new THREE.HemisphereLight(0xdfe8ff, 0x141820, 0.55));

  // The plinth: a slab of dark machined metal the regions stand on.
  const plinth = new THREE.Mesh(
    own(new THREE.BoxGeometry(WIDTH + 0.24, 0.05, deep + 0.24)),
    ownMaterial(
      new THREE.MeshStandardMaterial({ color: 0x23272e, metalness: 0.85, roughness: 0.42 }),
    ),
  );
  plinth.name = "plinth";
  plinth.position.y = -0.026;
  scene.add(plinth);

  // The table: map space (x east, y north, z up) laid flat and scaled to the world.
  const table = new THREE.Group();
  table.rotation.x = -Math.PI / 2;
  table.scale.setScalar(scale);
  scene.add(table);

  const lines = ownMaterial(
    new THREE.LineBasicMaterial({ color: 0x0b0e15, transparent: true, opacity: 0.45 }),
  );
  const regions: Region[] = shapes.map((s) => {
    const outlines = s.polygons.map(([outer, ...holes]) => {
      const shape = new THREE.Shape(open(outer ?? []).map(toMap));
      shape.holes = holes.map((h) => new THREE.Path(open(h).map(toMap)));
      return shape;
    });
    const glass = ownMaterial(
      new THREE.MeshPhysicalMaterial({
        color: GLASS,
        metalness: 0,
        roughness: 0.18,
        transparent: true,
        opacity: 0.38,
        clearcoat: 1,
        clearcoatRoughness: 0.12,
      }),
    );
    const prism = new THREE.Mesh(
      own(new THREE.ExtrudeGeometry(outlines, { depth: 1, bevelEnabled: false, curveSegments: 1 })),
      glass,
    );
    prism.userData.region = s.region;
    const group = new THREE.Group();
    group.name = `region:${s.region}`;
    group.add(prism);
    // The state lines are built either way and shown without lighter graphics, so the switch
    // can change while the map is showing.
    let borders: THREE.LineSegments | null = null;
    if (s.borders.length > 0) {
      const points: THREE.Vector3[] = [];
      for (const line of s.borders)
        for (let i = 1; i < line.length; i++) {
          const a = toMap(line[i - 1] as [number, number]);
          const b = toMap(line[i] as [number, number]);
          points.push(new THREE.Vector3(a.x, a.y, 0), new THREE.Vector3(b.x, b.y, 0));
        }
      borders = new THREE.LineSegments(
        own(new THREE.BufferGeometry().setFromPoints(points)),
        lines,
      );
      borders.name = `borders:${s.region}`;
      borders.visible = !lighter;
      group.add(borders);
    }
    table.add(group);
    return {
      id: s.region,
      group,
      prism,
      glass,
      topGeometry: own(new THREE.ShapeGeometry(outlines, 1)),
      top: null,
      topMaterial: null,
      borders,
      anchor: [s.anchor[0] - cx, -(s.anchor[1] - cy)],
      height: 0,
      target: 0,
    };
  });
  const byId = new Map(regions.map((r) => [r.id, r]));

  function placeHeight(r: Region): void {
    const d = depthOf(r.height);
    r.prism.scale.z = d;
    r.top?.position.set(0, 0, d + 0.4);
    r.borders?.position.set(0, 0, d + 0.8);
  }

  function setTop(r: Region, top: RegionTop): void {
    if (top.color) {
      if (!r.topMaterial || !r.top) {
        r.topMaterial = ownMaterial(
          new THREE.MeshStandardMaterial({
            color: 0x0b0e15,
            emissive: new THREE.Color(top.color),
            emissiveIntensity: TOP_GLOW,
            roughness: 0.35,
            metalness: 0.1,
          }),
        );
        r.top = new THREE.Mesh(r.topGeometry, r.topMaterial);
        r.top.name = `top:${r.id}`;
        r.top.userData.region = r.id;
      }
      r.topMaterial.emissive.set(top.color);
      if (!r.top.parent) r.group.add(r.top);
      r.glass.color.set(GLASS).lerp(new THREE.Color(top.color), 0.35);
    } else {
      if (r.top?.parent) r.group.remove(r.top);
      r.glass.color.set(GLASS);
    }
    r.target = clamp(top.height, 0, 1);
  }

  // The player's marks, in world space over their regions.
  let marks: Mark[] = [];
  let markGeometries: THREE.BufferGeometry[] = [];
  let markMaterials: THREE.Material[] = [];
  let markTextures: THREE.Texture[] = [];
  const markLayer = new THREE.Group();
  scene.add(markLayer);
  const markGeometry = <G extends THREE.BufferGeometry>(g: G): G => {
    markGeometries.push(g);
    return g;
  };
  const markMaterial = <M extends THREE.Material>(m: M): M => {
    markMaterials.push(m);
    return m;
  };

  function dataCenter(open: boolean): THREE.Group {
    const g = new THREE.Group();
    const metal = markMaterial(
      new THREE.MeshStandardMaterial({ color: 0x2b3038, metalness: 0.8, roughness: 0.4 }),
    );
    const h = open ? 0.06 : 0.03;
    const body = new THREE.Mesh(markGeometry(new THREE.BoxGeometry(0.045, h, 0.045)), metal);
    body.position.y = h / 2;
    g.add(body);
    if (open) {
      const lit = markMaterial(
        new THREE.MeshStandardMaterial({
          color: 0x111111,
          emissive: new THREE.Color(0xffd27a),
          emissiveIntensity: 1.4,
        }),
      );
      for (const y of [0.018, 0.034, 0.05]) {
        const strip = new THREE.Mesh(markGeometry(new THREE.BoxGeometry(0.047, 0.006, 0.047)), lit);
        strip.position.y = y;
        g.add(strip);
      }
    } else {
      const scaffold = new THREE.LineSegments(
        markGeometry(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.056, 0.066, 0.056))),
        markMaterial(new THREE.LineBasicMaterial({ color: 0xc9ced6 })),
      );
      scaffold.position.y = 0.033;
      g.add(scaffold);
    }
    return g;
  }

  function campaign(color: string): THREE.Mesh {
    const ring = new THREE.Mesh(
      markGeometry(new THREE.RingGeometry(0.08, 0.095, 48)),
      markMaterial(
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(color),
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      ),
    );
    ring.rotation.x = -Math.PI / 2;
    return ring;
  }

  function sponsor(text: string, color: string): THREE.Group {
    const g = new THREE.Group();
    const stick = new THREE.Mesh(
      markGeometry(new THREE.CylinderGeometry(0.005, 0.005, 0.1, 8)),
      markMaterial(
        new THREE.MeshStandardMaterial({ color: 0xb7bdc6, metalness: 0.9, roughness: 0.3 }),
      ),
    );
    stick.position.y = 0.05;
    const head = new THREE.Mesh(
      markGeometry(new THREE.SphereGeometry(0.017, 16, 12)),
      markMaterial(
        new THREE.MeshStandardMaterial({
          color: 0x111111,
          emissive: new THREE.Color(color),
          emissiveIntensity: 1.2,
        }),
      ),
    );
    head.position.y = 0.105;
    g.add(stick, head);
    const canvas = typeof document === "undefined" ? null : document.createElement("canvas");
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      canvas.width = 64;
      canvas.height = 64;
      ctx.font = "44px system-ui, 'Segoe UI Emoji', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 32, 36);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      markTextures.push(texture);
      const flag = new THREE.Sprite(markMaterial(new THREE.SpriteMaterial({ map: texture })));
      flag.scale.setScalar(0.09);
      flag.position.y = 0.16;
      g.add(flag);
    }
    return g;
  }

  function freeMarks(): void {
    for (const m of marks) markLayer.remove(m.object);
    for (const g of markGeometries) g.dispose();
    for (const m of markMaterials) m.dispose();
    for (const t of markTextures) t.dispose();
    marks = [];
    markGeometries = [];
    markMaterials = [];
    markTextures = [];
  }

  function buildMarks(all: RegionMarks, color: string): void {
    freeMarks();
    for (const region of REGION_IDS) {
      const m = all[region];
      if (!m) continue;
      const add = (name: string, object: THREE.Object3D, offset: THREE.Vector3, pulse = false) => {
        object.name = `${name}:${region}`;
        markLayer.add(object);
        marks.push({ region, object, offset, pulse });
      };
      if (m.dataCenter)
        add("dc", dataCenter(m.dataCenter === "open"), new THREE.Vector3(0.045, 0, 0.02));
      if (m.campaign) add("campaign", campaign(color), new THREE.Vector3(0, 0.004, 0), true);
      if (m.sponsor) add("sponsor", sponsor(m.sponsor, color), new THREE.Vector3(-0.045, 0, -0.02));
    }
  }

  let yaw: number = MAP_VIEW.yaw;
  let tilt: number = MAP_VIEW.tilt;
  let zoom: number = MAP_VIEW.zoom;
  let mode: "turn" | "tilt" | null = null;
  let hover: RegionId | null = null;
  let focus: RegionId | null = first.focus;

  // Framing: the camera finds the nearest distance at which the model's corners (the plinth, and
  // above it the tallest a region and its marks stand) all fit the canvas, and aims so they sit
  // centered on it, for every turn, tilt and canvas shape; zoom then moves it from there.
  const LOW = -0.052;
  const HIGH = BASE + RISE + 0.1;
  const halfW = WIDTH / 2 + 0.12;
  const halfD = deep / 2 + 0.12;
  const corners = [-1, 1].flatMap((sx) =>
    [-1, 1].flatMap((sz) => [LOW, HIGH].map((y) => new THREE.Vector3(sx * halfW, y, sz * halfD))),
  );
  /** The share of the frame the model fills on its tighter side. */
  const FILL = 0.94;
  const probe = new THREE.PerspectiveCamera(FOV, 1, 0.05, 80);
  const fits = new Map<string, { distance: number; aim: THREE.Vector3 }>();
  const seen = new THREE.Vector3();

  /** Puts `cam` `distance` away from `aim` at the current turn and tilt, looking at it. */
  function place(cam: THREE.PerspectiveCamera, aim: THREE.Vector3, distance: number): void {
    cam.position.set(
      aim.x + distance * Math.cos(tilt) * Math.sin(yaw),
      aim.y + distance * Math.sin(tilt),
      aim.z + distance * Math.cos(tilt) * Math.cos(yaw),
    );
    cam.lookAt(aim);
    cam.updateMatrixWorld();
  }

  /** Where the corners land on the probe's screen; null when one is behind it. */
  function extent(aim: THREE.Vector3, distance: number) {
    place(probe, aim, distance);
    let x0 = Number.POSITIVE_INFINITY;
    let x1 = Number.NEGATIVE_INFINITY;
    let y0 = Number.POSITIVE_INFINITY;
    let y1 = Number.NEGATIVE_INFINITY;
    for (const c of corners) {
      seen.copy(c).applyMatrix4(probe.matrixWorldInverse);
      if (seen.z > -probe.near) return null;
      seen.applyMatrix4(probe.projectionMatrix);
      x0 = Math.min(x0, seen.x);
      x1 = Math.max(x1, seen.x);
      y0 = Math.min(y0, seen.y);
      y1 = Math.max(y1, seen.y);
    }
    return { x0, x1, y0, y1 };
  }

  /** The nearest distance from `aim` at which every corner is inside the frame. */
  function nearest(aim: THREE.Vector3): number {
    let lo = 0.3;
    let hi = 60;
    for (let i = 0; i < 32; i++) {
      const mid = (lo + hi) / 2;
      const e = extent(aim, mid);
      if (e && Math.max(-e.x0, e.x1, -e.y0, e.y1) <= FILL) hi = mid;
      else lo = mid;
    }
    return hi;
  }

  /** The framing at zoom 1 for a canvas shape and the current turn and tilt (remembered). */
  function fit(aspect: number): { distance: number; aim: THREE.Vector3 } {
    const key = `${aspect.toFixed(3)} ${yaw.toFixed(4)} ${tilt.toFixed(4)}`;
    const known = fits.get(key);
    if (known) return known;
    probe.aspect = aspect;
    probe.updateProjectionMatrix();
    const t = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
    const toward = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
    const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const aim = new THREE.Vector3(0, (LOW + HIGH) / 2, 0);
    let distance = nearest(aim);
    // Perspective draws the near half larger: slide the aim until the model sits centered.
    for (let round = 0; round < 4; round++) {
      const e = extent(aim, distance);
      if (!e) break;
      const midX = (e.x0 + e.x1) / 2;
      const midY = (e.y0 + e.y1) / 2;
      if (Math.abs(midX) < 0.005 && Math.abs(midY) < 0.005) break;
      aim.addScaledVector(toward, (-midY * distance * t) / Math.max(0.2, Math.sin(tilt)));
      aim.addScaledVector(right, midX * distance * t * aspect);
      distance = nearest(aim);
    }
    if (fits.size > 256) fits.clear();
    const framing = { distance, aim };
    fits.set(key, framing);
    return framing;
  }

  /** Frames the whole map at zoom 1, whatever the canvas's shape, turn and tilt. */
  function placeCamera(): void {
    const { distance, aim } = fit(camera.aspect || 1);
    place(camera, aim, distance / zoom);
  }

  function shade(): void {
    for (const r of regions) {
      const dim = focus && focus !== r.id ? 0.35 : 1;
      const lit = hover === r.id ? 1.6 : 1;
      if (r.topMaterial) r.topMaterial.emissiveIntensity = TOP_GLOW * lit * dim;
      r.glass.opacity = (hover === r.id ? 0.55 : 0.38) * (dim < 1 ? 0.6 : 1);
    }
  }

  function anchor(region: RegionId): THREE.Vector3 {
    const r = byId.get(region);
    if (!r) return new THREE.Vector3();
    table.updateMatrixWorld(true);
    return new THREE.Vector3(r.anchor[0], r.anchor[1], depthOf(r.height) + 0.4).applyMatrix4(
      table.matrixWorld,
    );
  }

  function placeMarks(phase: number): void {
    for (const m of marks) {
      m.object.position.copy(anchor(m.region)).add(m.offset);
      if (m.pulse) {
        const mesh = m.object as THREE.Mesh;
        mesh.scale.setScalar(still ? 1.1 : 0.7 + 0.9 * phase);
        (mesh.material as THREE.MeshBasicMaterial).opacity = still ? 0.7 : 0.85 * (1 - phase);
      }
    }
  }

  /** The marks last built, so a new look rebuilds them only when they change (not every day). */
  let builtMarks = "";
  function setLook(look: MapLook): void {
    for (const r of regions) setTop(r, look.tops[r.id]);
    focus = look.focus;
    const wanted = JSON.stringify([look.marks, look.markColor]);
    if (wanted !== builtMarks) {
      builtMarks = wanted;
      buildMarks(look.marks, look.markColor);
    }
    shade();
  }

  setLook(first);
  for (const r of regions) {
    r.height = r.target;
    placeHeight(r);
  }
  placeCamera();
  placeMarks(0);

  const raycaster = new THREE.Raycaster();

  return {
    scene,
    camera,
    update(t, dt) {
      placeCamera();
      let moving = false;
      for (const r of regions) {
        if (r.height === r.target) continue;
        if (still || dt <= 0) r.height = r.target;
        else {
          const step = dt / EASE;
          r.height += clamp(r.target - r.height, -step, step);
          if (r.height !== r.target) moving = true;
        }
        placeHeight(r);
      }
      const pulsing = !still && marks.some((m) => m.pulse);
      placeMarks(pulsing ? (t % PULSE) / PULSE : 0);
      return moving || pulsing;
    },
    pointer(e) {
      if (e.type === "down") mode = e.button === 2 ? "tilt" : "turn";
      else if (e.type === "up") mode = null;
      // The near edge follows the pointer, as the core and the emblem turn with a drag.
      else if (mode === "turn") yaw = clamp(yaw - e.dx * 0.006, -MAP_LIMITS.yaw, MAP_LIMITS.yaw);
      else if (mode === "tilt")
        tilt = clamp(tilt + e.dy * 0.005, MAP_LIMITS.tiltMin, MAP_LIMITS.tiltMax);
      placeCamera();
    },
    setLook,
    setMode(next) {
      still = next.still;
      lighter = next.lighter;
      for (const r of regions) if (r.borders) r.borders.visible = !lighter;
    },
    setHover(region) {
      hover = region;
      shade();
    },
    pick(x, y) {
      placeCamera();
      scene.updateMatrixWorld(true);
      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
      const hit = raycaster
        .intersectObjects(table.children, true)
        .find((h) => h.object.userData.region !== undefined);
      return (hit?.object.userData.region as RegionId | undefined) ?? null;
    },
    zoom(by) {
      zoom = clamp(zoom * (by > 0 ? 1.12 : 1 / 1.12), MAP_LIMITS.zoomMin, MAP_LIMITS.zoomMax);
      placeCamera();
    },
    resetView() {
      yaw = MAP_VIEW.yaw;
      tilt = MAP_VIEW.tilt;
      zoom = MAP_VIEW.zoom;
      placeCamera();
    },
    view: () => ({ yaw, tilt, zoom }),
    anchor,
    dispose() {
      freeMarks();
      for (const g of geometries) g.dispose();
      for (const m of materials) m.dispose();
    },
  };
}
