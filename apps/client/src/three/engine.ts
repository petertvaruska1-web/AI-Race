import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createFrameMeter, due, type FramePolicy, framePolicy } from "../graphics/frames.ts";
import { reducedMotion } from "../motion.ts";
import { useGraphics } from "../store/graphics.ts";

// One shared 3D engine (design §8.0, ADR 0011 §1): a single WebGL renderer draws every 3D object
// into its own canvas on the page, only while the object is visible and changing, and never
// more often than the frame policy allows (R3: the player's graphics card is training their AI).

/** One 3D object: its scene and camera, and how it moves. */
export interface Stage3D {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  /** Advances the object by `dt` seconds (`t` since it started); true while it is still moving. */
  update(t: number, dt: number): boolean;
  /** Drags on the object's canvas, in CSS pixels; `button` says which button began it. */
  pointer?(e: { type: "down" | "move" | "up"; dx: number; dy: number; button?: number }): void;
  dispose(): void;
}

export type ViewState = {
  id: number;
  visible: boolean;
  /** Something changed that a still frame must show. */
  dirty: boolean;
  /** The object is animating on its own. */
  moving: boolean;
  /** When it was last drawn (ms). */
  last: number;
};

/**
 * The views to draw now: visible ones that changed, and visible moving ones whose next frame is
 * due. A still policy draws only changed ones; a stopped one (no frames, not still) draws none.
 */
export function dueViews(views: ViewState[], now: number, policy: FramePolicy): number[] {
  return views
    .filter((v) => {
      if (!v.visible) return false;
      if (v.dirty) return policy.still || policy.fps > 0;
      return v.moving && due(now, v.last, policy.fps);
    })
    .map((v) => v.id);
}

type Entry = {
  state: ViewState;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D | null;
  stage: Stage3D;
  started: number;
};

const entries = new Map<number, Entry>();
let nextId = 1;
let renderer: THREE.WebGLRenderer | null = null;
let environment: THREE.Texture | null = null;
let frame = 0;
let interactingUntil = 0;
const meter = createFrameMeter();
let lastFrame = 0;

/** The environment light every metal and glass surface reflects (made once). */
export function sharedEnvironment(r: THREE.WebGLRenderer): THREE.Texture {
  if (!environment) {
    const pmrem = new THREE.PMREMGenerator(r);
    environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
  }
  return environment;
}

/** Keeps drawing at the interactive rate for `ms` (a drag on a 3D object). */
export function markInteracting(ms = 1200): void {
  interactingUntil = performance.now() + ms;
  wake();
}

function theRenderer(): THREE.WebGLRenderer {
  if (renderer) return renderer;
  const r = new THREE.WebGLRenderer({
    canvas: document.createElement("canvas"),
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
    powerPreference: "low-power",
  });
  r.outputColorSpace = THREE.SRGBColorSpace;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.05;
  r.setClearColor(0x000000, 0);
  r.autoClear = false;
  renderer = r;
  return r;
}

function pixelRatio(): number {
  return useGraphics.getState().lighter ? 1 : Math.min(window.devicePixelRatio || 1, 1.5);
}

function draw(entry: Entry, now: number, still: boolean): void {
  const r = theRenderer();
  const ratio = pixelRatio();
  const cssW = entry.canvas.clientWidth;
  const cssH = entry.canvas.clientHeight;
  const w = Math.max(1, Math.round(cssW * ratio));
  const h = Math.max(1, Math.round(cssH * ratio));
  if (entry.canvas.width !== w || entry.canvas.height !== h) {
    entry.canvas.width = w;
    entry.canvas.height = h;
    entry.stage.camera.aspect = w / h;
    entry.stage.camera.updateProjectionMatrix();
  }
  // The shared surface grows to the largest view; each view uses its lower-left corner.
  const size = r.getSize(new THREE.Vector2());
  if (size.x < w || size.y < h) r.setSize(Math.max(size.x, w), Math.max(size.y, h), false);
  const full = r.domElement.height;
  const t = (now - entry.started) / 1000;
  const dt = entry.state.last ? Math.min(0.1, (now - entry.state.last) / 1000) : 0;
  entry.state.moving = entry.stage.update(t, still ? 0 : dt) && !still;
  r.setViewport(0, 0, w, h);
  r.setScissor(0, 0, w, h);
  r.setScissorTest(true);
  r.clear();
  r.render(entry.stage.scene, entry.stage.camera);
  if (entry.ctx) {
    entry.ctx.clearRect(0, 0, w, h);
    entry.ctx.drawImage(r.domElement, 0, full - h, w, h, 0, 0, w, h);
  }
  entry.state.dirty = false;
  entry.state.last = now;
}

function tick(now: number): void {
  frame = 0;
  if (entries.size === 0 || document.visibilityState === "hidden") return;
  const policy = framePolicy({
    reducedMotion: reducedMotion(),
    lighter: useGraphics.getState().lighter,
    hidden: false,
    interacting: now < interactingUntil,
  });
  const ids = dueViews(
    [...entries.values()].map((e) => e.state),
    now,
    policy,
  );
  if (ids.length > 0 && !policy.still) {
    if (lastFrame > 0 && meter.record(now - lastFrame, policy.fps)) {
      useGraphics.getState().setLighter(true, true);
    }
    lastFrame = now;
  }
  for (const id of ids) {
    const entry = entries.get(id);
    if (entry) draw(entry, now, policy.still);
  }
  const anyMoving = [...entries.values()].some((e) => e.state.visible && e.state.moving);
  const anyDirty = [...entries.values()].some((e) => e.state.visible && e.state.dirty);
  if ((anyMoving && !policy.still) || anyDirty) frame = requestAnimationFrame(tick);
}

function wake(): void {
  if (frame === 0 && entries.size > 0) frame = requestAnimationFrame(tick);
}

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") return;
    for (const e of entries.values()) e.state.dirty = true;
    lastFrame = 0;
    wake();
  });
  useGraphics.subscribe(() => {
    for (const e of entries.values()) e.state.dirty = true;
    wake();
  });
}

/** Draws `stage` into `canvas` from now on; the returned handle redraws, shows or detaches it. */
export function attachView(
  canvas: HTMLCanvasElement,
  stage: Stage3D,
): { markDirty(): void; setVisible(visible: boolean): void; detach(): void } {
  const id = nextId++;
  const r = theRenderer();
  stage.scene.environment = sharedEnvironment(r);
  const entry: Entry = {
    state: { id, visible: true, dirty: true, moving: true, last: 0 },
    canvas,
    ctx: canvas.getContext("2d"),
    stage,
    started: performance.now(),
  };
  entries.set(id, entry);
  wake();
  return {
    markDirty() {
      entry.state.dirty = true;
      wake();
    },
    setVisible(visible: boolean) {
      entry.state.visible = visible;
      if (visible) {
        entry.state.dirty = true;
        wake();
      }
    },
    detach() {
      entries.delete(id);
      stage.dispose();
    },
  };
}
