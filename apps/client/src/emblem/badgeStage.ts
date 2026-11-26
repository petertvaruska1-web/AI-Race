import type { Emblem, Palette } from "@airace/protocol";
import * as THREE from "three";
import { SVGLoader } from "three/addons/loaders/SVGLoader.js";
import type { Stage3D } from "../three/engine.ts";
import { darkMetal, metal } from "../three/materials.ts";
import { emblemSvg, framePath } from "./art.ts";

// The emblem as a 3D badge (design §8.2): the frame extruded and beveled in metal, the flat
// emblem's drawing on its face, a slow sway, drag to turn, and the bought logo motions.

/** The emblem's 100-unit box shown as about two units across. */
const SCALE = 0.02;
const TEXTURE = 512;

/** The frame's outline as three.js shapes (from its SVG path). */
function frameShapes(e: Emblem): THREE.Shape[] {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg"><path d="${framePath(e)}"/></svg>`;
  const data = new SVGLoader().parse(svg);
  return data.paths.flatMap((p) => SVGLoader.createShapes(p));
}

function frameGeometry(e: Emblem): THREE.ExtrudeGeometry {
  const geometry = new THREE.ExtrudeGeometry(frameShapes(e), {
    depth: 12,
    bevelEnabled: true,
    bevelThickness: 3,
    bevelSize: 2.4,
    bevelSegments: 3,
    curveSegments: 24,
  });
  // SVG's y runs down; center the 100-unit box on the origin, face toward the camera.
  geometry.translate(-50, -50, -6);
  return geometry;
}

export type BadgeStage = Stage3D & {
  setLook(e: Emblem, p: Palette): void;
  /** Called when the face's drawing has loaded, so a still badge draws again. */
  setRedraw(redraw: () => void): void;
};

/** Builds the badge for `emblem` in `palette`; `motion` is the bought logo look (`logo:*`). */
export function makeBadgeStage(
  emblem: Emblem,
  palette: Palette,
  o: { still: boolean; motion: string },
): BadgeStage {
  const scene = new THREE.Scene();
  // The shared room light (engine.ts sets it on every scene) at half strength: at full strength
  // it lit the enamel past white and the company's colors came out pale. three.js reads the
  // scene's strength here, not a material's own `envMapIntensity`.
  scene.environmentIntensity = 0.5;
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  camera.position.set(0, 0, 5.2);
  // A near key light: on a flat face a far (directional) light lit and glinted the whole face
  // at once; a near one leaves a highlight that travels across it as the badge sways.
  const key = new THREE.PointLight(0xffffff, 20, 0, 2);
  key.position.set(-1.6, 2.2, 3.4);
  const sweep = new THREE.DirectionalLight(0xffffff, 0);
  sweep.position.set(3, 1, 4);
  scene.add(key, sweep, new THREE.AmbientLight(0xffffff, 0.22));

  const badge = new THREE.Group();
  badge.name = "badge";
  badge.scale.set(SCALE, -SCALE, SCALE);
  scene.add(badge);

  let redraw: () => void = () => {};
  let look = { e: emblem, p: palette };
  let texture: THREE.CanvasTexture | null = null;

  // A satin enamel face: a mirror-bright coat reflected the lit room behind the camera and
  // washed the whole face white whenever the badge tilted up.
  const face = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0.05,
    roughness: 0.5,
    specularIntensity: 0.5,
    clearcoat: 0.5,
    clearcoatRoughness: 0.3,
  });
  const mesh = new THREE.Mesh(frameGeometry(emblem), [face, metal()]);
  mesh.name = "badge-face";
  badge.add(mesh);

  /** Draws the flat emblem into the face's texture once its image has loaded. */
  function paint(): void {
    if (typeof document === "undefined") return;
    const canvas = document.createElement("canvas");
    canvas.width = TEXTURE;
    canvas.height = TEXTURE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = new Image();
    image.onload = () => {
      ctx.clearRect(0, 0, TEXTURE, TEXTURE);
      ctx.drawImage(image, 0, 0, TEXTURE, TEXTURE);
      const next = new THREE.CanvasTexture(canvas);
      next.colorSpace = THREE.SRGBColorSpace;
      // The face's UVs are the shape's own units (0–100, y down, set before the geometry was
      // centered): u = x / 100 and v = 1 - y / 100 show the drawing 1:1 on the face.
      next.repeat.set(0.01, -0.01);
      next.offset.set(0, 1);
      next.anisotropy = 4;
      texture?.dispose();
      texture = next;
      face.map = next;
      face.emissiveMap = look.e.finish === "light" ? next : null;
      face.emissive.set(look.e.finish === "light" ? 0xffffff : 0x000000);
      face.emissiveIntensity = look.e.finish === "light" ? 1.1 : 0;
      face.needsUpdate = true;
      redraw();
    };
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(emblemSvg(look.e, look.p, { idPrefix: "badge", size: TEXTURE }))}`;
  }

  function finish(): void {
    mesh.material = [face, look.e.finish === "light" ? darkMetal() : metal()];
    face.transparent = look.e.finish === "glass";
    face.opacity = look.e.finish === "glass" ? 0.92 : 1;
  }
  finish();
  paint();

  // The orbit look: a spark circling the frame.
  let spark: THREE.Mesh | null = null;
  if (o.motion === "logo:orbit") {
    spark = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 16, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(palette.secondary) }),
    );
    spark.name = "orbit-spark";
    scene.add(spark);
  }

  let yaw = 0;
  let pitch = 0;
  let dragging = false;

  return {
    scene,
    camera,
    update(t) {
      const still = o.still;
      const sway = still || dragging ? 0 : 1;
      badge.rotation.set(
        pitch + Math.sin(t * 0.37) * 0.1 * sway,
        yaw + Math.sin(t * 0.5) * 0.45 * sway,
        0,
      );
      if (!still && o.motion === "logo:pulse") {
        const beat = Math.max(0, Math.sin(t * Math.PI * 2 * 1.1)) ** 8;
        badge.scale.set(SCALE * (1 + 0.05 * beat), -SCALE * (1 + 0.05 * beat), SCALE);
      }
      sweep.intensity = !still && o.motion === "logo:shimmer" ? 2.2 : 0;
      if (o.motion === "logo:shimmer") sweep.position.set(Math.sin(t * 0.9) * 5, 1, 4);
      if (spark) {
        const a = still ? 0.8 : t * 1.6;
        spark.position.set(Math.cos(a) * 1.15, Math.sin(a) * 1.15, 0.3);
      }
      return !still;
    },
    pointer(e) {
      if (e.type === "down") dragging = true;
      if (e.type === "up") dragging = false;
      if (e.type === "move") {
        yaw += e.dx * 0.012;
        pitch = Math.max(-0.9, Math.min(0.9, pitch + e.dy * 0.012));
      }
    },
    setLook(e, p) {
      const shapeChanged = framePath(e) !== framePath(look.e);
      look = { e, p };
      if (shapeChanged) {
        mesh.geometry.dispose();
        mesh.geometry = frameGeometry(e);
      }
      finish();
      paint();
    },
    setRedraw(fn) {
      redraw = fn;
    },
    dispose() {
      mesh.geometry.dispose();
      face.dispose();
      texture?.dispose();
      if (spark) {
        spark.geometry.dispose();
        (spark.material as THREE.Material).dispose();
      }
    },
  };
}
