import * as THREE from "three";
import type { OrbParams } from "../orb/params.ts";
import type { Stage3D } from "../three/engine.ts";
import { gearRing } from "../three/gear.ts";
import { metal } from "../three/materials.ts";

// The AI core in 3D (design §8.1, C60): a faceted crystal with a glowing heart, held by geared
// metal rings. Everything about how it looks comes from the measurements through `orbParams`
// (R5): size and heart from the overall score, a ring per skill, ring detail from words read,
// spin from boldness and inventiveness, wobble from inventiveness, breathing from steadiness,
// sparks from verbosity, glow from warmth, flicker from slips.

const HEART_VERTEX = `varying vec3 vWorld;
void main(){
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const HEART_FRAGMENT = `uniform vec3 uMain;uniform vec3 uSecond;uniform float uHeart;uniform float uTime;
uniform float uBeat;uniform float uRadius;uniform float uDim;
varying vec3 vWorld;
float hash(vec3 p){p=fract(p*0.3183099+0.1);p*=17.0;return fract(p.x*p.y*p.z*(p.x+p.y+p.z));}
void main(){
  vec3 n=normalize(cross(dFdx(vWorld),dFdy(vWorld)));
  vec3 V=normalize(cameraPosition-vWorld);
  if(dot(n,V)<0.0)n=-n;
  // Light inside the crystal: follow the bent ray through it and gather the glowing heart.
  vec3 rdi=refract(-V,n,0.69);vec3 q=vWorld;float acc=0.0;
  for(int k=0;k<22;k++){q+=rdi*(0.065*uRadius);acc+=exp(-length(q)/uRadius*4.0)*0.085;}
  float beat=0.85+0.15*sin(uTime*uBeat);
  vec3 heart=mix(uMain,vec3(1.0),0.45)*acc*2.6*beat*uHeart+uMain*0.16;
  float facet=hash(floor(n*7.0));
  vec3 L=normalize(vec3(-0.5,0.8,0.65));
  float diff=max(dot(n,L),0.0);
  vec3 body=mix(uMain*0.22,uMain*0.55,facet)*(0.5+0.5*diff);
  float fres=pow(1.0-max(dot(n,V),0.0),3.0);
  float spec=pow(max(dot(n,normalize(L+V)),0.0),70.0);
  vec3 col=mix(body+heart,mix(uSecond,vec3(1.0),0.6),fres*0.75)+vec3(1.0)*spec*0.95;
  gl_FragColor=vec4(col*uDim,1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

/** A soft round light for the halo and the sparks (no canvas needed). */
function glowTexture(): THREE.DataTexture {
  const n = 64;
  const data = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const d = Math.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n / 2);
      const a = Math.max(0, 1 - d) ** 2;
      const i = (y * n + x) * 4;
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = Math.round(a * 255);
    }
  const texture = new THREE.DataTexture(data, n, n);
  texture.needsUpdate = true;
  return texture;
}

type Mote = { pos: THREE.Vector3; vel: THREE.Vector3; life: number; color: THREE.Color };

/** Picks a color by share (shares add up to 1). */
function byShare(colors: string[], shares: number[]): string {
  let r = Math.random();
  for (let i = 0; i < colors.length; i++) {
    r -= shares[i] ?? 0;
    if (r <= 0) return colors[i] ?? "#ffffff";
  }
  return colors.at(-1) ?? "#ffffff";
}

const MOTES = 220;

/** The farthest any part of the core can reach: the 12th ring around the largest crystal. */
const MAX_REACH = 5.1;

export type CoreStage = Stage3D & {
  setParams(p: OrbParams): void;
  setStill(still: boolean): void;
};

/** Builds the core for `params`; `still` draws it without moving (reduced motion, lighter graphics). */
export function makeCoreStage(params: OrbParams, o: { still: boolean }): CoreStage {
  let p = params;
  let still = o.still;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  // The camera stays where the largest possible instrument fits (12 rings around the biggest
  // crystal), so a stronger AI shows as a bigger one (R5) and nothing jumps as rings fade in.
  camera.position.set(0, 0, (MAX_REACH * 1.12) / Math.tan(THREE.MathUtils.degToRad(15)));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(-3, 4, 5);
  scene.add(key, new THREE.AmbientLight(0xffffff, 0.25));
  const heartLight = new THREE.PointLight(0xffffff, 6, 6, 1.6);
  scene.add(heartLight);

  const rig = new THREE.Group();
  rig.name = "core";
  scene.add(rig);

  const U = {
    uMain: { value: new THREE.Color() },
    uSecond: { value: new THREE.Color() },
    uHeart: { value: 1 },
    uTime: { value: 0 },
    uBeat: { value: 2 },
    uRadius: { value: 1 },
    uDim: { value: 1 },
  };
  const heart = new THREE.ShaderMaterial({
    vertexShader: HEART_VERTEX,
    fragmentShader: HEART_FRAGMENT,
    uniforms: U,
  });
  const crystal = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 0), heart);
  crystal.name = "crystal";
  rig.add(crystal);

  const glow = glowTexture();
  const halo = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: glow,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      transparent: true,
    }),
  );
  halo.name = "halo";
  scene.add(halo);

  let rings: { mesh: THREE.Mesh; tilt: THREE.Group; speed: number; phase: number }[] = [];
  const ringMaterials: THREE.Material[] = [];

  const motesGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(MOTES * 3);
  const colors = new Float32Array(MOTES * 3);
  const positionAttr = new THREE.BufferAttribute(positions, 3);
  const colorAttr = new THREE.BufferAttribute(colors, 3);
  motesGeo.setAttribute("position", positionAttr);
  motesGeo.setAttribute("color", colorAttr);
  const motesMat = new THREE.PointsMaterial({
    size: 0.09,
    map: glow,
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const motes = new THREE.Points(motesGeo, motesMat);
  motes.name = "motes";
  scene.add(motes);
  let flying: Mote[] = [];

  const crystalRadius = () => (p.rings === 0 ? 0.55 : 0.75 + 0.9 * p.radius);
  /** The rings sit close around the crystal, so even ten of them stay a compact instrument. */
  const ringRadius = (i: number) => crystalRadius() * (1.42 + 0.17 * i);

  let fade: THREE.MeshStandardMaterial | null = null;
  let shape = "";

  /** The rings' geometry: rebuilt only when their number, detail or size changes. */
  function buildRings(): void {
    for (const r of rings) {
      rig.remove(r.tilt);
      r.mesh.geometry.dispose();
    }
    for (const m of ringMaterials) m.dispose();
    ringMaterials.length = 0;
    rings = [];
    fade = null;
    const count = Math.ceil(p.rings);
    for (let i = 0; i < count; i++) {
      let material: THREE.Material = metal();
      if (i === count - 1) {
        // The newest ring fades in as its skill nears its mark.
        fade = metal().clone();
        fade.transparent = true;
        ringMaterials.push(fade);
        material = fade;
      }
      const mesh = new THREE.Mesh(
        gearRing({
          radius: ringRadius(i),
          width: 0.06,
          depth: 0.055,
          teeth: 24 + Math.round(36 * p.ringDetail),
          toothDepth: 0.04,
        }),
        material,
      );
      mesh.name = `ring-${i}`;
      const tilt = new THREE.Group();
      tilt.rotation.set(1.05 + 0.62 * i, 0, 0.5 * i + 0.3);
      tilt.add(mesh);
      rig.add(tilt);
      rings.push({ mesh, tilt, speed: 0, phase: i * 1.7 });
    }
  }

  /** Everything else follows the measurements in place: colors, glow, speeds, the fade. */
  function tune(): void {
    const radius = crystalRadius();
    crystal.scale.setScalar(radius);
    const fraction = p.rings - Math.floor(p.rings);
    if (fade) fade.opacity = fraction > 0 ? fraction : 1;
    rings.forEach((r, i) => {
      r.speed = (0.32 + 0.1 * i) * (0.3 + p.swirl) * (i % 2 === 0 ? 1 : -1);
    });
    U.uMain.value.set(p.colors[0] ?? "#ffffff");
    U.uSecond.value.set(p.colors.at(-1) ?? "#ffffff");
    U.uHeart.value = 0.45 + 0.75 * p.brightness;
    U.uRadius.value = radius;
    U.uBeat.value = 2.4 - 1.6 * p.smoothness;
    heartLight.color.set(p.colors[0] ?? "#ffffff");
    heartLight.intensity = 2 + 8 * p.brightness;
    halo.material.color.set(p.colors[0] ?? "#ffffff");
    halo.material.opacity = 0.25 + 0.45 * p.brightness;
    halo.scale.setScalar(radius * (3.2 + 2.6 * p.glowSoftness));
  }

  function build(): void {
    const next = `${Math.ceil(p.rings)}|${p.ringDetail.toFixed(3)}|${crystalRadius().toFixed(3)}`;
    if (next !== shape) {
      shape = next;
      buildRings();
    }
    tune();
  }
  build();

  // Drag to turn it; it settles back into a slow turn.
  let yaw = 0.6;
  let pitch = 0.18;
  let spin = 0;
  let dragging = false;

  function spawn(dt: number): void {
    const r = crystalRadius();
    const make = (rate: number, inward: boolean, palette: string[], shares: number[] | null) => {
      const n = rate * dt;
      for (let i = 0; i < Math.floor(n) + (Math.random() < n % 1 ? 1 : 0); i++) {
        const dir = new THREE.Vector3().randomDirection();
        const color = new THREE.Color(
          shares
            ? byShare(palette, shares)
            : (palette[Math.floor(Math.random() * palette.length)] ?? "#ffffff"),
        );
        flying.push(
          inward
            ? {
                pos: dir.clone().multiplyScalar(3.4),
                vel: dir.clone().multiplyScalar(-(1.6 + Math.random())),
                life: 2.2,
                color,
              }
            : {
                pos: dir.clone().multiplyScalar(r),
                vel: dir.clone().multiplyScalar(0.6 + Math.random() * 0.8),
                life: 1,
                color,
              },
        );
      }
    };
    make(p.sparkRate * 24, false, p.colors, null);
    if (p.intake) make(p.intake.rate * 60, true, p.intake.colors, p.intake.shares);
    if (p.arrivals) make((0.15 + p.arrivals.rate) * 20, true, p.arrivals.colors, p.arrivals.shares);
    if (flying.length > MOTES) flying = flying.slice(-MOTES);
  }

  function writeMotes(): void {
    for (let i = 0; i < MOTES; i++) {
      const m = flying[i];
      const k = i * 3;
      if (!m) {
        positions[k] = positions[k + 1] = positions[k + 2] = 0;
        colors[k] = colors[k + 1] = colors[k + 2] = 0;
        continue;
      }
      positions[k] = m.pos.x;
      positions[k + 1] = m.pos.y;
      positions[k + 2] = m.pos.z;
      const fade = Math.min(1, m.life);
      colors[k] = m.color.r * fade;
      colors[k + 1] = m.color.g * fade;
      colors[k + 2] = m.color.b * fade;
    }
    positionAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    motesGeo.setDrawRange(0, Math.min(MOTES, flying.length));
  }

  return {
    scene,
    camera,
    update(t, dt) {
      if (!still) {
        if (!dragging) {
          spin += dt * (0.18 + 0.25 * p.swirl);
          pitch += (0.18 - pitch) * Math.min(1, dt * 1.5);
        }
        for (const r of rings) {
          r.mesh.rotation.z += dt * r.speed;
          r.tilt.rotation.y = Math.sin(t * (0.3 + p.turbulence) + r.phase) * 0.12 * p.turbulence;
        }
        crystal.scale.setScalar(
          crystalRadius() *
            (1 + 0.025 * Math.sin(t * (2.4 - 1.6 * p.smoothness)) * (1 - 0.6 * p.smoothness)),
        );
        spawn(dt);
        const r = crystalRadius();
        flying = flying.filter((m) => {
          m.pos.addScaledVector(m.vel, dt);
          m.life -= dt * 0.8;
          const arrived = m.vel.dot(m.pos) < 0 && m.pos.length() < r;
          return m.life > 0 && !arrived;
        });
        // An AI that often slips flickers now and then.
        const dim = Math.random() < p.flicker * 0.03 ? 0.55 : 1;
        U.uDim.value = dim;
      }
      U.uTime.value = t;
      rig.rotation.set(pitch, yaw + spin, 0);
      writeMotes();
      return !still;
    },
    pointer(e) {
      if (e.type === "down") dragging = true;
      if (e.type === "up") dragging = false;
      if (e.type === "move") {
        yaw += e.dx * 0.01;
        pitch = Math.max(-1.1, Math.min(1.1, pitch + e.dy * 0.01));
      }
    },
    setParams(next) {
      p = next;
      build();
    },
    setStill(next) {
      still = next;
    },
    dispose() {
      for (const r of rings) r.mesh.geometry.dispose();
      for (const m of ringMaterials) m.dispose();
      crystal.geometry.dispose();
      heart.dispose();
      halo.material.dispose();
      glow.dispose();
      motesGeo.dispose();
      motesMat.dispose();
    },
  };
}
