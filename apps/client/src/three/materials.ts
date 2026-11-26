import * as THREE from "three";

// The bench's shared 3D materials (design §8.0): machined metal, enamel, and things that glow.
// Cached by their arguments so every object of the same finish shares one material.

const cache = new Map<string, THREE.Material>();

function cached<T extends THREE.Material>(key: string, make: () => T): T {
  const hit = cache.get(key);
  if (hit) return hit as T;
  const m = make();
  cache.set(key, m);
  return m;
}

/** Bright machined metal (gears, frames, bolts). */
export function metal(): THREE.MeshStandardMaterial {
  return cached(
    "metal",
    () =>
      new THREE.MeshStandardMaterial({
        color: 0x8f959e,
        metalness: 1,
        roughness: 0.46,
        envMapIntensity: 0.75,
      }),
  );
}

/** Dark anodized metal (cradles, housings). */
export function darkMetal(): THREE.MeshStandardMaterial {
  return cached(
    "dark-metal",
    () => new THREE.MeshStandardMaterial({ color: 0x3a3f47, metalness: 1, roughness: 0.45 }),
  );
}

/** Glossy enamel in a company's color (emblem faces, map tops). */
export function enamel(color: string): THREE.MeshPhysicalMaterial {
  return cached(
    `enamel:${color}`,
    () =>
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(color),
        metalness: 0,
        roughness: 0.35,
        clearcoat: 1,
        clearcoatRoughness: 0.12,
      }),
  );
}

/** A dark surface whose light is its own: lamps, streams, glowing marks. */
export function glowing(color: string, intensity = 2.5): THREE.MeshStandardMaterial {
  return cached(
    `glow:${color}:${intensity}`,
    () =>
      new THREE.MeshStandardMaterial({
        color: 0x101010,
        emissive: new THREE.Color(color),
        emissiveIntensity: intensity,
        metalness: 0,
        roughness: 0.4,
      }),
  );
}
