import * as THREE from "three";

/**
 * A machined gear ring in the XY plane, centered on the origin: a flat ring of `width` around
 * `radius`, `depth` thick, with `teeth` square teeth of `toothDepth` on its outer edge.
 */
export function gearRing(o: {
  radius: number;
  width: number;
  depth: number;
  teeth: number;
  toothDepth: number;
}): THREE.BufferGeometry {
  const outer = o.radius + o.width / 2;
  const inner = o.radius - o.width / 2;
  const tip = outer + o.toothDepth;
  const shape = new THREE.Shape();
  const step = (Math.PI * 2) / o.teeth;
  for (let i = 0; i < o.teeth; i++) {
    const a = i * step;
    // A tooth fills the middle half of its step: base, rise, top, fall.
    const points: [number, number][] = [
      [a, outer],
      [a + step * 0.25, outer],
      [a + step * 0.3, tip],
      [a + step * 0.7, tip],
      [a + step * 0.75, outer],
    ];
    points.forEach(([angle, r], k) => {
      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r;
      if (i === 0 && k === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    });
  }
  shape.closePath();
  const hole = new THREE.Path();
  hole.absarc(0, 0, inner, 0, Math.PI * 2, true);
  shape.holes.push(hole);
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: o.depth,
    bevelEnabled: true,
    bevelThickness: o.depth * 0.22,
    bevelSize: Math.min(o.width, o.toothDepth) * 0.22,
    bevelSegments: 1,
    curveSegments: Math.max(24, o.teeth * 2),
  });
  geometry.translate(0, 0, -o.depth / 2);
  geometry.computeVertexNormals();
  return geometry;
}
