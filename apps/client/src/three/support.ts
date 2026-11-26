// Whether this browser can draw with WebGL. Everything that draws with it has a flat stand-in,
// so a computer without it (or the unit tests' DOM) still plays the whole game.

const cache = new Map<1 | 2, boolean>();

/** WebGL 2 for the 3D objects (three.js needs it); WebGL 1 is enough for the glass. */
export function webglAvailable(version: 1 | 2 = 2): boolean {
  const known = cache.get(version);
  if (known !== undefined) return known;
  let ok = false;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext(version === 2 ? "webgl2" : "webgl");
    ok = Boolean(gl);
    (gl as WebGLRenderingContext | null)?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    ok = false;
  }
  cache.set(version, ok);
  return ok;
}
