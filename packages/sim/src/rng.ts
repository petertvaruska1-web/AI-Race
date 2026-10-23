/** sfc32 state, four unsigned 32-bit integers (JSON-safe, kept in WorldState). */
export type RngState = { a: number; b: number; c: number; d: number };

function splitmix32(x: number): { value: number; next: number } {
  const next = (x + 0x9e3779b9) | 0;
  let z = next;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
  z ^= z >>> 16;
  return { value: z >>> 0, next };
}

/** Expands a seed into sfc32 state with splitmix32, then warms it up. */
export function seedRng(seed: number): RngState {
  let x = seed | 0;
  const words: number[] = [];
  for (let i = 0; i < 4; i++) {
    const r = splitmix32(x);
    words.push(r.value);
    x = r.next;
  }
  let s: RngState = { a: words[0] ?? 0, b: words[1] ?? 0, c: words[2] ?? 0, d: words[3] ?? 0 };
  for (let i = 0; i < 12; i++) s = rngNext(s).next;
  return s;
}

/** A value in [0, 1) and the next state; the given state is unchanged. */
export function rngNext(s: RngState): { value: number; next: RngState } {
  let { a, b, c, d } = s;
  const t = (((a + b) | 0) + d) | 0;
  d = (d + 1) | 0;
  a = b ^ (b >>> 9);
  b = (c + (c << 3)) | 0;
  c = (c << 21) | (c >>> 11);
  c = (c + t) | 0;
  return {
    value: (t >>> 0) / 4294967296,
    next: { a: a >>> 0, b: b >>> 0, c: c >>> 0, d: d >>> 0 },
  };
}

/** An integer in [0, n). */
export function rngInt(s: RngState, n: number): { value: number; next: RngState } {
  if (!Number.isInteger(n) || n <= 0) throw new Error(`rngInt needs a positive integer, got ${n}`);
  const r = rngNext(s);
  return { value: Math.floor(r.value * n), next: r.next };
}
