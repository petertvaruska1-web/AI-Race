/** JSON with object keys sorted, so equal values give equal strings. */
export function stableStringify(v: unknown): string {
  if (Array.isArray(v)) {
    return `[${v.map((x) => (x === undefined ? "null" : stableStringify(x))).join(",")}]`;
  }
  if (v !== null && typeof v === "object") {
    const obj = v as Record<string, unknown>;
    const keys = Object.keys(obj)
      .filter((k) => obj[k] !== undefined)
      .sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`).join(",")}}`;
  }
  return JSON.stringify(v) ?? "null";
}

/** A 64-bit cyrb53-style hash of `stableStringify(v)`, as 16 hex characters. */
export function stateHash(v: unknown): string {
  const str = stableStringify(v);
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h2 >>> 0).toString(16).padStart(8, "0") + (h1 >>> 0).toString(16).padStart(8, "0");
}
