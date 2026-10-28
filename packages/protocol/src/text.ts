import { z } from "zod";

/**
 * Trims and removes C0 control characters (U+0000–U+001F) and DEL (U+007F).
 * With `allowNewlines`, "\n" survives (chat). Other marks, bidi included, are
 * kept: clients render all player and model text as text (spec §11).
 */
export function cleanText(s: string, opts: { allowNewlines?: boolean } = {}): string {
  let out = "";
  for (const ch of s) {
    const c = ch.charCodeAt(0);
    const control = c === 0x7f || (c < 0x20 && !(opts.allowNewlines && c === 0x0a));
    if (!control) out += ch;
  }
  return out.trim();
}

/** A string cleaned by `cleanText`, then length-checked. */
export function zText(min: number, max: number, opts: { allowNewlines?: boolean } = {}) {
  return z
    .string()
    .overwrite((s) => cleanText(s, opts))
    .min(min)
    .max(max);
}

/** A single-line name. */
export function zName(min: number, max: number) {
  return zText(min, max);
}

/** Validated lengths of untrusted text (plan, Global Constraints). */
export const TEXT_LIMITS = {
  companyName: [2, 32],
  productName: [2, 24],
  displayName: [1, 24],
  worldName: [2, 40],
  probePrompt: [0, 120],
  chatMessage: [1, 500],
} as const;

/** Names every plain object already has (`__proto__`, `constructor`, …). */
const BUILT_IN_NAMES = new Set(Object.getOwnPropertyNames(Object.prototype));

/**
 * Server-made identifiers: UUIDs in production, short names in tests. A built-in
 * object name is refused: a lookup by it would find the object's own machinery
 * instead of an entry (and `__proto__` would reach every object).
 */
export const zId = z
  .string()
  .min(1)
  .max(64)
  .regex(/^[A-Za-z0-9_-]+$/)
  .refine((id) => !BUILT_IN_NAMES.has(id), "That id is reserved.");

export const Day = z.number().int().min(0);
