import { z } from "zod";

export type PatchPath = (string | number)[];
export type PatchOp =
  | { op: "set"; path: PatchPath; value: unknown }
  | { op: "del"; path: PatchPath };

const Path = z.array(z.union([z.string(), z.number().int().min(0)]));
export const PatchOp: z.ZodType<PatchOp> = z.discriminatedUnion("op", [
  z.object({ op: z.literal("set"), path: Path, value: z.unknown() }),
  z.object({ op: z.literal("del"), path: Path }),
]) as z.ZodType<PatchOp>;

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * The ops that turn JSON value `a` into `b`. Plain objects are compared key by
 * key; arrays element by element only when their lengths match, otherwise the
 * whole array is set.
 */
export function diff(a: unknown, b: unknown): PatchOp[] {
  const out: PatchOp[] = [];
  walk(a, b, [], out);
  return out;
}

function walk(a: unknown, b: unknown, path: PatchPath, out: PatchOp[]): void {
  if (Object.is(a, b)) return;
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    for (let i = 0; i < a.length; i++) walk(a[i], b[i], [...path, i], out);
    return;
  }
  if (isPlainObject(a) && isPlainObject(b)) {
    for (const key of Object.keys(a)) {
      if (!Object.hasOwn(b, key)) out.push({ op: "del", path: [...path, key] });
    }
    for (const key of Object.keys(b)) {
      if (Object.hasOwn(a, key)) walk(a[key], b[key], [...path, key], out);
      else out.push({ op: "set", path: [...path, key], value: b[key] });
    }
    return;
  }
  out.push({ op: "set", path, value: b });
}

/** Applies ops without changing `doc`; untouched branches are shared. */
export function applyPatch<T>(doc: T, ops: readonly PatchOp[]): T {
  let out: unknown = doc;
  for (const op of ops) out = applyAt(out, op, 0);
  return out as T;
}

function applyAt(node: unknown, op: PatchOp, depth: number): unknown {
  if (depth === op.path.length) {
    if (op.op === "del") throw new Error("cannot delete the whole document");
    return op.value;
  }
  const key = op.path[depth] as string | number;
  const last = depth === op.path.length - 1;
  if (Array.isArray(node) && typeof key === "number" && key < node.length) {
    const copy = node.slice();
    if (op.op === "del" && last) copy.splice(key, 1);
    else copy[key] = applyAt(node[key], op, depth + 1);
    return copy;
  }
  if (isPlainObject(node) && typeof key === "string") {
    if (!last && !Object.hasOwn(node, key))
      throw new Error(`patch path not found: ${op.path.join("/")}`);
    const copy: Record<string, unknown> = {};
    for (const k of Object.keys(node)) {
      if (op.op === "del" && last && k === key) continue;
      define(copy, k, node[k]);
    }
    if (!(op.op === "del" && last)) define(copy, key, applyAt(node[key], op, depth + 1));
    return copy;
  }
  throw new Error(`patch path not found: ${op.path.join("/")}`);
}

// Plain assignment of "__proto__" would replace the prototype.
function define(obj: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(obj, key, { value, writable: true, enumerable: true, configurable: true });
}
