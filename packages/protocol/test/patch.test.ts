import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { applyPatch, diff } from "../src/index.ts";

describe("diff and applyPatch", () => {
  it("rebuilds b from a for arbitrary JSON values", () => {
    fc.assert(
      fc.property(fc.jsonValue(), fc.jsonValue(), (a, b) => {
        expect(applyPatch(a, diff(a, b))).toEqual(b);
      }),
      { numRuns: 500 },
    );
  });

  it("finds no change between a value and itself", () => {
    fc.assert(
      fc.property(fc.jsonValue(), (a) => {
        expect(diff(a, a)).toEqual([]);
        expect(diff(a, structuredClone(a))).toEqual([]);
      }),
    );
  });

  it("never changes its input", () => {
    fc.assert(
      fc.property(fc.jsonValue(), fc.jsonValue(), (a, b) => {
        const before = structuredClone(a);
        applyPatch(a, diff(a, b));
        expect(a).toEqual(before);
      }),
    );
  });

  it("sets a whole array whose length changed", () => {
    expect(diff({ xs: [1, 2] }, { xs: [1, 2, 3] })).toEqual([
      { op: "set", path: ["xs"], value: [1, 2, 3] },
    ]);
  });

  it("diffs equal-length arrays element by element", () => {
    expect(diff({ xs: [1, { a: 2 }] }, { xs: [1, { a: 3 }] })).toEqual([
      { op: "set", path: ["xs", 1, "a"], value: 3 },
    ]);
  });

  it("deletes a removed key and sets an added one", () => {
    expect(diff({ a: 1, b: 2 }, { b: 2, c: 3 })).toEqual([
      { op: "del", path: ["a"] },
      { op: "set", path: ["c"], value: 3 },
    ]);
  });

  it("replaces the whole document at the root", () => {
    expect(diff(1, "x")).toEqual([{ op: "set", path: [], value: "x" }]);
    expect(applyPatch(1, [{ op: "set", path: [], value: "x" }])).toBe("x");
  });

  it("throws on a path that does not exist", () => {
    expect(() => applyPatch({ a: 1 }, [{ op: "set", path: ["b", "c"], value: 1 }])).toThrow();
  });

  it("stores a __proto__ key as data, not as the prototype", () => {
    const b = JSON.parse('{"__proto__": {"polluted": true}}');
    const out = applyPatch({}, diff({}, b)) as Record<string, unknown>;
    expect(Object.getPrototypeOf(out)).toBe(Object.prototype);
    expect(Object.hasOwn(out, "__proto__")).toBe(true);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });
});
