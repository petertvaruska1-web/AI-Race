import type { WorldState } from "@airace/sim";
import { describe, expect, it } from "vitest";
import { identityOf, identityOfPack, makeIdentity } from "../src/jobs/identity.ts";

type P = { name: string; versionId: string; status: "countdown" | "live" | "retired" };

function world(parts: {
  company: string;
  versions: { id: string; label: string; packKey?: string }[];
  products: P[];
}): WorldState {
  return {
    companies: {
      c1: {
        name: parts.company,
        versions: Object.fromEntries(
          parts.versions.map((v) => [
            v.id,
            { id: v.id, label: v.label, packKey: v.packKey ?? null },
          ]),
        ),
        products: Object.fromEntries(parts.products.map((p, i) => [`p${i}`, p])),
      },
    },
  } as unknown as WorldState;
}

describe("an AI's name and company for a job", () => {
  it("names a launched AI by its product and a new one by its version", () => {
    const s = world({
      company: "Pengu Labs",
      versions: [
        { id: "v1", label: "Brain 1" },
        { id: "v2", label: "Brain 2" },
      ],
      products: [{ name: "Zula", versionId: "v1", status: "live" }],
    });
    expect(identityOf(s, "v1")).toEqual({ name: "Zula", company: "Pengu Labs" });
    expect(identityOf(s, "v2")).toEqual({ name: "Brain 2", company: "Pengu Labs" });
    expect(identityOf(s, "nope")).toBeUndefined();
  });

  it("prefers the product on sale, and skips a retired product's name", () => {
    const s = world({
      company: "Pengu Labs",
      versions: [{ id: "v1", label: "Brain 1" }],
      products: [
        { name: "Old", versionId: "v1", status: "retired" },
        { name: "Soon", versionId: "v1", status: "countdown" },
        { name: "Now", versionId: "v1", status: "live" },
      ],
    });
    expect(identityOf(s, "v1")?.name).toBe("Now");
    const retired = world({
      company: "Pengu Labs",
      versions: [{ id: "v1", label: "Brain 1" }],
      products: [{ name: "Old", versionId: "v1", status: "retired" }],
    });
    expect(identityOf(retired, "v1")?.name).toBe("Brain 1");
  });

  it("finds a rival's AI by its pack key, with the rival company's name", () => {
    const s = world({
      company: "Budget AI Inc",
      versions: [{ id: "rv", label: "v2", packKey: "budget-a-v2" }],
      products: [{ name: "Thrifty", versionId: "rv", status: "live" }],
    });
    expect(identityOfPack(s, "budget-a-v2")).toEqual({ name: "Thrifty", company: "Budget AI Inc" });
    expect(identityOfPack(s, "other")).toBeUndefined();
  });

  it("cuts names to 60 characters and sends nothing for an empty name", () => {
    expect(makeIdentity("x".repeat(80), "y".repeat(70))).toEqual({
      name: "x".repeat(60),
      company: "y".repeat(60),
    });
    expect(makeIdentity("", "Pengu")).toBeUndefined();
    expect(makeIdentity("   ", "Pengu")).toBeUndefined();
    expect(makeIdentity("Zula", "")).toEqual({ name: "Zula", company: null });
    expect(makeIdentity("Zula", undefined)).toEqual({ name: "Zula", company: null });
  });
});
