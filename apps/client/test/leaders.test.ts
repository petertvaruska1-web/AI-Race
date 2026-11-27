import { REGION_IDS } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import { companyTops, leaderTops } from "../src/regions.ts";
import { readableOn } from "../src/theme/glass.ts";
import { apply, launchedWorld, rivalWorld, runningWorld, ticks, viewOf } from "./views.ts";

describe("who leads each region (the map's tops, the same for everybody)", () => {
  it("leaves every region plain glass before anyone has customers", () => {
    const tops = leaderTops(viewOf(runningWorld()));
    for (const r of REGION_IDS) expect(tops[r]).toEqual({ color: null, height: 0, leaderId: null });
  });

  it("tops each region with its leader, the same for every player", () => {
    let s = ticks(rivalWorld(), 30);
    s = apply(s, {
      kind: "system",
      event: { type: "memberJoined", userId: "u2", displayName: "Ben" },
    });
    const tops = leaderTops(viewOf(s, "u1"));
    expect(tops).toEqual(leaderTops(viewOf(s, "u2")));
    const view = viewOf(s);
    for (const r of REGION_IDS) {
      const lead = [...view.companies].sort(
        (a, b) =>
          b.customersByRegion[r] - a.customersByRegion[r] ||
          (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
      )[0];
      expect(tops[r].leaderId, r).toBe(lead && lead.customersByRegion[r] > 0 ? lead.id : null);
      if (lead && tops[r].leaderId)
        expect(tops[r].color).toBe(readableOn(lead.palette.primary, "dark"));
    }
    const led = REGION_IDS.filter((r) => tops[r].leaderId !== null).map((r) => tops[r].height);
    expect(led.length).toBeGreaterThan(0);
    expect(Math.max(...led)).toBe(1);
    expect(Math.min(...led)).toBeGreaterThanOrEqual(0.25);
  });

  it("breaks a tie the same way for every player, whatever their language", () => {
    const view = viewOf(launchedWorld());
    const [first, ...rest] = view.companies;
    if (!first) throw new Error("no company");
    const twin = { ...first, id: "B-twin", palette: { primary: "#ff9f1c", secondary: "#4f8cff" } };
    view.companies = [{ ...first, id: "a-first" }, twin, ...rest];
    const tied = REGION_IDS.filter((r) => first.customersByRegion[r] > 0);
    expect(tied.length).toBeGreaterThan(0);
    // By character code, "B" comes before "a" in every language; by most languages' rules it doesn't.
    for (const r of tied) expect(leaderTops(view)[r].leaderId, r).toBe("B-twin");
  });

  it("lifts a very dark leader's color so its top shows in the dark map", () => {
    const view = viewOf(launchedWorld());
    const acme = view.companies.find((c) => c.id === "c1");
    if (!acme) throw new Error("no Acme");
    acme.palette = { primary: "#101010", secondary: "#ffb020" };
    const tops = leaderTops(view);
    const led = REGION_IDS.filter((r) => tops[r].leaderId === "c1");
    expect(led.length).toBeGreaterThan(0);
    expect(readableOn("#101010", "dark")).not.toBe("#101010");
    for (const r of led) expect(tops[r].color).toBe(readableOn("#101010", "dark"));
  });

  it("shows one company's own regions for its window", () => {
    const view = viewOf(launchedWorld());
    const tops = companyTops(view, "c1");
    const acme = view.companies.find((c) => c.id === "c1");
    if (!acme) throw new Error("no Acme");
    for (const r of REGION_IDS) {
      if (acme.customersByRegion[r] > 0) {
        expect(tops[r].color, r).toBe(readableOn(acme.palette.primary, "dark"));
        expect(tops[r].height).toBeGreaterThanOrEqual(0.25);
        expect(tops[r].leaderId).toBe("c1");
      } else expect(tops[r], r).toEqual({ color: null, height: 0, leaderId: null });
    }
    expect(Math.max(...REGION_IDS.map((r) => tops[r].height))).toBe(1);
  });
});
