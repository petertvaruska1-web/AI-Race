import { DEPARTMENTS } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  departmentStrength,
  projectView,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, TICK, ticks } from "./pack.ts";

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const fund = (s: WorldState, department: string, budget: number) =>
  step(s, cmd({ type: "setDepartment", department, budget }));

describe("departments", () => {
  it("start unfunded, and every company has all four", () => {
    const s = runningWorld();
    for (const d of DEPARTMENTS) {
      expect(c1(s).departments[d]).toEqual({ budget: 0, staffed: 0 });
      expect(departmentStrength(c1(s), d)).toBe(0);
    }
  });

  it("a budget of $300,000 a month books $10,000 a day as Team", () => {
    const s = ok(fund(runningWorld(), "research", 300_000));
    const before = c1(s).ledger.current.costs.team;
    const cash = c1(s).cash;
    const next = ok(step(s, TICK));
    expect(c1(next).ledger.current.costs.team - before).toBe(10_000);
    expect(cash - c1(next).cash).toBeGreaterThanOrEqual(10_000);
  });

  it("a team grows toward its budget over about a month, with diminishing returns", () => {
    const s = ticks(ok(fund(runningWorld(), "data", 300_000)), 30);
    const staffed = c1(s).departments.data.staffed;
    expect(staffed / 300_000).toBeCloseTo(1 - (29 / 30) ** 30, 6); // about 63%
    expect(departmentStrength(c1(s), "data")).toBeCloseTo(1 - Math.exp(-staffed / 200_000), 9);
  });

  it("a cut to $0 halves the team in about 21 days; it doesn't vanish at once", () => {
    let s = ticks(ok(fund(runningWorld(), "sales", 200_000)), 60);
    const staffed = c1(s).departments.sales.staffed;
    s = ok(fund(s, "sales", 0));
    s = ok(step(s, TICK));
    expect(c1(s).departments.sales.staffed).toBeCloseTo(staffed * (29 / 30), 6);
    s = ticks(s, 20);
    expect(c1(s).departments.sales.staffed / staffed).toBeCloseTo((29 / 30) ** 21, 6);
    expect((29 / 30) ** 21).toBeCloseTo(0.49, 2);
  });

  it("refuses an off-step budget, too much, too little and an unknown department", () => {
    const s = runningWorld();
    const copy = "Budgets are $0–$500,000 a month, in $10,000 steps.";
    expect(rejected(fund(s, "research", 15_000))).toBe(copy);
    expect(rejected(fund(s, "research", 510_000))).toBe(copy);
    expect(rejected(fund(s, "research", -10_000))).toBe(copy);
    expect(rejected(fund(s, "lawyers", 10_000))).toBe("That request isn't valid.");
    expect(ok(fund(s, "support", 500_000)).companies.c1?.departments.support.budget).toBe(500_000);
  });

  it("pauses an absent owner's budgets on a day its cash is negative; they resume on return", () => {
    let s = ticks(ok(fund(runningWorld(), "research", 300_000)), 10);
    s = ok(step(s, sys({ type: "presence", userId: "u1", online: false })));
    c1(s).cash = -10;
    const team = c1(s).ledger.current.costs.team;
    const staffed = c1(s).departments.research.staffed;
    s = ok(step(s, TICK));
    expect(c1(s).ledger.current.costs.team).toBe(team); // no spend
    expect(c1(s).departments.research.staffed).toBeLessThan(staffed); // the team shrinks
    expect(c1(s).departments.research.budget).toBe(300_000); // the budget is kept
    s = ok(step(s, sys({ type: "presence", userId: "u1", online: true })));
    s = ok(step(s, TICK));
    expect(c1(s).ledger.current.costs.team).toBe(team + 10_000);
  });

  it("shows only the owner its departments, with each one's strength", () => {
    const s = ticks(ok(fund(runningWorld(), "research", 100_000)), 5);
    const view = projectView(s, "u1").me.company?.departments;
    expect(view?.research.budget).toBe(100_000);
    expect(view?.research.staffed).toBe(Math.round(c1(s).departments.research.staffed));
    expect(view?.research.strength).toBeCloseTo(departmentStrength(c1(s), "research"), 9);
    expect(view?.data).toEqual({ budget: 0, staffed: 0, strength: 0 });
    const joined = ok(step(s, sys({ type: "memberJoined", userId: "u2", displayName: "Ben" })));
    const other = projectView(joined, "u2");
    expect(JSON.stringify(other)).not.toContain('"staffed"');
  });
});
