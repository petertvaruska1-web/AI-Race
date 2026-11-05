import { describe, expect, it } from "vitest";
import {
  BALANCE_V2,
  type CompanyState,
  previewPlan,
  projectView,
  step,
  upgradeState,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys } from "./pack.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "BE" } as const;
const PALETTE = { primary: "#3a86ff", secondary: "#ffbe0b" };
const c = (s: WorldState, id: string) => s.companies[id] as CompanyState;
const plan = (coaching: boolean) => ({
  ...BALANCE_V2.defaultPlan,
  mix: { ...BALANCE_V2.defaultPlan.mix },
  coaching,
});

describe("lessons", () => {
  it("are counted in the world, for players' companies only, and shown only to their owner", () => {
    let s = runningWorld();
    s = ok(step(s, sys({ type: "lessonsCounted", companyId: "c1", count: 7 })));
    expect(c(s, "c1").lessons).toBe(7);
    s = ok(step(s, sys({ type: "memberJoined", userId: "u2", displayName: "Ben" })));
    s = ok(
      step(
        s,
        cmd({ type: "createCompany", name: "Beta", logo: LOGO, palette: PALETTE }, "c2", "u2"),
      ),
    );
    expect(projectView(s, "u1").me.company?.lessons).toBe(7);
    expect(projectView(s, "u2").me.company?.lessons).toBe(0);
    expect(JSON.stringify(projectView(s, "u2").companies)).not.toContain('"lessons"');
    for (const bad of [-1, 201, 2.5]) {
      expect(rejected(step(s, sys({ type: "lessonsCounted", companyId: "c1", count: bad })))).toBe(
        "That isn't valid.",
      );
    }
    expect(rejected(step(s, sys({ type: "lessonsCounted", companyId: "nope", count: 1 })))).toBe(
      "That isn't valid.",
    );
  });

  it("a plan can train on them from 5", () => {
    const four = previewPlan(BALANCE_V2, null, plan(true), 0, 0, 4);
    expect(four).toEqual({
      ok: false,
      reason: "Your AI needs at least 5 lessons first. You have 4.",
    });
    expect(previewPlan(BALANCE_V2, null, plan(true), 0, 0, 5).ok).toBe(true);
    expect(previewPlan(BALANCE_V2, null, plan(false), 0, 0, 0).ok).toBe(true);
    // Free: coaching doesn't change the price.
    const on = previewPlan(BALANCE_V2, null, plan(true), 0, 0, 5);
    const off = previewPlan(BALANCE_V2, null, plan(false), 0, 0, 5);
    if (!on.ok || !off.ok) throw new Error("preview failed");
    expect(on.preview.cost).toBe(off.preview.cost);
  });

  it("training refuses coaching under 5 lessons and records the switch on the run", () => {
    let s = runningWorld();
    expect(rejected(step(s, cmd({ type: "startTraining", plan: plan(true) }, "r1")))).toBe(
      "Your AI needs at least 5 lessons first. You have 0.",
    );
    s = ok(step(s, sys({ type: "lessonsCounted", companyId: "c1", count: 5 })));
    s = ok(step(s, cmd({ type: "startTraining", plan: plan(true) }, "r1")));
    expect(c(s, "c1").runs.r1?.plan.coaching).toBe(true);
  });

  it("an older world comes up with no lessons and plans without coaching", () => {
    const s = structuredClone(runningWorld()) as unknown as Record<string, unknown>;
    const companies = s.companies as Record<string, Record<string, unknown>>;
    for (const co of Object.values(companies)) delete co.lessons;
    const up = upgradeState(s, "sim-7");
    expect(c(up, "c1").lessons).toBe(0);
  });
});
