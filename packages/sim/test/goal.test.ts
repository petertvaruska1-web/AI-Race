import { GoalSpec } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  companyValue,
  createWorld,
  goalDay,
  goalProgress,
  type NewsItem,
  projectView,
  step,
  upgradeState,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, SYNTHETIC_PACK, sys } from "./pack.ts";

function world(spec: unknown, rivals = false): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Goals",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 7,
    balance: "balance-v2",
    inviteCode: "INV",
    goal: GoalSpec.parse(spec),
  });
  s = ok(
    step(
      s,
      cmd(
        {
          type: "createCompany",
          name: "Acme",
          logo: { shape: "circle", symbol: "spark", initials: "AC" },
          palette: { primary: "#ff9f1c", secondary: "#2ec4b6" },
        },
        "c1",
      ),
    ),
  );
  s = ok(step(s, cmd({ type: "startWorld" })));
  return rivals ? ok(step(s, sys({ type: "rivalsJoined", pack: SYNTHETIC_PACK }))) : s;
}

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const customers = (s: WorldState, id: string, n: number) => {
  s.market.customers[id] = { newEngland: { everyday: { free: n } } };
};
const goalNews = (s: WorldState) =>
  s.feed.filter((f): f is NewsItem => f.kind === "news" && f.topic === "goal");
const report = (profit: number) => ({ month: "2027-02", profit, grade: "C" }) as never;

describe("the goal", () => {
  it("is part of the world from the start, with its deadline", () => {
    const s = world({ goal: { kind: "customers", target: 1_000 }, deadlineYears: 2 });
    expect(s.goal.spec.goal).toEqual({ kind: "customers", target: 1_000 });
    expect(s.goal.deadlineDay).toBe(s.day + 730);
    expect(world({ goal: { kind: "none" }, deadlineYears: null }).goal.deadlineDay).toBeNull();
    expect(
      GoalSpec.safeParse({ goal: { kind: "customers", target: 10 }, deadlineYears: null }).success,
    ).toBe(false);
    expect(GoalSpec.safeParse({ goal: { kind: "value" }, deadlineYears: 3 }).success).toBe(false);
  });

  it("measures each kind of progress", () => {
    const s = world({ goal: { kind: "customers", target: 1_000 }, deadlineYears: null }, true);
    customers(s, "c1", 400);
    expect(goalProgress(s, c1(s))).toEqual({ value: 400, target: 1_000, done: false });
    s.goal.spec.goal = { kind: "profit", target: 5_000 };
    c1(s).reports.push(report(6_000));
    expect(goalProgress(s, c1(s))).toEqual({ value: 6_000, target: 5_000, done: true });
    s.goal.spec.goal = { kind: "beatArch" };
    const arch = s.rivals?.archId as string;
    customers(s, arch, 900);
    expect(goalProgress(s, c1(s))).toEqual({ value: 400, target: 900, done: false });
    customers(s, "c1", 901);
    expect(goalProgress(s, c1(s)).done).toBe(true);
    s.goal.spec.goal = { kind: "bestAi" };
    expect(goalProgress(s, c1(s)).done).toBe(false); // nothing on sale
  });

  it("values only the founders' share, never below zero", () => {
    const s = world({ goal: { kind: "value", target: 100_000 }, deadlineYears: null });
    const c = c1(s);
    c.cash = 100_000;
    c.fund.balance = 20_000;
    c.reports.push(report(1_000), report(2_000), report(3_000), report(-90_000));
    // The last three months average to −28,333: no future profit counted.
    expect(companyValue(s, c)).toBe(120_000);
    c.reports.push(report(91_000)); // last three: 3,000, −90,000, 91,000 → 1,333.33
    expect(companyValue(s, c)).toBe(Math.round(120_000 + 24 * (4_000 / 3)));
    c.ownership = 0.6;
    expect(companyValue(s, c)).toBe(Math.round(0.6 * (120_000 + 24 * (4_000 / 3))));
    c.cash = -500_000;
    expect(companyValue(s, c)).toBe(0);
  });

  it("is reached once, with one story, even on its deadline day", () => {
    const s = world({ goal: { kind: "customers", target: 1_000 }, deadlineYears: 1 });
    s.day = s.goal.deadlineDay as number;
    customers(s, "c1", 1_000);
    goalDay(s);
    expect(s.goal.reachedDay).toBe(s.day);
    expect(s.goal.reachedBy).toBe("c1");
    expect(goalNews(s).map((n) => [n.headline, n.sentiment])).toEqual([
      ["Acme reaches its goal: 1,000 customers", 0.8],
    ]);
    s.day += 1;
    goalDay(s);
    expect(goalNews(s)).toHaveLength(1);
    expect(s.goal.missedDay).toBeNull();
  });

  it("tells a missed deadline once, and play goes on", () => {
    const s = world({ goal: { kind: "customers", target: 1_000 }, deadlineYears: 1 });
    s.day = (s.goal.deadlineDay as number) + 1;
    goalDay(s);
    goalDay(s);
    expect(s.goal.missedDay).toBe(s.day);
    expect(goalNews(s).map((n) => n.headline)).toEqual(["Acme misses its goal deadline"]);
    customers(s, "c1", 5_000);
    s.day += 1;
    goalDay(s);
    expect(s.goal.reachedDay).toBeNull(); // too late to count
  });

  it("a rival goal never completes in a world without rivals", () => {
    const s = world({ goal: { kind: "beatArch" }, deadlineYears: null });
    customers(s, "c1", 1_000_000);
    goalDay(s);
    expect(s.goal.reachedDay).toBeNull();
    expect(goalProgress(s, c1(s)).done).toBe(false);
  });

  it("shows in the view: the world's goal and the player's progress", () => {
    const s = world({ goal: { kind: "customers", target: 1_000 }, deadlineYears: 5 });
    customers(s, "c1", 10);
    const view = projectView(s, "u1");
    expect(view.world.goal).toEqual({
      kind: "customers",
      target: 1_000,
      deadlineDay: s.day + 5 * 365,
      reachedDay: null,
      missedDay: null,
      reachedBy: null,
    });
    expect(view.me.company?.goalProgress).toEqual({ value: 10, target: 1_000, done: false });
    const none = world({ goal: { kind: "none" }, deadlineYears: null });
    expect(projectView(none, "u1").world.goal).toBeNull();
    expect(projectView(none, "u1").me.company?.goalProgress).toBeNull();
  });

  it("an upgraded world has no goal", () => {
    const s = world({ goal: { kind: "none" }, deadlineYears: null });
    const old = structuredClone(s) as unknown as Record<string, unknown>;
    delete old.goal;
    const up = upgradeState(old, "sim-5");
    expect(up.goal.spec.goal.kind).toBe("none");
    expect(projectView(up, "u1").world.goal).toBeNull();
  });
});
