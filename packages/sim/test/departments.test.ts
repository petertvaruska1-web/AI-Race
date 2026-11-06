import { DEPARTMENTS, type Department } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  awarenessOf,
  BALANCE_TEST,
  BALANCE_V2,
  type CompanyState,
  departmentStrength,
  departmentsDay,
  previewPlan,
  previewTraining,
  projectView,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, TICK, ticks, withRivals } from "./pack.ts";

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

// -- the four effects (Task 2) --------------------------------------------------------------

/** A steady team: budget and staff equal, so it neither grows nor shrinks. 140k ≈ 0.503. */
const STEADY = 140_000;
const S = 1 - Math.exp(-STEADY / 200_000);
function staff(s: WorldState, d: Department, dollars = STEADY): WorldState {
  const c = c1(s);
  c.departments[d] = { budget: dollars, staffed: dollars };
  return s;
}

const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
};

/** c1 has "Spark" (v1) live on free and personal, on the fast test balance. */
function live(): WorldState {
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  let s = ok(step(runningWorld("balance-test"), cmd({ type: "startTraining", plan }, "r1")));
  s = ok(
    step(
      s,
      sys({
        type: "trainingCompleted",
        runId: "r1",
        versionId: "v1",
        result: { steps: 8, finalLoss: 4, heldoutLosses: {}, unstable: false },
      }),
    ),
  );
  s = ticks(s, (c1(s).runs.r1?.endDay ?? 0) - s.day);
  s = ok(
    step(
      s,
      sys({
        type: "evalCompleted",
        versionId: "v1",
        scores: SCORES,
        overall: 42,
        personalityWords: [],
      }),
    ),
  );
  s = ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free", "personal"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
  return ticks(s, 3);
}

describe("what departments do", () => {
  const plan = () => ({ ...BALANCE_V2.defaultPlan, mix: { ...BALANCE_V2.defaultPlan.mix } });

  it("Research: a run reads 1 + 0.3·s times the text for the same money and time", () => {
    const s = runningWorld();
    const base = previewTraining(s, "c1", plan());
    const more = previewTraining(staff(structuredClone(s), "research"), "c1", plan());
    if (!base.ok || !more.ok) throw new Error("no preview");
    expect(more.preview.tokens).toBe(Math.round(base.preview.tokens * (1 + 0.3 * S)));
    expect(more.preview.cost).toBe(base.preview.cost);
    expect(more.preview.durationDays).toBe(base.preview.durationDays);
    expect(previewPlan(BALANCE_V2, null, plan(), 0, 0, 0, { research: 0.5 })).toMatchObject({
      preview: { tokens: Math.round((base.preview.tokens as number) * 1.15) },
    });
  });

  it("Data team: checks cost 1 − 0.5·s of their price", () => {
    const checked = { ...plan(), factCheck: true };
    const base = previewPlan(BALANCE_V2, null, checked, 0, 0, 0);
    const half = previewPlan(BALANCE_V2, null, checked, 0, 0, 0, { data: 0.5 });
    if (!base.ok || !half.ok) throw new Error("no preview");
    expect(half.preview.checks).toBe(Math.round(BALANCE_V2.factCheckCost * 0.75));
    expect(base.preview.cost - half.preview.cost).toBe(
      (base.preview.checks ?? 0) - (half.preview.checks ?? 0),
    );
    const s = staff(runningWorld(), "data");
    const mine = previewTraining(s, "c1", checked);
    expect(mine.ok && mine.preview.checks).toBe(
      Math.round(BALANCE_V2.factCheckCost * (1 - 0.5 * S)),
    );
  });

  it("Sales & marketing: awareness rises by 0.002·s·(1 − a) a day everywhere", () => {
    const s = staff(live(), "sales");
    const before = awarenessOf(s.market, "c1", "pacific", "students");
    const none = awarenessOf(s.market, "c1", "newEngland", "researchers");
    departmentsDay(s);
    expect(awarenessOf(s.market, "c1", "pacific", "students")).toBeCloseTo(
      before + 0.002 * S * (1 - before),
      12,
    );
    expect(awarenessOf(s.market, "c1", "newEngland", "researchers")).toBeCloseTo(
      none + 0.002 * S * (1 - none),
      12,
    );
  });

  it("Sales & marketing does nothing before there is an AI on sale", () => {
    const s = staff(runningWorld(), "sales");
    departmentsDay(s);
    expect(awarenessOf(s.market, "c1", "pacific", "students")).toBe(0);
  });

  it("Support & operations: a drop is 1 − 0.3·s as large, a rise unchanged", () => {
    const start = BALANCE_TEST.economy.feedback.satisfactionStart;
    const batch = [
      { group: "students", questionId: "q1", question: "Q?", reply: "A.", score: 1 },
      { group: "developers", questionId: "q2", question: "Q?", reply: "A.", score: 0 },
    ];
    const scored = sys({ type: "feedbackScored", productId: "p1", key: "w1", results: batch });
    const plain = ok(step(live(), scored));
    const helped = ok(step(staff(live(), "support"), scored));
    const sat = (w: WorldState, g: string) =>
      (w.market.satisfaction.p1 as Record<string, number> | undefined)?.[g] ?? start;
    const before = live();
    const was = (g: string) => sat(before, g);
    expect(sat(helped, "students") - was("students")).toBeCloseTo(
      sat(plain, "students") - was("students"),
      12,
    );
    expect(sat(helped, "developers") - was("developers")).toBeCloseTo(
      (sat(plain, "developers") - was("developers")) * (1 - 0.3 * S),
      12,
    );
    expect(sat(plain, "developers")).toBeLessThan(was("developers"));
  });

  it("Support & operations: serving costs 1 − 0.2·s", () => {
    const served = (w: WorldState) => {
      w.market.customers.c1 = { pacific: { everyday: { free: 50_000, personal: 20_000 } } };
      const before = c1(w).ledger.current.costs.serving;
      const after = ok(step(w, TICK));
      return c1(after).ledger.current.costs.serving - before;
    };
    const plain = served(live());
    const helped = served(staff(live(), "support"));
    expect(plain).toBeGreaterThan(50);
    expect(helped).toBeCloseTo(plain * (1 - 0.2 * S), -1);
  });

  it("a run keeps the text and price it was previewed with after the budget changes", () => {
    let s = staff(runningWorld(), "research", 500_000);
    s = ok(step(s, cmd({ type: "startTraining", plan: plan() }, "r1")));
    const recorded = structuredClone(c1(s).runs.r1?.preview);
    s = ok(step(s, cmd({ type: "setDepartment", department: "research", budget: 0 })));
    s = ticks(s, 20);
    expect(c1(s).runs.r1?.preview).toEqual(recorded);
    expect(recorded?.tokens).toBeGreaterThan(BALANCE_V2.lengths[1]?.tokens ?? 0);
  });
});

// -- rivals fund departments by style (Task 3) -----------------------------------------------

describe("rivals' departments", () => {
  const SHARES: Record<string, Partial<Record<Department, number>>> = {
    premium: { research: 0.06 },
    budget: { support: 0.05 },
    specialist: { research: 0.03, data: 0.03 },
    consumer: { sales: 0.06 },
  };
  /** Every rival's last closed month earned `revenue`. */
  function earned(revenue: number): WorldState {
    const s = withRivals();
    for (const c of Object.values(s.companies)) {
      if (!c.rival) continue;
      const month = structuredClone(c.ledger.current);
      for (const k of Object.keys(month.income)) (month.income as Record<string, number>)[k] = 0;
      month.income.personal = revenue;
      c.ledger.months = [month];
    }
    return s;
  }
  const rivals = (s: WorldState) => Object.values(s.companies).filter((c) => c.rival);

  it("fund their style's departments from last month's revenue, in $10,000 steps", () => {
    const s = ok(step(earned(1_234_567), TICK));
    expect(rivals(s).length).toBeGreaterThan(0);
    for (const c of rivals(s)) {
      const shares = SHARES[c.rival?.style ?? ""] ?? {};
      for (const d of DEPARTMENTS) {
        const want = Math.floor((1_234_567 * (shares[d] ?? 0)) / 10_000) * 10_000;
        expect(c.departments[d].budget, `${c.rival?.style} ${d}`).toBe(want);
      }
    }
  });

  it("fund at most $500,000 a month each, and nothing without revenue", () => {
    for (const c of rivals(ok(step(earned(100_000_000), TICK)))) {
      for (const d of DEPARTMENTS) expect(c.departments[d].budget).toBeLessThanOrEqual(500_000);
      expect(Math.max(...DEPARTMENTS.map((d) => c.departments[d].budget))).toBe(500_000);
    }
    for (const c of rivals(ok(step(earned(0), TICK)))) {
      for (const d of DEPARTMENTS) expect(c.departments[d].budget).toBe(0);
    }
  });
});
