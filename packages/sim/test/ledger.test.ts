import { type Command, GROUP_IDS, type GroupId, type PlanId } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  type CompanyState,
  createWorld,
  emptyMarket,
  emptyMonth,
  finance,
  type Input,
  ledgerDay,
  type MonthLedger,
  paramCount,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}
const cmd = (c: unknown, newId?: string): Input => ({
  kind: "command",
  userId: "u1",
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;
const ticks = (s: WorldState, n: number) => {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, { kind: "tick" }));
  return out;
};
const trainingPlan = (factCheck = false) => ({
  ...BALANCE_TEST.defaultPlan,
  mix: { ...BALANCE_TEST.defaultPlan.mix },
  factCheck,
});

/** A running world on day 0 (2027-01-01) where u1 owns company c1. */
function world(): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 5,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
  s = ok(
    step(
      s,
      cmd(
        {
          type: "createCompany",
          name: "Acme",
          logo: LOGO,
          palette: { primary: "#4f8cff", secondary: "#ffb020" },
        },
        "c1",
      ),
    ),
  );
  return ok(step(s, cmd({ type: "startWorld" })));
}

/** c1 trained and scored v1 and launched it on every plan; it is live. */
function live(): WorldState {
  let s = ok(step(world(), cmd({ type: "startTraining", plan: trainingPlan() }, "r1")));
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
  s = ticks(s, (s.companies.c1?.runs.r1?.endDay ?? 0) - s.day);
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
          plans: ["free", "personal", "business"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
  return ticks(s, 3);
}

const company = (s: WorldState): CompanyState => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};
const month = (s: WorldState): MonthLedger => company(s).ledger.current;

/** Clears the market and the month's bookings, so a test counts only its own days. */
function fresh(s: WorldState): WorldState {
  s.market = emptyMarket();
  company(s).ledger.current = emptyMonth(month(s).month);
  return s;
}

/** Puts `n` customers of `group` on `plan` (accounts, for the business plan). */
function give(s: WorldState, group: GroupId, plan: PlanId, n: number): void {
  const byRegion = s.market.customers.c1 ?? {};
  const byGroup = byRegion.pacific ?? {};
  byGroup[group] = { ...byGroup[group], [plan]: n };
  byRegion.pacific = byGroup;
  s.market.customers.c1 = byRegion;
}

const sum = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

describe("daily bookings", () => {
  it("books about N × $10 of personal income over 30 days, by group and plan", () => {
    const s = fresh(live());
    const cash = company(s).cash;
    give(s, "students", "personal", 1234);
    for (let d = 0; d < 30; d++) ledgerDay(s);
    const m = month(s);
    expect(Math.abs(m.income.personal - 12_340)).toBeLessThanOrEqual(15);
    expect(m.income.business).toBe(0);
    expect(m.revenueByGroup.students).toBe(m.income.personal);
    expect(m.revenueByPlan.personal).toBe(m.income.personal);
    expect(company(s).cash).toBe(cash + m.income.personal - m.costs.serving);
  });

  it("books business accounts once for every seat", () => {
    const s = fresh(live());
    give(s, "developers", "business", 101);
    for (let d = 0; d < 30; d++) ledgerDay(s);
    const seats = BALANCE_TEST.economy.groups.developers.seatsPerAccount;
    expect(month(s).income.business).toBe(101 * seats * 30);
    expect(month(s).revenueByPlan.business).toBe(month(s).income.business);
  });

  it("charges serving for free customers too, in step with the product's size", () => {
    const servingFor = (nLayer: number, dModel: number) => {
      const s = fresh(live());
      const v = company(s).versions.v1;
      if (!v) throw new Error("no version");
      v.nLayer = nLayer;
      v.dModel = dModel;
      give(s, "everyday", "free", 5_000_000);
      ledgerDay(s);
      expect(month(s).income.personal).toBe(0);
      return month(s).costs.serving;
    };
    const small = servingFor(2, 64);
    const big = servingFor(8, 256);
    const rate = BALANCE_TEST.economy.servingCostPerCustomerDayPerMParam;
    expect(Math.abs(small - (5_000_000 * rate * paramCount(2, 64)) / 1e6)).toBeLessThanOrEqual(0.5);
    expect(big / small).toBeCloseTo(paramCount(8, 256) / paramCount(2, 64), 1);
  });

  it("charges serving for every seat of a business account", () => {
    const s = fresh(live());
    give(s, "businesses", "business", 10_000);
    ledgerDay(s);
    const seats = BALANCE_TEST.economy.groups.businesses.seatsPerAccount;
    const rate = BALANCE_TEST.economy.servingCostPerCustomerDayPerMParam;
    const expected = (10_000 * seats * rate * paramCount(2, 64)) / 1e6;
    expect(Math.abs(month(s).costs.serving - expected)).toBeLessThanOrEqual(0.5);
  });

  it("books nothing on a day without customers", () => {
    const s = fresh(live());
    const cash = company(s).cash;
    ledgerDay(s);
    expect(sum(month(s).income)).toBe(0);
    expect(month(s).costs.serving).toBe(0);
    expect(company(s).cash).toBe(cash);
  });
});

describe("training money", () => {
  it("books a run under training, and its fact-checking under data checks", () => {
    const s = ok(step(world(), cmd({ type: "startTraining", plan: trainingPlan(true) }, "r1")));
    const cost = company(s).runs.r1?.preview.cost ?? 0;
    expect(month(s).costs.dataChecks).toBe(BALANCE_TEST.factCheckCost);
    expect(month(s).costs.training).toBe(cost - BALANCE_TEST.factCheckCost);
    expect(company(s).cash).toBe(BALANCE_TEST.foundingCash - cost);
    const plain = ok(step(world(), cmd({ type: "startTraining", plan: trainingPlan() }, "r1")));
    expect(month(plain).costs.dataChecks).toBe(0);
    expect(month(plain).costs.training).toBe(company(plain).runs.r1?.preview.cost);
  });

  it("books a refund back when a run fails", () => {
    let s = ok(step(world(), cmd({ type: "startTraining", plan: trainingPlan(true) }, "r1")));
    s = ok(step(s, sys({ type: "trainingFailed", runId: "r1", reason: "Stopped." })));
    expect(month(s).costs.training).toBe(0);
    expect(month(s).costs.dataChecks).toBe(0);
    expect(company(s).cash).toBe(BALANCE_TEST.foundingCash);
  });

  it("books a refund that arrives after its month closed into the current month", () => {
    let s = ok(step(world(), cmd({ type: "startTraining", plan: trainingPlan() }, "r1")));
    const cost = company(s).runs.r1?.preview.cost ?? 0;
    s = ticks(s, 31);
    s = ok(step(s, sys({ type: "trainingFailed", runId: "r1", reason: "Stopped." })));
    const [january] = company(s).ledger.months;
    expect(january?.costs.training).toBe(cost);
    expect(month(s).month).toBe("2027-02");
    expect(month(s).costs.training).toBe(-cost);
    expect(company(s).cash).toBe(BALANCE_TEST.foundingCash);
  });
});

describe("months", () => {
  it("closes the month into the history when the calendar turns", () => {
    let s = world();
    expect(month(s).month).toBe("2027-01");
    s = ticks(s, 30); // 2027-01-31
    expect(company(s).ledger.months).toEqual([]);
    s = ticks(s, 1); // 2027-02-01
    expect(company(s).ledger.months.map((m) => m.month)).toEqual(["2027-01"]);
    expect(month(s).month).toBe("2027-02");
  });

  it("records the customers at the month's end", () => {
    let s = live();
    s = ticks(s, 58 - s.day); // 2027-02-28
    const customers = Object.values(s.market.customers.c1 ?? {})
      .flatMap((byGroup) => Object.values(byGroup ?? {}))
      .reduce((total, byPlan) => total + sum(byPlan as Record<string, number>), 0);
    expect(customers).toBeGreaterThan(0);
    const before = s;
    s = ticks(s, 1); // 2027-03-01
    const february = company(s).ledger.months.at(-1);
    expect(february?.month).toBe("2027-02");
    expect(february?.customersEnd).toBe(Math.round(customers));
    const byGroup: Record<string, number> = february?.customersEndByGroup ?? {};
    for (const g of GROUP_IDS) {
      // The close counts the market as the last day of the month left it.
      const inGroup = Object.values(before.market.customers.c1 ?? {}).reduce(
        (n, groups) => n + sum((groups?.[g] ?? {}) as Record<string, number>),
        0,
      );
      expect(byGroup[g]).toBe(Math.round(inGroup));
    }
    expect(Math.abs(sum(byGroup) - (february?.customersEnd ?? 0))).toBeLessThanOrEqual(6);
  });

  it("books revenue by region, adding up to the income", () => {
    let s = live();
    s = ticks(s, 20);
    const m = month(s);
    expect(sum(m.revenueByRegion)).toBe(sum(m.income));
    expect(sum(m.revenueByPlan)).toBe(sum(m.income));
    expect(m.revenueByRegion.pacific).toBeGreaterThan(0);
  });

  it("keeps the last 24 months, newest last", () => {
    const s = ticks(world(), 800); // 2029-03-11
    const months = company(s).ledger.months.map((m) => m.month);
    expect(months).toHaveLength(24);
    expect(months[0]).toBe("2027-03");
    expect(months.at(-1)).toBe("2029-02");
    expect(month(s).month).toBe("2029-03");
  });
});

describe("financing", () => {
  it("moves cash and the month's financing together: loans and investors in, repayments out", () => {
    const s = fresh(live());
    const c = company(s);
    const cash = c.cash;
    finance(c, "loans", 1000.4);
    finance(c, "investors", 250);
    finance(c, "repaid", 300);
    expect(c.cash).toBe(cash + 1000 + 250 - 300);
    expect(month(s).financing).toEqual({ loans: 1000, investors: 250, repaid: 300 });
    expect(sum(month(s).income)).toBe(0);
    expect(sum(month(s).costs)).toBe(0);
  });
});

describe("money conservation", () => {
  it("keeps cash equal to founding cash plus every income less every cost, exactly", () => {
    let s = ticks(live(), 60);
    s = ok(step(s, cmd({ type: "startTraining", plan: trainingPlan(true) }, "r2")));
    s = ok(step(s, sys({ type: "trainingFailed", runId: "r2", reason: "Stopped." })));
    s = ticks(s, 40);
    const c = company(s);
    const months = [...c.ledger.months, c.ledger.current];
    finance(company(s), "loans", 5000);
    finance(company(s), "repaid", 1200);
    const net = months.reduce(
      (total, m) =>
        total +
        sum(m.income) -
        sum(m.costs) +
        m.financing.loans +
        m.financing.investors -
        m.financing.repaid,
      0,
    );
    expect(months.some((m) => m.income.personal > 0 && m.costs.serving > 0)).toBe(true);
    expect(Number.isInteger(c.cash)).toBe(true);
    expect(c.cash).toBe(BALANCE_TEST.foundingCash + net);
  });
});
