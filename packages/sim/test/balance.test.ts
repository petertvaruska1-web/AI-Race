import { type GroupId, PLAN_IDS, REGION_IDS, type RegionId } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  adoptionShare,
  BALANCE_TEST,
  BALANCE_V2,
  computeDay,
  customersByRegion,
  dataCentersDay,
  emptyMonth,
  servingLoad,
  step,
  totalCustomers,
  type WorldState,
} from "../src/index.ts";
import { afterLaunch, type DayReport, type Report, runBot, TYPICAL_FIRST_MODEL } from "./bots.ts";

// Balance targets for the game layer (design §14), played by headless bots.

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const firstPayingDay = (r: Report) =>
  (r.days.find((d) => d.paying >= 1)?.day ?? Number.POSITIVE_INFINITY) - r.launchDay;
const twiceAsGood = Object.fromEntries(
  Object.entries(TYPICAL_FIRST_MODEL).map(([k, v]) => [k, Math.min(100, 2 * v)]),
);

describe("balance-v2", { timeout: 60_000 }, () => {
  const typical = runBot(BALANCE_V2, {}, 365);

  it("brings the first paying customer within 7 game days of a typical launch", () => {
    expect(firstPayingDay(typical)).toBeLessThanOrEqual(7);
  });

  it("earns more than it pays for serving in the month ending 60 days after launch", () => {
    const month = typical.days.filter(
      (d) => d.day > typical.launchDay + 30 && d.day <= typical.launchDay + 60,
    );
    const income = sum(month.map((d) => d.income));
    expect(income).toBeGreaterThan(0);
    expect(income).toBeGreaterThan(sum(month.map((d) => d.serving)));
  });

  it("stays a small startup: under 2% of the people willing, all first year", () => {
    const e = BALANCE_V2.economy;
    for (const d of typical.days) {
      expect(d.customers).toBeLessThan(
        0.02 * REGION_IDS.reduce((n, r) => n + e.population[r], 0) * adoptionShare(e, d.day),
      );
    }
  });

  it("gives a model twice as good at least 1.5 times the customers by day 90", () => {
    const better = runBot(BALANCE_V2, { scores: twiceAsGood }, 90);
    expect(afterLaunch(better, 90).customers).toBeGreaterThanOrEqual(
      1.5 * afterLaunch(typical, 90).customers,
    );
  });

  it("loses more paying students than paying businesses when the personal price doubles", () => {
    const pricey = runBot(BALANCE_V2, { prices: { personal: 20 } }, 90);
    const paying = (r: Report, g: GroupId) => {
      const byPlan = afterLaunch(r, 90).byGroup[g];
      return byPlan.personal + byPlan.business;
    };
    const drop = (g: GroupId) => 1 - paying(pricey, g) / paying(typical, g);
    expect(drop("students")).toBeGreaterThan(drop("businesses"));
  });
});

describe("balance-test", () => {
  it("brings the first paying customer within 3 game days of launch", () => {
    expect(firstPayingDay(runBot(BALANCE_TEST, {}, 10))).toBeLessThanOrEqual(3);
  });
});

describe("balance-v2: a growing business", { timeout: 120_000 }, () => {
  const share = (d: DayReport, r: RegionId) => (d.byRegion[r] ?? 0) / Math.max(1, d.customers);

  it("lets regions diverge: a Pacific campaign raises the Pacific's share of customers", () => {
    const plain = runBot(BALANCE_V2, {}, 120);
    const aimed = runBot(
      BALANCE_V2,
      {
        campaigns: [
          { afterDays: 0, region: "pacific", group: "all", budgetPerDay: 2000, days: 30 },
        ],
      },
      120,
    );
    const ratio =
      share(afterLaunch(aimed, 90), "pacific") / share(afterLaunch(plain, 90), "pacific");
    expect(ratio, `Pacific share ratio ${ratio.toFixed(2)}`).toBeGreaterThanOrEqual(1.3);
  });

  it("makes a data center pay at scale: cheaper serving per customer, and its region gains", () => {
    const base = structuredClone(runBot(BALANCE_V2, {}, 120).final);
    const c = base.companies.c1;
    if (!c) throw new Error("no company");
    // 500,000 customers, spread as the company's own are.
    const scale = 500_000 / totalCustomers(base.market, "c1");
    for (const byGroup of Object.values(base.market.customers.c1 ?? {})) {
      for (const byPlan of Object.values(byGroup ?? {})) {
        for (const plan of PLAN_IDS) if (byPlan?.[plan]) byPlan[plan] *= scale;
      }
    }
    c.compute.cloudPflops = servingLoad(base, "c1");
    c.cash = 10_000_000;
    const owned = structuredClone(base);
    const oc = owned.companies.c1;
    if (!oc) throw new Error("no company");
    oc.dataCenters.dc1 = {
      id: "dc1",
      region: "pacific",
      startDay: owned.day - 60,
      openDay: owned.day,
      pflopsPerDay: BALANCE_V2.economy.dataCenter.pflopsPerDay,
    };
    const perCustomer = (w: WorldState) => {
      const co = w.companies.c1;
      if (!co) throw new Error("no company");
      co.ledger.current = emptyMonth(co.ledger.current.month);
      dataCentersDay(w);
      computeDay(w);
      const m = co.ledger.current.costs;
      return (m.serving + m.dataCenterUpkeep) / totalCustomers(w.market, "c1");
    };
    const cloud = perCustomer(structuredClone(base));
    const mine = perCustomer(structuredClone(owned));
    expect(mine / cloud, `owned/cloud cost ${(mine / cloud).toFixed(2)}`).toBeLessThanOrEqual(0.6);
    const later = (w: WorldState) => {
      let out = w;
      for (let i = 0; i < 30; i++) {
        const r = step(out, { kind: "tick" });
        if (!r.ok) throw new Error(r.reason);
        out = r.state;
      }
      const byRegion = customersByRegion(out.market, "c1");
      return (byRegion.pacific ?? 0) / totalCustomers(out.market, "c1");
    };
    expect(later(owned)).toBeGreaterThan(later(base));
  });

  it("recovers from a tripling of customers within 10 days", () => {
    // The spike lands after day 60's figures, so day 61 is the first day it is served.
    const r = runBot(BALANCE_V2, { spike: { afterDays: 60, factor: 3 } }, 90);
    expect(afterLaunch(r, 61).served).toBeLessThan(1);
    const back = r.days.find((d) => d.day > r.launchDay + 61 && d.served === 1);
    expect(back?.day ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(r.launchDay + 71);
  });

  it("spreads report-card grades over at least four letters", () => {
    const grades = new Set(
      [
        runBot(BALANCE_V2, {}, 365),
        runBot(BALANCE_V2, { scores: twiceAsGood }, 365),
        runBot(BALANCE_V2, { plans: ["free"] }, 365),
        runBot(BALANCE_V2, { prices: { personal: 40, business: 120 } }, 365),
        runBot(
          BALANCE_V2,
          {
            campaigns: [
              { afterDays: 30, region: "all", group: "all", budgetPerDay: 20_000, days: 14 },
            ],
          },
          365,
        ),
      ].flatMap((r) => r.reports.map((c) => c.grade)),
    );
    expect(grades.size, [...grades].join(" ")).toBeGreaterThanOrEqual(4);
  });

  it("keeps ads an investment: a $350,000 blitz takes two months or more to pay for itself", () => {
    const plain = runBot(BALANCE_V2, {}, 150);
    const blitz = runBot(
      BALANCE_V2,
      {
        campaigns: [0, 1, 2, 3, 4].map(() => ({
          afterDays: 0,
          region: "all" as const,
          group: "all" as const,
          budgetPerDay: 10_000,
          days: 7,
        })),
      },
      150,
    );
    const behind = (n: number) => afterLaunch(blitz, n).cash < afterLaunch(plain, n).cash;
    expect(behind(60)).toBe(true);
    expect(behind(140)).toBe(false);
  });

  it("rescues a spendthrift with a loan it then pays down", () => {
    const r = runBot(
      BALANCE_V2,
      {
        campaigns: [0, 1, 2, 3, 4].map(() => ({
          afterDays: 0,
          region: "all" as const,
          group: "all" as const,
          budgetPerDay: 10_000,
          days: 14,
        })),
      },
      365,
    );
    const offered = r.days.findIndex((d) => d.rescueOffered);
    expect(offered).toBeGreaterThanOrEqual(0);
    const loaned = r.days.findIndex((d, i) => i > offered && d.loan > 0);
    expect(loaned).toBeGreaterThan(offered);
    expect(r.days[loaned]?.cash ?? -1).toBeGreaterThanOrEqual(0);
    const owed = r.days.filter((_, i) => i >= loaned).map((d) => d.loan);
    expect(owed.at(-1) ?? 0).toBeLessThan(owed[0] ?? 0);
  });
});
