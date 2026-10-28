import { type GroupId, REGION_IDS } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import { adoptionShare, BALANCE_TEST, BALANCE_V2 } from "../src/index.ts";
import { afterLaunch, type Report, runBot, TYPICAL_FIRST_MODEL } from "./bots.ts";

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
