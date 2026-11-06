import { describe, expect, it } from "vitest";
import { step, upgradeState } from "../src/index.ts";
import { runningWorld, TICK, ticks } from "./pack.ts";

/** A sim-3 state: today's state without the fields sim-4 added. */
function sim3() {
  const s = structuredClone(ticks(runningWorld(), 3)) as unknown as Record<string, unknown> & {
    companies: Record<string, Record<string, unknown> & { versions: Record<string, object> }>;
  };
  delete s.rivals;
  delete s.legacyUntilSeq;
  delete s.leaders;
  delete s.lastRankingNews;
  delete s.points;
  for (const c of Object.values(s.companies)) {
    delete c.rival;
    delete c.startCash;
    delete c.lastRanks;
    delete c.battles;
    for (const v of Object.values(c.versions)) delete (v as { packKey?: unknown }).packKey;
  }
  return s;
}

describe("upgrading saved worlds", () => {
  it("brings a sim-3 world to sim-4 and keeps it running", () => {
    const up = upgradeState(sim3(), "sim-3");
    expect(up.rivals).toBeNull();
    expect(up.companies.c1?.rival).toBeNull();
    expect(up.companies.c1?.startCash).toBe(250_000);
    expect(up.companies.c1?.lastRanks).toBeNull();
    expect(up.leaders).toEqual({});
    expect(up.lastRankingNews).toEqual({});
    expect(up.points).toEqual({});
    expect(up.companies.c1?.battles).toEqual([]);
    const next = step(up, TICK);
    expect(next.ok).toBe(true);
  });
});

/** A sim-4 state: today's state without the fields sim-5 added. */
function sim4() {
  const s = structuredClone(ticks(runningWorld(), 3)) as unknown as Record<string, unknown> & {
    companies: Record<
      string,
      Record<string, unknown> & {
        ledger: { current: { costs: Record<string, number>; income: Record<string, number> } };
      }
    >;
  };
  delete s.legacyUntilSeq;
  for (const c of Object.values(s.companies)) {
    for (const k of ["showcases", "looks", "sponsorships", "apps", "fund", "loved"]) delete c[k];
    for (const k of ["events", "sponsorships", "looks", "appCosts", "investmentLosses", "overhead"])
      delete c.ledger.current.costs[k];
    for (const k of ["appRevenue", "investmentGains"]) delete c.ledger.current.income[k];
  }
  return s;
}

describe("upgrading a sim-4 world", () => {
  it("adds the playtest pack's fields with their starting values", () => {
    const up = upgradeState(sim4(), "sim-4", 40);
    const c1 = up.companies.c1;
    expect(c1?.showcases).toEqual([]);
    expect(c1?.looks).toEqual({
      owned: ["orb:classic", "logo:still"],
      orb: "orb:classic",
      logo: "logo:still",
    });
    expect(c1?.sponsorships).toEqual({});
    expect(c1?.apps).toEqual({});
    expect(c1?.fund).toEqual({ balance: 0 });
    expect(c1?.loved).toBe(0);
    expect(c1?.ledger.current.costs.overhead).toBe(0);
    expect(c1?.ledger.current.income.investmentGains).toBe(0);
    expect(up.legacyUntilSeq).toBe(40);
    expect(step(up, TICK).ok).toBe(true);
  });

  it("brings a sim-3 world all the way forward", () => {
    const up = upgradeState(sim3(), "sim-3", 5);
    expect(up.companies.c1?.fund).toEqual({ balance: 0 });
    expect(up.legacyUntilSeq).toBe(5);
  });

  it("starts a new world on today's rules", () => {
    expect(runningWorld().legacyUntilSeq).toBe(0);
  });
});

describe("upgrading a sim-8 world", () => {
  it("gives every department a budget of 0 and no team, and adds the Team cost line", () => {
    const s = structuredClone(ticks(runningWorld(), 40)) as unknown as {
      companies: Record<
        string,
        Record<string, unknown> & {
          ledger: {
            current: { costs: Record<string, number> };
            months: { costs: Record<string, number> }[];
          };
        }
      >;
    };
    for (const c of Object.values(s.companies)) {
      delete c.departments;
      for (const m of [...c.ledger.months, c.ledger.current]) delete m.costs.team;
    }
    const up = upgradeState(s, "sim-8");
    const c1 = up.companies.c1;
    expect(c1?.departments.research).toEqual({ budget: 0, staffed: 0 });
    expect(c1?.departments.support).toEqual({ budget: 0, staffed: 0 });
    expect(c1?.ledger.current.costs.team).toBe(0);
    expect(c1?.ledger.months.every((m) => m.costs.team === 0)).toBe(true);
    expect(c1?.lessons).toBeDefined();
    expect(step(up, TICK).ok).toBe(true);
  });
});
