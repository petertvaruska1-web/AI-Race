import { describe, expect, it } from "vitest";
import {
  BALANCE_V2,
  type CompanyState,
  getBalance,
  overheadForDay,
  projectView,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, ticks } from "./pack.ts";

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const invest = (s: WorldState, amount: number) => step(s, cmd({ type: "invest", amount }));
const withdraw = (s: WorldState, amount: number) => step(s, cmd({ type: "withdraw", amount }));

/** Plays until the next game month starts. */
function toNextMonth(s: WorldState): WorldState {
  const month = c1(s).ledger.current.month;
  let t = s;
  while (c1(t).ledger.current.month === month) t = ticks(t, 1);
  return t;
}

describe("the investment fund", () => {
  it("moves money from cash into the fund and back, without booking it as income or cost", () => {
    const s = runningWorld();
    const cash = c1(s).cash;
    const in1 = ok(invest(s, 100_000));
    expect(c1(in1).cash).toBe(cash - 100_000);
    expect(c1(in1).fund.balance).toBe(100_000);
    const out = ok(withdraw(in1, 40_000));
    expect(c1(out).cash).toBe(cash - 60_000);
    expect(c1(out).fund.balance).toBe(60_000);
    expect(c1(out).ledger.current.income.investmentGains).toBe(0);
  });

  it("refuses more than there is", () => {
    const s = runningWorld();
    expect(rejected(invest(s, c1(s).cash + 1))).toMatch(/You have/);
    expect(rejected(withdraw(s, 1))).toMatch(/fund holds/);
  });

  it("gains or loses at each month end, seeded, booked in the month that closes", () => {
    const s = ok(invest(runningWorld(), 200_000));
    const a = toNextMonth(s);
    const b = toNextMonth(s);
    expect(c1(a).fund.balance).toBe(c1(b).fund.balance); // same seed, same month
    const closed = c1(a).ledger.months.at(-1);
    const change = c1(a).fund.balance - 200_000;
    expect(change).not.toBe(0);
    if (change > 0) expect(closed?.income.investmentGains).toBe(change);
    else expect(closed?.costs.investmentLosses).toBe(-change);
    expect(c1(a).cash + c1(a).fund.balance).toBeGreaterThan(0);
  });

  it("averages a small gain a month with occasional dips", () => {
    const f = getBalance("balance-v2").economy.fund;
    expect(f.meanMonthly).toBeGreaterThan(0);
    expect(f.dipChance).toBeCloseTo(1 / 12);
  });

  it("covers a month-end shortfall before a rescue is offered", () => {
    const s = structuredClone(ok(invest(runningWorld(), 100_000)));
    c1(s).cash = -5_000;
    const t = toNextMonth(s);
    expect(c1(t).rescue).toBeNull();
    expect(c1(t).cash).toBeGreaterThanOrEqual(0);
    expect(c1(t).fund.balance).toBeLessThan(100_000);
  });

  it("shows the fund to its owner", () => {
    const s = ok(invest(runningWorld(), 1_000));
    expect(projectView(s, "u1").me.company?.fund).toEqual({ balance: 1_000 });
  });
});

describe("office and support bills", () => {
  const share = (monthly: number) =>
    (overheadForDay(BALANCE_V2.economy, monthly / 30) * 30) / monthly;

  it("are nothing without revenue, and a bigger company pays a bigger share", () => {
    expect(overheadForDay(BALANCE_V2.economy, 0)).toBe(0);
    expect(share(465_000)).toBeGreaterThan(0.13);
    expect(share(465_000)).toBeLessThan(0.17);
    expect(share(60_000)).toBeLessThan(share(465_000));
    expect(share(14_000_000)).toBeGreaterThan(share(1_700_000));
    expect(share(14_000_000)).toBeLessThan(0.25);
  });

  it("are booked each day there is revenue, but not on a replayed old day", () => {
    const s = structuredClone(runningWorld());
    const day = ok(step(s, { kind: "tick" }));
    expect(c1(day).ledger.current.costs.overhead).toBe(0); // nothing sold yet
    const paid = structuredClone(s);
    c1(paid).ledger.current.income.personal = 0;
    expect(overheadForDay(BALANCE_V2.economy, 1_000)).toBeGreaterThan(0);
    const legacy = { ...structuredClone(s), legacyUntilSeq: s.seq + 1 };
    expect(c1(ok(step(legacy, { kind: "tick" }))).ledger.current.costs.overhead).toBe(0);
  });
});
