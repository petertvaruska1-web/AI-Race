import type { Command } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  createWorld,
  type Input,
  monthOfDay,
  projectView,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
const r = BALANCE_TEST.economy.rescue;

function ok(res: StepResult): WorldState {
  if (!res.ok) throw new Error(`rejected: ${res.reason}`);
  return res.state;
}
const cmd = (c: unknown, newId?: string): Input => ({
  kind: "command",
  userId: "u1",
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const ticks = (s: WorldState, n: number) => {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, { kind: "tick" }));
  return out;
};
const toNextMonth = (s: WorldState) => {
  let out = s;
  const month = monthOfDay(out.day);
  while (monthOfDay(out.day) === month) out = ticks(out, 1);
  return out;
};
const c1 = (s: WorldState) => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};
const rescueNews = (s: WorldState) =>
  s.feed
    .filter((f) => f.kind === "news" && f.topic === "rescue")
    .map((f) => (f.kind === "news" ? f.headline : ""));

/** A running world where c1 has `cash` dollars on the last day of January. */
function broke(cash: number): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 5,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
  const palette = { primary: "#4f8cff", secondary: "#ffb020" };
  s = ok(step(s, cmd({ type: "createCompany", name: "Acme", logo: LOGO, palette }, "c1")));
  s = ticks(ok(step(s, cmd({ type: "startWorld" }))), 29); // 2027-01-30
  const next = structuredClone(s);
  c1(next).cash = cash;
  return next;
}

describe("rescue", () => {
  it("is offered at a month end with cash below zero, once, with the news", () => {
    expect(c1(toNextMonth(broke(10))).rescue).toBeNull();
    let s = toNextMonth(broke(-5000));
    expect(c1(s).rescue).toEqual({ offeredDay: s.day, deadlineDay: s.day + r.decideDays });
    expect(rescueNews(s)).toEqual(["Acme runs out of cash"]);
    s = ticks(s, r.decideDays - 1);
    expect(rescueNews(s)).toHaveLength(1);
    expect(c1(s).rescue).not.toBeNull();
  });

  it("by loan, restores cash to the cushion as financing, once", () => {
    let s = toNextMonth(broke(-5000));
    const cash = c1(s).cash;
    s = ok(step(s, cmd({ type: "chooseRescue", kind: "loan" })));
    const amount = r.cushion - cash;
    expect(c1(s).cash).toBe(r.cushion);
    expect(c1(s).ledger.current.financing.loans).toBe(amount);
    expect(c1(s).loan).toEqual({
      balance: amount,
      principalPerMonth: Math.ceil(amount / r.loanTermMonths),
    });
    expect(c1(s).rescue).toBeNull();
    expect(rescueNews(s).at(-1)).toBe("Acme takes an emergency loan");
    const again = step(s, cmd({ type: "chooseRescue", kind: "loan" }));
    expect(again.ok ? "" : again.reason).toBe("There is no rescue to choose.");
  });

  it("by investor, restores cash and gives the investor a quarter of the company", () => {
    let s = toNextMonth(broke(-5000));
    const cash = c1(s).cash;
    s = ok(step(s, cmd({ type: "chooseRescue", kind: "investor" })));
    expect(c1(s).cash).toBe(r.cushion);
    expect(c1(s).ledger.current.financing.investors).toBe(r.cushion - cash);
    expect(c1(s).ownership).toBe(0.75);
    expect(c1(s).loan).toBeNull();
    expect(rescueNews(s).at(-1)).toBe("Investors take a 25% stake in Acme");
    const pub = projectView(s, "u1").companies.find((c) => c.id === "c1");
    expect(pub?.ownership).toBe(0.75);
  });

  it("takes the loan at the deadline, or lapses if the cash came back", () => {
    let s = toNextMonth(broke(-5000));
    s = ticks(s, r.decideDays);
    expect(c1(s).rescue).toBeNull();
    expect(c1(s).loan).not.toBeNull();
    expect(c1(s).cash).toBe(r.cushion);

    let back = toNextMonth(broke(-5000));
    back = structuredClone(back);
    c1(back).cash = 50_000;
    back = ticks(back, r.decideDays);
    expect(c1(back).rescue).toBeNull();
    expect(c1(back).loan).toBeNull();
    expect(c1(back).ledger.current.financing.loans).toBe(0);
  });

  it("charges the loan's interest and a repayment at each month end until it is paid", () => {
    let s = toNextMonth(broke(-5000));
    s = ok(step(s, cmd({ type: "chooseRescue", kind: "loan" })));
    const loan = c1(s).loan;
    if (!loan) throw new Error("no loan");
    s = toNextMonth(s);
    expect(c1(s).ledger.current.costs.interest).toBe(Math.round(loan.balance * r.loanMonthlyRate));
    expect(c1(s).ledger.current.financing.repaid).toBe(loan.principalPerMonth);
    expect(c1(s).loan?.balance).toBe(loan.balance - loan.principalPerMonth);
    const view = projectView(s, "u1");
    expect(view.me.company?.loan).toEqual({
      balance: loan.balance - loan.principalPerMonth,
      principalPerMonth: loan.principalPerMonth,
      monthlyRate: r.loanMonthlyRate,
    });
    // A company that can pay (this one has no income of its own) clears it within its term.
    s = structuredClone(s);
    c1(s).cash = 10_000_000;
    for (let m = 0; m < r.loanTermMonths; m++) s = toNextMonth(s);
    expect(c1(s).loan).toBeNull();
  });

  it("adds a second loan to the first", () => {
    let s = toNextMonth(broke(-5000));
    s = ok(step(s, cmd({ type: "chooseRescue", kind: "loan" })));
    const first = c1(s).loan?.balance ?? 0;
    s = structuredClone(s);
    c1(s).cash = -20_000;
    s = toNextMonth(s);
    const paid = c1(s).ledger.current.financing.repaid;
    s = ok(step(s, cmd({ type: "chooseRescue", kind: "loan" })));
    const balance = c1(s).loan?.balance ?? 0;
    expect(balance).toBe(first - paid + c1(s).ledger.current.financing.loans);
    expect(c1(s).loan?.principalPerMonth).toBe(Math.ceil(balance / r.loanTermMonths));
  });
});
