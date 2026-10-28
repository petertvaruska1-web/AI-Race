import { type Command, GROUP_IDS, PLAN_IDS, WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  createWorld,
  customersOf,
  HISTORY_DAYS,
  type Input,
  projectView,
  type StepResult,
  step,
  upgradeState,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}
const cmd = (userId: string, c: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const ticks = (s: WorldState, n: number) => {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, { kind: "tick" }));
  return out;
};
const company = (s: WorldState, id = "c1"): CompanyState => {
  const c = s.companies[id];
  if (!c) throw new Error("no company");
  return c;
};

/** A running world where u1 owns c1 and u2 owns c2. */
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
  const palette = { primary: "#4f8cff", secondary: "#ffb020" };
  s = ok(step(s, cmd("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette }, "c1")));
  s = ok(
    step(s, { kind: "system", event: { type: "memberJoined", userId: "u2", displayName: "Bo" } }),
  );
  s = ok(
    step(
      s,
      cmd(
        "u2",
        {
          type: "createCompany",
          name: "Bolt",
          logo: LOGO,
          palette: { primary: "#22c55e", secondary: "#a855f7" },
        },
        "c2",
      ),
    ),
  );
  return ok(step(s, cmd("u1", { type: "startWorld" })));
}

/** Gives c1 customers on day `s.day`, as if a product had won them. */
function withCustomers(s: WorldState): WorldState {
  const next = structuredClone(s);
  next.market.customers.c1 = {
    national: { students: { free: 10.4, personal: 2.6 }, businesses: { business: 3.5 } },
  };
  return next;
}

describe("day history", () => {
  it("records each company's day: cash, customers by group, paying customers and fame", () => {
    const s = ticks(withCustomers(world()), 1);
    const c = company(s);
    const sample = c.history.at(-1);
    const expected = Object.fromEntries(
      GROUP_IDS.map((g) => [
        g,
        Math.round(PLAN_IDS.reduce((n, p) => n + customersOf(s.market, "c1", "national", g, p), 0)),
      ]),
    );
    const paying = ["personal", "business"] as const;
    expect(sample).toEqual({
      day: s.day,
      cash: c.cash,
      customers: expected,
      paying: Math.round(
        GROUP_IDS.reduce(
          (n, g) =>
            n + paying.reduce((m, p) => m + customersOf(s.market, "c1", "national", g, p), 0),
          0,
        ),
      ),
      followers: Math.round(c.fame.followers),
      reputation: Math.round(c.fame.reputation),
    });
    expect(company(s, "c2").history.at(-1)?.day).toBe(s.day);
  });

  it("adds one sample a day and keeps the last 120 days", () => {
    const s = ticks(world(), HISTORY_DAYS + 10);
    const days = company(s).history.map((h) => h.day);
    expect(HISTORY_DAYS).toBe(120);
    expect(days).toHaveLength(120);
    expect(days[0]).toBe(s.day - 119);
    expect(days.at(-1)).toBe(s.day);
  });

  it("opens a world saved before history existed with an empty history", () => {
    const old = structuredClone(ticks(world(), 2)) as WorldState;
    for (const c of Object.values(old.companies)) delete (c as Partial<CompanyState>).history;
    const s = ticks(upgradeState(old), 1);
    expect(company(s).history.map((h) => h.day)).toEqual([s.day]);
  });

  it("shows the history keyed by day, in its owner's view only (R2)", () => {
    const s = ticks(withCustomers(world()), 3);
    const mine = WorldView.parse(projectView(s, "u1"));
    const theirs = WorldView.parse(projectView(s, "u2"));
    const history = mine.me.company?.history ?? {};
    expect(Object.keys(history)).toEqual(["1", "2", "3"]);
    expect(history["3"]).toEqual({
      cash: company(s).cash,
      customers: company(s).history.at(-1)?.customers,
      paying: company(s).history.at(-1)?.paying,
      followers: company(s).history.at(-1)?.followers,
      reputation: company(s).history.at(-1)?.reputation,
    });
    expect(Object.keys(theirs.me.company?.history ?? {})).toEqual(["1", "2", "3"]);
    expect(theirs.me.company?.history["3"]?.cash).toBe(company(s, "c2").cash);
    expect(JSON.stringify(theirs.companies.find((c) => c.id === "c1"))).not.toContain("history");
  });
});
