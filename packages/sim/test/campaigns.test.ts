import { type Command, GROUP_IDS, PLAN_IDS } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  createWorld,
  customersOf,
  type Input,
  projectView,
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
const refused = (s: WorldState, input: Input): string => {
  const r = step(s, input);
  if (r.ok) throw new Error("accepted");
  return r.reason;
};
const c1 = (s: WorldState) => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};

/** A running world on the test balance where u1 owns c1, with a scored v1. */
function scored(): WorldState {
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
  s = ok(step(s, cmd({ type: "startWorld" })));
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  s = ok(step(s, cmd({ type: "startTraining", plan }, "r1")));
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
  return ok(
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
}

/** `scored()` with v1 launched as Spark on every plan after a 3-day countdown. */
function announced(): WorldState {
  return ok(
    step(
      scored(),
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
}

const campaign = (over: Record<string, unknown> = {}) => ({
  type: "startCampaign",
  region: "pacific",
  group: "developers",
  budgetPerDay: 1000,
  days: 10,
  ...over,
});

const pacificDevelopers = (s: WorldState) =>
  PLAN_IDS.reduce((n, p) => n + customersOf(s.market, "c1", "pacific", "developers", p), 0);

describe("ad campaigns", () => {
  it("refuse plainly, changing nothing, when they can't run", () => {
    expect(refused(scored(), cmd(campaign(), "k1"))).toBe("Launch an AI before advertising it.");
    const s = announced();
    expect(refused(s, cmd(campaign({ budgetPerDay: 99 }), "k1"))).toMatch(/\$100–\$100,000/);
    expect(refused(s, cmd(campaign({ budgetPerDay: 100_001 }), "k1"))).toMatch(/\$100–\$100,000/);
    expect(refused(s, cmd(campaign({ days: 6 }), "k1"))).toMatch(/7–90 game days/);
    expect(refused(s, cmd(campaign({ days: 91 }), "k1"))).toMatch(/7–90 game days/);
    let five = s;
    for (const id of ["k1", "k2", "k3", "k4", "k5"]) five = ok(step(five, cmd(campaign(), id)));
    expect(refused(five, cmd(campaign(), "k6"))).toBe("You can run up to 5 campaigns at once.");
    const poor = structuredClone(s);
    c1(poor).cash = 999;
    expect(refused(poor, cmd(campaign(), "k1"))).toMatch(/at least \$1,000 for a day/);
    expect(refused(s, cmd({ type: "stopCampaign", campaignId: "nope" }))).toBe(
      "That campaign isn't running.",
    );
    expect(c1(s).campaigns).toEqual({});
  });

  it("spend their budget on exactly as many days as they run, then end", () => {
    let s = ok(step(announced(), cmd(campaign(), "k1")));
    const before = c1(s).ledger.current.costs.marketing;
    s = ticks(s, 9);
    expect(c1(s).campaigns.k1?.spent).toBe(9000);
    s = ticks(s, 1);
    expect(c1(s).campaigns.k1).toBeUndefined();
    s = ticks(s, 5);
    const months = [...c1(s).ledger.months, c1(s).ledger.current];
    const marketing = months.reduce((n, m) => n + m.costs.marketing, 0);
    expect(marketing - before).toBe(10_000);
  });

  it("raise awareness only where they aim, winning more customers there", () => {
    const base = ticks(announced(), 3);
    const started = ok(step(base, cmd(campaign({ days: 30 }), "k1")));
    const aware = (s: WorldState, r: string, g: string) =>
      s.market.awareness.c1?.[r as "pacific"]?.[g as "developers"] ?? 0;
    // Its first day: only the cell it aims at moves.
    const aimed = ticks(started, 1);
    const plain = ticks(base, 1);
    expect(aware(aimed, "pacific", "developers")).toBeGreaterThan(
      aware(plain, "pacific", "developers"),
    );
    expect(aware(aimed, "newEngland", "students")).toBe(aware(plain, "newEngland", "students"));
    // A month on, it has won customers there.
    expect(pacificDevelopers(ticks(started, 30))).toBeGreaterThan(
      pacificDevelopers(ticks(base, 30)),
    );
  });

  it("aimed at everyone everywhere, reach every region and group", () => {
    const base = ticks(announced(), 3);
    const wide = ticks(ok(step(base, cmd(campaign({ region: "all", group: "all" }), "k1"))), 3);
    const plain = ticks(base, 3);
    for (const g of GROUP_IDS) {
      const r = "midSouth";
      expect(wide.market.awareness.c1?.[r]?.[g] ?? 0).toBeGreaterThan(
        plain.market.awareness.c1?.[r]?.[g] ?? 0,
      );
    }
  });

  it("stop spending as soon as they are stopped", () => {
    let s = ticks(ok(step(announced(), cmd(campaign(), "k1"))), 2);
    s = ok(step(s, cmd({ type: "stopCampaign", campaignId: "k1" })));
    const spent = c1(s).ledger.current.costs.marketing;
    s = ticks(s, 3);
    expect(c1(s).campaigns.k1).toBeUndefined();
    expect(c1(s).ledger.current.costs.marketing).toBe(spent);
  });

  it("build hype for a launch counting down", () => {
    const s = announced();
    const quiet = ticks(s, 2);
    const loud = ticks(ok(step(s, cmd(campaign({ budgetPerDay: 50_000 }), "k1"))), 2);
    expect(loud.companies.c1?.products.p1?.hype ?? 0).toBeGreaterThan(
      quiet.companies.c1?.products.p1?.hype ?? 0,
    );
  });

  it("show only to their owner", () => {
    const s = ok(step(announced(), cmd(campaign(), "k1")));
    const view = projectView(s, "u1");
    expect(view.me.company?.campaigns).toEqual([
      {
        id: "k1",
        region: "pacific",
        group: "developers",
        budgetPerDay: 1000,
        startDay: s.day,
        endDay: s.day + 10,
        spent: 0,
      },
    ]);
  });
});
