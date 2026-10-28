import { type Command, GROUP_IDS, PLAN_IDS, REGION_IDS, WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  adoptable,
  adoptionShare,
  BALANCE_TEST,
  BALANCE_V2,
  createWorld,
  customersByRegion,
  customersOf,
  type Input,
  projectView,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

const e = BALANCE_V2.economy;
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

/** A running world on the test balance where u1 owns company c1. */
function world(seed = 5): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
  const palette = { primary: "#4f8cff", secondary: "#ffb020" };
  s = ok(step(s, cmd({ type: "createCompany", name: "Acme", logo: LOGO, palette }, "c1")));
  return ok(step(s, cmd({ type: "startWorld" })));
}

/** c1's scored v1, launched on every plan and live for `days` days. */
function launched(days: number): WorldState {
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  let s = ok(step(world(), cmd({ type: "startTraining", plan }, "r1")));
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
  return ticks(s, 3 + days);
}

const inRegion = (s: WorldState, region: (typeof REGION_IDS)[number], group: string) =>
  PLAN_IDS.reduce(
    (n, p) => n + customersOf(s.market, "c1", region, group as (typeof GROUP_IDS)[number], p),
    0,
  );

describe("regions", () => {
  it("are the nine US divisions with their people, group mixes and neighbours", () => {
    expect(REGION_IDS).toHaveLength(9);
    expect(REGION_IDS.reduce((n, r) => n + e.population[r], 0)).toBe(336_100_000);
    for (const r of REGION_IDS) {
      expect(GROUP_IDS.reduce((n, g) => n + e.groupMix[r][g], 0)).toBeCloseTo(1, 9);
      expect(e.neighbors[r].length).toBeGreaterThan(0);
      for (const n of e.neighbors[r]) expect(e.neighbors[n]).toContain(r);
    }
    expect(e.groupMix.pacific.developers).toBeGreaterThan(e.groupMix.midSouth.developers);
  });

  it("count a region's willing people from its own population and mix", () => {
    const want = e.population.pacific * adoptionShare(e, 100) * e.groupMix.pacific.developers;
    expect(adoptable(e, 100, "pacific", "developers")).toBeCloseTo(want, 6);
  });

  it("all get customers from a launch, each in its own mix", () => {
    const s = launched(10);
    const byRegion = customersByRegion(s.market, "c1");
    for (const r of REGION_IDS) expect(byRegion[r]).toBeGreaterThan(0);
    const share = (r: (typeof REGION_IDS)[number]) =>
      inRegion(s, r, "developers") / (byRegion[r] ?? 1);
    expect(share("pacific")).toBeGreaterThan(share("midSouth"));
  });

  it("pass word of mouth to neighbouring regions, not distant ones", () => {
    const s0 = structuredClone(world());
    s0.market.customers.c1 = { pacific: { developers: { free: 50_000 } } };
    const s = ticks(s0, 1);
    expect(s.market.awareness.c1?.mountain?.developers ?? 0).toBeGreaterThan(0);
    expect(s.market.awareness.c1?.newEngland?.developers ?? 0).toBe(0);
  });

  it("give each spotlight customer a region, the same for the same seed", () => {
    const a = world(7).spotlights.map((p) => p.region);
    expect(a.every((r) => (REGION_IDS as readonly string[]).includes(r))).toBe(true);
    expect(world(7).spotlights.map((p) => p.region)).toEqual(a);
  });

  it("show in the view: willing people by region, public rounded counts, private exact ones", () => {
    const s = launched(10);
    const mine = WorldView.parse(projectView(s, "u1"));
    const regions = mine.world.market.regions;
    const sum = REGION_IDS.reduce((n, r) => n + regions[r].adoptable, 0);
    expect(Math.abs(sum - mine.world.market.adoptable)).toBeLessThanOrEqual(9);
    const exact = customersByRegion(s.market, "c1");
    const pub = mine.companies.find((c) => c.id === "c1");
    for (const r of REGION_IDS) {
      const whole = Math.round(exact[r] ?? 0);
      const step3 = whole < 1000 ? 1 : 10 ** (String(whole).length - 3);
      expect(pub?.customersByRegion[r]).toBe(Math.round(whole / step3) * step3);
      const byGroup = mine.me.company?.customers.byRegion[r];
      const total = GROUP_IDS.reduce((n, g) => n + (byGroup?.[g] ?? 0), 0);
      expect(Math.abs(total - whole)).toBeLessThanOrEqual(6);
    }
    const today = mine.me.company?.history[String(s.day)];
    const regionsTotal = REGION_IDS.reduce((n, r) => n + (today?.regions[r] ?? 0), 0);
    expect(Math.abs(regionsTotal - (mine.me.company?.customers.total ?? 0))).toBeLessThanOrEqual(9);
  });
});
