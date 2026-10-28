import { type Command, GROUP_IDS } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  adoptionShare,
  BALANCE_TEST,
  BALANCE_V2,
  choiceShares,
  createWorld,
  type Input,
  productAppeal,
  type StepResult,
  stateHash,
  step,
  type VersionState,
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

/** c1 trained and scored v1, and launched it on the free and personal plans (live). */
function launched(): WorldState {
  let s = world();
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

const total = (s: WorldState) =>
  Object.values(s.market.customers.c1?.national ?? {}).reduce(
    (sum, byPlan) => sum + Object.values(byPlan).reduce((a, b) => a + b, 0),
    0,
  );

const version = (scores: Record<string, number>): VersionState => ({
  id: "v",
  label: "v1",
  runId: "r",
  parentId: null,
  nLayer: 4,
  dModel: 128,
  ctxLen: 256,
  createdDay: 0,
  trainedTokens: 1,
  scores,
  overall: 40,
  personalityWords: [],
  traits: null,
  published: false,
});

describe("choice", () => {
  it("splits evenly between equal options and the no-AI choice", () => {
    const shares = choiceShares([0, 0], 0);
    expect(shares[0]).toBeCloseTo(1 / 3, 9);
    expect(shares[1]).toBeCloseTo(1 / 3, 9);
    const big = choiceShares([3, 1, -2], 0.5);
    expect(big.reduce((a, b) => a + b, 0)).toBeLessThan(1);
    expect(big.every((x) => x > 0)).toBe(true);
  });

  it("rewards the skills a group cares about", () => {
    const e = BALANCE_V2.economy;
    const coder = version({ ...SCORES, coding: 90 });
    const weak = version({ ...SCORES, coding: 10 });
    expect(productAppeal(e, "developers", coder, "personal", 10, 50, 0.5)).toBeGreaterThan(
      productAppeal(e, "developers", weak, "personal", 10, 50, 0.5),
    );
  });

  it("puts students off a price rise more than businesses", () => {
    const e = BALANCE_V2.economy;
    const v = version(SCORES);
    const drop = (g: "students" | "businesses") =>
      productAppeal(e, g, v, "personal", 10, 50, 0.5) -
      productAppeal(e, g, v, "personal", 20, 50, 0.5);
    expect(drop("students")).toBeGreaterThan(drop("businesses"));
  });
});

describe("customers", () => {
  it("has no customers without a live product", () => {
    expect(total(ticks(world(), 30))).toBe(0);
  });

  it("grows customers day by day after a launch, never past the people willing", () => {
    let s = launched();
    const day0 = total(s);
    s = ticks(s, 1);
    const day1 = total(s);
    s = ticks(s, 5);
    const day6 = total(s);
    expect(day1).toBeGreaterThan(day0);
    expect(day6).toBeGreaterThan(day1);
    const willing =
      BALANCE_TEST.economy.population.national * adoptionShare(BALANCE_TEST.economy, s.day);
    expect(day6).toBeLessThanOrEqual(willing);
  });

  it("loses a closed plan's customers gradually, never below zero", () => {
    let s = ticks(launched(), 10);
    const personal = () =>
      Object.values(s.market.customers.c1?.national ?? {}).reduce(
        (a, g) => a + (g.personal ?? 0),
        0,
      );
    const before = personal();
    expect(before).toBeGreaterThan(0);
    s = ok(step(s, cmd({ type: "setPlan", plan: "personal", productId: null })));
    s = ticks(s, 1);
    expect(personal()).toBeLessThan(before);
    expect(personal()).toBeGreaterThan(0);
    s = ticks(s, 400);
    expect(personal()).toBeGreaterThanOrEqual(0);
  });

  it("replays the market exactly", () => {
    expect(stateHash(ticks(launched(), 20))).toBe(stateHash(ticks(launched(), 20)));
  });
});

describe("spotlights", () => {
  it("draws six spotlight customers, one per group, the same for the same seed", () => {
    const a = world(9).spotlights;
    expect(a.map((p) => p.group).sort()).toEqual([...GROUP_IDS].sort());
    expect(world(9).spotlights).toEqual(a);
    expect(world(10).spotlights).not.toEqual(a);
    for (const p of a) {
      expect(p.name.length).toBeGreaterThan(1);
      expect(p.handle).toMatch(/^@[a-z0-9_]+$/);
      expect(p.region).toBe("national");
    }
  });
});
