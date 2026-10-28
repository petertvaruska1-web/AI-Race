import type { Command } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  createWorld,
  type Input,
  liveProductFor,
  type StepResult,
  stateHash,
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

function rejected(r: StepResult): string {
  if (r.ok) throw new Error("expected a rejection");
  return r.reason;
}

const cmd = (c: unknown, newId?: string): Input => ({
  kind: "command",
  userId: "u1",
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;
const TICK: Input = { kind: "tick" };

function ticks(s: WorldState, n: number): WorldState {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, TICK));
  return out;
}

/** A running world where u1's company c1 has trained version v1 (scored unless `scored` is false). */
function withVersion(scored = true): WorldState {
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
  if (scored) {
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
  }
  return s;
}

const launch = (over: Record<string, unknown> = {}, id = "p1"): Input =>
  cmd(
    {
      type: "launchProduct",
      versionId: "v1",
      name: "Spark",
      plans: ["free", "personal"],
      countdownDays: 3,
      ...over,
    },
    id,
  );

const company = (s: WorldState) => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};

describe("launching products", () => {
  it("refuses to launch an AI that has not been scored", () => {
    expect(rejected(step(withVersion(false), launch()))).toMatch(/scored/i);
  });

  it("counts down, builds hype, then goes live on its plans", () => {
    let s = ok(step(withVersion(), launch()));
    const p = company(s).products.p1;
    expect(p).toMatchObject({
      status: "countdown",
      announcedDay: s.day,
      launchDay: s.day + 3,
      hype: 0,
    });
    expect(company(s).plans.personal.productId).toBeNull();
    s = ticks(s, 2);
    expect(company(s).products.p1?.status).toBe("countdown");
    expect(company(s).products.p1?.hype).toBeGreaterThan(0);
    s = ticks(s, 1);
    expect(company(s).products.p1?.status).toBe("live");
    expect(company(s).plans.free.productId).toBe("p1");
    expect(company(s).plans.personal.productId).toBe("p1");
    expect(company(s).plans.business.productId).toBeNull();
    expect(liveProductFor(company(s), "personal")?.id).toBe("p1");
    expect(liveProductFor(company(s), "business")).toBeNull();
  });

  it("allows at most three products on sale or counting down", () => {
    let s = withVersion();
    s = ok(step(s, launch({ name: "One" }, "p1")));
    s = ok(step(s, launch({ name: "Two", plans: [] }, "p2")));
    s = ok(step(s, launch({ name: "Three", plans: [] }, "p3")));
    expect(rejected(step(s, launch({ name: "Four", plans: [] }, "p4")))).toMatch(/three|3/i);
  });

  it("keeps names unique, clean and within 2–24 characters", () => {
    let s = ok(step(withVersion(), launch({ name: "Spark" })));
    expect(rejected(step(s, launch({ name: "SPARK" }, "p2")))).toMatch(/name/i);
    expect(rejected(step(s, launch({ name: "x".repeat(25) }, "p2")))).toMatch(/2–24/);
    s = ok(step(s, launch({ name: "<b>Swift</b>", plans: [] }, "p2")));
    expect(company(s).products.p2?.name).toBe("<b>Swift</b>");
  });

  it("refuses a product id that another company already uses", () => {
    let s = ok(step(withVersion(), launch()));
    s = ok(step(s, sys({ type: "memberJoined", userId: "u2", displayName: "Ben" })));
    s = ok(
      step(s, {
        kind: "command",
        userId: "u2",
        cmd: {
          type: "createCompany",
          name: "Beta",
          logo: LOGO,
          palette: { primary: "#4f8cff", secondary: "#ffb020" },
        },
        newId: "c2",
      }),
    );
    const theirs = step(s, { ...launch(), userId: "u2" } as Input);
    expect(rejected(theirs)).toMatch(/id is already used/i);
  });

  it("refuses countdowns outside 3–14 days", () => {
    expect(rejected(step(withVersion(), launch({ countdownDays: 2 })))).toMatch(/3–14/);
    expect(rejected(step(withVersion(), launch({ countdownDays: 15 })))).toMatch(/3–14/);
  });
});

describe("plans and prices", () => {
  it("puts plans only on live products, and closes them", () => {
    let s = ok(step(withVersion(), launch({ plans: [] })));
    expect(rejected(step(s, cmd({ type: "setPlan", plan: "business", productId: "p1" })))).toMatch(
      /sale/i,
    );
    s = ticks(s, 3);
    s = ok(step(s, cmd({ type: "setPlan", plan: "business", productId: "p1" })));
    expect(company(s).plans.business.productId).toBe("p1");
    s = ok(step(s, cmd({ type: "setPlan", plan: "business", productId: null })));
    expect(company(s).plans.business.productId).toBeNull();
  });

  it("keeps prices within their bounds", () => {
    const s = withVersion();
    expect(company(s).plans.personal.price).toBe(10);
    expect(company(s).plans.business.price).toBe(30);
    expect(rejected(step(s, cmd({ type: "setPrice", plan: "personal", price: 0 })))).toMatch(
      /\$1–\$200/,
    );
    expect(rejected(step(s, cmd({ type: "setPrice", plan: "personal", price: 201 })))).toMatch(
      /\$1–\$200/,
    );
    expect(
      company(ok(step(s, cmd({ type: "setPrice", plan: "business", price: 45 })))).plans.business
        .price,
    ).toBe(45);
  });

  it("retires a product only once no plan uses it", () => {
    let s = ticks(ok(step(withVersion(), launch())), 3);
    expect(rejected(step(s, cmd({ type: "retireProduct", productId: "p1" })))).toMatch(
      /move its plans/i,
    );
    s = ok(step(s, cmd({ type: "setPlan", plan: "free", productId: null })));
    s = ok(step(s, cmd({ type: "setPlan", plan: "personal", productId: null })));
    s = ok(step(s, cmd({ type: "retireProduct", productId: "p1" })));
    expect(company(s).products.p1?.status).toBe("retired");
  });

  it("refuses ids named like built-in object parts, changing nothing anywhere", () => {
    const s = ticks(ok(step(withVersion(), launch())), 3);
    const before = stateHash(s);
    const plan = {
      ...BALANCE_TEST.defaultPlan,
      mix: { ...BALANCE_TEST.defaultPlan.mix },
      parentVersionId: "__proto__",
    };
    const polluted = () =>
      "status" in Object.prototype || Object.hasOwn(Object, "status") || "status" in {};
    for (const bad of [
      cmd({ type: "retireProduct", productId: "__proto__" }),
      cmd({ type: "retireProduct", productId: "constructor" }),
      cmd({ type: "setPlan", plan: "free", productId: "__proto__" }),
      launch({ versionId: "__proto__", name: "Ghost" }, "p2"),
      launch({ versionId: "constructor", name: "Ghost" }, "p3"),
      cmd({ type: "startTraining", plan }, "r9"),
    ]) {
      const r = step(s, bad);
      const dirty = polluted();
      // Clean up before asserting, so a failure here can't spill into other tests.
      delete (Object.prototype as unknown as Record<string, unknown>).status;
      delete (Object as unknown as Record<string, unknown>).status;
      expect(dirty).toBe(false);
      expect(r.ok).toBe(false);
    }
    expect(stateHash(s)).toBe(before);
  });

  it("leaves the world untouched by every refusal", () => {
    const s = ticks(ok(step(withVersion(), launch())), 3);
    const before = stateHash(s);
    for (const bad of [
      launch({ name: "SPARK" }, "p2"),
      cmd({ type: "setPrice", plan: "personal", price: 999 }),
      cmd({ type: "retireProduct", productId: "p1" }),
      cmd({ type: "setPlan", plan: "free", productId: "nope" }),
    ]) {
      expect(step(s, bad).ok).toBe(false);
      expect(stateHash(s)).toBe(before);
    }
  });
});
