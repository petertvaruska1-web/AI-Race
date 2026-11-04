import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  type CompanyState,
  type NewsItem,
  step,
  type TrainingResult,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, runningWorld, sys, ticks, withRivals } from "./pack.ts";

const RESULT: TrainingResult = { steps: 1, finalLoss: 3, heldoutLosses: {}, unstable: false };

const plan = (over: object) => ({ ...BALANCE_TEST.defaultPlan, ...over });
const webOnly = { ...BALANCE_TEST.defaultPlan.mix };

/** Trains one version on the test balance with `over` applied to the plan; returns its id. */
function train(s0: WorldState, runId: string, over: object): WorldState {
  let s = ok(step(s0, cmd({ type: "startTraining", plan: plan(over) }, runId)));
  s = ok(
    step(s, sys({ type: "trainingCompleted", runId, versionId: `ver-${runId}`, result: RESULT })),
  );
  return ticks(s, 40);
}

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;

/** c1 sells `versionId` as "Spark" to `customers` everyday customers. */
function selling(s0: WorldState, versionId: string, customers = 150): WorldState {
  const scored = ok(
    step(
      s0,
      sys({
        type: "evalCompleted",
        versionId,
        scores: { language: 10 },
        overall: 10,
        personalityWords: [],
      }),
    ),
  );
  let s = ok(
    step(
      scored,
      cmd(
        { type: "launchProduct", versionId, name: "Spark", plans: ["free"], countdownDays: 3 },
        "p1",
      ),
    ),
  );
  s = ticks(s, 3);
  s.market.customers.c1 = { newEngland: { everyday: { free: customers } } };
  s.market.satisfaction.p1 = { everyday: 0.5, students: 0.03 };
  return s;
}

const batch = (key: string, knowledgeScore: number) =>
  sys({
    type: "feedbackScored",
    productId: "p1",
    key,
    results: [
      {
        group: "everyday",
        questionId: "knowledge-customer-0003",
        question: "What is the capital of France?",
        reply: "Berlin",
        score: knowledgeScore,
      },
    ],
  });

const scandals = (s: WorldState) =>
  s.feed.filter((f): f is NewsItem => f.kind === "news" && f.topic === "scandal");

describe("the data-quality scandal", () => {
  it("versions remember learning from unchecked web data, and pass it on", () => {
    let s = runningWorld("balance-test");
    s = train(s, "r1", { mix: webOnly, factCheck: false });
    expect(c1(s).versions["ver-r1"]?.unchecked).toBe(true);
    s = train(s, "r2", { mix: { ...webOnly, web: 0 }, parentVersionId: "ver-r1" });
    expect(c1(s).versions["ver-r2"]?.unchecked).toBe(true); // learned it from its parent
    s = train(s, "r3", { mix: webOnly, factCheck: true });
    expect(c1(s).versions["ver-r3"]?.unchecked).toBe(false);
    s = train(s, "r4", { mix: webOnly, cleaning: "expert" }); // Fact-check plus checks facts
    expect(c1(s).versions["ver-r4"]?.unchecked).toBe(false);
  });

  it("fires once when customers catch a false fact at scale, and hurts satisfaction", () => {
    let s = train(runningWorld("balance-test"), "r1", { mix: webOnly, factCheck: false });
    s = selling(s, "ver-r1");
    s = ok(step(s, batch("w1", 1)));
    expect(scandals(s)).toHaveLength(0);
    const before = structuredClone(s.market.satisfaction.p1);
    s = ok(step(s, batch("w2", 0)));
    expect(scandals(s).map((n) => [n.headline, n.sentiment])).toEqual([
      ["Customers catch Spark stating false facts: Acme trained it on unchecked web data", -0.7],
    ]);
    expect(c1(s).products.p1?.scandal).toBe(true);
    // Every group's satisfaction falls by 0.05 after the batch's own update, never below 0.
    expect(s.market.satisfaction.p1?.students).toBe(0);
    expect(s.market.satisfaction.p1?.everyday).toBeLessThan((before?.everyday ?? 0) - 0.05);
    s = ok(step(s, batch("w3", 0)));
    expect(scandals(s)).toHaveLength(1);
  });

  it("doesn't fire for a fact-checked AI, below the customer mark, or for rivals", () => {
    const checked = selling(
      train(runningWorld("balance-test"), "r1", { mix: webOnly, factCheck: true }),
      "ver-r1",
    );
    expect(scandals(ok(step(checked, batch("w1", 0))))).toHaveLength(0);
    const small = selling(
      train(runningWorld("balance-test"), "r1", { mix: webOnly, factCheck: false }),
      "ver-r1",
      99,
    );
    expect(scandals(ok(step(small, batch("w1", 0))))).toHaveLength(0);
    const rivals = ticks(withRivals("balance-test"), 8);
    const arch = rivals.companies["rival-1"] as CompanyState;
    const live = Object.values(arch.products).find((p) => p.status === "live");
    expect(live).toBeDefined();
    rivals.market.customers["rival-1"] = { newEngland: { everyday: { free: 1_000_000 } } };
    const r = step(rivals, {
      ...batch("w1", 0),
      event: { ...(batch("w1", 0) as { event: object }).event, productId: live?.id },
    } as never);
    expect(r.ok).toBe(true);
    if (r.ok) expect(scandals(r.state)).toHaveLength(0);
  });
});
