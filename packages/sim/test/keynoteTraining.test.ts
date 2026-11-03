import type { TrainingPlan } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_V2,
  type CompanyState,
  getBalance,
  previewPlan,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, ticks } from "./pack.ts";

const c1 = (s: WorldState) => s.companies.c1 as CompanyState;
const E = getBalance("balance-test").economy;

function scored(): WorldState {
  const s = structuredClone(runningWorld("balance-test"));
  c1(s).versions.v1 = {
    id: "v1",
    label: "v1",
    runId: "r1",
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: s.day,
    trainedTokens: 1,
    scores: { knowledge: 50 },
    overall: 30,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  };
  return s;
}
const launch = (
  s: WorldState,
  keynote: { questionId: string } | null = { questionId: "knowledge-customer-0003" },
) =>
  step(
    s,
    cmd(
      {
        type: "launchProduct",
        versionId: "v1",
        name: "Spark",
        plans: ["free"],
        countdownDays: 3,
        keynote,
      },
      "p1",
    ),
  );
const keynote = (s: WorldState, score: number, reply = "Paris is the capital.") =>
  step(
    s,
    sys({
      type: "keynoteResolved",
      productId: "p1",
      prompt: "What is the capital of France?",
      reply,
      score,
    }),
  );

describe("launch keynotes", () => {
  it("charge their fee when the launch is announced", () => {
    const s = scored();
    const after = ok(launch(s));
    expect(c1(after).cash).toBe(c1(s).cash - E.keynote.fee);
    expect(c1(after).ledger.current.costs.events).toBe(E.keynote.fee);
    expect(c1(after).products.p1?.keynote).toMatchObject({
      questionId: "knowledge-customer-0003",
      status: "waiting",
    });
    expect(c1(ok(launch(s, null))).cash).toBe(c1(s).cash);
  });

  it("a great answer before launch day makes a bigger launch and quotes the real reply", () => {
    const plain = ticks(ok(launch(scored(), null)), 4);
    const great = ticks(ok(keynote(ok(launch(scored())), 1)), 4);
    const aware = (s: WorldState) =>
      Object.values(s.market.awareness.c1 ?? {})
        .flatMap((r) => Object.values(r ?? {}))
        .reduce((a, b) => a + b, 0);
    expect(aware(great)).toBeGreaterThan(aware(plain));
    const news = great.feed.filter((f) => f.kind === "news" && f.topic === "keynote");
    expect(news[0]?.kind === "news" && news[0].headline).toBe(
      'Live on stage, Spark said: "Paris is the capital."',
    );
    expect(c1(great).products.p1?.keynote?.applied).toBe(true);
  });

  it("a bad answer flops", () => {
    const flop = ticks(ok(keynote(ok(launch(scored())), 0, "pigmpton.")), 4);
    const news = flop.feed.filter((f) => f.kind === "news" && f.topic === "keynote");
    expect(news[0]?.kind === "news" && news[0].headline).toBe(
      'Acme\'s keynote flops: on stage, Spark said "pigmpton."',
    );
    expect(news[0]?.kind === "news" && news[0].sentiment).toBeLessThan(0);
  });

  it("an answer after launch day still makes the news, once; a failed one is refunded", () => {
    const late = ok(keynote(ticks(ok(launch(scored())), 4), 0.8));
    expect(late.feed.filter((f) => f.kind === "news" && f.topic === "keynote")).toHaveLength(1);
    expect(keynote(late, 0.8).ok).toBe(false);
    const s = ok(launch(scored()));
    const failed = ok(step(s, sys({ type: "keynoteResolved", productId: "p1", failed: true })));
    expect(c1(failed).cash).toBe(c1(s).cash + E.keynote.fee);
    expect(c1(failed).products.p1?.keynote?.status).toBe("failed");
  });
});

describe("training extras", () => {
  const plan = (change: Partial<TrainingPlan> = {}): TrainingPlan => ({
    ...BALANCE_V2.defaultPlan,
    mix: { ...BALANCE_V2.defaultPlan.mix },
    ...change,
  });
  const b = BALANCE_V2;
  const base = previewPlan(b, null, plan());
  if (!base.ok) throw new Error(base.reason);

  it("tutoring packs add 10% of the tokens each, and their fee", () => {
    const r = previewPlan(b, null, plan({ tutoring: { dataset: "code", packs: 3 } }));
    if (!r.ok) throw new Error(r.reason);
    expect(r.preview.tokens).toBe(Math.round(base.preview.tokens * 1.3));
    expect(r.preview.cost).toBeGreaterThan(base.preview.cost + 3 * b.economy.tutoring.packFee - 1);
    expect(r.preview.durationDays).toBeGreaterThanOrEqual(base.preview.durationDays);
  });

  it("Fact-check plus costs by the text it checks", () => {
    const r = previewPlan(b, null, plan({ cleaning: "expert" }));
    if (!r.ok) throw new Error(r.reason);
    const expert = Math.round((base.preview.tokens / 1000) * b.economy.expertCheckPerThousand);
    expect(r.preview.cost).toBe(base.preview.cost + expert);
    expect(r.preview.checks).toBe(expert);
  });

  it("learning from customers needs ten loved conversations and costs by the conversation", () => {
    expect(previewPlan(b, null, plan({ learnFromCustomers: true }), 0, 9).ok).toBe(false);
    const r = previewPlan(b, null, plan({ learnFromCustomers: true }), 0, 50);
    if (!r.ok) throw new Error(r.reason);
    expect(r.preview.cost).toBe(base.preview.cost + 50 * b.economy.lovedConversationFee);
  });

  it("counts loved replies from customer feedback, up to 400", () => {
    const s = ticks(ok(launch(scored(), null)), 4);
    const results = [0.9, 0.8, 0.3].map((score, k) => ({
      group: "students",
      questionId: `q${k}`,
      question: "Q?",
      reply: "A",
      score,
    }));
    const fed = ok(step(s, sys({ type: "feedbackScored", productId: "p1", key: "w1", results })));
    expect(c1(fed).loved).toBe(2);
    const full = structuredClone(fed);
    c1(full).loved = 399;
    const more = ok(
      step(full, sys({ type: "feedbackScored", productId: "p1", key: "w2", results })),
    );
    expect(c1(more).loved).toBe(400);
  });

  it("a run that learns from customers is refused before any money moves without enough", () => {
    const s = ticks(ok(launch(scored(), null)), 4);
    const r = step(
      s,
      cmd(
        {
          type: "startTraining",
          plan: plan({ ...getBalance("balance-test").defaultPlan, learnFromCustomers: true }),
        },
        "r9",
      ),
    );
    expect(rejected(r)).toMatch(/loved/);
  });
});
