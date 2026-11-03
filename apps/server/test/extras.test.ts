import { GROUP_IDS } from "@airace/protocol";
import { BALANCE_V2, type CompanyState } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { toTrainRunConfig } from "../src/training/config.ts";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import { closeSockets, plan, send, setup, start, toDay, trainVersion, worker } from "./flows.ts";
import type { TestServer } from "./harness.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

const EVAL = {
  report: {
    suite: "bench-v1",
    scores: { knowledge: { score: 60, raw: 0.6, n: 40 } },
    overall: 50,
    missing: [],
    items: [],
    seconds: 1,
  },
  traits: { curious: 0.4 },
  personalityWords: [],
};

type Owner = Awaited<ReturnType<typeof setup>>;
const companyOf = (o: Owner) => {
  const s = o.actor.state();
  return s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
};

/** The owner's first AI, trained and scored; `keynote` books a launch keynote with it. */
async function launched(s: TestServer, w: FakeWorker, o: Owner, keynote: string | null) {
  const versionId = await trainVersion(s, w, o);
  const evalLease = await w.nextLease();
  w.done(evalLease.jobId, EVAL);
  await eventually(() => companyOf(o).versions[versionId]?.scores);
  await send(o.ws, o.worldId, {
    type: "launchProduct",
    versionId,
    name: "Spark",
    plans: ["free", "personal"],
    countdownDays: 3,
    keynote: keynote ? { questionId: keynote } : null,
  });
  const product = Object.values(companyOf(o).products)[0];
  if (!product) throw new Error("no product");
  return { versionId, productId: product.id };
}

describe("launch keynotes", () => {
  it("ask the AI its demo question as a job and tell the world the real reply", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "keynote"] });
    const o = await setup(server);
    const { productId, versionId } = await launched(server, w, o, "knowledge-customer-0003");
    const lease = await w.nextLease();
    expect(lease.jobType).toBe("keynote");
    expect(lease.spec).toMatchObject({ versionId, questionId: "knowledge-customer-0003" });
    w.done(lease.jobId, {
      questionId: "knowledge-customer-0003",
      prompt: "What is the capital of France?",
      reply: "Paris. <|end|>",
      score: 1,
    });
    await eventually(() => companyOf(o).products[productId]?.keynote?.status === "done");
    expect(companyOf(o).products[productId]?.keynote?.result?.reply).toBe("Paris. <|end|>");
  });

  it("refund a keynote whose job fails for good", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "keynote"] });
    const o = await setup(server);
    const { productId } = await launched(server, w, o, "no-such-question");
    const cash = companyOf(o).cash;
    const lease = await w.nextLease();
    w.failed(lease.jobId, "no customer question 'no-such-question'", false);
    await eventually(() => companyOf(o).products[productId]?.keynote?.status === "failed");
    expect(companyOf(o).cash).toBeGreaterThan(cash);
  });
});

describe("partner app reviews", () => {
  it("ask a live app's AI two fresh questions a week and post the reviews", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "appReview"] });
    const o = await setup(server);
    const { productId, versionId } = await launched(server, w, o, null);
    await toDay(server, o.actor, o.actor.state().day + 4);
    await send(o.ws, o.worldId, { type: "fundApp", kind: "puzzles", productId });
    const app = await eventually(() => Object.values(companyOf(o).apps)[0]);
    const lease = await w.nextLease();
    expect(lease.jobType).toBe("appReview");
    expect(lease.spec).toMatchObject({ versionId, skill: "pattern", n: 2 });
    w.done(lease.jobId, {
      reviews: [
        { prompt: "Q1?", reply: "A1", score: 1 },
        { prompt: "Q2?", reply: "A2", score: 0.5 },
      ],
    });
    await eventually(() => (companyOf(o).apps[app.id]?.reviews.length ?? 0) === 2);
    expect(companyOf(o).apps[app.id]?.rating).toBe(4);
  });
});

describe("loved replies and training extras", () => {
  it("keep the replies customers loved, and a run that learns from them gets them at 5%", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "feedback"] });
    const o = await setup(server);
    const { productId } = await launched(server, w, o, null);
    await toDay(server, o.actor, o.actor.state().day + 4);
    const lease = await w.nextLease();
    expect(lease.jobType).toBe("feedback");
    const results = GROUP_IDS.flatMap((group) =>
      [0, 1].map((k) => ({
        group,
        questionId: `q-${group}-${k}`,
        question: `What would ${group} ask ${k}?`,
        reply: `A loved reply ${k}.`,
        score: 0.9,
      })),
    );
    w.done(lease.jobId, { results });
    await eventually(() => companyOf(o).loved === 12);
    const rows = await server.ctx.db
      .select()
      .from(schema.lovedReplies)
      .where(eq(schema.lovedReplies.companyId, companyOf(o).id));
    expect(rows).toHaveLength(12);
    expect(productId).toBeTruthy();

    const before = Object.keys(companyOf(o).runs).length;
    await send(o.ws, o.worldId, {
      type: "startTraining",
      plan: { ...plan(), learnFromCustomers: true },
    });
    await eventually(() => Object.keys(companyOf(o).runs).length > before);
    const train = await eventually(async () => {
      const jobs = await server?.ctx.db
        .select()
        .from(schema.jobs)
        .where(eq(schema.jobs.type, "train"));
      return jobs?.at(-1);
    });
    const config = (
      train.payload as { config: { coaching: [string, string][]; mixture: Record<string, number> } }
    ).config;
    expect(config.coaching).toHaveLength(12);
    expect(config.coaching[0]?.[1]).toMatch(/A loved reply/);
    expect(config.mixture.coaching).toBeCloseTo(0.05);
  });

  it("tutoring packs add their dataset's share on top, and expert cleaning passes through", () => {
    const base = toTrainRunConfig(BALANCE_V2.defaultPlan, {
      runId: "r",
      seed: 1,
      balance: BALANCE_V2,
      parent: null,
    });
    const config = toTrainRunConfig(
      { ...BALANCE_V2.defaultPlan, cleaning: "expert", tutoring: { dataset: "code", packs: 2 } },
      { runId: "r", seed: 1, balance: BALANCE_V2, parent: null },
    );
    const budget = base.token_budget as number;
    expect(config.token_budget).toBe(Math.round(budget * 1.2));
    const mix = config.mixture as Record<string, number>;
    const total = Object.values(mix).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1);
    const baseMix = base.mixture as Record<string, number>;
    const baseTotal = Object.values(baseMix).reduce((a, b) => a + b, 0);
    const codeShare = ((baseMix.code ?? 0) / baseTotal) * (1 / 1.2) + 0.2 / 1.2;
    expect(mix.code).toBeCloseTo(codeShare);
    expect((config.prep as { cleaning: string }).cleaning).toBe("expert");
  });
});
