import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FeedbackJobSpec, GROUP_IDS, type JobType } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import type { WorldActor } from "../src/worlds/actor.ts";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import { closeSockets, owner, send, setup, start, toDay, trainVersion, worker } from "./flows.ts";
import type { TestServer } from "./harness.ts";
import type { WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

const ALL_JOBS: { jobTypes: JobType[] } = { jobTypes: ["train", "eval", "generate", "feedback"] };
const EVAL = {
  report: {
    suite: "bench-v1",
    scores: { language: { score: 41.5, raw: 0.415, n: 40 } },
    overall: 41.5,
    missing: [],
    items: [],
    seconds: 1,
  },
  traits: { curious: 0.4 },
  personalityWords: ["curious"],
};
const RESULTS = GROUP_IDS.map((group, i) => ({
  group,
  questionId: `q-${group}`,
  question: `What would ${group} ask?`,
  reply: i === 5 ? "<b>x</b> <|end|>" : `A reply for ${group}.`,
  score: i / 5,
}));

type Owner = { ws: WsClient; worldId: string; userId: string; actor: WorldActor };

function companyOf(o: Owner): CompanyState {
  const s = o.actor.state();
  const company = s.companies[s.members[o.userId]?.companyId ?? ""];
  if (!company) throw new Error("no company");
  return company;
}

/** Trains and scores the owner's first AI, then launches it on free and personal in 3 days. */
async function launch(s: TestServer, w: FakeWorker, o: Owner) {
  const versionId = await trainVersion(s, w, o);
  const evalLease = await w.nextLease();
  expect(evalLease.jobType).toBe("eval");
  w.done(evalLease.jobId, EVAL);
  await eventually(() => companyOf(o).versions[versionId]?.scores);
  await send(o.ws, o.worldId, {
    type: "launchProduct",
    versionId,
    name: "Spark",
    plans: ["free", "personal"],
    countdownDays: 3,
  });
  const product = Object.values(companyOf(o).products)[0];
  if (!product) throw new Error("no product");
  return { versionId, productId: product.id, launchDay: product.launchDay };
}

const feedbackJobs = async (s: TestServer) =>
  s.ctx.db.select().from(schema.jobs).where(eq(schema.jobs.type, "feedback"));

describe("customer feedback jobs", () => {
  it("asks a live product's AI six customer questions and posts its best and worst replies", async () => {
    server = await start();
    const w = await worker(server, ALL_JOBS);
    const o = await setup(server);
    const l = await launch(server, w, o);
    await toDay(server, o.actor, l.launchDay);

    const lease = await w.nextLease();
    expect(lease.jobType).toBe("feedback");
    const spec = FeedbackJobSpec.parse(lease.spec);
    expect(spec).toMatchObject({ productId: l.productId, versionId: l.versionId, perGroup: 1 });
    expect([...spec.groups].sort()).toEqual([...GROUP_IDS].sort());
    expect(await (await fetch(spec.modelTarUrl)).text()).toBe("fake-checkpoint");

    w.done(lease.jobId, { results: RESULTS });
    await eventually(() => o.actor.state().feed.some((i) => i.kind === "post"));
    o.ws.send({ v: 1, type: "resync", worldId: o.worldId });
    const view = (await o.ws.next("snapshot")).view;
    const posts = view.feed.filter((i) => i.kind === "post");
    expect(posts).toHaveLength(2);
    expect(posts.map((p) => p.kind === "post" && p.reply)).toContain("<b>x</b> <|end|>");
  });

  it("asks once per week and launch, and launches once, across restarts", async () => {
    const dir = mkdtempSync(join(tmpdir(), "airace-feedback-"));
    temps.push(dir);
    const over = { dbDir: join(dir, "db"), blobDir: join(dir, "blobs") };
    server = await start(over);
    let w = await worker(server, ALL_JOBS);
    const o = await setup(server);
    const l = await launch(server, w, o);

    // Restart in the middle of the countdown.
    await server.close();
    closeSockets();
    server = await start(over);
    w = await worker(server, ALL_JOBS);
    let ws = await owner(server, o.cookie, o.worldId);
    let actor = await server.ctx.worlds.get(o.worldId);
    await toDay(server, actor, l.launchDay);
    await eventually(async () => (await feedbackJobs(server as TestServer)).length === 2);
    const keys = await server.ctx.db.select().from(schema.feedbackRequests);
    expect(keys.map((k) => k.key).sort()).toEqual(
      ["launch", `w${Math.floor(l.launchDay / 7)}`].sort(),
    );

    // Restart on the launch day: nothing is asked twice, nothing launches twice.
    await server.close();
    closeSockets();
    server = await start(over);
    w = await worker(server, ALL_JOBS);
    ws = await owner(server, o.cookie, o.worldId);
    actor = await server.ctx.worlds.get(o.worldId);
    await toDay(server, actor, l.launchDay + 1);
    const s = actor.state();
    const company = s.companies[s.members[o.userId]?.companyId ?? ""];
    expect(Object.values(company?.products ?? {}).map((p) => p.status)).toEqual(["live"]);
    expect(s.feed.filter((i) => i.kind === "news" && i.topic === "launch")).toHaveLength(1);
    expect(await feedbackJobs(server)).toHaveLength(2);
    expect(ws).toBeDefined();
  });

  it("serves a chat reply before a waiting feedback job", async () => {
    server = await start();
    // The services build real specs at lease time; here the raw payload is the spec.
    server.ctx.broker.onPrepare("feedback", async (job) => job.payload);
    server.ctx.broker.onPrepare("generate", async (job) => job.payload);
    const f = await server.ctx.broker.enqueue("feedback", { productId: "p" }, null);
    const g = await server.ctx.broker.enqueue("generate", { text: "hi" }, null);
    const w = await worker(server, {
      jobTypes: ["generate", "feedback"],
      slots: { train: 0, eval: 0, generate: 1, feedback: 1 },
    });
    expect((await w.nextLease()).jobId).toBe(g);
    expect((await w.nextLease()).jobId).toBe(f);
  });
});
