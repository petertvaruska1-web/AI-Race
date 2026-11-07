import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FeedbackJobSpec, GROUP_IDS, type JobType } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import type { WorldActor } from "../src/worlds/actor.ts";
import { eventually, FakeWorker } from "./fakeWorker.ts";
import {
  closeSockets,
  openSockets,
  owner,
  send,
  setup,
  start,
  toDay,
  trainVersion,
  worker,
} from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
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
async function launch(s: TestServer, w: FakeWorker, o: Owner, name = "Spark") {
  const versionId = await trainVersion(s, w, o);
  const evalLease = await w.nextLease();
  expect(evalLease.jobType).toBe("eval");
  w.done(evalLease.jobId, EVAL);
  await eventually(() => companyOf(o).versions[versionId]?.scores);
  await send(o.ws, o.worldId, {
    type: "launchProduct",
    versionId,
    name,
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

  it("drops questions that waited more than 14 game days for a card, and answers the rest", async () => {
    server = await start();
    const w = await worker(server, ALL_JOBS);
    const o = await setup(server);
    const l = await launch(server, w, o);
    w.close();
    await eventually(() => server?.ctx.broker.connections().length === 0);
    await toDay(server, o.actor, l.launchDay + 20);
    const day = o.actor.state().day;
    const asked = await feedbackJobs(server);
    const stale = asked.filter(
      (j) => (j.payload as { enqueuedDay: number }).enqueuedDay < day - 14,
    );
    const fresh = asked.filter(
      (j) => (j.payload as { enqueuedDay: number }).enqueuedDay >= day - 14,
    );
    expect(stale.length).toBeGreaterThan(0);
    expect(fresh.length).toBeGreaterThan(0);

    const back = await worker(server, { ...ALL_JOBS, workerId: "back" });
    for (const _ of fresh) {
      const lease = await back.nextLease();
      expect(fresh.map((j) => j.id)).toContain(lease.jobId);
      back.done(lease.jobId, { results: RESULTS });
    }
    await eventually(async () =>
      (await feedbackJobs(server as TestServer)).every(
        (j) => j.status === "done" || j.status === "cancelled",
      ),
    );
    for (const job of await feedbackJobs(server)) {
      const old = stale.some((j) => j.id === job.id);
      expect([job.status, old ? job.error : null]).toEqual(
        old ? ["cancelled", "stale"] : ["done", null],
      );
    }
    // A dropped week is never asked again.
    await toDay(server, o.actor, day + 1);
    const keyOf = (j: { payload: unknown }) => (j.payload as { key: string }).key;
    const after = await feedbackJobs(server);
    for (const j of stale) expect(after.filter((a) => keyOf(a) === keyOf(j))).toHaveLength(1);
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

describe("licensed answers", () => {
  async function keyOf(s: TestServer, cookie: string): Promise<string> {
    const res = await api(s, "/api/helper/key", { cookie });
    return ((await res.json()) as { key: string }).key;
  }

  /** Ana (host, house card) sells Spark; Ben (his own card) sells Owl and licenses Spark for students. */
  async function licensedWorld(s: TestServer) {
    const house = await worker(s, ALL_JOBS);
    const ana = await setup(s);
    const spark = await launch(s, house, ana, "Spark");
    const g = await createGuest(s, "Ben");
    const res = await api(s, "/api/worlds/join", {
      body: { code: ana.actor.state().inviteCode },
      cookie: g.cookie,
    });
    expect(res.status).toBe(200);
    const ben = {
      ...g,
      ws: await owner(s, g.cookie, ana.worldId),
      worldId: ana.worldId,
      actor: ana.actor,
    };
    await send(ben.ws, ben.worldId, {
      type: "createCompany",
      name: "Bolt",
      logo: { shape: "circle", symbol: "spark", initials: "BO" },
      palette: { primary: "#4f8cff", secondary: "#ffb020" },
    });
    const bensCard = await FakeWorker.connect(s, {
      ...ALL_JOBS,
      token: await keyOf(s, ben.cookie),
      workerId: "ben-pc",
      ready: true,
    });
    openSockets.push(bensCard);
    const owl = await launch(s, bensCard, ben, "Owl");
    await toDay(s, ana.actor, Math.max(spark.launchDay, owl.launchDay) + 1);
    await send(ana.ws, ana.worldId, {
      type: "offerLicense",
      versionId: spark.versionId,
      priceCents: 300,
      months: 6,
      groups: ["students"],
      exclusive: false,
    });
    const offerId = ana.actor.state().licenses.offers[0]?.id;
    await send(ben.ws, ben.worldId, {
      type: "takeLicense",
      offerId,
      productId: owl.productId,
      groups: ["students"],
    });
    expect(ana.actor.state().licenses.deals).toHaveLength(1);
    return { house, ana, ben, bensCard, spark, owl };
  }

  it("splits a licensee's batch: the licensed groups' questions go to the licensed AI on its maker's card", async () => {
    server = await start();
    const { house, ana, ben, spark, owl } = await licensedWorld(server);
    const dealId = ana.actor.state().licenses.deals[0]?.id as string;
    await toDay(server, ana.actor, ana.actor.state().day + 7);
    const week = `w${Math.floor(ana.actor.state().day / 7)}`;
    const jobs = (await feedbackJobs(server)).filter(
      (j) => (j.payload as { productId: string }).productId === owl.productId,
    );
    const own = jobs.find((j) => (j.payload as { key: string }).key === week);
    const lic = jobs.find((j) => (j.payload as { key: string }).key === `${week}~${dealId}`);
    expect(own?.ownerUserId).toBe(ben.userId);
    expect(lic?.ownerUserId).toBe(ana.userId);
    expect(lic?.payload).toMatchObject({ versionId: spark.versionId, groups: ["students"] });
    expect((own?.payload as { groups?: string[] } | undefined)?.groups).not.toContain("students");

    // The house card (Ana's) answers the licensed questions with Spark.
    const REPLY = "Spark says <b>4</b> <|end|> 😀";
    for (;;) {
      const lease = await house.nextLease();
      const spec = FeedbackJobSpec.parse(lease.spec);
      if (lease.jobId !== lic?.id) {
        house.done(lease.jobId, { results: RESULTS });
        continue;
      }
      expect(spec).toMatchObject({ versionId: spark.versionId, groups: ["students"] });
      const before = ana.actor.state().market.satisfaction[owl.productId]?.students ?? 0.5;
      house.done(lease.jobId, {
        results: [
          { group: "students", questionId: "q1", question: "2 + 2?", reply: REPLY, score: 1 },
          // A stray group the job wasn't asked about is dropped.
          { group: "developers", questionId: "q2", question: "Code?", reply: "x", score: 0 },
        ],
      });
      const post = await eventually(() =>
        ana.actor.state().feed.find((i) => i.kind === "post" && i.reply === REPLY),
      );
      expect(post).toMatchObject({
        companyId: companyOf(ben).id,
        productName: "Owl",
        poweredBy: "Spark",
      });
      expect(ana.actor.state().feed.some((i) => i.kind === "post" && i.reply === "x")).toBe(false);
      expect(ana.actor.state().market.satisfaction[owl.productId]?.students ?? 0).toBeGreaterThan(
        before,
      );
      break;
    }
  }, 90_000);
});
