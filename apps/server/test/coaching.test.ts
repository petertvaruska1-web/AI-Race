import { randomUUID } from "node:crypto";
import { EvalJobSpec, TrainJobSpec } from "@airace/protocol";
import { BALANCE_TEST } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { NOT_OWN_AI, PICK_A_REPLY } from "../src/coaching/service.ts";
import { schema } from "../src/db/index.ts";
import { toTrainRunConfig } from "../src/training/config.ts";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import { closeSockets, owner, plan, send, setup, start, trainVersion, worker } from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
import type { WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

async function chat(ws: WsClient, worldId: string, versionId: string, text: string) {
  const cid = randomUUID();
  ws.send({ v: 1, type: "chat", cid, worldId, threadId: null, versionId, text });
  return (await ws.next("ack", (m) => m.cid === cid)).data as { threadId: string };
}

async function coach(
  ws: WsClient,
  worldId: string,
  threadId: string,
  idx: number,
  kind: "liked" | "written",
  text?: string,
): Promise<string | null> {
  const cid = randomUUID();
  ws.send({ v: 1, type: "lesson", cid, worldId, threadId, idx, kind, ...(text ? { text } : {}) });
  return Promise.race([
    ws.next("ack", (m) => m.cid === cid).then(() => null),
    ws.next("reject", (m) => m.cid === cid).then((m) => m.reason),
  ]);
}

async function lessonsOf(ws: WsClient, worldId: string) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q: { kind: "lessons", worldId } });
  return ws.next("queryResult", (m) => m.id === id);
}

/** Ana's trained v1, and a finished conversation with it: "What is 2+2?" → "Four." */
async function talked(s: TestServer) {
  const w: FakeWorker = await worker(s, {
    jobTypes: ["train", "generate"],
    slots: { train: 1, eval: 0, generate: 1 },
  });
  const o = await setup(s);
  const versionId = await trainVersion(s, w, o);
  const { threadId } = await chat(o.ws, o.worldId, versionId, "What is 2+2?");
  const lease = await w.nextLease();
  w.done(lease.jobId, { text: "Four." });
  await eventually(async () => {
    const rows = await s.ctx.db
      .select()
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId));
    return rows.length === 2;
  });
  const companyId = o.actor.state().members[o.userId]?.companyId ?? "";
  return { ...o, w, versionId, threadId, companyId };
}

const countOf = (o: Awaited<ReturnType<typeof talked>>) =>
  o.actor.state().companies[o.companyId]?.lessons;

describe("lessons from the Playground", () => {
  it("a like and a written answer each make one lesson with the exact texts; a like twice makes one", async () => {
    server = await start();
    const o = await talked(server);
    expect(await coach(o.ws, o.worldId, o.threadId, 1, "liked")).toBeNull();
    expect(await coach(o.ws, o.worldId, o.threadId, 1, "liked")).toBeNull();
    expect(await coach(o.ws, o.worldId, o.threadId, 1, "written", "  It is 4.  ")).toBeNull();
    const rows = await server.ctx.db.select().from(schema.lessons);
    expect(rows.map((r) => [r.kind, r.question, r.answer]).sort()).toEqual([
      ["liked", "What is 2+2?", "Four."],
      ["written", "What is 2+2?", "It is 4."],
    ]);
    await eventually(() => countOf(o) === 2);
    const listed = (await lessonsOf(o.ws, o.worldId)).data as {
      lessons: { id: number; kind: string; question: string; answer: string }[];
    };
    expect(listed.lessons.map((l) => l.kind).sort()).toEqual(["liked", "written"]);

    // Deleting one counts it out.
    const cid = randomUUID();
    o.ws.send({
      v: 1,
      type: "lessonDelete",
      cid,
      worldId: o.worldId,
      lessonId: listed.lessons[0]?.id ?? 0,
    });
    await o.ws.next("ack", (m) => m.cid === cid);
    await eventually(() => countOf(o) === 1);
  });

  it("refuses a user message, an unfinished reply, a link, and someone else's AI", async () => {
    server = await start();
    const o = await talked(server);
    expect(await coach(o.ws, o.worldId, o.threadId, 0, "liked")).toBe(PICK_A_REPLY);
    expect(await coach(o.ws, o.worldId, o.threadId, 7, "liked")).toBe(PICK_A_REPLY);
    expect(await coach(o.ws, o.worldId, o.threadId, 1, "written")).toBe(
      "Write an answer of 1–500 characters.",
    );
    expect(await coach(o.ws, o.worldId, o.threadId, 1, "written", "see www.example.com")).toBe(
      "Answers can't hold links, email addresses or phone numbers.",
    );

    // Ben tries Ana's shared AI: his conversation, but not his AI.
    await send(o.ws, o.worldId, {
      type: "publishVersion",
      versionId: o.versionId,
      published: true,
    });
    const ben = await createGuest(server, "Ben");
    await api(server, "/api/worlds/join", {
      body: { code: o.actor.state().inviteCode },
      cookie: ben.cookie,
    });
    const benWs = await owner(server, ben.cookie, o.worldId);
    await send(benWs, o.worldId, {
      type: "createCompany",
      name: "Bolt",
      logo: { shape: "circle", symbol: "spark", initials: "BO" },
      palette: { primary: "#4f8cff", secondary: "#ffb020" },
    });
    const theirs = await chat(benWs, o.worldId, o.versionId, "Hi");
    const lease = await o.w.nextLease();
    o.w.done(lease.jobId, { text: "Hello." });
    await eventually(async () => {
      const rows = await server?.ctx.db
        .select()
        .from(schema.chatMessages)
        .where(eq(schema.chatMessages.threadId, theirs.threadId));
      return rows?.length === 2;
    });
    expect(await coach(benWs, o.worldId, theirs.threadId, 1, "liked")).toBe(NOT_OWN_AI);
    // Nor someone else's conversation.
    expect(await coach(benWs, o.worldId, o.threadId, 1, "liked")).toBe(NOT_OWN_AI);
    expect((await lessonsOf(benWs, o.worldId)).data).toEqual({ lessons: [] });
    expect(await server.ctx.db.select().from(schema.lessons)).toEqual([]);
  });

  it("keeps the newest 200", async () => {
    server = await start();
    const o = await talked(server);
    await server.ctx.db.insert(schema.lessons).values(
      Array.from({ length: 200 }, (_, k) => ({
        worldId: o.worldId,
        companyId: o.companyId,
        question: `Q${k}`,
        answer: `A${k}`,
        kind: "written",
        sourceThread: randomUUID(),
        sourceIdx: 1,
        at: k,
      })),
    );
    expect(await coach(o.ws, o.worldId, o.threadId, 1, "liked")).toBeNull();
    const rows = await server.ctx.db.select().from(schema.lessons);
    expect(rows).toHaveLength(200);
    expect(rows.some((r) => r.question === "Q0")).toBe(false);
    await eventually(() => countOf(o) === 200);
  });

  it("a run with coaching on trains on the lessons", async () => {
    server = await start();
    const o = await talked(server);
    for (let k = 0; k < 5; k++) {
      expect(await coach(o.ws, o.worldId, o.threadId, 1, "written", `Answer ${k}`)).toBeNull();
    }
    // A written answer to the same reply replaces the last one: five need five replies.
    await server.ctx.db.insert(schema.lessons).values(
      Array.from({ length: 4 }, (_, k) => ({
        worldId: o.worldId,
        companyId: o.companyId,
        question: `Q${k}`,
        answer: `A${k}`,
        kind: "written",
        sourceThread: randomUUID(),
        sourceIdx: 1,
        at: k,
      })),
    );
    await o.actor.submit({
      kind: "system",
      event: { type: "lessonsCounted", companyId: o.companyId, count: 5 },
    });
    await send(o.ws, o.worldId, { type: "startTraining", plan: { ...plan(), coaching: true } });
    const lease = await o.w.nextLease();
    const config = TrainJobSpec.parse(lease.spec).config as {
      coaching?: [string, string][];
      mixture: Record<string, number>;
    };
    expect(config.coaching).toHaveLength(5);
    expect(config.coaching).toContainEqual(["What is 2+2?", "Answer 4"]);
    expect(config.mixture.coaching).toBeCloseTo(0.05);
  });
});

describe("creativity knows the company's own text", () => {
  it("an eval carries the company's notebook and lessons, so copying them isn't new writing", async () => {
    server = await start();
    const o = await talked(server);
    expect(
      await coach(o.ws, o.worldId, o.threadId, 1, "written", "It is four, my friend."),
    ).toBeNull();
    const cid = randomUUID();
    o.ws.send({
      v: 1,
      type: "notebookSave",
      cid,
      worldId: o.worldId,
      text: "The marmalade lighthouse sings to the sea.",
    });
    await o.ws.next("ack", (m) => m.cid === cid);
    // v1's eval waited for a card that measures; this one takes it now.
    const measurer = await worker(server, { jobTypes: ["eval"] });
    const lease = await measurer.nextLease();
    const spec = EvalJobSpec.parse(lease.spec);
    expect(spec.versionId).toBe(o.versionId);
    expect(spec.ownText).toEqual(
      expect.arrayContaining([
        "The marmalade lighthouse sings to the sea.",
        "What is 2+2?",
        "It is four, my friend.",
      ]),
    );
  });
});

describe("the training config", () => {
  const base = { runId: "r", seed: 1, balance: BALANCE_TEST, parent: null };
  it("adds the lessons as coaching pairs at 5%, 10% with Learn from customers", () => {
    const lessons: [string, string][] = [["Q", "A"]];
    const loved: [string, string][] = [["L", "R"]];
    const on = toTrainRunConfig({ ...plan(), coaching: true }, { ...base, lessons });
    expect(on.coaching).toEqual(lessons);
    expect((on.mixture as Record<string, number>).coaching).toBeCloseTo(0.05);
    const both = toTrainRunConfig(
      { ...plan(), coaching: true, learnFromCustomers: true },
      { ...base, lessons, coaching: loved },
    );
    expect(both.coaching).toEqual([...loved, ...lessons]);
    const mix = both.mixture as Record<string, number>;
    expect(mix.coaching).toBeCloseTo(0.1);
    expect(Object.values(mix).reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    const off = toTrainRunConfig({ ...plan(), coaching: false }, { ...base, lessons });
    expect(off.coaching).toBeUndefined();
  });
});
