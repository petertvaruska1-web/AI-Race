import { randomUUID } from "node:crypto";
import { InterviewJobSpec } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import {
  closeSockets,
  openSockets,
  send,
  setup,
  start,
  toDay,
  trainVersion,
  worker,
} from "./flows.ts";
import { createGuest, type TestServer } from "./harness.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

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

/** The owner's first AI, scored and on sale as "Spark". */
async function onSale(s: TestServer, w: FakeWorker, o: Owner) {
  const versionId = await trainVersion(s, w, o);
  const evalLease = await w.nextLease();
  w.done(evalLease.jobId, EVAL);
  await eventually(() => companyOf(o).versions[versionId]?.scores);
  await send(o.ws, o.worldId, {
    type: "launchProduct",
    versionId,
    name: "Spark",
    plans: ["free"],
    countdownDays: 3,
  });
  const product = Object.values(companyOf(o).products)[0];
  if (!product) throw new Error("no product");
  await toDay(s, o.actor, product.launchDay);
  await eventually(() => companyOf(o).products[product.id]?.status === "live");
  return { versionId, productId: product.id };
}

async function ask(
  ws: WsClient,
  o: Owner,
  versionId: string,
  productId: string,
  text: string,
  cardId: string | null,
  threadId: string | null,
) {
  const cid = randomUUID();
  ws.send({
    v: 1,
    type: "chat",
    cid,
    worldId: o.worldId,
    threadId,
    versionId,
    text,
    interview: { productId, cardId },
  });
  const ack = await ws.next("ack", (m) => m.cid === cid);
  return (ack.data as { threadId: string }).threadId;
}

async function publish(ws: WsClient, o: Owner, threadId: string, idxs: number[]) {
  const cid = randomUUID();
  ws.send({ v: 1, type: "interviewPublish", cid, worldId: o.worldId, threadId, idxs });
  return Promise.race([
    ws.next("ack", (m) => m.cid === cid).then((m) => ({ ok: true as const, data: m.data })),
    ws.next("reject", (m) => m.cid === cid).then((m) => ({ ok: false as const, reason: m.reason })),
  ]);
}

/** An interview of two answers: idx 1 and 3 are the AI's. */
async function interview(s: TestServer, w: FakeWorker, o: Owner) {
  const { versionId, productId } = await onSale(s, w, o);
  const threadId = await ask(
    o.ws,
    o,
    versionId,
    productId,
    "Tell us a joke.",
    "interview-open-02",
    null,
  );
  const first = await w.nextLease();
  expect(first.jobType).toBe("generate");
  w.done(first.jobId, { text: "Why did the robot cross the road? <|end|>" });
  await eventually(async () => {
    const rows = await s.ctx.db
      .select()
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId));
    return rows.length === 2;
  });
  await ask(o.ws, o, versionId, productId, "What is <b>your</b> name?", null, threadId);
  const second = await w.nextLease();
  w.done(second.jobId, { text: "I am Spark." });
  await eventually(async () => {
    const rows = await s.ctx.db
      .select()
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId));
    return rows.length === 4;
  });
  return { versionId, productId, threadId, firstJob: first.jobId };
}

describe("interviews", () => {
  it("are held on the server, ahead of ordinary chats, and published as the AI said them", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "generate", "interview"] });
    const o = await setup(server);
    const { productId, threadId, firstJob } = await interview(server, w, o);
    expect((await server.ctx.broker.getJob(firstJob))?.priority).toBe(1);
    const [thread] = await server.ctx.db
      .select()
      .from(schema.chatThreads)
      .where(eq(schema.chatThreads.id, threadId));
    expect(thread).toMatchObject({ kind: "interview", productId });

    const r = await publish(o.ws, o, threadId, [1, 3]);
    expect(r.ok).toBe(true);
    const lease = await w.nextLease();
    expect(lease.jobType).toBe("interview");
    expect((await server.ctx.broker.getJob(lease.jobId))?.priority).toBe(1);
    const spec = InterviewJobSpec.parse(lease.spec);
    expect(spec.answers).toEqual([
      {
        cardId: "interview-open-02",
        question: "Tell us a joke.",
        reply: "Why did the robot cross the road? <|end|>",
      },
      { cardId: null, question: "What is <b>your</b> name?", reply: "I am Spark." },
    ]);
    w.done(lease.jobId, {
      answers: [
        { score: 0.9, checked: false },
        { score: 0.8, checked: false },
      ],
    });
    await eventually(() => companyOf(o).interviews.length === 1);
    expect(companyOf(o).interviews[0]?.answers.map((a) => a.reply)).toEqual([
      "Why did the robot cross the road? <|end|>",
      "I am Spark.",
    ]);
    expect(companyOf(o).interviews[0]?.id).toBe(
      (r as { data: { interviewId: string } }).data.interviewId,
    );

    // The cooldown, from the world's rules.
    expect(await publish(o.ws, o, threadId, [1])).toMatchObject({
      ok: false,
      reason: expect.stringMatching(/^You gave an interview recently/),
    });
  });

  it("refuse a player's own message, another player's thread, and an ordinary chat", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "generate", "interview"] });
    const o = await setup(server);
    const { versionId, threadId } = await interview(server, w, o);
    expect(await publish(o.ws, o, threadId, [0])).toEqual({
      ok: false,
      reason: "Pick answers from this interview.",
    });
    expect(await publish(o.ws, o, threadId, [9])).toMatchObject({ ok: false });

    const ben = await createGuest(server, "Ben");
    await o.actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    const benWs = await connectAs(server, ben.cookie);
    openSockets.push(benWs);
    expect(await publish(benWs, o, threadId, [1])).toEqual({
      ok: false,
      reason: "That interview isn't yours.",
    });

    const cid = randomUUID();
    o.ws.send({
      v: 1,
      type: "chat",
      cid,
      worldId: o.worldId,
      threadId: null,
      versionId,
      text: "hi",
    });
    const plain = (await o.ws.next("ack", (m) => m.cid === cid)).data as { threadId: string };
    expect(await publish(o.ws, o, plain.threadId, [1])).toEqual({
      ok: false,
      reason: "That interview isn't yours.",
    });
  });

  it("a scoring that fails for good changes nothing, and the player is told", async () => {
    server = await start();
    const w = await worker(server, { jobTypes: ["train", "eval", "generate", "interview"] });
    const o = await setup(server);
    const { threadId } = await interview(server, w, o);
    expect((await publish(o.ws, o, threadId, [1])).ok).toBe(true);
    const lease = await w.nextLease();
    w.failed(lease.jobId, "boom", false);
    const told = await o.ws.next("event", (m) => m.kind === "interview");
    expect(told.text).toBe("The interview couldn't be scored. Try publishing again.");
    expect(companyOf(o).interviews).toEqual([]);
    expect(companyOf(o).lastInterviewDay).toBeNull();
  });
});
