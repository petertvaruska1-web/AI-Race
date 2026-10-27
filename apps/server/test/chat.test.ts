import { randomUUID } from "node:crypto";
import { GenerateJobSpec } from "@airace/protocol";
import { asc, eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { REPLY_FAILED } from "../src/chat/service.ts";
import { schema } from "../src/db/index.ts";
import type { FakeWorker } from "./fakeWorker.ts";
import {
  closeSockets,
  openSockets,
  owner,
  send,
  setup,
  start,
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

/** Sends a chat message; resolves to the ack's threadId or the reject reason. */
async function chat(
  ws: WsClient,
  worldId: string,
  versionId: string,
  text: string,
  threadId: string | null = null,
): Promise<{ threadId?: string; reason?: string }> {
  const cid = randomUUID();
  ws.send({ v: 1, type: "chat", cid, worldId, threadId, versionId, text });
  return Promise.race([
    ws.next("ack", (m) => m.cid === cid).then((m) => m.data as { threadId: string }),
    ws.next("reject", (m) => m.cid === cid).then((m) => ({ reason: m.reason })),
  ]);
}

async function query(ws: WsClient, q: unknown) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q });
  return ws.next("queryResult", (m) => m.id === id);
}

/** A world whose owner has one trained version and a worker for chat. */
async function withVersion(s: TestServer) {
  // No eval slot, so the version's evaluation never takes the worker's attention here.
  const w: FakeWorker = await worker(s, {
    jobTypes: ["train", "generate"],
    slots: { train: 1, eval: 0, generate: 1 },
  });
  const o = await setup(s);
  const versionId = await trainVersion(s, w, o);
  return { ...o, w, versionId };
}

describe("playground chat", () => {
  it("streams a reply to the asker and keeps the conversation", async () => {
    server = await start();
    const o = await withVersion(server);
    const sent = await chat(o.ws, o.worldId, o.versionId, "Tell me a story <|end|>");
    expect(sent.threadId).toBeTruthy();
    const threadId = sent.threadId ?? "";

    const lease = await o.w.nextLease();
    expect(lease.jobType).toBe("generate");
    const spec = GenerateJobSpec.parse(lease.spec);
    expect(spec).toMatchObject({
      versionId: o.versionId,
      maxNewTokens: 96,
      temperature: 0.8,
      topP: 0.95,
    });
    expect(spec.history).toEqual([["user", "Tell me a story <|end|>"]]);
    expect(await (await fetch(spec.modelTarUrl)).text()).toBe("fake-checkpoint");

    o.ws.send({ v: 1, type: "subscribe", sub: `chat:${o.worldId}:${threadId}` });
    expect((await query(o.ws, { kind: "ping" })).ok).toBe(true); // the subscribe is in place
    o.w.chunk(lease.jobId, "Hel");
    o.w.chunk(lease.jobId, "lo");
    o.w.done(lease.jobId, { text: "Hello" });
    const first = await o.ws.next("stream");
    const second = await o.ws.next("stream");
    const last = await o.ws.next("stream");
    expect([first.data, second.data, last.data]).toEqual([
      { kind: "chunk", text: "Hel" },
      { kind: "chunk", text: "lo" },
      { kind: "done", text: "Hello" },
    ]);

    o.ws.close();
    const again = await owner(server, o.cookie, o.worldId);
    const thread = await query(again, { kind: "chatThread", worldId: o.worldId, threadId });
    expect(thread.ok).toBe(true);
    expect((thread.data as { messages: { role: string; text: string }[] }).messages).toMatchObject([
      { role: "user", text: "Tell me a story <|end|>" },
      { role: "ai", text: "Hello" },
    ]);
    const list = await query(again, { kind: "chatThreads", worldId: o.worldId });
    expect((list.data as { threads: { id: string }[] }).threads.map((t) => t.id)).toEqual([
      threadId,
    ]);

    // The next message in the thread carries the conversation so far.
    expect((await chat(again, o.worldId, o.versionId, "More!", threadId)).threadId).toBe(threadId);
    const next = GenerateJobSpec.parse((await o.w.nextLease()).spec);
    expect(next.history).toEqual([
      ["user", "Tell me a story <|end|>"],
      ["ai", "Hello"],
      ["user", "More!"],
    ]);
  });

  it("keeps conversations private and AIs to their owners", async () => {
    server = await start();
    const o = await withVersion(server);
    const { threadId } = await chat(o.ws, o.worldId, o.versionId, "Hi");

    const ben = await createGuest(server, "Ben");
    await o.actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    const benWs = await connectAs(server, ben.cookie);
    openSockets.push(benWs);
    benWs.send({ v: 1, type: "watch", worldId: o.worldId });
    await benWs.next("snapshot");
    await send(benWs, o.worldId, {
      type: "createCompany",
      name: "Beta",
      logo: { shape: "shield", symbol: null, initials: "B" },
      palette: { primary: "#ff7a45", secondary: "#ffd166" },
    });

    expect((await query(benWs, { kind: "chatThread", worldId: o.worldId, threadId })).ok).toBe(
      false,
    );
    benWs.send({ v: 1, type: "subscribe", sub: `chat:${o.worldId}:${threadId}` });
    expect((await benWs.next("error")).reason).toMatch(/can't follow/i);
    expect((await chat(benWs, o.worldId, o.versionId, "Hi")).reason).toMatch(/your own AIs/i);
    expect((await chat(benWs, o.worldId, o.versionId, "Hi", threadId ?? null)).reason).toBeTruthy();
  });

  it("refuses an over-long message and a second one while a reply is coming", async () => {
    server = await start();
    const o = await withVersion(server);
    expect((await chat(o.ws, o.worldId, o.versionId, "x".repeat(501))).reason).toMatch(/1–500/);
    const { threadId } = await chat(o.ws, o.worldId, o.versionId, "First");
    expect((await chat(o.ws, o.worldId, o.versionId, "Second", threadId ?? null)).reason).toMatch(
      /wait for the reply/i,
    );
    const rows = await server.ctx.db
      .select()
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId ?? ""))
      .orderBy(asc(schema.chatMessages.idx));
    expect(rows.map((r) => r.text)).toEqual(["First"]);
  });

  it("tells the asker when a reply fails, and lets them try again", async () => {
    server = await start();
    const o = await withVersion(server);
    const { threadId } = await chat(o.ws, o.worldId, o.versionId, "Hi");
    o.ws.send({ v: 1, type: "subscribe", sub: `chat:${o.worldId}:${threadId}` });
    expect((await query(o.ws, { kind: "ping" })).ok).toBe(true);
    const lease = await o.w.nextLease();
    o.w.failed(lease.jobId, "out of memory", false);
    expect((await o.ws.next("stream")).data).toEqual({ kind: "failed", reason: REPLY_FAILED });
    expect((await chat(o.ws, o.worldId, o.versionId, "Again?", threadId ?? null)).threadId).toBe(
      threadId,
    );
  });
});
