import { randomUUID } from "node:crypto";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { MAKER_OFFLINE, NOT_SHARED, TRIED_A_LOT } from "../src/chat/service.ts";
import { schema } from "../src/db/index.ts";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import { closeSockets, owner, send, setup, start, trainVersion, worker } from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
import type { WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

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

/** Ana (host, house card) shares her trained v1; Ben joins and founds a company. */
async function sharedWorld(s: TestServer) {
  const w: FakeWorker = await worker(s, {
    jobTypes: ["train", "generate"],
    slots: { train: 1, eval: 0, generate: 1 },
  });
  const ana = await setup(s);
  const versionId = await trainVersion(s, w, ana);
  await send(ana.ws, ana.worldId, { type: "publishVersion", versionId, published: true });
  const ben = await createGuest(s, "Ben");
  const res = await api(s, "/api/worlds/join", {
    body: { code: ana.actor.state().inviteCode },
    cookie: ben.cookie,
  });
  expect(res.status).toBe(200);
  const benWs = await owner(s, ben.cookie, ana.worldId);
  await send(benWs, ana.worldId, {
    type: "createCompany",
    name: "Bolt",
    logo: { shape: "circle", symbol: "spark", initials: "BO" },
    palette: { primary: "#4f8cff", secondary: "#ffb020" },
  });
  return { w, ana, ben: { ...ben, ws: benWs }, versionId };
}

describe("trying a friend's shared AI", () => {
  it("is answered by its maker's card, privately to the asker; the maker sees only a count", async () => {
    server = await start();
    const { w, ana, ben, versionId } = await sharedWorld(server);

    const sent = await chat(ben.ws, ana.worldId, versionId, "Hi there");
    expect(sent.threadId).toBeTruthy();
    const lease = await w.nextLease();
    expect(lease.jobType).toBe("generate");
    const [job] = await server.ctx.db
      .select()
      .from(schema.jobs)
      .where(eq(schema.jobs.id, lease.jobId));
    expect(job?.ownerUserId).toBe(ana.userId);

    ben.ws.send({ v: 1, type: "subscribe", sub: `chat:${ana.worldId}:${sent.threadId}` });
    expect((await query(ben.ws, { kind: "ping" })).ok).toBe(true);
    ana.ws.send({ v: 1, type: "subscribe", sub: `chat:${ana.worldId}:${sent.threadId}` });
    expect((await ana.ws.next("error")).reason).toMatch(/can't follow/i);
    w.done(lease.jobId, { text: "Hello Ben" });
    expect((await ben.ws.next("stream")).data).toEqual({ kind: "done", text: "Hello Ben" });

    const anaThreads = await query(ana.ws, { kind: "chatThreads", worldId: ana.worldId });
    expect((anaThreads.data as { threads: unknown[] }).threads).toEqual([]);
    const tries = await query(ana.ws, { kind: "versionTries", worldId: ana.worldId });
    expect(tries.data).toEqual({ versions: [{ versionId, tries7d: 1 }] });
    expect(JSON.stringify(tries.data)).not.toContain("Hi there");
  });

  it("refuses once the maker stops sharing it", async () => {
    server = await start();
    const { w, ana, ben, versionId } = await sharedWorld(server);
    const sent = await chat(ben.ws, ana.worldId, versionId, "Hi");
    const lease = await w.nextLease();
    w.done(lease.jobId, { text: "Hello" });
    await eventually(async () => {
      const rows = await server?.ctx.db.select().from(schema.chatMessages);
      return rows?.some((r) => r.role === "ai") ?? false;
    });
    await send(ana.ws, ana.worldId, { type: "publishVersion", versionId, published: false });
    expect(
      (await chat(ben.ws, ana.worldId, versionId, "Again?", sent.threadId ?? null)).reason,
    ).toBe(NOT_SHARED);
    expect((await chat(ben.ws, ana.worldId, versionId, "New?")).reason).toBe(NOT_SHARED);
  });

  it("refuses while the maker's computer is off (Fair mode off)", async () => {
    server = await start();
    const { w, ana, ben, versionId } = await sharedWorld(server);
    w.close();
    await eventually(() => server?.ctx.broker.connections().length === 0);
    expect((await chat(ben.ws, ana.worldId, versionId, "Anyone?")).reason).toBe(MAKER_OFFLINE);
  });

  it("limits tries to 10 messages in 10 minutes per AI", async () => {
    server = await start();
    const { w, ana, ben, versionId } = await sharedWorld(server);
    let threadId: string | null = null;
    for (let i = 0; i < 10; i++) {
      const sent = await chat(ben.ws, ana.worldId, versionId, `Try ${i}`, threadId);
      expect(sent.reason).toBeUndefined();
      threadId = sent.threadId ?? null;
      const lease = await w.nextLease();
      w.done(lease.jobId, { text: "ok" });
      await eventually(async () => {
        const rows = await server?.ctx.db
          .select()
          .from(schema.chatMessages)
          .where(eq(schema.chatMessages.threadId, threadId ?? ""));
        return rows?.length === 2 * (i + 1);
      });
    }
    expect((await chat(ben.ws, ana.worldId, versionId, "One more", threadId)).reason).toBe(
      TRIED_A_LOT,
    );
  });
});
