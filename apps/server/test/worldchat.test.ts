import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { KEPT_MESSAGES, SLOW_DOWN } from "../src/worldchat/service.ts";
import { closeSockets, openSockets, owner } from "./flows.ts";
import { api, createGuest, createGuestWorld, startTestServer, type TestServer } from "./harness.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

async function post(ws: WsClient, worldId: string, text: string): Promise<string | null> {
  const cid = randomUUID();
  ws.send({ v: 1, type: "worldChat", cid, worldId, text });
  return Promise.race([
    ws.next("ack", (m) => m.cid === cid).then(() => null),
    ws.next("reject", (m) => m.cid === cid).then((m) => m.reason),
  ]);
}

async function history(ws: WsClient, worldId: string) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q: { kind: "worldChat", worldId } });
  return ws.next("queryResult", (m) => m.id === id);
}

/** Ana's world with Ben joined; both watching it. */
async function two(s: TestServer) {
  const ana = await createGuestWorld(s);
  const anaWs = await owner(s, ana.cookie, ana.worldId);
  const ben = await createGuest(s, "Ben");
  const actor = await s.ctx.worlds.get(ana.worldId);
  await api(s, "/api/worlds/join", {
    body: { code: actor.state().inviteCode },
    cookie: ben.cookie,
  });
  const benWs = await owner(s, ben.cookie, ana.worldId);
  return { ana: { ...ana, ws: anaWs }, ben: { ...ben, ws: benWs } };
}

describe("the world chat", () => {
  it("delivers a message to everyone in the world, and only to them", async () => {
    server = await startTestServer();
    const { ana, ben } = await two(server);
    ben.ws.send({ v: 1, type: "subscribe", sub: `worldchat:${ana.worldId}` });
    expect((await history(ben.ws, ana.worldId)).data).toEqual({ messages: [] });
    expect(await post(ana.ws, ana.worldId, "  Good luck, Ben!  ")).toBeNull();
    const pushed = await ben.ws.next("stream", (m) => m.sub === `worldchat:${ana.worldId}`);
    expect(pushed.data).toMatchObject({ name: "Ana", companyId: null, text: "Good luck, Ben!" });
    expect(
      ((await history(ben.ws, ana.worldId)).data as { messages: unknown[] }).messages,
    ).toHaveLength(1);

    const eve = await createGuest(server, "Eve");
    const eveWs = await connectAs(server, eve.cookie);
    openSockets.push(eveWs);
    eveWs.send({ v: 1, type: "subscribe", sub: `worldchat:${ana.worldId}` });
    expect((await eveWs.next("error")).reason).toMatch(/can't follow/i);
    expect((await history(eveWs, ana.worldId)).ok).toBe(false);
    expect(await post(eveWs, ana.worldId, "Hi")).toBe(
      "That world doesn't exist or you're not in it.",
    );
  });

  it("refuses links, empty or long messages, and the AI's own characters", async () => {
    server = await startTestServer();
    const { ana } = await two(server);
    expect(await post(ana.ws, ana.worldId, "see https://example.com")).toBe(
      "Messages can't hold links, email addresses or phone numbers.",
    );
    expect(await post(ana.ws, ana.worldId, "   ")).toBe("Messages are 1–300 characters.");
    expect(await post(ana.ws, ana.worldId, "x".repeat(301))).toBe("Messages are 1–300 characters.");
    expect(await post(ana.ws, ana.worldId, "hi <|end|>")).toBe(
      "That text has characters the AI uses for itself.",
    );
    expect(await server.ctx.db.select().from(schema.worldChat)).toEqual([]);
  });

  it("asks a player to slow down after five messages in ten seconds", async () => {
    server = await startTestServer();
    const { ana } = await two(server);
    for (let i = 0; i < 5; i++) expect(await post(ana.ws, ana.worldId, `m${i}`)).toBeNull();
    expect(await post(ana.ws, ana.worldId, "one more")).toBe(SLOW_DOWN);
    await server.scheduler.advance(10_001);
    expect(await post(ana.ws, ana.worldId, "later")).toBeNull();
  });

  it("keeps the last 200 messages", async () => {
    server = await startTestServer();
    const { ana } = await two(server);
    for (let i = 0; i < KEPT_MESSAGES + 5; i++) {
      await server.ctx.worldChat.add(ana.worldId, ana.userId, `m${i}`);
    }
    const rows = await server.ctx.db.select().from(schema.worldChat);
    expect(rows).toHaveLength(KEPT_MESSAGES);
    const shown = (await history(ana.ws, ana.worldId)).data as { messages: { text: string }[] };
    expect(shown.messages[0]?.text).toBe("m5");
    expect(shown.messages.at(-1)?.text).toBe(`m${KEPT_MESSAGES + 4}`);
  });
});
