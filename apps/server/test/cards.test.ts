import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { FakeWorker } from "./fakeWorker.ts";
import { closeSockets, openSockets } from "./flows.ts";
import { api, createGuest, createGuestWorld, startTestServer, type TestServer } from "./harness.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

type Card = {
  userId: string;
  playerName: string;
  gpuName: string | null;
  vramMb: number;
  speed: number;
  online: boolean;
  house: boolean;
};

async function cards(ws: WsClient, worldId: string) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q: { kind: "cards", worldId } });
  return ws.next("queryResult", (m) => m.id === id);
}

async function keyOf(s: TestServer, cookie: string): Promise<string> {
  const res = await api(s, "/api/helper/key", { cookie });
  return ((await res.json()) as { key: string }).key;
}

describe("cards in a world", () => {
  it("every player sees whose graphics card is online, and never a key", async () => {
    server = await startTestServer();
    const ana = await createGuestWorld(server);
    const ben = await createGuest(server, "Ben");
    const actor = await server.ctx.worlds.get(ana.worldId);
    await actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    const benWs = await connectAs(server, ben.cookie);
    openSockets.push(benWs);

    const before = (await cards(benWs, ana.worldId)).data as { cards: Card[] };
    expect(before.cards.map((c) => [c.playerName, c.online])).toEqual([
      ["Ana", false],
      ["Ben", false],
    ]);

    benWs.send({ v: 1, type: "subscribe", sub: `cards:${ana.worldId}` });
    const key = await keyOf(server, ana.cookie);
    const w = await FakeWorker.connect(server, {
      token: key,
      workerId: "ana-pc",
      gpuName: "NVIDIA GeForce RTX 3050",
      speed: 450_000,
      vramMb: 8192,
    });
    const pushed = await benWs.next("stream", (m) => m.sub === `cards:${ana.worldId}`);
    const list = (pushed.data as { cards: Card[] }).cards;
    expect(list.find((c) => c.playerName === "Ana")).toMatchObject({
      gpuName: "NVIDIA GeForce RTX 3050",
      vramMb: 8192,
      speed: 1,
      online: true,
      house: false,
    });
    expect(JSON.stringify(list)).not.toContain(key.split(".")[1]);
    expect(JSON.stringify(list)).not.toContain("ana-pc");

    w.close();
    const gone = await benWs.next(
      "stream",
      (m) =>
        m.sub === `cards:${ana.worldId}` &&
        (m.data as { cards: Card[] }).cards.every((c) => !c.online),
    );
    expect(gone).toBeTruthy();
  });

  it("marks a card whose helper is from an older version of the game", async () => {
    server = await startTestServer();
    const ana = await createGuestWorld(server);
    await FakeWorker.connect(server, { engine: 1 });
    const anaWs = await connectAs(server, ana.cookie);
    openSockets.push(anaWs);
    const mine = (await cards(anaWs, ana.worldId)).data as { cards: Card[] };
    expect(mine.cards).toEqual([expect.objectContaining({ online: true, outdated: true })]);
  });

  it("the house card counts as the world host's, and outsiders can't look", async () => {
    server = await startTestServer();
    const ana = await createGuestWorld(server);
    await FakeWorker.connect(server, { gpuName: "NVIDIA GeForce RTX 3050", speed: 225_000 });
    const anaWs = await connectAs(server, ana.cookie);
    openSockets.push(anaWs);
    const mine = (await cards(anaWs, ana.worldId)).data as { cards: Card[] };
    expect(mine.cards).toEqual([
      expect.objectContaining({ playerName: "Ana", online: true, house: true, speed: 0.5 }),
    ]);
    const eve = await createGuest(server, "Eve");
    const eveWs = await connectAs(server, eve.cookie);
    openSockets.push(eveWs);
    expect((await cards(eveWs, ana.worldId)).ok).toBe(false);
  });
});
