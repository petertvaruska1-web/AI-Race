import { randomUUID } from "node:crypto";
import { applyPatch, WorldView } from "@airace/protocol";
import { type Input, projectView } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { CLOSE_UNAUTHORIZED } from "../src/realtime/gateway.ts";
import { createGuest, createGuestWorld, startTestServer, type TestServer } from "./harness.ts";
import { connectAs, settle, WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
const clients: WsClient[] = [];
afterEach(async () => {
  for (const c of clients.splice(0)) c.close();
  await server?.close();
  server = null;
});

async function connect(s: TestServer, cookie: string): Promise<WsClient> {
  const c = await connectAs(s, cookie);
  clients.push(c);
  return c;
}

const COMPANY = {
  type: "createCompany",
  name: "Acme",
  logo: { shape: "circle", symbol: "spark", initials: "AC" },
  palette: { primary: "#4f8cff", secondary: "#ffb020" },
};

function command(worldId: string, cmd: unknown, cid: string = randomUUID()) {
  return { v: 1, type: "command", cid, worldId, cmd };
}

async function presenceOffRows(s: TestServer, worldId: string): Promise<number> {
  const rows = await s.ctx.db
    .select({ payload: schema.worldInputs.payload })
    .from(schema.worldInputs)
    .where(eq(schema.worldInputs.worldId, worldId));
  return rows.filter((r) => {
    const input = r.payload as Input;
    return input.kind === "system" && input.event.type === "presence" && !input.event.online;
  }).length;
}

describe("sign-in", () => {
  it("closes a socket whose hello is bad, missing or late", async () => {
    server = await startTestServer();
    const bad = await WsClient.open(`${server.wsUrl}/ws`);
    bad.send({ v: 1, type: "hello", token: "forged" });
    expect((await bad.closed).code).toBe(CLOSE_UNAUTHORIZED);

    const garbage = await WsClient.open(`${server.wsUrl}/ws`);
    garbage.send("not json");
    expect((await garbage.closed).code).toBe(CLOSE_UNAUTHORIZED);

    const silent = await WsClient.open(`${server.wsUrl}/ws`);
    await settle();
    await server.scheduler.advance(5000);
    expect((await silent.closed).code).toBe(CLOSE_UNAUTHORIZED);
  });
});

describe("world views", () => {
  it("sends a snapshot, then patches that rebuild the player's view", async () => {
    server = await startTestServer();
    const { userId, cookie, worldId } = await createGuestWorld(server);
    const ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    const snap = await ws.next("snapshot");
    expect(WorldView.safeParse(snap.view).success).toBe(true);
    expect(snap.view.members[0]?.online).toBe(true);

    const cid = randomUUID();
    ws.send(command(worldId, COMPANY, cid));
    expect(await ws.next("ack", (m) => m.cid === cid)).toMatchObject({ duplicate: false });
    await server.scheduler.advance(250);
    const patch = await ws.next("patch");
    expect(patch.from).toBe(snap.version);
    const actor = await server.ctx.worlds.get(worldId);
    expect(patch.to).toBe(actor.state().seq);
    expect(applyPatch(snap.view, patch.ops)).toEqual(projectView(actor.state(), userId));
  });

  it("acks a resent command id as a duplicate and applies it once", async () => {
    server = await startTestServer();
    const { cookie, worldId } = await createGuestWorld(server);
    const ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    await ws.next("snapshot");
    const cid = randomUUID();
    ws.send(command(worldId, COMPANY, cid));
    ws.send(command(worldId, COMPANY, cid));
    expect(await ws.next("ack", (m) => m.cid === cid)).toMatchObject({ duplicate: false });
    expect(await ws.next("ack", (m) => m.cid === cid)).toMatchObject({ duplicate: true });
    const actor = await server.ctx.worlds.get(worldId);
    expect(Object.keys(actor.state().companies)).toHaveLength(1);
  });

  it("gives new campaigns and data centers an id, so the rules can judge them", async () => {
    server = await startTestServer();
    const { cookie, worldId } = await createGuestWorld(server);
    const ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    await ws.next("snapshot");
    for (const cmd of [COMPANY, { type: "startWorld" }]) {
      const cid = randomUUID();
      ws.send(command(worldId, cmd, cid));
      await ws.next("ack", (m) => m.cid === cid);
    }
    const reasons: string[] = [];
    for (const cmd of [
      { type: "startCampaign", region: "pacific", group: "all", budgetPerDay: 500, days: 7 },
      { type: "buildDataCenter", region: "pacific" },
    ]) {
      const cid = randomUUID();
      ws.send(command(worldId, cmd, cid));
      reasons.push((await ws.next("reject", (m) => m.cid === cid)).reason);
    }
    expect(reasons[0]).toBe("Launch an AI before advertising it.");
    expect(reasons[1]).toMatch(/^A data center costs \$1,500,000\./);
  });

  it("answers resync with a fresh snapshot", async () => {
    server = await startTestServer();
    const { cookie, worldId } = await createGuestWorld(server);
    const ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    await ws.next("snapshot");
    ws.send(command(worldId, COMPANY));
    await ws.next("ack");
    ws.send({ v: 1, type: "resync", worldId });
    const again = await ws.next("snapshot");
    const actor = await server.ctx.worlds.get(worldId);
    expect(again.version).toBe(actor.state().seq);
    expect(again.view.me.company?.cash).toBeGreaterThan(0);
  });

  it("refuses to show a world to someone who is not in it", async () => {
    server = await startTestServer();
    const { worldId } = await createGuestWorld(server);
    const ben = await createGuest(server, "Ben");
    const ws = await connect(server, ben.cookie);
    ws.send({ v: 1, type: "watch", worldId });
    expect((await ws.next("error")).reason).toMatch(/not in it/i);
    const cid = randomUUID();
    ws.send(command(worldId, COMPANY, cid));
    expect(await ws.next("reject", (m) => m.cid === cid)).toBeTruthy();
    ws.send({ v: 1, type: "watch", worldId: "not-a-world" });
    expect((await ws.next("error")).reason).toMatch(/doesn't exist/i);
  });

  it("answers queries", async () => {
    server = await startTestServer();
    const { cookie } = await createGuestWorld(server);
    const ws = await connect(server, cookie);
    ws.send({ v: 1, type: "query", id: "q1", q: { kind: "ping" } });
    expect(await ws.next("queryResult")).toEqual({
      v: 1,
      type: "queryResult",
      id: "q1",
      ok: true,
      data: { pong: true },
    });
  });
});

describe("presence", () => {
  it("keeps a player online across a quick reconnect and freezes the world after they leave", async () => {
    server = await startTestServer();
    const { cookie, worldId } = await createGuestWorld(server);
    let ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    await ws.next("snapshot");
    ws.send(command(worldId, COMPANY));
    await ws.next("ack");
    ws.send(command(worldId, { type: "startWorld" }));
    await ws.next("ack");
    const actor = await server.ctx.worlds.get(worldId);
    await server.scheduler.advance(1000);
    expect(actor.state().day).toBeGreaterThan(0);
    const offBefore = await presenceOffRows(server, worldId);

    ws.close();
    await settle();
    await server.scheduler.advance(4000);
    ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    await ws.next("snapshot");
    await server.scheduler.advance(2000);
    expect(await presenceOffRows(server, worldId)).toBe(offBefore);
    expect(actor.state().members[Object.keys(actor.state().members)[0] ?? ""]?.online).toBe(true);

    ws.close();
    await settle();
    await server.scheduler.advance(6000);
    expect(await presenceOffRows(server, worldId)).toBe(offBefore + 1);
    const frozenDay = actor.state().day;
    await server.scheduler.advance(5000);
    expect(actor.state().day).toBe(frozenDay);
  });
});

describe("limits", () => {
  it("slows down a flood of commands", async () => {
    server = await startTestServer();
    const { cookie, worldId } = await createGuestWorld(server);
    const ws = await connect(server, cookie);
    ws.send({ v: 1, type: "watch", worldId });
    await ws.next("snapshot");
    const cids = Array.from({ length: 25 }, () => randomUUID());
    for (const cid of cids) ws.send(command(worldId, { type: "setSpeed", speed: 2 }, cid));
    const outcomes: string[] = [];
    for (const cid of cids) {
      const ack = ws.next("ack", (m) => m.cid === cid, 3000).then(() => "ack");
      const reject = ws.next("reject", (m) => m.cid === cid, 3000).then((m) => m.reason);
      outcomes.push(await Promise.any([ack, reject]));
    }
    expect(outcomes.slice(0, 20).every((o) => o === "ack")).toBe(true);
    for (const o of outcomes.slice(20)) expect(o).toMatch(/slow down/i);
  });
});
