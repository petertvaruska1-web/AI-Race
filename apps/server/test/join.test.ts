import { afterEach, describe, expect, it } from "vitest";
import { eventually } from "./fakeWorker.ts";
import { closeSockets, DAY_MS, openSockets, owner, send } from "./flows.ts";
import { api, createGuest, startTestServer, type TestServer } from "./harness.ts";
import { connectAs, settle } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

async function newWorld(s: TestServer, cookie: string, body: Record<string, unknown>) {
  const res = await api(s, "/api/worlds", { body, cookie });
  expect(res.status).toBe(200);
  const { worldId } = (await res.json()) as { worldId: string };
  const actor = await s.ctx.worlds.get(worldId);
  return { worldId, actor, code: actor.state().inviteCode };
}

async function join(s: TestServer, cookie: string, code: string) {
  return api(s, "/api/worlds/join", { body: { code }, cookie });
}

const company = (name: string) => ({
  type: "createCompany",
  name,
  logo: { shape: "circle", symbol: "spark", initials: name.slice(0, 2).toUpperCase() },
  palette: { primary: "#4f8cff", secondary: "#ffb020" },
});

describe("joining a friend's world", () => {
  it("joins by invite code and sees the host's company without its private parts", async () => {
    server = await startTestServer();
    const ana = await createGuest(server, "Ana");
    const w = await newWorld(server, ana.cookie, { name: "Friends" });
    const anaWs = await owner(server, ana.cookie, w.worldId);
    await send(anaWs, w.worldId, company("Acme"));

    const ben = await createGuest(server, "Ben");
    const res = await join(server, ben.cookie, w.code.toLowerCase());
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ worldId: w.worldId });
    // Joining again is fine and changes nothing.
    expect((await join(server, ben.cookie, w.code)).status).toBe(200);
    expect(Object.keys(w.actor.state().members)).toHaveLength(2);

    const listed = await api(server, "/api/worlds", { cookie: ben.cookie });
    expect(((await listed.json()) as { worlds: { id: string }[] }).worlds.map((x) => x.id)).toEqual(
      [w.worldId],
    );
    const benWs = await owner(server, ben.cookie, w.worldId);
    benWs.send({ v: 1, type: "resync", worldId: w.worldId });
    const view = (await benWs.next("snapshot")).view;
    const acme = view.companies.find((c) => c.name === "Acme");
    expect(acme).toBeTruthy();
    expect(JSON.stringify(acme)).not.toMatch(/"cash"|"runs"/);
    expect(view.me.company).toBeNull();
    expect(view.world.fairMode).toBe(false);
  });

  it("refuses a code that matches no world", async () => {
    server = await startTestServer();
    const ben = await createGuest(server, "Ben");
    const res = await join(server, ben.cookie, "ZZZZZZ");
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ error: "That invite code doesn't match a world." });
  });

  it("refuses the 9th player", async () => {
    server = await startTestServer();
    const ana = await createGuest(server, "Ana");
    const w = await newWorld(server, ana.cookie, { name: "Crowd" });
    for (let i = 2; i <= 8; i++) {
      const g = await createGuest(server, `P${i}`);
      expect((await join(server, g.cookie, w.code)).status).toBe(200);
    }
    const ninth = await createGuest(server, "Nine");
    const res = await join(server, ninth.cookie, w.code);
    expect(res.status).toBe(409);
    expect(await res.json()).toEqual({ error: "This world is full (8 players)." });
    const rows = await api(server, "/api/worlds", { cookie: ninth.cookie });
    expect(((await rows.json()) as { worlds: unknown[] }).worlds).toEqual([]);
  });

  it("a world made with Fair mode on says so", async () => {
    server = await startTestServer();
    const ana = await createGuest(server, "Ana");
    const w = await newWorld(server, ana.cookie, { name: "Fair", fairMode: true });
    expect(w.actor.state().fairMode).toBe(true);
    const ben = await createGuest(server, "Ben");
    await join(server, ben.cookie, w.code);
    const benWs = await owner(server, ben.cookie, w.worldId);
    benWs.send({ v: 1, type: "resync", worldId: w.worldId });
    expect((await benWs.next("snapshot")).view.world.fairMode).toBe(true);
  });

  it("the host away: a friend runs the clock; everyone away: time stops until someone is back", async () => {
    server = await startTestServer();
    const ana = await createGuest(server, "Ana");
    const w = await newWorld(server, ana.cookie, { name: "Clock" });
    const anaWs = await owner(server, ana.cookie, w.worldId);
    await send(anaWs, w.worldId, company("Acme"));
    const ben = await createGuest(server, "Ben");
    await join(server, ben.cookie, w.code);
    const benWs = await owner(server, ben.cookie, w.worldId);
    await send(benWs, w.worldId, company("Bolt"));
    await send(anaWs, w.worldId, { type: "startWorld" });

    anaWs.close();
    // Past the presence grace, once the server has seen the socket close.
    await eventually(async () => {
      await server?.scheduler.advance(1_000);
      return w.actor.state().members[ana.userId]?.online === false;
    });
    await send(benWs, w.worldId, { type: "setPaused", paused: true });
    expect(w.actor.state().paused).toBe(true);
    await send(benWs, w.worldId, { type: "setPaused", paused: false });

    benWs.close();
    // Past the presence grace, once the server has seen the socket close.
    await eventually(async () => {
      await server?.scheduler.advance(1_000);
      return w.actor.state().members[ben.userId]?.online === false;
    });
    const day = w.actor.state().day;
    await server.scheduler.advance(5 * DAY_MS);
    expect(w.actor.state().day).toBe(day);

    const back = await connectAs(server, ben.cookie);
    openSockets.push(back);
    back.send({ v: 1, type: "watch", worldId: w.worldId });
    await back.next("snapshot");
    await server.scheduler.advance(3 * DAY_MS);
    await settle(50);
    expect(w.actor.state().day).toBeGreaterThan(day);
  });
});
