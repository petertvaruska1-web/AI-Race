import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { INVITE_ALPHABET } from "../src/worlds/inviteCode.ts";
import { api, createGuest, createGuestWorld, startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

describe("world routes", () => {
  it("creates a world with its creator seated and lists it", async () => {
    server = await startTestServer();
    const { userId, cookie, worldId } = await createGuestWorld(server, "Ana", "  Friday night ");
    const list = await api(server, "/api/worlds", { cookie });
    expect(await list.json()).toEqual({
      worlds: [{ id: worldId, name: "Friday night", phase: "setup", day: 0, memberCount: 1 }],
    });

    const [world] = await server.ctx.db
      .select()
      .from(schema.worlds)
      .where(eq(schema.worlds.id, worldId));
    expect(world?.hostUserId).toBe(userId);
    expect(world?.inviteCode).toMatch(new RegExp(`^[${INVITE_ALPHABET}]{6}$`));
    const members = await server.ctx.db.select().from(schema.memberships);
    expect(members.map((m) => [m.userId, m.displayName])).toEqual([[userId, "Ana"]]);
    const snaps = await server.ctx.db.select().from(schema.worldSnapshots);
    // Snapshot 0 is the starting state; listing loaded the world, which froze it
    // (nobody connected), so a later snapshot follows.
    expect(snaps.map((s) => s.seq).sort()[0]).toBe(0);
    const state = (await server.ctx.worlds.get(worldId)).state();
    expect(state.members[userId]).toMatchObject({ displayName: "Ana", companyId: null });
    expect(state.inviteCode).toBe(world?.inviteCode);
  });

  it("lists a world saved by an older version as outdated, without breaking the list", async () => {
    server = await startTestServer();
    const { cookie, worldId: oldId } = await createGuestWorld(server, "Ana", "Old world");
    const created = await api(server, "/api/worlds", {
      method: "POST",
      cookie,
      body: { name: "New world" },
    });
    const { worldId: newId } = (await created.json()) as { worldId: string };
    await server.ctx.worlds.evict(oldId);
    await server.ctx.db
      .update(schema.worldSnapshots)
      .set({ simVersion: "sim-1" })
      .where(eq(schema.worldSnapshots.worldId, oldId));
    const list = await api(server, "/api/worlds", { cookie });
    expect(list.status).toBe(200);
    expect(await list.json()).toEqual({
      worlds: [
        { id: oldId, name: "Old world", outdated: true },
        { id: newId, name: "New world", phase: "setup", day: 0, memberCount: 1 },
      ],
    });
  });

  it("lists only the caller's worlds", async () => {
    server = await startTestServer();
    await createGuestWorld(server, "Ana");
    const ben = await createGuest(server, "Ben");
    expect(await (await api(server, "/api/worlds", { cookie: ben.cookie })).json()).toEqual({
      worlds: [],
    });
  });

  it("needs a session and a 2–40 character name", async () => {
    server = await startTestServer();
    expect((await api(server, "/api/worlds", { body: { name: "Lab" } })).status).toBe(401);
    const { cookie } = await createGuest(server);
    expect((await api(server, "/api/worlds", { body: { name: "x" }, cookie })).status).toBe(400);
    expect(
      (await api(server, "/api/worlds", { body: { name: "x".repeat(41) }, cookie })).status,
    ).toBe(400);
  });
});
