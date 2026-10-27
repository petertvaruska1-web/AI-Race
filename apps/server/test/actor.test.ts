import { randomUUID } from "node:crypto";
import type { Command } from "@airace/protocol";
import { type Input, stateHash } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { type SubmitResult, WorldActor } from "../src/worlds/actor.ts";
import { createGuestWorld, startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

const command = (userId: string, cmd: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: cmd as Command,
  ...(newId ? { newId } : {}),
});
const presence = (userId: string, online: boolean): Input => ({
  kind: "system",
  event: { type: "presence", userId, online },
});

function accepted(r: SubmitResult): number {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.seq;
}

/** A world whose host is online, has a company and has started the clock. */
async function runningWorld(s: TestServer) {
  const { userId, worldId } = await createGuestWorld(s);
  const actor = await s.ctx.worlds.get(worldId);
  accepted(await actor.submit(presence(userId, true)));
  const company = {
    type: "createCompany",
    name: "Acme",
    logo: { shape: "circle", symbol: "spark", initials: "AC" },
    palette: { primary: "#4f8cff", secondary: "#ffb020" },
  };
  accepted(await actor.submit(command(userId, company, randomUUID())));
  accepted(await actor.submit(command(userId, { type: "startWorld" })));
  return { userId, worldId, actor };
}

async function inputRows(s: TestServer, worldId: string) {
  return s.ctx.db.select().from(schema.worldInputs).where(eq(schema.worldInputs.worldId, worldId));
}

async function snapshotSeqs(s: TestServer, worldId: string): Promise<number[]> {
  const rows = await s.ctx.db
    .select({ seq: schema.worldSnapshots.seq })
    .from(schema.worldSnapshots)
    .where(eq(schema.worldSnapshots.worldId, worldId));
  return rows.map((r) => r.seq).sort((a, b) => a - b);
}

describe("world clock", () => {
  it("ticks a day every 8 s at 1x and every 4 s at 2x, and snapshots while running", async () => {
    server = await startTestServer({ balance: "balance-v2" });
    const { userId, worldId, actor } = await runningWorld(server);
    await server.scheduler.advance(3 * 8000);
    expect(actor.state().day).toBe(3);
    expect((await snapshotSeqs(server, worldId)).length).toBeGreaterThan(1);

    accepted(await actor.submit(command(userId, { type: "setSpeed", speed: 2 })));
    await server.scheduler.advance(3 * 8000);
    expect(actor.state().day).toBe(9);
  });

  it("stands still while paused or when everyone is away, then unloads", async () => {
    server = await startTestServer({ balance: "balance-v2" });
    const { userId, worldId, actor } = await runningWorld(server);
    accepted(await actor.submit(command(userId, { type: "setPaused", paused: true })));
    const pausedSeq = actor.state().seq;
    expect(await snapshotSeqs(server, worldId)).toContain(pausedSeq);
    await server.scheduler.advance(5 * 8000);
    expect(actor.state().day).toBe(0);

    accepted(await actor.submit(command(userId, { type: "setPaused", paused: false })));
    await server.scheduler.advance(8000);
    expect(actor.state().day).toBe(1);

    accepted(await actor.submit(presence(userId, false)));
    expect(await snapshotSeqs(server, worldId)).toContain(actor.state().seq);
    await server.scheduler.advance(5 * 8000);
    expect(actor.state().day).toBe(1);
    expect(server.ctx.worlds.isLoaded(worldId)).toBe(true);
    await server.scheduler.advance(60_000);
    expect(server.ctx.worlds.isLoaded(worldId)).toBe(false);
    expect(actor.isClosed()).toBe(true);
  });
});

describe("inputs", () => {
  it("applies a command id once", async () => {
    server = await startTestServer();
    const { userId, worldId, actor } = await runningWorld(server);
    const cid = randomUUID();
    const first = await actor.submit(command(userId, { type: "setSpeed", speed: 4 }), {
      cid,
      userId,
    });
    const second = await actor.submit(command(userId, { type: "setSpeed", speed: 4 }), {
      cid,
      userId,
    });
    expect(first).toMatchObject({ ok: true, duplicate: false });
    expect(second).toEqual({ ok: true, seq: accepted(first), duplicate: true });
    expect((await inputRows(server, worldId)).filter((r) => r.cid === cid)).toHaveLength(1);
  });

  it("logs nothing for a rejected command", async () => {
    server = await startTestServer();
    const { userId, worldId, actor } = await runningWorld(server);
    const before = (await inputRows(server, worldId)).length;
    const r = await actor.submit(command(userId, { type: "startWorld" }), {
      cid: randomUUID(),
      userId,
    });
    expect(r).toMatchObject({ ok: false });
    expect(await inputRows(server, worldId)).toHaveLength(before);
  });
});

describe("recovery", () => {
  it("rebuilds the live state from a snapshot plus the later inputs", async () => {
    server = await startTestServer({ balance: "balance-v2" });
    const { userId, worldId, actor } = await runningWorld(server);
    await actor.snapshot("test");
    const snapSeq = actor.state().seq;
    await server.scheduler.advance(2 * 8000);
    accepted(await actor.submit(command(userId, { type: "setSpeed", speed: 4 })));
    expect(actor.state().seq).toBeGreaterThan(snapSeq);
    const live = actor.state();

    await server.ctx.worlds.evict(worldId);
    const reloaded = await WorldActor.load(server.ctx, worldId);
    expect(stateHash(reloaded.state())).toBe(stateHash(live));
    await reloaded.unload();

    const again = await server.ctx.worlds.get(worldId);
    expect(again.state().members[userId]?.online).toBe(false);
    accepted(await again.submit(presence(userId, true)));
    await server.scheduler.advance(2000);
    expect(again.state().day).toBe(live.day + 1);
  });

  it("refuses a snapshot from another simulation version", async () => {
    server = await startTestServer();
    const { worldId } = await createGuestWorld(server);
    await server.ctx.db
      .update(schema.worldSnapshots)
      .set({ simVersion: "sim-0" })
      .where(eq(schema.worldSnapshots.worldId, worldId));
    await expect(server.ctx.worlds.get(worldId)).rejects.toThrow(/sim-0/);
  });

  it("snapshots every world on shutdown", async () => {
    server = await startTestServer();
    const { userId, worldId, actor } = await runningWorld(server);
    accepted(await actor.submit(command(userId, { type: "setSpeed", speed: 2 })));
    const seq = actor.state().seq;
    await server.ctx.worlds.shutdown();
    expect(await snapshotSeqs(server, worldId)).toContain(seq);
  });
});
