import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import { BattleJobSpec } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import { closeSockets, send, setup, start, toDay, trainVersion, worker } from "./flows.ts";
import { createGuest, type TestServer } from "./harness.ts";
import { writeFixturePack } from "./rivalPack.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

const EVAL = {
  report: {
    suite: "bench-v1",
    scores: { knowledge: { score: 50, raw: 0.5, n: 40 }, coding: { score: 30, raw: 0.3, n: 40 } },
    overall: 40,
    missing: [],
    items: [],
    seconds: 1,
  },
  traits: { verbosity: 0.4 },
  personalityWords: [],
};

const ROUND = (k: number) => ({
  prompt: `Q${k}?`,
  a: { reply: `mine ${k} <|end|>`, score: 1 },
  b: { reply: `theirs ${k}`, score: 0 },
});

type Ready = {
  ws: WsClient;
  worldId: string;
  userId: string;
  cookie: string;
  actor: Awaited<ReturnType<typeof setup>>["actor"];
  w: FakeWorker;
  battleId: string;
};

const mine = (o: { actor: Ready["actor"]; userId: string }) => {
  const s = o.actor.state();
  return s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
};

/** A world where the player's AI is on sale, a rival's too, and a knowledge battle has started. */
async function battleStarted(s: TestServer): Promise<Ready> {
  const w = await worker(s, { jobTypes: ["train", "eval", "battle"] });
  const o = await setup(s);
  await eventually(() => o.actor.state().rivals);
  const versionId = await trainVersion(s, w, o);
  const evalLease = await w.nextLease();
  w.done(evalLease.jobId, EVAL);
  await eventually(() => mine(o).versions[versionId]?.scores);
  await send(o.ws, o.worldId, {
    type: "launchProduct",
    versionId,
    name: "Spark",
    plans: ["free", "personal"],
    countdownDays: 3,
  });
  await toDay(s, o.actor, o.actor.state().day + 4);
  const arch = o.actor.state().companies["rival-1"] as CompanyState;
  const target = Object.values(arch.products).find((p) => p.status === "live");
  if (!target) throw new Error("the arch-rival has nothing on sale");
  await send(o.ws, o.worldId, {
    type: "startBattle",
    rivalProductId: target.id,
    skill: "knowledge",
  });
  const battle = await eventually(() => mine(o).battles.find((b) => b.status === "running"));
  return { ...o, w, battleId: battle.id };
}

async function query(ws: WsClient, q: unknown) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q });
  return ws.next("queryResult", (m) => m.id === id);
}

function startWithPack() {
  const pack = writeFixturePack();
  temps.push(pack.dir);
  return start({ rivalsDir: pack.dir });
}

describe("the battle service", () => {
  it("runs a battle as a job, streams both replies live and applies the result", async () => {
    server = await startWithPack();
    const o = await battleStarted(server);
    const lease = await o.w.nextLease();
    expect(lease.jobType).toBe("battle");
    const spec = BattleJobSpec.parse(lease.spec);
    expect(spec).toMatchObject({ battleId: o.battleId, skill: "knowledge", rounds: 3 });
    expect(await (await fetch(spec.a.modelTarUrl)).text()).toBe("fake-checkpoint");
    expect(await (await fetch(spec.b.modelTarUrl)).text()).toBe(`fake-rival-${spec.b.versionId}`);
    expect((await server.ctx.broker.getJob(lease.jobId))?.priority).toBe(1);

    o.ws.send({ v: 1, type: "subscribe", sub: `battle:${o.worldId}:${o.battleId}` });
    expect((await query(o.ws, { kind: "ping" })).ok).toBe(true);
    o.w.progress(lease.jobId, { kind: "battleText", round: 0, side: "a", text: "Par" });
    o.w.progress(lease.jobId, { kind: "battleText", round: 0, side: "b", text: "Lon" });
    expect((await o.ws.next("stream")).data).toEqual({
      kind: "text",
      round: 0,
      side: "mine",
      text: "Par",
    });
    expect((await o.ws.next("stream")).data).toEqual({
      kind: "text",
      round: 0,
      side: "theirs",
      text: "Lon",
    });
    const sofar = await query(o.ws, { kind: "battle", worldId: o.worldId, battleId: o.battleId });
    expect(sofar.data).toEqual({ rounds: [{ mine: "Par", theirs: "Lon" }] });

    o.w.done(lease.jobId, { rounds: [0, 1, 2].map(ROUND) });
    expect((await o.ws.next("stream")).data).toEqual({ kind: "done" });
    const battle = await eventually(() =>
      mine(o).battles.find((b) => b.id === o.battleId && b.status === "done"),
    );
    expect(battle.outcome).toBe("win");
    expect(battle.rounds?.[0]?.mine.reply).toBe("mine 0 <|end|>");
    const companyId = mine(o).id;
    expect(o.actor.state().points.knowledge?.[companyId]).toBe(1016);
  });

  it("calls a battle off and refunds the fee when the job fails", async () => {
    server = await startWithPack();
    const o = await battleStarted(server);
    const cash = mine(o).cash;
    const lease = await o.w.nextLease();
    o.w.failed(lease.jobId, "the model could not load", false);
    await eventually(() => mine(o).battles.find((b) => b.status === "calledOff"));
    expect(mine(o).cash).toBe(cash + 5_000);
    expect(o.actor.state().points).toEqual({});
  });

  it("applies a result once even when the world was put away meanwhile", async () => {
    server = await startWithPack();
    const o = await battleStarted(server);
    const lease = await o.w.nextLease();
    await server.ctx.worlds.evict(o.worldId);
    o.w.done(lease.jobId, { rounds: [0, 1, 2].map(ROUND) });
    const actor = await server.ctx.worlds.get(o.worldId);
    const done = await eventually(() => {
      const s = actor.state();
      const c = s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
      return c.battles.find((b) => b.id === o.battleId && b.status === "done");
    });
    expect(done.outcome).toBe("win");
    await server.ctx.worlds.evict(o.worldId);
    const again = await server.ctx.worlds.get(o.worldId);
    await new Promise((r) => setTimeout(r, 50));
    const battleJobs = await server.ctx.db
      .select()
      .from(schema.jobs)
      .where(eq(schema.jobs.type, "battle"));
    expect(battleJobs).toHaveLength(1);
    expect(again.state().points.knowledge).toBeDefined();
  });

  it("keeps a battle's live replies to its owner", async () => {
    server = await startWithPack();
    const o = await battleStarted(server);
    const stranger = await createGuest(server, "Eve");
    const ws = await connectAs(server, stranger.cookie);
    ws.send({ v: 1, type: "subscribe", sub: `battle:${o.worldId}:${o.battleId}` });
    expect((await ws.next("error")).reason).toBe("You can't follow that.");
    const q = await query(ws, { kind: "battle", worldId: o.worldId, battleId: o.battleId });
    expect(q.ok).toBe(false);
    ws.close();
  });
});
