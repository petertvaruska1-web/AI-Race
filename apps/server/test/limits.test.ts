import { randomUUID } from "node:crypto";
import { TrainJobSpec } from "@airace/protocol";
import { BALANCE_V2, paramCount } from "@airace/sim";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { biggestBrain } from "../src/helpers/limits.ts";
import { resumeKey } from "../src/training/service.ts";
import { createWorldRecord } from "../src/worlds/routes.ts";
import { eventually, FakeWorker } from "./fakeWorker.ts";
import {
  closeSockets,
  openSockets,
  owner,
  plan,
  RESULT,
  runOf,
  send,
  start,
  worker,
} from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
import { settle, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

const TINY = paramCount(2, 64); // BALANCE_TEST's only brain

async function attempt(ws: WsClient, worldId: string, cmd: unknown): Promise<string | null> {
  const cid = randomUUID();
  ws.send({ v: 1, type: "command", cid, worldId, cmd });
  return Promise.race([
    ws.next("ack", (m) => m.cid === cid).then(() => null),
    ws.next("reject", (m) => m.cid === cid).then((m) => m.reason),
  ]);
}

async function limit(ws: WsClient, worldId: string) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q: { kind: "trainingLimit", worldId } });
  return ws.next("queryResult", (m) => m.id === id);
}

async function keyOf(s: TestServer, cookie: string): Promise<string> {
  const res = await api(s, "/api/helper/key", { cookie });
  return ((await res.json()) as { key: string }).key;
}

/** A running world (Fair mode as given) whose host Ana has a company and a socket. */
async function anaWorld(s: TestServer, fairMode: boolean) {
  const ana = await createGuest(s, "Ana");
  const { worldId } = await createWorldRecord(s.ctx, {
    name: "Friends",
    hostUserId: ana.userId,
    hostName: "Ana",
    fairMode,
  });
  const ws = await owner(s, ana.cookie, worldId);
  await send(ws, worldId, {
    type: "createCompany",
    name: "Acme",
    logo: { shape: "circle", symbol: "spark", initials: "AC" },
    palette: { primary: "#4f8cff", secondary: "#ffb020" },
  });
  await send(ws, worldId, { type: "startWorld" });
  const actor = await s.ctx.worlds.get(worldId);
  return { ...ana, worldId, ws, actor };
}

describe("the biggest brain a card can train", () => {
  it("is the largest brain whose parameters fit", () => {
    const small = paramCount(6, 192);
    expect(biggestBrain(BALANCE_V2, small)?.id).toBe("small");
    expect(biggestBrain(BALANCE_V2, small - 1)?.id).toBe("tiny");
    expect(biggestBrain(BALANCE_V2, 1_000)).toBeNull();
  });
});

describe("training fits the player's card", () => {
  it("refuses training with no card online, and the refusal is never logged", async () => {
    server = await start();
    const o = await anaWorld(server, false);
    expect(await attempt(o.ws, o.worldId, { type: "startTraining", plan: plan() })).toBe(
      "Connect your computer to do this.",
    );
    const company = Object.values(o.actor.state().companies).find((c) => !c.rival);
    expect(Object.keys(company?.runs ?? {})).toEqual([]);
    const logged = await server.ctx.db.select().from(schema.worldInputs);
    expect(logged.length).toBeGreaterThan(0);
    expect(JSON.stringify(logged)).not.toContain("startTraining");
  });

  it("refuses a brain too big for the player's card (Fair off) and lets a fitting one through", async () => {
    server = await start();
    const o = await anaWorld(server, false);
    const small = await worker(server, { maxParams: TINY - 1, speed: 20_000 });
    expect(await attempt(o.ws, o.worldId, { type: "startTraining", plan: plan() })).toBe(
      "Your graphics card is too small to train an AI.",
    );
    expect((await limit(o.ws, o.worldId)).data).toEqual({
      maxBrain: null,
      tokensPerSecond: 20_000,
    });
    small.close();
    await worker(server, { workerId: "big", maxParams: TINY, speed: 30_000 });
    await eventually(
      async () =>
        ((await limit(o.ws, o.worldId)).data as { maxBrain: string | null }).maxBrain === "tiny",
    );
    expect(await attempt(o.ws, o.worldId, { type: "startTraining", plan: plan() })).toBeNull();
  });

  it("Fair off: a friend's card never counts for the host", async () => {
    server = await start();
    const o = await anaWorld(server, false);
    const ben = await createGuest(server, "Ben");
    await o.actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    const bens = await FakeWorker.connect(server, {
      token: await keyOf(server, ben.cookie),
      workerId: "ben-pc",
    });
    openSockets.push(bens);
    await eventually(() => server?.ctx.broker.connections().length === 1);
    expect(await attempt(o.ws, o.worldId, { type: "startTraining", plan: plan() })).toBe(
      "Connect your computer to do this.",
    );
  });

  it("Fair on: the pool's biggest card sets the limit", async () => {
    server = await start();
    const o = await anaWorld(server, true);
    const ben = await createGuest(server, "Ben");
    await o.actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    await worker(server, { maxParams: TINY - 1 });
    expect(await attempt(o.ws, o.worldId, { type: "startTraining", plan: plan() })).toBe(
      "The shared cards are too small to train an AI.",
    );
    const bens = await FakeWorker.connect(server, {
      token: await keyOf(server, ben.cookie),
      workerId: "ben-pc",
      maxParams: TINY,
      jobTypes: ["generate"],
    });
    openSockets.push(bens);
    await eventually(() => server?.ctx.broker.connections().length === 2);
    expect(await attempt(o.ws, o.worldId, { type: "startTraining", plan: plan() })).toBeNull();
  });
});

describe("runs that move (Fair mode)", () => {
  it("saves progress to the host and continues on the next card", async () => {
    server = await start();
    const o = await anaWorld(server, true);
    const first = await worker(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const lease = await first.nextLease();
    const spec = TrainJobSpec.parse(lease.spec);
    const run = runOf(o.actor, o.userId);
    expect(spec.resumeUrl).toContain(encodeURIComponent(resumeKey(run.id)).replace(/%2F/g, "/"));
    expect(spec.resumeFromUrl).toBeNull();
    expect(
      (await fetch(spec.resumeUrl ?? "", { method: "PUT", body: "resume-state" })).status,
    ).toBe(204);
    first.close();
    await eventually(() => server?.ctx.broker.connections().length === 0);
    await server.scheduler.advance(40_000); // the lease runs out

    const second = await worker(server, { workerId: "second" });
    const again = await second.nextLease();
    expect(again.jobId).toBe(lease.jobId);
    const moved = TrainJobSpec.parse(again.spec);
    expect(moved.resumeFromUrl).toBeTruthy();
    expect(await (await fetch(moved.resumeFromUrl ?? "")).text()).toBe("resume-state");

    await fetch(moved.outTarUrl, { method: "PUT", body: "ckpt" });
    second.done(again.jobId, RESULT);
    await eventually(async () => !(await server?.ctx.blobs.exists(resumeKey(run.id))));
  });

  it("Fair off: no progress goes to the host", async () => {
    server = await start();
    const o = await anaWorld(server, false);
    const w = await worker(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const spec = TrainJobSpec.parse((await w.nextLease()).spec);
    expect(spec.resumeUrl).toBeNull();
    expect(spec.resumeFromUrl).toBeNull();
    await settle(10);
  });
});
