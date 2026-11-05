import { afterEach, describe, expect, it } from "vitest";
import { createWorldRecord } from "../src/worlds/routes.ts";
import { eventually, FakeWorker } from "./fakeWorker.ts";
import { api, createGuest, SAMPLE_CATALOG, startTestServer, type TestServer } from "./harness.ts";
import { settle } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

async function keyOf(s: TestServer, cookie: string): Promise<string> {
  const res = await api(s, "/api/helper/key", { cookie });
  return ((await res.json()) as { key: string }).key;
}

/** Ana's world (Fair mode as given) with Ben joined. */
async function world(s: TestServer, fairMode: boolean) {
  const ana = await createGuest(s, "Ana");
  const ben = await createGuest(s, "Ben");
  const { worldId } = await createWorldRecord(s.ctx, {
    name: "Friends",
    hostUserId: ana.userId,
    hostName: "Ana",
    fairMode,
  });
  const actor = await s.ctx.worlds.get(worldId);
  await actor.submit({
    kind: "system",
    event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
  });
  return { ana, ben, worldId };
}

const job = (
  s: TestServer,
  worldId: string,
  owner: string,
  o: { companyId?: string; minParams?: number } = {},
) => s.ctx.broker.enqueue("generate", { text: "x" }, worldId, { owner, ...o });

describe("who runs a job", () => {
  it("Fair off: a friend's job never goes to the house card; it waits for their own", async () => {
    server = await startTestServer({ catalogPath: SAMPLE_CATALOG });
    const { ben, worldId } = await world(server, false);
    const house = await FakeWorker.connect(server, { jobTypes: ["generate"] });
    const id = await job(server, worldId, ben.userId);
    await settle(200);
    expect(house.leases).toEqual([]);
    const bens = await FakeWorker.connect(server, {
      token: await keyOf(server, ben.cookie),
      workerId: "ben-pc",
      jobTypes: ["generate"],
    });
    expect((await bens.nextLease()).jobId).toBe(id);
  });

  it("Fair off: the house card does the host's own jobs", async () => {
    server = await startTestServer({ catalogPath: SAMPLE_CATALOG });
    const { ana, worldId } = await world(server, false);
    const house = await FakeWorker.connect(server, { jobTypes: ["generate"] });
    const id = await job(server, worldId, ana.userId);
    expect((await house.nextLease()).jobId).toBe(id);
  });

  it("Fair on: any member's card, the house card included, takes anyone's job", async () => {
    server = await startTestServer({ catalogPath: SAMPLE_CATALOG });
    const { ben, worldId } = await world(server, true);
    const house = await FakeWorker.connect(server, { jobTypes: ["generate"] });
    const id = await job(server, worldId, ben.userId);
    expect((await house.nextLease()).jobId).toBe(id);
  });

  it("Fair on: the company that has used the cards least goes first", async () => {
    server = await startTestServer({ catalogPath: SAMPLE_CATALOG });
    const { ana, ben, worldId } = await world(server, true);
    server.ctx.broker.fairShare.add(worldId, "c-ana", 60_000, 0);
    const first = await job(server, worldId, ana.userId, { companyId: "c-ana" });
    const second = await job(server, worldId, ben.userId, { companyId: "c-ben" });
    const house = await FakeWorker.connect(server, {
      jobTypes: ["generate"],
      slots: { generate: 1 },
    });
    expect((await house.nextLease()).jobId).toBe(second);
    expect(first).not.toBe(second);
  });

  it("a job too big for a card waits for one that can hold it", async () => {
    server = await startTestServer({ catalogPath: SAMPLE_CATALOG });
    const { ana, worldId } = await world(server, true);
    const small = await FakeWorker.connect(server, {
      jobTypes: ["generate"],
      maxParams: 1_000_000,
    });
    const id = await job(server, worldId, ana.userId, { minParams: 5_000_000 });
    await settle(200);
    expect(small.leases).toEqual([]);
    const big = await FakeWorker.connect(server, {
      token: await keyOf(server, ana.cookie),
      workerId: "ana-pc",
      jobTypes: ["generate"],
      maxParams: 50_000_000,
    });
    expect((await big.nextLease()).jobId).toBe(id);
  });

  it("the faster of two free cards takes a job", async () => {
    server = await startTestServer({ catalogPath: SAMPLE_CATALOG });
    const { ana, ben, worldId } = await world(server, true);
    const slow = await FakeWorker.connect(server, {
      token: await keyOf(server, ana.cookie),
      workerId: "ana-pc",
      jobTypes: ["generate"],
      speed: 10_000,
    });
    const fast = await FakeWorker.connect(server, {
      token: await keyOf(server, ben.cookie),
      workerId: "ben-pc",
      jobTypes: ["generate"],
      speed: 60_000,
    });
    await eventually(() => server?.ctx.broker.connections().length === 2);
    const id = await job(server, worldId, ana.userId);
    expect((await fast.nextLease()).jobId).toBe(id);
    expect(slow.leases).toEqual([]);
  });
});
