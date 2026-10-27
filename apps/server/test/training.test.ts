import { randomUUID } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EvalJobSpec, TrainJobSpec } from "@airace/protocol";
import { BALANCE_TEST, BALANCE_V2 } from "@airace/sim";
import { eq } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { toTrainRunConfig } from "../src/training/config.ts";
import { revealedCount } from "../src/training/reveal.ts";
import { TRAINING_STOPPED } from "../src/training/service.ts";
import { eventually } from "./fakeWorker.ts";
import {
  cashOf,
  closeSockets,
  openSockets,
  owner,
  plan,
  RESULT,
  runOf,
  send,
  setup,
  start,
  toDay,
  worker,
} from "./flows.ts";
import { createGuest, type TestServer } from "./harness.ts";
import { connectAs, settle, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

async function telemetry(ws: WsClient, worldId: string, runId: string) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q: { kind: "runTelemetry", worldId, runId } });
  return ws.next("queryResult", (m) => m.id === id);
}

describe("trainer settings", () => {
  it("turns the default plan into the committed Python config", () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL("../../../packages/protocol/fixtures/train-config.default.json", import.meta.url),
        "utf8",
      ),
    );
    const config = toTrainRunConfig(BALANCE_V2.defaultPlan, {
      runId: "run-default",
      seed: 12345,
      balance: BALANCE_V2,
      parent: null,
    });
    expect(config).toEqual(fixture);
  });

  it("drops unused datasets and points a grown model at its parent", () => {
    const p = {
      ...plan(),
      mix: { ...plan().mix, code: 0, web: 40 },
      review: 0.2,
      probePrompt: "Hi",
    };
    const config = toTrainRunConfig(p, {
      runId: "r",
      seed: 1,
      balance: BALANCE_TEST,
      parent: { dir: "parent" },
    });
    expect(config.mixture).not.toHaveProperty("code");
    expect(config).toMatchObject({ replay: 0.2, parent_dir: "parent", probe_prompts: ["Hi"] });
  });

  it("reveals the events game time has reached", () => {
    const run = { startDay: 10, endDay: 20, totalSteps: 100 };
    const events = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((step) => ({ step }));
    expect(revealedCount(run, 10, 0, events)).toBe(0);
    expect(revealedCount(run, 15, 0, events)).toBe(5);
    expect(revealedCount(run, 15, 0.5, events)).toBe(5);
    expect(revealedCount(run, 15, 0.99, events)).toBe(5);
    expect(revealedCount(run, 16, 0, events)).toBe(6);
    expect(revealedCount(run, 30, 0, events)).toBe(10);
  });
});

describe("training flow", () => {
  it("runs from the command to a scored version, revealing progress at game pace", async () => {
    server = await start();
    const w = await worker(server);
    const o = await setup(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });

    const lease = await w.nextLease();
    expect(lease.jobType).toBe("train");
    const spec = TrainJobSpec.parse(lease.spec);
    const run = runOf(o.actor, o.userId);
    expect(spec.runId).toBe(run.id);
    expect(spec.parentTarUrl).toBeNull();
    expect(spec.config).toEqual(
      toTrainRunConfig(run.plan, {
        runId: run.id,
        seed: run.seed,
        balance: BALANCE_TEST,
        parent: null,
      }),
    );
    await eventually(() => runOf(o.actor, o.userId).computeStarted);

    const sub = `training:${o.worldId}:${run.id}`;
    o.ws.send({ v: 1, type: "subscribe", sub });
    const ben = await createGuest(server, "Ben");
    await o.actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    const benWs = await connectAs(server, ben.cookie);
    openSockets.push(benWs);
    benWs.send({ v: 1, type: "subscribe", sub });
    expect((await benWs.next("error")).reason).toMatch(/can't follow/i);
    expect((await telemetry(benWs, o.worldId, run.id)).ok).toBe(false);

    expect((await fetch(spec.outTarUrl, { method: "PUT", body: "fake-checkpoint" })).status).toBe(
      204,
    );
    for (let step = 1; step <= 8; step++)
      w.progress(lease.jobId, { kind: "step", step, loss: 6 - step * 0.3 });
    await eventually(
      async () => (await server?.ctx.db.select().from(schema.trainingEvents))?.length === 8,
    );

    const total = run.endDay - run.startDay;
    await toDay(server, o.actor, run.startDay + total / 2);
    const half = await telemetry(o.ws, o.worldId, run.id);
    expect(half.ok).toBe(true);
    const shown = (half.data as { events: unknown[] }).events.length;
    expect(shown).toBeGreaterThanOrEqual(3);
    expect(shown).toBeLessThanOrEqual(5);

    await toDay(server, o.actor, run.startDay + Math.round(total * 0.6));
    w.done(lease.jobId, RESULT);
    await eventually(() => runOf(o.actor, o.userId).held !== null);
    expect(runOf(o.actor, o.userId).status).toBe("training");

    await toDay(server, o.actor, run.endDay);
    await eventually(() => runOf(o.actor, o.userId).status === "completed");
    o.ws.send({ v: 1, type: "resync", worldId: o.worldId });
    const view = (await o.ws.next("snapshot")).view;
    expect(view.me.company?.versions.map((v) => v.label)).toEqual(["v1"]);
    await eventually(
      async () => (await server?.ctx.db.select().from(schema.modelVersions))?.length === 1,
    );

    const evalLease = await w.nextLease();
    expect(evalLease.jobType).toBe("eval");
    const evalSpec = EvalJobSpec.parse(evalLease.spec);
    expect(evalSpec.population).toHaveLength(2); // the sample catalog's reference traits
    expect(await (await fetch(evalSpec.modelTarUrl)).text()).toBe("fake-checkpoint");
    w.done(evalLease.jobId, {
      report: {
        suite: "bench-v1",
        scores: { language: { score: 41.5, raw: 0.415, n: 40 } },
        overall: 41.5,
        missing: ["creativity"],
        items: [],
        seconds: 1,
      },
      traits: { curious: 0.4 },
      personalityWords: ["curious"],
    });
    const versionId = runOf(o.actor, o.userId).versionId ?? "";
    await eventually(() => {
      const s = o.actor.state();
      return s.companies[s.members[o.userId]?.companyId ?? ""]?.versions[versionId]?.scores;
    });
    const s = o.actor.state();
    expect(s.companies[s.members[o.userId]?.companyId ?? ""]?.versions[versionId]).toMatchObject({
      scores: { language: 41.5, creativity: null },
      overall: 41.5,
      personalityWords: ["curious"],
    });

    // Every event reached the owner's stream by the end, and no one else's.
    await server.scheduler.advance(1000);
    await eventually(() => {
      const streamed = o.ws
        .unread("stream")
        .flatMap((m) =>
          m.type === "stream" ? ((m.data as { events?: unknown[] }).events ?? []) : [],
        );
      return streamed.length === 8;
    });
    expect(benWs.unread("stream")).toEqual([]);
  });

  it("finishes a run exactly once across a server restart", async () => {
    const dir = mkdtempSync(join(tmpdir(), "airace-training-"));
    temps.push(dir);
    const over = { dbDir: join(dir, "db"), blobDir: join(dir, "blobs") };
    server = await start(over);
    const w = await worker(server);
    const o = await setup(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const lease = await w.nextLease();
    const spec = TrainJobSpec.parse(lease.spec);
    await fetch(spec.outTarUrl, { method: "PUT", body: "fake-checkpoint" });
    for (let step = 1; step <= 3; step++) w.progress(lease.jobId, { kind: "step", step });
    await eventually(
      async () => (await server?.ctx.db.select().from(schema.trainingEvents))?.length === 3,
    );
    const run = runOf(o.actor, o.userId);
    closeSockets();
    await server.close();

    server = await start(over);
    const back = await worker(server, { running: [lease.jobId] });
    await owner(server, o.cookie, o.worldId);
    const actor = await server.ctx.worlds.get(o.worldId);
    await settle(100);
    expect(back.leases).toEqual([]);
    back.done(lease.jobId, RESULT);
    await toDay(server, actor, run.endDay);
    await eventually(() => runOf(actor, o.userId).status === "completed");
    const s = actor.state();
    expect(
      Object.keys(s.companies[s.members[o.userId]?.companyId ?? ""]?.versions ?? {}),
    ).toHaveLength(1);
    expect(cashOf(actor, o.userId)).toBe(BALANCE_TEST.foundingCash - run.preview.cost);
    await eventually(
      async () => (await server?.ctx.db.select().from(schema.modelVersions))?.length === 1,
    );
  });
});

describe("waiting and failing", () => {
  it("waits for compute when no worker is online, then applies a late result at once", async () => {
    server = await start();
    const o = await setup(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const run = runOf(o.actor, o.userId);
    const jobs = await eventually(async () => {
      const rows = await server?.ctx.db
        .select()
        .from(schema.jobs)
        .where(eq(schema.jobs.type, "train"));
      return rows?.length ? rows : null;
    });
    expect(jobs?.[0]?.status).toBe("queued");
    expect(runOf(o.actor, o.userId)).toMatchObject({ status: "training", computeStarted: false });

    await toDay(server, o.actor, run.endDay);
    expect(runOf(o.actor, o.userId).status).toBe("finalizing");
    const w = await worker(server);
    const lease = await w.nextLease();
    await fetch(TrainJobSpec.parse(lease.spec).outTarUrl, { method: "PUT", body: "ckpt" });
    w.done(lease.jobId, RESULT);
    await eventually(() => runOf(o.actor, o.userId).status === "completed");
  });

  it("refunds a run that fails for good", async () => {
    server = await start();
    const w = await worker(server);
    const o = await setup(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const lease = await w.nextLease();
    w.failed(lease.jobId, "the config is impossible", false);
    await eventually(() => runOf(o.actor, o.userId).status === "failed");
    expect(runOf(o.actor, o.userId).failure).toBe(TRAINING_STOPPED);
    expect(cashOf(o.actor, o.userId)).toBe(BALANCE_TEST.foundingCash);
  });

  it("refunds a run whose finished AI was never saved", async () => {
    server = await start();
    const w = await worker(server);
    const o = await setup(server);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const lease = await w.nextLease();
    w.done(lease.jobId, RESULT);
    await eventually(() => runOf(o.actor, o.userId).status === "failed");
    expect(runOf(o.actor, o.userId).failure).toMatch(/couldn't be saved/);
    expect(cashOf(o.actor, o.userId)).toBe(BALANCE_TEST.foundingCash);
  });
});
