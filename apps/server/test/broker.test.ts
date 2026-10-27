import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { CLOSE_BAD_WORKER } from "../src/jobs/workerGateway.ts";
import { eventually, FakeWorker, type FakeWorkerOptions } from "./fakeWorker.ts";
import { SAMPLE_CATALOG, startTestServer, type TestServer } from "./harness.ts";
import { settle } from "./wsClient.ts";

let server: TestServer | null = null;
const workers: FakeWorker[] = [];
const temps: string[] = [];
afterEach(async () => {
  for (const w of workers.splice(0)) w.close();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

const start = (over = {}) => startTestServer({ catalogPath: SAMPLE_CATALOG, ...over });

async function worker(s: TestServer, o: FakeWorkerOptions = {}): Promise<FakeWorker> {
  const w = await FakeWorker.connect(s, o);
  workers.push(w);
  return w;
}

const trainSpec = () => ({
  runId: randomUUID(),
  config: { seed: 1 },
  parentTarUrl: null,
  outTarUrl: "http://127.0.0.1/blobs/out.tar?exp=1&sig=a",
});
const RESULT = { steps: 10, final_loss: 4.2, heldout_losses: {}, status: "completed" };
const TRAIN_ONLY: FakeWorkerOptions = {
  jobTypes: ["train"],
  slots: { train: 1, eval: 0, generate: 0 },
};

describe("leasing", () => {
  it("leases a train job to a worker with a train slot", async () => {
    server = await start();
    const w = await worker(server, TRAIN_ONLY);
    const spec = trainSpec();
    const id = await server.ctx.broker.enqueue("train", spec, null);
    const lease = await w.nextLease();
    expect(lease).toMatchObject({ jobId: id, jobType: "train", spec, leaseMs: 30_000 });
    expect(await server.ctx.broker.getJob(id)).toMatchObject({
      status: "leased",
      leaseWorker: "fake-worker",
      attempts: 1,
    });
  });

  it("serves a chat reply before a waiting training job", async () => {
    server = await start();
    const w = await worker(server, {
      jobTypes: ["train", "generate"],
      slots: { train: 1, eval: 0, generate: 1 },
    });
    const a = await server.ctx.broker.enqueue("train", trainSpec(), null);
    expect((await w.nextLease()).jobId).toBe(a);
    const b = await server.ctx.broker.enqueue("train", trainSpec(), null);
    const g = await server.ctx.broker.enqueue("generate", { text: "hi" }, null);
    expect((await w.nextLease()).jobId).toBe(g);
    w.done(a, RESULT);
    expect((await w.nextLease()).jobId).toBe(b);
  });

  it("gives nothing to a worker with other training data", async () => {
    server = await start();
    const w = await worker(server, { dataIdentity: "someone-elses-data" });
    const id = await server.ctx.broker.enqueue("train", trainSpec(), null);
    await settle(200);
    expect(w.leases).toEqual([]);
    expect((await server.ctx.broker.getJob(id))?.status).toBe("queued");
  });

  it("refuses a worker with the wrong token", async () => {
    server = await start();
    const w = await worker(server, { token: "wrong" });
    expect(await w.closed).toBe(CLOSE_BAD_WORKER);
  });
});

describe("lost and returning workers", () => {
  it("requeues a lease after 30 s without a heartbeat and gives it to another worker", async () => {
    server = await start();
    const w1 = await worker(server, { ...TRAIN_ONLY, workerId: "w-1" });
    const id = await server.ctx.broker.enqueue("train", trainSpec(), null);
    await w1.nextLease();
    const w2 = await worker(server, { ...TRAIN_ONLY, workerId: "w-2" });
    await settle(100);
    expect(w2.leases).toEqual([]);

    // A heartbeat keeps the lease alive past its first 30 s.
    await server.scheduler.advance(20_000);
    w1.heartbeat();
    await settle(100);
    await server.scheduler.advance(20_000);
    expect((await server.ctx.broker.getJob(id))?.leaseWorker).toBe("w-1");

    await server.scheduler.advance(16_000);
    expect((await w2.nextLease()).jobId).toBe(id);
    expect(await server.ctx.broker.getJob(id)).toMatchObject({ leaseWorker: "w-2", attempts: 2 });
    expect(await w1.closed).toBe(4408);
  });

  it("keeps a job with the worker that reconnects still running it", async () => {
    server = await start();
    const w1 = await worker(server, { ...TRAIN_ONLY, workerId: "w-1" });
    const id = await server.ctx.broker.enqueue("train", trainSpec(), null);
    await w1.nextLease();
    w1.close();
    await settle(100);
    const w2 = await worker(server, { ...TRAIN_ONLY, workerId: "w-2" });
    const back = await worker(server, { ...TRAIN_ONLY, workerId: "w-1", running: [id] });
    await settle(100);
    expect(back.leases).toEqual([]);
    expect(w2.leases).toEqual([]);
    back.done(id, RESULT);
    await eventually(async () => (await server?.ctx.broker.getJob(id))?.status === "done");
  });

  it("tells a returning worker to drop a job that is no longer its", async () => {
    server = await start();
    const back = await worker(server, { running: [randomUUID()] });
    await eventually(() => back.cancels.length === 1);
  });
});

describe("results", () => {
  it("stores a finished job's result and tells the listeners", async () => {
    server = await start();
    const seen: unknown[] = [];
    server.ctx.broker.onDone("train", (job, result) => {
      seen.push([job.id, result]);
    });
    const w = await worker(server);
    const id = await server.ctx.broker.enqueue("train", trainSpec(), null);
    await w.nextLease();
    w.done(id, RESULT);
    await eventually(() => seen.length === 1);
    expect(seen).toEqual([[id, RESULT]]);
    expect(await server.ctx.broker.getJob(id)).toMatchObject({
      status: "done",
      result: RESULT,
      leaseWorker: null,
    });
  });

  it("retries a retryable failure and gives up on the rest", async () => {
    server = await start();
    const failures: string[] = [];
    server.ctx.broker.onFailed("train", (_job, error) => {
      failures.push(error);
    });
    const w = await worker(server);
    const id = await server.ctx.broker.enqueue("train", trainSpec(), null);
    await w.nextLease();
    w.failed(id, "out of memory", true);
    expect((await w.nextLease()).jobId).toBe(id);
    w.failed(id, "the plan is impossible", false);
    await eventually(() => failures.length === 1);
    expect(failures).toEqual(["the plan is impossible"]);
    expect((await server.ctx.broker.getJob(id))?.status).toBe("failed");
  });

  it("requeues a lease left over from before a restart once it expires", async () => {
    const dir = mkdtempSync(join(tmpdir(), "airace-broker-"));
    temps.push(dir);
    server = await start({ dbDir: join(dir, "db") });
    const w = await worker(server);
    const id = await server.ctx.broker.enqueue("train", trainSpec(), null);
    await w.nextLease();
    w.close();
    await server.close();

    server = await start({ dbDir: join(dir, "db") });
    expect((await server.ctx.broker.getJob(id))?.status).toBe("leased");
    await server.scheduler.advance(31_000);
    await server.ctx.broker.reconcile();
    expect(await server.ctx.broker.getJob(id)).toMatchObject({
      status: "queued",
      leaseWorker: null,
    });
    const fresh = await worker(server, { workerId: "after-restart" });
    expect((await fresh.nextLease()).jobId).toBe(id);
  });
});
