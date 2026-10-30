import type { JobType, ServerToWorker, WorkerCapabilities } from "@airace/protocol";
import WebSocket from "ws";
import type { TestServer } from "./harness.ts";

type Lease = Extract<ServerToWorker, { type: "lease" }>;

export type FakeWorkerOptions = {
  workerId?: string;
  token?: string;
  dataIdentity?: string;
  jobTypes?: JobType[];
  slots?: Partial<WorkerCapabilities["slots"]>;
  running?: string[];
};

/** A scripted ML worker for tests (never shipped): records leases, sends what the test says. */
export class FakeWorker {
  readonly leases: Lease[] = [];
  readonly cancels: string[] = [];
  private waiters: (() => void)[] = [];
  readonly closed: Promise<number>;

  private constructor(private readonly ws: WebSocket) {
    ws.on("message", (raw) => {
      const msg = JSON.parse(raw.toString()) as ServerToWorker;
      if (msg.type === "lease") this.leases.push(msg);
      if (msg.type === "cancel") this.cancels.push(msg.jobId);
      if (msg.type === "wwelcome") this.welcomed = true;
      for (const wake of this.waiters.splice(0)) wake();
    });
    this.closed = new Promise((resolve) => ws.on("close", (code) => resolve(code)));
  }

  private welcomed = false;
  private taken = 0;

  static async connect(s: TestServer, o: FakeWorkerOptions = {}): Promise<FakeWorker> {
    const ws = new WebSocket(`${s.wsUrl}/worker`);
    const worker = new FakeWorker(ws);
    await new Promise<void>((resolve, reject) => {
      ws.once("open", () => resolve());
      ws.once("error", reject);
    });
    const jobTypes = o.jobTypes ?? ["train", "eval", "generate"];
    worker.send({
      type: "whello",
      token: o.token ?? s.ctx.opts.workerToken,
      workerId: o.workerId ?? "fake-worker",
      capabilities: {
        device: "cpu",
        vramMb: 0,
        maxParams: 50_000_000,
        jobTypes,
        slots: { train: 1, eval: 1, generate: 1, feedback: 1, battle: 1, ...o.slots },
        dataIdentity: o.dataIdentity ?? s.ctx.catalog?.dataIdentity ?? "none",
      },
      running: o.running ?? [],
    });
    if (o.token === undefined) await worker.until(() => worker.welcomed, "wwelcome");
    return worker;
  }

  send(msg: Record<string, unknown>): void {
    this.ws.send(JSON.stringify({ v: 1, ...msg }));
  }

  /** The next lease not yet returned by an earlier call. */
  async nextLease(timeoutMs = 3000): Promise<Lease> {
    await this.until(() => this.leases.length > this.taken, "lease", timeoutMs);
    return this.leases[this.taken++] as Lease;
  }

  heartbeat(): void {
    this.send({ type: "heartbeat" });
  }
  progress(jobId: string, event: unknown): void {
    this.send({ type: "progress", jobId, event });
  }
  chunk(jobId: string, text: string): void {
    this.send({ type: "chunk", jobId, text });
  }
  done(jobId: string, result: unknown): void {
    this.send({ type: "done", jobId, result });
  }
  failed(jobId: string, error: string, retryable: boolean): void {
    this.send({ type: "failed", jobId, error, retryable });
  }

  close(): void {
    this.ws.close();
  }

  private async until(ok: () => boolean, what: string, timeoutMs = 3000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!ok()) {
      const left = deadline - Date.now();
      if (left <= 0) throw new Error(`fake worker: no ${what} within ${timeoutMs} ms`);
      await new Promise<void>((resolve) => {
        const t = setTimeout(resolve, left);
        this.waiters.push(() => {
          clearTimeout(t);
          resolve();
        });
      });
    }
  }
}

/** Polls `check` (real time) until it returns a truthy value. */
export async function eventually<T>(
  check: () => Promise<T> | T,
  timeoutMs = 3000,
): Promise<NonNullable<T>> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await check();
    if (value) return value as NonNullable<T>;
    if (Date.now() > deadline) throw new Error("condition not met in time");
    await new Promise((r) => setTimeout(r, 20));
  }
}
