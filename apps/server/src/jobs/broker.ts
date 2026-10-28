import { randomUUID } from "node:crypto";
import type { JobType, ServerToWorker, WorkerCapabilities, WorkerMessage } from "@airace/protocol";
import { and, asc, eq, inArray, lt } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { schema } from "../db/index.ts";
import type { TimerHandle } from "../scheduler.ts";

export type JobRow = typeof schema.jobs.$inferSelect;
export type JobStatus = "queued" | "leased" | "done" | "failed" | "cancelled";
type Handler<T> = (job: JobRow, value: T) => void | Promise<void>;

/**
 * Lower runs first (game-layer design §12): a player waiting on a chat reply, then
 * the player's training, then scoring, then weekly customer feedback.
 */
export const JOB_PRIORITY: Record<JobType, number> = {
  generate: 0,
  train: 2,
  eval: 3,
  feedback: 4,
};
export const LEASE_MS = 30_000;
export const MAX_ATTEMPTS = 3;
const SWEEP_MS = 5_000;
export const LOST_WORKER = "The computer running it stopped responding.";

/** A connected worker, as the broker sees it. */
export type WorkerConn = {
  workerId: string;
  caps: WorkerCapabilities;
  send(msg: ServerToWorker): void;
  close(code: number, reason: string): void;
  /** Jobs leased to this connection, by id. */
  leases: Map<string, JobType>;
};

/**
 * The durable job queue (spec §5.4). Jobs live in the `jobs` table; workers
 * lease them while they keep sending heartbeats. A lease that goes quiet for
 * 30 s is requeued (the job fails after 3 attempts). Every state change runs
 * through one queue, so a job is never leased twice.
 */
export class JobBroker {
  private workers = new Map<string, WorkerConn>();
  private progress = new Map<JobType, Handler<unknown>[]>();
  private done = new Map<JobType, Handler<unknown>[]>();
  private failed = new Map<JobType, Handler<string>[]>();
  private leased = new Map<JobType, Handler<{ attempt: number }>[]>();
  private chunks: Handler<string>[] = [];
  private prepare = new Map<JobType, (job: JobRow) => Promise<unknown>>();
  private queue: Promise<unknown> = Promise.resolve();
  private sweepTimer: TimerHandle | null = null;
  private closed = false;

  constructor(private readonly ctx: AppContext) {}

  /** Requeues leases that expired while the server was down, then checks leases every 5 s. */
  async start(): Promise<void> {
    await this.reconcile();
    this.scheduleSweep();
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.sweepTimer) this.ctx.opts.scheduler.clearTimeout(this.sweepTimer);
    this.sweepTimer = null;
    await this.queue;
  }

  onProgress(type: JobType, fn: Handler<unknown>): void {
    push(this.progress, type, fn);
  }
  onChunk(fn: Handler<string>): void {
    this.chunks.push(fn);
  }
  onDone(type: JobType, fn: Handler<unknown>): void {
    push(this.done, type, fn);
  }
  onFailed(type: JobType, fn: Handler<string>): void {
    push(this.failed, type, fn);
  }
  /** Fires each time a job is handed to a worker (`attempt` 1 is the first time). */
  onLeased(type: JobType, fn: Handler<{ attempt: number }>): void {
    push(this.leased, type, fn);
  }

  /**
   * Builds the spec a worker receives when the job is leased (e.g. fresh signed URLs, so
   * a job that waits in the queue never carries expired links). Without one, the payload is sent.
   */
  onPrepare(type: JobType, fn: (job: JobRow) => Promise<unknown>): void {
    this.prepare.set(type, fn);
  }

  async enqueue(type: JobType, payload: unknown, worldId: string | null): Promise<string> {
    const id = randomUUID();
    const now = this.ctx.opts.scheduler.now();
    await this.ctx.db.insert(schema.jobs).values({
      id,
      type,
      priority: JOB_PRIORITY[type],
      worldId,
      payload,
      status: "queued",
      createdAt: now,
      updatedAt: now,
    });
    void this.kick();
    return id;
  }

  async getJob(id: string): Promise<JobRow | null> {
    const [row] = await this.ctx.db.select().from(schema.jobs).where(eq(schema.jobs.id, id));
    return row ?? null;
  }

  /** Stops a job; a worker running it is told to drop it. */
  cancel(jobId: string): Promise<void> {
    return this.serial(async () => {
      const job = await this.getJob(jobId);
      if (!job || job.status === "done" || job.status === "failed") return;
      await this.update(jobId, { status: "cancelled", leaseWorker: null, leaseUntil: null });
      const conn = job.leaseWorker ? this.workers.get(job.leaseWorker) : undefined;
      if (conn?.leases.delete(jobId)) conn.send({ v: 1, type: "cancel", jobId });
    });
  }

  /** A worker said hello. Jobs it still runs stay with it; others it names are cancelled. */
  attach(conn: WorkerConn, running: string[]): Promise<void> {
    return this.serial(async () => {
      const previous = this.workers.get(conn.workerId);
      if (previous && previous !== conn) previous.close(4409, "replaced by a new connection");
      this.workers.set(conn.workerId, conn);
      const now = this.ctx.opts.scheduler.now();
      for (const jobId of running) {
        const job = await this.getJob(jobId);
        const mine = job?.status === "leased" && job.leaseWorker === conn.workerId;
        // A job whose lease ran out but nobody else took is still this worker's work.
        if (job && (mine || job.status === "queued")) {
          await this.update(jobId, {
            status: "leased",
            leaseWorker: conn.workerId,
            leaseUntil: now + LEASE_MS,
          });
          conn.leases.set(jobId, job.type as JobType);
        } else {
          conn.send({ v: 1, type: "cancel", jobId });
        }
      }
      await this.dispatch();
    });
  }

  detach(conn: WorkerConn): void {
    // Its leases stay until they expire: the worker may come back with `running`.
    if (this.workers.get(conn.workerId) === conn) this.workers.delete(conn.workerId);
  }

  async handle(conn: WorkerConn, msg: WorkerMessage): Promise<void> {
    if (this.workers.get(conn.workerId) !== conn) return;
    switch (msg.type) {
      case "whello":
        return;
      case "heartbeat":
        await this.ctx.db
          .update(schema.jobs)
          .set({ leaseUntil: this.ctx.opts.scheduler.now() + LEASE_MS })
          .where(and(eq(schema.jobs.status, "leased"), eq(schema.jobs.leaseWorker, conn.workerId)));
        return;
      case "progress": {
        const job = await this.leasedTo(conn, msg.jobId);
        if (job) await this.fire(this.progress.get(job.type as JobType), job, msg.event);
        return;
      }
      case "chunk": {
        const job = await this.leasedTo(conn, msg.jobId);
        if (job) await this.fire(this.chunks, job, msg.text);
        return;
      }
      case "done":
        return this.serial(async () => {
          const job = await this.leasedTo(conn, msg.jobId);
          if (!job) return;
          conn.leases.delete(job.id);
          const finished = await this.update(job.id, {
            status: "done",
            result: msg.result,
            leaseWorker: null,
            leaseUntil: null,
          });
          await this.fire(this.done.get(job.type as JobType), finished, msg.result);
          await this.dispatch();
        });
      case "failed":
        return this.serial(async () => {
          const job = await this.leasedTo(conn, msg.jobId);
          if (!job) return;
          conn.leases.delete(job.id);
          await this.retryOrFail(job, msg.error, msg.retryable);
          await this.dispatch();
        });
    }
  }

  /** Requeues (or fails) every lease that has gone 30 s without a heartbeat. */
  reconcile(): Promise<void> {
    return this.serial(async () => {
      const now = this.ctx.opts.scheduler.now();
      const expired = await this.ctx.db
        .select()
        .from(schema.jobs)
        .where(and(eq(schema.jobs.status, "leased"), lt(schema.jobs.leaseUntil, now)))
        .orderBy(asc(schema.jobs.seq));
      for (const job of expired) {
        const conn = job.leaseWorker ? this.workers.get(job.leaseWorker) : undefined;
        if (conn) {
          // A worker that misses its heartbeats is treated as gone.
          this.workers.delete(conn.workerId);
          conn.close(4408, "missed heartbeats");
        }
        await this.retryOrFail(job, LOST_WORKER, true);
      }
      await this.dispatch();
    });
  }

  private async retryOrFail(job: JobRow, error: string, retryable: boolean): Promise<void> {
    if (retryable && job.attempts < MAX_ATTEMPTS) {
      await this.update(job.id, { status: "queued", leaseWorker: null, leaseUntil: null });
      return;
    }
    const failed = await this.update(job.id, {
      status: "failed",
      error,
      leaseWorker: null,
      leaseUntil: null,
    });
    await this.fire(this.failed.get(job.type as JobType), failed, error);
  }

  private scheduleSweep(): void {
    if (this.closed) return;
    this.sweepTimer = this.ctx.opts.scheduler.setTimeout(async () => {
      this.sweepTimer = null;
      await this.reconcile().catch((err) => this.ctx.log.error({ err }, "lease sweep failed"));
      this.scheduleSweep();
    }, SWEEP_MS);
  }

  private kick(): Promise<void> {
    return this.serial(() => this.dispatch()).catch((err) =>
      this.ctx.log.error({ err }, "job dispatch failed"),
    );
  }

  /** Gives each worker's free slots the oldest, most urgent jobs it can run. Call inside `serial`. */
  private async dispatch(): Promise<void> {
    const identity = this.ctx.catalog?.dataIdentity;
    if (this.closed || !identity) return;
    const workers = [...this.workers.values()].sort((a, b) => (a.workerId < b.workerId ? -1 : 1));
    for (const conn of workers) {
      // A worker with other data would train on something the game doesn't describe.
      if (conn.caps.dataIdentity !== identity) continue;
      for (;;) {
        const free = conn.caps.jobTypes.filter((t) => conn.caps.slots[t] > countLeases(conn, t));
        if (free.length === 0) break;
        const [next] = await this.ctx.db
          .select()
          .from(schema.jobs)
          .where(and(eq(schema.jobs.status, "queued"), inArray(schema.jobs.type, free)))
          .orderBy(asc(schema.jobs.priority), asc(schema.jobs.seq))
          .limit(1);
        if (!next) break;
        let spec: unknown = next.payload;
        const prepare = this.prepare.get(next.type as JobType);
        if (prepare) {
          try {
            spec = await prepare(next);
          } catch (err) {
            this.ctx.log.error({ err, jobId: next.id }, "could not prepare job");
            await this.retryOrFail(next, "The job could not be prepared.", false);
            continue;
          }
        }
        const job = await this.update(next.id, {
          status: "leased",
          leaseWorker: conn.workerId,
          leaseUntil: this.ctx.opts.scheduler.now() + LEASE_MS,
          attempts: next.attempts + 1,
        });
        const type = job.type as JobType;
        conn.leases.set(job.id, type);
        conn.send({
          v: 1,
          type: "lease",
          jobId: job.id,
          jobType: type,
          spec,
          leaseMs: LEASE_MS,
        });
        this.ctx.log.info({ jobId: job.id, workerId: conn.workerId, type }, "job leased");
        await this.fire(this.leased.get(type), job, { attempt: job.attempts });
      }
    }
  }

  private async leasedTo(conn: WorkerConn, jobId: string): Promise<JobRow | null> {
    const job = await this.getJob(jobId);
    if (job?.status === "leased" && job.leaseWorker === conn.workerId) return job;
    this.ctx.log.debug(
      { jobId, workerId: conn.workerId },
      "message for a job not leased to this worker",
    );
    return null;
  }

  private async update(id: string, set: Partial<JobRow>): Promise<JobRow> {
    const [row] = await this.ctx.db
      .update(schema.jobs)
      .set({ ...set, updatedAt: this.ctx.opts.scheduler.now() })
      .where(eq(schema.jobs.id, id))
      .returning();
    if (!row) throw new Error(`job ${id} vanished`);
    return row;
  }

  private async fire<T>(handlers: Handler<T>[] | undefined, job: JobRow, value: T): Promise<void> {
    for (const fn of handlers ?? []) {
      try {
        await fn(job, value);
      } catch (err) {
        this.ctx.log.error({ err, jobId: job.id }, "job handler failed");
      }
    }
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch(() => {});
    return run;
  }
}

function push<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const list = map.get(key);
  if (list) list.push(value);
  else map.set(key, [value]);
}

function countLeases(conn: WorkerConn, type: JobType): number {
  let n = 0;
  for (const t of conn.leases.values()) if (t === type) n++;
  return n;
}
