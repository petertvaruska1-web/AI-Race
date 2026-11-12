import { randomUUID } from "node:crypto";
import {
  type JobType,
  type ServerToWorker,
  TRAINING_ENGINE,
  type WorkerCapabilities,
  type WorkerMessage,
} from "@airace/protocol";
import { and, asc, eq, lt, notInArray, or } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { schema } from "../db/index.ts";
import type { TimerHandle } from "../scheduler.ts";
import { FairShare } from "./fairShare.ts";

export type JobRow = typeof schema.jobs.$inferSelect;
export type JobStatus = "queued" | "leased" | "done" | "failed" | "cancelled";

/** What a prepare hook returns to cancel a job instead of handing it out (it waited too long). */
export const DROP_JOB = "drop";
/** Customer questions older than this many game days are dropped rather than answered (Stage 2). */
export const STALE_AFTER_DAYS = 14;

/** Whether a job asked on `enqueuedDay` is too old to answer on `day` (old jobs have no day). */
export function isStale(enqueuedDay: number | undefined, day: number): boolean {
  return enqueuedDay !== undefined && day - enqueuedDay > STALE_AFTER_DAYS;
}
type Handler<T> = (job: JobRow, value: T) => void | Promise<void>;

/**
 * Lower runs first (game-layer design §12): a player waiting on a chat reply, then
 * a battle, the player's training, scoring, weekly customer feedback, and last a
 * chat with a rival's AI (a `generate` job enqueued with priority 5).
 */
export const JOB_PRIORITY: Record<JobType, number> = {
  generate: 0,
  minigame: 0, // the player is watching, like a chat
  battle: 1,
  keynote: 1,
  interview: 1, // the player just published and is waiting for the reaction
  train: 2,
  eval: 3,
  feedback: 4,
  appReview: 4,
};
export const LEASE_MS = 30_000;
/** How many queued jobs one dispatch looks at. */
const DISPATCH_WINDOW = 500;
export const MAX_ATTEMPTS = 3;
const SWEEP_MS = 5_000;
/** How long a "the player is in the Playground" sign lasts (the client sends one every 15 s). */
export const PLAYGROUND_MS = 30_000;
/** Jobs that wait while their world's player is in the Playground, so the AI answers them first. */
export const HELD_IN_PLAYGROUND: JobType[] = ["feedback", "appReview"];
export const LOST_WORKER = "The computer running it stopped responding.";

/** What routing needs to know about a job's world. */
type RouteWorld = { fairMode: boolean; hostUserId: string; members: Set<string>; day: number };

export type EnqueueOptions = {
  priority?: number;
  /** The player whose card does it (Fair off), or whose share it counts against (Fair on). */
  owner?: string | null;
  companyId?: string | null;
  /** The biggest model it loads, in parameters. */
  minParams?: number;
};

/**
 * Whether `conn` may run `job` (Stage 2): it must hold the job's model; then, with Fair mode
 * off, only its owner's cards (the house card counts as the world host's); with it on, any card
 * of a member of the job's world, or the house card. A job without a world runs anywhere.
 */
export function canRun(job: JobRow, conn: WorkerConn, world: RouteWorld | null): boolean {
  if (conn.caps.maxParams < job.minParams) return false;
  if (!job.worldId || !world) return true;
  if (world.fairMode) return conn.ownerUserId === null || world.members.has(conn.ownerUserId);
  const owner = job.ownerUserId ?? world.hostUserId;
  return conn.ownerUserId === null ? owner === world.hostUserId : conn.ownerUserId === owner;
}

/** A connected worker, as the broker sees it. */
export type WorkerConn = {
  workerId: string;
  caps: WorkerCapabilities;
  send(msg: ServerToWorker): void;
  close(code: number, reason: string): void;
  /** Jobs leased to this connection, by id. */
  leases: Map<string, JobType>;
  /** The player whose graphics card this is; null for the house card (the PC running the game). */
  ownerUserId: string | null;
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
  /** When each world last said its player is in the Playground. */
  private playground = new Map<string, number>();
  /** Card time per company in Fair-mode worlds. */
  readonly fairShare = new FairShare();
  /** When each lease began, for the Fair-mode share. */
  private leasedAt = new Map<string, number>();

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

  /**
   * The world's player is in the Playground: its customer feedback waits until
   * `PLAYGROUND_MS` after the last sign (the lease sweep picks it up again).
   */
  setPlayground(worldId: string): void {
    this.playground.set(worldId, this.ctx.opts.scheduler.now());
  }

  private playgroundWorlds(): string[] {
    const now = this.ctx.opts.scheduler.now();
    for (const [id, at] of this.playground)
      if (now - at >= PLAYGROUND_MS) this.playground.delete(id);
    return [...this.playground.keys()];
  }

  async enqueue(
    type: JobType,
    payload: unknown,
    worldId: string | null,
    opts: EnqueueOptions = {},
  ): Promise<string> {
    const id = randomUUID();
    const now = this.ctx.opts.scheduler.now();
    await this.ctx.db.insert(schema.jobs).values({
      id,
      type,
      priority: opts.priority ?? JOB_PRIORITY[type],
      worldId,
      ownerUserId: opts.owner ?? null,
      companyId: opts.companyId ?? null,
      minParams: opts.minParams ?? 0,
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
  private cardsChanged = new Set<() => void>();

  /** Called whenever a card connects or leaves. */
  onCardsChanged(fn: () => void): void {
    this.cardsChanged.add(fn);
  }

  private cardsMoved(): void {
    for (const fn of this.cardsChanged) fn();
  }

  /** Every connected card (for the cards view and the routing). */
  connections(): WorkerConn[] {
    return [...this.workers.values()];
  }

  attach(conn: WorkerConn, running: string[]): Promise<void> {
    return this.serial(async () => {
      const previous = this.workers.get(conn.workerId);
      if (previous && previous !== conn) previous.close(4409, "replaced by a new connection");
      this.workers.set(conn.workerId, conn);
      this.cardsMoved();
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
    if (this.workers.get(conn.workerId) === conn) {
      this.workers.delete(conn.workerId);
      this.cardsMoved();
    }
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
          await this.noteCardTime(job);
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
          await this.noteCardTime(job);
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

  /**
   * Gives each card's free slots the most urgent jobs it may run (`canRun`), fastest cards
   * first. Jobs go by priority, then (in Fair-mode worlds) the company that has used the cards
   * least, then age. Call inside `serial`.
   */
  private async dispatch(): Promise<void> {
    const identity = this.ctx.catalog?.dataIdentity;
    if (this.closed || !identity) return;
    // A worker with other data would train on something the game doesn't describe, and one
    // with another training engine would train it differently: neither gets work.
    const workers = [...this.workers.values()]
      .filter((c) => c.caps.dataIdentity === identity && c.caps.engine === TRAINING_ENGINE)
      .sort((a, b) => b.caps.speed - a.caps.speed || (a.workerId < b.workerId ? -1 : 1));
    if (workers.length === 0) return;
    const playing = this.playgroundWorlds();
    const notHeld =
      playing.length === 0
        ? undefined
        : or(
            notInArray(schema.jobs.type, HELD_IN_PLAYGROUND),
            notInArray(schema.jobs.worldId, playing),
          );
    const queued = await this.ctx.db
      .select()
      .from(schema.jobs)
      .where(and(eq(schema.jobs.status, "queued"), notHeld))
      .orderBy(asc(schema.jobs.priority), asc(schema.jobs.seq))
      .limit(DISPATCH_WINDOW);
    if (queued.length === 0) return;
    const worlds = new Map<string, RouteWorld | null>();
    for (const job of queued) {
      if (job.worldId && !worlds.has(job.worldId)) {
        worlds.set(job.worldId, await this.routeWorld(job.worldId));
      }
    }
    const usage = (job: JobRow) => {
      const w = job.worldId ? worlds.get(job.worldId) : null;
      if (!w?.fairMode || !job.worldId) return 0;
      return this.fairShare.usage(job.worldId, job.companyId ?? job.ownerUserId ?? "", w.day);
    };
    queued.sort((a, b) => a.priority - b.priority || usage(a) - usage(b) || a.seq - b.seq);
    for (const conn of workers) {
      for (;;) {
        const free = conn.caps.jobTypes.filter((t) => conn.caps.slots[t] > countLeases(conn, t));
        if (free.length === 0) break;
        const index = queued.findIndex(
          (j) =>
            (free as string[]).includes(j.type) &&
            canRun(j, conn, j.worldId ? (worlds.get(j.worldId) ?? null) : null),
        );
        if (index < 0) break;
        const [next] = queued.splice(index, 1);
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
          if (spec === DROP_JOB) {
            await this.update(next.id, { status: "cancelled", error: "stale" });
            this.ctx.log.info({ jobId: next.id, type: next.type }, "stale job dropped");
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
        this.leasedAt.set(job.id, this.ctx.opts.scheduler.now());
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

  /** What routing needs to know about a world (null: none, or it can't be opened). */
  private async routeWorld(worldId: string): Promise<RouteWorld | null> {
    try {
      const actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
      const s = actor.state();
      return {
        fairMode: s.fairMode === true,
        hostUserId: s.hostUserId,
        members: new Set(Object.keys(s.members)),
        day: s.day,
      };
    } catch {
      return null;
    }
  }

  /** Adds a finished lease's card time to its company's Fair-mode share. */
  private async noteCardTime(job: JobRow): Promise<void> {
    const at = this.leasedAt.get(job.id);
    this.leasedAt.delete(job.id);
    if (at === undefined || !job.worldId) return;
    const world = await this.routeWorld(job.worldId);
    if (!world?.fairMode) return;
    const company = job.companyId ?? job.ownerUserId ?? "";
    this.fairShare.add(job.worldId, company, this.ctx.opts.scheduler.now() - at, world.day);
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
