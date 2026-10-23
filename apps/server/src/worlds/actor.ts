import { getBalance, type Input, step, type WorldState } from "@airace/sim";
import { and, asc, desc, eq, gt } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { SIM_VERSION } from "../config.ts";
import { schema } from "../db/index.ts";
import type { Logger } from "../log.ts";
import type { TimerHandle } from "../scheduler.ts";

export type SubmitMeta = { cid?: string; userId?: string };
export type SubmitResult =
  | { ok: true; seq: number; duplicate: boolean }
  | { ok: false; reason: string };
export type ChangeListener = (state: WorldState, seq: number) => void;

const SNAPSHOT_EVERY_MS = 15_000;
const UNLOAD_AFTER_FREEZE_MS = 60_000;

/** True for a Postgres unique-constraint violation, however the driver wraps it. */
export function isUniqueViolation(err: unknown): boolean {
  for (let e = err as { code?: string; cause?: unknown } | undefined; e; e = e.cause as typeof e) {
    if (e.code === "23505") return true;
  }
  return false;
}

/**
 * The single writer of one world (spec §5.1). Inputs are applied one at a time:
 * dedupe by command id, `step`, append to the input log, swap the state, notify.
 * Replaying the log from the latest snapshot rebuilds the same state.
 */
export class WorldActor {
  private current: WorldState;
  private queue: Promise<unknown> = Promise.resolve();
  private listeners = new Set<ChangeListener>();
  private tickTimer: { handle: TimerHandle; ms: number } | null = null;
  private snapshotTimer: TimerHandle | null = null;
  private unloadTimer: TimerHandle | null = null;
  private lastSnapshotSeq: number;
  private closed = false;
  private readonly log: Logger;

  private constructor(
    private readonly ctx: AppContext,
    readonly worldId: string,
    state: WorldState,
    snapshotSeq: number,
    private readonly onUnload: (actor: WorldActor) => void,
  ) {
    this.current = state;
    this.lastSnapshotSeq = snapshotSeq;
    this.log = ctx.log.child({ worldId });
  }

  /** The latest snapshot plus a replay of every later input. */
  static async load(
    ctx: AppContext,
    worldId: string,
    onUnload: (actor: WorldActor) => void = () => {},
  ): Promise<WorldActor> {
    const [snap] = await ctx.db
      .select()
      .from(schema.worldSnapshots)
      .where(eq(schema.worldSnapshots.worldId, worldId))
      .orderBy(desc(schema.worldSnapshots.seq))
      .limit(1);
    if (!snap) throw new Error(`World ${worldId} not found.`);
    if (snap.simVersion !== SIM_VERSION) {
      throw new Error(
        `World ${worldId} was saved by simulation ${snap.simVersion}, but this server runs ${SIM_VERSION}. ` +
          "It can't be replayed safely; start it with a matching server version.",
      );
    }
    let state = snap.state as WorldState;
    const inputs = await ctx.db
      .select({ seq: schema.worldInputs.seq, payload: schema.worldInputs.payload })
      .from(schema.worldInputs)
      .where(and(eq(schema.worldInputs.worldId, worldId), gt(schema.worldInputs.seq, snap.seq)))
      .orderBy(asc(schema.worldInputs.seq));
    for (const row of inputs) {
      const r = step(state, row.payload as Input);
      if (!r.ok || r.state.seq !== row.seq) {
        throw new Error(`World ${worldId} replay diverged at input ${row.seq}.`);
      }
      state = r.state;
    }
    const actor = new WorldActor(ctx, worldId, state, snap.seq, onUnload);
    actor.reschedule();
    return actor;
  }

  state(): WorldState {
    return this.current;
  }

  onChange(listener: ChangeListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Applies one input after every earlier one. */
  submit(input: Input, meta: SubmitMeta = {}): Promise<SubmitResult> {
    const run = this.queue.then(() => this.apply(input, meta));
    this.queue = run.catch(() => {});
    return run;
  }

  private async apply(input: Input, meta: SubmitMeta): Promise<SubmitResult> {
    if (this.closed) return { ok: false, reason: "This world is closed." };
    if (meta.cid && meta.userId) {
      const dup = await this.findCommand(meta.userId, meta.cid);
      if (dup !== null) return { ok: true, seq: dup, duplicate: true };
    }
    const r = step(this.current, input);
    if (!r.ok) return r;
    const seq = r.state.seq;
    try {
      await this.ctx.db.insert(schema.worldInputs).values({
        worldId: this.worldId,
        seq,
        kind: input.kind,
        payload: input,
        cid: meta.cid ?? null,
        userId: meta.userId ?? null,
        at: this.ctx.opts.scheduler.now(),
      });
    } catch (err) {
      if (meta.cid && meta.userId && isUniqueViolation(err)) {
        const dup = await this.findCommand(meta.userId, meta.cid);
        if (dup !== null) return { ok: true, seq: dup, duplicate: true };
      }
      throw err;
    }
    const before = this.current;
    this.current = r.state;
    for (const listener of this.listeners) {
      try {
        listener(r.state, seq);
      } catch (err) {
        this.log.error({ err }, "world change listener failed");
      }
    }
    await this.afterChange(before, r.state);
    return { ok: true, seq, duplicate: false };
  }

  private async findCommand(userId: string, cid: string): Promise<number | null> {
    const [row] = await this.ctx.db
      .select({ seq: schema.worldInputs.seq })
      .from(schema.worldInputs)
      .where(
        and(
          eq(schema.worldInputs.worldId, this.worldId),
          eq(schema.worldInputs.userId, userId),
          eq(schema.worldInputs.cid, cid),
        ),
      )
      .limit(1);
    return row ? row.seq : null;
  }

  private anyoneOnline(s: WorldState): boolean {
    return Object.values(s.members).some((m) => m.online);
  }

  private async afterChange(before: WorldState, after: WorldState): Promise<void> {
    if (this.unloadTimer) {
      this.ctx.opts.scheduler.clearTimeout(this.unloadTimer);
      this.unloadTimer = null;
    }
    const paused = !before.paused && after.paused;
    const froze = this.anyoneOnline(before) && !this.anyoneOnline(after);
    if (paused || froze) await this.snapshot(froze ? "freeze" : "pause");
    if (froze) {
      this.unloadTimer = this.ctx.opts.scheduler.setTimeout(
        () => this.unload(),
        UNLOAD_AFTER_FREEZE_MS,
      );
    }
    this.reschedule();
  }

  /** Ticks run while the world runs, is not paused and someone is online (C1). */
  private reschedule(): void {
    if (this.closed) return;
    const s = this.current;
    const sched = this.ctx.opts.scheduler;
    const live = s.phase === "running" && !s.paused && this.anyoneOnline(s);
    const ms = getBalance(s.balance).tickMsAt1x / s.speed;
    if (this.tickTimer && (!live || this.tickTimer.ms !== ms)) {
      sched.clearTimeout(this.tickTimer.handle);
      this.tickTimer = null;
    }
    if (live && !this.tickTimer) {
      this.tickTimer = {
        ms,
        handle: sched.setTimeout(() => {
          this.tickTimer = null;
          return this.submit({ kind: "tick" }).catch((err) =>
            this.log.error({ err }, "tick failed"),
          );
        }, ms),
      };
    }
    if (live && !this.snapshotTimer) {
      this.snapshotTimer = sched.setTimeout(() => {
        this.snapshotTimer = null;
        return this.snapshot("periodic").then(() => this.reschedule());
      }, SNAPSHOT_EVERY_MS);
    }
    if (!live && this.snapshotTimer) {
      sched.clearTimeout(this.snapshotTimer);
      this.snapshotTimer = null;
    }
  }

  /** Saves the current state; a no-op when nothing changed since the last one. */
  async snapshot(reason: string): Promise<void> {
    const s = this.current;
    if (s.seq === this.lastSnapshotSeq) return;
    await this.ctx.db
      .insert(schema.worldSnapshots)
      .values({
        worldId: this.worldId,
        seq: s.seq,
        simVersion: SIM_VERSION,
        state: s,
        at: this.ctx.opts.scheduler.now(),
      })
      .onConflictDoNothing();
    this.lastSnapshotSeq = s.seq;
    this.log.debug({ seq: s.seq, reason }, "world snapshot");
  }

  /** Stops timers and refuses further inputs. Does not snapshot (see `WorldRegistry.shutdown`). */
  async unload(): Promise<void> {
    if (this.closed) return;
    await this.queue;
    this.closed = true;
    const sched = this.ctx.opts.scheduler;
    for (const t of [this.tickTimer?.handle, this.snapshotTimer, this.unloadTimer]) {
      if (t) sched.clearTimeout(t);
    }
    this.tickTimer = null;
    this.snapshotTimer = null;
    this.unloadTimer = null;
    this.listeners.clear();
    this.onUnload(this);
  }

  isClosed(): boolean {
    return this.closed;
  }
}
