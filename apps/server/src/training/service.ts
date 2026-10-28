import { randomUUID } from "node:crypto";
import { EvalJobResult, type Query, TrainJobResult } from "@airace/protocol";
import { getBalance, type RunState, type SystemEvent, type WorldState } from "@airace/sim";
import { and, asc, eq, ne } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import type { TimerHandle } from "../scheduler.ts";
import type { WorldActor } from "../worlds/actor.ts";
import { toTrainRunConfig, totalSteps } from "./config.ts";
import { type RevealRun, revealedCount } from "./reveal.ts";

/** Signed links live long enough for a slow CPU run; they are made fresh at each lease. */
const JOB_URL_TTL_MS = 12 * 60 * 60 * 1000;
const REVEAL_EVERY_MS = 500;
export const TRAINING_STOPPED =
  "Training stopped because the computer running it had a problem. Your money was refunded.";
const SAVE_FAILED = "The finished AI couldn't be saved. Your money was refunded.";

/** What a train job carries; the worker gets a spec built from it at lease time. */
export type TrainPayload = {
  worldId: string;
  runId: string;
  companyId: string;
  versionId: string;
  parentVersionId: string | null;
  config: Record<string, unknown>;
};
type EvalPayload = { worldId: string; versionId: string };
type StoredEvent = { step: number; event: unknown };
type ActiveRun = RevealRun & { worldId: string; runId: string; done: boolean };

export const checkpointKey = (versionId: string) => `checkpoints/${versionId}.tar`;

/**
 * From a player's "start training" to a scored version (plan Task 14):
 * accepted runs become train jobs; progress is stored and revealed at game pace;
 * results go back into the world as inputs; new versions get an eval job.
 * Everything it does is repeatable, so a restart can safely run it again.
 */
export class TrainingService {
  private knownRuns = new Set<string>();
  private knownVersions = new Set<string>();
  private events = new Map<string, StoredEvent[]>();
  private revealed = new Map<string, number>();
  private published = new Map<string, number>();
  private active = new Map<string, ActiveRun>();
  private dayClock = new Map<string, { day: number; at: number }>();
  private queue: Promise<unknown> = Promise.resolve();
  private timer: TimerHandle | null = null;
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, gateway, worlds } = ctx;
    worlds.onLoad((actor) => this.attach(actor));
    broker.onPrepare("train", (job) => this.trainSpec(job.payload as TrainPayload));
    broker.onLeased("train", (job, { attempt }) =>
      this.trainLeased(job.payload as TrainPayload, attempt),
    );
    broker.onProgress("train", (job, event) =>
      this.trainProgress(job.payload as TrainPayload, event),
    );
    broker.onDone("train", (job, result) => this.trainDone(job.payload as TrainPayload, result));
    broker.onFailed("train", (job) =>
      this.trainFailed(job.payload as TrainPayload, TRAINING_STOPPED),
    );
    broker.onPrepare("eval", (job) => this.evalSpec(job.payload as EvalPayload));
    broker.onDone("eval", (job, result) => this.evalDone(job.payload as EvalPayload, result));
    broker.onFailed("eval", (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "evaluation failed; the version keeps no scores");
    });
    gateway.registerStream("training", async (userId, sub) => {
      const [, worldId, runId] = sub.split(":");
      return !!(worldId && runId && (await this.ownRun(userId, worldId, runId)));
    });
    gateway.registerQuery("runTelemetry", (userId, q) => this.telemetry(userId, q));
    this.scheduleReveal();
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.timer) this.ctx.opts.scheduler.clearTimeout(this.timer);
    this.timer = null;
    await this.queue;
  }

  // --- following worlds -----------------------------------------------------

  private attach(actor: WorldActor): void {
    const now = () => this.ctx.opts.scheduler.now();
    this.dayClock.set(actor.worldId, { day: actor.state().day, at: now() });
    actor.onChange((state) => {
      if (this.dayClock.get(actor.worldId)?.day !== state.day) {
        this.dayClock.set(actor.worldId, { day: state.day, at: now() });
      }
      void this.serial(() => this.sync(actor));
    });
    // Not awaited: this runs while the registry is still loading the actor.
    void this.serial(() => this.sync(actor));
  }

  /** Makes sure every run has its job and every version its rows and evaluation. */
  private async sync(actor: WorldActor): Promise<void> {
    if (this.closed || actor.isClosed()) return;
    const s = actor.state();
    for (const companyId of Object.keys(s.companies).sort()) {
      const company = s.companies[companyId];
      if (!company) continue;
      for (const run of Object.values(company.runs)) {
        if (run.status === "training" || run.status === "finalizing") {
          this.track(s.worldId, run);
          await this.once(this.knownRuns, run.id, () => this.ensureRun(s, companyId, run));
        } else {
          const a = this.active.get(run.id);
          if (a) a.done = true; // its remaining events go out, then it is dropped
        }
      }
      for (const version of Object.values(company.versions)) {
        await this.once(this.knownVersions, version.id, () =>
          this.ensureVersion(actor, companyId, version.id),
        );
      }
    }
  }

  private async once(seen: Set<string>, id: string, fn: () => Promise<void>): Promise<void> {
    if (seen.has(id)) return;
    seen.add(id);
    try {
      await fn();
    } catch (err) {
      seen.delete(id); // tried again on the next change
      this.ctx.log.error({ err, id }, "training sync failed");
    }
  }

  private track(worldId: string, run: RunState): void {
    if (this.active.has(run.id)) return;
    this.active.set(run.id, {
      worldId,
      runId: run.id,
      startDay: run.startDay,
      endDay: run.endDay,
      totalSteps: totalSteps(run.preview.tokens),
      done: false,
    });
  }

  private async ensureRun(s: WorldState, companyId: string, run: RunState): Promise<void> {
    const { db, broker } = this.ctx;
    const [row] = await db
      .select()
      .from(schema.trainingRuns)
      .where(eq(schema.trainingRuns.id, run.id));
    if (!row?.jobId) {
      const versionId = row?.versionId ?? randomUUID();
      if (!row) {
        // The row comes first, so a crash before the job leaves a row to finish from.
        await db.insert(schema.trainingRuns).values({
          id: run.id,
          worldId: s.worldId,
          companyId,
          jobId: null,
          versionId,
          status: "queued",
          createdAt: this.ctx.opts.scheduler.now(),
        });
      }
      const payload: TrainPayload = {
        worldId: s.worldId,
        runId: run.id,
        companyId,
        versionId,
        parentVersionId: run.plan.parentVersionId,
        config: toTrainRunConfig(run.plan, {
          runId: run.id,
          seed: run.seed,
          balance: getBalance(s.balance),
          parent: run.plan.parentVersionId ? { dir: "parent" } : null,
        }),
      };
      const jobId = await broker.enqueue("train", payload, s.worldId);
      await db.update(schema.trainingRuns).set({ jobId }).where(eq(schema.trainingRuns.id, run.id));
      return;
    }
    // After a restart: a job that ended while its world was not applied gets applied now.
    const job = await broker.getJob(row.jobId);
    if (job?.status === "done") await this.trainDone(job.payload as TrainPayload, job.result);
    if (job?.status === "failed" || job?.status === "cancelled") {
      await this.trainFailed(job.payload as TrainPayload, TRAINING_STOPPED);
    }
  }

  private async ensureVersion(
    actor: WorldActor,
    companyId: string,
    versionId: string,
  ): Promise<void> {
    const { db, broker } = this.ctx;
    const s = actor.state();
    const v = s.companies[companyId]?.versions[versionId];
    if (!v) return;
    const [row] = await db
      .select()
      .from(schema.modelVersions)
      .where(eq(schema.modelVersions.id, versionId));
    if (!row) {
      let [lineage] = await db
        .select()
        .from(schema.lineages)
        .where(
          and(eq(schema.lineages.worldId, s.worldId), eq(schema.lineages.companyId, companyId)),
        );
      if (!lineage) {
        [lineage] = await db
          .insert(schema.lineages)
          .values({ id: randomUUID(), worldId: s.worldId, companyId })
          .returning();
      }
      if (!lineage) throw new Error("lineage insert failed");
      await db.insert(schema.modelVersions).values({
        id: v.id,
        lineageId: lineage.id,
        worldId: s.worldId,
        companyId,
        parentId: v.parentId,
        runId: v.runId,
        blobKey: checkpointKey(v.id),
        shape: { nLayer: v.nLayer, dModel: v.dModel, ctxLen: v.ctxLen },
        createdAt: this.ctx.opts.scheduler.now(),
      });
      await broker.enqueue("eval", { worldId: s.worldId, versionId: v.id }, s.worldId);
      return;
    }
    if (v.scores === null) {
      const [stored] = await db
        .select()
        .from(schema.evalResults)
        .where(eq(schema.evalResults.versionId, versionId));
      if (stored) {
        const traits = EvalJobResult.shape.traits.safeParse(stored.fingerprint);
        await this.applyEval(
          s.worldId,
          versionId,
          stored.report,
          null,
          traits.success ? traits.data : null,
        );
      }
    }
  }

  // --- train jobs -------------------------------------------------------------

  private async trainSpec(p: TrainPayload): Promise<unknown> {
    const { ctx } = this;
    if (p.parentVersionId && !(await ctx.blobs.exists(checkpointKey(p.parentVersionId)))) {
      throw new Error(`parent checkpoint ${p.parentVersionId} is missing`);
    }
    return {
      runId: p.runId,
      config: p.config,
      parentTarUrl: p.parentVersionId
        ? signBlobUrl(ctx, checkpointKey(p.parentVersionId), "GET", JOB_URL_TTL_MS)
        : null,
      outTarUrl: signBlobUrl(ctx, checkpointKey(p.versionId), "PUT", JOB_URL_TTL_MS),
    };
  }

  private async trainLeased(p: TrainPayload, attempt: number): Promise<void> {
    const { db } = this.ctx;
    if (attempt > 1) {
      // A new attempt starts again from the parent (ADR 0008 item 8): earlier telemetry is void.
      await db.delete(schema.trainingEvents).where(eq(schema.trainingEvents.runId, p.runId));
      this.events.set(p.runId, []);
      this.revealed.set(p.runId, 0);
      this.published.set(p.runId, 0);
      this.ctx.gateway.publish(streamName(p), { kind: "reset" });
    }
    await db
      .update(schema.trainingRuns)
      .set({ status: "running" })
      .where(eq(schema.trainingRuns.id, p.runId));
    await this.submit(p.worldId, { type: "computeStarted", runId: p.runId });
  }

  private async trainProgress(p: TrainPayload, event: unknown): Promise<void> {
    const list = await this.loadEvents(p.runId);
    const given = (event as { step?: unknown } | null)?.step;
    const step = Number.isInteger(given) ? (given as number) : (list.at(-1)?.step ?? 0);
    await this.ctx.db
      .insert(schema.trainingEvents)
      .values({ runId: p.runId, idx: list.length, step, event });
    list.push({ step, event });
  }

  private async trainDone(p: TrainPayload, result: unknown): Promise<void> {
    const parsed = TrainJobResult.safeParse(result);
    if (!parsed.success) {
      this.ctx.log.error({ runId: p.runId, issues: parsed.error.issues }, "bad training result");
      return this.trainFailed(p, TRAINING_STOPPED);
    }
    if (!(await this.ctx.blobs.exists(checkpointKey(p.versionId)))) {
      this.ctx.log.error({ runId: p.runId }, "training finished without a saved checkpoint");
      return this.trainFailed(p, SAVE_FAILED);
    }
    await this.ctx.db
      .update(schema.trainingRuns)
      .set({ status: "done" })
      .where(eq(schema.trainingRuns.id, p.runId));
    const r = parsed.data;
    await this.submit(p.worldId, {
      type: "trainingCompleted",
      runId: p.runId,
      versionId: p.versionId,
      result: {
        steps: r.steps,
        finalLoss: r.final_loss,
        heldoutLosses: r.heldout_losses,
        unstable: r.status === "unstable_stopped",
      },
    });
  }

  private async trainFailed(p: TrainPayload, reason: string): Promise<void> {
    await this.ctx.db
      .update(schema.trainingRuns)
      .set({ status: "failed" })
      .where(eq(schema.trainingRuns.id, p.runId));
    await this.submit(p.worldId, { type: "trainingFailed", runId: p.runId, reason });
  }

  // --- eval jobs ----------------------------------------------------------------

  private async evalSpec(p: EvalPayload): Promise<unknown> {
    const { ctx } = this;
    // Personality words are relative: this AI against the world's other AIs and the references.
    const others = await ctx.db
      .select({ fingerprint: schema.evalResults.fingerprint })
      .from(schema.evalResults)
      .innerJoin(schema.modelVersions, eq(schema.modelVersions.id, schema.evalResults.versionId))
      .where(
        and(eq(schema.modelVersions.worldId, p.worldId), ne(schema.modelVersions.id, p.versionId)),
      );
    const population = [
      ...others.map((o) => o.fingerprint).filter((f): f is Record<string, number> => !!f),
      ...(ctx.catalog?.referenceTraits ?? []),
    ];
    return {
      versionId: p.versionId,
      modelTarUrl: signBlobUrl(ctx, checkpointKey(p.versionId), "GET", JOB_URL_TTL_MS),
      population,
    };
  }

  private async evalDone(p: EvalPayload, result: unknown): Promise<void> {
    const parsed = EvalJobResult.safeParse(result);
    if (!parsed.success) {
      this.ctx.log.error(
        { versionId: p.versionId, issues: parsed.error.issues },
        "bad eval result",
      );
      return;
    }
    const { report, traits, personalityWords } = parsed.data;
    await this.ctx.db
      .insert(schema.evalResults)
      .values({
        versionId: p.versionId,
        report: { ...report, personalityWords },
        fingerprint: traits,
      })
      .onConflictDoUpdate({
        target: schema.evalResults.versionId,
        set: { report: { ...report, personalityWords }, fingerprint: traits },
      });
    await this.applyEval(p.worldId, p.versionId, report, personalityWords, traits);
  }

  private async applyEval(
    worldId: string,
    versionId: string,
    report: unknown,
    words: string[] | null,
    traits: Record<string, number> | null,
  ): Promise<void> {
    const r = report as {
      scores?: Record<string, unknown>;
      overall?: unknown;
      missing?: unknown;
      personalityWords?: unknown;
    };
    const scores: Record<string, number | null> = {};
    for (const [category, s] of Object.entries(r.scores ?? {})) {
      const value = typeof s === "number" ? s : (s as { score?: unknown } | null)?.score;
      scores[category] = typeof value === "number" ? value : null;
    }
    if (Array.isArray(r.missing))
      for (const m of r.missing) if (typeof m === "string") scores[m] = null;
    const personalityWords =
      words ??
      (Array.isArray(r.personalityWords)
        ? r.personalityWords.filter((w) => typeof w === "string")
        : []);
    await this.submit(worldId, {
      type: "evalCompleted",
      versionId,
      scores,
      overall: typeof r.overall === "number" ? r.overall : null,
      personalityWords,
      traits,
    });
  }

  // --- game-pace telemetry -------------------------------------------------------

  private async loadEvents(runId: string): Promise<StoredEvent[]> {
    let list = this.events.get(runId);
    if (!list) {
      const rows = await this.ctx.db
        .select()
        .from(schema.trainingEvents)
        .where(eq(schema.trainingEvents.runId, runId))
        .orderBy(asc(schema.trainingEvents.idx));
      list = this.events.get(runId) ?? rows.map((r) => ({ step: r.step, event: r.event }));
      this.events.set(runId, list);
    }
    return list;
  }

  /** How far into the current game day the world is, 0 to just under 1. */
  private dayFraction(s: WorldState): number {
    const live =
      s.phase === "running" && !s.paused && Object.values(s.members).some((m) => m.online);
    const clock = this.dayClock.get(s.worldId);
    if (!live || !clock) return 0;
    const dayMs = getBalance(s.balance).tickMsAt1x / s.speed;
    return Math.min(0.999, Math.max(0, (this.ctx.opts.scheduler.now() - clock.at) / dayMs));
  }

  /** Events revealed so far; never fewer than before (pausing doesn't take any back). */
  private revealNow(
    runId: string,
    run: RevealRun,
    s: WorldState,
    list: StoredEvent[],
    done: boolean,
  ): number {
    const count = done ? list.length : revealedCount(run, s.day, this.dayFraction(s), list);
    const n = Math.max(this.revealed.get(runId) ?? 0, count);
    this.revealed.set(runId, n);
    return n;
  }

  private scheduleReveal(): void {
    if (this.closed) return;
    this.timer = this.ctx.opts.scheduler.setTimeout(() => {
      this.timer = null;
      this.publishReveals();
      this.scheduleReveal();
    }, REVEAL_EVERY_MS);
  }

  private publishReveals(): void {
    for (const run of [...this.active.values()]) {
      const actor = this.ctx.worlds.peek(run.worldId);
      const list = this.events.get(run.runId);
      if (!actor || actor.isClosed() || !list) {
        if (!list) void this.loadEvents(run.runId);
        continue;
      }
      const n = this.revealNow(run.runId, run, actor.state(), list, run.done);
      const from = this.published.get(run.runId) ?? 0;
      if (n > from) {
        this.ctx.gateway.publish(streamName(run), {
          kind: "events",
          from,
          events: list.slice(from, n).map((e) => e.event),
          totalSteps: run.totalSteps,
        });
        this.published.set(run.runId, n);
      }
      if (run.done && n >= list.length) this.active.delete(run.runId);
    }
  }

  /** The run if it belongs to `userId`'s company in `worldId`. */
  private async ownRun(
    userId: string,
    worldId: string,
    runId: string,
  ): Promise<{ s: WorldState; run: RunState } | null> {
    let actor = this.ctx.worlds.peek(worldId);
    if (!actor) {
      try {
        actor = await this.ctx.worlds.get(worldId);
      } catch {
        return null;
      }
    }
    const s = actor.state();
    const companyId = s.members[userId]?.companyId;
    const run = companyId ? s.companies[companyId]?.runs[runId] : undefined;
    return run ? { s, run } : null;
  }

  private async telemetry(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "runTelemetry") throw new HttpError(400, "Unknown request.");
    const owned = await this.ownRun(userId, q.worldId, q.runId);
    if (!owned) throw new HttpError(403, "That isn't one of your training runs.");
    const { s, run } = owned;
    const list = await this.loadEvents(run.id);
    const reveal = {
      startDay: run.startDay,
      endDay: run.endDay,
      totalSteps: totalSteps(run.preview.tokens),
    };
    const done = run.status === "completed" || run.status === "failed";
    const n = this.revealNow(run.id, reveal, s, list, done);
    return {
      runId: run.id,
      totalSteps: reveal.totalSteps,
      events: list.slice(0, n).map((e) => e.event),
    };
  }

  // --- helpers ---------------------------------------------------------------------

  private async submit(worldId: string, event: SystemEvent): Promise<void> {
    const actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
    const r = await actor.submit({ kind: "system", event });
    if (!r.ok)
      this.ctx.log.debug(
        { worldId, event: event.type, reason: r.reason },
        "system input not applied",
      );
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch((err) => this.ctx.log.error({ err }, "training service step failed"));
    return run;
  }
}

function streamName(r: { worldId: string; runId: string }): string {
  return `training:${r.worldId}:${r.runId}`;
}
