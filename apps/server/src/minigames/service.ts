import { randomInt, randomUUID } from "node:crypto";
import {
  MinigameJobResult,
  MinigameStepEvent,
  MinigameTextEvent,
  type Query,
  type MinigameJobResult as Result,
} from "@airace/protocol";
import type { CompanyState, ShowcaseState, WorldState } from "@airace/sim";
import { and, desc, eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import type { EnqueueOptions } from "../jobs/broker.ts";
import { forCompany, paramsOf } from "../jobs/owners.ts";
import type { MinigameMessage } from "../realtime/gateway.ts";
import { checkpointKey } from "../training/service.ts";
import type { WorldActor } from "../worlds/actor.ts";

const MODEL_URL_TTL_MS = 12 * 60 * 60 * 1000;
const RUNS_SHOWN = 20;

type MinigamePayload = { runId: string };
type Row = typeof schema.minigameRuns.$inferSelect;

const stream = (r: { worldId: string; id: string }) => `minigame:${r.worldId}:${r.id}`;

/**
 * Playground minigames (playtest pack §4). A practice run or a duel comes from the player's
 * `minigame` message; a showcase from the world (`startShowcase`). Each is one `minigame`
 * job on the worker, which plays it with the real AIs. Its steps stream live to the
 * player on `minigame:<world>:<run>`; a showcase's score goes back to the world as
 * `showcaseResolved`. Runs are private to their player (R2).
 */
export class MinigameService {
  private playing = new Map<string, string>(); // userId -> runId
  private known = new Set<string>();
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, worlds, gateway } = ctx;
    gateway.setMinigameHandler((userId, msg) => this.start(userId, msg));
    worlds.onLoad((actor) => this.attach(actor));
    broker.onPrepare("minigame", (job) => this.spec((job.payload as MinigamePayload).runId));
    broker.onProgress("minigame", (job, event) =>
      this.progress((job.payload as MinigamePayload).runId, event),
    );
    broker.onDone("minigame", (job, result) =>
      this.finish((job.payload as MinigamePayload).runId, result),
    );
    broker.onFailed("minigame", (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "minigame failed");
      return this.finish((job.payload as MinigamePayload).runId, null);
    });
    gateway.registerStream("minigame", async (userId, sub) => {
      const [, worldId, runId] = sub.split(":");
      if (!worldId || !runId) return false;
      const row = await this.row(runId);
      return row?.userId === userId && row.worldId === worldId;
    });
    gateway.registerQuery("minigameRuns", (userId, q) => this.runs(userId, q));
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.queue;
  }

  // -- practice and duels ------------------------------------------------------------------------

  private async start(
    userId: string,
    msg: MinigameMessage,
  ): Promise<{ ok: true; data: unknown } | { ok: false; reason: string }> {
    let actor: WorldActor;
    try {
      actor = this.ctx.worlds.peek(msg.worldId) ?? (await this.ctx.worlds.get(msg.worldId));
    } catch {
      return { ok: false, reason: "That world isn't available." };
    }
    const s = actor.state();
    const companyId = s.members[userId]?.companyId;
    const company = companyId ? s.companies[companyId] : undefined;
    if (!company) return { ok: false, reason: "Found your company first." };
    const version = company.versions[msg.versionId];
    if (!version) return { ok: false, reason: "Pick one of your own AIs to play." };
    if (msg.game === "story" && typeof version.scores?.creativity !== "number") {
      return { ok: false, reason: "Stories can't be judged on this computer yet." };
    }
    if (this.playing.has(userId)) return { ok: false, reason: "Your last game is still going." };
    let opponentPack: string | null = null;
    let opponentName: string | null = null;
    if (msg.opponent) {
      const found = rivalOnSale(s, msg.opponent);
      if (!found) return { ok: false, reason: "That rival AI isn't on sale." };
      opponentPack = found.packKey;
      opponentName = `${found.company.name}'s ${found.productName}`;
    }
    const runId = randomUUID();
    this.playing.set(userId, runId);
    await this.ctx.db.insert(schema.minigameRuns).values({
      id: runId,
      worldId: msg.worldId,
      userId,
      companyId: company.id,
      game: msg.game,
      difficulty: msg.difficulty,
      settings: msg.settings,
      versionId: msg.versionId,
      opponentPack,
      opponentName,
      showcaseId: null,
      seed: randomInt(2 ** 31),
      status: "running",
      createdAt: this.ctx.opts.scheduler.now(),
    });
    await this.enqueue(runId, msg.worldId, forCompany(s, company.id, paramsOf(s, msg.versionId)));
    return { ok: true, data: { runId } };
  }

  private async enqueue(runId: string, worldId: string, routing: EnqueueOptions): Promise<void> {
    const jobId = await this.ctx.broker.enqueue(
      "minigame",
      { runId } satisfies MinigamePayload,
      worldId,
      routing,
    );
    await this.ctx.db
      .update(schema.minigameRuns)
      .set({ jobId })
      .where(eq(schema.minigameRuns.id, runId));
  }

  // -- showcases -------------------------------------------------------------------------------

  private attach(actor: WorldActor): void {
    actor.onChange(() => void this.serial(() => this.sync(actor)));
    void this.serial(() => this.sync(actor));
  }

  /** Every running showcase gets its run once; one that ended while its world was away is applied. */
  private async sync(actor: WorldActor): Promise<void> {
    if (this.closed || actor.isClosed()) return;
    const s = actor.state();
    for (const companyId of Object.keys(s.companies).sort()) {
      const company = s.companies[companyId] as CompanyState;
      for (const x of company.showcases) {
        if (x.status !== "running" || this.known.has(x.id)) continue;
        this.known.add(x.id);
        await this.requestShowcase(s, company, x);
      }
    }
  }

  private async requestShowcase(
    s: WorldState,
    company: CompanyState,
    x: ShowcaseState,
  ): Promise<void> {
    const { db, broker } = this.ctx;
    const [existing] = await db
      .select()
      .from(schema.minigameRuns)
      .where(eq(schema.minigameRuns.showcaseId, x.id));
    if (existing) {
      // After a restart: a job that ended while its world was not loaded is applied now.
      const job = existing.jobId ? await broker.getJob(existing.jobId) : null;
      if (job?.status === "done") await this.finish(existing.id, job.result);
      if (job?.status === "failed" || job?.status === "cancelled")
        await this.finish(existing.id, null);
      return;
    }
    const runId = randomUUID();
    await db.insert(schema.minigameRuns).values({
      id: runId,
      worldId: s.worldId,
      userId: company.ownerUserId ?? randomUUID(),
      companyId: company.id,
      game: x.game,
      difficulty: x.difficulty,
      settings: x.settings,
      versionId: x.versionId,
      opponentPack: null,
      opponentName: null,
      showcaseId: x.id,
      seed: x.seed,
      status: "running",
      createdAt: this.ctx.opts.scheduler.now(),
    });
    await this.enqueue(runId, s.worldId, forCompany(s, company.id, paramsOf(s, x.versionId)));
  }

  // -- the job ---------------------------------------------------------------------------------

  private async row(runId: string): Promise<Row | null> {
    const [row] = await this.ctx.db
      .select()
      .from(schema.minigameRuns)
      .where(eq(schema.minigameRuns.id, runId));
    return row ?? null;
  }

  private async spec(runId: string): Promise<unknown> {
    const row = await this.row(runId);
    if (!row) throw new Error(`minigame run ${runId} is gone`);
    const sides: { side: "a" | "b"; versionId: string; modelTarUrl: string }[] = [
      {
        side: "a",
        versionId: row.versionId,
        modelTarUrl: signBlobUrl(this.ctx, checkpointKey(row.versionId), "GET", MODEL_URL_TTL_MS),
      },
    ];
    if (row.opponentPack) {
      sides.push({
        side: "b",
        versionId: row.opponentPack,
        modelTarUrl: await this.ctx.rivals.modelUrl(row.opponentPack),
      });
    }
    return {
      runId: row.id,
      game: row.game,
      difficulty: row.difficulty,
      seed: row.seed,
      settings: row.settings,
      sides,
    };
  }

  private async progress(runId: string, event: unknown): Promise<void> {
    const step = MinigameStepEvent.safeParse(event);
    const text = step.success ? null : MinigameTextEvent.safeParse(event);
    if (!step.success && !text?.success) return;
    const row = await this.row(runId);
    if (!row) return;
    if (step.success) {
      const { kind: _kind, side, ...rest } = step.data;
      this.ctx.gateway.publish(stream(row), { kind: "step", side, step: rest });
    } else if (text?.success) {
      const { side, attempt, text: piece } = text.data;
      this.ctx.gateway.publish(stream(row), { kind: "text", side, attempt, text: piece });
    }
  }

  /** A finished (or failed: `result` null) run: stored, streamed, and a showcase told to its world. */
  private async finish(runId: string, result: unknown): Promise<void> {
    const row = await this.row(runId);
    if (row?.status !== "running") return;
    if (this.playing.get(row.userId) === runId) this.playing.delete(row.userId);
    const parsed = result === null ? null : MinigameJobResult.safeParse(result);
    const ok: Result | null = parsed?.success ? parsed.data : null;
    if (parsed && !parsed.success) {
      this.ctx.log.error({ runId, issues: parsed.error.issues }, "bad minigame result");
    }
    await this.ctx.db
      .update(schema.minigameRuns)
      .set(
        ok
          ? {
              status: "done",
              result: ok,
              score: ok.sides.a.score,
              maxScore: ok.sides.a.maxScore,
              opponentScore: ok.sides.b?.score ?? null,
            }
          : { status: "failed" },
      )
      .where(eq(schema.minigameRuns.id, runId));
    this.ctx.gateway.publish(stream(row), ok ? { kind: "done", result: ok } : { kind: "failed" });
    if (row.showcaseId) await this.resolveShowcase(row, ok);
  }

  private async resolveShowcase(row: Row, ok: Result | null): Promise<void> {
    const actor = this.ctx.worlds.peek(row.worldId) ?? (await this.ctx.worlds.get(row.worldId));
    const showcaseId = row.showcaseId as string;
    const event = ok
      ? {
          type: "showcaseResolved" as const,
          showcaseId,
          score: ok.sides.a.score,
          maxScore: ok.sides.a.maxScore,
        }
      : { type: "showcaseResolved" as const, showcaseId, failed: true as const };
    const r = await actor.submit({ kind: "system", event });
    if (!r.ok) this.ctx.log.info({ showcaseId, reason: r.reason }, "showcase result not applied");
  }

  // -- the player's runs -----------------------------------------------------------------------

  private async runs(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "minigameRuns") throw new HttpError(400, "That request isn't valid.");
    const rows = await this.ctx.db
      .select()
      .from(schema.minigameRuns)
      .where(
        and(eq(schema.minigameRuns.worldId, q.worldId), eq(schema.minigameRuns.userId, userId)),
      )
      .orderBy(desc(schema.minigameRuns.createdAt));
    const bests: Record<string, number> = {};
    for (const r of rows) {
      if (r.status !== "done" || r.score === null) continue;
      const key = `${r.game}:${r.difficulty}`;
      bests[key] = Math.max(bests[key] ?? 0, r.score);
    }
    return {
      bests,
      runs: rows.slice(0, RUNS_SHOWN).map((r) => ({
        id: r.id,
        game: r.game,
        difficulty: r.difficulty,
        versionId: r.versionId,
        opponentName: r.opponentName,
        showcase: r.showcaseId !== null,
        status: r.status,
        score: r.score,
        maxScore: r.maxScore,
        opponentScore: r.opponentScore,
        createdAt: r.createdAt,
        result: r.result,
      })),
    };
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch((err) => this.ctx.log.error({ err }, "minigame service step failed"));
    return run;
  }
}

/** A rival's product on sale, with its version's pack key. */
function rivalOnSale(
  s: WorldState,
  productId: string,
): { company: CompanyState; productName: string; packKey: string } | null {
  for (const id of Object.keys(s.companies).sort()) {
    const c = s.companies[id] as CompanyState;
    const p = c.products[productId];
    const packKey = p ? c.versions[p.versionId]?.packKey : null;
    if (c.rival && p?.status === "live" && packKey)
      return { company: c, productName: p.name, packKey };
  }
  return null;
}
