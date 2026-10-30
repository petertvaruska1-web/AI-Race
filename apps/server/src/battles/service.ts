import { BattleJobResult, BattleTextEvent, type Query } from "@airace/protocol";
import { type BattleState, type CompanyState, getBalance, type WorldState } from "@airace/sim";
import { eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import { checkpointKey } from "../training/service.ts";
import type { WorldActor } from "../worlds/actor.ts";

const MODEL_URL_TTL_MS = 12 * 60 * 60 * 1000;

type BattlePayload = { worldId: string; companyId: string; battleId: string };
type Side = "mine" | "theirs";

/**
 * Battles (game-layer design §10): a battle the world started becomes one
 * `battle` job. Both replies stream live on `battle:<world>:<battle>` to the
 * battle's owner; the scored rounds come back as `battleResolved`, and a job
 * that finally fails calls the battle off (the sim refunds the fee).
 */
export class BattleService {
  private known = new Set<string>();
  private transcripts = new Map<string, { mine: string; theirs: string }[]>();
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, worlds, gateway } = ctx;
    worlds.onLoad((actor) => this.attach(actor));
    broker.onPrepare("battle", (job) => this.spec(job.payload as BattlePayload));
    broker.onProgress("battle", (job, event) => this.progress(job.payload as BattlePayload, event));
    broker.onDone("battle", (job, result) => this.done(job.payload as BattlePayload, result));
    broker.onFailed("battle", (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "battle failed; it is called off");
      return this.resolve(job.payload as BattlePayload, { failed: true });
    });
    gateway.registerStream("battle", async (userId, sub) => {
      const [, worldId, battleId] = sub.split(":");
      return !!(worldId && battleId && (await this.own(userId, worldId, battleId)));
    });
    gateway.registerQuery("battle", (userId, q) => this.transcript(userId, q));
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.queue;
  }

  private attach(actor: WorldActor): void {
    actor.onChange(() => void this.serial(() => this.sync(actor)));
    void this.serial(() => this.sync(actor));
  }

  /** Every running battle gets its job once; one that finished while its world was away is applied. */
  private async sync(actor: WorldActor): Promise<void> {
    if (this.closed || actor.isClosed()) return;
    const s = actor.state();
    for (const companyId of Object.keys(s.companies).sort()) {
      const company = s.companies[companyId] as CompanyState;
      for (const battle of company.battles) {
        if (battle.status !== "running" || this.known.has(battle.id)) continue;
        this.known.add(battle.id);
        await this.request(s, companyId, battle);
      }
    }
  }

  private async request(s: WorldState, companyId: string, battle: BattleState): Promise<void> {
    const { db, broker } = this.ctx;
    const payload: BattlePayload = { worldId: s.worldId, companyId, battleId: battle.id };
    const fresh = await db
      .insert(schema.battleRequests)
      .values({ battleId: battle.id, worldId: s.worldId })
      .onConflictDoNothing()
      .returning();
    if (fresh.length > 0) {
      const jobId = await broker.enqueue("battle", payload, s.worldId);
      await db
        .update(schema.battleRequests)
        .set({ jobId })
        .where(eq(schema.battleRequests.battleId, battle.id));
      return;
    }
    // After a restart: a job that ended while its world was not loaded is applied now.
    const [row] = await db
      .select()
      .from(schema.battleRequests)
      .where(eq(schema.battleRequests.battleId, battle.id));
    const job = row?.jobId ? await broker.getJob(row.jobId) : null;
    if (job?.status === "done") await this.done(payload, job.result);
    if (job?.status === "failed" || job?.status === "cancelled") {
      await this.resolve(payload, { failed: true });
    }
  }

  private async battleOf(p: BattlePayload): Promise<{ s: WorldState; battle: BattleState } | null> {
    const actor = this.ctx.worlds.peek(p.worldId) ?? (await this.ctx.worlds.get(p.worldId));
    const s = actor.state();
    const battle = s.companies[p.companyId]?.battles.find((b) => b.id === p.battleId);
    return battle ? { s, battle } : null;
  }

  private async spec(p: BattlePayload): Promise<unknown> {
    const found = await this.battleOf(p);
    if (!found) throw new Error(`battle ${p.battleId} is gone`);
    const { s, battle } = found;
    return {
      battleId: battle.id,
      skill: battle.skill,
      rounds: getBalance(s.balance).economy.battles.rounds,
      seed: battle.seed,
      a: {
        versionId: battle.mine.versionId,
        modelTarUrl: signBlobUrl(
          this.ctx,
          checkpointKey(battle.mine.versionId),
          "GET",
          MODEL_URL_TTL_MS,
        ),
      },
      b: {
        versionId: battle.theirs.packKey,
        modelTarUrl: await this.ctx.rivals.modelUrl(battle.theirs.packKey),
      },
    };
  }

  private progress(p: BattlePayload, event: unknown): void {
    const parsed = BattleTextEvent.safeParse(event);
    if (!parsed.success) return;
    const { round, side, text } = parsed.data;
    const who: Side = side === "a" ? "mine" : "theirs";
    const rounds = this.transcripts.get(p.battleId) ?? [];
    while (rounds.length <= round) rounds.push({ mine: "", theirs: "" });
    const entry = rounds[round] as { mine: string; theirs: string };
    entry[who] += text;
    this.transcripts.set(p.battleId, rounds);
    this.ctx.gateway.publish(streamName(p), { kind: "text", round, side: who, text });
  }

  private async done(p: BattlePayload, result: unknown): Promise<void> {
    const parsed = BattleJobResult.safeParse(result);
    if (!parsed.success) {
      this.ctx.log.error(
        { battleId: p.battleId, issues: parsed.error.issues },
        "bad battle result",
      );
      return this.resolve(p, { failed: true });
    }
    const rounds = parsed.data.rounds.map((r) => ({ prompt: r.prompt, mine: r.a, theirs: r.b }));
    await this.resolve(p, { rounds });
  }

  private async resolve(
    p: BattlePayload,
    outcome: { failed: true } | { rounds: BattleState["rounds"] },
  ): Promise<void> {
    const actor = this.ctx.worlds.peek(p.worldId) ?? (await this.ctx.worlds.get(p.worldId));
    const event =
      "failed" in outcome
        ? { type: "battleResolved" as const, battleId: p.battleId, failed: true as const }
        : { type: "battleResolved" as const, battleId: p.battleId, rounds: outcome.rounds ?? [] };
    const r = await actor.submit({ kind: "system", event });
    if (!r.ok) {
      this.ctx.log.info({ battleId: p.battleId, reason: r.reason }, "battle result not applied");
    }
    this.transcripts.delete(p.battleId);
    this.ctx.gateway.publish(streamName(p), { kind: "failed" in outcome ? "calledOff" : "done" });
  }

  private async own(userId: string, worldId: string, battleId: string): Promise<boolean> {
    let actor: WorldActor;
    try {
      actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
    } catch {
      return false;
    }
    const s = actor.state();
    const companyId = s.members[userId]?.companyId;
    return !!companyId && !!s.companies[companyId]?.battles.some((b) => b.id === battleId);
  }

  private async transcript(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "battle") throw new HttpError(400, "That request isn't valid.");
    if (!(await this.own(userId, q.worldId, q.battleId))) {
      throw new HttpError(404, "That battle isn't yours.");
    }
    return { rounds: this.transcripts.get(q.battleId) ?? [] };
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch((err) => this.ctx.log.error({ err }, "battle service step failed"));
    return run;
  }
}

function streamName(p: { worldId: string; battleId: string }): string {
  return `battle:${p.worldId}:${p.battleId}`;
}
