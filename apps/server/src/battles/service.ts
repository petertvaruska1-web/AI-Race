import { createHash } from "node:crypto";
import { BattleJobResult, BattleTextEvent, type Query } from "@airace/protocol";
import {
  type BattleRound,
  type BattleState,
  type CompanyState,
  getBalance,
  type WorldState,
} from "@airace/sim";
import { eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import { trainingCards } from "../helpers/limits.ts";
import { forCompany, paramsOf } from "../jobs/owners.ts";
import type { BattleVoteMessage } from "../realtime/gateway.ts";
import type { TimerHandle } from "../scheduler.ts";
import { checkpointKey } from "../training/service.ts";
import type { WorldActor } from "../worlds/actor.ts";

const MODEL_URL_TTL_MS = 12 * 60 * 60 * 1000;
/** How long the other players may vote on a friends' battle's stories (real time). */
export const VOTE_MS = 300_000;
export const BATTLE_OFFLINE = "Its maker's computer is offline, so it can't battle right now.";
export const NOT_YOUR_VOTE = "You can't vote on your own battle.";
export const VOTE_CLOSED = "Voting on that battle is over.";

/** A battle job; `side` is set for one side of a battle between friends' AIs. */
type BattlePayload = { worldId: string; companyId: string; battleId: string; side?: "a" | "b" };
type Side = "mine" | "theirs";

/** An open vote on a friends' battle's story rounds, kept in memory (only totals go out). */
type Ballot = {
  p: BattlePayload;
  rounds: BattleRound[];
  /** The two players whose AIs battle: they can't vote. */
  parties: Set<string>;
  /** The other members online when the vote opened. */
  voters: Set<string>;
  story: number[];
  leftIsMine: Map<number, boolean>;
  picks: Map<number, Map<string, Side>>;
  timer: TimerHandle;
};

/**
 * Battles (game-layer design §10): a battle the world started becomes one `battle` job. Both
 * replies stream live on `battle:<world>:<battle>` to the battle's owner; the scored rounds come
 * back as `battleResolved`, and a job that finally fails calls the battle off (the sim refunds
 * the fee).
 *
 * A battle between two players' AIs (Stage 2) runs as two side jobs, each on its own owner's
 * card with only its own model, joined here. Its story rounds go to a vote of the world's other
 * online players for five minutes; a round's winner is the story with more votes (a tie or no
 * votes: the judge's scores).
 */
export class BattleService {
  private known = new Set<string>();
  private transcripts = new Map<string, { mine: string; theirs: string }[]>();
  private ballots = new Map<string, Ballot>();
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, worlds, gateway } = ctx;
    worlds.onLoad((actor) => this.attach(actor));
    broker.onPrepare("battle", (job) => this.spec(job.payload as BattlePayload));
    broker.onProgress("battle", (job, event) => this.progress(job.payload as BattlePayload, event));
    broker.onDone("battle", (job, result) => this.done(job.payload as BattlePayload, result));
    broker.onFailed("battle", async (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "battle failed; it is called off");
      const p = job.payload as BattlePayload;
      if (p.side) await this.cancelSides(p.battleId);
      return this.resolve(whole(p), { failed: true });
    });
    gateway.registerStream("battle", async (userId, sub) => {
      const [, worldId, battleId] = sub.split(":");
      return !!(worldId && battleId && (await this.own(userId, worldId, battleId)));
    });
    gateway.registerStream("battlevote", async (userId, sub) => {
      const s = await this.stateOf(sub.split(":")[1] ?? "");
      return !!s?.members[userId];
    });
    gateway.registerQuery("battle", (userId, q) => this.transcript(userId, q));
    gateway.setBattleVoteHandler(async (userId, msg) => this.vote(userId, msg));
    gateway.setCommandGuard("startBattle", async (userId, worldId, cmd) => {
      if (cmd.type !== "startBattle") return null;
      const s = await this.stateOf(worldId);
      return s ? this.defenderAway(s, userId, cmd.rivalProductId) : null;
    });
  }

  async close(): Promise<void> {
    this.closed = true;
    for (const b of this.ballots.values()) this.ctx.opts.scheduler.clearTimeout(b.timer);
    this.ballots.clear();
    await this.queue;
  }

  private async stateOf(worldId: string): Promise<WorldState | null> {
    try {
      const actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
      return actor.state();
    } catch {
      return null;
    }
  }

  /** With Fair mode off, a friend's AI battles on its maker's card, which must be on. */
  private defenderAway(s: WorldState, userId: string, productId: string): string | null {
    if (s.fairMode) return null;
    for (const c of Object.values(s.companies)) {
      if (!c.products[productId]) continue;
      if (c.rival || !c.ownerUserId || c.ownerUserId === userId) return null;
      const cards = trainingCards(this.ctx.broker.connections(), s, c.ownerUserId);
      return cards.length > 0 ? null : BATTLE_OFFLINE;
    }
    return null;
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
        if (battle.opponentKind === "player") await this.requestSides(s, companyId, battle);
        else await this.request(s, companyId, battle);
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
      const jobId = await broker.enqueue(
        "battle",
        payload,
        s.worldId,
        forCompany(
          s,
          companyId,
          Math.max(paramsOf(s, battle.mine.versionId), paramsOf(s, battle.theirs.versionId)),
        ),
      );
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

  /** A battle between two players' AIs: side a on the challenger's card, side b on the defender's. */
  private async requestSides(s: WorldState, companyId: string, battle: BattleState): Promise<void> {
    const { db, broker } = this.ctx;
    const base: BattlePayload = { worldId: s.worldId, companyId, battleId: battle.id };
    const fresh = await db
      .insert(schema.battleRequests)
      .values({ battleId: battle.id, worldId: s.worldId })
      .onConflictDoNothing()
      .returning();
    if (fresh.length > 0) {
      const defender = s.companies[battle.rivalId];
      if (defender?.ownerUserId && this.defenderAwayFor(s, defender.ownerUserId)) {
        this.ctx.gateway.publish(streamName(base), { kind: "calledOff", reason: BATTLE_OFFLINE });
        return this.resolve(base, { failed: true });
      }
      const jobId = await broker.enqueue(
        "battle",
        { ...base, side: "a" },
        s.worldId,
        forCompany(s, companyId, paramsOf(s, battle.mine.versionId)),
      );
      const sideBJobId = await broker.enqueue(
        "battle",
        { ...base, side: "b" },
        s.worldId,
        forCompany(s, battle.rivalId, paramsOf(s, battle.theirs.versionId)),
      );
      await db
        .update(schema.battleRequests)
        .set({ jobId, sideBJobId })
        .where(eq(schema.battleRequests.battleId, battle.id));
      return;
    }
    // After a restart: sides that ended while the world was not loaded are applied now.
    const [a, b] = await this.sideJobs(battle.id);
    if (a?.status === "done" && b?.status === "done") return this.join(base, a.result, b.result);
    if ([a?.status, b?.status].some((x) => x === "failed" || x === "cancelled")) {
      await this.cancelSides(battle.id);
      await this.resolve(base, { failed: true });
    }
  }

  private defenderAwayFor(s: WorldState, ownerUserId: string): boolean {
    return !s.fairMode && trainingCards(this.ctx.broker.connections(), s, ownerUserId).length === 0;
  }

  private async sideJobs(battleId: string) {
    const [row] = await this.ctx.db
      .select()
      .from(schema.battleRequests)
      .where(eq(schema.battleRequests.battleId, battleId));
    const get = (id: string | null | undefined) => (id ? this.ctx.broker.getJob(id) : null);
    return Promise.all([get(row?.jobId), get(row?.sideBJobId)]);
  }

  private async cancelSides(battleId: string): Promise<void> {
    for (const job of await this.sideJobs(battleId)) {
      if (job && (job.status === "queued" || job.status === "leased")) {
        await this.ctx.broker.cancel(job.id);
      }
    }
  }

  private async battleOf(p: BattlePayload): Promise<{ s: WorldState; battle: BattleState } | null> {
    const s = await this.stateOf(p.worldId);
    const battle = s?.companies[p.companyId]?.battles.find((b) => b.id === p.battleId);
    return s && battle ? { s, battle } : null;
  }

  private async spec(p: BattlePayload): Promise<unknown> {
    const found = await this.battleOf(p);
    if (!found) throw new Error(`battle ${p.battleId} is gone`);
    const { s, battle } = found;
    const base = {
      battleId: battle.id,
      skill: battle.skill,
      rounds: getBalance(s.balance).economy.battles.rounds,
      seed: battle.seed,
    };
    const model = async (versionId: string, companyId: string) => ({
      versionId,
      modelTarUrl: signBlobUrl(this.ctx, checkpointKey(versionId), "GET", MODEL_URL_TTL_MS),
      ownText: await this.ctx.coaching.ownText(s, companyId),
    });
    if (p.side) {
      // One side only, with only its own model (a friend's weights stay off your card).
      const side =
        p.side === "a"
          ? await model(battle.mine.versionId, p.companyId)
          : await model(battle.theirs.versionId, battle.rivalId);
      return { ...base, sides: [p.side], [p.side]: side };
    }
    return {
      ...base,
      a: await model(battle.mine.versionId, p.companyId),
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
    if (p.side && side !== p.side) return;
    const who: Side = side === "a" ? "mine" : "theirs";
    const rounds = this.transcripts.get(p.battleId) ?? [];
    while (rounds.length <= round) rounds.push({ mine: "", theirs: "" });
    const entry = rounds[round] as { mine: string; theirs: string };
    entry[who] += text;
    this.transcripts.set(p.battleId, rounds);
    this.ctx.gateway.publish(streamName(p), { kind: "text", round, side: who, text });
  }

  private async done(p: BattlePayload, result: unknown): Promise<void> {
    if (p.side) {
      // A side waits for the other; the second to finish joins them.
      const [a, b] = await this.sideJobs(p.battleId);
      if (a?.status !== "done" || b?.status !== "done") return;
      return this.join(whole(p), a.result, b.result);
    }
    const parsed = BattleJobResult.safeParse(result);
    const rounds = parsed.success
      ? parsed.data.rounds.map((r) =>
          r.a && r.b ? { prompt: r.prompt, expected: r.expected, mine: r.a, theirs: r.b } : null,
        )
      : [];
    if (!parsed.success || rounds.some((r) => r === null)) {
      this.ctx.log.error({ battleId: p.battleId }, "bad battle result");
      return this.resolve(p, { failed: true });
    }
    await this.resolve(p, { rounds: rounds as BattleRound[] });
  }

  /** Joins the two sides' rounds of a friends' battle, then puts its stories to a vote. */
  private async join(p: BattlePayload, resultA: unknown, resultB: unknown): Promise<void> {
    const a = BattleJobResult.safeParse(resultA);
    const b = BattleJobResult.safeParse(resultB);
    const rounds: BattleRound[] = [];
    if (a.success && b.success && a.data.rounds.length === b.data.rounds.length) {
      for (const [k, ra] of a.data.rounds.entries()) {
        const rb = b.data.rounds[k];
        if (!ra.a || !rb?.b || rb.prompt !== ra.prompt) break;
        rounds.push({ prompt: ra.prompt, expected: ra.expected, mine: ra.a, theirs: rb.b });
      }
    }
    if (rounds.length === 0 || !a.success || rounds.length !== a.data.rounds.length) {
      this.ctx.log.error({ battleId: p.battleId }, "the two sides of a battle don't match");
      return this.resolve(p, { failed: true });
    }
    await this.openVote(p, rounds);
  }

  private async openVote(p: BattlePayload, rounds: BattleRound[]): Promise<void> {
    if (this.ballots.has(p.battleId)) return;
    const found = await this.battleOf(p);
    if (found?.battle.status !== "running") return;
    const { s, battle } = found;
    const parties = new Set(
      [s.companies[p.companyId]?.ownerUserId, s.companies[battle.rivalId]?.ownerUserId].filter(
        (x): x is string => !!x,
      ),
    );
    const voters = new Set(
      Object.values(s.members)
        .filter((m) => m.online && !parties.has(m.userId))
        .map((m) => m.userId),
    );
    const story = rounds.flatMap((r, k) => (r.expected === null ? [k] : []));
    if (voters.size === 0 || story.length === 0) return this.resolve(p, { rounds });
    const leftIsMine = new Map(story.map((k) => [k, leftIsMineFor(p.battleId, k)]));
    const ballot: Ballot = {
      p,
      rounds,
      parties,
      voters,
      story,
      leftIsMine,
      picks: new Map(),
      timer: this.ctx.opts.scheduler.setTimeout(
        () => void this.serial(() => this.closeVote(p.battleId)),
        VOTE_MS,
      ),
    };
    this.ballots.set(p.battleId, ballot);
    // Unlabelled: no company, player, score or side, in a random (but fixed) order.
    this.ctx.gateway.publish(voteStream(p.worldId), {
      kind: "open",
      battleId: p.battleId,
      closesInMs: VOTE_MS,
      rounds: story.map((k) => {
        const r = rounds[k] as BattleRound;
        const [left, right] = leftIsMine.get(k)
          ? [r.mine.reply, r.theirs.reply]
          : [r.theirs.reply, r.mine.reply];
        return { round: k, prompt: r.prompt, left, right };
      }),
    });
  }

  private async vote(
    userId: string,
    msg: BattleVoteMessage,
  ): Promise<{ ok: true } | { ok: false; reason: string }> {
    const s = await this.stateOf(msg.worldId);
    if (!s?.members[userId]) {
      return { ok: false, reason: "That world doesn't exist or you're not in it." };
    }
    const ballot = this.ballots.get(msg.battleId);
    if (!ballot || ballot.p.worldId !== msg.worldId) return { ok: false, reason: VOTE_CLOSED };
    if (ballot.parties.has(userId)) return { ok: false, reason: NOT_YOUR_VOTE };
    if (!ballot.voters.has(userId)) return { ok: false, reason: VOTE_CLOSED };
    if (!ballot.story.includes(msg.round)) {
      return { ok: false, reason: "That round isn't a story." };
    }
    const left = ballot.leftIsMine.get(msg.round) ? "mine" : "theirs";
    const pick: Side = msg.pick === "left" ? left : left === "mine" ? "theirs" : "mine";
    const picks = ballot.picks.get(msg.round) ?? new Map<string, Side>();
    picks.set(userId, pick);
    ballot.picks.set(msg.round, picks);
    const everyone = ballot.story.every(
      (k) => (ballot.picks.get(k)?.size ?? 0) >= ballot.voters.size,
    );
    if (everyone) void this.serial(() => this.closeVote(msg.battleId));
    return { ok: true };
  }

  private async closeVote(battleId: string): Promise<void> {
    const ballot = this.ballots.get(battleId);
    if (!ballot) return;
    this.ballots.delete(battleId);
    this.ctx.opts.scheduler.clearTimeout(ballot.timer);
    const totals = (k: number) => {
      const all = [...(ballot.picks.get(k)?.values() ?? [])];
      return {
        mine: all.filter((x) => x === "mine").length,
        theirs: all.filter((x) => x === "theirs").length,
      };
    };
    const rounds = ballot.rounds.map((r, k) => {
      const t = totals(k);
      return ballot.story.includes(k) && t.mine + t.theirs > 0 ? { ...r, votes: t } : r;
    });
    this.ctx.gateway.publish(voteStream(ballot.p.worldId), {
      kind: "closed",
      battleId,
      rounds: ballot.story.map((k) => {
        const t = totals(k);
        const [left, right] = ballot.leftIsMine.get(k) ? [t.mine, t.theirs] : [t.theirs, t.mine];
        return { round: k, left, right };
      }),
    });
    await this.resolve(ballot.p, { rounds });
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
    const s = await this.stateOf(worldId);
    const companyId = s?.members[userId]?.companyId;
    return !!companyId && !!s?.companies[companyId]?.battles.some((b) => b.id === battleId);
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

/** The battle a side job belongs to, without its side. */
function whole(p: BattlePayload): BattlePayload {
  return { worldId: p.worldId, companyId: p.companyId, battleId: p.battleId };
}

/** Which story shows on the left, fixed per battle and round (seeded by the battle id). */
function leftIsMineFor(battleId: string, round: number): boolean {
  return (createHash("sha256").update(`${battleId}:${round}`).digest()[0] ?? 0) % 2 === 0;
}

function streamName(p: { worldId: string; battleId: string }): string {
  return `battle:${p.worldId}:${p.battleId}`;
}

function voteStream(worldId: string): string {
  return `battlevote:${worldId}`;
}
