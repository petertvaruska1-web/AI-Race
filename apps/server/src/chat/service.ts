import { randomInt, randomUUID } from "node:crypto";
import { GenerateJobResult, type Query } from "@airace/protocol";
import type { WorldState } from "@airace/sim";
import { and, asc, desc, eq, gte, inArray, ne } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import { trainingCards } from "../helpers/limits.ts";
import { forPlayer, paramsOf } from "../jobs/owners.ts";
import type { ChatMessage } from "../realtime/gateway.ts";
import { checkpointKey } from "../training/service.ts";

const HISTORY_TURNS = 12;
const RATE_WINDOW_MS = 10 * 60 * 1000;
const RATE_MAX_MESSAGES = 30;
const MODEL_URL_TTL_MS = 60 * 60 * 1000;
const RECENT_CIDS = 2000;
/** Tries of one friend's AI: 10 messages in 10 minutes per asker (Stage 2). */
const TRY_WINDOW_MS = 10 * 60 * 1000;
const TRY_MAX_MESSAGES = 10;
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
export const REPLY_FAILED = "The AI couldn't answer right now. Try again.";

type GeneratePayload = {
  worldId: string;
  threadId: string;
  versionId: string;
  history: ["user" | "ai", string][];
  seed: number;
  /** Set when the AI is a rival's: its model comes from the rival pack. */
  packKey?: string;
};

/** Rival chats wait behind everything the player is doing (game-layer design §12). */
const RIVAL_CHAT_PRIORITY = 5;
/** An interview runs right after ordinary chats: the press is waiting. */
const INTERVIEW_PRIORITY = 1;
export const NOT_YOURS =
  "You can only talk to your own AIs, rivals' AIs on sale and AIs your friends share.";
export const NOT_SHARED = "That AI isn't shared.";
export const TRIED_A_LOT = "You've tried this AI a lot. Take a short break.";
export const MAKER_OFFLINE = "Its maker's computer is offline, so it can't answer right now.";
type ChatResult =
  | { ok: true; duplicate: boolean; data: { threadId: string } }
  | { ok: false; reason: string };

/**
 * The playground (plan Task 15): a message to one of the player's AIs becomes a
 * generate job; its text streams to the asker on `chat:<worldId>:<threadId>`
 * and is stored with the conversation. The model's words are shown as they
 * come, never rewritten (R1).
 */
export class ChatService {
  /** Threads waiting for a reply, with the job that will give it. */
  private inFlight = new Map<string, string>();
  private rate = new Map<string, number[]>();
  /** Recent tries of a friend's AI, by `<asker>|<versionId>`. */
  private tries = new Map<string, number[]>();
  /** Recently handled chat ids, so a resent message is acked once. */
  private seen = new Map<string, string>();

  constructor(private readonly ctx: AppContext) {
    const { broker, gateway } = ctx;
    gateway.setChatHandler((userId, msg) => this.send(userId, msg));
    gateway.registerStream("chat", async (userId, sub) => {
      const [, worldId, threadId] = sub.split(":");
      return !!(worldId && threadId && (await this.ownThread(userId, worldId, threadId)));
    });
    gateway.registerQuery("chatThreads", (userId, q) => this.threads(userId, q));
    gateway.registerQuery("chatThread", (userId, q) => this.thread(userId, q));
    gateway.registerQuery("versionTries", (userId, q) => this.versionTries(userId, q));
    broker.onPrepare("generate", (job) => this.spec(job.payload as GeneratePayload));
    broker.onChunk((job, text) => {
      if (job.type !== "generate") return;
      const p = job.payload as GeneratePayload;
      gateway.publish(stream(p), { kind: "chunk", text });
    });
    broker.onDone("generate", (job, result) => this.done(job.payload as GeneratePayload, result));
    broker.onFailed("generate", (job) => this.failed(job.payload as GeneratePayload));
  }

  private async send(userId: string, msg: ChatMessage): Promise<ChatResult> {
    const { ctx } = this;
    const seenKey = `${userId}|${msg.cid}`;
    const earlier = this.seen.get(seenKey);
    if (earlier) return { ok: true, duplicate: true, data: { threadId: earlier } };

    let actor: Awaited<ReturnType<typeof ctx.worlds.get>>;
    try {
      actor = await ctx.worlds.get(msg.worldId);
    } catch {
      return { ok: false, reason: "That world doesn't exist or you're not in it." };
    }
    const s = actor.state();
    const companyId = s.members[userId]?.companyId;
    if (!s.members[userId])
      return { ok: false, reason: "That world doesn't exist or you're not in it." };
    // Your own AIs, a rival's AI while it is on sale, or an AI another player shares (Stage 2).
    const version = companyId ? s.companies[companyId]?.versions[msg.versionId] : undefined;
    const packKey = version ? undefined : rivalOnSale(s, msg.versionId);
    const friend = version || packKey ? null : friendsVersion(s, msg.versionId);
    if (!version && !packKey && !friend) return { ok: false, reason: NOT_YOURS };
    if (friend && !friend.published) return { ok: false, reason: NOT_SHARED };
    const conns = ctx.broker.connections();
    if (friend && !s.fairMode && trainingCards(conns, s, friend.maker).length === 0) {
      return { ok: false, reason: MAKER_OFFLINE };
    }
    // An interview is with your own AI on sale, under that product's name.
    const interview = msg.interview ?? null;
    if (interview) {
      const product = companyId ? s.companies[companyId]?.products[interview.productId] : undefined;
      if (!product || product.versionId !== msg.versionId) {
        return { ok: false, reason: "That AI isn't one of yours." };
      }
      if (product.status !== "live") return { ok: false, reason: "That AI isn't on sale." };
    }

    let threadId = msg.threadId;
    if (threadId) {
      const thread = await this.ownThread(userId, msg.worldId, threadId);
      if (!thread) return { ok: false, reason: "That conversation isn't yours." };
      if (thread.versionId !== msg.versionId) {
        return { ok: false, reason: "That conversation is with another AI. Start a new one." };
      }
      const interviewing = thread.kind === "interview";
      if (
        interviewing !== (interview !== null) ||
        (interview && thread.productId !== interview.productId)
      ) {
        return { ok: false, reason: "Start a new conversation for that." };
      }
      if (this.inFlight.has(threadId))
        return { ok: false, reason: "Wait for the reply before sending another message." };
    }

    const now = ctx.opts.scheduler.now();
    if (friend) {
      const key = `${userId}|${msg.versionId}`;
      const tried = (this.tries.get(key) ?? []).filter((t) => t > now - TRY_WINDOW_MS);
      if (tried.length >= TRY_MAX_MESSAGES) {
        this.tries.set(key, tried);
        return { ok: false, reason: TRIED_A_LOT };
      }
      tried.push(now);
      this.tries.set(key, tried);
    }
    const recent = (this.rate.get(userId) ?? []).filter((t) => t > now - RATE_WINDOW_MS);
    if (recent.length >= RATE_MAX_MESSAGES) {
      this.rate.set(userId, recent);
      return {
        ok: false,
        reason: "You've sent a lot of messages. Take a short break and try again.",
      };
    }
    recent.push(now);
    this.rate.set(userId, recent);

    if (!threadId) {
      threadId = randomUUID();
      await ctx.db.insert(schema.chatThreads).values({
        id: threadId,
        worldId: msg.worldId,
        userId,
        versionId: msg.versionId,
        createdAt: now,
        kind: interview ? "interview" : "chat",
        productId: interview?.productId ?? null,
      });
    }
    const idx = await this.nextIdx(threadId);
    // The text goes to the model as plain text, `<|end|>` included (M1 Review Focus 1).
    await ctx.db.insert(schema.chatMessages).values({
      threadId,
      idx,
      role: "user",
      text: msg.text,
      at: now,
      cardId: interview?.cardId ?? null,
    });
    const turns = await ctx.db
      .select({ role: schema.chatMessages.role, text: schema.chatMessages.text })
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId))
      .orderBy(desc(schema.chatMessages.idx))
      .limit(HISTORY_TURNS);
    const history = turns
      .reverse()
      .map((t): ["user" | "ai", string] => [t.role === "ai" ? "ai" : "user", t.text]);
    const payload: GeneratePayload = {
      worldId: msg.worldId,
      threadId,
      versionId: msg.versionId,
      history,
      seed: randomInt(2 ** 31),
      ...(packKey ? { packKey } : {}),
    };
    this.inFlight.set(threadId, "pending");
    const jobId = await ctx.broker.enqueue("generate", payload, msg.worldId, {
      ...forPlayer(s, userId, paramsOf(s, msg.versionId)),
      // A friend's AI answers on its maker's card (Fair off); the asker's share pays (Fair on).
      ...(friend ? { owner: friend.maker } : {}),
      ...(packKey
        ? { priority: RIVAL_CHAT_PRIORITY }
        : interview
          ? { priority: INTERVIEW_PRIORITY }
          : {}),
    });
    this.inFlight.set(threadId, jobId);
    this.remember(seenKey, threadId);
    return { ok: true, duplicate: false, data: { threadId } };
  }

  private async spec(p: GeneratePayload): Promise<unknown> {
    return {
      versionId: p.packKey ?? p.versionId,
      modelTarUrl: p.packKey
        ? await this.ctx.rivals.modelUrl(p.packKey)
        : signBlobUrl(this.ctx, checkpointKey(p.versionId), "GET", MODEL_URL_TTL_MS),
      history: p.history,
      maxNewTokens: 96,
      // How an AI talks everywhere (ml: airace_ml/infer/serving.py; ADR 0009): calmer than the
      // old 0.8 / 0.95, which made the same AIs ramble.
      temperature: 0.6,
      topP: 0.9,
      seed: p.seed,
    };
  }

  private async done(p: GeneratePayload, result: unknown): Promise<void> {
    const parsed = GenerateJobResult.safeParse(result);
    if (!parsed.success) return this.failed(p);
    const { text } = parsed.data;
    await this.ctx.db.insert(schema.chatMessages).values({
      threadId: p.threadId,
      idx: await this.nextIdx(p.threadId),
      role: "ai",
      text,
      at: this.ctx.opts.scheduler.now(),
    });
    this.inFlight.delete(p.threadId);
    this.ctx.gateway.publish(stream(p), { kind: "done", text });
  }

  private async failed(p: GeneratePayload): Promise<void> {
    this.inFlight.delete(p.threadId);
    this.ctx.gateway.publish(stream(p), { kind: "failed", reason: REPLY_FAILED });
  }

  private async threads(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "chatThreads") throw new HttpError(400, "Unknown request.");
    const rows = await this.ctx.db
      .select()
      .from(schema.chatThreads)
      .where(and(eq(schema.chatThreads.worldId, q.worldId), eq(schema.chatThreads.userId, userId)))
      .orderBy(asc(schema.chatThreads.createdAt));
    return {
      threads: rows.map((t) => ({
        id: t.id,
        versionId: t.versionId,
        createdAt: t.createdAt,
        kind: t.kind,
        productId: t.productId,
      })),
    };
  }

  /** How often other players tried each of your shared AIs in the last 7 days: counts only (R2). */
  private async versionTries(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "versionTries") throw new HttpError(400, "Unknown request.");
    let s: WorldState;
    try {
      s = (await this.ctx.worlds.get(q.worldId)).state();
    } catch {
      throw new HttpError(403, "That world doesn't exist or you're not in it.");
    }
    if (!s.members[userId])
      throw new HttpError(403, "That world doesn't exist or you're not in it.");
    const companyId = s.members[userId]?.companyId;
    const company = companyId ? s.companies[companyId] : undefined;
    const shared = Object.values(company?.versions ?? {})
      .filter((v) => v.published)
      .sort((a, b) => a.createdDay - b.createdDay || (a.id < b.id ? -1 : 1));
    if (shared.length === 0) return { versions: [] };
    const since = this.ctx.opts.scheduler.now() - WEEK_MS;
    const rows = await this.ctx.db
      .select({ versionId: schema.chatThreads.versionId })
      .from(schema.chatMessages)
      .innerJoin(schema.chatThreads, eq(schema.chatThreads.id, schema.chatMessages.threadId))
      .where(
        and(
          eq(schema.chatThreads.worldId, q.worldId),
          ne(schema.chatThreads.userId, userId),
          inArray(
            schema.chatThreads.versionId,
            shared.map((v) => v.id),
          ),
          eq(schema.chatMessages.role, "user"),
          gte(schema.chatMessages.at, since),
        ),
      );
    return {
      versions: shared.map((v) => ({
        versionId: v.id,
        tries7d: rows.filter((r) => r.versionId === v.id).length,
      })),
    };
  }

  private async thread(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "chatThread") throw new HttpError(400, "Unknown request.");
    const thread = await this.ownThread(userId, q.worldId, q.threadId);
    if (!thread) throw new HttpError(403, "That conversation isn't yours.");
    const messages = await this.ctx.db
      .select({
        role: schema.chatMessages.role,
        text: schema.chatMessages.text,
        at: schema.chatMessages.at,
        cardId: schema.chatMessages.cardId,
      })
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, thread.id))
      .orderBy(asc(schema.chatMessages.idx));
    return {
      threadId: thread.id,
      versionId: thread.versionId,
      kind: thread.kind,
      productId: thread.productId,
      waiting: this.inFlight.has(thread.id),
      messages,
    };
  }

  private async ownThread(userId: string, worldId: string, threadId: string) {
    try {
      const [thread] = await this.ctx.db
        .select()
        .from(schema.chatThreads)
        .where(eq(schema.chatThreads.id, threadId));
      return thread && thread.userId === userId && thread.worldId === worldId ? thread : null;
    } catch {
      return null; // not a valid id
    }
  }

  private async nextIdx(threadId: string): Promise<number> {
    const [last] = await this.ctx.db
      .select({ idx: schema.chatMessages.idx })
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId))
      .orderBy(desc(schema.chatMessages.idx))
      .limit(1);
    return last ? last.idx + 1 : 0;
  }

  private remember(key: string, threadId: string): void {
    this.seen.set(key, threadId);
    if (this.seen.size > RECENT_CIDS) {
      const oldest = this.seen.keys().next().value;
      if (oldest !== undefined) this.seen.delete(oldest);
    }
  }
}

function stream(p: { worldId: string; threadId: string }): string {
  return `chat:${p.worldId}:${p.threadId}`;
}

/** Another player's AI `versionId`: who made it and whether it is shared. */
function friendsVersion(
  s: WorldState,
  versionId: string,
): { maker: string; published: boolean } | null {
  for (const c of Object.values(s.companies)) {
    const v = c.versions[versionId];
    if (v && !c.rival && c.ownerUserId) return { maker: c.ownerUserId, published: v.published };
  }
  return null;
}

/** The pack key of a rival's AI that is on sale under `versionId`, else undefined. */
function rivalOnSale(s: WorldState, versionId: string): string | undefined {
  for (const c of Object.values(s.companies)) {
    if (!c.rival) continue;
    const onSale = Object.values(c.products).some(
      (p) => p.status === "live" && p.versionId === versionId,
    );
    const key = c.versions[versionId]?.packKey;
    if (onSale && key) return key;
  }
  return undefined;
}
