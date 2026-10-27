import { randomInt, randomUUID } from "node:crypto";
import { GenerateJobResult, type Query } from "@airace/protocol";
import { and, asc, desc, eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import type { ChatMessage } from "../realtime/gateway.ts";
import { checkpointKey } from "../training/service.ts";

const HISTORY_TURNS = 12;
const RATE_WINDOW_MS = 10 * 60 * 1000;
const RATE_MAX_MESSAGES = 30;
const MODEL_URL_TTL_MS = 60 * 60 * 1000;
const RECENT_CIDS = 2000;
export const REPLY_FAILED = "The AI couldn't answer right now. Try again.";

type GeneratePayload = {
  worldId: string;
  threadId: string;
  versionId: string;
  history: ["user" | "ai", string][];
  seed: number;
};
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
    // Stage 1: only your own AIs. Stage 2 adds the AIs other companies share (Task 26).
    const version = companyId ? s.companies[companyId]?.versions[msg.versionId] : undefined;
    if (!version) return { ok: false, reason: "You can only talk to your own AIs for now." };

    let threadId = msg.threadId;
    if (threadId) {
      const thread = await this.ownThread(userId, msg.worldId, threadId);
      if (!thread) return { ok: false, reason: "That conversation isn't yours." };
      if (thread.versionId !== msg.versionId) {
        return { ok: false, reason: "That conversation is with another AI. Start a new one." };
      }
      if (this.inFlight.has(threadId))
        return { ok: false, reason: "Wait for the reply before sending another message." };
    }

    const now = ctx.opts.scheduler.now();
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
      });
    }
    const idx = await this.nextIdx(threadId);
    // The text goes to the model as plain text, `<|end|>` included (M1 Review Focus 1).
    await ctx.db
      .insert(schema.chatMessages)
      .values({ threadId, idx, role: "user", text: msg.text, at: now });
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
    };
    this.inFlight.set(threadId, "pending");
    const jobId = await ctx.broker.enqueue("generate", payload, msg.worldId);
    this.inFlight.set(threadId, jobId);
    this.remember(seenKey, threadId);
    return { ok: true, duplicate: false, data: { threadId } };
  }

  private async spec(p: GeneratePayload): Promise<unknown> {
    return {
      versionId: p.versionId,
      modelTarUrl: signBlobUrl(this.ctx, checkpointKey(p.versionId), "GET", MODEL_URL_TTL_MS),
      history: p.history,
      maxNewTokens: 96,
      temperature: 0.8,
      topP: 0.95,
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
      threads: rows.map((t) => ({ id: t.id, versionId: t.versionId, createdAt: t.createdAt })),
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
      })
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, thread.id))
      .orderBy(asc(schema.chatMessages.idx));
    return {
      threadId: thread.id,
      versionId: thread.versionId,
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
