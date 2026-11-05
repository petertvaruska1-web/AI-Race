import type { Query } from "@airace/protocol";
import type { WorldState } from "@airace/sim";
import { and, asc, desc, eq, lt } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { schema } from "../db/index.ts";
import { textProblem } from "../notebook/filter.ts";
import type { WorldChatMessage } from "../realtime/gateway.ts";

export const KEPT_MESSAGES = 200;
export const MAX_CHARS = 300;
export const SLOW_DOWN = "Slow down a little.";
const RATE_WINDOW_MS = 10_000;
const RATE_MAX = 5;
const NOT_IN_WORLD = "That world doesn't exist or you're not in it.";

/** A world chat line as everyone in the world sees it. */
export type WorldChatLine = { name: string; companyId: string | null; text: string; at: number };

/**
 * The world chat (Stage 2): the players of a world talk to each other. Members only; each
 * message is pushed on `worldchat:<worldId>` and the last 200 are kept. The filter only blocks
 * (links, phone numbers, the AI's own characters, blocked words); it never changes a word.
 */
export class WorldChatService {
  private rate = new Map<string, number[]>();

  constructor(private readonly ctx: AppContext) {
    ctx.gateway.setWorldChatHandler((userId, msg) => this.post(userId, msg));
    ctx.gateway.registerQuery("worldChat", (userId, q) => this.history(userId, q));
    ctx.gateway.registerStream("worldchat", async (userId, sub) => {
      const s = await this.stateOf(sub.split(":")[1] ?? "");
      return !!s?.members[userId];
    });
  }

  private async stateOf(worldId: string): Promise<WorldState | null> {
    try {
      const actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
      return actor.state();
    } catch {
      return null;
    }
  }

  private async post(
    userId: string,
    msg: WorldChatMessage,
  ): Promise<{ ok: true } | { ok: false; reason: string }> {
    const s = await this.stateOf(msg.worldId);
    if (!s?.members[userId]) return { ok: false, reason: NOT_IN_WORLD };
    const text = msg.text.trim();
    if (text.length === 0 || text.length > MAX_CHARS) {
      return { ok: false, reason: `Messages are 1–${MAX_CHARS} characters.` };
    }
    const problem = textProblem(text);
    if (problem) return { ok: false, reason: problem };
    const now = this.ctx.opts.scheduler.now();
    const key = `${msg.worldId}|${userId}`;
    const recent = (this.rate.get(key) ?? []).filter((t) => t > now - RATE_WINDOW_MS);
    if (recent.length >= RATE_MAX) {
      this.rate.set(key, recent);
      return { ok: false, reason: SLOW_DOWN };
    }
    recent.push(now);
    this.rate.set(key, recent);
    await this.add(msg.worldId, userId, text);
    return { ok: true };
  }

  /** Stores and pushes a message that passed the checks, keeping the world's last 200. */
  async add(worldId: string, userId: string, text: string): Promise<void> {
    const s = await this.stateOf(worldId);
    const member = s?.members[userId];
    if (!member) return;
    const at = this.ctx.opts.scheduler.now();
    const { db } = this.ctx;
    await db
      .insert(schema.worldChat)
      .values({ worldId, userId, name: member.displayName, text, at });
    const [oldest] = await db
      .select({ id: schema.worldChat.id })
      .from(schema.worldChat)
      .where(eq(schema.worldChat.worldId, worldId))
      .orderBy(desc(schema.worldChat.id))
      .offset(KEPT_MESSAGES - 1)
      .limit(1);
    if (oldest) {
      await db
        .delete(schema.worldChat)
        .where(and(eq(schema.worldChat.worldId, worldId), lt(schema.worldChat.id, oldest.id)));
    }
    const line: WorldChatLine = { name: member.displayName, companyId: member.companyId, text, at };
    this.ctx.gateway.publish(`worldchat:${worldId}`, line);
  }

  private async history(userId: string, q: Query): Promise<{ messages: WorldChatLine[] }> {
    if (q.kind !== "worldChat") throw new HttpError(400, "Unknown request.");
    const s = await this.stateOf(q.worldId);
    if (!s?.members[userId]) throw new HttpError(403, NOT_IN_WORLD);
    const rows = await this.ctx.db
      .select()
      .from(schema.worldChat)
      .where(eq(schema.worldChat.worldId, q.worldId))
      .orderBy(asc(schema.worldChat.id));
    return {
      messages: rows.map((r) => ({
        name: r.name,
        companyId: s.members[r.userId]?.companyId ?? null,
        text: r.text,
        at: r.at,
      })),
    };
  }
}
