import type { Query } from "@airace/protocol";
import { LESSONS_MAX, type WorldState } from "@airace/sim";
import { and, asc, desc, eq, inArray, lt } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { schema } from "../db/index.ts";
import { textProblem } from "../notebook/filter.ts";
import type { LessonDeleteMessage, LessonMessage } from "../realtime/gateway.ts";

export const NOT_OWN_AI = "Coach your own AI's replies.";
export const PICK_A_REPLY = "Pick a reply from this conversation.";
export const ANSWER_LENGTH = "Write an answer of 1–500 characters.";
const NOT_IN_WORLD = "That world doesn't exist or you're not in it.";
const MAX_ANSWER = 500;

type Result = { ok: true } | { ok: false; reason: string };
export type LessonView = {
  id: number;
  kind: "liked" | "written";
  question: string;
  answer: string;
  at: number;
};

/**
 * Coaching (C45): a player likes one of their own AI's replies in the Playground, or writes the
 * answer they wished for. Each becomes a lesson `{question, answer}` kept on the server (R2), at
 * most 200 per company (the newest replace the oldest); the world only counts them. A plan with
 * "Train on your lessons" on gets them as coaching pairs.
 */
export class CoachingService {
  constructor(private readonly ctx: AppContext) {
    ctx.gateway.setLessonHandler((userId, msg) => this.lesson(userId, msg));
    ctx.gateway.setLessonDeleteHandler((userId, msg) => this.remove(userId, msg));
    ctx.gateway.registerQuery("lessons", (userId, q) => this.list(userId, q));
  }

  private async stateOf(worldId: string): Promise<WorldState | null> {
    try {
      const actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
      return actor.state();
    } catch {
      return null;
    }
  }

  private async lesson(userId: string, msg: LessonMessage): Promise<Result> {
    const { db } = this.ctx;
    const s = await this.stateOf(msg.worldId);
    if (!s?.members[userId]) return { ok: false, reason: NOT_IN_WORLD };
    const companyId = s.members[userId]?.companyId;
    const company = companyId ? s.companies[companyId] : undefined;
    let thread: typeof schema.chatThreads.$inferSelect | undefined;
    try {
      [thread] = await db
        .select()
        .from(schema.chatThreads)
        .where(eq(schema.chatThreads.id, msg.threadId));
    } catch {
      thread = undefined;
    }
    // Your own conversation with one of your own AIs (not a rival's or a friend's).
    if (
      !company ||
      !thread ||
      thread.userId !== userId ||
      thread.worldId !== msg.worldId ||
      !company.versions[thread.versionId]
    ) {
      return { ok: false, reason: NOT_OWN_AI };
    }
    const turns = await db
      .select()
      .from(schema.chatMessages)
      .where(
        and(
          eq(schema.chatMessages.threadId, thread.id),
          inArray(schema.chatMessages.idx, [msg.idx - 1, msg.idx]),
        ),
      );
    const reply = turns.find((m) => m.idx === msg.idx);
    const question = turns.find((m) => m.idx === msg.idx - 1);
    if (reply?.role !== "ai" || question?.role !== "user") {
      return { ok: false, reason: PICK_A_REPLY };
    }
    let answer = reply.text;
    if (msg.kind === "written") {
      const text = (msg.text ?? "").trim();
      if (text.length === 0 || text.length > MAX_ANSWER)
        return { ok: false, reason: ANSWER_LENGTH };
      const problem = textProblem(text, { many: "Answers", one: "That answer" });
      if (problem) return { ok: false, reason: problem };
      answer = text;
    }
    const row = {
      worldId: msg.worldId,
      companyId: company.id,
      question: question.text,
      answer,
      kind: msg.kind,
      sourceThread: thread.id,
      sourceIdx: msg.idx,
      at: this.ctx.opts.scheduler.now(),
    };
    // A like twice is one lesson; a new written answer to the same reply replaces the last.
    await db
      .insert(schema.lessons)
      .values(row)
      .onConflictDoUpdate({
        target: [schema.lessons.sourceThread, schema.lessons.sourceIdx, schema.lessons.kind],
        set: { answer: row.answer, at: row.at },
      });
    await this.keepNewest(msg.worldId, company.id);
    await this.recount(msg.worldId, company.id);
    return { ok: true };
  }

  private async remove(userId: string, msg: LessonDeleteMessage): Promise<Result> {
    const s = await this.stateOf(msg.worldId);
    const companyId = s?.members[userId]?.companyId;
    if (!s || !companyId) return { ok: false, reason: NOT_IN_WORLD };
    await this.ctx.db
      .delete(schema.lessons)
      .where(
        and(
          eq(schema.lessons.id, msg.lessonId),
          eq(schema.lessons.worldId, msg.worldId),
          eq(schema.lessons.companyId, companyId),
        ),
      );
    await this.recount(msg.worldId, companyId);
    return { ok: true };
  }

  /** The newest 200 stay; older ones go. */
  private async keepNewest(worldId: string, companyId: string): Promise<void> {
    const mine = and(eq(schema.lessons.worldId, worldId), eq(schema.lessons.companyId, companyId));
    const [edge] = await this.ctx.db
      .select({ id: schema.lessons.id })
      .from(schema.lessons)
      .where(mine)
      .orderBy(desc(schema.lessons.id))
      .offset(LESSONS_MAX - 1)
      .limit(1);
    if (edge)
      await this.ctx.db.delete(schema.lessons).where(and(mine, lt(schema.lessons.id, edge.id)));
  }

  private async recount(worldId: string, companyId: string): Promise<void> {
    const rows = await this.of(worldId, companyId);
    const actor = this.ctx.worlds.peek(worldId) ?? (await this.ctx.worlds.get(worldId));
    const r = await actor.submit({
      kind: "system",
      event: { type: "lessonsCounted", companyId, count: Math.min(rows.length, LESSONS_MAX) },
    });
    if (!r.ok) this.ctx.log.warn({ worldId, reason: r.reason }, "lessons not counted");
  }

  /** A company's lessons, oldest first (private, R2). */
  async of(worldId: string, companyId: string) {
    return this.ctx.db
      .select()
      .from(schema.lessons)
      .where(and(eq(schema.lessons.worldId, worldId), eq(schema.lessons.companyId, companyId)))
      .orderBy(asc(schema.lessons.id));
  }

  /** The lessons as the trainer's coaching pairs `[question, answer]`. */
  async pairs(worldId: string, companyId: string): Promise<[string, string][]> {
    return (await this.of(worldId, companyId)).map((l) => [l.question, l.answer]);
  }

  private async list(userId: string, q: Query): Promise<{ lessons: LessonView[] }> {
    if (q.kind !== "lessons") throw new HttpError(400, "Unknown request.");
    const s = await this.stateOf(q.worldId);
    if (!s?.members[userId]) throw new HttpError(403, NOT_IN_WORLD);
    const companyId = s.members[userId]?.companyId;
    if (!companyId) return { lessons: [] };
    const rows = await this.of(q.worldId, companyId);
    return {
      lessons: rows.map((l) => ({
        id: l.id,
        kind: l.kind === "written" ? "written" : "liked",
        question: l.question,
        answer: l.answer,
        at: l.at,
      })),
    };
  }
}
