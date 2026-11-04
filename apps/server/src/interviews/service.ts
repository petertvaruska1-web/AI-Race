import { randomUUID } from "node:crypto";
import { InterviewJobResult } from "@airace/protocol";
import { dateOfDay, getBalance } from "@airace/sim";
import { asc, eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { schema } from "../db/index.ts";
import type { InterviewPublishMessage } from "../realtime/gateway.ts";

/** Published interviews run right after chats: the player is waiting for the reaction. */
const INTERVIEW_PRIORITY = 1;
export const SCORING_FAILED = "The interview couldn't be scored. Try publishing again.";
const NOT_YOURS = "That interview isn't yours.";
const PICK = "Pick answers from this interview.";

type InterviewPayload = { worldId: string; userId: string; requestId: string };
type Result = { ok: true; data: { interviewId: string } } | { ok: false; reason: string };

/**
 * Interviews (game-layer design §10): the player interviews their AI in an interview thread
 * the server keeps, then publishes the AI messages they pick. The answers come from the stored
 * conversation, never from the client, so only what the AI really said reaches the press (R1).
 * An `interview` job scores them; the world gets `interviewScored`.
 */
export class InterviewService {
  constructor(private readonly ctx: AppContext) {
    const { broker, gateway } = ctx;
    gateway.setInterviewHandler((userId, msg) => this.publish(userId, msg));
    broker.onPrepare("interview", async (job) => {
      const p = job.payload as InterviewPayload;
      return { interviewId: p.requestId, answers: await this.answers(p.requestId) };
    });
    broker.onDone("interview", (job, result) => this.done(job.payload as InterviewPayload, result));
    broker.onFailed("interview", (job, error) => {
      const p = job.payload as InterviewPayload;
      ctx.log.warn({ jobId: job.id, error }, "interview scoring failed; nothing was published");
      gateway.notify(p.userId, p.worldId, "interview", SCORING_FAILED);
    });
  }

  private async publish(userId: string, msg: InterviewPublishMessage): Promise<Result> {
    const { ctx } = this;
    let actor: Awaited<ReturnType<typeof ctx.worlds.get>>;
    try {
      actor = await ctx.worlds.get(msg.worldId);
    } catch {
      return { ok: false, reason: "That world doesn't exist or you're not in it." };
    }
    const s = actor.state();
    const company = s.companies[s.members[userId]?.companyId ?? ""];
    if (!company) return { ok: false, reason: NOT_YOURS };
    let thread: typeof schema.chatThreads.$inferSelect | undefined;
    try {
      [thread] = await ctx.db
        .select()
        .from(schema.chatThreads)
        .where(eq(schema.chatThreads.id, msg.threadId));
    } catch {
      thread = undefined; // not a valid id
    }
    if (
      !thread ||
      thread.userId !== userId ||
      thread.worldId !== msg.worldId ||
      thread.kind !== "interview" ||
      !thread.productId
    ) {
      return { ok: false, reason: NOT_YOURS };
    }
    const product = company.products[thread.productId];
    if (!product) return { ok: false, reason: "That AI isn't one of yours." };
    if (product.status !== "live") return { ok: false, reason: "That AI isn't on sale." };
    const { cooldownDays } = getBalance(s.balance).economy.interviews;
    const last = company.lastInterviewDay;
    if (last !== null && s.day < last + cooldownDays) {
      return {
        ok: false,
        reason: `You gave an interview recently. Next one on ${dateOfDay(last + cooldownDays)}.`,
      };
    }
    const messages = await this.messages(thread.id);
    const byIdx = new Map(messages.map((m) => [m.idx, m]));
    const idxs = [...new Set(msg.idxs)].sort((a, b) => a - b);
    const answersOnly = idxs.every(
      (i) => byIdx.get(i)?.role === "ai" && byIdx.get(i - 1)?.role === "user",
    );
    if (!answersOnly) return { ok: false, reason: PICK };

    const requestId = randomUUID();
    await ctx.db.insert(schema.interviewRequests).values({
      id: requestId,
      worldId: msg.worldId,
      companyId: company.id,
      productId: product.id,
      threadId: thread.id,
      idxs,
      createdAt: ctx.opts.scheduler.now(),
    });
    const payload: InterviewPayload = { worldId: msg.worldId, userId, requestId };
    const jobId = await ctx.broker.enqueue("interview", payload, msg.worldId, {
      priority: INTERVIEW_PRIORITY,
    });
    await ctx.db
      .update(schema.interviewRequests)
      .set({ jobId })
      .where(eq(schema.interviewRequests.id, requestId));
    return { ok: true, data: { interviewId: requestId } };
  }

  private messages(threadId: string) {
    return this.ctx.db
      .select()
      .from(schema.chatMessages)
      .where(eq(schema.chatMessages.threadId, threadId))
      .orderBy(asc(schema.chatMessages.idx));
  }

  private async request(requestId: string) {
    const [row] = await this.ctx.db
      .select()
      .from(schema.interviewRequests)
      .where(eq(schema.interviewRequests.id, requestId));
    if (!row) throw new Error(`interview ${requestId} is missing`);
    return row;
  }

  /** The picked answers, each with the question asked just before it, from the stored thread. */
  private async answers(requestId: string) {
    const row = await this.request(requestId);
    const byIdx = new Map((await this.messages(row.threadId)).map((m) => [m.idx, m]));
    return (row.idxs as number[]).map((i) => {
      const asked = byIdx.get(i - 1);
      return {
        cardId: asked?.cardId ?? null,
        question: asked?.text ?? "",
        reply: byIdx.get(i)?.text ?? "",
      };
    });
  }

  private async done(p: InterviewPayload, result: unknown): Promise<void> {
    const parsed = InterviewJobResult.safeParse(result);
    const answers = await this.answers(p.requestId);
    if (!parsed.success || parsed.data.answers.length !== answers.length) {
      this.ctx.log.error({ requestId: p.requestId }, "bad interview result");
      this.ctx.gateway.notify(p.userId, p.worldId, "interview", SCORING_FAILED);
      return;
    }
    const row = await this.request(p.requestId);
    const actor = this.ctx.worlds.peek(p.worldId) ?? (await this.ctx.worlds.get(p.worldId));
    const r = await actor.submit({
      kind: "system",
      event: {
        type: "interviewScored",
        interviewId: p.requestId,
        productId: row.productId,
        answers: answers.map((a, k) => ({
          question: a.question,
          reply: a.reply,
          score: parsed.data.answers[k]?.score ?? 0,
          checked: parsed.data.answers[k]?.checked ?? false,
        })),
      },
    });
    if (!r.ok) this.ctx.gateway.notify(p.userId, p.worldId, "interview", r.reason);
  }
}
