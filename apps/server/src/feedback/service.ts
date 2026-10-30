import { createHash } from "node:crypto";
import { FeedbackJobResult, GROUP_IDS } from "@airace/protocol";
import { getBalance, type WorldState } from "@airace/sim";
import { and, eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { signBlobUrl } from "../blobs.ts";
import { schema } from "../db/index.ts";
import { checkpointKey } from "../training/service.ts";
import type { WorldActor } from "../worlds/actor.ts";

const JOB_URL_TTL_MS = 12 * 60 * 60 * 1000;

/** What a feedback job carries; the worker gets a spec built from it at lease time. */
type FeedbackPayload = {
  worldId: string;
  productId: string;
  versionId: string;
  key: string;
  perGroup: number;
};

/**
 * Customer feedback (game-layer design §7): each live product's AI answers a
 * question from every customer group's pool, once on launch and then once per
 * game week, as a low-priority job. The scored replies go back into the world
 * as `feedbackScored`. A batch is recorded before it is queued, so a restart
 * never asks for it twice; a batch that finally fails is skipped.
 */
export class FeedbackService {
  private known = new Set<string>();
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, worlds } = ctx;
    worlds.onLoad((actor) => this.attach(actor));
    broker.onPrepare("feedback", (job) => this.spec(job.payload as FeedbackPayload));
    broker.onDone("feedback", (job, result) => this.done(job.payload as FeedbackPayload, result));
    broker.onFailed("feedback", (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "customer feedback failed; the batch is skipped");
    });
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.queue;
  }

  private attach(actor: WorldActor): void {
    actor.onChange(() => void this.serial(() => this.sync(actor)));
    // Not awaited: this runs while the registry is still loading the actor.
    void this.serial(() => this.sync(actor));
  }

  /** Every live product gets its launch batch once, then one batch per game week. */
  private async sync(actor: WorldActor): Promise<void> {
    if (this.closed || actor.isClosed()) return;
    const s = actor.state();
    const week = `w${Math.floor(s.day / 7)}`;
    for (const companyId of Object.keys(s.companies).sort()) {
      const company = s.companies[companyId];
      if (!company) continue;
      for (const productId of Object.keys(company.products).sort()) {
        const product = company.products[productId];
        if (product?.status !== "live") continue;
        const packKey = company.rival ? company.versions[product.versionId]?.packKey : null;
        if (packKey) {
          await this.answerFromPack(actor, productId, packKey, week);
          continue;
        }
        for (const key of ["launch", week])
          await this.request(s, productId, product.versionId, key);
      }
    }
  }

  private async request(
    s: WorldState,
    productId: string,
    versionId: string,
    key: string,
  ): Promise<void> {
    const memo = `${productId}|${key}`;
    if (this.known.has(memo)) return;
    const fresh = await this.ctx.db
      .insert(schema.feedbackRequests)
      .values({ productId, key })
      .onConflictDoNothing()
      .returning();
    this.known.add(memo);
    if (fresh.length === 0) return;
    const payload: FeedbackPayload = {
      worldId: s.worldId,
      productId,
      versionId,
      key,
      perGroup: getBalance(s.balance).economy.feedback.perGroup,
    };
    const jobId = await this.ctx.broker.enqueue("feedback", payload, s.worldId);
    await this.ctx.db
      .update(schema.feedbackRequests)
      .set({ jobId })
      .where(
        and(eq(schema.feedbackRequests.productId, productId), eq(schema.feedbackRequests.key, key)),
      );
  }

  /**
   * A rival's weekly batch comes from the replies its AI wrote when the pack was
   * built: no job and no request row (the sim refuses a week it already has).
   */
  private async answerFromPack(
    actor: WorldActor,
    productId: string,
    packKey: string,
    key: string,
  ): Promise<void> {
    const memo = `${productId}|${key}`;
    if (this.known.has(memo)) return;
    this.known.add(memo);
    const s = actor.state();
    const perGroup = getBalance(s.balance).economy.feedback.perGroup;
    const results = this.ctx.rivals.feedbackFor(productId, packKey, key, perGroup);
    if (results.length === 0) return;
    const r = await actor.submit({
      kind: "system",
      event: { type: "feedbackScored", productId, key, results },
    });
    if (!r.ok)
      this.ctx.log.debug({ productId, key, reason: r.reason }, "rival feedback not applied");
  }

  private async spec(p: FeedbackPayload): Promise<unknown> {
    return {
      productId: p.productId,
      versionId: p.versionId,
      modelTarUrl: signBlobUrl(this.ctx, checkpointKey(p.versionId), "GET", JOB_URL_TTL_MS),
      groups: [...GROUP_IDS],
      perGroup: p.perGroup,
      seed: seedOf(p.productId, p.key),
    };
  }

  private async done(p: FeedbackPayload, result: unknown): Promise<void> {
    const parsed = FeedbackJobResult.safeParse(result);
    if (!parsed.success) {
      this.ctx.log.error({ productId: p.productId, issues: parsed.error.issues }, "bad feedback");
      return;
    }
    const actor = this.ctx.worlds.peek(p.worldId) ?? (await this.ctx.worlds.get(p.worldId));
    const r = await actor.submit({
      kind: "system",
      event: {
        type: "feedbackScored",
        productId: p.productId,
        key: p.key,
        results: parsed.data.results,
      },
    });
    if (!r.ok) {
      this.ctx.log.info(
        { worldId: p.worldId, productId: p.productId, key: p.key, reason: r.reason },
        "customer feedback not applied",
      );
    }
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch((err) => this.ctx.log.error({ err }, "feedback service step failed"));
    return run;
  }
}

/** The same seed for the same batch, so a retried job asks the same questions. */
export function seedOf(productId: string, key: string): number {
  return createHash("sha256").update(`${productId}:${key}`).digest().readUInt32BE(0) & 0x7fffffff;
}
