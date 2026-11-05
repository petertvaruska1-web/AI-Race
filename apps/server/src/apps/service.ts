import { AppReviewJobResult } from "@airace/protocol";
import { type CompanyState, getBalance } from "@airace/sim";
import type { AppContext } from "../app.ts";
import { signBlobUrl } from "../blobs.ts";
import { seedOf } from "../feedback/service.ts";
import { DROP_JOB, isStale } from "../jobs/broker.ts";
import { forCompany, paramsOf } from "../jobs/owners.ts";
import { requestOnce } from "../requests.ts";
import { checkpointKey } from "../training/service.ts";
import type { WorldActor } from "../worlds/actor.ts";

const JOB_URL_TTL_MS = 12 * 60 * 60 * 1000;
const REVIEWS_PER_WEEK = 2;

type AppReviewPayload = {
  worldId: string;
  appId: string;
  versionId: string;
  skill: string;
  key: string;
  /** The game day it was asked; reviews unanswered for 14 days are dropped (Stage 2). */
  enqueuedDay?: number;
};

/**
 * Partner app reviews (playtest pack §5): once a game week, each running app's AI answers two
 * fresh questions in the app's skill (an `appReview` job, held while its player is in the
 * Playground). The scored real replies go back as `appReviewed`; a batch that fails is skipped.
 */
export class AppReviewService {
  private known = new Set<string>();
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, worlds } = ctx;
    worlds.onLoad((actor) => {
      actor.onChange(() => void this.serial(() => this.sync(actor)));
      void this.serial(() => this.sync(actor));
    });
    broker.onPrepare("appReview", async (job) => {
      const p = job.payload as AppReviewPayload;
      if (p.enqueuedDay !== undefined) {
        const actor = ctx.worlds.peek(p.worldId) ?? (await ctx.worlds.get(p.worldId));
        if (isStale(p.enqueuedDay, actor.state().day)) return DROP_JOB;
      }
      return {
        versionId: p.versionId,
        modelTarUrl: signBlobUrl(ctx, checkpointKey(p.versionId), "GET", JOB_URL_TTL_MS),
        skill: p.skill,
        n: REVIEWS_PER_WEEK,
        seed: seedOf(p.appId, p.key),
      };
    });
    broker.onDone("appReview", (job, result) => this.done(job.payload as AppReviewPayload, result));
    broker.onFailed("appReview", (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "app reviews failed; the week is skipped");
    });
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.queue;
  }

  private async sync(actor: WorldActor): Promise<void> {
    if (this.closed || actor.isClosed()) return;
    const s = actor.state();
    const kinds = getBalance(s.balance).economy.apps.kinds;
    const key = `w${Math.floor(s.day / 7)}`;
    for (const companyId of Object.keys(s.companies).sort()) {
      const company = s.companies[companyId] as CompanyState;
      for (const app of Object.values(company.apps)) {
        const product = company.products[app.productId];
        if (app.status !== "live" || product?.status !== "live") continue;
        const memo = `${app.id}|${key}`;
        if (this.known.has(memo)) continue;
        this.known.add(memo);
        const payload: AppReviewPayload = {
          worldId: s.worldId,
          appId: app.id,
          versionId: product.versionId,
          skill: kinds[app.kind].skill,
          key,
          enqueuedDay: s.day,
        };
        const earlier = await requestOnce(
          this.ctx,
          `appReview:${app.id}:${key}`,
          s.worldId,
          "appReview",
          payload,
          forCompany(s, companyId, paramsOf(s, product.versionId)),
        );
        if (earlier?.status === "done") await this.done(payload, earlier.result);
      }
    }
  }

  private async done(p: AppReviewPayload, result: unknown): Promise<void> {
    const parsed = AppReviewJobResult.safeParse(result);
    if (!parsed.success) {
      this.ctx.log.error({ appId: p.appId, issues: parsed.error.issues }, "bad app reviews");
      return;
    }
    const actor = this.ctx.worlds.peek(p.worldId) ?? (await this.ctx.worlds.get(p.worldId));
    const r = await actor.submit({
      kind: "system",
      event: { type: "appReviewed", appId: p.appId, key: p.key, reviews: parsed.data.reviews },
    });
    if (!r.ok)
      this.ctx.log.debug(
        { appId: p.appId, key: p.key, reason: r.reason },
        "app reviews not applied",
      );
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch((err) => this.ctx.log.error({ err }, "app review service step failed"));
    return run;
  }
}
