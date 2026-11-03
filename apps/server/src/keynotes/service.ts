import { KeynoteJobResult } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import type { AppContext } from "../app.ts";
import { signBlobUrl } from "../blobs.ts";
import { seedOf } from "../feedback/service.ts";
import { requestOnce } from "../requests.ts";
import { checkpointKey } from "../training/service.ts";
import type { WorldActor } from "../worlds/actor.ts";

const JOB_URL_TTL_MS = 12 * 60 * 60 * 1000;

type KeynotePayload = { worldId: string; productId: string; versionId: string; questionId: string };

/**
 * Launch keynotes (playtest pack §5): a launch announced with a keynote becomes one `keynote`
 * job, in which the product's AI answers its demo question live. The real reply and its score
 * go back as `keynoteResolved`; a job that finally fails calls the keynote off (refunded).
 */
export class KeynoteService {
  private known = new Set<string>();
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    const { broker, worlds } = ctx;
    worlds.onLoad((actor) => {
      actor.onChange(() => void this.serial(() => this.sync(actor)));
      void this.serial(() => this.sync(actor));
    });
    broker.onPrepare("keynote", async (job) => {
      const p = job.payload as KeynotePayload;
      return {
        versionId: p.versionId,
        modelTarUrl: signBlobUrl(ctx, checkpointKey(p.versionId), "GET", JOB_URL_TTL_MS),
        questionId: p.questionId,
        seed: seedOf(p.productId, "keynote"),
      };
    });
    broker.onDone("keynote", (job, result) => this.done(job.payload as KeynotePayload, result));
    broker.onFailed("keynote", (job, error) => {
      ctx.log.warn({ jobId: job.id, error }, "keynote failed; it is called off");
      return this.resolve(job.payload as KeynotePayload, null);
    });
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.queue;
  }

  private async sync(actor: WorldActor): Promise<void> {
    if (this.closed || actor.isClosed()) return;
    const s = actor.state();
    for (const companyId of Object.keys(s.companies).sort()) {
      const company = s.companies[companyId] as CompanyState;
      for (const product of Object.values(company.products)) {
        const k = product.keynote;
        if (k?.status !== "waiting" || this.known.has(product.id)) continue;
        this.known.add(product.id);
        const payload: KeynotePayload = {
          worldId: s.worldId,
          productId: product.id,
          versionId: product.versionId,
          questionId: k.questionId,
        };
        const earlier = await requestOnce(
          this.ctx,
          `keynote:${product.id}`,
          s.worldId,
          "keynote",
          payload,
        );
        // After a restart: a job that ended while its world was not loaded is applied now.
        if (earlier?.status === "done") await this.done(payload, earlier.result);
        if (earlier?.status === "failed" || earlier?.status === "cancelled")
          await this.resolve(payload, null);
      }
    }
  }

  private async done(p: KeynotePayload, result: unknown): Promise<void> {
    const parsed = KeynoteJobResult.safeParse(result);
    if (!parsed.success) {
      this.ctx.log.error(
        { productId: p.productId, issues: parsed.error.issues },
        "bad keynote result",
      );
      return this.resolve(p, null);
    }
    return this.resolve(p, parsed.data);
  }

  private async resolve(p: KeynotePayload, r: KeynoteJobResult | null): Promise<void> {
    const actor = this.ctx.worlds.peek(p.worldId) ?? (await this.ctx.worlds.get(p.worldId));
    const event = r
      ? {
          type: "keynoteResolved" as const,
          productId: p.productId,
          prompt: r.prompt,
          reply: r.reply,
          score: r.score,
        }
      : { type: "keynoteResolved" as const, productId: p.productId, failed: true as const };
    const out = await actor.submit({ kind: "system", event });
    if (!out.ok)
      this.ctx.log.info({ productId: p.productId, reason: out.reason }, "keynote not applied");
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch((err) => this.ctx.log.error({ err }, "keynote service step failed"));
    return run;
  }
}
