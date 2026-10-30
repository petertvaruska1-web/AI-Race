import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { join } from "node:path";
import { GROUP_IDS, RivalReplies, type RivalReply } from "@airace/protocol";
import type { FeedbackResult, WorldState } from "@airace/sim";
import type { AppContext } from "../app.ts";
import { signBlobUrl } from "../blobs.ts";
import { seedOf } from "../feedback/service.ts";
import type { WorldActor } from "../worlds/actor.ts";
import { type LoadedPack, loadRivalPack } from "./pack.ts";

const MODEL_URL_TTL_MS = 12 * 60 * 60 * 1000;

/**
 * Rivals (game-layer design §9): seats them in running worlds from the rival
 * pack with one logged `rivalsJoined` input, answers their customer feedback
 * from the replies written at build time (no job), and places their models in
 * the blob store when a chat or battle needs one.
 */
export class RivalService {
  readonly loaded: LoadedPack | null;
  private joined = new Set<string>();
  private replyCache = new Map<string, RivalReply[]>();
  private placing = new Map<string, Promise<void>>();

  constructor(private readonly ctx: AppContext) {
    this.loaded = loadRivalPack(ctx.opts.rivalsDir ?? null, ctx.catalog, ctx.log);
    ctx.worlds.onLoad((actor) => this.attach(actor));
  }

  get installed(): boolean {
    return this.loaded !== null;
  }

  private attach(actor: WorldActor): void {
    actor.onChange(() => this.sync(actor));
    // Not awaited: this runs while the registry is still loading the actor.
    queueMicrotask(() => this.sync(actor));
  }

  /** Rivals join a running world that has none, once the pack is installed. */
  private sync(actor: WorldActor): void {
    if (!this.loaded || actor.isClosed()) return;
    const s = actor.state();
    if (s.phase !== "running" || s.rivals !== null || this.joined.has(s.worldId)) return;
    this.joined.add(s.worldId);
    void actor
      .submit({ kind: "system", event: { type: "rivalsJoined", pack: this.loaded.pack } })
      .then((r) => {
        if (!r.ok) this.ctx.log.info({ worldId: s.worldId, reason: r.reason }, "rivals not seated");
      });
  }

  /** A rival version's answers to the customer questions, read once. */
  replies(packKey: string): RivalReply[] {
    let list = this.replyCache.get(packKey);
    if (!list) {
      if (!this.loaded) return [];
      const file = join(this.loaded.dir, "replies", `${packKey}.json`);
      list = RivalReplies.parse(JSON.parse(readFileSync(file, "utf8")));
      this.replyCache.set(packKey, list);
    }
    return list;
  }

  /** A week's feedback batch for a rival product: `perGroup` stored replies per group, the same for the same batch. */
  feedbackFor(productId: string, packKey: string, key: string, perGroup: number): FeedbackResult[] {
    const all = this.replies(packKey);
    const seed = seedOf(productId, key);
    const out: FeedbackResult[] = [];
    GROUP_IDS.forEach((group, g) => {
      const mine = all.filter((r) => r.group === group);
      if (mine.length === 0) return;
      const first = (seed + g * 7919) % mine.length;
      for (let k = 0; k < Math.min(perGroup, mine.length); k++) {
        const r = mine[(first + k) % mine.length] as RivalReply;
        out.push({ ...r });
      }
    });
    return out;
  }

  /** The traits of every rival AI released in `s`, for personality words. */
  traitsIn(s: WorldState): Record<string, number>[] {
    const out: Record<string, number>[] = [];
    for (const c of Object.values(s.companies)) {
      if (!c.rival) continue;
      for (const v of Object.values(c.versions)) if (v.traits) out.push({ ...v.traits });
    }
    return out;
  }

  /** A worker's download URL for a rival AI, placing its checkpoint in the blob store first. */
  async modelUrl(packKey: string): Promise<string> {
    if (!this.loaded) throw new Error("no rival pack");
    const identity = createHash("sha256").update(this.loaded.pack.dataIdentity).digest("hex");
    const key = `rivals/${identity.slice(0, 12)}/${packKey}.tar`;
    let placing = this.placing.get(key);
    if (!placing) {
      const file = join(this.loaded.dir, "models", `${packKey}.tar`);
      placing = (async () => {
        if (!(await this.ctx.blobs.exists(key)))
          await this.ctx.blobs.put(key, createReadStream(file));
      })();
      this.placing.set(key, placing);
      placing.catch(() => this.placing.delete(key));
    }
    await placing;
    return signBlobUrl(this.ctx, key, "GET", MODEL_URL_TTL_MS);
  }
}
