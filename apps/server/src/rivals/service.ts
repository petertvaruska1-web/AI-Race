import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  type Breakdown,
  GROUP_IDS,
  type RivalPackName,
  RivalReplies,
  type RivalReply,
} from "@airace/protocol";
import type { BalanceVersion, FeedbackResult, WorldState } from "@airace/sim";
import type { AppContext } from "../app.ts";
import { signBlobUrl } from "../blobs.ts";
import { seedOf } from "../feedback/service.ts";
import type { WorldActor } from "../worlds/actor.ts";
import { type LoadedPack, loadRivalPack } from "./pack.ts";

const MODEL_URL_TTL_MS = 12 * 60 * 60 * 1000;

/** The pack each world's rules seat their rivals from (a world records its rules). */
export const PACK_FOR_RULES: Record<BalanceVersion, RivalPackName> = {
  "balance-v2": "rivals-v1",
  "balance-v3": "rivals-v2",
  "balance-v4": "rivals-v3",
  "balance-test": "rivals-v3",
};

/**
 * Rivals (game-layer design §9): seats them in running worlds from a rival
 * pack with one logged `rivalsJoined` input, answers their customer feedback
 * from the replies written at build time (no job), and places their models in
 * the blob store when a chat or battle needs one.
 *
 * Every pack stays installed: the current one (built on the current training data, which new
 * worlds draw from) and the earlier ones, whatever data they were built on, for the rivals
 * already seated from them and for worlds of the rules they were built for. No two packs share
 * an AI's key, so a key alone says which pack a rival AI is in.
 */
export class RivalService {
  /** The current pack (`rivalsDir`), when it is installed and built on the current data. */
  readonly current: LoadedPack | null;
  /** The earlier packs (`earlierRivalsDirs`) that are installed, in that order. */
  readonly earlier: LoadedPack[];
  private joined = new Set<string>();
  private replyCache = new Map<string, RivalReply[]>();
  private placing = new Map<string, Promise<void>>();

  constructor(private readonly ctx: AppContext) {
    this.current = loadRivalPack(ctx.opts.rivalsDir ?? null, ctx.catalog, ctx.log);
    const taken = new Set(this.current?.keys ?? []);
    this.earlier = [];
    for (const dir of ctx.opts.earlierRivalsDirs ?? []) {
      const p = loadRivalPack(dir, ctx.catalog, ctx.log, { current: false });
      if (!p) continue;
      const shared = [...p.keys].filter((k) => taken.has(k));
      if (shared.length > 0) {
        ctx.log.error(
          { dir, shared: shared.slice(0, 3) },
          "this rival pack's AIs have the keys of another installed pack; it is not loaded",
        );
        continue;
      }
      for (const k of p.keys) taken.add(k);
      this.earlier.push(p);
    }
    ctx.worlds.onLoad((actor) => this.attach(actor));
  }

  private get packs(): LoadedPack[] {
    return this.current ? [this.current, ...this.earlier] : this.earlier;
  }

  /**
   * The pack a world of these rules draws its rivals from: the pack built for its rules, else
   * the current one. An earlier pack never seats a world of other rules.
   */
  packFor(balance: BalanceVersion): LoadedPack | null {
    return this.packs.find((p) => p.pack.version === PACK_FOR_RULES[balance]) ?? this.current;
  }

  /** The pack a rival AI is in. */
  private packOf(packKey: string): LoadedPack | null {
    return this.packs.find((p) => p.keys.has(packKey)) ?? null;
  }

  /** Whether worlds made now get rivals. */
  get installed(): boolean {
    return this.packFor(this.ctx.opts.balance) !== null;
  }

  private attach(actor: WorldActor): void {
    actor.onChange(() => this.sync(actor));
    // Not awaited: this runs while the registry is still loading the actor.
    queueMicrotask(() => this.sync(actor));
  }

  /** Rivals join a running world that has none, once the pack is installed. */
  private sync(actor: WorldActor): void {
    if (actor.isClosed()) return;
    const s = actor.state();
    const from = this.packFor(s.balance);
    if (!from) return;
    if (s.phase !== "running" || s.rivals !== null || this.joined.has(s.worldId)) return;
    this.joined.add(s.worldId);
    void actor
      .submit({ kind: "system", event: { type: "rivalsJoined", pack: from.pack } })
      .then((r) => {
        if (!r.ok) this.ctx.log.info({ worldId: s.worldId, reason: r.reason }, "rivals not seated");
      });
  }

  /** A rival version's answers to the customer questions, read once. */
  replies(packKey: string): RivalReply[] {
    let list = this.replyCache.get(packKey);
    if (!list) {
      const from = this.packOf(packKey);
      if (!from) return [];
      const file = join(from.dir, "replies", `${packKey}.json`);
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

  /** Every rival AI released in `s`, as discoveries compare with it: its breakdown and traits. */
  peersIn(s: WorldState): { breakdown: Breakdown; traits: Record<string, number> }[] {
    const byKey = new Map(
      this.packs.flatMap((p) =>
        p.pack.lineages.flatMap((l) => l.versions.map((v) => [v.key, v] as const)),
      ),
    );
    const out: { breakdown: Breakdown; traits: Record<string, number> }[] = [];
    for (const c of Object.values(s.companies)) {
      if (!c.rival) continue;
      for (const v of Object.values(c.versions)) {
        const packed = v.packKey ? byKey.get(v.packKey) : undefined;
        if (packed) out.push({ breakdown: { ...packed.breakdown }, traits: { ...packed.traits } });
      }
    }
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
    const from = this.packOf(packKey);
    if (!from) throw new Error(`no rival pack holds ${packKey}`);
    const identity = createHash("sha256").update(from.pack.dataIdentity).digest("hex");
    const key = `rivals/${identity.slice(0, 12)}/${packKey}.tar`;
    let placing = this.placing.get(key);
    if (!placing) {
      const file = join(from.dir, "models", `${packKey}.tar`);
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
