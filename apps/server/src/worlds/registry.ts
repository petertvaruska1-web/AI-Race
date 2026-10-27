import type { Input } from "@airace/sim";
import type { AppContext } from "../app.ts";
import { WorldActor } from "./actor.ts";

/** One actor per world, loaded on first use (spec §5.1). */
export class WorldRegistry {
  private actors = new Map<string, Promise<WorldActor>>();
  private loaded = new Map<string, WorldActor>();
  private loadHooks: ((actor: WorldActor) => Promise<void> | void)[] = [];

  constructor(private readonly ctx: AppContext) {}

  get(worldId: string): Promise<WorldActor> {
    let pending = this.actors.get(worldId);
    if (!pending) {
      pending = this.load(worldId);
      this.actors.set(worldId, pending);
      pending.catch(() => {
        if (this.actors.get(worldId) === pending) this.actors.delete(worldId);
      });
    }
    return pending;
  }

  private async load(worldId: string): Promise<WorldActor> {
    const actor = await WorldActor.load(this.ctx, worldId, (a) => {
      // Only forget this actor, never a newer one loaded after an eviction.
      if (this.loaded.get(worldId) !== a) return;
      this.loaded.delete(worldId);
      this.actors.delete(worldId);
    });
    this.loaded.set(worldId, actor);
    // Nobody is connected to a world that was just loaded: everyone it saved as
    // online is now away, so it stays frozen until someone comes back (C1).
    const s = actor.state();
    for (const id of Object.keys(s.members).sort()) {
      if (s.members[id]?.online) {
        const offline: Input = {
          kind: "system",
          event: { type: "presence", userId: id, online: false },
        };
        await actor.submit(offline);
      }
    }
    for (const hook of this.loadHooks) await hook(actor);
    return actor;
  }

  /** Runs `fn` for every actor once it is loaded (services attach their listeners here). */
  onLoad(fn: (actor: WorldActor) => Promise<void> | void): void {
    this.loadHooks.push(fn);
  }

  /** The actor if it is loaded, without loading it. */
  peek(worldId: string): WorldActor | null {
    return this.loaded.get(worldId) ?? null;
  }

  isLoaded(worldId: string): boolean {
    return this.actors.has(worldId);
  }

  /** Drops an actor without saving, as a crash would (tests). */
  async evict(worldId: string): Promise<void> {
    const pending = this.actors.get(worldId);
    this.actors.delete(worldId);
    const actor = pending ? await pending : undefined;
    if (actor && this.loaded.get(worldId) === actor) this.loaded.delete(worldId);
    await actor?.unload();
  }

  /** Snapshots and unloads every actor. */
  async shutdown(): Promise<void> {
    const actors = await Promise.allSettled([...this.actors.values()]);
    for (const r of actors) {
      if (r.status !== "fulfilled") continue;
      await r.value.snapshot("shutdown");
      await r.value.unload();
    }
    this.actors.clear();
    this.loaded.clear();
  }
}
