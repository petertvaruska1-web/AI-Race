import { diff, type WorldView } from "@airace/protocol";
import { projectView } from "@airace/sim";
import type { Scheduler, TimerHandle } from "../scheduler.ts";
import type { WorldActor } from "../worlds/actor.ts";
import type { Session } from "./session.ts";

/** World changes are batched into at most one patch per socket per 250 ms. */
export const PATCH_COALESCE_MS = 250;

/**
 * One socket watching one world. It remembers the view it last sent, so each
 * patch is the difference from what that player already has (R2: every view
 * comes from `projectView` for this player).
 */
export class WorldWatch {
  private lastView!: WorldView;
  private lastVersion = 0;
  private timer: TimerHandle | null = null;
  private closed = false;
  private readonly off: () => void;

  constructor(
    private readonly session: Session,
    private readonly actor: WorldActor,
    private readonly userId: string,
    private readonly scheduler: Scheduler,
  ) {
    this.off = actor.onChange(() => this.schedule());
    this.sendSnapshot();
  }

  sendSnapshot(): void {
    const s = this.actor.state();
    this.lastView = projectView(s, this.userId);
    this.lastVersion = s.seq;
    this.session.send({
      v: 1,
      type: "snapshot",
      worldId: this.actor.worldId,
      version: s.seq,
      view: this.lastView,
    });
  }

  private schedule(): void {
    if (this.closed || this.timer) return;
    this.timer = this.scheduler.setTimeout(() => {
      this.timer = null;
      this.flush();
    }, PATCH_COALESCE_MS);
  }

  private flush(): void {
    if (this.closed) return;
    const s = this.actor.state();
    if (!s.members[this.userId]) return;
    const view = projectView(s, this.userId);
    const ops = diff(this.lastView, view);
    // An empty diff sends nothing and keeps `lastVersion`, so patches still chain.
    if (ops.length === 0) return;
    this.session.send({
      v: 1,
      type: "patch",
      worldId: this.actor.worldId,
      from: this.lastVersion,
      to: s.seq,
      ops,
    });
    this.lastView = view;
    this.lastVersion = s.seq;
  }

  close(): void {
    this.closed = true;
    this.off();
    if (this.timer) this.scheduler.clearTimeout(this.timer);
    this.timer = null;
  }
}
