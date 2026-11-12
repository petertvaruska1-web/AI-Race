import { type Query, TRAINING_ENGINE } from "@airace/protocol";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import type { WorkerConn } from "../jobs/broker.ts";

/**
 * Training tokens a second on the helper's speed test that count as 1.0× speed: what an RTX 3050
 * measures there (about 450,000; the old 30,000 made that card read "15× speed"). The client's
 * minute estimate (`realMinutes`) uses the same number.
 */
export const SPEED_REFERENCE = 450_000;

/** A graphics card as every player in the world sees it (never its key or worker id). */
export type CardView = {
  userId: string;
  playerName: string;
  gpuName: string | null;
  vramMb: number;
  /** × of the reference speed, to one decimal. */
  speed: number;
  online: boolean;
  /** Its helper is from an older version of the game: it gets no work until set up again. */
  outdated: boolean;
  /** The PC running the game (it counts as the world host's card). */
  house: boolean;
};

/**
 * The world's graphics cards (Stage 2): each member's connected helpers, the house card as the
 * world host's, and an offline row for a member with none. Cards are real-time facts, not world
 * state, so they come by query and stream (`cards:<worldId>`), pushed when a card comes or goes.
 */
export class CardsService {
  /** Worlds someone follows the cards of, so changes are pushed only there. */
  private followed = new Set<string>();

  constructor(private readonly ctx: AppContext) {
    ctx.gateway.registerQuery("cards", (userId, q) => this.query(userId, q));
    ctx.gateway.registerStream("cards", async (userId, sub) => {
      const worldId = sub.split(":")[1] ?? "";
      const ok = await this.isMember(userId, worldId);
      if (ok) this.followed.add(worldId);
      return ok;
    });
    ctx.broker.onCardsChanged(() => void this.push());
  }

  private async isMember(userId: string, worldId: string): Promise<boolean> {
    try {
      const actor = await this.ctx.worlds.get(worldId);
      return !!actor.state().members[userId];
    } catch {
      return false;
    }
  }

  private async query(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "cards") throw new HttpError(400, "Unknown request.");
    if (!(await this.isMember(userId, q.worldId))) {
      throw new HttpError(403, "That world doesn't exist or you're not in it.");
    }
    return { cards: await this.cardsIn(q.worldId) };
  }

  /** Every member's cards in join order; the house card belongs to the world's host. */
  async cardsIn(worldId: string): Promise<CardView[]> {
    const s = (await this.ctx.worlds.get(worldId)).state();
    const conns = this.ctx.broker.connections();
    const members = Object.values(s.members).sort((a, b) => a.joinedSeq - b.joinedSeq);
    const out: CardView[] = [];
    for (const m of members) {
      const mine = conns.filter(
        (c) => c.ownerUserId === m.userId || (c.ownerUserId === null && m.userId === s.hostUserId),
      );
      if (mine.length === 0) {
        out.push({
          userId: m.userId,
          playerName: m.displayName,
          gpuName: null,
          vramMb: 0,
          speed: 0,
          online: false,
          outdated: false,
          house: false,
        });
      }
      for (const c of mine) out.push(view(m.userId, m.displayName, c));
    }
    return out;
  }

  private async push(): Promise<void> {
    for (const worldId of this.followed) {
      try {
        this.ctx.gateway.publish(`cards:${worldId}`, { cards: await this.cardsIn(worldId) });
      } catch {
        this.followed.delete(worldId); // the world went away
      }
    }
  }
}

function view(userId: string, playerName: string, c: WorkerConn): CardView {
  return {
    userId,
    playerName,
    gpuName: c.caps.gpuName,
    vramMb: c.caps.vramMb,
    speed: Math.round((10 * c.caps.speed) / SPEED_REFERENCE) / 10,
    online: true,
    outdated: c.caps.engine !== TRAINING_ENGINE,
    house: c.ownerUserId === null,
  };
}
