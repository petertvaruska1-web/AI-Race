import { type Command, type Query, TRAINING_ENGINE } from "@airace/protocol";
import {
  type Balance,
  type BrainSize,
  getBalance,
  paramCount,
  previewTraining,
  type WorldState,
} from "@airace/sim";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import type { WorkerConn } from "../jobs/broker.ts";

export const NO_CARD = "Connect your computer to do this.";

/** The largest brain whose parameters fit in `maxParams`, or null when none does. */
export function biggestBrain(b: Balance, maxParams: number): BrainSize | null {
  let best: BrainSize | null = null;
  for (const brain of b.brainSizes) {
    const p = paramCount(brain.nLayer, brain.dModel);
    if (p <= maxParams && (!best || p > paramCount(best.nLayer, best.dModel))) best = brain;
  }
  return best;
}

const named = (id: string) => id.charAt(0).toUpperCase() + id.slice(1);

/**
 * The cards that train for `userId` in this world: with Fair mode off, their own helpers (and the
 * house card when they host the world); with it on, every member's card and the house card.
 */
export function trainingCards(conns: WorkerConn[], s: WorldState, userId: string): WorkerConn[] {
  // A helper that needs an update gets no work (the broker), so it trains for nobody.
  const current = conns.filter((c) => c.caps.engine === TRAINING_ENGINE);
  if (s.fairMode) {
    return current.filter((c) => c.ownerUserId === null || !!s.members[c.ownerUserId]);
  }
  return current.filter((c) =>
    c.ownerUserId === null ? userId === s.hostUserId : c.ownerUserId === userId,
  );
}

export type TrainingLimit = { maxBrain: string | null; tokensPerSecond: number };

export function trainingLimitOf(cards: WorkerConn[], b: Balance): TrainingLimit {
  const maxParams = Math.max(0, ...cards.map((c) => c.caps.maxParams));
  const tokensPerSecond = Math.max(0, ...cards.map((c) => c.caps.speed));
  return {
    maxBrain: cards.length ? (biggestBrain(b, maxParams)?.id ?? null) : null,
    tokensPerSecond,
  };
}

/** Why `userId` can't start this training here, or null when a card can take it. */
export function trainingRefusal(
  conns: WorkerConn[],
  s: WorldState,
  userId: string,
  cmd: Extract<Command, { type: "startTraining" }>,
): string | null {
  const companyId = s.members[userId]?.companyId;
  if (!companyId) return null; // the world says why
  const preview = previewTraining(s, companyId, cmd.plan);
  if (!preview.ok) return null;
  const cards = trainingCards(conns, s, userId);
  if (cards.length === 0) return NO_CARD;
  const maxParams = Math.max(...cards.map((c) => c.caps.maxParams));
  if (preview.preview.params <= maxParams) return null;
  const brain = biggestBrain(getBalance(s.balance), maxParams);
  if (s.fairMode) {
    return brain
      ? `The shared cards can train up to the ${named(brain.id)} brain.`
      : "The shared cards are too small to train an AI.";
  }
  return brain
    ? `Your graphics card can train up to the ${named(brain.id)} brain.`
    : "Your graphics card is too small to train an AI.";
}

/** Training fits the cards that do it (Stage 2): the refusal before logging, and the Lab's limit. */
export class LimitsService {
  constructor(private readonly ctx: AppContext) {
    ctx.gateway.setCommandGuard("startTraining", async (userId, worldId, cmd) => {
      if (cmd.type !== "startTraining") return null;
      const s = (await ctx.worlds.get(worldId)).state();
      return trainingRefusal(ctx.broker.connections(), s, userId, cmd);
    });
    ctx.gateway.registerQuery("trainingLimit", (userId, q) => this.query(userId, q));
  }

  private async query(userId: string, q: Query): Promise<TrainingLimit> {
    if (q.kind !== "trainingLimit") throw new HttpError(400, "Unknown request.");
    let s: WorldState;
    try {
      s = (await this.ctx.worlds.get(q.worldId)).state();
    } catch {
      throw new HttpError(403, "That world doesn't exist or you're not in it.");
    }
    if (!s.members[userId])
      throw new HttpError(403, "That world doesn't exist or you're not in it.");
    const cards = trainingCards(this.ctx.broker.connections(), s, userId);
    return trainingLimitOf(cards, getBalance(s.balance));
  }
}
