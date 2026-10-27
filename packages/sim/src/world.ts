import type { BalanceVersion } from "./balance.ts";
import { emptyMarket } from "./market.ts";
import { seedRng } from "./rng.ts";
import { drawSpotlights } from "./spotlights.ts";
import type { WorldState } from "./state.ts";

export function createWorld(p: {
  worldId: string;
  name: string;
  hostUserId: string;
  hostName: string;
  seed: number;
  balance: BalanceVersion;
  inviteCode: string;
}): WorldState {
  const { spotlights, rng } = drawSpotlights(seedRng(p.seed));
  return {
    worldId: p.worldId,
    name: p.name,
    balance: p.balance,
    rng,
    phase: "setup",
    day: 0,
    speed: 1,
    paused: false,
    hostUserId: p.hostUserId,
    members: {
      [p.hostUserId]: {
        userId: p.hostUserId,
        displayName: p.hostName,
        joinedSeq: 0,
        online: true,
        companyId: null,
      },
    },
    companies: {},
    inviteCode: p.inviteCode,
    seq: 0,
    market: emptyMarket(),
    spotlights,
    feed: [],
    feedSeq: 0,
  };
}

/** The host while online, else the earliest-joined online member (C1), else null. */
export function effectiveHost(state: WorldState): string | null {
  if (state.members[state.hostUserId]?.online) return state.hostUserId;
  let best: { userId: string; joinedSeq: number } | null = null;
  for (const m of Object.values(state.members)) {
    if (m.online && (best === null || m.joinedSeq < best.joinedSeq)) best = m;
  }
  return best?.userId ?? null;
}
