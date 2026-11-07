import { type GoalSpec, NO_GOAL } from "@airace/protocol";
import { type Balance, type BalanceVersion, getBalance } from "./balance.ts";
import { newLicenses } from "./licenseState.ts";
import { emptyMarket } from "./market.ts";
import { seedRng } from "./rng.ts";
import { drawSpotlights } from "./spotlights.ts";
import type { CompanyState, WorldState } from "./state.ts";

/** Founder grant: extra founding cash per game day of the world's age (Stage 2). */
const FOUNDER_GRANT_PER_DAY = 2_000;

export function createWorld(p: {
  worldId: string;
  name: string;
  hostUserId: string;
  hostName: string;
  seed: number;
  balance: BalanceVersion;
  inviteCode: string;
  goal?: GoalSpec;
  /** Everyone's graphics cards shared equally (C36); fixed for the world's life. */
  fairMode?: boolean;
}): WorldState {
  const { spotlights, rng } = drawSpotlights(seedRng(p.seed), getBalance(p.balance).economy);
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
    legacyUntilSeq: 0,
    sim6FromSeq: 0,
    sim9FromSeq: 0,
    sim10FromSeq: 0,
    licenses: newLicenses(),
    rivals: null,
    leaders: {},
    lastRankingNews: {},
    points: {},
    market: emptyMarket(),
    spotlights,
    feed: [],
    feedSeq: 0,
    goal: newGoal(p.goal ?? NO_GOAL),
    fairMode: p.fairMode ?? false,
  };
}

/** Founding cash for a company founded on `day`: a grant that grows with the world's age, up to double. */
export function foundingCashOn(b: Balance, day: number): number {
  return Math.min(2 * b.foundingCash, b.foundingCash + FOUNDER_GRANT_PER_DAY * day);
}

/** Whether a player company's owner is away: their company runs on autopilot (C40). */
export function ownerAway(s: WorldState, company: CompanyState): boolean {
  if (!company.ownerUserId || company.rival) return false;
  return s.members[company.ownerUserId]?.online === false;
}

/** A goal as the world keeps it; the deadline is set when the world starts. */
export function newGoal(spec: GoalSpec): WorldState["goal"] {
  return { spec, deadlineDay: null, reachedDay: null, reachedBy: null, missedDay: null };
}

/**
 * Whether the input being applied was logged by an older simulation (an upgraded world
 * replaying its log): rules that changed how money moves keep their old values for it.
 * The input being applied will get seq `s.seq + 1`.
 */
export function onLegacyRules(s: WorldState): boolean {
  return s.seq < (s.legacyUntilSeq ?? 0);
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
