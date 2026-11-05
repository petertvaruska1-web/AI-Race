import { paramCount, type WorldState } from "@airace/sim";
import type { EnqueueOptions } from "./broker.ts";

/** Routing for a job done for `companyId`: its owner's card, its share (Stage 2). */
export function forCompany(s: WorldState, companyId: string, minParams = 0): EnqueueOptions {
  return { owner: s.companies[companyId]?.ownerUserId ?? null, companyId, minParams };
}

/** Routing for a job a player asked for themselves (a chat, a battle, a game). */
export function forPlayer(s: WorldState, userId: string, minParams = 0): EnqueueOptions {
  return { owner: userId, companyId: s.members[userId]?.companyId ?? null, minParams };
}

/** The parameters of a version (a player's or a rival's), 0 when it isn't known. */
export function paramsOf(s: WorldState, versionId: string): number {
  for (const c of Object.values(s.companies)) {
    const v = c.versions[versionId];
    if (v) return paramCount(v.nLayer, v.dModel);
  }
  return 0;
}
