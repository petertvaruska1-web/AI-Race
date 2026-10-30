import { getBalance } from "./balance.ts";
import type { WorldState } from "./state.ts";

// Worlds saved by an older simulation, brought forward so they keep playing.
// Each step only adds the newer fields with their starting values.

/** The simulation versions `upgradeState` can bring forward. */
export const UPGRADABLE = ["sim-3"] as const;
export type UpgradableVersion = (typeof UPGRADABLE)[number];

type Loose = Record<string, unknown>;

/** A world saved by sim-3 (Phase 2) as sim-4: no rivals yet; companies keep their founding cash. */
function fromSim3(s: Loose): Loose {
  const founding = getBalance(s.balance as WorldState["balance"]).foundingCash;
  const companies = s.companies as Record<string, Loose>;
  for (const c of Object.values(companies)) {
    c.rival ??= null;
    c.startCash ??= founding;
    c.lastRanks ??= null;
    for (const r of c.reports as Loose[]) r.ranks ??= null;
    for (const v of Object.values(c.versions as Record<string, Loose>)) v.packKey ??= null;
  }
  s.rivals ??= null;
  s.leaders ??= {};
  s.lastRankingNews ??= {};
  return s;
}

export function upgradeState(state: unknown, from: UpgradableVersion): WorldState {
  const s = structuredClone(state) as Loose;
  if (from === "sim-3") fromSim3(s);
  return s as unknown as WorldState;
}
