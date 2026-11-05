import { COST_CATEGORIES, INCOME_CATEGORIES, NO_GOAL } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { newCompanyExtras } from "./companyExtras.ts";
import type { WorldState } from "./state.ts";
import { newGoal } from "./world.ts";

// Worlds saved by an older simulation, brought forward so they keep playing.
// Each step only adds the newer fields with their starting values.

/** The simulation versions `upgradeState` can bring forward. */
export const UPGRADABLE = ["sim-3", "sim-4", "sim-5", "sim-6", "sim-7"] as const;
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
    c.battles ??= [];
    for (const r of c.reports as Loose[]) r.ranks ??= null;
    for (const v of Object.values(c.versions as Record<string, Loose>)) v.packKey ??= null;
  }
  s.rivals ??= null;
  s.leaders ??= {};
  s.lastRankingNews ??= {};
  s.points ??= {};
  return s;
}

/**
 * A world saved by sim-4 (Phase 3) as sim-5: the playtest pack's company fields, the
 * new money categories in every month, and the seq up to which inputs replay on the old rules.
 */
function fromSim4(s: Loose, legacyUntilSeq: number): Loose {
  const companies = s.companies as Record<string, Loose>;
  for (const c of Object.values(companies)) {
    const extras = newCompanyExtras() as unknown as Loose;
    for (const [k, v] of Object.entries(extras)) c[k] ??= v;
    const ledger = c.ledger as { current: Loose; months: Loose[] };
    for (const m of [...ledger.months, ledger.current]) {
      const income = m.income as Record<string, number>;
      const costs = m.costs as Record<string, number>;
      for (const k of INCOME_CATEGORIES) income[k] ??= 0;
      for (const k of COST_CATEGORIES) costs[k] ??= 0;
    }
  }
  s.legacyUntilSeq ??= legacyUntilSeq;
  return s;
}

/**
 * A world saved by sim-5 (the playtest pack) as sim-6 (Phase 4): the notebook joins every
 * plan with a share of 0, and every company starts without one.
 */
function fromSim5(s: Loose, lastLoggedSeq: number): Loose {
  s.sim6FromSeq ??= lastLoggedSeq;
  s.goal ??= newGoal(NO_GOAL);
  const companies = s.companies as Record<string, Loose>;
  for (const c of Object.values(companies)) {
    const extras = newCompanyExtras() as unknown as Loose;
    for (const [k, v] of Object.entries(extras)) c[k] ??= v;
    for (const r of Object.values(c.runs as Record<string, Loose>)) {
      const plan = r.plan as { mix: Record<string, number> };
      plan.mix.notebook ??= 0;
      r.notebookHash ??= null;
    }
  }
  return s;
}

/** A world saved by sim-6 (Phase 4) as sim-7 (Stage 2): no Fair mode. */
function fromSim6(s: Loose): Loose {
  s.fairMode ??= false;
  return s;
}

/** A sim-7 world as sim-8 (coaching): no lessons yet; no stored plan trains on them. */
function fromSim7(s: Loose): Loose {
  for (const c of Object.values(s.companies as Record<string, Loose>)) {
    c.lessons ??= 0;
    for (const r of Object.values((c.runs ?? {}) as Record<string, Loose>)) {
      const plan = r.plan as Loose | undefined;
      if (plan) plan.coaching ??= false;
    }
  }
  return s;
}

/**
 * Brings an older world forward. `legacyUntilSeq` is the last logged input that will be
 * replayed on it: those inputs keep the old rules where the new ones would change money
 * (they were accepted under them), so replay can't diverge.
 */
export function upgradeState(
  state: unknown,
  from: UpgradableVersion,
  legacyUntilSeq?: number,
): WorldState {
  const s = structuredClone(state) as Loose;
  if (from === "sim-3") fromSim3(s);
  if (from === "sim-3" || from === "sim-4") fromSim4(s, legacyUntilSeq ?? (s.seq as number));
  if (from !== "sim-6" && from !== "sim-7") fromSim5(s, legacyUntilSeq ?? (s.seq as number));
  if (from !== "sim-7") fromSim6(s);
  fromSim7(s);
  return s as unknown as WorldState;
}
