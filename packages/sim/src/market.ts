import {
  GROUP_IDS,
  type GroupId,
  PLAN_IDS,
  type PlanId,
  REGION_IDS,
  type RegionId,
} from "@airace/protocol";
import { adoptionShare, type Economy, getBalance } from "./balance.ts";
import { liveProductFor } from "./products.ts";
import { paramCount } from "./shape.ts";
import type { MarketState, VersionState, WorldState } from "./state.ts";

// The customer market (game-layer design §5). Every day each customer group
// weighs each company's plans against using no AI at all; customers drift
// toward those preferences as fast as awareness lets them, and unhappy ones
// leave. All of it is deterministic: fixed iteration order, no randomness.

export function emptyMarket(): MarketState {
  return { customers: {}, awareness: {}, satisfaction: {} };
}

/** How appealing one plan's product is to a group (a utility; higher is better). */
export function productAppeal(
  economy: Economy,
  group: GroupId,
  version: VersionState,
  plan: PlanId,
  price: number,
  reputation: number,
  satisfaction: number,
  served = 1,
): number {
  const g = economy.groups[group];
  const u = economy.utility;
  let skill = 0;
  for (const [category, weight] of Object.entries(g.weights)) {
    skill += (weight ?? 0) * ((version.scores?.[category] ?? 0) / 100);
  }
  const params = paramCount(version.nLayer, version.dModel);
  const speed = 1 - params / (params + 5e6);
  const cost = plan === "free" ? u.freePenalty : g.priceSensitivity * Math.log(1 + price / 10);
  const fit = plan === "business" && g.buysBusiness ? u.businessFit : 0;
  return (
    u.score * skill -
    u.price * cost +
    fit +
    u.speed * g.speedPreference * speed +
    u.reputation * (reputation / 100 - 0.5) +
    u.satisfaction * (satisfaction - 0.5) +
    u.reliability * (served - 1)
  );
}

/** Logit shares of each option, beside an outside "no AI" option of utility `noAi`. */
export function choiceShares(utilities: number[], noAi: number): number[] {
  const top = Math.max(noAi, ...utilities);
  const weights = utilities.map((u) => Math.exp(u - top));
  const total = weights.reduce((a, b) => a + b, Math.exp(noAi - top));
  return weights.map((w) => w / total);
}

/** Customers of a company in one region, group and plan (0 when none). */
export function customersOf(
  m: MarketState,
  companyId: string,
  region: string,
  group: GroupId,
  plan: PlanId,
): number {
  return m.customers[companyId]?.[region]?.[group]?.[plan] ?? 0;
}

/** All of a company's customers (a business account counts once). */
export function totalCustomers(m: MarketState, companyId: string): number {
  let n = 0;
  for (const region of REGION_IDS) {
    for (const group of GROUP_IDS) {
      for (const plan of PLAN_IDS) n += customersOf(m, companyId, region, group, plan);
    }
  }
  return n;
}

/** A company's customers in each region, over every group and plan. */
export function customersByRegion(m: MarketState, companyId: string): Record<RegionId, number> {
  const out = {} as Record<RegionId, number>;
  for (const region of REGION_IDS) {
    let n = 0;
    for (const group of GROUP_IDS) {
      for (const plan of PLAN_IDS) n += customersOf(m, companyId, region, group, plan);
    }
    out[region] = n;
  }
  return out;
}

function setCustomers(
  m: MarketState,
  companyId: string,
  region: string,
  group: GroupId,
  plan: PlanId,
  n: number,
): void {
  const byRegion = m.customers[companyId] ?? {};
  const byGroup = byRegion[region] ?? {};
  const byPlan = byGroup[group] ?? {};
  byPlan[plan] = Math.max(0, n);
  byGroup[group] = byPlan;
  byRegion[region] = byGroup;
  m.customers[companyId] = byRegion;
}

export function awarenessOf(
  m: MarketState,
  companyId: string,
  region: string,
  group: GroupId,
): number {
  return m.awareness[companyId]?.[region]?.[group] ?? 0;
}

export function setAwareness(
  m: MarketState,
  companyId: string,
  region: string,
  group: GroupId,
  a: number,
): void {
  const byRegion = m.awareness[companyId] ?? {};
  const byGroup = byRegion[region] ?? {};
  byGroup[group] = Math.min(1, Math.max(0, a));
  byRegion[region] = byGroup;
  m.awareness[companyId] = byRegion;
}

/** People in a region and group willing to use an AI today. */
export function adoptable(economy: Economy, day: number, region: string, group: GroupId): number {
  const population = economy.population[region as keyof Economy["population"]] ?? 0;
  const mix = economy.groupMix[region as keyof Economy["groupMix"]]?.[group] ?? 0;
  return population * adoptionShare(economy, day) * mix;
}

/** The daily market step. Call after products' launch days are applied. */
export function marketDay(s: WorldState): void {
  const e = getBalance(s.balance).economy;
  const m = s.market;
  const companyIds = Object.keys(s.companies).sort();

  // Launch days lift awareness everywhere, more with more hype.
  for (const id of companyIds) {
    const company = s.companies[id];
    if (!company) continue;
    for (const p of Object.values(company.products)) {
      if (p.status !== "live" || p.launchDay !== s.day) continue;
      for (const region of REGION_IDS) {
        for (const group of GROUP_IDS) {
          const a = awarenessOf(m, id, region, group);
          setAwareness(m, id, region, group, a + e.diffusion.launchBoost * (0.5 + p.hype));
        }
      }
    }
  }

  // Word of mouth that reaches neighbouring regions, applied after every cell's own.
  const spills: { id: string; region: RegionId; group: GroupId; spread: number }[] = [];

  for (const region of REGION_IDS) {
    for (const group of GROUP_IDS) {
      const g = e.groups[group];
      const people = adoptable(e, s.day, region, group);
      const options: { companyId: string; plan: PlanId; utility: number; satisfaction: number }[] =
        [];
      for (const id of companyIds) {
        const company = s.companies[id];
        if (!company) continue;
        for (const plan of PLAN_IDS) {
          if (plan === "business" && !g.buysBusiness) continue;
          const product = liveProductFor(company, plan);
          const version = product ? company.versions[product.versionId] : undefined;
          if (!product || !version) continue;
          const satisfaction = m.satisfaction[product.id]?.[group] ?? e.feedback.satisfactionStart;
          const price = plan === "free" ? 0 : company.plans[plan].price;
          const utility = productAppeal(
            e,
            group,
            version,
            plan,
            price,
            company.fame.reputation,
            satisfaction,
            company.compute.served,
          );
          options.push({ companyId: id, plan, utility, satisfaction });
        }
      }
      const shares = choiceShares(
        options.map((o) => o.utility),
        e.noAiUtility[group],
      );
      const offered = new Set(options.map((o) => `${o.companyId}|${o.plan}`));
      options.forEach((o, i) => {
        // Business customers are accounts, each paying for several seats.
        const perCustomer = o.plan === "business" ? g.seatsPerAccount : 1;
        const target = (people * (shares[i] ?? 0)) / perCustomer;
        let n = customersOf(m, o.companyId, region, group, o.plan);
        const aware = awarenessOf(m, o.companyId, region, group);
        n +=
          target > n
            ? (target - n) * e.diffusion.adoptRate * aware
            : (target - n) * e.diffusion.switchRate;
        n -= n * g.churnPerDay * (1 - o.satisfaction);
        setCustomers(m, o.companyId, region, group, o.plan, n);
      });
      // A plan that is no longer offered loses its customers gradually.
      for (const id of companyIds) {
        for (const plan of PLAN_IDS) {
          if (offered.has(`${id}|${plan}`)) continue;
          const n = customersOf(m, id, region, group, plan);
          if (n > 0) setCustomers(m, id, region, group, plan, n * (1 - e.diffusion.switchRate));
        }
      }
      // Word of mouth: happy customers make more people aware; awareness fades slowly.
      for (const id of companyIds) {
        const company = s.companies[id];
        if (!company) continue;
        const mine = PLAN_IDS.reduce(
          (sum, plan) => sum + customersOf(m, id, region, group, plan),
          0,
        );
        const a = awarenessOf(m, id, region, group);
        const spread = people > 0 ? (e.diffusion.wordOfMouth * mine) / people : 0;
        if (a > 0 || spread > 0)
          setAwareness(m, id, region, group, a + (1 - a) * spread - a * e.diffusion.awarenessDecay);
        if (spread > 0) spills.push({ id, region, group, spread });
      }
    }
  }

  for (const { id, region, group, spread } of spills) {
    for (const near of e.neighbors[region]) {
      const a = awarenessOf(m, id, near, group);
      setAwareness(m, id, near, group, a + (1 - a) * spread * e.diffusion.neighborSpill);
    }
  }
}
