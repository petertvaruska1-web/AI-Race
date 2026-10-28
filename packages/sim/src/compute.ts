import {
  type Command,
  GROUP_IDS,
  PLAN_IDS,
  REGION_IDS,
  REGION_PLACES,
  type RegionId,
} from "@airace/protocol";
import { type Balance, type Economy, getBalance } from "./balance.ts";
import { tellNews } from "./feed.ts";
import { book } from "./ledger.ts";
import { customersOf } from "./market.ts";
import { liveProductFor } from "./products.ts";
import { paramCount } from "./shape.ts";
import type { CompanyState, ComputeState, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

// Serving capacity (game-layer design §8), in PFLOP a game day. Every customer,
// free ones too, uses compute each day in step with the size of the AI behind
// their plan. Owned data centers serve first; the cloud rents the rest, but it
// can only grow by a share a day, so a sudden spike leaves some questions
// unanswered: the company is less appealing and its customers less satisfied.

export function newCompute(b: Balance): ComputeState {
  return {
    cloudPflops: b.economy.compute.cloudStartPflops,
    trainingShare: 0,
    loadPflops: 0,
    served: 1,
    lastOverloadNews: null,
  };
}

/** Today's serving: the compute it needs, and what that costs at cloud prices. */
function demand(s: WorldState, companyId: string): { pflops: number; dollars: number } {
  const b = getBalance(s.balance);
  const e = b.economy;
  const company = s.companies[companyId];
  if (!company) return { pflops: 0, dollars: 0 };
  let dollars = 0;
  for (const plan of PLAN_IDS) {
    const product = liveProductFor(company, plan);
    const version = product ? company.versions[product.versionId] : undefined;
    // A closed plan's customers are leaving: they use no compute.
    if (!version) continue;
    const millions = paramCount(version.nLayer, version.dModel) / 1e6;
    for (const group of GROUP_IDS) {
      // A business account uses compute for each of its seats.
      const seats = plan === "business" ? e.groups[group].seatsPerAccount : 1;
      let n = 0;
      for (const region of REGION_IDS) n += customersOf(s.market, companyId, region, group, plan);
      dollars += n * seats * e.servingCostPerCustomerDayPerMParam * millions;
    }
  }
  return { pflops: dollars / b.cloudPricePerPflop, dollars };
}

/** The compute (PFLOP) the company's customers need today. */
export function servingLoad(s: WorldState, companyId: string): number {
  return demand(s, companyId).pflops;
}

/** The company's open data-center capacity (PFLOP a day). */
export function ownedPflops(company: CompanyState, day: number): number {
  let pflops = 0;
  for (const dc of Object.values(company.dataCenters)) {
    if (dc.openDay <= day) pflops += dc.pflopsPerDay;
  }
  return pflops;
}

/** How close a region is to the company's open data centers: 1 in it, less next door, else 0. */
export function closeness(
  company: CompanyState,
  region: RegionId,
  day: number,
  economy: Economy,
): number {
  let best = 0;
  for (const dc of Object.values(company.dataCenters)) {
    if (dc.openDay > day) continue;
    if (dc.region === region) return economy.nearby.region;
    if (economy.neighbors[dc.region].includes(region)) best = economy.nearby.neighbor;
  }
  return best;
}

type ComputeCommand = Extract<Command, { type: "buildDataCenter" | "setTrainingShare" }>;

export function computeCommand(
  s: WorldState,
  company: CompanyState,
  cmd: ComputeCommand,
  newId: string | undefined,
): Handled {
  if (cmd.type === "setTrainingShare") {
    if (!Number.isFinite(cmd.share) || cmd.share < 0 || cmd.share > 1) {
      return no("Choose a share from 0% to 100%.");
    }
    company.compute.trainingShare = Math.round(cmd.share * 100) / 100;
    return OK;
  }
  const dc = getBalance(s.balance).economy.dataCenter;
  if (s.phase !== "running") return no("The world is not running yet.");
  if (!newId) return no("Missing data center id.");
  if (company.dataCenters[newId]) return no("That data center id is already used.");
  if (Object.keys(company.dataCenters).length >= dc.maxPerCompany) {
    return no(`You can own at most ${dc.maxPerCompany} data centers.`);
  }
  if (company.cash < dc.cost) {
    return no(
      `A data center costs ${formatMoney(dc.cost)}. You have ${formatMoney(company.cash)}.`,
    );
  }
  book(company, "dataCenterBuild", dc.cost);
  company.dataCenters[newId] = {
    id: newId,
    region: cmd.region,
    startDay: s.day,
    openDay: s.day + dc.buildDays,
    pflopsPerDay: dc.pflopsPerDay,
  };
  return OK;
}

/** Data centers that open today make the news; every open one costs a day of upkeep. Call before `computeDay`. */
export function dataCentersDay(s: WorldState): void {
  const dc = getBalance(s.balance).economy.dataCenter;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    for (const key of Object.keys(company.dataCenters).sort()) {
      const center = company.dataCenters[key];
      if (!center || center.openDay > s.day) continue;
      if (center.openDay === s.day) {
        const headline = `${company.name} opens a data center in ${REGION_PLACES[center.region]}`;
        tellNews(s, company, "dataCenter", headline, 0.5);
      }
      book(company, "dataCenterUpkeep", dc.upkeepPerMonth / 30);
    }
  }
}

/**
 * The daily serving step, after the market: rents the cloud room needed (it
 * grows at most `cloudGrowthPerDay` a day), books serving, and when some
 * questions go unanswered lowers satisfaction and, now and then, makes the news.
 */
export function computeDay(s: WorldState): void {
  const c = getBalance(s.balance).economy.compute;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    const { pflops: load, dollars } = demand(s, id);
    const owned = ownedPflops(company, s.day) * (1 - company.compute.trainingShare);
    const need = Math.max(0, load - owned);
    const room = Math.max(
      c.cloudStartPflops,
      company.compute.cloudPflops * (1 + c.cloudGrowthPerDay),
    );
    const cloud = Math.min(need, room);
    const served = load > 0 ? Math.min(1, (owned + cloud) / load) : 1;
    // The cloud's share of the load costs cloud prices; owned centers' share a fraction of them.
    const fromCloud = load > 0 ? Math.min(cloud, need) / load : 0;
    const fromOwned = load > 0 ? Math.min(owned, load) / load : 0;
    book(company, "serving", dollars * (fromCloud + fromOwned * c.ownedServingPriceShare));
    company.compute.cloudPflops = cloud;
    company.compute.loadPflops = load;
    company.compute.served = served;
    if (served >= 1) continue;

    const start = getBalance(s.balance).economy.feedback.satisfactionStart;
    for (const productId of Object.keys(company.products).sort()) {
      const sat = s.market.satisfaction[productId] ?? {};
      for (const group of GROUP_IDS) {
        const now = sat[group] ?? start;
        sat[group] = Math.max(0, now - (1 - served) * c.overloadSatisfactionHit);
      }
      s.market.satisfaction[productId] = sat;
    }
    const last = company.compute.lastOverloadNews;
    if (served < c.overloadNewsBelow && (last === null || s.day - last >= c.overloadNewsGapDays)) {
      company.compute.lastOverloadNews = s.day;
      const headline = `${company.name}'s AI buckles under demand: only ${Math.round(served * 100)}% of questions answered`;
      tellNews(s, company, "overload", headline, -0.5);
    }
  }
}
