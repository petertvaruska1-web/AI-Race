import { GROUP_IDS, PLAN_IDS, REGION_IDS } from "@airace/protocol";
import { type Economy, getBalance } from "./balance.ts";
import { customersOf } from "./market.ts";
import { liveProductFor } from "./products.ts";
import type { CompanyState, WorldState } from "./state.ts";

// Reputation and followers (game-layer design §7). Reputation follows how happy
// the customers are and how the company's news reads; followers come from
// launches and viral posts, and fade slowly.

/** Moves a company's news mood toward a new story's sentiment. */
export function noteNews(economy: Economy, company: CompanyState, sentiment: number): void {
  const { fame } = company;
  fame.newsSentiment += economy.fame.newsAlpha * (sentiment - fame.newsSentiment);
}

/** Satisfaction averaged over the company's customers; the starting value while it has none. */
function customerSatisfaction(s: WorldState, economy: Economy, company: CompanyState): number {
  const start = economy.feedback.satisfactionStart;
  let weighted = 0;
  let total = 0;
  for (const plan of PLAN_IDS) {
    const product = liveProductFor(company, plan);
    if (!product) continue;
    for (const group of GROUP_IDS) {
      let n = 0;
      for (const region of REGION_IDS) n += customersOf(s.market, company.id, region, group, plan);
      weighted += n * (s.market.satisfaction[product.id]?.[group] ?? start);
      total += n;
    }
  }
  return total > 0 ? weighted / total : start;
}

/** The daily fame step: reputation drifts toward its target, and followers fade. */
export function fameDay(s: WorldState): void {
  const e = getBalance(s.balance).economy;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    const { fame } = company;
    const news = (fame.newsSentiment + 1) / 2;
    const target =
      100 *
      ((1 - e.fame.newsWeight) * customerSatisfaction(s, e, company) + e.fame.newsWeight * news);
    const next = fame.reputation + e.fame.reputationRate * (target - fame.reputation);
    fame.reputation = Math.min(100, Math.max(0, next));
    fame.followers -= fame.followers * e.fame.followersDecayPerDay;
  }
}
