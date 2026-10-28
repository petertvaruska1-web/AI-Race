import { GROUP_IDS, type GroupId, PLAN_IDS, REGION_IDS } from "@airace/protocol";
import { customersByRegion, customersOf } from "./market.ts";
import type { DaySample, WorldState } from "./state.ts";

/** Days of history kept per company: four game months, day by day (immersion design I3). */
export const HISTORY_DAYS = 120;

/** Records today's sample for every company, dropping days older than `HISTORY_DAYS`. */
export function historyDay(s: WorldState): void {
  for (const id of Object.keys(s.companies).sort()) {
    const c = s.companies[id];
    if (!c) continue;
    const customers = {} as Record<GroupId, number>;
    let paying = 0;
    for (const g of GROUP_IDS) {
      let n = 0;
      for (const region of REGION_IDS) {
        for (const plan of PLAN_IDS) {
          const k = customersOf(s.market, id, region, g, plan);
          n += k;
          if (plan !== "free") paying += k;
        }
      }
      customers[g] = Math.round(n);
    }
    const regions = customersByRegion(s.market, id);
    for (const r of REGION_IDS) regions[r] = Math.round(regions[r]);
    const sample: DaySample = {
      day: s.day,
      cash: c.cash,
      customers,
      regions,
      paying: Math.round(paying),
      followers: Math.round(c.fame.followers),
      reputation: Math.round(c.fame.reputation),
    };
    c.history.push(sample);
    if (c.history.length > HISTORY_DAYS) c.history.splice(0, c.history.length - HISTORY_DAYS);
  }
}
