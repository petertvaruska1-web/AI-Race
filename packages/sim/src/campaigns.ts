import { type Command, GROUP_IDS, REGION_IDS } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { book } from "./ledger.ts";
import { adoptable, awarenessOf, setAwareness } from "./market.ts";
import type { CompanyState, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

// Ad campaigns (game-layer design §5): a daily budget aimed at a region (or the
// whole country) and a group (or everyone) for a number of days. Each day it
// books its budget as marketing and makes more of the people it aims at aware
// of the company; during a launch countdown it also builds hype.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type CampaignCommand = Extract<Command, { type: "startCampaign" | "stopCampaign" }>;

export function campaignCommand(
  s: WorldState,
  company: CompanyState,
  cmd: CampaignCommand,
  newId: string | undefined,
): Handled {
  const c = getBalance(s.balance).economy.campaigns;
  if (cmd.type === "stopCampaign") {
    if (!company.campaigns[cmd.campaignId]) return no("That campaign isn't running.");
    delete company.campaigns[cmd.campaignId];
    return OK;
  }
  if (s.phase !== "running") return no("The world is not running yet.");
  if (!newId) return no("Missing campaign id.");
  if (company.campaigns[newId]) return no("That campaign id is already used.");
  const selling = Object.values(company.products).some((p) => p.status !== "retired");
  if (!selling) return no("Launch an AI before advertising it.");
  if (cmd.budgetPerDay < c.budget.min || cmd.budgetPerDay > c.budget.max) {
    return no(
      `A campaign's daily budget must be ${formatMoney(c.budget.min)}–${formatMoney(c.budget.max)}.`,
    );
  }
  if (cmd.days < c.days.min || cmd.days > c.days.max) {
    return no(`A campaign runs ${c.days.min}–${c.days.max} game days.`);
  }
  if (Object.keys(company.campaigns).length >= c.maxRunning) {
    return no(`You can run up to ${c.maxRunning} campaigns at once.`);
  }
  if (company.cash < cmd.budgetPerDay) {
    return no(`You need at least ${formatMoney(cmd.budgetPerDay)} for a day of this campaign.`);
  }
  company.campaigns[newId] = {
    id: newId,
    region: cmd.region,
    group: cmd.group,
    budgetPerDay: cmd.budgetPerDay,
    startDay: s.day,
    endDay: s.day + cmd.days,
    spent: 0,
  };
  return OK;
}

/**
 * A day of every running campaign. Call before the products and the market:
 * it books the day's budget, raises awareness where it aims, adds hype to
 * countdowns, and ends campaigns on their last day. A campaign spends on the
 * ticks after the day it started, `days` times.
 */
export function campaignsDay(s: WorldState): void {
  const e = getBalance(s.balance).economy;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    for (const key of Object.keys(company.campaigns).sort()) {
      const k = company.campaigns[key];
      if (!k || s.day <= k.startDay) continue;
      book(company, "marketing", k.budgetPerDay);
      k.spent += k.budgetPerDay;
      const regions = k.region === "all" ? REGION_IDS : [k.region];
      const groups = k.group === "all" ? GROUP_IDS : [k.group];
      const cells = regions.flatMap((region) =>
        groups.map((group) => ({ region, group, people: adoptable(e, s.day, region, group) })),
      );
      const people = cells.reduce((n, cell) => n + cell.people, 0);
      // The reach is spread over the cells by their people, so each gets the same share.
      const share =
        people > 0 ? Math.min(1, k.budgetPerDay / e.campaigns.costPerReach / people) : 0;
      for (const cell of cells) {
        if (cell.people <= 0) continue;
        const a = awarenessOf(s.market, id, cell.region, cell.group);
        setAwareness(
          s.market,
          id,
          cell.region,
          cell.group,
          a + (1 - a) * share * e.campaigns.awarenessPerReach,
        );
      }
      for (const productId of Object.keys(company.products).sort()) {
        const p = company.products[productId];
        if (p?.status === "countdown") p.hype += k.budgetPerDay * e.campaigns.hypePerDollar;
      }
      if (s.day >= k.endDay) delete company.campaigns[key];
    }
  }
}
