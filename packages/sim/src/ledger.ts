import {
  COST_CATEGORIES,
  type CostCategory,
  GROUP_IDS,
  type GroupId,
  INCOME_CATEGORIES,
  type IncomeCategory,
  PLAN_IDS,
  type PlanId,
  REGION_IDS,
  type RegionId,
} from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { monthOfDay } from "./calendar.ts";
import { customersOf, totalCustomers } from "./market.ts";
import { liveProductFor } from "./products.ts";
import { reportMonth } from "./reports.ts";
import type { CompanyState, LedgerState, MonthLedger, RunState, WorldState } from "./state.ts";

// The money ledger (game-layer design §8). Every change of cash is booked here
// in whole dollars, so cash always equals founding cash plus income less costs,
// plus financing (rescue loans and investors in, repayments out).

const MONTHS_KEPT = 24;
/** Prices are monthly; each day bills a 30th. */
const BILLING_DAYS = 30;

function zeros<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((k) => [k, 0])) as Record<K, number>;
}

export function emptyMonth(month: string): MonthLedger {
  return {
    month,
    income: zeros(INCOME_CATEGORIES),
    costs: zeros(COST_CATEGORIES),
    revenueByGroup: zeros(GROUP_IDS),
    revenueByPlan: zeros(PLAN_IDS),
    revenueByRegion: zeros(REGION_IDS),
    customersEnd: 0,
    customersEndByGroup: zeros(GROUP_IDS),
    financing: { loans: 0, investors: 0, repaid: 0 },
  };
}

/** A new company's ledger, open on the month of `day`. */
export function newLedger(day: number): LedgerState {
  return { current: emptyMonth(monthOfDay(day)), months: [] };
}

const isIncome = (c: IncomeCategory | CostCategory): c is IncomeCategory =>
  (INCOME_CATEGORIES as readonly string[]).includes(c);

/**
 * Books an amount (negative for a refund) into the open month. Rounds once,
 * then changes cash and the month together.
 */
export function book(
  company: CompanyState,
  category: IncomeCategory | CostCategory,
  amount: number,
  detail: { group?: GroupId; plan?: PlanId; region?: RegionId } = {},
): void {
  const dollars = Math.round(amount);
  if (dollars === 0) return;
  const m = company.ledger.current;
  if (isIncome(category)) {
    company.cash += dollars;
    m.income[category] += dollars;
    if (detail.group) m.revenueByGroup[detail.group] += dollars;
    if (detail.plan) m.revenueByPlan[detail.plan] += dollars;
    if (detail.region) m.revenueByRegion[detail.region] += dollars;
  } else {
    company.cash -= dollars;
    m.costs[category] += dollars;
  }
}

/**
 * Books money that is neither income nor a cost: a rescue loan or an investor's
 * money in, a loan repayment out. Rounds once, like `book`.
 */
export function finance(
  company: CompanyState,
  kind: "loans" | "investors" | "repaid",
  amount: number,
): void {
  const dollars = Math.round(amount);
  if (dollars === 0) return;
  company.cash += kind === "repaid" ? -dollars : dollars;
  company.ledger.current.financing[kind] += dollars;
}

/** Books a run's price (sign 1) or its refund (sign -1); fact-checking goes under data checks. */
export function bookTraining(
  s: WorldState,
  company: CompanyState,
  run: Pick<RunState, "plan" | "preview">,
  sign: 1 | -1,
): void {
  const checks = run.plan.factCheck ? getBalance(s.balance).factCheckCost : 0;
  book(company, "training", sign * (run.preview.cost - checks));
  book(company, "dataChecks", sign * checks);
}

/** A day of revenue for every company (serving is booked by `computeDay`). Call after the market step. */
export function ledgerDay(s: WorldState): void {
  const e = getBalance(s.balance).economy;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    for (const plan of PLAN_IDS) {
      const product = liveProductFor(company, plan);
      const version = product ? company.versions[product.versionId] : undefined;
      // A closed plan's customers are leaving: they no longer pay.
      if (!version || plan === "free") continue;
      for (const group of GROUP_IDS) {
        // A business account pays for each of its seats.
        const seats = plan === "business" ? e.groups[group].seatsPerAccount : 1;
        for (const region of REGION_IDS) {
          const users = customersOf(s.market, id, region, group, plan) * seats;
          const amount = (users * company.plans[plan].price) / BILLING_DAYS;
          book(company, plan, amount, { group, plan, region });
        }
      }
    }
  }
}

/** Closes the open month with its closing customers (in all and by group), and opens `month`. */
export function closeMonth(
  company: CompanyState,
  month: string,
  customers: number,
  byGroup: Record<GroupId, number>,
): void {
  const ledger = company.ledger;
  ledger.current.customersEnd = Math.round(customers);
  for (const g of GROUP_IDS) ledger.current.customersEndByGroup[g] = Math.round(byGroup[g]);
  ledger.months.push(ledger.current);
  if (ledger.months.length > MONTHS_KEPT)
    ledger.months.splice(0, ledger.months.length - MONTHS_KEPT);
  ledger.current = emptyMonth(month);
}

/** On the first day of a game month, closes every company's previous month. Call first in the tick. */
export function closeMonths(s: WorldState): void {
  const month = monthOfDay(s.day);
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (company && company.ledger.current.month !== month) {
      const byGroup = zeros(GROUP_IDS);
      for (const region of REGION_IDS) {
        for (const g of GROUP_IDS) {
          for (const plan of PLAN_IDS) byGroup[g] += customersOf(s.market, id, region, g, plan);
        }
      }
      closeMonth(company, month, totalCustomers(s.market, id), byGroup);
      const closed = company.ledger.months.at(-1);
      if (closed) reportMonth(s, company, closed, company.ledger.months.at(-2));
    }
  }
}
