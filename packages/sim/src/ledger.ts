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
} from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { monthOfDay } from "./calendar.ts";
import { customersOf, totalCustomers } from "./market.ts";
import { liveProductFor } from "./products.ts";
import { paramCount } from "./shape.ts";
import type { CompanyState, LedgerState, MonthLedger, RunState, WorldState } from "./state.ts";

// The money ledger (game-layer design §8). Every change of cash is booked here
// in whole dollars, so cash always equals founding cash plus income less costs.

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
    customersEnd: 0,
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
  detail: { group?: GroupId; plan?: PlanId } = {},
): void {
  const dollars = Math.round(amount);
  if (dollars === 0) return;
  const m = company.ledger.current;
  if (isIncome(category)) {
    company.cash += dollars;
    m.income[category] += dollars;
    if (detail.group) m.revenueByGroup[detail.group] += dollars;
    if (detail.plan) m.revenueByPlan[detail.plan] += dollars;
  } else {
    company.cash -= dollars;
    m.costs[category] += dollars;
  }
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

/** A day of revenue and serving costs for every company. Call after the market step. */
export function ledgerDay(s: WorldState): void {
  const e = getBalance(s.balance).economy;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company) continue;
    let serving = 0;
    for (const plan of PLAN_IDS) {
      const product = liveProductFor(company, plan);
      const version = product ? company.versions[product.versionId] : undefined;
      // A closed plan's customers are leaving: they neither pay nor use compute.
      if (!version) continue;
      const millions = paramCount(version.nLayer, version.dModel) / 1e6;
      for (const group of GROUP_IDS) {
        let n = 0;
        for (const region of REGION_IDS) n += customersOf(s.market, id, region, group, plan);
        // A business account pays, and uses compute, for each of its seats.
        const users = plan === "business" ? n * e.groups[group].seatsPerAccount : n;
        serving += users * e.servingCostPerCustomerDayPerMParam * millions;
        if (plan !== "free") {
          book(company, plan, (users * company.plans[plan].price) / BILLING_DAYS, { group, plan });
        }
      }
    }
    book(company, "serving", serving);
  }
}

/** Closes the open month with its closing customer count, and opens `month`. */
export function closeMonth(company: CompanyState, month: string, customers: number): void {
  const ledger = company.ledger;
  ledger.current.customersEnd = Math.round(customers);
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
      closeMonth(company, month, totalCustomers(s.market, id));
    }
  }
}
