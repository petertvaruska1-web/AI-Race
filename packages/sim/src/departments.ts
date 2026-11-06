import {
  type Command,
  DEPARTMENTS,
  type Department,
  GROUP_IDS,
  REGION_IDS,
} from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { book } from "./ledger.ts";
import { awarenessOf, setAwareness } from "./market.ts";
import type { CompanyState, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";
import { ownerAway } from "./world.ts";

// Departments (C43): four monthly budgets. Each day a company books its budgets as Team and
// each team grows (or shrinks) by 1/30 of the gap to its budget; a team's strength, with
// diminishing returns, scales its department's effect.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/** A department's strength, 0 to 1: `1 − exp(−staffed / scale)`. */
export function departmentStrength(
  c: Pick<CompanyState, "departments">,
  d: Department,
  scale = getBalance("balance-v2").economy.departments.scale,
): number {
  const staffed = c.departments?.[d]?.staffed ?? 0;
  return staffed > 0 ? 1 - Math.exp(-staffed / scale) : 0;
}

/** A company's strengths in a world, by its balance. */
export function strengthsOf(s: WorldState, c: CompanyState): Record<Department, number> {
  const { scale } = getBalance(s.balance).economy.departments;
  return Object.fromEntries(DEPARTMENTS.map((d) => [d, departmentStrength(c, d, scale)])) as Record<
    Department,
    number
  >;
}

export function departmentCommand(
  s: WorldState,
  company: CompanyState,
  cmd: Extract<Command, { type: "setDepartment" }>,
): Handled {
  const b = getBalance(s.balance).economy.departments;
  if (!(DEPARTMENTS as readonly string[]).includes(cmd.department)) {
    return no("That request isn't valid.");
  }
  const budget = cmd.budget;
  if (!Number.isInteger(budget) || budget < 0 || budget > b.maxBudget || budget % b.step !== 0) {
    return no(
      `Budgets are $0–${formatMoney(b.maxBudget)} a month, in ${formatMoney(b.step)} steps.`,
    );
  }
  company.departments[cmd.department].budget = budget;
  return OK;
}

/**
 * A day of every company's departments. Call before the campaigns and every effect: it books
 * the day's budgets (`budget / 30` each) and moves each team toward its budget. An absent
 * owner's budgets pause on a day the company has no cash: nothing is spent and the teams shrink.
 */
export function departmentsDay(s: WorldState): void {
  const { growDays } = getBalance(s.balance).economy.departments;
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company?.departments) continue;
    const paused = company.cash < 0 && ownerAway(s, company);
    let spend = 0;
    for (const d of DEPARTMENTS) {
      const dept = company.departments[d];
      const target = paused ? 0 : dept.budget;
      if (!paused) spend += dept.budget / 30;
      dept.staffed += (target - dept.staffed) / growDays;
      if (dept.staffed < 1 && target === 0) dept.staffed = 0;
    }
    book(company, "team", spend);
    salesDay(s, company);
  }
}

/**
 * Sales & marketing, after the teams grow: in every region and group, awareness rises by
 * `sales · s · (1 − awareness)` a day, once the company has an AI on sale.
 */
function salesDay(s: WorldState, company: CompanyState): void {
  const e = getBalance(s.balance).economy.departments;
  const strength = departmentStrength(company, "sales", e.scale);
  if (strength <= 0) return;
  if (!Object.values(company.products).some((p) => p.status === "live")) return;
  for (const region of REGION_IDS) {
    for (const group of GROUP_IDS) {
      const a = awarenessOf(s.market, company.id, region, group);
      setAwareness(s.market, company.id, region, group, a + e.sales * strength * (1 - a));
    }
  }
}
