import type { Command } from "@airace/protocol";
import { type Economy, getBalance } from "./balance.ts";
import { rngNext } from "./rng.ts";
import type { CompanyState, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

// The investment fund (playtest pack §5): money moved out of cash earns or loses a
// little at each month end. Moving money in or out is neither income nor a cost; a
// month's gain or loss is booked in that month but stays in the fund.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type FundCommand = Extract<Command, { type: "invest" | "withdraw" }>;

export function fundCommand(company: CompanyState, cmd: FundCommand): Handled {
  if (cmd.type === "invest") {
    if (cmd.amount > company.cash) {
      return no(`You have ${formatMoney(company.cash)} to invest.`);
    }
    company.cash -= cmd.amount;
    company.fund.balance += cmd.amount;
    return OK;
  }
  if (cmd.amount > company.fund.balance) {
    return no(`The fund holds ${formatMoney(company.fund.balance)}.`);
  }
  company.fund.balance -= cmd.amount;
  company.cash += cmd.amount;
  return OK;
}

/** This month's return as a share of the fund: usually a little up or down, sometimes a dip. */
function monthlyReturn(s: WorldState, f: Economy["fund"]): number {
  const a = rngNext(s.rng);
  const b = rngNext(a.next);
  const c = rngNext(b.next);
  s.rng = c.next;
  if (a.value < f.dipChance) return -(f.dipMin + b.value * (f.dipMax - f.dipMin));
  // A normal draw (Box-Muller) from the other two.
  const normal = Math.sqrt(-2 * Math.log(1 - b.value)) * Math.cos(2 * Math.PI * c.value);
  return f.meanMonthly + f.sdMonthly * normal;
}

/** At a month end, before the month closes: the fund's gain or loss, booked in that month. */
export function fundMonthEnd(s: WorldState, company: CompanyState): void {
  if (company.fund.balance <= 0) return;
  const f = getBalance(s.balance).economy.fund;
  const change = Math.round(company.fund.balance * monthlyReturn(s, f));
  if (change === 0) return;
  company.fund.balance += change;
  const m = company.ledger.current;
  if (change > 0) m.income.investmentGains += change;
  else m.costs.investmentLosses += -change;
}

/** Cash below zero at a month end is first covered from the fund, before any rescue. */
export function coverShortfall(company: CompanyState): void {
  if (company.cash >= 0 || company.fund.balance <= 0) return;
  const moved = Math.min(company.fund.balance, -company.cash);
  company.fund.balance -= moved;
  company.cash += moved;
}
