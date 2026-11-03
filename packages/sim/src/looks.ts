import type { Command } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { book } from "./ledger.ts";
import type { CompanyState, WorldState } from "./state.ts";
import { formatMoney } from "./training.ts";

// Looks (playtest pack §5): orb styles and logo motions, bought once and worn at will.
// They change only how the company looks; the orb still shows real measurements.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

type LookCommand = Extract<Command, { type: "buyLook" | "useLook" }>;

function wear(company: CompanyState, lookId: string): void {
  if (lookId.startsWith("orb:")) company.looks.orb = lookId;
  else company.looks.logo = lookId;
}

export function lookCommand(s: WorldState, company: CompanyState, cmd: LookCommand): Handled {
  const price = getBalance(s.balance).economy.looks[cmd.lookId];
  if (price === undefined) return no("There's no such look.");
  const owned = company.looks.owned.includes(cmd.lookId);
  if (cmd.type === "useLook") {
    if (!owned) return no("Buy it first.");
    wear(company, cmd.lookId);
    return OK;
  }
  if (owned) return no("You already own it.");
  if (company.cash < price) {
    return no(`That look costs ${formatMoney(price)}. You have ${formatMoney(company.cash)}.`);
  }
  book(company, "looks", price);
  company.looks.owned.push(cmd.lookId);
  wear(company, cmd.lookId);
  return OK;
}
