import type { Command } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { tellNews } from "./feed.ts";
import { book, finance } from "./ledger.ts";
import type { CompanyState, WorldState } from "./state.ts";
import { trashTalk } from "./trashTalk.ts";

// The rescue (game-layer design §8, C19). A month that ends with cash below zero
// opens an offer: an emergency loan with heavy interest, or an investor who takes
// a share of the company. Without a choice the game takes the loan when the offer
// runs out. A company cannot otherwise die.

type Handled = { ok: true } | { ok: false; reason: string };
const OK: Handled = { ok: true };
const no = (reason: string): Handled => ({ ok: false, reason });

/** Restores cash to the cushion by loan or investor, and closes the offer. */
function rescue(s: WorldState, company: CompanyState, kind: "loan" | "investor"): void {
  const r = getBalance(s.balance).economy.rescue;
  company.rescue = null;
  const amount = Math.max(0, Math.round(r.cushion - company.cash));
  if (amount === 0) return;
  if (!company.rival) trashTalk(s, "playerRescued", { player: company.name });
  if (kind === "loan") {
    finance(company, "loans", amount);
    const balance = (company.loan?.balance ?? 0) + amount;
    company.loan = { balance, principalPerMonth: Math.ceil(balance / r.loanTermMonths) };
    tellNews(s, company, "rescue", `${company.name} takes an emergency loan`, -0.3);
  } else {
    finance(company, "investors", amount);
    company.ownership = Math.round(company.ownership * (1 - r.investorStake) * 1e4) / 1e4;
    const stake = Math.round(r.investorStake * 100);
    tellNews(s, company, "rescue", `Investors take a ${stake}% stake in ${company.name}`, -0.2);
  }
}

export function rescueCommand(
  s: WorldState,
  company: CompanyState,
  cmd: Extract<Command, { type: "chooseRescue" }>,
): Handled {
  if (!company.rescue) return no("There is no rescue to choose.");
  rescue(s, company, cmd.kind);
  return OK;
}

/**
 * At a month close, after the report card: the loan's interest and repayment
 * (booked in the new month), then an offer if cash is below zero.
 */
export function rescueMonth(s: WorldState, company: CompanyState): void {
  const r = getBalance(s.balance).economy.rescue;
  const loan = company.loan;
  if (loan) {
    book(company, "interest", loan.balance * r.loanMonthlyRate);
    const repaid = Math.min(loan.balance, loan.principalPerMonth);
    finance(company, "repaid", repaid);
    loan.balance -= repaid;
    if (loan.balance <= 0) company.loan = null;
  }
  if (company.cash < 0 && !company.rescue) {
    company.rescue = { offeredDay: s.day, deadlineDay: s.day + r.decideDays };
    tellNews(s, company, "rescue", `${company.name} runs out of cash`, -0.6);
  }
}

/** An offer that runs out takes the loan if cash is still below zero, else lapses. Call after the ledger. */
export function rescueDay(s: WorldState): void {
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company?.rescue || s.day < company.rescue.deadlineDay) continue;
    if (company.cash < 0) rescue(s, company, "loan");
    else company.rescue = null;
  }
}
