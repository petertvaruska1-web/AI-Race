import type { Goal } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { tellNews } from "./feed.ts";
import { leagueTables } from "./league.ts";
import { totalCustomers } from "./market.ts";
import type { CompanyState, WorldState } from "./state.ts";

// The optional goal (game-layer design §11), chosen when the world is created. The first
// player company to reach it wins a victory screen; play always goes on.

const grouped = (n: number) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/** The goal in plain words, as its story tells it. */
export function goalWords(goal: Goal): string {
  switch (goal.kind) {
    case "customers":
      return `${grouped(goal.target)} customers`;
    case "profit":
      return `$${grouped(goal.target)} profit in a month`;
    case "bestAi":
      return "the best AI overall";
    case "beatArch":
      return "more customers than the arch-rival";
    case "value":
      return `a company value of $${grouped(goal.target)}`;
    default:
      return "no goal";
  }
}

/**
 * What the company is worth to its founders: cash and the fund, less any loan, plus
 * `valueMonths` of its recent profit (the mean of the last three closed months, never
 * below 0), times the founders' share. Never below 0.
 */
export function companyValue(s: WorldState, company: CompanyState): number {
  const months = getBalance(s.balance).economy.goal.valueMonths;
  const recent = company.reports.slice(-3).map((r) => r.profit);
  const profit = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
  const worth =
    company.cash +
    company.fund.balance -
    (company.loan?.balance ?? 0) +
    months * Math.max(0, profit);
  return Math.round(company.ownership * Math.max(0, worth));
}

export type GoalProgress = { value: number; target: number | null; done: boolean };

/** How far a company is towards the world's goal (null when there is none). */
export function goalProgress(s: WorldState, company: CompanyState): GoalProgress {
  const goal = s.goal.spec.goal;
  switch (goal.kind) {
    case "customers": {
      const value = totalCustomers(s.market, company.id);
      return { value, target: goal.target, done: value >= goal.target };
    }
    case "profit": {
      const value = company.reports.at(-1)?.profit ?? 0;
      return { value, target: goal.target, done: value >= goal.target };
    }
    case "value": {
      const value = companyValue(s, company);
      return { value, target: goal.target, done: value >= goal.target };
    }
    case "bestAi": {
      const rows = leagueTables(s).overall;
      const rank = rows.findIndex((r) => r.companyId === company.id) + 1;
      const live = Object.values(company.products).some((p) => p.status === "live");
      const top = rows[0];
      const done = s.rivals !== null && live && top?.companyId === company.id && top.value !== null;
      return { value: rank, target: 1, done };
    }
    case "beatArch": {
      const value = totalCustomers(s.market, company.id);
      if (!s.rivals) return { value, target: null, done: false };
      const target = totalCustomers(s.market, s.rivals.archId);
      return { value, target, done: value > target };
    }
    default:
      return { value: 0, target: null, done: false };
  }
}

/**
 * The daily goal step, after the milestones: the first player company to reach the goal
 * wins it (once, even on the deadline day); a deadline that passes first is told once.
 */
export function goalDay(s: WorldState): void {
  const g = s.goal;
  if (g.spec.goal.kind === "none" || g.reachedDay !== null || g.missedDay !== null) return;
  const players = Object.keys(s.companies)
    .sort()
    .map((id) => s.companies[id] as CompanyState)
    .filter((c) => !c.rival);
  if (g.deadlineDay === null || s.day <= g.deadlineDay) {
    const winner = players.find((c) => goalProgress(s, c).done);
    if (winner) {
      g.reachedDay = s.day;
      g.reachedBy = winner.id;
      const headline = `${winner.name} reaches its goal: ${goalWords(g.spec.goal)}`;
      tellNews(s, winner, "goal", headline, 0.8);
    }
    return;
  }
  g.missedDay = s.day;
  for (const c of players) tellNews(s, c, "goal", `${c.name} misses its goal deadline`, -0.2);
}
