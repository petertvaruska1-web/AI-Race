import {
  BENCH_CATEGORIES,
  type BenchCategory,
  LEAGUE_TABLES,
  type LeagueTable,
} from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { SKILL_WORDS, tellNews } from "./feed.ts";
import { totalCustomers } from "./market.ts";
import type { CompanyState, WorldState } from "./state.ts";

// League tables (game-layer design §7 "Fame"): every company ranked by customers,
// last month's revenue, overall score and each skill, all from measurements.

export type LeagueRow = { companyId: string; value: number | null };

/** The tables whose new #1 makes news; revenue stays private (R2). */
const NEWS_TABLES: readonly LeagueTable[] = ["customers", "overall", ...BENCH_CATEGORIES];

/** The report card's three tables. */
export const CARD_TABLES = ["customers", "revenue", "overall"] as const;

function liveVersions(c: CompanyState) {
  return Object.values(c.products)
    .filter((p) => p.status === "live")
    .map((p) => c.versions[p.versionId])
    .filter((v) => v !== undefined);
}

function best(values: (number | null | undefined)[]): number | null {
  const known = values.filter((v): v is number => typeof v === "number");
  return known.length > 0 ? Math.max(...known) : null;
}

function tableValue(s: WorldState, c: CompanyState, table: LeagueTable): number | null {
  if (table === "customers") return totalCustomers(s.market, c.id);
  if (table === "revenue") {
    const income = c.ledger.months.at(-1)?.income;
    return income ? income.personal + income.business : 0;
  }
  const versions = liveVersions(c);
  if (table === "overall") return best(versions.map((v) => v.overall));
  return best(versions.map((v) => v.scores?.[table]));
}

/** Every table, best first; ties go to the lower company id; no value comes last. */
export function leagueTables(s: WorldState): Record<LeagueTable, LeagueRow[]> {
  const ids = Object.keys(s.companies).sort();
  const out = {} as Record<LeagueTable, LeagueRow[]>;
  for (const table of LEAGUE_TABLES) {
    const rows = ids.map((id) => ({
      companyId: id,
      value: tableValue(s, s.companies[id] as CompanyState, table),
    }));
    rows.sort((a, b) => {
      if (a.value === null || b.value === null) return a.value === null ? 1 : -1;
      return b.value - a.value || a.companyId.localeCompare(b.companyId);
    });
    out[table] = rows;
  }
  return out;
}

/** A company's place (from 1) in each of the report card's tables. */
export function cardRanks(
  s: WorldState,
  companyId: string,
): Record<(typeof CARD_TABLES)[number], number> {
  const tables = leagueTables(s);
  const place = (t: LeagueTable) => tables[t].findIndex((r) => r.companyId === companyId) + 1;
  return { customers: place("customers"), revenue: place("revenue"), overall: place("overall") };
}

function headlineFor(company: CompanyState, table: LeagueTable): string {
  if (table === "customers") return `${company.name} now has the most customers`;
  if (table === "overall") return `${company.name} now has the best AI overall`;
  return `${company.name} takes #1 in ${SKILL_WORDS[table as BenchCategory]}`;
}

/**
 * The daily league step: records each table's leader. With rivals in the world, a
 * change of leader is news, at most one story a day and none for the same table
 * within `newsGapDays` (C13: rare and meaningful). A first leader is not a change.
 */
export function leagueDay(s: WorldState): void {
  const gap = getBalance(s.balance).economy.league.newsGapDays;
  const tables = leagueTables(s);
  let told = false;
  for (const table of NEWS_TABLES) {
    const top = tables[table][0];
    if (!top || top.value === null || top.value <= 0) continue;
    const before = s.leaders[table];
    if (before === top.companyId) continue;
    s.leaders[table] = top.companyId;
    if (before === undefined || !s.rivals || told) continue;
    const last = s.lastRankingNews[table];
    if (last !== undefined && s.day - last < gap) continue;
    const company = s.companies[top.companyId] as CompanyState;
    tellNews(s, company, "ranking", headlineFor(company, table), 0.4);
    s.lastRankingNews[table] = s.day;
    told = true;
  }
}
