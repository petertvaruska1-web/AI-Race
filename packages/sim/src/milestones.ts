import { BENCH_CATEGORIES, type EvalSignals, type MilestoneId } from "@airace/protocol";
import { payingCustomers, SKILL_WORDS } from "./feed.ts";
import { totalCustomers } from "./market.ts";
import type { CompanyState, VersionState, WorldState } from "./state.ts";

// Milestones (game-layer design §11) for the trophy shelf. Each is earned once, only from
// a measurement or the company's state (R5): nothing is listed before it is earned.

const CUSTOMER_MILESTONES: [number, MilestoneId][] = [
  [1_000, "customers1k"],
  [100_000, "customers100k"],
  [1_000_000, "customers1m"],
];

/** Records a milestone once; rivals earn none. */
export function noteMilestone(
  s: WorldState,
  company: CompanyState,
  id: MilestoneId,
  detail: string | null,
): void {
  if (company.rival || company.milestones.some((m) => m.id === id)) return;
  company.milestones.push({ id, day: s.day, detail });
}

/** What one eval measured: only a measurement that was made can earn a milestone. */
export function evalMilestones(
  s: WorldState,
  company: CompanyState,
  version: VersionState,
  signals: EvalSignals | null | undefined,
): void {
  if (!signals) return;
  if (signals.legibleReplies !== null && signals.legibleReplies >= 1) {
    noteMilestone(s, company, "firstSentence", version.label);
  }
  if (signals.correctFacts >= 1) noteMilestone(s, company, "firstFact", version.label);
  if (signals.passingPrograms >= 1) noteMilestone(s, company, "firstProgram", version.label);
}

/** The daily step, after the league: milestones read from each player company's state. */
export function milestonesDay(s: WorldState): void {
  for (const id of Object.keys(s.companies).sort()) {
    const company = s.companies[id];
    if (!company || company.rival) continue;
    const customers = totalCustomers(s.market, id);
    for (const [mark, milestone] of CUSTOMER_MILESTONES) {
      if (customers >= mark) noteMilestone(s, company, milestone, null);
    }
    if (payingCustomers(s, id) >= 1) noteMilestone(s, company, "firstPaying", null);
    if (s.rivals) {
      const skill = BENCH_CATEGORIES.find((table) => s.leaders[table] === id);
      if (skill) noteMilestone(s, company, "numberOne", SKILL_WORDS[skill]);
      const archId = s.rivals.archId;
      const won = company.battles.find((b) => b.rivalId === archId && b.outcome === "win");
      if (won) noteMilestone(s, company, "beatArch", won.theirs?.productName ?? null);
    }
    if (Object.values(company.dataCenters).some((d) => d.openDay <= s.day)) {
      noteMilestone(s, company, "dataCenter", null);
    }
    const profitable = company.reports.find((r) => r.profit > 0);
    if (profitable) noteMilestone(s, company, "firstProfit", profitable.month);
    const a = company.reports.find((r) => r.grade === "A");
    if (a) noteMilestone(s, company, "gradeA", a.month);
  }
}
