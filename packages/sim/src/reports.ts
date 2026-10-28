import { GROUP_IDS, type GroupId, PLAN_IDS, REGION_IDS } from "@airace/protocol";
import { getBalance } from "./balance.ts";
import { monthOfDay } from "./calendar.ts";
import { customersOf } from "./market.ts";
import { liveProductFor } from "./products.ts";
import type {
  CompanyState,
  MomentPost,
  MonthLedger,
  PostItem,
  ReportCard,
  WorldState,
} from "./state.ts";

// The monthly report card (game-layer design §8): at each month end after the
// company's first launch, its money against last month, customer growth by
// group, how satisfied customers are, the month's best and worst posts (quoted
// verbatim, R1) and its biggest story, and a grade from A to F.

const REPORTS_KEPT = 24;
const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

/**
 * The grade, from points: profit (1 above zero, 1 more above last month's, or
 * above zero again when there was no last month) + growth (1 for 10% or more,
 * or from none to some; 0.5 for any) + mood (1 at 0.6 or more; 0.5 at 0.45).
 */
export function gradeFor(m: {
  profit: number;
  lastProfit: number | null;
  start: number;
  end: number;
  mood: number | null;
}): ReportCard["grade"] {
  let points = m.profit > 0 ? 1 : 0;
  if (m.lastProfit === null ? m.profit > 0 : m.profit > m.lastProfit) points += 1;
  if (m.start > 0) {
    if (m.end >= m.start * 1.1) points += 1;
    else if (m.end > m.start) points += 0.5;
  } else if (m.end > 0) points += 1;
  if (m.mood !== null) {
    if (m.mood >= 0.6) points += 1;
    else if (m.mood >= 0.45) points += 0.5;
  }
  if (points >= 3.5) return "A";
  if (points >= 2.5) return "B";
  if (points >= 1.5) return "C";
  if (points >= 0.5) return "D";
  return "F";
}

/** Customer-weighted satisfaction over the company's live plans (null with no customers). */
function moodOf(s: WorldState, company: CompanyState): number | null {
  const start = getBalance(s.balance).economy.feedback.satisfactionStart;
  let weighted = 0;
  let people = 0;
  for (const plan of PLAN_IDS) {
    const product = liveProductFor(company, plan);
    if (!product) continue;
    for (const group of GROUP_IDS) {
      let n = 0;
      for (const region of REGION_IDS) n += customersOf(s.market, company.id, region, group, plan);
      weighted += n * (s.market.satisfaction[product.id]?.[group] ?? start);
      people += n;
    }
  }
  return people > 0 ? weighted / people : null;
}

function moment(p: PostItem): MomentPost {
  return {
    author: { name: p.author.name, handle: p.author.handle, group: p.author.group },
    productName: p.productName,
    question: p.question,
    reply: p.reply,
    tone: p.tone,
  };
}

/**
 * Makes the card for `closed` once the company has launched a product before
 * the close. Call at the month close, after the ledger moved `closed` into its
 * history; `previous` is the month before it, if any.
 */
export function reportMonth(
  s: WorldState,
  company: CompanyState,
  closed: MonthLedger,
  previous: MonthLedger | undefined,
): void {
  const launched = Object.values(company.products).some(
    (p) => p.status !== "countdown" && p.launchDay < s.day,
  );
  if (!launched) return;

  const revenue = total(closed.income);
  const costs = total(closed.costs);
  const lastProfit = previous ? total(previous.income) - total(previous.costs) : null;
  const growthByGroup = {} as Record<GroupId, number>;
  for (const g of GROUP_IDS) {
    growthByGroup[g] = closed.customersEndByGroup[g] - (previous?.customersEndByGroup[g] ?? 0);
  }
  const mood = moodOf(s, company);

  // The feed is oldest first, so on a tie the earliest wins.
  const posts = s.feed.filter(
    (f): f is PostItem =>
      f.kind === "post" && f.companyId === company.id && monthOfDay(f.day) === closed.month,
  );
  let best: PostItem | null = null;
  let worst: PostItem | null = null;
  for (const p of posts) {
    if (!best || p.sentiment > best.sentiment) best = p;
    if (!worst || p.sentiment < worst.sentiment) worst = p;
  }
  if (worst === best) worst = null;
  let headline: { text: string; strength: number } | null = null;
  for (const f of s.feed) {
    if (f.kind !== "news" || f.companyId !== company.id || monthOfDay(f.day) !== closed.month)
      continue;
    if (!headline || Math.abs(f.sentiment) > headline.strength) {
      headline = { text: f.headline, strength: Math.abs(f.sentiment) };
    }
  }

  const start = previous?.customersEnd ?? 0;
  const profit = revenue - costs;
  company.reports.push({
    month: closed.month,
    revenue,
    costs,
    profit,
    lastProfit,
    customersStart: start,
    customersEnd: closed.customersEnd,
    growthByGroup,
    mood,
    best: best ? moment(best) : null,
    worst: worst ? moment(worst) : null,
    headline: headline?.text ?? null,
    grade: gradeFor({ profit, lastProfit, start, end: closed.customersEnd, mood }),
  });
  if (company.reports.length > REPORTS_KEPT) {
    company.reports.splice(0, company.reports.length - REPORTS_KEPT);
  }
}
