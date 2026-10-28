import {
  BENCH_CATEGORIES,
  type Command,
  GROUP_IDS,
  type GroupId,
  PLAN_IDS,
  type PlanId,
  REGION_IDS,
  type RegionId,
} from "@airace/protocol";
import {
  type Balance,
  type CompanyState,
  createWorld,
  customersByRegion,
  customersOf,
  type Input,
  type ReportCard,
  step,
  totalCustomers,
  type WorldState,
} from "../src/index.ts";

// Headless companies for the balance harness: each feeds `step` the inputs a
// simple player would send, with fake training results, scores and weekly
// customer feedback.

/** Scores of a typical first model, from the M1 gate's first models. */
export const TYPICAL_FIRST_MODEL: Record<string, number> = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};
export const TYPICAL_OVERALL = 42;

export type BotScript = {
  scores?: Record<string, number>;
  plans?: PlanId[];
  prices?: { personal?: number; business?: number };
  countdownDays?: number;
  /** A second, better model launched as another product this many days after the first launch. */
  retrain?: { afterDays: number; scores: Record<string, number> };
  /** Ad campaigns, each started this many days after launch. */
  campaigns?: {
    afterDays: number;
    region: RegionId | "all";
    group: GroupId | "all";
    budgetPerDay: number;
    days: number;
  }[];
  /** On this day after launch every customer is multiplied by `factor` (a viral spike). */
  spike?: { afterDays: number; factor: number };
};

export type DayReport = {
  day: number;
  /** All customers (a business account counts once). */
  customers: number;
  /** Personal customers plus business accounts. */
  paying: number;
  byGroup: Record<GroupId, Record<PlanId, number>>;
  cash: number;
  /** This day's bookings. */
  income: number;
  serving: number;
  costs: number;
  byRegion: Record<RegionId, number>;
  /** The share of the day's questions answered. */
  served: number;
  /** What is still owed on rescue loans. */
  loan: number;
  rescueOffered: boolean;
};
export type Report = {
  launchDay: number;
  days: DayReport[];
  /** The company's report cards at the end. */
  reports: ReportCard[];
  /** The world at the end, for follow-up experiments. */
  final: WorldState;
};

const USER = "bot";
const COMPANY = "c1";

function apply(s: WorldState, input: Input): WorldState {
  const r = step(s, input);
  if (!r.ok) throw new Error(`bot input refused: ${r.reason}`);
  return r.state;
}
const cmd = (c: unknown, newId?: string): Input => ({
  kind: "command",
  userId: USER,
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;

function company(s: WorldState): CompanyState {
  const c = s.companies[COMPANY];
  if (!c) throw new Error("no bot company");
  return c;
}

function totals(c: CompanyState): { income: number; serving: number; costs: number } {
  let income = 0;
  let serving = 0;
  let costs = 0;
  for (const m of [...c.ledger.months, c.ledger.current]) {
    income += m.income.personal + m.income.business;
    serving += m.costs.serving;
    costs += Object.values(m.costs).reduce((a, b) => a + b, 0);
  }
  return { income, serving, costs };
}

/** A group's feedback score for a model: its weighted skill, as the checks would roughly find. */
function feedbackScore(balance: Balance, group: GroupId, scores: Record<string, number>): number {
  let skill = 0;
  for (const [category, weight] of Object.entries(balance.economy.groups[group].weights)) {
    skill += (weight ?? 0) * ((scores[category] ?? 0) / 100);
  }
  return Math.min(1, Math.max(0, skill));
}

/** Trains a version to completion and scores it. Returns the state on its scoring day. */
function trainAndScore(
  s: WorldState,
  balance: Balance,
  runId: string,
  versionId: string,
  parentVersionId: string | null,
  scores: Record<string, number>,
): WorldState {
  const plan = {
    ...balance.defaultPlan,
    mix: { ...balance.defaultPlan.mix },
    parentVersionId,
  };
  let out = apply(s, cmd({ type: "startTraining", plan }, runId));
  out = apply(
    out,
    sys({
      type: "trainingCompleted",
      runId,
      versionId,
      result: { steps: 100, finalLoss: 4, heldoutLosses: {}, unstable: false },
    }),
  );
  const end = company(out).runs[runId]?.endDay ?? out.day;
  while (out.day < end) out = apply(out, { kind: "tick" });
  const overall =
    BENCH_CATEGORIES.reduce((sum, c) => sum + (scores[c] ?? 0), 0) / BENCH_CATEGORIES.length;
  return apply(
    out,
    sys({ type: "evalCompleted", versionId, scores, overall, personalityWords: [], traits: null }),
  );
}

/**
 * Plays one scripted company: found, train, score, launch (default prices
 * unless the script sets them), then `days` more days with weekly feedback.
 */
export function runBot(balance: Balance, script: BotScript, days: number, seed = 1): Report {
  const scores = script.scores ?? TYPICAL_FIRST_MODEL;
  let s = createWorld({
    worldId: "bot-world",
    name: "Balance",
    hostUserId: USER,
    hostName: "Bot",
    seed,
    balance: balance.version,
    inviteCode: "BOT",
  });
  s = apply(
    s,
    cmd(
      {
        type: "createCompany",
        name: "Bot Co",
        logo: { shape: "circle", symbol: "spark", initials: "BC" },
        palette: { primary: "#4f8cff", secondary: "#ffb020" },
      },
      COMPANY,
    ),
  );
  s = apply(s, cmd({ type: "startWorld" }));
  s = trainAndScore(s, balance, "r1", "v1", null, scores);
  if (script.prices?.personal !== undefined) {
    s = apply(s, cmd({ type: "setPrice", plan: "personal", price: script.prices.personal }));
  }
  if (script.prices?.business !== undefined) {
    s = apply(s, cmd({ type: "setPrice", plan: "business", price: script.prices.business }));
  }
  const plans = script.plans ?? ["free", "personal", "business"];
  const countdownDays = script.countdownDays ?? 3;
  s = apply(
    s,
    cmd({ type: "launchProduct", versionId: "v1", name: "Bot One", plans, countdownDays }, "p1"),
  );
  const launchDay = s.day + countdownDays;
  const report: Report = { launchDay, days: [], reports: [], final: s };
  let before = totals(company(s));
  let modelScores: Record<string, number> = scores;
  let liveProduct = "p1";
  let week = -1;
  while (s.day < launchDay + days) {
    s = apply(s, { kind: "tick" });
    for (const [i, k] of (script.campaigns ?? []).entries()) {
      if (s.day !== launchDay + k.afterDays) continue;
      const { afterDays: _, ...campaign } = k;
      s = apply(s, cmd({ type: "startCampaign", ...campaign }, `ad${i}`));
    }
    if (script.spike && s.day === launchDay + script.spike.afterDays) {
      const factor = script.spike.factor;
      s = structuredClone(s);
      for (const byGroup of Object.values(s.market.customers[COMPANY] ?? {})) {
        for (const byPlan of Object.values(byGroup ?? {})) {
          for (const plan of PLAN_IDS) if (byPlan?.[plan]) byPlan[plan] *= factor;
        }
      }
    }
    // A bot never chooses a rescue: at the deadline the game takes the loan.
    const retrain = script.retrain;
    if (retrain && s.day === launchDay + retrain.afterDays) {
      s = trainAndScore(s, balance, "r2", "v2", "v1", retrain.scores);
      s = apply(
        s,
        cmd(
          { type: "launchProduct", versionId: "v2", name: "Bot Two", plans, countdownDays },
          "p2",
        ),
      );
      modelScores = retrain.scores;
      liveProduct = "p2";
    }
    const product = company(s).products[liveProduct];
    const thisWeek = Math.floor(s.day / 7);
    if (product?.status === "live" && thisWeek !== week) {
      week = thisWeek;
      const results = GROUP_IDS.map((group) => ({
        group,
        questionId: `q-${group}`,
        question: "A customer question",
        reply: "A reply.",
        score: feedbackScore(balance, group, modelScores),
      }));
      s = apply(
        s,
        sys({ type: "feedbackScored", productId: liveProduct, key: `w${thisWeek}`, results }),
      );
    }
    const c = company(s);
    const now = totals(c);
    const byGroup = {} as Record<GroupId, Record<PlanId, number>>;
    let paying = 0;
    for (const group of GROUP_IDS) {
      const byPlan = {} as Record<PlanId, number>;
      for (const plan of PLAN_IDS) {
        byPlan[plan] = REGION_IDS.reduce(
          (n, region) => n + customersOf(s.market, COMPANY, region, group, plan),
          0,
        );
        if (plan !== "free") paying += byPlan[plan];
      }
      byGroup[group] = byPlan;
    }
    report.days.push({
      day: s.day,
      customers: totalCustomers(s.market, COMPANY),
      paying,
      byGroup,
      cash: c.cash,
      income: now.income - before.income,
      serving: now.serving - before.serving,
      costs: now.costs - before.costs,
      byRegion: customersByRegion(s.market, COMPANY),
      served: c.compute.served,
      loan: c.loan?.balance ?? 0,
      rescueOffered: c.rescue !== null,
    });
    before = now;
  }
  report.reports = company(s).reports;
  report.final = s;
  return report;
}

/** The report's day `launchDay + n`. */
export function afterLaunch(report: Report, n: number): DayReport {
  const d = report.days.find((x) => x.day === report.launchDay + n);
  if (!d) throw new Error(`no day ${n} after launch`);
  return d;
}
