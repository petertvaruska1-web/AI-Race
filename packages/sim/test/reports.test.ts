import { type Command, GROUP_IDS } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  BALANCE_TEST,
  createWorld,
  gradeFor,
  type Input,
  monthOfDay,
  projectView,
  type StepResult,
  step,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}
const cmd = (c: unknown, newId?: string): Input => ({
  kind: "command",
  userId: "u1",
  cmd: c as Command,
  ...(newId ? { newId } : {}),
});
const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;
const ticks = (s: WorldState, n: number) => {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, { kind: "tick" }));
  return out;
};
/** Ticks until the calendar turns to a new month (the close happens on its first day). */
const toNextMonth = (s: WorldState) => {
  let out = s;
  const month = monthOfDay(out.day);
  while (monthOfDay(out.day) === month) out = ticks(out, 1);
  return out;
};
const c1 = (s: WorldState) => {
  const c = s.companies.c1;
  if (!c) throw new Error("no company");
  return c;
};

function world(): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 5,
    balance: "balance-test",
    inviteCode: "ABC234",
  });
  const palette = { primary: "#4f8cff", secondary: "#ffb020" };
  s = ok(step(s, cmd({ type: "createCompany", name: "Acme", logo: LOGO, palette }, "c1")));
  return ok(step(s, cmd({ type: "startWorld" })));
}

/** c1's scored v1 launched as Spark on every plan; live once the countdown ends. */
function launched(): WorldState {
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  let s = ok(step(world(), cmd({ type: "startTraining", plan }, "r1")));
  s = ok(
    step(
      s,
      sys({
        type: "trainingCompleted",
        runId: "r1",
        versionId: "v1",
        result: { steps: 8, finalLoss: 4, heldoutLosses: {}, unstable: false },
      }),
    ),
  );
  s = ticks(s, (s.companies.c1?.runs.r1?.endDay ?? 0) - s.day);
  s = ok(
    step(
      s,
      sys({
        type: "evalCompleted",
        versionId: "v1",
        scores: SCORES,
        overall: 42,
        personalityWords: [],
      }),
    ),
  );
  s = ok(
    step(
      s,
      cmd(
        {
          type: "launchProduct",
          versionId: "v1",
          name: "Spark",
          plans: ["free", "personal", "business"],
          countdownDays: 3,
        },
        "p1",
      ),
    ),
  );
  return ticks(s, 4);
}

const HOSTILE = "<b>bold</b> <|end|>\n‮evil";

describe("report cards", () => {
  it("wait for the first launch", () => {
    const s = toNextMonth(toNextMonth(world()));
    expect(c1(s).ledger.months.length).toBeGreaterThan(0);
    expect(c1(s).reports).toEqual([]);
  });

  it("sum up the first month that closes after a launch", () => {
    const s = toNextMonth(launched());
    const card = c1(s).reports.at(-1);
    const closed = c1(s).ledger.months.at(-1);
    const previous = c1(s).ledger.months.at(-2);
    if (!card || !closed) throw new Error("no card");
    const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);
    expect(c1(s).reports).toHaveLength(1);
    expect(card.month).toBe(closed.month);
    expect(card.revenue).toBe(total(closed.income));
    expect(card.costs).toBe(total(closed.costs));
    expect(card.profit).toBe(card.revenue - card.costs);
    expect(card.lastProfit).toBe(previous ? total(previous.income) - total(previous.costs) : null);
    expect(card.customersEnd).toBe(closed.customersEnd);
    expect(card.customersStart).toBe(previous?.customersEnd ?? 0);
    for (const g of GROUP_IDS) {
      expect(card.growthByGroup[g]).toBe(
        closed.customersEndByGroup[g] - (previous?.customersEndByGroup[g] ?? 0),
      );
    }
    expect(["A", "B", "C", "D", "F"]).toContain(card.grade);
  });

  it("grade from profit, growth and mood", () => {
    const strong = { profit: 5000, lastProfit: 2000, start: 1000, end: 1200, mood: 0.7 };
    expect(gradeFor(strong)).toBe("A");
    // Points: profit 2 (above 0, above last month) + growth 0.5 + mood 0 = 2.5.
    expect(gradeFor({ ...strong, end: 1050, mood: 0.3 })).toBe("B");
    expect(gradeFor({ ...strong, profit: -10, lastProfit: -500, end: 1050 })).toBe("B");
    // Points: profit 0 + growth 0.5 + mood 1 = 1.5.
    expect(gradeFor({ ...strong, profit: -10, lastProfit: 0, end: 1050 })).toBe("C");
    expect(gradeFor({ profit: -100, lastProfit: 0, start: 1000, end: 900, mood: 0.5 })).toBe("D");
    expect(gradeFor({ profit: -100, lastProfit: 0, start: 1000, end: 900, mood: null })).toBe("F");
    // A first month with customers from nothing counts as growth; no last month counts profit twice.
    expect(gradeFor({ profit: 10, lastProfit: null, start: 0, end: 5, mood: null })).toBe("B");
  });

  it("quote the month's best and worst posts exactly, and its biggest story", () => {
    let s = launched();
    s = ok(
      step(
        s,
        sys({
          type: "feedbackScored",
          productId: "p1",
          key: "w99",
          results: [
            {
              group: "students",
              questionId: "q1",
              question: "Capital of France?",
              reply: "Paris.",
              score: 0.95,
            },
            {
              group: "developers",
              questionId: "q2",
              question: "Sum 2+2?",
              reply: HOSTILE,
              score: 0.05,
            },
          ],
        }),
      ),
    );
    s = toNextMonth(s);
    const card = c1(s).reports.at(-1);
    expect(card?.best?.reply).toBe("Paris.");
    expect(card?.best?.question).toBe("Capital of France?");
    expect(card?.best?.productName).toBe("Spark");
    expect(card?.worst?.reply).toBe(HOSTILE);
    expect(card?.worst?.tone).toBe("roast");
    const month = card?.month ?? "";
    const stories = s.feed.filter(
      (f) => f.kind === "news" && f.companyId === "c1" && monthOfDay(f.day) === month,
    );
    const strongest = [...stories].sort((a, b) => Math.abs(b.sentiment) - Math.abs(a.sentiment))[0];
    expect(card?.headline).toBe(strongest?.kind === "news" ? strongest.headline : null);
  });

  it("leave the moments empty in a month without posts or news, and keep the last 24", () => {
    let s = toNextMonth(toNextMonth(launched()));
    const quiet = c1(s).reports.at(-1);
    expect(quiet?.best).toBeNull();
    expect(quiet?.worst).toBeNull();
    expect(quiet?.headline).toBeNull();
    s = ticks(s, 800);
    expect(c1(s).reports).toHaveLength(24);
    const view = projectView(s, "u1");
    expect(view.me.company?.reports).toHaveLength(12);
    expect(view.me.company?.reports.at(-1)?.month).toBe(c1(s).reports.at(-1)?.month);
  });
});
