import { BENCH_CATEGORIES, LEAGUE_TABLES, WorldView } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  leagueTables,
  projectView,
  step,
  type WorldState,
} from "../src/index.ts";
import { cmd, ok, runningWorld, sys, ticks, withRivals } from "./pack.ts";

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

/** Gives c1 a scored version and launches it on every plan, with `scores`. */
function launch(s0: WorldState, scores: Record<string, number> = SCORES): WorldState {
  const s = structuredClone(s0);
  const c1 = s.companies.c1 as CompanyState;
  c1.versions.v1 = {
    id: "v1",
    label: "v1",
    runId: "r1",
    parentId: null,
    nLayer: 4,
    dModel: 128,
    ctxLen: 256,
    createdDay: s.day,
    trainedTokens: 1,
    scores,
    overall: Object.values(scores).reduce((a, b) => a + b, 0) / 8,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  };
  const launched = ok(
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
  return ticks(launched, 3);
}

describe("league tables", () => {
  it("has a table for customers, revenue, overall and each skill", () => {
    expect(LEAGUE_TABLES).toEqual(["customers", "revenue", "overall", ...BENCH_CATEGORIES]);
    const t = leagueTables(withRivals());
    expect(Object.keys(t).sort()).toEqual([...LEAGUE_TABLES].sort());
  });

  it("lists the best first, ties by company id, and nothing-on-sale last with no value", () => {
    const s = ticks(withRivals(), 40);
    const t = leagueTables(s);
    const coding = t.coding;
    const values = coding.map((r) => r.value);
    const known = values.filter((v): v is number => v !== null);
    expect(known).toEqual([...known].sort((a, b) => b - a));
    expect(values.slice(known.length).every((v) => v === null)).toBe(true);
    expect(coding.find((r) => r.companyId === "c1")?.value).toBeNull();
    expect(coding.at(-1)?.value).toBeNull();
  });

  it("shows other companies' revenue as a place only, never an amount", () => {
    let s = launch(withRivals());
    s = ticks(s, 70);
    const view = WorldView.parse(projectView(s, "u1"));
    for (const row of view.world.league.tables.revenue) {
      if (row.companyId === "c1") expect(row.value).not.toBeNull();
      else expect(row.value).toBeNull();
    }
    for (const c of Object.values(s.companies)) {
      if (c.id === "c1") continue;
      const income = c.ledger.months.at(-1)?.income;
      const amount = income ? income.personal + income.business : 0;
      if (amount > 0) expect(JSON.stringify(view.world.league)).not.toContain(`:${amount}`);
    }
  });
});

describe("news of a new #1", () => {
  it("tells one story when the leader changes, none on a flip-flop within 14 days, none without rivals", () => {
    let s = ticks(withRivals(), 40);
    const before = s.feed.filter((f) => f.kind === "news" && f.topic === "ranking").length;
    s = launch(s, { ...SCORES, coding: 100, language: 10, pattern: 10 });
    s = ticks(s, 1);
    const stories = s.feed.filter((f) => f.kind === "news" && f.topic === "ranking");
    expect(stories.length).toBe(before + 1);
    const story = stories.at(-1);
    expect(story?.kind === "news" && story.headline).toBe("Acme takes #1 in coding");
    expect(s.leaders.coding).toBe("c1");

    // The player falls back and takes it again within 14 days: no second story.
    const c1 = s.companies.c1 as CompanyState;
    (c1.versions.v1 as { scores: Record<string, number> }).scores.coding = 0;
    s = ticks(s, 2);
    (
      (s.companies.c1 as CompanyState).versions.v1 as { scores: Record<string, number> }
    ).scores.coding = 100;
    s = ticks(s, 2);
    const again = s.feed.filter(
      (f) => f.kind === "news" && f.topic === "ranking" && f.headline.includes("coding"),
    );
    expect(again).toHaveLength(1);

    const alone = ticks(launch(runningWorld()), 5);
    expect(alone.feed.some((f) => f.kind === "news" && f.topic === "ranking")).toBe(false);
  });
});

describe("rank changes on the report card", () => {
  it("lists customers, revenue and overall places from last month to this, only with rivals", () => {
    let s = launch(withRivals());
    s = ticks(s, 75);
    const cards = s.companies.c1?.reports ?? [];
    expect(cards.length).toBeGreaterThanOrEqual(2);
    const last = cards.at(-1);
    expect(last?.ranks?.map((r) => r.table)).toEqual(["customers", "revenue", "overall"]);
    for (const r of last?.ranks ?? []) expect(r.to).toBeGreaterThanOrEqual(1);
    expect(cards.at(-1)?.ranks?.[0]?.from).toBe(cards.at(-2)?.ranks?.[0]?.to);

    const alone = ticks(launch(runningWorld()), 75);
    expect(alone.companies.c1?.reports.at(-1)?.ranks).toBeNull();
  });
});

describe("the view", () => {
  it("parses with league tables, rival rows included", () => {
    const view = WorldView.parse(projectView(launch(withRivals()), "u1"));
    expect(view.world.league.tables.customers).toHaveLength(5);
    expect(view.world.league.points.coding).toEqual([]);
    expect(sys).toBeDefined();
  });
});
