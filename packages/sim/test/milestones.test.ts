import { MILESTONE_IDS, MILESTONE_NAMES } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  type CompanyState,
  milestonesDay,
  projectView,
  step,
  upgradeState,
  type VersionState,
  type WorldState,
} from "../src/index.ts";
import { ok, runningWorld, sys, TICK, ticks, withRivals } from "./pack.ts";

function version(id: string, label = "v1"): VersionState {
  return {
    id,
    label,
    runId: `r-${id}`,
    parentId: null,
    nLayer: 2,
    dModel: 64,
    ctxLen: 64,
    createdDay: 0,
    trainedTokens: 1,
    scores: null,
    overall: null,
    personalityWords: [],
    traits: null,
    published: false,
    packKey: null,
  } as VersionState;
}

function withVersion(s0: WorldState, id = "v1", label = "v1"): WorldState {
  const s = structuredClone(s0);
  (s.companies.c1 as CompanyState).versions[id] = version(id, label);
  return s;
}

const evalOf = (versionId: string, signals?: unknown) =>
  sys({
    type: "evalCompleted",
    versionId,
    scores: { language: 10 },
    overall: 10,
    personalityWords: [],
    traits: null,
    ...(signals === undefined ? {} : { signals }),
  });

const ids = (s: WorldState, company = "c1") =>
  (s.companies[company] as CompanyState).milestones.map((m) => m.id);

function customers(s: WorldState, n: number, plan: "free" | "personal" = "free"): void {
  s.market.customers.c1 = { newEngland: { everyday: { [plan]: n } } };
}

describe("milestones", () => {
  it("names twelve milestones", () => {
    expect(MILESTONE_IDS).toHaveLength(12);
    expect(MILESTONE_NAMES.firstSentence).toBe("First full sentence");
    expect(MILESTONE_NAMES.gradeA).toBe("An A on the report card");
  });

  it("come from an eval's measurements, each once", () => {
    let s = withVersion(runningWorld());
    s = ok(step(s, evalOf("v1", { legibleReplies: 0, correctFacts: 0, passingPrograms: 0 })));
    expect(ids(s)).toEqual([]);
    s = ok(step(s, evalOf("v1", { legibleReplies: 3, correctFacts: 1, passingPrograms: 0 })));
    expect(ids(s)).toEqual(["firstSentence", "firstFact"]);
    const first = (s.companies.c1 as CompanyState).milestones[0];
    expect(first).toEqual({ id: "firstSentence", day: s.day, detail: "v1" });
    s = withVersion(s, "v2", "v2");
    s = ok(step(s, evalOf("v2", { legibleReplies: 9, correctFacts: 5, passingPrograms: 2 })));
    expect(ids(s)).toEqual(["firstSentence", "firstFact", "firstProgram"]);
  });

  it("never come from a measurement that wasn't made", () => {
    let s = withVersion(runningWorld());
    s = ok(step(s, evalOf("v1"))); // an eval from before signals
    s = ok(step(s, evalOf("v1", { legibleReplies: null, correctFacts: 0, passingPrograms: 0 })));
    expect(ids(s)).toEqual([]);
  });

  it("come daily from the company's state", () => {
    const s = ticks(runningWorld(), 1);
    customers(s, 999);
    milestonesDay(s);
    expect(ids(s)).toEqual([]);
    customers(s, 1_000);
    milestonesDay(s);
    expect(ids(s)).toEqual(["customers1k"]);
    customers(s, 2_000_000, "personal"); // a big jump earns every mark it passes
    milestonesDay(s);
    expect(ids(s)).toEqual(["customers1k", "customers100k", "customers1m", "firstPaying"]);
    milestonesDay(s);
    expect(ids(s)).toHaveLength(4);
  });

  it("count a data center once it opens, a profitable month and an A", () => {
    const s = ticks(runningWorld(), 1);
    const c1 = s.companies.c1 as CompanyState;
    c1.dataCenters.d1 = {
      id: "d1",
      region: "newEngland",
      startDay: 0,
      openDay: s.day + 5,
      pflopsPerDay: 1,
    };
    milestonesDay(s);
    expect(ids(s)).toEqual([]);
    s.day += 5;
    milestonesDay(s);
    expect(ids(s)).toEqual(["dataCenter"]);
    const card = { month: "2027-02", profit: -5, grade: "C" };
    c1.reports.push(card as never);
    milestonesDay(s);
    expect(ids(s)).toEqual(["dataCenter"]);
    c1.reports.push({ ...card, month: "2027-03", profit: 10, grade: "A" } as never);
    milestonesDay(s);
    expect(ids(s)).toEqual(["dataCenter", "firstProfit", "gradeA"]);
    expect(c1.milestones[1]?.detail).toBe("2027-03");
  });

  it("count #1 in a skill only with rivals, and a battle won against the arch-rival", () => {
    const solo = ticks(runningWorld(), 1);
    solo.leaders.knowledge = "c1";
    milestonesDay(solo);
    expect(ids(solo)).toEqual([]);
    const s = ticks(withRivals(), 1);
    s.leaders.knowledge = "c1";
    milestonesDay(s);
    expect(ids(s)).toEqual(["numberOne"]);
    expect((s.companies.c1 as CompanyState).milestones[0]?.detail).toBe("knowledge");
    const c1 = s.companies.c1 as CompanyState;
    const battle = { id: "b1", rivalId: "rival-2", outcome: "win", status: "done" };
    c1.battles.push(battle as never);
    milestonesDay(s);
    expect(ids(s)).not.toContain("beatArch");
    c1.battles.push({ ...battle, id: "b2", rivalId: s.rivals?.archId } as never);
    milestonesDay(s);
    expect(ids(s)).toContain("beatArch");
  });

  it("are never earned by rivals", () => {
    const s = ticks(withRivals(), 1);
    s.market.customers["rival-1"] = { newEngland: { everyday: { personal: 5_000 } } };
    milestonesDay(s);
    expect(ids(s, "rival-1")).toEqual([]);
  });

  it("show on the trophy shelf, oldest first, to their owner only", () => {
    const s = ticks(runningWorld(), 1);
    customers(s, 1_000);
    milestonesDay(s);
    const view = projectView(s, "u1");
    expect(view.me.company?.milestones).toEqual([{ id: "customers1k", day: s.day, detail: null }]);
  });

  it("catch up in an upgraded world from its state, but not from measurements", () => {
    let s = withVersion(ticks(runningWorld(), 1));
    const old = structuredClone(s) as unknown as {
      companies: Record<string, Record<string, unknown>>;
    };
    for (const c of Object.values(old.companies)) delete c.milestones;
    s = upgradeState(old, "sim-5");
    expect(ids(s)).toEqual([]);
    s = ok(step(s, TICK));
    expect(ids(s)).toEqual([]); // nothing measured since: no invented history
  });
});
