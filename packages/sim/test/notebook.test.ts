import { describe, expect, it } from "vitest";
import { BALANCE_V2, projectView, step, upgradeState } from "../src/index.ts";
import { cmd, ok, rejected, runningWorld, sys, TICK, ticks } from "./pack.ts";

const HASH = "a".repeat(64);
const plan = (notebook: number) => ({
  ...BALANCE_V2.defaultPlan,
  mix: { ...BALANCE_V2.defaultPlan.mix, notebook },
});

describe("the notebook", () => {
  it("is a ninth share of the plan, 0 by default", () => {
    expect(BALANCE_V2.defaultPlan.mix.notebook).toBe(0);
    // A plan logged before the notebook still reads, with no notebook share.
    const { notebook: _, ...eight } = BALANCE_V2.defaultPlan.mix;
    const s = ok(
      step(runningWorld(), cmd({ type: "startTraining", plan: { ...plan(0), mix: eight } }, "r1")),
    );
    expect(s.companies.c1?.runs.r1?.plan.mix.notebook).toBe(0);
    expect(s.companies.c1?.runs.r1?.notebookHash).toBeNull();
  });

  it("can't be trained on before something is written in it", () => {
    const s = runningWorld();
    expect(s.companies.c1?.notebook).toBeNull();
    expect(rejected(step(s, cmd({ type: "startTraining", plan: plan(10) }, "r1")))).toBe(
      "Write something in your notebook first.",
    );
    const empty = ok(
      step(s, sys({ type: "notebookSaved", companyId: "c1", hash: HASH, words: 0 })),
    );
    expect(rejected(step(empty, cmd({ type: "startTraining", plan: plan(10) }, "r1")))).toBe(
      "Write something in your notebook first.",
    );
  });

  it("is saved by the server as a fingerprint and a word count, and a run remembers which", () => {
    let s = ticks(runningWorld(), 2);
    s = ok(step(s, sys({ type: "notebookSaved", companyId: "c1", hash: HASH, words: 120 })));
    expect(s.companies.c1?.notebook).toEqual({ hash: HASH, words: 120, savedDay: s.day });
    s = ok(step(s, cmd({ type: "startTraining", plan: plan(10) }, "r1")));
    expect(s.companies.c1?.runs.r1?.notebookHash).toBe(HASH);
    // Saving again later doesn't change what the running run trains on.
    s = ok(
      step(s, sys({ type: "notebookSaved", companyId: "c1", hash: "b".repeat(64), words: 3 })),
    );
    expect(s.companies.c1?.runs.r1?.notebookHash).toBe(HASH);
  });

  it("refuses a bad save", () => {
    const s = runningWorld();
    const save = (e: object) => step(s, sys({ type: "notebookSaved", companyId: "c1", ...e }));
    expect(rejected(save({ hash: "nope", words: 1 }))).toMatch(/notebook/i);
    expect(rejected(save({ hash: HASH, words: 5_001 }))).toMatch(/notebook/i);
    expect(rejected(save({ hash: HASH, words: -1 }))).toMatch(/notebook/i);
    expect(
      rejected(step(s, sys({ type: "notebookSaved", companyId: "c9", hash: HASH, words: 1 }))),
    ).toMatch(/company/i);
  });

  it("shows its owner the word count, and nobody its fingerprint", () => {
    const s = ok(
      step(runningWorld(), sys({ type: "notebookSaved", companyId: "c1", hash: HASH, words: 7 })),
    );
    const view = projectView(s, "u1");
    expect(view.me.company?.notebook).toEqual({ words: 7, savedDay: s.day });
    expect(JSON.stringify(view)).not.toContain(HASH);
  });
});

describe("upgrading a sim-5 world", () => {
  it("gives every plan a notebook share of 0 and keeps playing", () => {
    let s = ok(step(runningWorld(), cmd({ type: "startTraining", plan: plan(0) }, "r1")));
    const old = structuredClone(s) as unknown as {
      companies: Record<
        string,
        Record<string, unknown> & {
          runs: Record<string, { plan: { mix: Record<string, number> }; notebookHash?: unknown }>;
        }
      >;
    };
    for (const c of Object.values(old.companies)) {
      delete c.notebook;
      for (const r of Object.values(c.runs)) {
        delete r.plan.mix.notebook;
        delete r.notebookHash;
      }
    }
    s = upgradeState(old, "sim-5");
    expect(s.companies.c1?.notebook).toBeNull();
    expect(s.companies.c1?.runs.r1?.plan.mix.notebook).toBe(0);
    expect(s.companies.c1?.runs.r1?.notebookHash).toBeNull();
    expect(step(s, TICK).ok).toBe(true);
  });
});
