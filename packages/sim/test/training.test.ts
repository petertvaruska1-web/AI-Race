import type { Command, TrainingPlan } from "@airace/protocol";
import { describe, expect, it } from "vitest";
import {
  activeRun,
  BALANCE_V1,
  createWorld,
  type Input,
  previewTraining,
  runProgress,
  type StepResult,
  step,
  type TrainingResult,
  type WorldState,
} from "../src/index.ts";

const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
const RESULT: TrainingResult = {
  steps: 1536,
  finalLoss: 3.9,
  heldoutLosses: { web: 4.1, books: null },
  unstable: false,
};

function ok(r: StepResult): WorldState {
  if (!r.ok) throw new Error(`rejected: ${r.reason}`);
  return r.state;
}

function rejected(r: StepResult): string {
  if (r.ok) throw new Error("expected a rejection");
  return r.reason;
}

const cmd = (userId: string, c: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: c as Command,
  ...(newId === undefined ? {} : { newId }),
});

const plan = (change: Partial<TrainingPlan> = {}): TrainingPlan => ({
  ...BALANCE_V1.defaultPlan,
  mix: { ...BALANCE_V1.defaultPlan.mix },
  ...change,
});

const train = (userId: string, runId: string, p: TrainingPlan = plan()): Input =>
  cmd(userId, { type: "startTraining", plan: p }, runId);

const sys = (event: unknown): Input => ({ kind: "system", event }) as Input;
const completed = (runId: string, versionId: string): Input =>
  sys({ type: "trainingCompleted", runId, versionId, result: RESULT });
const TICK: Input = { kind: "tick" };

function ticks(s: WorldState, n: number): WorldState {
  let out = s;
  for (let i = 0; i < n; i++) out = ok(step(out, TICK));
  return out;
}

/** A running world: u1 owns c1 and u2 owns c2. */
function running(start = true): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Lab",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 3,
    balance: "balance-v1",
    inviteCode: "ABCD-1234",
  });
  s = ok(step(s, sys({ type: "memberJoined", userId: "u2", displayName: "Ben" })));
  s = ok(
    step(
      s,
      cmd(
        "u1",
        {
          type: "createCompany",
          name: "Acme",
          logo: LOGO,
          palette: { primary: "#4f8cff", secondary: "#ffb020" },
        },
        "c1",
      ),
    ),
  );
  s = ok(
    step(
      s,
      cmd(
        "u2",
        {
          type: "createCompany",
          name: "Beta",
          logo: LOGO,
          palette: { primary: "#ff7a45", secondary: "#ffd166" },
        },
        "c2",
      ),
    ),
  );
  if (start) s = ok(step(s, cmd("u1", { type: "startWorld" })));
  return s;
}

/** Trains and completes one run for `userId`, returning the state with its version. */
function trained(
  s: WorldState,
  userId: string,
  runId: string,
  versionId: string,
  p = plan(),
): WorldState {
  let out = ok(step(s, train(userId, runId, p)));
  out = ok(step(out, completed(runId, versionId)));
  const company = Object.values(out.companies).find((c) => c.ownerUserId === userId);
  const run = company?.runs[runId];
  if (!run) throw new Error("run missing");
  return ticks(out, run.endDay - out.day);
}

describe("previewTraining", () => {
  it("prices the default plan from compute: $41,596 over 13 game days", () => {
    const r = previewTraining(running(), "c1", plan());
    expect(r).toEqual({
      ok: true,
      preview: {
        params: 1_377_408,
        tokens: 12_582_912,
        flops: 6 * 1_377_408 * 12_582_912,
        cost: 41_596,
        durationDays: 13,
      },
    });
  });

  it("adds the fact-check fee", () => {
    const r = previewTraining(running(), "c1", plan({ factCheck: true }));
    expect(r.ok && r.preview.cost).toBe(41_596 + BALANCE_V1.factCheckCost);
  });
});

describe("startTraining", () => {
  it("charges the cost, schedules the run in game days and draws its seed from the world", () => {
    const before = ticks(running(), 2);
    const s = ok(step(before, train("u1", "r1")));
    const run = s.companies.c1?.runs.r1;
    expect(run).toMatchObject({
      status: "training",
      computeStarted: false,
      startDay: 2,
      endDay: 15,
      held: null,
      failure: null,
      versionId: null,
    });
    expect(Number.isInteger(run?.seed)).toBe(true);
    expect(s.companies.c1?.cash).toBe(BALANCE_V1.foundingCash - 41_596);
    expect(s.rng).not.toEqual(before.rng);
    expect(activeRun(s.companies.c1 ?? never())).toBe(run);
  });

  it.each<[string, (s: WorldState) => Input, RegExp]>([
    ["a second run while one is active", () => train("u1", "r2"), /already training/i],
    [
      "an all-zero mix",
      () =>
        train(
          "u1",
          "r2",
          plan({
            mix: {
              web: 0,
              books: 0,
              educational: 0,
              conversations: 0,
              code: 0,
              reasoning: 0,
              facts: 0,
              creative: 0,
            },
          }),
        ),
      /dataset/i,
    ],
    ["an unknown brain size", () => train("u1", "r2", plan({ brainSize: "huge" })), /brain size/i],
    ["an unknown length", () => train("u1", "r2", plan({ length: "forever" })), /length/i],
    [
      "an unknown attention span",
      () => train("u1", "r2", plan({ attentionSpan: 300 })),
      /attention/i,
    ],
    [
      "another company's parent",
      () => train("u1", "r2", plan({ parentVersionId: "v-beta" })),
      /yours/i,
    ],
    ["a missing run id", () => cmd("u1", { type: "startTraining", plan: plan() }), /id/i],
  ])("rejects %s", (_label, input, reason) => {
    let s = trained(running(), "u2", "rb", "v-beta");
    if (_label.startsWith("a second")) s = ok(step(s, train("u1", "r1")));
    const r = step(s, input(s));
    expect(rejected(r)).toMatch(reason);
  });

  it("rejects a plan it cannot pay for", () => {
    const s = running();
    const poor = structuredClone(s);
    const c1 = poor.companies.c1 ?? never();
    c1.cash = 41_595;
    expect(rejected(step(poor, train("u1", "r1")))).toMatch(/cash/i);
  });

  it("rejects a smaller brain than the parent's", () => {
    const s = trained(running(), "u1", "r1", "v-a", plan({ brainSize: "small" }));
    expect(
      rejected(step(s, train("u1", "r2", plan({ brainSize: "tiny", parentVersionId: "v-a" })))),
    ).toMatch(/smaller/i);
    expect(
      ok(step(s, train("u1", "r2", plan({ brainSize: "small", parentVersionId: "v-a" })))).companies
        .c1?.runs.r2,
    ).toBeTruthy();
  });

  it("rejects training before the world starts", () => {
    expect(rejected(step(running(false), train("u1", "r1")))).toMatch(/not running/i);
  });
});

describe("completion at game pace", () => {
  it("holds an early result until the run's last day", () => {
    let s = ok(step(running(), train("u1", "r1")));
    s = ok(step(s, sys({ type: "computeStarted", runId: "r1" })));
    s = ok(step(s, completed("r1", "v-a")));
    expect(s.companies.c1?.runs.r1).toMatchObject({
      status: "training",
      held: { versionId: "v-a", result: RESULT },
    });
    expect(s.companies.c1?.versions).toEqual({});
    s = ticks(s, 12);
    expect(s.companies.c1?.runs.r1?.status).toBe("training");
    s = ticks(s, 1);
    expect(s.companies.c1?.runs.r1).toMatchObject({
      status: "completed",
      versionId: "v-a",
      held: null,
    });
    expect(s.companies.c1?.versions["v-a"]).toMatchObject({
      label: "v1",
      runId: "r1",
      parentId: null,
      nLayer: 4,
      dModel: 128,
      ctxLen: 256,
      createdDay: 13,
      trainedTokens: 12_582_912,
      scores: null,
      published: false,
    });
  });

  it("waits in finalizing when the game time is up before the result", () => {
    let s = ticks(ok(step(running(), train("u1", "r1"))), 13);
    expect(s.companies.c1?.runs.r1?.status).toBe("finalizing");
    expect(activeRun(s.companies.c1 ?? never())?.id).toBe("r1");
    s = ok(step(s, completed("r1", "v-a")));
    expect(s.companies.c1?.runs.r1?.status).toBe("completed");
    expect(s.companies.c1?.versions["v-a"]?.createdDay).toBe(13);
  });

  it("applies a result once", () => {
    const s = trained(running(), "u1", "r1", "v-a");
    expect(rejected(step(s, completed("r1", "v-a")))).toMatch(/already applied/i);
    const held = ok(step(ok(step(running(), train("u1", "r1"))), completed("r1", "v-a")));
    expect(rejected(step(held, completed("r1", "v-a")))).toMatch(/already/i);
  });

  it("numbers versions per company and adds the parent's tokens", () => {
    let s = trained(running(), "u1", "r1", "v-a");
    s = trained(s, "u2", "rb", "v-beta");
    s = trained(s, "u1", "r2", "v-b", plan({ parentVersionId: "v-a" }));
    expect(s.companies.c1?.versions["v-b"]).toMatchObject({
      label: "v2",
      parentId: "v-a",
      trainedTokens: 2 * 12_582_912,
    });
    expect(s.companies.c2?.versions["v-beta"]?.label).toBe("v1");
  });

  it("marks compute as started, repeatably", () => {
    let s = ok(step(running(), train("u1", "r1")));
    s = ok(step(s, sys({ type: "computeStarted", runId: "r1" })));
    s = ok(step(s, sys({ type: "computeStarted", runId: "r1" })));
    expect(s.companies.c1?.runs.r1?.computeStarted).toBe(true);
    expect(rejected(step(s, sys({ type: "computeStarted", runId: "nope" })))).toMatch(/unknown/i);
  });
});

describe("failures and evaluations", () => {
  it("refunds a failed run", () => {
    let s = ok(step(running(), train("u1", "r1")));
    s = ok(
      step(
        s,
        sys({ type: "trainingFailed", runId: "r1", reason: "The computer running it stopped." }),
      ),
    );
    expect(s.companies.c1?.runs.r1).toMatchObject({
      status: "failed",
      failure: "The computer running it stopped.",
    });
    expect(s.companies.c1?.cash).toBe(BALANCE_V1.foundingCash);
    expect(activeRun(s.companies.c1 ?? never())).toBeNull();
    expect(
      rejected(step(s, sys({ type: "trainingFailed", runId: "r1", reason: "again" }))),
    ).toMatch(/finished/i);
  });

  it("fills a version's scores and personality", () => {
    let s = trained(running(), "u1", "r1", "v-a");
    s = ok(
      step(
        s,
        sys({
          type: "evalCompleted",
          versionId: "v-a",
          scores: { language: 41.5, coding: null },
          overall: 20.1,
          personalityWords: ["curious", "brief"],
        }),
      ),
    );
    expect(s.companies.c1?.versions["v-a"]).toMatchObject({
      scores: { language: 41.5, coding: null },
      overall: 20.1,
      personalityWords: ["curious", "brief"],
    });
    expect(
      rejected(
        step(
          s,
          sys({
            type: "evalCompleted",
            versionId: "nope",
            scores: {},
            overall: null,
            personalityWords: [],
          }),
        ),
      ),
    ).toMatch(/unknown/i);
  });
});

describe("runProgress", () => {
  it("runs from 0 to 1 across the run's game days", () => {
    const run = ok(step(running(), train("u1", "r1"))).companies.c1?.runs.r1 ?? never();
    expect(runProgress(run, 0)).toBe(0);
    expect(runProgress(run, 6.5)).toBe(0.5);
    expect(runProgress(run, 13)).toBe(1);
    expect(runProgress(run, 40)).toBe(1);
  });
});

function never(): never {
  throw new Error("missing");
}

describe("world-unique ids", () => {
  it("refuses a version id that a waiting result already holds", () => {
    let s = ok(step(running(), train("u1", "r1")));
    s = ok(step(s, train("u2", "r2")));
    s = ok(step(s, completed("r1", "v-same")));
    expect(rejected(step(s, completed("r2", "v-same")))).toMatch(/already used/i);
  });

  it("refuses a run id another company already uses", () => {
    const s = ok(step(running(), train("u1", "r1")));
    expect(rejected(step(s, train("u2", "r1")))).toMatch(/already used/i);
  });
});

describe("review", () => {
  it("needs an earlier AI to review", () => {
    expect(rejected(step(running(), train("u1", "r1", plan({ review: 0.2 }))))).toMatch(/review/i);
  });
});
