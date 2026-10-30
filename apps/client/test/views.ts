import { type Command, RivalPack, type WorldView } from "@airace/protocol";
import {
  BALANCE_TEST,
  type BalanceVersion,
  createWorld,
  type Input,
  projectView,
  step,
  type WorldState,
} from "@airace/sim";
import RIVAL_PACK_JSON from "../../../packages/protocol/fixtures/rivals/rivals.json";

// Real world views, made by the sim, for the client tests.

export const LOGO = { shape: "circle", symbol: "spark", initials: "AC" } as const;
export const PALETTE = { primary: "#4f8cff", secondary: "#ffb020" };

export function testWorld(balance: BalanceVersion = "balance-test"): WorldState {
  return createWorld({
    worldId: "w1",
    name: "Friday night",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 1,
    balance,
    inviteCode: "ABC234",
  });
}

export function apply(s: WorldState, input: Input): WorldState {
  const r = step(s, input);
  if (!r.ok) throw new Error(r.reason);
  return r.state;
}

export const command = (userId: string, cmd: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: cmd as Command,
  ...(newId ? { newId } : {}),
});

/** A running world where u1 owns company c1. */
export function runningWorld(balance: BalanceVersion = "balance-test"): WorldState {
  let s = testWorld(balance);
  s = apply(
    s,
    command("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }, "c1"),
  );
  return apply(s, command("u1", { type: "startWorld" }));
}

export function viewOf(s: WorldState, userId = "u1"): WorldView {
  return projectView(s, userId);
}

/** A running world where c1 has trained version v1 (scored when `scores` is given). */
export function trainedWorld(
  scores?: Record<string, number | null>,
  conversations = 12,
): WorldState {
  const plan = {
    ...BALANCE_TEST.defaultPlan,
    mix: { ...BALANCE_TEST.defaultPlan.mix, conversations },
  };
  let s = apply(runningWorld(), command("u1", { type: "startTraining", plan }, "r1"));
  s = apply(s, {
    kind: "system",
    event: {
      type: "trainingCompleted",
      runId: "r1",
      versionId: "v1",
      result: { steps: 8, finalLoss: 4, heldoutLosses: {}, unstable: false },
    },
  });
  const end = s.companies.c1?.runs.r1?.endDay ?? 0;
  while (s.day < end) s = apply(s, { kind: "tick" });
  if (scores) {
    s = apply(s, {
      kind: "system",
      event: {
        type: "evalCompleted",
        versionId: "v1",
        scores,
        overall: 30,
        personalityWords: ["curious"],
      },
    });
  }
  return s;
}

/** Scores like the M1 gate's typical first model. */
export const SCORES = {
  language: 60,
  reasoning: 8,
  pattern: 80,
  knowledge: 44,
  coding: 38,
  creativity: 2,
  consistency: 37,
  instruction: 63,
};

export function ticks(s: WorldState, n: number): WorldState {
  let out = s;
  for (let i = 0; i < n; i++) out = apply(out, { kind: "tick" });
  return out;
}

/** c1's scored v1 launched as "Spark" on the free and personal plans, live for `days` days. */
export function launchedWorld(days = 5): WorldState {
  const s = apply(
    trainedWorld(SCORES),
    command(
      "u1",
      {
        type: "launchProduct",
        versionId: "v1",
        name: "Spark",
        plans: ["free", "personal"],
        countdownDays: 3,
      },
      "p1",
    ),
  );
  return ticks(s, 3 + days);
}

/** A scored customer-feedback batch for product p1: one reply per [group, score, reply]. */
export function feedback(key: string, replies: [string, number, string][]): Input {
  return {
    kind: "system",
    event: {
      type: "feedbackScored",
      productId: "p1",
      key,
      results: replies.map(([group, score, reply]) => ({
        group,
        questionId: `q-${group}`,
        question: "What is the capital of France?",
        reply,
        score,
      })),
    },
  } as Input;
}

/** The shared rival pack fixture, on the test world's balance. */
export const RIVAL_PACK = RivalPack.parse(RIVAL_PACK_JSON);

/** `launchedWorld` with the fixture's rivals joined and `days` more days, so their first AIs are on sale. */
export function rivalWorld(days = 10): WorldState {
  const s = apply(launchedWorld(), {
    kind: "system",
    event: { type: "rivalsJoined", pack: RIVAL_PACK },
  });
  return ticks(s, days);
}
