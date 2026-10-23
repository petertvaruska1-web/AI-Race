import type { Command, TrainingPlan } from "@airace/protocol";
import fc from "fast-check";
import {
  BALANCE_V1,
  createWorld,
  type Input,
  PALETTE_PRESETS,
  step,
  type WorldState,
} from "../src/index.ts";

// Small id pools make collisions (rejected inputs) and hits (completions of
// real runs) both common. Ids are distinctive so a text search finds only them.
export const USERS = ["u1", "u2", "u3", "u4"] as const;
const COMPANY_IDS = ["CO_1", "CO_2", "CO_3", "CO_4", "CO_5"];
export const RUN_IDS = ["RUN_1", "RUN_2", "RUN_3", "RUN_4", "RUN_5", "RUN_6", "RUN_7", "RUN_8"];
export const VERSION_IDS = ["VER_1", "VER_2", "VER_3", "VER_4", "VER_5", "VER_6", "VER_7", "VER_8"];
const NAMES = ["Acme", "acme", "Beta Labs", "Cortex", "Delta", "A"];

export function newWorld(): WorldState {
  return createWorld({
    worldId: "w1",
    name: "Property world",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 11,
    balance: "balance-v1",
    inviteCode: "INVITE-42",
  });
}

// Weighted towards the two players the setup prefix seats.
const user = fc.constantFrom("u1", "u1", "u2", "u2", "u3", "u4");

const plan = (userId: string): fc.Arbitrary<TrainingPlan> =>
  fc
    .record({
      brainSize: fc.constantFrom("tiny", "tiny", "small", "huge"),
      length: fc.constantFrom("short", "standard"),
      attentionSpan: fc.constantFrom(128, 256, 256, 300),
      factCheck: fc.boolean(),
      allZero: fc.nat(9).map((x) => x === 0),
      prompt: fc.nat(3),
      parent: fc.option(fc.constantFrom(...VERSION_IDS), { freq: 2 }),
    })
    .map((r) => ({
      ...BALANCE_V1.defaultPlan,
      mix: r.allZero
        ? {
            web: 0,
            books: 0,
            educational: 0,
            conversations: 0,
            code: 0,
            reasoning: 0,
            facts: 0,
            creative: 0,
          }
        : { ...BALANCE_V1.defaultPlan.mix },
      brainSize: r.brainSize,
      length: r.length,
      attentionSpan: r.attentionSpan,
      factCheck: r.factCheck,
      // A prompt unique to its owner, so finding it in a view is a leak.
      probePrompt: r.prompt === 0 ? "" : `SECRET_PROBE_${userId}_${r.prompt}`,
      parentVersionId: r.parent,
    }));

const command = (userId: string, cmd: unknown, newId?: string): Input => ({
  kind: "command",
  userId,
  cmd: cmd as Command,
  ...(newId === undefined ? {} : { newId }),
});

const commandInput: fc.Arbitrary<Input> = user.chain((u) =>
  fc.oneof(
    fc
      .record({
        name: fc.constantFrom(...NAMES),
        palette: fc.constantFrom(...PALETTE_PRESETS.map((p) => p.palette), {
          primary: "#101010",
          secondary: "#ffffff",
        }),
        id: fc.constantFrom(...COMPANY_IDS),
      })
      .map((r) =>
        command(
          u,
          {
            type: "createCompany",
            name: r.name,
            logo: { shape: "circle", symbol: null, initials: "" },
            palette: r.palette,
          },
          r.id,
        ),
      ),
    fc.constant(command(u, { type: "startWorld" })),
    fc.constantFrom(1, 2, 3, 4).map((speed) => command(u, { type: "setSpeed", speed })),
    // Mostly resumes: a paused world never reaches a run's last day.
    fc.nat(3).map((x) => command(u, { type: "setPaused", paused: x === 0 })),
    {
      arbitrary: fc
        .tuple(plan(u), fc.constantFrom(...RUN_IDS))
        .map(([p, id]) => command(u, { type: "startTraining", plan: p }, id)),
      weight: 4,
    },
  ),
);

const systemInput: fc.Arbitrary<Input> = fc.oneof(
  fc.tuple(user, fc.constantFrom("Ben", "Cy", "Dee")).map(
    ([userId, displayName]): Input => ({
      kind: "system",
      event: { type: "memberJoined", userId, displayName },
    }),
  ),
  fc.tuple(user, fc.boolean()).map(
    ([userId, online]): Input => ({
      kind: "system",
      event: { type: "presence", userId, online },
    }),
  ),
  fc
    .constantFrom(...RUN_IDS)
    .map((runId): Input => ({ kind: "system", event: { type: "computeStarted", runId } })),
  {
    weight: 4,
    arbitrary: fc.tuple(fc.constantFrom(...RUN_IDS), fc.constantFrom(...VERSION_IDS)).map(
      ([runId, versionId]): Input => ({
        kind: "system",
        event: {
          type: "trainingCompleted",
          runId,
          versionId,
          result: { steps: 100, finalLoss: 4.2, heldoutLosses: { web: 4.5 }, unstable: false },
        },
      }),
    ),
  },
  fc.constantFrom(...RUN_IDS).map(
    (runId): Input => ({
      kind: "system",
      event: { type: "trainingFailed", runId, reason: "The computer running it stopped." },
    }),
  ),
  fc.tuple(fc.constantFrom(...VERSION_IDS), fc.double({ min: 0, max: 100, noNaN: true })).map(
    ([versionId, overall]): Input => ({
      kind: "system",
      event: {
        type: "evalCompleted",
        versionId,
        scores: { language: overall },
        overall,
        personalityWords: ["curious"],
      },
    }),
  ),
);

const TICK: Input = { kind: "tick" };

/** Random command, tick and system sequences across 1–4 users; many are invalid. */
export function arbInputs(n: number): fc.Arbitrary<Input[]> {
  // A short setup prefix makes running worlds with companies common.
  const setup = fc.constant<Input[]>([
    command(
      "u1",
      {
        type: "createCompany",
        name: "Host Co",
        logo: { shape: "hexagon", symbol: "spark", initials: "HC" },
        palette: { primary: "#4f8cff", secondary: "#ffb020" },
      },
      "CO_0",
    ),
    { kind: "system", event: { type: "memberJoined", userId: "u2", displayName: "Ben" } },
    command("u1", { type: "startWorld" }),
  ]);
  const body = fc.array(
    fc.oneof(
      { arbitrary: fc.constant(TICK), weight: 4 },
      { arbitrary: commandInput, weight: 3 },
      { arbitrary: systemInput, weight: 3 },
    ),
    {
      minLength: Math.floor(n / 2),
      maxLength: n,
    },
  );
  return fc
    .tuple(
      fc.nat(9).map((x) => x > 0),
      setup,
      body,
    )
    .map(([withSetup, pre, rest]) => (withSetup ? [...pre, ...rest] : rest));
}

/** Applies inputs in order, skipping rejected ones. */
export function runInputs(state: WorldState, inputs: readonly Input[]): WorldState {
  let s = state;
  for (const input of inputs) {
    const r = step(s, input);
    if (r.ok) s = r.state;
  }
  return s;
}
