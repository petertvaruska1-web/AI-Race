import { type Command, GROUP_IDS, PLAN_IDS, type TrainingPlan } from "@airace/protocol";
import fc from "fast-check";
import {
  BALANCE_V2,
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
export const PRODUCT_IDS = ["PROD_1", "PROD_2", "PROD_3", "PROD_4"];
export const CAMPAIGN_IDS = ["AD_1", "AD_2"];
const PRODUCT_NAMES = ["Spark", "spark", "Nova", "<b>Bolt</b> 🚀", "X"];
/** Replies a real model might write; posts must carry them unchanged. */
const REPLIES = ["<b>x</b> <|end|>", "", "😀😀", "x".repeat(2000), "Paris."];

export function newWorld(): WorldState {
  return createWorld({
    worldId: "w1",
    name: "Property world",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 11,
    balance: "balance-v2",
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
      ...BALANCE_V2.defaultPlan,
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
        : { ...BALANCE_V2.defaultPlan.mix },
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
    {
      arbitrary: fc
        .record({
          version: fc.constantFrom(...VERSION_IDS),
          name: fc.constantFrom(...PRODUCT_NAMES),
          plans: fc.subarray([...PLAN_IDS]),
          days: fc.constantFrom(3, 3, 14, 2),
          id: fc.constantFrom(...PRODUCT_IDS),
        })
        .map((r) =>
          command(
            u,
            {
              type: "launchProduct",
              versionId: r.version,
              name: r.name,
              plans: r.plans,
              countdownDays: r.days,
            },
            r.id,
          ),
        ),
      weight: 2,
    },
    fc
      .record({
        plan: fc.constantFrom(...PLAN_IDS),
        product: fc.option(fc.constantFrom(...PRODUCT_IDS), { freq: 3 }),
      })
      .map((r) => command(u, { type: "setPlan", plan: r.plan, productId: r.product })),
    fc
      .record({
        plan: fc.constantFrom("personal", "business"),
        price: fc.constantFrom(1, 10, 25, 200, 201, 0, 500),
      })
      .map((r) => command(u, { type: "setPrice", plan: r.plan, price: r.price })),
    fc
      .constantFrom(...PRODUCT_IDS)
      .map((id) => command(u, { type: "retireProduct", productId: id })),
    fc
      .record({
        region: fc.constantFrom("pacific", "midSouth", "all"),
        group: fc.constantFrom("developers", "everyday", "all"),
        budget: fc.constantFrom(100, 2000, 99, 100_000),
        days: fc.constantFrom(7, 30, 6),
        id: fc.constantFrom(...CAMPAIGN_IDS),
      })
      .map((r) =>
        command(
          u,
          {
            type: "startCampaign",
            region: r.region,
            group: r.group,
            budgetPerDay: r.budget,
            days: r.days,
          },
          r.id,
        ),
      ),
    fc
      .constantFrom(...CAMPAIGN_IDS)
      .map((id) => command(u, { type: "stopCampaign", campaignId: id })),
  ),
);

const feedbackScored = (productId: string, key: string, results: unknown[]): Input =>
  ({ kind: "system", event: { type: "feedbackScored", productId, key, results } }) as Input;

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
        scores: { language: overall, coding: overall / 2 },
        overall,
        personalityWords: ["curious"],
        traits: { curious: overall / 100 },
      },
    }),
  ),
  {
    weight: 3,
    arbitrary: fc
      .record({
        product: fc.constantFrom(...PRODUCT_IDS),
        key: fc.constantFrom("launch", "w1", "w2", "w3", "w4"),
        results: fc.array(
          fc.record({
            group: fc.constantFrom(...GROUP_IDS),
            score: fc.oneof(fc.double({ min: 0, max: 1, noNaN: true }), fc.constantFrom(0, 1, 1.5)),
            reply: fc.constantFrom(...REPLIES),
          }),
          { maxLength: 6 },
        ),
      })
      .map((r) =>
        feedbackScored(
          r.product,
          r.key,
          r.results.map((x) => ({
            group: x.group,
            questionId: `q-${x.group}`,
            question: "What would you ask an AI?",
            reply: x.reply,
            score: x.score,
          })),
        ),
      ),
  },
);

const TICK: Input = { kind: "tick" };

/** Host Co trains, scores and launches "Spark" on PROD_1, which goes live and gets feedback. */
const LAUNCH: Input[] = [
  command(
    "u1",
    {
      type: "startTraining",
      plan: { ...BALANCE_V2.defaultPlan, mix: { ...BALANCE_V2.defaultPlan.mix } },
    },
    "RUN_1",
  ),
  {
    kind: "system",
    event: {
      type: "trainingCompleted",
      runId: "RUN_1",
      versionId: "VER_1",
      result: { steps: 100, finalLoss: 4.2, heldoutLosses: { web: 4.5 }, unstable: false },
    },
  },
  ...Array.from({ length: 14 }, () => TICK),
  {
    kind: "system",
    event: {
      type: "evalCompleted",
      versionId: "VER_1",
      scores: { language: 60, pattern: 80, coding: 38 },
      overall: 42,
      personalityWords: ["curious"],
      traits: { curious: 0.4 },
    },
  },
  command(
    "u1",
    {
      type: "launchProduct",
      versionId: "VER_1",
      name: "Spark",
      plans: ["free", "personal"],
      countdownDays: 3,
    },
    "PROD_1",
  ),
  TICK,
  TICK,
  TICK,
  feedbackScored(
    "PROD_1",
    "launch",
    GROUP_IDS.map((group, i) => ({
      group,
      questionId: `q-${group}`,
      question: "What would you ask an AI?",
      reply: REPLIES[i % REPLIES.length],
      score: i / 5,
    })),
  ),
];

/**
 * Random command, tick and system sequences across 1–4 users; many are invalid.
 * A third of the worlds with the setup prefix also launch a product first.
 */
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
      fc.nat(2).map((x) => x === 0),
      setup,
      body,
    )
    .map(([withSetup, launched, pre, rest]) =>
      withSetup ? [...pre, ...(launched ? LAUNCH : []), ...rest] : rest,
    );
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
