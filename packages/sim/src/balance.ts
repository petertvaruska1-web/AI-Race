import type { DatasetId, GroupId, RegionId, Speed, TrainingPlan } from "@airace/protocol";

export type BalanceVersion = "balance-v2" | "balance-test";
export type BrainSize = { id: string; nLayer: number; dModel: number };
export type LengthOption = { id: string; tokens: number };

export type GroupTraits = {
  /** How much the group cares about each bench category (any subset of the eight). */
  weights: Partial<Record<string, number>>;
  /** How strongly price puts the group off (applied to ln(1 + price/10)). */
  priceSensitivity: number;
  /** -1 wants the smartest AI, +1 the fastest. */
  speedPreference: number;
  /** Share of unhappy customers lost per day (scaled by 1 - satisfaction). */
  churnPerDay: number;
  /** Whether the group buys business seats rather than personal plans. */
  buysBusiness: boolean;
  /** Seats a business account of this group pays for. */
  seatsPerAccount: number;
};

/** The game layer's market and money constants (game-layer design §5–§8), tuned by the balance harness. */
export type Economy = {
  population: Record<RegionId, number>;
  /** Share of people willing to use an AI: start + (end - start) / (1 + exp(-steepness * (years - midYear))). */
  adoption: { start: number; end: number; midYear: number; steepness: number };
  groupMix: Record<RegionId, Record<GroupId, number>>;
  groups: Record<GroupId, GroupTraits>;
  prices: {
    personal: { default: number; min: number; max: number };
    business: { default: number; min: number; max: number };
  };
  noAiUtility: Record<GroupId, number>;
  utility: {
    score: number;
    price: number;
    speed: number;
    reputation: number;
    satisfaction: number;
  };
  diffusion: {
    adoptRate: number;
    switchRate: number;
    wordOfMouth: number;
    awarenessDecay: number;
    launchBoost: number;
  };
  servingCostPerCustomerDayPerMParam: number;
  fame: {
    reputationStart: number;
    reputationRate: number;
    newsWeight: number;
    followersPerHype: number;
    followersDecayPerDay: number;
    viralScore: number;
    viralFollowers: number;
  };
  feedback: { perGroup: number; satisfactionAlpha: number; satisfactionStart: number };
  feedCap: number;
  hypePerDay: number;
  maxProducts: number;
  countdown: { min: number; max: number };
};

/** Every tunable number of the economy. A world records its version (replay). */
export type Balance = {
  version: BalanceVersion;
  /** Real milliseconds per game day at speed 1. */
  tickMsAt1x: number;
  speeds: readonly Speed[];
  foundingCash: number;
  /** Rented cloud compute per game day, in PFLOP (1e15 operations). */
  cloudPflopsPerDay: number;
  /** Dollars per PFLOP of rented compute. */
  cloudPricePerPflop: number;
  factCheckCost: number;
  maxMembers: number;
  brainSizes: readonly BrainSize[];
  attentionSpans: readonly number[];
  lengths: readonly LengthOption[];
  defaultPlan: TrainingPlan;
  economy: Economy;
};

const EVEN_MIX: Record<DatasetId, number> = {
  web: 12,
  books: 12,
  educational: 12,
  conversations: 12,
  code: 12,
  reasoning: 12,
  facts: 12,
  creative: 12,
};

function defaultPlan(brainSize: string, attentionSpan: number, length: string): TrainingPlan {
  return {
    mix: { ...EVEN_MIX },
    cleaning: "standard",
    removeDuplicates: false,
    factCheck: false,
    variety: "natural",
    brainSize,
    attentionSpan,
    length,
    learningStyle: 0.5,
    review: 0,
    probePrompt: "",
    parentVersionId: null,
  };
}

// The first model (tiny, standard length) is 0.104 PFLOP: 13 game days
// (104 s at 1x) and $41,596, about a sixth of the founding cash.
const ECONOMY_V2: Economy = {
  population: { national: 335_000_000 },
  adoption: { start: 0.1, end: 0.6, midYear: 4, steepness: 1.2 },
  groupMix: {
    national: {
      students: 0.14,
      creators: 0.1,
      developers: 0.07,
      businesses: 0.06,
      everyday: 0.58,
      researchers: 0.05,
    },
  },
  groups: {
    students: {
      weights: { knowledge: 0.35, reasoning: 0.25, language: 0.2, instruction: 0.2 },
      priceSensitivity: 1.6,
      speedPreference: 0,
      churnPerDay: 0.006,
      buysBusiness: false,
      seatsPerAccount: 1,
    },
    creators: {
      weights: { creativity: 0.4, language: 0.35, instruction: 0.25 },
      priceSensitivity: 1,
      speedPreference: 0,
      churnPerDay: 0.004,
      buysBusiness: false,
      seatsPerAccount: 1,
    },
    developers: {
      weights: { coding: 0.55, instruction: 0.25, reasoning: 0.2 },
      priceSensitivity: 0.9,
      speedPreference: 0.5,
      churnPerDay: 0.004,
      buysBusiness: true,
      seatsPerAccount: 3,
    },
    businesses: {
      weights: { consistency: 0.3, instruction: 0.3, knowledge: 0.2, language: 0.2 },
      priceSensitivity: 0.5,
      speedPreference: -0.4,
      churnPerDay: 0.003,
      buysBusiness: true,
      seatsPerAccount: 8,
    },
    everyday: {
      weights: { language: 0.45, knowledge: 0.25, instruction: 0.15, creativity: 0.15 },
      priceSensitivity: 1.4,
      speedPreference: 0.4,
      churnPerDay: 0.006,
      buysBusiness: false,
      seatsPerAccount: 1,
    },
    researchers: {
      weights: { reasoning: 0.4, knowledge: 0.3, pattern: 0.3 },
      priceSensitivity: 0.6,
      speedPreference: -0.5,
      churnPerDay: 0.003,
      buysBusiness: true,
      seatsPerAccount: 4,
    },
  },
  prices: {
    personal: { default: 10, min: 1, max: 200 },
    business: { default: 30, min: 5, max: 500 },
  },
  noAiUtility: {
    students: 1,
    creators: 1,
    developers: 1,
    businesses: 1,
    everyday: 1,
    researchers: 1,
  },
  utility: { score: 4, price: 1, speed: 0.5, reputation: 1, satisfaction: 1.5 },
  diffusion: {
    adoptRate: 0.05,
    switchRate: 0.03,
    wordOfMouth: 0.02,
    awarenessDecay: 0.01,
    launchBoost: 0.01,
  },
  servingCostPerCustomerDayPerMParam: 0.0004,
  fame: {
    reputationStart: 50,
    reputationRate: 0.03,
    newsWeight: 0.3,
    followersPerHype: 500,
    followersDecayPerDay: 0.002,
    viralScore: 0.9,
    viralFollowers: 1000,
  },
  feedback: { perGroup: 1, satisfactionAlpha: 0.3, satisfactionStart: 0.5 },
  feedCap: 200,
  hypePerDay: 0.05,
  maxProducts: 3,
  countdown: { min: 3, max: 14 },
};

export const BALANCE_V2: Balance = {
  version: "balance-v2",
  tickMsAt1x: 8000,
  speeds: [1, 2, 4],
  foundingCash: 250_000,
  cloudPflopsPerDay: 0.008,
  cloudPricePerPflop: 400_000,
  factCheckCost: 5000,
  maxMembers: 8,
  brainSizes: [
    { id: "tiny", nLayer: 4, dModel: 128 },
    { id: "small", nLayer: 6, dModel: 192 },
    { id: "medium", nLayer: 8, dModel: 256 },
  ],
  attentionSpans: [128, 256, 512],
  lengths: [
    { id: "short", tokens: 6_291_456 },
    { id: "standard", tokens: 12_582_912 },
    { id: "long", tokens: 25_165_824 },
  ],
  defaultPlan: defaultPlan("tiny", 256, "standard"),
  economy: ECONOMY_V2,
};

/** Fast worlds for end-to-end and quick tests. */
export const BALANCE_TEST: Balance = {
  ...BALANCE_V2,
  version: "balance-test",
  tickMsAt1x: 200,
  brainSizes: [{ id: "tiny", nLayer: 2, dModel: 64 }],
  attentionSpans: [64],
  lengths: [{ id: "short", tokens: 65_536 }],
  cloudPflopsPerDay: 0.000005, // the 2x64 test model trains in 30 game days (6 s at 1x)
  defaultPlan: defaultPlan("tiny", 64, "short"),
  // Customers arrive within a few game days of a launch, so end-to-end tests stay short.
  economy: {
    ...ECONOMY_V2,
    diffusion: { ...ECONOMY_V2.diffusion, adoptRate: 0.5, launchBoost: 0.2 },
  },
};

export function getBalance(version: BalanceVersion): Balance {
  return version === "balance-v2" ? BALANCE_V2 : BALANCE_TEST;
}

/** Share of people willing to use an AI on a game day (an S-curve from `start` toward `end`). */
export function adoptionShare(economy: Economy, day: number): number {
  const { start, end, midYear, steepness } = economy.adoption;
  return start + (end - start) / (1 + Math.exp(-steepness * (day / 365 - midYear)));
}
