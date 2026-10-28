import {
  type DatasetId,
  GROUP_IDS,
  type GroupId,
  type RegionId,
  type Speed,
  type TrainingPlan,
} from "@airace/protocol";

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
  /** Regions that touch; word of mouth and data centers reach them. Symmetric. */
  neighbors: Record<RegionId, readonly RegionId[]>;
  groups: Record<GroupId, GroupTraits>;
  prices: {
    personal: { default: number; min: number; max: number };
    business: { default: number; min: number; max: number };
  };
  noAiUtility: Record<GroupId, number>;
  /**
   * Utility weights. `freePenalty` is what the free plan's limits cost it against
   * paid plans; `businessFit` is what the business plan's team features are worth
   * to the groups that buy it.
   */
  utility: {
    score: number;
    price: number;
    speed: number;
    reputation: number;
    satisfaction: number;
    freePenalty: number;
    businessFit: number;
    /** What unanswered questions cost: times (served share − 1). */
    reliability: number;
  };
  diffusion: {
    adoptRate: number;
    switchRate: number;
    wordOfMouth: number;
    awarenessDecay: number;
    launchBoost: number;
    /** The share of a cell's word of mouth that also reaches each neighbouring region. */
    neighborSpill: number;
  };
  servingCostPerCustomerDayPerMParam: number;
  /**
   * Ad campaigns: daily budgets and lengths allowed, how many at once, dollars per
   * person reached in a day, how much of a full reach turns into awareness, and the
   * hype a dollar adds to a launch counting down.
   */
  /**
   * Serving capacity in PFLOP a day: the cloud's starting room and how fast it can
   * grow, owned compute's price as a share of the cloud's, and what an overload
   * does to satisfaction and the news.
   */
  compute: {
    cloudStartPflops: number;
    cloudGrowthPerDay: number;
    ownedServingPriceShare: number;
    overloadSatisfactionHit: number;
    overloadNewsBelow: number;
    overloadNewsGapDays: number;
  };
  campaigns: {
    budget: { min: number; max: number };
    days: { min: number; max: number };
    maxRunning: number;
    costPerReach: number;
    awarenessPerReach: number;
    hypePerDollar: number;
  };
  fame: {
    reputationStart: number;
    reputationRate: number;
    newsWeight: number;
    followersPerHype: number;
    followersDecayPerDay: number;
    viralScore: number;
    viralFollowers: number;
    /** How far each news story moves a company's news mood (a moving average). */
    newsAlpha: number;
  };
  /** The sentiment (-1..1) of good-news stories; a viral post carries its own. */
  newsSentiment: { launch: number; customers: number; firstPaying: number };
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

/** The country's mix of groups among people willing to use an AI. */
const NATIONAL_MIX: Record<GroupId, number> = {
  students: 0.14,
  creators: 0.1,
  developers: 0.07,
  businesses: 0.06,
  everyday: 0.58,
  researchers: 0.05,
};

/** The national mix tilted by a region's multipliers (unlisted groups x1), summing to 1. */
function tilted(by: Partial<Record<GroupId, number>>): Record<GroupId, number> {
  const raw = GROUP_IDS.map((g) => NATIONAL_MIX[g] * (by[g] ?? 1));
  const sum = raw.reduce((a, b) => a + b, 0);
  return Object.fromEntries(GROUP_IDS.map((g, i) => [g, (raw[i] ?? 0) / sum])) as Record<
    GroupId,
    number
  >;
}

// The first model (tiny, standard length) is 0.104 PFLOP: 13 game days
// (104 s at 1x) and $41,596, about a sixth of the founding cash.
const ECONOMY_V2: Economy = {
  // The census divisions' people (2023 estimates).
  population: {
    newEngland: 15_100_000,
    midAtlantic: 42_000_000,
    greatLakes: 47_200_000,
    plains: 21_800_000,
    southAtlantic: 68_700_000,
    midSouth: 19_800_000,
    southCentral: 43_000_000,
    mountain: 25_500_000,
    pacific: 53_000_000,
  },
  adoption: { start: 0.1, end: 0.6, midYear: 4, steepness: 1.2 },
  groupMix: {
    newEngland: tilted({ students: 1.4, researchers: 1.8, developers: 1.1 }),
    midAtlantic: tilted({ businesses: 1.5, creators: 1.2, researchers: 1.2 }),
    greatLakes: tilted({ businesses: 1.1 }),
    plains: tilted({ everyday: 1.15, developers: 0.7, researchers: 0.7 }),
    southAtlantic: tilted({ businesses: 1.1 }),
    midSouth: tilted({ everyday: 1.2, creators: 0.9, developers: 0.6, researchers: 0.6 }),
    southCentral: tilted({ businesses: 1.2 }),
    mountain: tilted({ developers: 1.2 }),
    pacific: tilted({ developers: 1.9, creators: 1.4, researchers: 1.3, everyday: 0.85 }),
  },
  neighbors: {
    newEngland: ["midAtlantic"],
    midAtlantic: ["newEngland", "greatLakes", "southAtlantic"],
    greatLakes: ["midAtlantic", "plains", "midSouth", "southAtlantic"],
    plains: ["greatLakes", "midSouth", "southCentral", "mountain"],
    southAtlantic: ["midAtlantic", "greatLakes", "midSouth"],
    midSouth: ["greatLakes", "plains", "southAtlantic", "southCentral"],
    southCentral: ["plains", "midSouth", "mountain"],
    mountain: ["plains", "southCentral", "pacific"],
    pacific: ["mountain"],
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
  // Tuned by the balance harness (test/balance.test.ts), one reason each:
  // - noAiUtility 6: a typical first AI alone in the market wins about 0.4% of the
  //   people willing within a year (a small startup), not the half it won at 1.
  noAiUtility: {
    students: 6,
    creators: 6,
    developers: 6,
    businesses: 6,
    everyday: 6,
    researchers: 6,
  },
  // - score 2: an AI twice as good wins about 5x the customers by day 90 (15x at 4).
  // - freePenalty 0.5: price-sensitive groups prefer free, the rest pay (about 40% pay).
  // - businessFit 1: businesses, developers and researchers prefer the business plan.
  utility: {
    score: 2,
    price: 1,
    speed: 0.5,
    reputation: 1,
    satisfaction: 1.5,
    freePenalty: 0.5,
    businessFit: 1,
    reliability: 3,
  },
  // - wordOfMouth 0.5 and launchBoost 0.1: about 1,300 customers on launch day and
  //   steady growth all year; at 0.02 and 0.01 a small company never got known.
  diffusion: {
    adoptRate: 0.05,
    switchRate: 0.03,
    wordOfMouth: 0.5,
    awarenessDecay: 0.01,
    launchBoost: 0.1,
    neighborSpill: 0.15,
  },
  // - 0.01: serving costs a tiny AI about 5% of its income; a medium AI costs four times
  //   as much per customer, so size is a real trade-off (at 0.0004 it was free).
  servingCostPerCustomerDayPerMParam: 0.01,
  compute: {
    cloudStartPflops: 0.00025,
    cloudGrowthPerDay: 0.2,
    ownedServingPriceShare: 0.3,
    overloadSatisfactionHit: 0.05,
    overloadNewsBelow: 0.9,
    overloadNewsGapDays: 30,
  },
  campaigns: {
    budget: { min: 100, max: 100_000 },
    days: { min: 7, max: 90 },
    maxRunning: 5,
    costPerReach: 0.05,
    awarenessPerReach: 0.5,
    hypePerDollar: 0.000002,
  },
  fame: {
    reputationStart: 50,
    reputationRate: 0.03,
    newsWeight: 0.3,
    followersPerHype: 500,
    followersDecayPerDay: 0.002,
    viralScore: 0.9,
    viralFollowers: 1000,
    newsAlpha: 0.3,
  },
  newsSentiment: { launch: 0.5, customers: 0.6, firstPaying: 0.8 },
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
