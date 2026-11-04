import {
  type AppKind,
  type BattleSkill,
  type DatasetId,
  GROUP_IDS,
  type GroupId,
  type MixId,
  type PlanId,
  type RegionId,
  type RivalStyle,
  type Speed,
  type SponsorKind,
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
    /** What a nearby data center is worth: times closeness, more for groups that want speed. */
    nearby: number;
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
  /** One size of data center: its price, build time, capacity (PFLOP a day) and upkeep. */
  dataCenter: {
    cost: number;
    buildDays: number;
    pflopsPerDay: number;
    upkeepPerMonth: number;
    maxPerCompany: number;
  };
  /**
   * The rescue: cash restored to `cushion`; a loan's monthly interest and term; an
   * investor's share; days to choose before the game takes the loan.
   */
  rescue: {
    cushion: number;
    loanMonthlyRate: number;
    loanTermMonths: number;
    investorStake: number;
    decideDays: number;
  };
  /** Closeness to an open data center: in the region, or in a neighbouring one. */
  nearby: { region: number; neighbor: number };
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
  /** Rival companies (game-layer design §9): their money, release schedule and habits. */
  rivals: RivalBalance;
  /** A table's #1 makes news at most once in this many days. */
  league: { newsGapDays: number };
  /** Battles against rivals (game-layer design §10): fee, rounds, cooldown and ranking points. */
  battles: {
    fee: number;
    rounds: number;
    cooldownDays: number;
    /** A total score difference within this is a draw. */
    drawMargin: number;
    pointsStart: number;
    pointsK: number;
    sentiment: { win: number; draw: number; loss: number };
    followers: { win: number; loss: number };
    kept: number;
  };
  /** The investment fund's month-end returns (playtest pack §5). */
  fund: {
    meanMonthly: number;
    sdMonthly: number;
    /** The chance of a dip in any month, and how deep one goes. */
    dipChance: number;
    dipMin: number;
    dipMax: number;
  };
  /** Office and support: base × R^exponent a month for monthly revenue R (C35). */
  overhead: { base: number; exponent: number };
  /** Sponsorships: what each kind of event costs, how long it runs, whose awareness it lifts. */
  sponsorships: {
    kinds: Record<SponsorKind, { price: number; days: number; groups: GroupId[]; rate: number }>;
  };
  /** Partner apps: what each costs and earns, and how its users follow the AI's measured skill. */
  apps: {
    kinds: Record<
      AppKind,
      {
        skill: BattleSkill;
        group: GroupId;
        setup: number;
        upkeep: number;
        /** Users a perfect AI with a 3-star rating settles at. */
        capacity: number;
        revenuePerUser: number;
      }
    >;
    /** Users close this share of the gap to their target each day. */
    dailyGrowth: number;
    /** fit = (score/100)^fitPower; an unmeasured skill fits at unmeasuredFit. */
    fitPower: number;
    unmeasuredFit: number;
    reviewsKept: number;
  };
  /** A launch keynote's fee; below `stumble` the live demo is a flop. */
  keynote: { fee: number; stumble: number };
  /** Tutoring packs: each adds `packShare` of the run's tokens from one dataset, plus a fee. */
  tutoring: { packFee: number; packShare: number };
  /** Fact-check plus: dollars per thousand tokens of text checked. */
  expertCheckPerThousand: number;
  /** Discoveries: how big a difference counts, how many an eval reports and how many are kept. */
  discoveries: {
    minPoints: number;
    minZ: number;
    minPeers: number;
    perEval: number;
    publishFollowers: number;
    kept: number;
  };
  /** Interviews: how often, how many answers, and how the press reacts. */
  interviews: {
    cooldownDays: number;
    maxAnswers: number;
    followersPerPoint: number;
    viralBelow: number;
    viralFollowers: number;
    kept: number;
  };
  /** The goal: how many months of recent profit a company's value counts. */
  goal: { valueMonths: number };
  /** The data-quality scandal: how big a company must be for it to make news, and its cost. */
  scandal: { minCustomers: number; satisfactionHit: number; sentiment: number };
  /** The player's notebook: how many words it holds. */
  notebook: { maxWords: number };
  /** Learn from customers: dollars per loved conversation used, and how many it needs. */
  lovedConversationFee: number;
  lovedMin: number;
  lovedMax: number;
  /** A feedback reply scoring this or more is one customers loved. */
  lovedScore: number;
  /** Looks by id, with their price (the free ones cost 0). */
  looks: Record<string, number>;
  /** Playground games played live for the press (playtest pack §4.3). */
  showcases: {
    fee: number;
    cooldownDays: number;
    /** Followers × (1 + followers × (2p − 1)) for a run that got a share p of the way. */
    followers: number;
    /** Below this share of the way, the news says it stumbled. */
    stumble: number;
    kept: number;
  };
};

export type RivalBalance = {
  /** Times the founding cash, by style; the arch-rival gets `archBonus` times more. */
  startCash: Record<RivalStyle, number>;
  archBonus: number;
  /** Days after joining of each rival's first release (the arch-rival first). */
  firstRelease: readonly [number, number, number, number];
  releaseGapDays: number;
  jitterDays: number;
  countdownDays: Record<RivalStyle, number>;
  plans: Record<RivalStyle, readonly PlanId[]>;
  /** Times each plan's default price. */
  priceFactor: Record<RivalStyle, number>;
  reviewEveryDays: number;
  priceStep: number;
  ad: {
    shareOfCash: number;
    maxPerDay: Record<RivalStyle, number>;
    days: number;
    minCash: number;
  };
  dataCenterAtServingPerMonth: number;
  trashTalkGapDays: number;
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

const EVEN_MIX: Record<MixId, number> = {
  web: 12,
  books: 12,
  educational: 12,
  conversations: 12,
  code: 12,
  reasoning: 12,
  facts: 12,
  creative: 12,
  notebook: 0,
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
    tutoring: null,
    learnFromCustomers: false,
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
    nearby: 0.6,
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
  dataCenter: {
    cost: 1_500_000,
    buildDays: 60,
    pflopsPerDay: 0.05,
    upkeepPerMonth: 30_000,
    maxPerCompany: 6,
  },
  nearby: { region: 1, neighbor: 0.4 },
  rescue: {
    cushion: 100_000,
    loanMonthlyRate: 0.04,
    loanTermMonths: 24,
    investorStake: 0.25,
    decideDays: 14,
  },
  // - costPerReach 0.15: ads pay back in about 3 months (a $350k blitz in about 87 days) and a
  //   30-day Pacific campaign lifts the Pacific's share 1.34 times; at 0.05 one blitz left a
  //   company $633k richer within 110 days, so advertising was free money.
  campaigns: {
    budget: { min: 100, max: 100_000 },
    days: { min: 7, max: 90 },
    maxRunning: 5,
    costPerReach: 0.15,
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
  rivals: {
    startCash: { premium: 3, budget: 1, specialist: 1.5, consumer: 2 },
    archBonus: 1.5,
    firstRelease: [12, 25, 40, 55],
    releaseGapDays: 110,
    jitterDays: 10,
    countdownDays: { premium: 10, budget: 3, specialist: 7, consumer: 5 },
    plans: {
      premium: ["personal", "business"],
      budget: ["free", "personal"],
      specialist: ["personal", "business"],
      consumer: ["free", "personal", "business"],
    },
    priceFactor: { premium: 1.5, budget: 0.5, specialist: 1.1, consumer: 0.8 },
    reviewEveryDays: 30,
    priceStep: 0.1,
    ad: {
      shareOfCash: 0.01,
      // Tuned by the rival balance targets: the crowd-pleaser's ads at $4,000 a day pushed
      // it to 48% of all AI users within a year; $2,000 keeps it under half.
      maxPerDay: { premium: 3000, budget: 500, specialist: 1000, consumer: 2000 },
      days: 30,
      minCash: 200_000,
    },
    dataCenterAtServingPerMonth: 60_000,
    trashTalkGapDays: 21,
  },
  league: { newsGapDays: 14 },
  battles: {
    fee: 5_000,
    rounds: 5,
    cooldownDays: 7,
    // 0.1: with partial credit for near misses, only almost equal totals are a draw.
    drawMargin: 0.1,
    pointsStart: 1000,
    pointsK: 32,
    sentiment: { win: 0.4, draw: 0.1, loss: -0.3 },
    followers: { win: 0.05, loss: -0.02 },
    kept: 20,
  },
  // About 10% a year with a 1-in-12 dip of 8-15%: parking cash is a choice, not free money.
  fund: { meanMonthly: 0.008, sdMonthly: 0.03, dipChance: 1 / 12, dipMin: 0.08, dipMax: 0.15 },
  // A typical company has $1.7M revenue a month at 180,000 customers after a year, with costs
  // of about 5% (harness, 2026-11-03). 0.0407 × R^1.1 is 12% of a $60k month, 15% of $465k,
  // 17% of $1.7M and 21% of $14M: bigger companies pay a bigger share, and still grow rich.
  overhead: { base: 0.0407, exponent: 1.1 },
  // Priced against a typical company's $1.7M a month after a year (harness, 2026-11-03).
  // `rate` is the share of the unaware crowd that hears of the company each day it runs.
  sponsorships: {
    kinds: {
      stadium: { price: 400_000, days: 90, groups: ["everyday", "students"], rate: 0.004 },
      conference: {
        price: 150_000,
        days: 30,
        groups: ["developers", "businesses", "researchers"],
        rate: 0.008,
      },
      festival: { price: 100_000, days: 60, groups: ["creators", "everyday"], rate: 0.005 },
    },
  },
  // A good app pays back its setup in about 6-12 months; a weak AI's app never does.
  apps: {
    kinds: {
      homework: {
        skill: "knowledge",
        group: "students",
        setup: 250_000,
        upkeep: 20_000,
        capacity: 60_000,
        revenuePerUser: 2,
      },
      stories: {
        skill: "creativity",
        group: "everyday",
        setup: 200_000,
        upkeep: 15_000,
        capacity: 80_000,
        revenuePerUser: 1.5,
      },
      code: {
        skill: "coding",
        group: "developers",
        setup: 300_000,
        upkeep: 25_000,
        capacity: 25_000,
        revenuePerUser: 5,
      },
      puzzles: {
        skill: "pattern",
        group: "students",
        setup: 150_000,
        upkeep: 10_000,
        capacity: 40_000,
        revenuePerUser: 1.2,
      },
      office: {
        skill: "instruction",
        group: "businesses",
        setup: 400_000,
        upkeep: 30_000,
        capacity: 20_000,
        revenuePerUser: 8,
      },
    },
    dailyGrowth: 0.03,
    fitPower: 1.5,
    unmeasuredFit: 0.1,
    reviewsKept: 6,
  },
  keynote: { fee: 50_000, stumble: 0.3 },
  tutoring: { packFee: 15_000, packShare: 0.1 },
  // $4 per thousand tokens: about $100,000 to check a long run's text.
  expertCheckPerThousand: 4,
  notebook: { maxWords: 5_000 },
  scandal: { minCustomers: 100_000, satisfactionHit: 0.05, sentiment: -0.7 },
  goal: { valueMonths: 24 },
  interviews: {
    cooldownDays: 14,
    maxAnswers: 5,
    followersPerPoint: 0.1,
    viralBelow: 0.1,
    viralFollowers: 1_000,
    kept: 10,
  },
  discoveries: {
    minPoints: 25,
    minZ: 2,
    minPeers: 3,
    perEval: 3,
    publishFollowers: 0.03,
    kept: 30,
  },
  lovedConversationFee: 150,
  lovedMin: 10,
  lovedMax: 400,
  lovedScore: 0.75,
  // Status for the rich: a supernova orb is more than a month of a typical company's revenue.
  looks: {
    "orb:classic": 0,
    "orb:nebula": 100_000,
    "orb:crystal": 250_000,
    "orb:plasma": 500_000,
    "orb:aurora": 1_000_000,
    "orb:supernova": 2_500_000,
    "logo:still": 0,
    "logo:pulse": 50_000,
    "logo:shimmer": 150_000,
    "logo:orbit": 400_000,
  },
  // A showcase costs four battles: it is a press event, and a great run gains 8% followers.
  showcases: { fee: 20_000, cooldownDays: 7, followers: 0.08, stumble: 0.3, kept: 20 },
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
    // Rivals release within days, so end-to-end tests meet them; countdowns are the shortest allowed.
    rivals: {
      ...ECONOMY_V2.rivals,
      firstRelease: [1, 2, 3, 4],
      releaseGapDays: 20,
      jitterDays: 0,
      countdownDays: { premium: 3, budget: 3, specialist: 3, consumer: 3 },
      trashTalkGapDays: 2,
    },
    battles: { ...ECONOMY_V2.battles, rounds: 3, cooldownDays: 2 },
    showcases: { ...ECONOMY_V2.showcases, cooldownDays: 2 },
    // The first scored AI already shows a discovery, so end-to-end tests meet one.
    discoveries: { ...ECONOMY_V2.discoveries, minPoints: 1, minZ: 0, minPeers: 1 },
    interviews: { ...ECONOMY_V2.interviews, cooldownDays: 1 },
    scandal: { ...ECONOMY_V2.scandal, minCustomers: 100 },
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
