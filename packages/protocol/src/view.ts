import { z } from "zod";
import { Difficulty, GameId } from "./minigames.ts";
import { ARCHETYPES, RIVAL_STYLES } from "./rivals.ts";
import { Day, zId } from "./text.ts";
import { DiscoveryKind } from "./worker.ts";
import {
  APP_KINDS,
  BATTLE_SKILLS,
  COST_CATEGORIES,
  GOAL_KINDS,
  GroupId,
  INCOME_CATEGORIES,
  LEAGUE_TABLES,
  Logo,
  MILESTONE_IDS,
  NEWS_TOPICS,
  Palette,
  PlanId,
  POST_TONES,
  REGION_IDS,
  SPONSOR_KINDS,
  Speed,
  TrainingPlan,
} from "./world.ts";

// Views are strict: an unexpected key (a leaked private field) fails parsing.

export const ModelShapeView = z.strictObject({
  nLayer: z.number().int().positive(),
  dModel: z.number().int().positive(),
  ctxLen: z.number().int().positive(),
});

export const TrainingPreview = z.strictObject({
  params: z.number().int().positive(),
  tokens: z.number().int().positive(),
  flops: z.number().positive(),
  cost: z.number().int().min(0),
  /** Of the cost, what checking and preparing data takes (absent before sim-5). */
  checks: z.number().int().min(0).optional(),
  durationDays: z.number().int().positive(),
});
export type TrainingPreview = z.infer<typeof TrainingPreview>;

export const Scores = z.record(z.string(), z.number().nullable());

export const RUN_STATUSES = ["training", "finalizing", "completed", "failed"] as const;
export type RunStatus = (typeof RUN_STATUSES)[number];

export const RunView = z.strictObject({
  id: zId,
  status: z.enum(RUN_STATUSES),
  computeStarted: z.boolean(),
  plan: TrainingPlan,
  preview: TrainingPreview,
  startDay: Day,
  endDay: Day,
  progress: z.number().min(0).max(1),
  failure: z.string().nullable(),
  versionId: zId.nullable(),
});
export type RunView = z.infer<typeof RunView>;

export const VersionView = z.strictObject({
  id: zId,
  label: z.string(),
  runId: zId,
  parentId: zId.nullable(),
  shape: ModelShapeView,
  createdDay: Day,
  trainedTokens: z.number().int().positive(),
  scores: Scores.nullable(),
  overall: z.number().nullable(),
  personalityWords: z.array(z.string()),
  /** Measured style traits (the owner's only). */
  traits: z.record(z.string(), z.number()).nullable(),
  published: z.boolean(),
});
export type VersionView = z.infer<typeof VersionView>;

/** A shared AI as others see it (spec §6.4): no brain shape, data or training detail (R2). */
export const PublicVersion = z.strictObject({
  id: zId,
  label: z.string(),
  createdDay: Day,
  scores: Scores.nullable(),
  overall: z.number().nullable(),
  personalityWords: z.array(z.string()),
});
export type PublicVersion = z.infer<typeof PublicVersion>;

const Count = z.number().int().min(0);
const Reputation = z.number().int().min(0).max(100);
const Price = z.number().int().positive();

export const PRODUCT_STATUSES = ["countdown", "live", "retired"] as const;
export type ProductStatus = (typeof PRODUCT_STATUSES)[number];

/** A product as its owner sees it. */
export const ProductView = z.strictObject({
  id: zId,
  name: z.string(),
  versionId: zId,
  status: z.enum(PRODUCT_STATUSES),
  announcedDay: Day,
  launchDay: Day,
  hype: z.number().min(0),
});
export type ProductView = z.infer<typeof ProductView>;

/** A product as everyone sees it, from its announcement until it is retired. */
export const PublicProduct = z.strictObject({
  id: zId,
  name: z.string(),
  status: z.enum(["countdown", "live"]),
  launchDay: Day,
  /** Null during the countdown: an AI stays private until it is released (R2). */
  version: PublicVersion.nullable(),
});
export type PublicProduct = z.infer<typeof PublicProduct>;

export const PlansView = z.strictObject({
  free: z.strictObject({ productId: zId.nullable() }),
  personal: z.strictObject({ productId: zId.nullable(), price: Price }),
  business: z.strictObject({ productId: zId.nullable(), price: Price }),
});
export type PlansView = z.infer<typeof PlansView>;

/** One game month of the owner's money, in whole dollars. */
export const MonthLedgerView = z.strictObject({
  month: z.string().regex(/^\d{4}-\d{2}$/),
  income: z.record(z.enum(INCOME_CATEGORIES), z.number().int()),
  /** A refund can make a month's cost negative. */
  costs: z.record(z.enum(COST_CATEGORIES), z.number().int()),
  revenueByGroup: z.record(GroupId, z.number().int()),
  revenueByPlan: z.record(PlanId, z.number().int()),
  revenueByRegion: z.record(z.enum(REGION_IDS), z.number().int()),
  customersEnd: Count,
  customersEndByGroup: z.record(GroupId, Count),
  /** Money that is neither income nor a cost: rescue loans and investors in, repayments out. */
  financing: z.strictObject({ loans: Count, investors: Count, repaid: Count }),
});
export type MonthLedgerView = z.infer<typeof MonthLedgerView>;

export const SpotlightView = z.strictObject({
  id: zId,
  name: z.string(),
  handle: z.string(),
  group: GroupId,
  region: z.enum(REGION_IDS),
  bio: z.string(),
});
export type SpotlightView = z.infer<typeof SpotlightView>;

const feedBase = {
  id: zId,
  day: Day,
  companyId: zId.nullable(),
  sentiment: z.number().min(-1).max(1),
};

/** A customer post: a game-written frame around a real reply. */
export const PostView = z.strictObject({
  ...feedBase,
  kind: z.literal("post"),
  author: z.strictObject({
    name: z.string(),
    handle: z.string(),
    group: GroupId,
    spotlightId: zId.nullable(),
  }),
  productName: z.string(),
  question: z.string(),
  /** Real model output, exactly as written (R1). Clients render it as text. */
  reply: z.string(),
  tone: z.enum(POST_TONES),
  /** A partner app's review: its stars. */
  stars: z.number().int().min(1).max(5).optional(),
  /** Written by a news outlet about an interview, not by a customer. */
  press: z.literal(true).optional(),
});
export type PostView = z.infer<typeof PostView>;

/** A news story: game commentary, never an AI's words. */
export const NewsView = z.strictObject({
  ...feedBase,
  kind: z.literal("news"),
  topic: z.enum(NEWS_TOPICS),
  headline: z.string(),
});
export type NewsView = z.infer<typeof NewsView>;

export const FeedItemView = z.discriminatedUnion("kind", [PostView, NewsView]);
export type FeedItemView = z.infer<typeof FeedItemView>;

export const LeagueRowView = z.strictObject({
  companyId: zId,
  rank: z.number().int().positive(),
  value: z.number().nullable(),
});
export type LeagueRowView = z.infer<typeof LeagueRowView>;

/** Ranking points from battles (an Elo rating per skill). */
export const PointsRowView = z.strictObject({
  companyId: zId,
  rank: z.number().int().positive(),
  points: z.number().int(),
});
export type PointsRowView = z.infer<typeof PointsRowView>;

/** All another player may see of a company (R2, spec §6.4 and game-layer §13.2). */
export const PublicCompany = z.strictObject({
  id: zId,
  name: z.string(),
  logo: Logo,
  palette: Palette,
  /** How the logo moves (`logo:<motion>`; a look the company bought). */
  logoMotion: z.string(),
  /** Its running sponsorships, as markers on the map. */
  sponsorships: z.array(
    z.strictObject({ region: z.enum(REGION_IDS), kind: z.enum(SPONSOR_KINDS) }),
  ),
  /** Null for a rival company: no player owns it. */
  ownerUserId: zId.nullable(),
  /** What kind of rival it is; null for a player's company. The archetype is the arch-rival's only. */
  rival: z
    .strictObject({ style: z.enum(RIVAL_STYLES), archetype: z.enum(ARCHETYPES).nullable() })
    .nullable(),
  publishedVersions: z.array(PublicVersion),
  products: z.array(PublicProduct),
  /** Prices of the open plans; null (or false) for a closed plan. */
  prices: z.strictObject({
    free: z.boolean(),
    personal: Price.nullable(),
    business: Price.nullable(),
  }),
  /** Rounded to three significant figures. */
  customers: Count,
  /** Rounded to three significant figures. */
  customersByRegion: z.record(z.enum(REGION_IDS), Count),
  /** The founders' share of the company, after any investor rescue (1 = all of it). */
  ownership: z.number().gt(0).max(1),
  reputation: Reputation,
  followers: Count,
});
export type PublicCompany = z.infer<typeof PublicCompany>;

/** One day of the owner's company, for its charts (immersion design). */
export const DaySampleView = z.strictObject({
  cash: z.number().int(),
  customers: z.record(GroupId, Count),
  regions: z.record(z.enum(REGION_IDS), Count),
  /** Customers on the personal and business plans. */
  paying: Count,
  followers: Count,
  reputation: Reputation,
});
export type DaySampleView = z.infer<typeof DaySampleView>;

/** A running ad campaign (its owner's only). */
export const CampaignView = z.strictObject({
  id: zId,
  region: z.union([z.enum(REGION_IDS), z.literal("all")]),
  group: z.union([GroupId, z.literal("all")]),
  budgetPerDay: z.number().int().positive(),
  startDay: Day,
  endDay: Day,
  spent: z.number().int().min(0),
});
export type CampaignView = z.infer<typeof CampaignView>;

/** A data center, building or open (its owner's only). */
export const DataCenterView = z.strictObject({
  id: zId,
  region: z.enum(REGION_IDS),
  startDay: Day,
  openDay: Day,
  open: z.boolean(),
});
export type DataCenterView = z.infer<typeof DataCenterView>;

/** A post kept on a report card; its reply is real model output, verbatim (R1). */
export const MomentPostView = z.strictObject({
  author: z.strictObject({ name: z.string(), handle: z.string(), group: GroupId }),
  productName: z.string(),
  question: z.string(),
  reply: z.string(),
  tone: z.enum(POST_TONES),
});

/** A month's report card (its owner's only). */
export const ReportCardView = z.strictObject({
  month: z.string().regex(/^\d{4}-\d{2}$/),
  revenue: z.number().int(),
  costs: z.number().int(),
  profit: z.number().int(),
  lastProfit: z.number().int().nullable(),
  customersStart: Count,
  customersEnd: Count,
  growthByGroup: z.record(GroupId, z.number().int()),
  mood: z.number().min(0).max(1).nullable(),
  best: MomentPostView.nullable(),
  worst: MomentPostView.nullable(),
  headline: z.string().nullable(),
  grade: z.enum(["A", "B", "C", "D", "F"]),
  /** Places against rivals from last month's card to this one; null in a world without rivals. */
  ranks: z
    .array(
      z.strictObject({
        table: z.enum(["customers", "revenue", "overall"]),
        from: z.number().int().positive().nullable(),
        to: z.number().int().positive(),
      }),
    )
    .nullable(),
});
export type ReportCardView = z.infer<typeof ReportCardView>;

const BattleSideView = z.strictObject({ productId: zId, versionId: zId, productName: z.string() });
const BattleAnswerView = z.strictObject({
  /** Real model output, exactly as written (R1). */
  reply: z.string(),
  score: z.number().min(0).max(1),
});

/** A battle against a rival (its owner's only); the replies are real, verbatim. */
export const BattleView = z.strictObject({
  id: zId,
  day: Day,
  skill: z.enum(BATTLE_SKILLS),
  rivalId: zId,
  mine: BattleSideView,
  theirs: BattleSideView,
  status: z.enum(["running", "done", "calledOff"]),
  rounds: z
    .array(
      z.strictObject({
        prompt: z.string(),
        /** The right answer (none for a story). */
        expected: z.string().nullable(),
        mine: BattleAnswerView,
        theirs: BattleAnswerView,
        /** A story round between friends' AIs: the other players' vote totals (they decide it). */
        votes: z
          .strictObject({ mine: z.number().int().min(0), theirs: z.number().int().min(0) })
          .nullable(),
      }),
    )
    .nullable(),
  outcome: z.enum(["win", "draw", "loss"]).nullable(),
  /** Ranking points won or lost by it. */
  pointsChange: z.number().int().nullable(),
  /** Whether `rivalId` is a rival company or another player's. */
  opponentKind: z.enum(["rival", "player"]),
  /** A friend challenged this company: its side is `mine`. */
  defending: z.boolean(),
});
export type BattleView = z.infer<typeof BattleView>;

/** A showcase: a Playground game played live for the press. */
export const ShowcaseView = z.strictObject({
  id: zId,
  day: Day,
  game: GameId,
  difficulty: Difficulty,
  productId: zId,
  versionId: zId,
  productName: z.string(),
  status: z.enum(["running", "done", "calledOff"]),
  score: z.number().nullable(),
  maxScore: z.number().nullable(),
});
export type ShowcaseView = z.infer<typeof ShowcaseView>;

/** What the playtest pack's purchases cost, from the world's balance. */
export const ShopView = z.strictObject({
  showcase: z.strictObject({ fee: z.number().int(), cooldownDays: z.number().int() }),
  sponsorships: z.array(
    z.strictObject({
      kind: z.enum(SPONSOR_KINDS),
      price: z.number().int(),
      days: z.number().int(),
      groups: z.array(GroupId),
    }),
  ),
  looks: z.array(z.strictObject({ id: z.string(), price: z.number().int() })),
  apps: z.array(
    z.strictObject({
      kind: z.enum(APP_KINDS),
      skill: z.enum(BATTLE_SKILLS),
      group: GroupId,
      setup: z.number().int(),
      upkeep: z.number().int(),
    }),
  ),
});
export type ShopView = z.infer<typeof ShopView>;

/** One department as its owner sees it (C43). */
const DepartmentView = z.strictObject({
  budget: z.number().int().min(0),
  staffed: z.number().int().min(0),
  strength: z.number().min(0).max(1),
});

/** A public license offer (C44): terms any company in the world can take. */
export const LicenseOfferView = z.strictObject({
  id: zId,
  licensorId: zId,
  versionId: zId,
  /** The licensed AI's product name when it was offered. */
  aiName: z.string(),
  priceCents: z.number().int(),
  months: z.number().int(),
  groups: z.array(GroupId),
  exclusive: z.boolean(),
  day: Day,
});
export type LicenseOfferView = z.infer<typeof LicenseOfferView>;

const LicenseSide = z.enum(["licensor", "licensee"]);

/** One of the company's running licenses, seen from its side. */
export const LicenseDealView = z.strictObject({
  id: zId,
  role: LicenseSide,
  otherId: zId,
  otherName: z.string(),
  versionId: zId,
  aiName: z.string(),
  /** The licensee's product the licensed AI powers. */
  productId: zId,
  productName: z.string(),
  priceCents: z.number().int(),
  groups: z.array(GroupId),
  startDay: Day,
  endDay: Day,
  exclusive: z.boolean(),
  /** Royalties built up so far this month, in whole dollars; paid at the month's close. */
  owed: z.number().int().min(0),
});
export type LicenseDealView = z.infer<typeof LicenseDealView>;

/** A proposed license waiting for the maker's answer: `licensor` is asked, `licensee` asked. */
export const LicenseProposalView = z.strictObject({
  id: zId,
  role: LicenseSide,
  otherId: zId,
  otherName: z.string(),
  versionId: zId,
  aiName: z.string(),
  productId: zId,
  productName: z.string(),
  priceCents: z.number().int(),
  months: z.number().int(),
  groups: z.array(GroupId),
  day: Day,
  expiresDay: Day,
});
export type LicenseProposalView = z.infer<typeof LicenseProposalView>;

export const PrivateCompany = z.strictObject({
  id: zId,
  cash: z.number().int(),
  runs: z.array(RunView),
  versions: z.array(VersionView),
  products: z.array(ProductView),
  plans: PlansView,
  customers: z.strictObject({
    total: Count,
    byGroup: z.record(GroupId, z.record(PlanId, Count)),
    byRegion: z.record(z.enum(REGION_IDS), z.record(GroupId, Count)),
  }),
  /** Per product, each group's satisfaction (0..1). */
  satisfaction: z.record(zId, z.record(GroupId, z.number().min(0).max(1))),
  ledger: z.strictObject({ current: MonthLedgerView, months: z.array(MonthLedgerView).max(12) }),
  fame: z.strictObject({ reputation: Reputation, followers: Count }),
  /**
   * The last 120 days keyed by day ("37"): a new day patches as one entry, where
   * a growing array would be resent whole (immersion design I2).
   */
  history: z.record(z.string().regex(/^\d+$/), DaySampleView),
  campaigns: z.array(CampaignView),
  dataCenters: z.array(DataCenterView),
  /** The last 20 battles, newest last. */
  battles: z.array(BattleView),
  looks: z.strictObject({ owned: z.array(z.string()), orb: z.string(), logo: z.string() }),
  /** Every sponsorship, running or over. */
  sponsorships: z.array(
    z.strictObject({
      id: zId,
      region: z.enum(REGION_IDS),
      kind: z.enum(SPONSOR_KINDS),
      startDay: Day,
      endDay: Day,
    }),
  ),
  /** Partner apps, running or closed. */
  apps: z.array(
    z.strictObject({
      id: zId,
      kind: z.enum(APP_KINDS),
      productId: zId,
      productName: z.string(),
      startDay: Day,
      users: z.number().int(),
      rating: z.number().min(1).max(5),
      status: z.enum(["live", "closed"]),
      reviews: z.array(
        z.strictObject({
          day: Day,
          prompt: z.string(),
          /** Real model output, exactly as written (R1). */
          reply: z.string(),
          stars: z.number().int().min(1).max(5),
        }),
      ),
    }),
  ),
  /** Feedback replies customers loved, which "Learn from customers" can train on. */
  loved: z.number().int().min(0),
  /** Lessons the player gave their AI in the Playground (coaching), up to 200. */
  lessons: z.number().int().min(0),
  /** The notebook's size and when it was saved; its text stays on the server. */
  notebook: z.strictObject({ words: z.number().int().min(0), savedDay: Day }).nullable(),
  /** What the scoring found in this company's AIs, with their real evidence (owner only, R2). */
  discoveries: z.array(
    z.strictObject({
      id: z.string(),
      versionId: zId,
      versionLabel: z.string(),
      day: Day,
      kind: DiscoveryKind,
      about: z.string(),
      title: z.string(),
      category: z.string(),
      tag: z.string(),
      value: z.number(),
      compared: z.number(),
      evidence: z.strictObject({ prompt: z.string(), reply: z.string() }).nullable(),
      published: z.boolean(),
    }),
  ),
  /** Published interviews, newest last; the AI's answers verbatim (R1). */
  interviews: z.array(
    z.strictObject({
      id: zId,
      day: Day,
      productId: zId,
      productName: z.string(),
      outlet: z.string(),
      mean: z.number().min(0).max(1),
      answers: z.array(
        z.strictObject({
          question: z.string(),
          reply: z.string(),
          score: z.number().min(0).max(1),
          checked: z.boolean(),
        }),
      ),
    }),
  ),
  /** The first day another interview can be published (null before the first). */
  nextInterviewDay: Day.nullable(),
  /** Progress towards the world's goal (null without one). */
  goalProgress: z
    .strictObject({ value: z.number(), target: z.number().nullable(), done: z.boolean() })
    .nullable(),
  /** The trophy shelf: milestones earned, oldest first (never a list of what is left). */
  milestones: z.array(
    z.strictObject({ id: z.enum(MILESTONE_IDS), day: Day, detail: z.string().nullable() }),
  ),
  /** The investment fund (it counts toward the company's value). */
  fund: z.strictObject({ balance: z.number().int() }),
  /** The four departments: monthly budget, the team grown so far, and its strength (0–1). */
  departments: z.strictObject({
    research: DepartmentView,
    data: DepartmentView,
    sales: DepartmentView,
    support: DepartmentView,
  }),
  /** Licenses this company gives or takes, and proposals waiting for an answer (C44). */
  licenses: z.strictObject({
    deals: z.array(LicenseDealView),
    proposals: z.array(LicenseProposalView),
  }),
  /** The last 20 showcases, newest last. */
  showcases: z.array(ShowcaseView),
  shop: ShopView,
  /** The last 12 report cards, newest last. */
  reports: z.array(ReportCardView).max(12),
  /** A rescue offer waiting for a choice, until its deadline. */
  rescue: z.strictObject({ deadlineDay: Day }).nullable(),
  loan: z
    .strictObject({
      balance: z.number().int().min(0),
      principalPerMonth: z.number().int().min(0),
      monthlyRate: z.number().min(0),
    })
    .nullable(),
  /** Serving capacity in PFLOP a day, and the share of today's questions answered. */
  compute: z.strictObject({
    cloudPflops: z.number().min(0),
    ownedPflops: z.number().min(0),
    trainingShare: z.number().min(0).max(1),
    loadPflops: z.number().min(0),
    served: z.number().min(0).max(1),
  }),
});
export type PrivateCompany = z.infer<typeof PrivateCompany>;

export const WorldView = z.strictObject({
  world: z.strictObject({
    id: zId,
    name: z.string(),
    phase: z.enum(["setup", "running"]),
    /** Which economy the world runs on; clients price plans with it. */
    balance: z.enum(["balance-v2", "balance-test"]),
    day: Day,
    date: z.string(),
    speed: Speed,
    paused: z.boolean(),
    hostUserId: zId,
    effectiveHostUserId: zId.nullable(),
    inviteCode: z.string(),
    /** The share of people willing to use an AI today, and how many people that is. */
    market: z.strictObject({
      adoption: z.number().min(0).max(1),
      adoptable: Count,
      regions: z.record(z.enum(REGION_IDS), z.strictObject({ adoptable: Count })),
    }),
    /** Whether rivals have joined this world (they need the rival pack). */
    rivals: z.enum(["joined", "none"]),
    /** Everyone's graphics cards are pooled and shared equally (chosen at creation). */
    fairMode: z.boolean(),
    /** The world's goal, if it has one; `reachedBy` is the company that won it. */
    goal: z
      .strictObject({
        kind: z.enum(GOAL_KINDS).exclude(["none"]),
        target: z.number().nullable(),
        deadlineDay: Day.nullable(),
        reachedDay: Day.nullable(),
        missedDay: Day.nullable(),
        reachedBy: zId.nullable(),
      })
      .nullable(),
    /** Every company's place in each table; other companies' revenue shows as a place only (R2). */
    league: z.strictObject({
      tables: z.record(z.enum(LEAGUE_TABLES), z.array(LeagueRowView)),
      points: z.record(z.enum(BATTLE_SKILLS), z.array(PointsRowView)),
    }),
  }),
  members: z.array(
    z.strictObject({
      userId: zId,
      displayName: z.string(),
      online: z.boolean(),
      companyId: zId.nullable(),
    }),
  ),
  companies: z.array(PublicCompany),
  /** Every open license offer in the world, oldest first (C44). */
  licenseOffers: z.array(LicenseOfferView),
  /** The newest 50 posts and news, oldest first; public whatever company they are about. */
  feed: z.array(FeedItemView).max(50),
  spotlights: z.array(SpotlightView),
  me: z.strictObject({
    userId: zId,
    company: PrivateCompany.nullable(),
  }),
});
export type WorldView = z.infer<typeof WorldView>;
