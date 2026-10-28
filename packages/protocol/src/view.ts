import { z } from "zod";
import { Day, zId } from "./text.ts";
import {
  COST_CATEGORIES,
  GroupId,
  INCOME_CATEGORIES,
  Logo,
  NEWS_TOPICS,
  Palette,
  PlanId,
  POST_TONES,
  REGION_IDS,
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

/** All another player may see of a company (R2, spec §6.4 and game-layer §13.2). */
export const PublicCompany = z.strictObject({
  id: zId,
  name: z.string(),
  logo: Logo,
  palette: Palette,
  ownerUserId: zId,
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
});
export type ReportCardView = z.infer<typeof ReportCardView>;

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
  /** The newest 50 posts and news, oldest first; public whatever company they are about. */
  feed: z.array(FeedItemView).max(50),
  spotlights: z.array(SpotlightView),
  me: z.strictObject({
    userId: zId,
    company: PrivateCompany.nullable(),
  }),
});
export type WorldView = z.infer<typeof WorldView>;
