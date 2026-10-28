import type {
  Command,
  CostCategory,
  GroupId,
  IncomeCategory,
  Logo,
  NewsTopic,
  Palette,
  PlanId,
  PostTone,
  RegionId,
  RunStatus,
  Speed,
  TrainingPlan,
  TrainingPreview,
} from "@airace/protocol";
import type { BalanceVersion } from "./balance.ts";
import type { RngState } from "./rng.ts";

export type Member = {
  userId: string;
  displayName: string;
  /** The `seq` of the input that added the member; orders the host handover. */
  joinedSeq: number;
  online: boolean;
  companyId: string | null;
};

/** What a worker reports when a training job ends. */
export type TrainingResult = {
  steps: number;
  finalLoss: number | null;
  heldoutLosses: Record<string, number | null>;
  unstable: boolean;
};

export type RunState = {
  id: string;
  plan: TrainingPlan;
  preview: TrainingPreview;
  /** Drawn from the world rng, so a replayed run trains the same model. */
  seed: number;
  status: RunStatus;
  computeStarted: boolean;
  startDay: number;
  endDay: number;
  /** A result that arrived before its game time; applied by the tick that reaches `endDay`. */
  held: { versionId: string; result: TrainingResult } | null;
  failure: string | null;
  versionId: string | null;
};

export type VersionState = {
  id: string;
  label: string;
  runId: string;
  parentId: string | null;
  nLayer: number;
  dModel: number;
  ctxLen: number;
  createdDay: number;
  trainedTokens: number;
  scores: Record<string, number | null> | null;
  overall: number | null;
  personalityWords: string[];
  /** Measured style traits, from the same evaluation as the scores. */
  traits: Record<string, number> | null;
  published: boolean;
};

export type ProductState = {
  id: string;
  name: string;
  versionId: string;
  status: "countdown" | "live" | "retired";
  announcedDay: number;
  launchDay: number;
  hype: number;
  /** Plans it powers from launch day. */
  launchPlans: PlanId[];
  /** Feedback batches already applied (the last 20), so a result is never counted twice. */
  feedbackKeys: string[];
};

export type PlansState = {
  free: { productId: string | null };
  personal: { productId: string | null; price: number };
  business: { productId: string | null; price: number };
};

export type Spotlight = {
  id: string;
  name: string;
  handle: string;
  group: GroupId;
  region: RegionId;
  bio: string;
};

/** Customer counts are fractional (they move smoothly); views round them. */
export type MarketState = {
  customers: Record<
    string,
    Partial<Record<string, Partial<Record<GroupId, Partial<Record<PlanId, number>>>>>>
  >;
  awareness: Record<string, Partial<Record<string, Partial<Record<GroupId, number>>>>>;
  satisfaction: Record<string, Partial<Record<GroupId, number>>>;
};

export type FameState = {
  /** 0..100 */
  reputation: number;
  /** Fractional; views round. */
  followers: number;
  /** A moving average (-1..1) of the sentiment of the company's news. */
  newsSentiment: number;
};

/** One scored customer question, as the worker reports it. `reply` is real model output. */
export type FeedbackResult = {
  group: GroupId;
  questionId: string;
  question: string;
  reply: string;
  /** 0..1 */
  score: number;
};

type FeedBase = {
  id: string;
  day: number;
  companyId: string | null;
  /** -1..1 */
  sentiment: number;
};

/** A customer post: a game-written frame around a reply quoted verbatim (R1). */
export type PostItem = FeedBase & {
  kind: "post";
  author: { name: string; handle: string; group: GroupId; spotlightId: string | null };
  productName: string;
  question: string;
  reply: string;
  tone: PostTone;
};

/** A news story: game commentary over real events, never an AI's words. */
export type NewsItem = FeedBase & { kind: "news"; topic: NewsTopic; headline: string };

export type FeedItem = PostItem | NewsItem;

/** One game month of bookings, in whole dollars. */
export type MonthLedger = {
  /** "2027-03" */
  month: string;
  income: Record<IncomeCategory, number>;
  costs: Record<CostCategory, number>;
  revenueByGroup: Record<GroupId, number>;
  revenueByPlan: Record<PlanId, number>;
  revenueByRegion: Record<RegionId, number>;
  /** Customers when the month closed (0 while it is open). */
  customersEnd: number;
  customersEndByGroup: Record<GroupId, number>;
  /** Money that is neither income nor a cost (rescues, Phase 2). */
  financing: { loans: number; investors: number; repaid: number };
};

/** One day of a company, for its charts. Whole numbers. */
export type DaySample = {
  day: number;
  cash: number;
  customers: Record<GroupId, number>;
  regions: Record<RegionId, number>;
  /** Customers on the personal and business plans. */
  paying: number;
  followers: number;
  reputation: number;
};

export type LedgerState = {
  current: MonthLedger;
  /** The last 24 closed months, newest last. */
  months: MonthLedger[];
};

export type CompanyState = {
  id: string;
  ownerUserId: string;
  name: string;
  logo: Logo;
  palette: Palette;
  cash: number;
  runs: Record<string, RunState>;
  versions: Record<string, VersionState>;
  versionSeq: number;
  products: Record<string, ProductState>;
  plans: PlansState;
  fame: FameState;
  ledger: LedgerState;
  /** Milestones already in the news, so each is told once. */
  announced: { customers: number; firstPaying: boolean };
  /** The last `HISTORY_DAYS` days, oldest first. */
  history: DaySample[];
};

export type WorldState = {
  worldId: string;
  name: string;
  balance: BalanceVersion;
  rng: RngState;
  phase: "setup" | "running";
  day: number;
  speed: Speed;
  paused: boolean;
  hostUserId: string;
  members: Record<string, Member>;
  companies: Record<string, CompanyState>;
  inviteCode: string;
  /** Accepted inputs so far. */
  seq: number;
  market: MarketState;
  spotlights: Spotlight[];
  /** Posts and news, oldest first, capped at the balance's `feedCap`. */
  feed: FeedItem[];
  /** Feed items made so far; numbers their ids. */
  feedSeq: number;
};

export type SystemEvent =
  | { type: "memberJoined"; userId: string; displayName: string }
  | { type: "presence"; userId: string; online: boolean }
  | { type: "trainingCompleted"; runId: string; versionId: string; result: TrainingResult }
  | { type: "trainingFailed"; runId: string; reason: string }
  | { type: "computeStarted"; runId: string }
  | {
      type: "evalCompleted";
      versionId: string;
      scores: Record<string, number | null>;
      overall: number | null;
      personalityWords: string[];
      traits?: Record<string, number> | null;
    }
  | {
      type: "feedbackScored";
      productId: string;
      /** "w12" for a weekly batch, "launch" for the launch-day burst. */
      key: string;
      results: FeedbackResult[];
    };

/** Everything that can change a world. Logged, so replay reproduces the state. */
export type Input =
  | { kind: "command"; userId: string; cmd: Command; newId?: string }
  | { kind: "tick" }
  | { kind: "system"; event: SystemEvent };

export type StepResult = { ok: true; state: WorldState } | { ok: false; reason: string };
