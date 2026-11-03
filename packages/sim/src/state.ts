import type {
  AppKind,
  Archetype,
  BattleSkill,
  Command,
  CostCategory,
  Difficulty,
  EvalSignals,
  GameId,
  GroupId,
  IncomeCategory,
  LeagueTable,
  Logo,
  MilestoneId,
  MinigameSettings,
  NewsTopic,
  Palette,
  PlanId,
  PostTone,
  RegionId,
  RivalPack,
  RivalPackVersion,
  RivalStyle,
  RunStatus,
  Speed,
  SponsorKind,
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
  /** The notebook the run trains on (its text's sha256), when the plan has a notebook share. */
  notebookHash: string | null;
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
  /** A rival's version: its key in the rival pack (its `runId` is "pack"). Null for players. */
  packKey: string | null;
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
  /** The launch keynote, if one was booked (absent before sim-5). */
  keynote?: KeynoteState | null;
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
  /** A partner app's review: its stars (1-5). */
  stars?: number;
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

/** Serving capacity (PFLOP a day) and how much of today's demand it met. */
export type ComputeState = {
  cloudPflops: number;
  /** The share of owned capacity lent to training (0..1). */
  trainingShare: number;
  loadPflops: number;
  /** The share of today's questions answered (0..1). */
  served: number;
  lastOverloadNews: number | null;
};

/** A post kept on a report card: who said it, and the AI's reply quoted verbatim (R1). */
export type MomentPost = {
  author: { name: string; handle: string; group: GroupId };
  productName: string;
  question: string;
  reply: string;
  tone: PostTone;
};

/** A month's report card. Game commentary, except the quoted posts. */
export type ReportCard = {
  month: string;
  revenue: number;
  costs: number;
  profit: number;
  lastProfit: number | null;
  customersStart: number;
  customersEnd: number;
  growthByGroup: Record<GroupId, number>;
  /** Customer-weighted satisfaction (0..1) at the close; null without customers. */
  mood: number | null;
  best: MomentPost | null;
  worst: MomentPost | null;
  headline: string | null;
  grade: "A" | "B" | "C" | "D" | "F";
  /** Places against rivals, last card to this one; null without rivals. */
  ranks: { table: "customers" | "revenue" | "overall"; from: number | null; to: number }[] | null;
};

/** A data center: it serves (and trains) from `openDay` with its capacity. */
export type DataCenterState = {
  id: string;
  region: RegionId;
  startDay: number;
  openDay: number;
  pflopsPerDay: number;
};

/** A running ad campaign; it spends on the ticks after `startDay`, through `endDay`. */
export type CampaignState = {
  id: string;
  region: RegionId | "all";
  group: GroupId | "all";
  budgetPerDay: number;
  startDay: number;
  endDay: number;
  spent: number;
};

/** A rival version waiting for its release day (the pack's measurements, R5). */
export type PackRelease = RivalPackVersion & { releaseDay: number };

/** What makes a company a rival: its pack lineage and business style (game-layer design §9). */
export type RivalState = {
  lineageId: string;
  style: RivalStyle;
  focus: GroupId;
  /** Set only for the arch-rival. */
  archetype: Archetype | null;
  /** Which of its style's product words names its AIs. */
  productWord: number;
  upcoming: PackRelease[];
  /** Versions released so far. */
  released: number;
  lastReviewDay: number;
  lastTrashTalkDay: number | null;
  /** Numbers the ids of its campaigns and data centers. */
  counters: { campaigns: number; dataCenters: number };
};

/** One round of a battle: the challenge and both AIs' real replies with their scores (R1). */
/** A showcase: a minigame played live for the press (playtest pack §4.3). */
export type ShowcaseState = {
  id: string;
  day: number;
  game: GameId;
  difficulty: Difficulty;
  settings: MinigameSettings;
  productId: string;
  versionId: string;
  productName: string;
  seed: number;
  status: "running" | "done" | "calledOff";
  score: number | null;
  maxScore: number | null;
};

/** Looks the company owns and wears (playtest pack §5). */
export type LooksState = { owned: string[]; orb: string; logo: string };

export type SponsorshipState = {
  id: string;
  region: RegionId;
  kind: SponsorKind;
  startDay: number;
  endDay: number;
};

export type AppReviewState = {
  day: number;
  prompt: string;
  reply: string;
  score: number;
  stars: number;
};

/** A partner app built on one of the company's products. */
export type AppState = {
  id: string;
  kind: AppKind;
  productId: string;
  startDay: number;
  users: number;
  /** 1-5 stars, from its reviews (3 until the first ones). */
  rating: number;
  status: "live" | "closed";
  /** The last reviews, newest last. */
  reviews: AppReviewState[];
  /** Review batches already applied, so a result is never counted twice. */
  reviewKeys: string[];
};

/** The launch keynote's live demo, before and after it happens. */
export type KeynoteState = {
  questionId: string;
  status: "waiting" | "done" | "failed";
  result: { prompt: string; reply: string; score: number } | null;
  /** Whether its effect on the launch has been applied. */
  applied: boolean;
};

export type BattleRound = {
  prompt: string;
  /** The right answer, shown under the replies (none for a story; absent before sim-5). */
  expected?: string | null;
  mine: { reply: string; score: number };
  theirs: { reply: string; score: number };
};

/** A battle between the player's AI and a rival's (game-layer design §10). */
export type BattleState = {
  id: string;
  day: number;
  skill: BattleSkill;
  rivalId: string;
  mine: { productId: string; versionId: string; productName: string };
  theirs: { productId: string; versionId: string; productName: string; packKey: string };
  /** Picks the challenges, so a retried battle asks the same ones. */
  seed: number;
  status: "running" | "done" | "calledOff";
  rounds: BattleRound[] | null;
  outcome: "win" | "draw" | "loss" | null;
  /** The player's ranking points won (or lost) by this battle. */
  pointsChange: number | null;
};

export type CompanyState = {
  id: string;
  /** Null for a rival: no player owns it. */
  ownerUserId: string | null;
  /** Cash at founding; the money identity starts from it. */
  startCash: number;
  rival: RivalState | null;
  /** The last battles, newest last. */
  battles: BattleState[];
  /** League places at the last report card (for its rank changes). */
  lastRanks: { customers: number; revenue: number; overall: number } | null;
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
  campaigns: Record<string, CampaignState>;
  compute: ComputeState;
  dataCenters: Record<string, DataCenterState>;
  /** The last 24 report cards, newest last. */
  reports: ReportCard[];
  /** A rescue offer waiting for a choice. */
  rescue: { offeredDay: number; deadlineDay: number } | null;
  loan: { balance: number; principalPerMonth: number } | null;
  /** The founders' share of the company (1 = all of it); investor rescues take part. */
  ownership: number;
  /** The last showcases, newest last. */
  showcases: ShowcaseState[];
  looks: LooksState;
  sponsorships: Record<string, SponsorshipState>;
  apps: Record<string, AppState>;
  /** The investment fund: money moved out of cash, earning or losing at month ends. */
  fund: { balance: number };
  /** Feedback replies customers loved (score 0.75 or more), up to 400; the server keeps their text. */
  loved: number;
  /** The player's notebook: the server keeps its text; the world only its fingerprint (R2). */
  notebook: { hash: string; words: number; savedDay: number } | null;
  /** Milestones earned, oldest first; each once (players only). */
  milestones: { id: MilestoneId; day: number; detail: string | null }[];
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
  /**
   * Inputs up to this one were logged by an older simulation and replay on its rules
   * (an upgraded world); 0 for a world made by this one.
   */
  legacyUntilSeq: number;
  /** Null until rivals join (they need the rival pack). */
  rivals: { joinedDay: number; archId: string; archPassed: boolean } | null;
  /** Each league table's leader, and the day its last #1 story ran. */
  leaders: Partial<Record<LeagueTable, string>>;
  lastRankingNews: Partial<Record<LeagueTable, number>>;
  /** Ranking points per battle skill (Elo), by company. */
  points: Partial<Record<BattleSkill, Record<string, number>>>;
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
      /** What the milestones need (absent from evals logged before sim-6). */
      signals?: EvalSignals | null;
    }
  | { type: "rivalsJoined"; pack: RivalPack }
  | { type: "battleResolved"; battleId: string; rounds: BattleRound[] }
  | { type: "battleResolved"; battleId: string; failed: true }
  | { type: "showcaseResolved"; showcaseId: string; score: number; maxScore: number }
  | { type: "showcaseResolved"; showcaseId: string; failed: true }
  | { type: "keynoteResolved"; productId: string; prompt: string; reply: string; score: number }
  | { type: "keynoteResolved"; productId: string; failed: true }
  /** The server stored a notebook's text; only its sha256 and word count enter the world. */
  | { type: "notebookSaved"; companyId: string; hash: string; words: number }
  | {
      type: "appReviewed";
      appId: string;
      /** "w12": the week's batch, so a result is applied once. */
      key: string;
      reviews: { prompt: string; reply: string; score: number }[];
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
