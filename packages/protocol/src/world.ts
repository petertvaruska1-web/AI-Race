import { z } from "zod";
import { Difficulty, GameId, MinigameSettings } from "./minigames.ts";
import { TEXT_LIMITS, zId, zName, zText } from "./text.ts";

export const DATASET_IDS = [
  "web",
  "books",
  "educational",
  "conversations",
  "code",
  "reasoning",
  "facts",
  "creative",
] as const;
export type DatasetId = (typeof DATASET_IDS)[number];

/** What a training plan mixes: the eight datasets and the player's own notebook. */
export const MIX_IDS = [...DATASET_IDS, "notebook"] as const;
export type MixId = (typeof MIX_IDS)[number];

/** The eight bench-v1 skill categories. */
export const BENCH_CATEGORIES = [
  "language",
  "reasoning",
  "pattern",
  "knowledge",
  "coding",
  "creativity",
  "consistency",
  "instruction",
] as const;
export type BenchCategory = (typeof BENCH_CATEGORIES)[number];

/** League tables (game-layer design §7): customers, last month's revenue, overall and each skill. */
export const LEAGUE_TABLES = ["customers", "revenue", "overall", ...BENCH_CATEGORIES] as const;
export type LeagueTable = (typeof LEAGUE_TABLES)[number];

/** Skills two AIs can battle in: their replies are written text a check or the judge scores. */
export const BATTLE_SKILLS = [
  "knowledge",
  "coding",
  "instruction",
  "pattern",
  "creativity",
] as const;
export type BattleSkill = (typeof BATTLE_SKILLS)[number];

/** Customer groups (game-layer design §5). */
export const GROUP_IDS = [
  "students",
  "creators",
  "developers",
  "businesses",
  "everyday",
  "researchers",
] as const;
export const GroupId = z.enum(GROUP_IDS);
export type GroupId = z.infer<typeof GroupId>;

export const PLAN_IDS = ["free", "personal", "business"] as const;
export const PlanId = z.enum(PLAN_IDS);
export type PlanId = z.infer<typeof PlanId>;

/** Ledger categories (game-layer design §8). Marketing, data centers and interest come in Phase 2. */
export const INCOME_CATEGORIES = [
  "personal",
  "business",
  /** Partner apps' revenue share. */
  "appRevenue",
  /** What the investment fund gained (it stays in the fund until withdrawn). */
  "investmentGains",
] as const;
export type IncomeCategory = (typeof INCOME_CATEGORIES)[number];
export const COST_CATEGORIES = [
  "training",
  "serving",
  "dataChecks",
  "marketing",
  "dataCenterBuild",
  "dataCenterUpkeep",
  "interest",
  /** Launch keynotes and showcases. */
  "events",
  "sponsorships",
  "looks",
  /** Partner apps' setup and upkeep. */
  "appCosts",
  /** What the investment fund lost. */
  "investmentLosses",
  /** Office and support, which grow with the company (C35). */
  "overhead",
] as const;
export type CostCategory = (typeof COST_CATEGORIES)[number];

/** News stories in Phase 1 (game-layer design §7). Later phases add more. */
export const NEWS_TOPICS = [
  "launch",
  "customers",
  "firstPaying",
  "viral",
  "overload",
  "dataCenter",
  "rescue",
  "battle",
  "ranking",
  "trashTalk",
  "showcase",
  "keynote",
  "sponsorship",
  "app",
  "interview",
  "discovery",
  "goal",
  "scandal",
] as const;
export type NewsTopic = (typeof NEWS_TOPICS)[number];
/** Milestones (game-layer design §11): each earned once, only from a measurement or the state (R5). */
export const MILESTONE_IDS = [
  "firstSentence",
  "firstFact",
  "firstProgram",
  "customers1k",
  "customers100k",
  "customers1m",
  "firstPaying",
  "firstProfit",
  "numberOne",
  "beatArch",
  "dataCenter",
  "gradeA",
] as const;
export type MilestoneId = (typeof MILESTONE_IDS)[number];
export const MILESTONE_NAMES: Record<MilestoneId, string> = {
  firstSentence: "First full sentence",
  firstFact: "First correct fact",
  firstProgram: "First working program",
  customers1k: "First 1,000 customers",
  customers100k: "100,000 customers",
  customers1m: "1,000,000 customers",
  firstPaying: "First paying customer",
  firstProfit: "First profitable month",
  numberOne: "#1 in a skill",
  beatArch: "Beat the arch-rival in a battle",
  dataCenter: "A data center opened",
  gradeA: "An A on the report card",
};

/** A customer post's tone, picked from its reply's score. */
export const POST_TONES = ["praise", "joke", "roast"] as const;
export type PostTone = (typeof POST_TONES)[number];

/** Market regions. Phase 1 has one; Phase 2 adds the 9 US divisions. */
/** The nine US census divisions (game-layer design §5). */
export const REGION_IDS = [
  "newEngland",
  "midAtlantic",
  "greatLakes",
  "plains",
  "southAtlantic",
  "midSouth",
  "southCentral",
  "mountain",
  "pacific",
] as const;
export type RegionId = (typeof REGION_IDS)[number];

/** The regions in plain words (R7). */
export const REGION_NAMES: Record<RegionId, string> = {
  newEngland: "New England",
  midAtlantic: "Mid-Atlantic",
  greatLakes: "Great Lakes",
  plains: "Great Plains",
  southAtlantic: "South Atlantic",
  midSouth: "Mid-South",
  southCentral: "South Central",
  mountain: "Mountain West",
  pacific: "Pacific",
};

/** How a sentence names a region after "in": "in the Pacific", "in New England". */
export const REGION_PLACES: Record<RegionId, string> = {
  newEngland: "New England",
  midAtlantic: "the Mid-Atlantic",
  greatLakes: "the Great Lakes",
  plains: "the Great Plains",
  southAtlantic: "the South Atlantic",
  midSouth: "the Mid-South",
  southCentral: "South Central",
  mountain: "the Mountain West",
  pacific: "the Pacific",
};

export const Hex = z.string().regex(/^#[0-9a-f]{6}$/i);
export type Hex = z.infer<typeof Hex>;

export const Palette = z.object({ primary: Hex, secondary: Hex });
export type Palette = z.infer<typeof Palette>;

export const LOGO_SHAPES = ["circle", "rounded-square", "hexagon", "shield", "diamond"] as const;
export type LogoShape = (typeof LOGO_SHAPES)[number];

export const LOGO_SYMBOLS = [
  "spark",
  "atom",
  "leaf",
  "bolt",
  "star",
  "wave",
  "eye",
  "flame",
  "rocket",
  "brain",
  "cube",
  "moon",
  "sun",
  "anchor",
  "crown",
  "feather",
] as const;
export type LogoSymbol = (typeof LOGO_SYMBOLS)[number];

export const Logo = z.object({
  shape: z.enum(LOGO_SHAPES),
  symbol: z.enum(LOGO_SYMBOLS).nullable(),
  initials: z.string().regex(/^[A-Z0-9]{0,2}$/),
});
export type Logo = z.infer<typeof Logo>;

/** Integer weights; the trainer normalizes them. */
export const MixWeight = z.number().int().min(0).max(100);

/** The player-facing training plan. Each control maps to a real parameter (R7). */
const DatasetMix = Object.fromEntries(DATASET_IDS.map((id) => [id, MixWeight])) as Record<
  DatasetId,
  typeof MixWeight
>;

export const TrainingPlan = z.object({
  /** Every dataset's weight; the notebook's is 0 in plans from before it existed. */
  mix: z.strictObject({ ...DatasetMix, notebook: MixWeight.default(0) }),
  /** "expert" is Fact-check plus: the strictest cleaning, with duplicates and false facts removed. */
  cleaning: z.enum(["light", "standard", "thorough", "expert"]),
  removeDuplicates: z.boolean(),
  factCheck: z.boolean(),
  variety: z.enum(["natural", "balanced"]),
  brainSize: z.string().min(1).max(32),
  attentionSpan: z.number().int().positive(),
  length: z.string().min(1).max(32),
  learningStyle: z.number().min(0).max(1),
  review: z.number().min(0).max(0.5),
  probePrompt: zText(...TEXT_LIMITS.probePrompt),
  parentVersionId: zId.nullable(),
  /** Tutoring packs: extra practice from one dataset on top of the length (10% each). */
  tutoring: z
    .object({ dataset: z.enum(DATASET_IDS), packs: z.number().int().min(1).max(5) })
    .nullable()
    .default(null),
  /** Train on the conversations customers loved (playtest pack §5). */
  learnFromCustomers: z.boolean().default(false),
});
export type TrainingPlan = z.infer<typeof TrainingPlan>;

/** Sponsorships (playtest pack §5): what kind of event the company's name goes on. */
export const SPONSOR_KINDS = ["stadium", "conference", "festival"] as const;
export type SponsorKind = (typeof SPONSOR_KINDS)[number];

/** Partner apps built on the player's AI, each betting on one skill. */
export const APP_KINDS = ["homework", "stories", "code", "puzzles", "office"] as const;
export type AppKind = (typeof APP_KINDS)[number];
export const APP_NAMES: Record<AppKind, string> = {
  homework: "Homework Helper",
  stories: "Story Time",
  code: "Code Buddy",
  puzzles: "Puzzle Pal",
  office: "Office Assistant",
};

/** A look's id: `orb:<style>` or `logo:<motion>` (the sim's catalog says which exist). */
export const LookId = z.string().regex(/^(orb|logo):[a-z]{2,20}$/);

export const SPEEDS = [1, 2, 4] as const;
export const Speed = z.literal(SPEEDS);
export type Speed = z.infer<typeof Speed>;

export const Command = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("createCompany"),
    name: zName(...TEXT_LIMITS.companyName),
    logo: Logo,
    palette: Palette,
  }),
  z.object({ type: z.literal("startWorld") }),
  z.object({ type: z.literal("setSpeed"), speed: Speed }),
  z.object({ type: z.literal("setPaused"), paused: z.boolean() }),
  z.object({ type: z.literal("startTraining"), plan: TrainingPlan }),
  z.object({ type: z.literal("publishVersion"), versionId: zId, published: z.boolean() }),
  z.object({
    type: z.literal("launchProduct"),
    versionId: zId,
    name: zName(...TEXT_LIMITS.productName),
    plans: z.array(PlanId).max(3),
    countdownDays: z.number().int().min(0).max(365),
    /** A launch keynote: the AI answers this customer question live on stage. */
    keynote: z
      .object({ questionId: z.string().min(1).max(64) })
      .nullable()
      .default(null),
  }),
  z.object({ type: z.literal("setPlan"), plan: PlanId, productId: zId.nullable() }),
  z.object({
    type: z.literal("setPrice"),
    plan: z.enum(["personal", "business"]),
    price: z.number().int(),
  }),
  z.object({ type: z.literal("retireProduct"), productId: zId }),
  z.object({
    type: z.literal("startCampaign"),
    region: z.union([z.enum(REGION_IDS), z.literal("all")]),
    group: z.union([GroupId, z.literal("all")]),
    budgetPerDay: z.number().int(),
    days: z.number().int(),
  }),
  z.object({ type: z.literal("stopCampaign"), campaignId: zId }),
  z.object({ type: z.literal("buildDataCenter"), region: z.enum(REGION_IDS) }),
  z.object({ type: z.literal("setTrainingShare"), share: z.number() }),
  z.object({ type: z.literal("chooseRescue"), kind: z.enum(["loan", "investor"]) }),
  z.object({
    type: z.literal("startBattle"),
    rivalProductId: zId,
    skill: z.enum(BATTLE_SKILLS),
  }),
  z.object({
    type: z.literal("startShowcase"),
    game: GameId,
    difficulty: Difficulty,
    versionId: zId,
    settings: MinigameSettings,
  }),
  z.object({ type: z.literal("sponsor"), region: z.enum(REGION_IDS), kind: z.enum(SPONSOR_KINDS) }),
  z.object({ type: z.literal("buyLook"), lookId: LookId }),
  z.object({ type: z.literal("useLook"), lookId: LookId }),
  z.object({ type: z.literal("fundApp"), kind: z.enum(APP_KINDS), productId: zId }),
  z.object({ type: z.literal("closeApp"), appId: zId }),
  z.object({ type: z.literal("invest"), amount: z.number().int().positive() }),
  z.object({ type: z.literal("withdraw"), amount: z.number().int().positive() }),
]);
export type Command = z.infer<typeof Command>;
export type CommandType = Command["type"];
