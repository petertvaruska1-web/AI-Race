import { z } from "zod";
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
export const INCOME_CATEGORIES = ["personal", "business"] as const;
export type IncomeCategory = (typeof INCOME_CATEGORIES)[number];
export const COST_CATEGORIES = [
  "training",
  "serving",
  "dataChecks",
  "marketing",
  "dataCenterBuild",
  "dataCenterUpkeep",
  "interest",
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
] as const;
export type NewsTopic = (typeof NEWS_TOPICS)[number];
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
export const TrainingPlan = z.object({
  mix: z.record(z.enum(DATASET_IDS), MixWeight),
  cleaning: z.enum(["light", "standard", "thorough"]),
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
});
export type TrainingPlan = z.infer<typeof TrainingPlan>;

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
]);
export type Command = z.infer<typeof Command>;
export type CommandType = Command["type"];
