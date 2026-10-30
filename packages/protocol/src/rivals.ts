import { z } from "zod";
import { zId } from "./text.ts";
import { GroupId } from "./world.ts";

// The rival pack (game-layer design §9): AIs built once on this computer by
// `airace-ml build-rivals`, measured, with their answers to the customer
// questions written ahead of time. The server loads `rivals.json` like the
// catalog; a world draws its four rivals from it.

export const RIVAL_STYLES = ["premium", "budget", "specialist", "consumer"] as const;
export type RivalStyle = (typeof RIVAL_STYLES)[number];

/** The arch-rival's surprise type, drawn per world. */
export const ARCHETYPES = ["giant", "upstart", "lab"] as const;
export type Archetype = (typeof ARCHETYPES)[number];

export const ARCHETYPE_WORDS: Record<Archetype, string> = {
  giant: "an arrogant tech giant",
  upstart: "a cheeky upstart",
  lab: "a mysterious lab",
};

export const STYLE_WORDS: Record<RivalStyle, string> = {
  premium: "Premium: big, pricey AIs",
  budget: "Budget: cheap and quick",
  specialist: "Specialist: great at one thing",
  consumer: "Crowd-pleaser: an AI for everyone",
};

/** The arch-rival's style follows its archetype. */
export const ARCHETYPE_STYLE: Record<Archetype, RivalStyle> = {
  giant: "premium",
  upstart: "consumer",
  lab: "specialist",
};

const Positive = z.number().int().positive();

export const RivalPackVersion = z.strictObject({
  key: zId,
  nLayer: Positive,
  dModel: Positive,
  ctxLen: Positive,
  trainedTokens: z.number().int().min(0),
  scores: z.record(z.string(), z.number().nullable()),
  overall: z.number().nullable(),
  traits: z.record(z.string(), z.number()),
  personalityWords: z.array(z.string()),
  /** "<category>/<tag>" -> [mean item score x 100, item count] (BenchReport.tag_breakdown). */
  breakdown: z.record(z.string(), z.tuple([z.number(), z.number().int().min(0)])),
});
export type RivalPackVersion = z.infer<typeof RivalPackVersion>;

export const RivalPackLineage = z.strictObject({
  id: zId,
  style: z.enum(RIVAL_STYLES),
  /** The customer group its business aims at. */
  focus: GroupId,
  versions: z.array(RivalPackVersion).min(1).max(5),
});
export type RivalPackLineage = z.infer<typeof RivalPackLineage>;

export const RivalPack = z
  .strictObject({
    version: z.literal("rivals-v1"),
    dataIdentity: z.string().min(1),
    tokenizer: z.literal("tok-v1"),
    suite: z.literal("bench-v1"),
    /** A quick build for checks: tiny models, few measurements. */
    quick: z.boolean(),
    lineages: z.array(RivalPackLineage).length(8),
  })
  .refine(
    (p) => RIVAL_STYLES.every((s) => p.lineages.filter((l) => l.style === s).length === 2),
    "A rival pack needs exactly two lineages of each style.",
  )
  .refine((p) => {
    const keys = p.lineages.flatMap((l) => l.versions.map((v) => v.key));
    const ids = p.lineages.map((l) => l.id);
    return new Set(keys).size === keys.length && new Set(ids).size === ids.length;
  }, "Lineage ids and version keys must be unique.");
export type RivalPack = z.infer<typeof RivalPack>;

/** One customer question a rival AI answered at build time; `reply` is real model output (R1). */
export const RivalReply = z.strictObject({
  group: GroupId,
  questionId: z.string().min(1),
  question: z.string(),
  reply: z.string(),
  score: z.number().min(0).max(1),
});
export type RivalReply = z.infer<typeof RivalReply>;

export const RivalReplies = z.array(RivalReply);
