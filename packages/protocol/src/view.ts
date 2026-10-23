import { z } from "zod";
import { Day, zId } from "./text.ts";
import { Logo, Palette, Speed, TrainingPlan } from "./world.ts";

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

/** All another player may see of a company (R2, spec §6.4). */
export const PublicCompany = z.strictObject({
  id: zId,
  name: z.string(),
  logo: Logo,
  palette: Palette,
  ownerUserId: zId,
  publishedVersions: z.array(PublicVersion),
});
export type PublicCompany = z.infer<typeof PublicCompany>;

export const PrivateCompany = z.strictObject({
  id: zId,
  cash: z.number().int(),
  runs: z.array(RunView),
  versions: z.array(VersionView),
});
export type PrivateCompany = z.infer<typeof PrivateCompany>;

export const WorldView = z.strictObject({
  world: z.strictObject({
    id: zId,
    name: z.string(),
    phase: z.enum(["setup", "running"]),
    day: Day,
    date: z.string(),
    speed: Speed,
    paused: z.boolean(),
    hostUserId: zId,
    effectiveHostUserId: zId.nullable(),
    inviteCode: z.string(),
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
  me: z.strictObject({
    userId: zId,
    company: PrivateCompany.nullable(),
  }),
});
export type WorldView = z.infer<typeof WorldView>;
