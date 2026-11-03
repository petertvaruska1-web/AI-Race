import { z } from "zod";
import {
  Difficulty,
  GameId,
  MinigameSettings,
  MinigameSideResult,
  MinigameStep,
} from "./minigames.ts";
import { zId } from "./text.ts";
import { BATTLE_SKILLS, GroupId } from "./world.ts";

// Messages between the server and ML workers (plan Task 13). The Python worker
// (ml/, Task 17) loads the fixtures in fixtures/worker/ to check it speaks the
// same shapes. Envelope keys are camelCase; Python-made payloads (training
// config and results) keep Python's snake_case.

const v = z.literal(1);

export const JOB_TYPES = [
  "train",
  "eval",
  "generate",
  "feedback",
  "battle",
  "minigame",
  "keynote",
  "appReview",
  "interview",
] as const;
export const JobType = z.enum(JOB_TYPES);
export type JobType = z.infer<typeof JobType>;

const Slots = z.number().int().min(0);
export const JobId = z.uuid();

export const WorkerCapabilities = z.object({
  device: z.enum(["cuda", "cpu"]),
  vramMb: z.number().int().min(0),
  maxParams: z.number().int().positive(),
  jobTypes: z.array(JobType).min(1),
  /** A worker from before feedback or battle jobs sends no slot for them: it gets none. */
  slots: z.object({
    train: Slots,
    eval: Slots,
    generate: Slots,
    feedback: Slots.default(0),
    battle: Slots.default(0),
    minigame: Slots.default(0),
    keynote: Slots.default(0),
    appReview: Slots.default(0),
    interview: Slots.default(0),
  }),
  /** Which corpus, tokenizer and evaluators the worker has installed (ADR 0008 item 2). */
  dataIdentity: z.string().min(1),
});
export type WorkerCapabilities = z.infer<typeof WorkerCapabilities>;

export const WorkerMessage = z.discriminatedUnion("type", [
  z.object({
    v,
    type: z.literal("whello"),
    token: z.string().min(1).max(512),
    workerId: z.string().min(1).max(64),
    capabilities: WorkerCapabilities,
    /** Jobs this worker is still running, e.g. after a dropped connection. */
    running: z.array(JobId),
  }),
  z.object({ v, type: z.literal("heartbeat") }),
  z.object({ v, type: z.literal("progress"), jobId: JobId, event: z.unknown() }),
  z.object({ v, type: z.literal("chunk"), jobId: JobId, text: z.string() }),
  z.object({ v, type: z.literal("done"), jobId: JobId, result: z.unknown() }),
  z.object({
    v,
    type: z.literal("failed"),
    jobId: JobId,
    error: z.string(),
    retryable: z.boolean(),
  }),
]);
export type WorkerMessage = z.infer<typeof WorkerMessage>;

export const ServerToWorker = z.discriminatedUnion("type", [
  z.object({ v, type: z.literal("wwelcome") }),
  z.object({
    v,
    type: z.literal("lease"),
    jobId: JobId,
    // `type` names the message, so the job's type has its own key.
    jobType: JobType,
    spec: z.unknown(),
    leaseMs: z.number().int().positive(),
  }),
  z.object({ v, type: z.literal("cancel"), jobId: JobId }),
]);
export type ServerToWorker = z.infer<typeof ServerToWorker>;

const Url = z.url();

export const TrainJobSpec = z.object({
  runId: z.uuid(),
  /** Python `TrainRunConfig` JSON (snake_case), made by the server's training service. */
  config: z.record(z.string(), z.unknown()),
  parentTarUrl: Url.nullable(),
  outTarUrl: Url,
});
export type TrainJobSpec = z.infer<typeof TrainJobSpec>;

export const TrainJobResult = z.object({
  steps: z.number().int().min(0),
  final_loss: z.number().nullable(),
  heldout_losses: z.record(z.string(), z.number().nullable()),
  status: z.enum(["completed", "unstable_stopped"]),
});
export type TrainJobResult = z.infer<typeof TrainJobResult>;

/** `"<category>/<tag>"` -> [score 0-100, item count], from a benchmark report. */
export const Breakdown = z.record(z.string(), z.tuple([z.number(), z.number().int().min(0)]));
export type Breakdown = z.infer<typeof Breakdown>;

export const DISCOVERY_KINDS = ["strength", "weakness", "better", "worse", "changed"] as const;
export const DiscoveryKind = z.enum(DISCOVERY_KINDS);
export type DiscoveryKind = z.infer<typeof DiscoveryKind>;

/** Something the scoring found in an AI, with one real question and reply as proof (R1). */
export const Discovery = z.object({
  key: z.string().min(1).max(120),
  kind: DiscoveryKind,
  about: z.string().min(1).max(120),
  title: z.string().min(1).max(200),
  category: z.string().min(1).max(40),
  tag: z.string().min(1).max(80),
  value: z.number(),
  compared: z.number(),
  evidence: z.object({ prompt: z.string(), reply: z.string() }).nullable(),
});
export type Discovery = z.infer<typeof Discovery>;

export const DiscoveryThresholds = z.object({
  minPoints: z.number().min(0),
  minZ: z.number().min(0),
  minPeers: z.number().int().min(1),
  perEval: z.number().int().min(0),
});
export type DiscoveryThresholds = z.infer<typeof DiscoveryThresholds>;

const Traits = z.record(z.string(), z.number());

export const EvalJobSpec = z.object({
  versionId: z.uuid(),
  modelTarUrl: Url,
  /** Trait vectors of other AIs; personality words are relative to them. */
  population: z.array(Traits),
  /** The version this one was trained from, for "better" and "worse" findings. */
  previous: z
    .object({ label: z.string(), breakdown: Breakdown, traits: Traits })
    .nullable()
    .default(null),
  /** The world's other AIs, for strengths and weaknesses. */
  peers: z.array(z.object({ breakdown: Breakdown, traits: Traits })).default([]),
  thresholds: DiscoveryThresholds.default({ minPoints: 25, minZ: 2, minPeers: 3, perEval: 3 }),
});
export type EvalJobSpec = z.infer<typeof EvalJobSpec>;

/** What the milestones need from one eval; legibility is null without the judge. */
export const EvalSignals = z.object({
  legibleReplies: z.number().int().min(0).nullable(),
  correctFacts: z.number().int().min(0),
  passingPrograms: z.number().int().min(0),
});
export type EvalSignals = z.infer<typeof EvalSignals>;

export const EvalJobResult = z.object({
  report: z.record(z.string(), z.unknown()),
  traits: Traits,
  personalityWords: z.array(z.string()),
  signals: EvalSignals.nullable().default(null),
  breakdown: Breakdown.default({}),
  discoveries: z.array(Discovery).default([]),
});
export type EvalJobResult = z.infer<typeof EvalJobResult>;

export const GenerateJobSpec = z.object({
  /** A player version's id, or a rival AI's pack key. */
  versionId: zId,
  modelTarUrl: Url,
  history: z.array(z.tuple([z.enum(["user", "ai"]), z.string()])),
  maxNewTokens: z.number().int().positive(),
  temperature: z.number().positive(),
  topP: z.number().gt(0).max(1),
  seed: z.number().int().min(0),
});
export type GenerateJobSpec = z.infer<typeof GenerateJobSpec>;

export const GenerateJobResult = z.object({ text: z.string() });
export type GenerateJobResult = z.infer<typeof GenerateJobResult>;

/** A product's AI answers customer questions from each group's pool (game-layer design §7). */
export const FeedbackJobSpec = z.object({
  productId: z.uuid(),
  versionId: z.uuid(),
  modelTarUrl: Url,
  groups: z.array(GroupId).min(1),
  perGroup: z.number().int().positive(),
  seed: z.number().int().min(0),
});
export type FeedbackJobSpec = z.infer<typeof FeedbackJobSpec>;

export const FeedbackJobResult = z.object({
  results: z.array(
    z.object({
      group: GroupId,
      questionId: z.string().min(1),
      question: z.string(),
      /** Real model output, exactly as generated (R1). */
      reply: z.string(),
      /** What the question's automatic check gave the reply, 0..1. */
      score: z.number().min(0).max(1),
    }),
  ),
});
export type FeedbackJobResult = z.infer<typeof FeedbackJobResult>;

const BattleSide = z.object({ versionId: zId, modelTarUrl: Url });

/** Two AIs answer the same fresh challenges (game-layer design §10); `b` is the rival. */
export const BattleJobSpec = z.object({
  battleId: zId,
  skill: z.enum(BATTLE_SKILLS),
  rounds: z.number().int().min(1).max(5),
  seed: z.number().int().min(0),
  a: BattleSide,
  b: BattleSide,
});
export type BattleJobSpec = z.infer<typeof BattleJobSpec>;

/** A piece of a battle reply, sent as a `progress` event while it is written. */
export const BattleTextEvent = z.object({
  kind: z.literal("battleText"),
  round: z.number().int().min(0),
  side: z.enum(["a", "b"]),
  text: z.string(),
});
export type BattleTextEvent = z.infer<typeof BattleTextEvent>;

const BattleAnswer = z.object({
  /** Real model output, exactly as generated (R1). */
  reply: z.string(),
  score: z.number().min(0).max(1),
});

export const BattleJobResult = z.object({
  rounds: z
    .array(
      z.object({
        prompt: z.string(),
        /** The right answer, or null for a story. */
        expected: z.string().nullable().default(null),
        a: BattleAnswer,
        b: BattleAnswer,
      }),
    )
    .min(1),
});
export type BattleJobResult = z.infer<typeof BattleJobResult>;

const Side = z.enum(["a", "b"]);

/** One or two AIs play a Playground game (playtest pack §4); `b` is a rival's, in a duel. */
export const MinigameJobSpec = z.object({
  runId: zId,
  game: GameId,
  difficulty: Difficulty,
  seed: z.number().int().min(0),
  settings: MinigameSettings,
  sides: z
    .array(z.object({ side: Side, versionId: zId, modelTarUrl: Url }))
    .min(1)
    .max(2),
});
export type MinigameJobSpec = z.infer<typeof MinigameJobSpec>;

/** A step as it happens, sent as a `progress` event. */
export const MinigameStepEvent = MinigameStep.extend({
  kind: z.literal("minigameStep"),
  side: Side,
});
export type MinigameStepEvent = z.infer<typeof MinigameStepEvent>;

/** A piece of a typed answer or story, sent as a `progress` event while it is written. */
export const MinigameTextEvent = z.object({
  kind: z.literal("minigameText"),
  side: Side,
  attempt: z.number().int().min(0),
  text: z.string(),
});
export type MinigameTextEvent = z.infer<typeof MinigameTextEvent>;

export const MinigameJobResult = z.object({
  sides: z.object({ a: MinigameSideResult, b: MinigameSideResult.optional() }),
});
export type MinigameJobResult = z.infer<typeof MinigameJobResult>;

/** A launch keynote's live demo: the product's AI answers one customer question. */
export const KeynoteJobSpec = z.object({
  versionId: zId,
  modelTarUrl: Url,
  questionId: z.string().min(1),
  seed: z.number().int().min(0),
});
export type KeynoteJobSpec = z.infer<typeof KeynoteJobSpec>;

export const KeynoteJobResult = z.object({
  questionId: z.string().min(1),
  prompt: z.string(),
  /** Real model output, exactly as generated (R1). */
  reply: z.string(),
  score: z.number().min(0).max(1),
});
export type KeynoteJobResult = z.infer<typeof KeynoteJobResult>;

/** A partner app's users try the AI on fresh questions in the app's skill. */
export const AppReviewJobSpec = z.object({
  versionId: zId,
  modelTarUrl: Url,
  skill: z.enum(BATTLE_SKILLS),
  n: z.number().int().min(1).max(5),
  seed: z.number().int().min(0),
});
export type AppReviewJobSpec = z.infer<typeof AppReviewJobSpec>;

export const AppReviewJobResult = z.object({
  reviews: z
    .array(
      z.object({
        prompt: z.string(),
        /** Real model output, exactly as generated (R1). */
        reply: z.string(),
        score: z.number().min(0).max(1),
      }),
    )
    .min(1),
});
export type AppReviewJobResult = z.infer<typeof AppReviewJobResult>;

/** Scores the answers a player published from an interview; no model is loaded. */
export const InterviewJobSpec = z.object({
  interviewId: zId,
  answers: z
    .array(
      z.object({
        /** The prompt card asked, or null for the player's own question. */
        cardId: z.string().min(1).max(80).nullable(),
        question: z.string(),
        /** The AI's answer, exactly as it was said (R1). */
        reply: z.string(),
      }),
    )
    .min(1)
    .max(5),
});
export type InterviewJobSpec = z.infer<typeof InterviewJobSpec>;

export const InterviewJobResult = z.object({
  /** One per answer, in the same order. */
  answers: z
    .array(z.object({ score: z.number().min(0).max(1), checked: z.boolean() }))
    .min(1)
    .max(5),
});
export type InterviewJobResult = z.infer<typeof InterviewJobResult>;

/** Schemas by name, for the shared fixtures (`{schema, message}`). */
export const WORKER_SCHEMAS = {
  WorkerMessage,
  ServerToWorker,
  TrainJobSpec,
  TrainJobResult,
  EvalJobSpec,
  EvalJobResult,
  GenerateJobSpec,
  GenerateJobResult,
  FeedbackJobSpec,
  FeedbackJobResult,
  BattleJobSpec,
  BattleJobResult,
  MinigameJobSpec,
  MinigameJobResult,
  KeynoteJobSpec,
  KeynoteJobResult,
  AppReviewJobSpec,
  AppReviewJobResult,
  InterviewJobSpec,
  InterviewJobResult,
} as const;
