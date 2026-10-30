import { z } from "zod";
import { zId } from "./text.ts";
import { BATTLE_SKILLS, GroupId } from "./world.ts";

// Messages between the server and ML workers (plan Task 13). The Python worker
// (ml/, Task 17) loads the fixtures in fixtures/worker/ to check it speaks the
// same shapes. Envelope keys are camelCase; Python-made payloads (training
// config and results) keep Python's snake_case.

const v = z.literal(1);

export const JOB_TYPES = ["train", "eval", "generate", "feedback", "battle"] as const;
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

export const EvalJobSpec = z.object({
  versionId: z.uuid(),
  modelTarUrl: Url,
  /** Trait vectors of other AIs; personality words are relative to them. */
  population: z.array(z.record(z.string(), z.number())),
});
export type EvalJobSpec = z.infer<typeof EvalJobSpec>;

export const EvalJobResult = z.object({
  report: z.record(z.string(), z.unknown()),
  traits: z.record(z.string(), z.number()),
  personalityWords: z.array(z.string()),
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
  rounds: z.array(z.object({ prompt: z.string(), a: BattleAnswer, b: BattleAnswer })).min(1),
});
export type BattleJobResult = z.infer<typeof BattleJobResult>;

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
} as const;
