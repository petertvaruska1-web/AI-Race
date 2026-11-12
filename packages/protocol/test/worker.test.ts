import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ServerToWorker, TRAINING_ENGINE, WORKER_SCHEMAS, WorkerMessage } from "../src/index.ts";

const DIR = new URL("../fixtures/worker/", import.meta.url);

describe("worker protocol fixtures", () => {
  const files = readdirSync(DIR).filter((f) => f.endsWith(".json"));

  it("covers every message type", () => {
    expect(files.length).toBeGreaterThanOrEqual(14);
  });

  it.each(files)("%s parses with its schema", (file) => {
    const { schema, message } = JSON.parse(readFileSync(new URL(file, DIR), "utf8"));
    const zodSchema = WORKER_SCHEMAS[schema as keyof typeof WORKER_SCHEMAS];
    expect(zodSchema, `unknown schema ${schema}`).toBeDefined();
    expect(zodSchema.safeParse(message).error).toBeUndefined();
  });

  it("checks job specs inside leases", () => {
    for (const file of files.filter((f) => f.startsWith("lease-"))) {
      const { message } = JSON.parse(readFileSync(new URL(file, DIR), "utf8"));
      const lease = ServerToWorker.parse(message);
      if (lease.type !== "lease") throw new Error("not a lease");
      const spec = {
        train: WORKER_SCHEMAS.TrainJobSpec,
        eval: WORKER_SCHEMAS.EvalJobSpec,
        generate: WORKER_SCHEMAS.GenerateJobSpec,
        feedback: WORKER_SCHEMAS.FeedbackJobSpec,
        battle: WORKER_SCHEMAS.BattleJobSpec,
        minigame: WORKER_SCHEMAS.MinigameJobSpec,
        keynote: WORKER_SCHEMAS.KeynoteJobSpec,
        appReview: WORKER_SCHEMAS.AppReviewJobSpec,
        interview: WORKER_SCHEMAS.InterviewJobSpec,
      }[lease.jobType];
      expect(spec.safeParse(lease.spec).error).toBeUndefined();
    }
  });

  it("reads a hello from a helper that doesn't name or time its card", () => {
    const { message } = JSON.parse(readFileSync(new URL("whello.json", DIR), "utf8"));
    const { gpuName: _, speed: __, engine: ___, ...older } = message.capabilities;
    const parsed = WorkerMessage.parse({ ...message, capabilities: older });
    if (parsed.type !== "whello") throw new Error("not a hello");
    expect(parsed.capabilities.gpuName).toBe("unknown");
    expect(parsed.capabilities.speed).toBe(0);
    expect(parsed.capabilities.engine).toBe(1); // from before engines were named
  });

  it("expects the training engine the shared hello names (the helper tests the same file)", () => {
    const { message } = JSON.parse(readFileSync(new URL("whello.json", DIR), "utf8"));
    expect(message.capabilities.engine).toBe(TRAINING_ENGINE);
  });

  it("reads an eval spec and result from before discoveries", () => {
    const spec = WORKER_SCHEMAS.EvalJobSpec.parse({
      versionId: "6fa459ea-ee8a-4ca4-894e-db77e160355e",
      modelTarUrl: "http://127.0.0.1:8787/blobs/a.tar",
      population: [],
    });
    expect(spec.previous).toBeNull();
    expect(spec.peers).toEqual([]);
    expect(spec.thresholds).toEqual({ minPoints: 25, minZ: 2, minPeers: 3, perEval: 3 });
    const result = WORKER_SCHEMAS.EvalJobResult.parse({
      report: {},
      traits: {},
      personalityWords: [],
    });
    expect(result).toMatchObject({ signals: null, breakdown: {}, discoveries: [] });
  });

  it("gives an older worker no battle slot", () => {
    const { message } = JSON.parse(readFileSync(new URL("whello.json", DIR), "utf8"));
    delete message.capabilities.slots.battle;
    const hello = WorkerMessage.parse(message);
    if (hello.type !== "whello") throw new Error("not a hello");
    expect(hello.capabilities.slots.battle).toBe(0);
  });

  it("lets a chat name a rival AI by its pack key", () => {
    const { message } = JSON.parse(readFileSync(new URL("lease-generate.json", DIR), "utf8"));
    const spec = { ...message.spec, versionId: "budget-a-v2" };
    expect(WORKER_SCHEMAS.GenerateJobSpec.safeParse(spec).success).toBe(true);
  });

  it("rejects a worker message without a job id or version", () => {
    expect(WorkerMessage.safeParse({ v: 1, type: "done", result: {} }).success).toBe(false);
    expect(WorkerMessage.safeParse({ v: 2, type: "heartbeat" }).success).toBe(false);
  });
});
