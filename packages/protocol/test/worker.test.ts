import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ServerToWorker, WORKER_SCHEMAS, WorkerMessage } from "../src/index.ts";

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
      }[lease.jobType];
      expect(spec.safeParse(lease.spec).error).toBeUndefined();
    }
  });

  it("rejects a worker message without a job id or version", () => {
    expect(WorkerMessage.safeParse({ v: 1, type: "done", result: {} }).success).toBe(false);
    expect(WorkerMessage.safeParse({ v: 2, type: "heartbeat" }).success).toBe(false);
  });
});
