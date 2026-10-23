import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { BalanceVersion } from "@airace/sim";
import { buildApp } from "./app.ts";
import { realScheduler } from "./scheduler.ts";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

/** Reads a secret from the environment, else from `file`, creating it on first start. */
function secretFrom(envName: string, file: string): string {
  const fromEnv = process.env[envName];
  if (fromEnv) return fromEnv;
  if (existsSync(file)) return readFileSync(file, "utf8").trim();
  mkdirSync(dirname(file), { recursive: true });
  const value = randomBytes(32).toString("base64url");
  writeFileSync(file, `${value}\n`, { mode: 0o600 });
  return value;
}

const data = resolve(process.env.AIRACE_DATA ?? join(REPO_ROOT, "data"));
const serverDir = join(data, "server");
const port = Number(process.env.PORT ?? 8787);
const balance = (process.env.AIRACE_BALANCE ?? "balance-v1") as BalanceVersion;
if (balance !== "balance-v1" && balance !== "balance-test") {
  throw new Error(`AIRACE_BALANCE must be balance-v1 or balance-test, got ${balance}`);
}
const catalog = join(data, "catalog", "v1", "catalog.json");

const { app, close } = await buildApp({
  dbDir: join(serverDir, "db"),
  blobDir: join(serverDir, "blobs"),
  catalogPath: existsSync(catalog) ? catalog : null,
  secret: secretFrom("AIRACE_SECRET", join(serverDir, "secret")),
  // The local worker reads the same file, so `npm run dev` needs no setup.
  workerToken: secretFrom("AIRACE_WORKER_TOKEN", join(serverDir, "worker-token")),
  balance,
  scheduler: realScheduler,
  publicUrl: process.env.AIRACE_PUBLIC_URL ?? `http://localhost:${port}`,
  logLevel: process.env.LOG_LEVEL ?? "info",
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    app.log.info({ signal }, "shutting down");
    void close().then(() => process.exit(0));
  });
}

await app.listen({ port, host: process.env.HOST ?? "127.0.0.1" });
