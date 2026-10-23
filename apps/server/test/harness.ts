import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FastifyInstance } from "fastify";
import { type AppContext, buildApp } from "../src/app.ts";
import type { ServerOptions } from "../src/config.ts";
import { type FakeScheduler, fakeScheduler } from "../src/scheduler.ts";

export type TestServer = {
  url: string;
  wsUrl: string;
  app: FastifyInstance;
  ctx: AppContext;
  scheduler: FakeScheduler;
  close(): Promise<void>;
};

/** A real server on a free port: in-memory database, temp blob dir, fake clock. */
export async function startTestServer(over: Partial<ServerOptions> = {}): Promise<TestServer> {
  const tmp = mkdtempSync(join(tmpdir(), "airace-server-"));
  const scheduler = (over.scheduler as FakeScheduler | undefined) ?? fakeScheduler();
  const { app, ctx, close } = await buildApp({
    dbDir: null,
    blobDir: join(tmp, "blobs"),
    catalogPath: null,
    secret: "test-secret",
    workerToken: "test-worker-token",
    balance: "balance-test",
    publicUrl: "http://127.0.0.1",
    logLevel: "warn",
    ...over,
    scheduler,
  });
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (!address || typeof address === "string") throw new Error("server has no port");
  const url = `http://127.0.0.1:${address.port}`;
  return {
    url,
    wsUrl: url.replace("http", "ws"),
    app,
    ctx,
    scheduler,
    async close() {
      await close();
      rmSync(tmp, { recursive: true, force: true });
    },
  };
}
