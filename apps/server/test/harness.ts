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
  // Signed URLs must point at the port the server actually got.
  if (over.publicUrl === undefined) ctx.opts.publicUrl = url;
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

/** Calls the API with an optional session cookie and JSON body. */
export async function api(
  s: TestServer,
  path: string,
  init: { method?: string; body?: unknown; cookie?: string } = {},
): Promise<Response> {
  const headers: Record<string, string> = {};
  if (init.body !== undefined) headers["content-type"] = "application/json";
  if (init.cookie) headers.cookie = `airace_session=${init.cookie}`;
  return fetch(`${s.url}${path}`, {
    method: init.method ?? (init.body === undefined ? "GET" : "POST"),
    headers,
    ...(init.body === undefined ? {} : { body: JSON.stringify(init.body) }),
  });
}

/** Signs up a guest; returns its id and session cookie value. */
export async function createGuest(
  s: TestServer,
  displayName = "Ana",
): Promise<{ userId: string; cookie: string }> {
  const res = await api(s, "/api/guest", { body: { displayName } });
  if (res.status !== 200) throw new Error(`guest failed: ${res.status}`);
  const { userId } = (await res.json()) as { userId: string };
  const cookie = /airace_session=([^;]+)/.exec(res.headers.get("set-cookie") ?? "")?.[1];
  if (!cookie) throw new Error("no session cookie");
  return { userId, cookie };
}

/** A guest with a new world; returns ids and the cookie. */
export async function createGuestWorld(
  s: TestServer,
  displayName = "Ana",
  name = "Friday night",
): Promise<{ userId: string; cookie: string; worldId: string }> {
  const guest = await createGuest(s, displayName);
  const res = await api(s, "/api/worlds", { body: { name }, cookie: guest.cookie });
  if (res.status !== 200) throw new Error(`world failed: ${res.status}`);
  const { worldId } = (await res.json()) as { worldId: string };
  return { ...guest, worldId };
}
