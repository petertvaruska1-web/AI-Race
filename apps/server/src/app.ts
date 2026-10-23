import { mkdirSync } from "node:fs";
import cookie from "@fastify/cookie";
import Fastify, { type FastifyBaseLogger, type FastifyInstance } from "fastify";
import { registerAuthRoutes } from "./auth.ts";
import { type ServerOptions, SIM_VERSION } from "./config.ts";
import { type Db, openDb } from "./db/index.ts";
import { createLogger, type Logger } from "./log.ts";
import { WorldRegistry } from "./worlds/registry.ts";
import { registerWorldRoutes } from "./worlds/routes.ts";

/** Everything a route or service needs; one per running server. */
export type AppContext = {
  opts: ServerOptions;
  db: Db;
  log: Logger;
  worlds: WorldRegistry;
  /** Run on close, latest registered first (actors, gateways, brokers). */
  onClose: (fn: () => Promise<void>) => void;
};

export async function buildApp(
  opts: ServerOptions,
): Promise<{ app: FastifyInstance; ctx: AppContext; close(): Promise<void> }> {
  mkdirSync(opts.blobDir, { recursive: true });
  const log = createLogger(opts.logLevel ?? "info");
  const database = await openDb(opts.dbDir);
  const closers: (() => Promise<void>)[] = [];
  const ctx: AppContext = {
    opts,
    db: database.db,
    log,
    worlds: undefined as unknown as WorldRegistry, // set just below; it needs ctx
    onClose: (fn) => {
      closers.push(fn);
    },
  };
  ctx.worlds = new WorldRegistry(ctx);
  ctx.onClose(() => ctx.worlds.shutdown());

  const app = Fastify({
    loggerInstance: log as FastifyBaseLogger,
    bodyLimit: 64 * 1024,
  });
  await app.register(cookie, { secret: opts.secret });

  app.get("/api/health", async () => ({ ok: true, sim: SIM_VERSION }));
  registerAuthRoutes(app, ctx);
  registerWorldRoutes(app, ctx);

  let closed = false;
  return {
    app,
    ctx,
    async close() {
      if (closed) return;
      closed = true;
      // Stop taking requests first; then services (actors snapshot) close, newest first.
      await app.close();
      for (const fn of closers.reverse()) await fn();
      await database.close();
    },
  };
}
