import { mkdirSync } from "node:fs";
import type { Catalog } from "@airace/protocol";
import cookie from "@fastify/cookie";
import Fastify, { type FastifyBaseLogger, type FastifyInstance } from "fastify";
import { HttpError, registerAuthRoutes } from "./auth.ts";
import { type BlobStore, LocalFsBlobStore, registerBlobRoutes } from "./blobs.ts";
import { loadCatalog, NO_TRAINING_DATA, publicCatalog } from "./catalog.ts";
import { type ServerOptions, SIM_VERSION } from "./config.ts";
import { type Db, openDb } from "./db/index.ts";
import { JobBroker } from "./jobs/broker.ts";
import { registerWorkerRoutes } from "./jobs/workerGateway.ts";
import { createLogger, type Logger } from "./log.ts";
import { Gateway } from "./realtime/gateway.ts";
import { WorldRegistry } from "./worlds/registry.ts";
import { registerWorldRoutes } from "./worlds/routes.ts";

/** Everything a route or service needs; one per running server. */
export type AppContext = {
  opts: ServerOptions;
  db: Db;
  log: Logger;
  worlds: WorldRegistry;
  gateway: Gateway;
  broker: JobBroker;
  blobs: BlobStore;
  /** Null until the training data is installed. */
  catalog: Catalog | null;
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
    // Set just below: both need ctx.
    worlds: undefined as unknown as WorldRegistry,
    gateway: undefined as unknown as Gateway,
    broker: undefined as unknown as JobBroker,
    blobs: new LocalFsBlobStore(opts.blobDir),
    catalog: loadCatalog(opts.catalogPath, log),
    onClose: (fn) => {
      closers.push(fn);
    },
  };
  ctx.worlds = new WorldRegistry(ctx);
  ctx.onClose(() => ctx.worlds.shutdown());
  ctx.gateway = new Gateway(ctx);
  ctx.onClose(() => ctx.gateway.close());
  ctx.broker = new JobBroker(ctx);
  ctx.onClose(() => ctx.broker.close());
  ctx.gateway.registerQuery("catalog", async () => {
    if (!ctx.catalog) throw new HttpError(503, NO_TRAINING_DATA);
    return publicCatalog(ctx.catalog);
  });

  const app = Fastify({
    loggerInstance: log as FastifyBaseLogger,
    bodyLimit: 64 * 1024,
  });
  await app.register(cookie, { secret: opts.secret });
  await ctx.gateway.register(app);
  registerWorkerRoutes(app, ctx);

  app.get("/api/health", async () => ({ ok: true, sim: SIM_VERSION }));
  registerAuthRoutes(app, ctx);
  registerWorldRoutes(app, ctx);
  registerBlobRoutes(app, ctx);

  await ctx.broker.start();

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
