import { existsSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Catalog } from "@airace/protocol";
import cookie from "@fastify/cookie";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyBaseLogger, type FastifyInstance } from "fastify";
import { AppReviewService } from "./apps/service.ts";
import { HttpError, registerAuthRoutes } from "./auth.ts";
import { BattleService } from "./battles/service.ts";
import { type BlobStore, LocalFsBlobStore, registerBlobRoutes } from "./blobs.ts";
import { loadCatalog, NO_TRAINING_DATA, publicCatalog } from "./catalog.ts";
import { ChatService } from "./chat/service.ts";
import { CoachingService } from "./coaching/service.ts";
import { type ServerOptions, SIM_VERSION } from "./config.ts";
import { type Db, openDb } from "./db/index.ts";
import { FeedbackService } from "./feedback/service.ts";
import { CardsService } from "./helpers/cards.ts";
import { registerHelperDownloads } from "./helpers/downloads.ts";
import { registerHelperKeyRoutes } from "./helpers/keys.ts";
import { LimitsService } from "./helpers/limits.ts";
import { InterviewService } from "./interviews/service.ts";
import { JobBroker } from "./jobs/broker.ts";
import { registerWorkerRoutes } from "./jobs/workerGateway.ts";
import { KeynoteService } from "./keynotes/service.ts";
import { createLogger, type Logger } from "./log.ts";
import { MinigameService } from "./minigames/service.ts";
import { NotebookService } from "./notebook/service.ts";
import { Gateway } from "./realtime/gateway.ts";
import { RivalService } from "./rivals/service.ts";
import { TrainingService } from "./training/service.ts";
import { WorldChatService } from "./worldchat/service.ts";
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
  training: TrainingService;
  chat: ChatService;
  feedback: FeedbackService;
  rivals: RivalService;
  battles: BattleService;
  minigames: MinigameService;
  keynotes: KeynoteService;
  appReviews: AppReviewService;
  interviews: InterviewService;
  notebooks: NotebookService;
  cards: CardsService;
  worldChat: WorldChatService;
  coaching: CoachingService;
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
    training: undefined as unknown as TrainingService,
    chat: undefined as unknown as ChatService,
    feedback: undefined as unknown as FeedbackService,
    rivals: undefined as unknown as RivalService,
    battles: undefined as unknown as BattleService,
    minigames: undefined as unknown as MinigameService,
    keynotes: undefined as unknown as KeynoteService,
    appReviews: undefined as unknown as AppReviewService,
    interviews: undefined as unknown as InterviewService,
    notebooks: undefined as unknown as NotebookService,
    cards: undefined as unknown as CardsService,
    worldChat: undefined as unknown as WorldChatService,
    coaching: undefined as unknown as CoachingService,
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
  ctx.training = new TrainingService(ctx);
  ctx.onClose(() => ctx.training.close());
  ctx.rivals = new RivalService(ctx);
  ctx.chat = new ChatService(ctx);
  ctx.gateway.setPlaygroundHandler((worldId) => ctx.broker.setPlayground(worldId));
  ctx.feedback = new FeedbackService(ctx);
  ctx.onClose(() => ctx.feedback.close());
  ctx.battles = new BattleService(ctx);
  ctx.onClose(() => ctx.battles.close());
  ctx.minigames = new MinigameService(ctx);
  ctx.onClose(() => ctx.minigames.close());
  ctx.keynotes = new KeynoteService(ctx);
  ctx.onClose(() => ctx.keynotes.close());
  ctx.appReviews = new AppReviewService(ctx);
  ctx.onClose(() => ctx.appReviews.close());
  ctx.interviews = new InterviewService(ctx);
  ctx.notebooks = new NotebookService(ctx);
  ctx.cards = new CardsService(ctx);
  new LimitsService(ctx);
  ctx.worldChat = new WorldChatService(ctx);
  ctx.coaching = new CoachingService(ctx);
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
  registerHelperKeyRoutes(app, ctx);

  app.get("/api/health", async () => ({ ok: true, sim: SIM_VERSION }));
  registerAuthRoutes(app, ctx);
  registerWorldRoutes(app, ctx);
  registerBlobRoutes(app, ctx);
  registerHelperDownloads(app, ctx);
  await serveClient(app, opts.clientDist ?? null);

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

/**
 * The built web app, when the host plays from their PC (`npm run host`): its files at `/`, and
 * `index.html` for the links the app handles itself (`/join/<code>`, `/w/<world>`).
 */
async function serveClient(app: FastifyInstance, dist: string | null): Promise<void> {
  if (!dist || !existsSync(join(dist, "index.html"))) return;
  await app.register(fastifyStatic, { root: resolve(dist), prefix: "/", index: "index.html" });
  for (const path of ["/join/*", "/w/*"]) {
    app.get(path, (_req, reply) => reply.sendFile("index.html"));
  }
}
