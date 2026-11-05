import {
  bigint,
  doublePrecision,
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  serial,
  text,
  unique,
  uuid,
} from "drizzle-orm/pg-core";

// Times are epoch milliseconds from the server's scheduler, so tests with a
// fake clock see consistent values.
const ms = (name: string) => bigint(name, { mode: "number" });

export const users = pgTable("users", {
  id: uuid("id").primaryKey(),
  displayName: text("display_name").notNull(),
  createdAt: ms("created_at").notNull(),
});

export const authIdentities = pgTable("auth_identities", {
  id: uuid("id").primaryKey(),
  userId: uuid("user_id")
    .notNull()
    .references(() => users.id),
  kind: text("kind").notNull(), // "guest"
  secretHash: text("secret_hash").notNull(),
});

export const sessions = pgTable("sessions", {
  id: uuid("id").primaryKey(),
  userId: uuid("user_id")
    .notNull()
    .references(() => users.id),
  tokenHash: text("token_hash").notNull().unique(),
  createdAt: ms("created_at").notNull(),
  lastSeenAt: ms("last_seen_at").notNull(),
});

export const worlds = pgTable("worlds", {
  id: uuid("id").primaryKey(),
  name: text("name").notNull(),
  hostUserId: uuid("host_user_id")
    .notNull()
    .references(() => users.id),
  inviteCode: text("invite_code").notNull().unique(),
  balance: text("balance").notNull(),
  seed: integer("seed").notNull(),
  createdAt: ms("created_at").notNull(),
  ownerInstance: text("owner_instance"),
  leaseUntil: ms("lease_until"),
});

export const memberships = pgTable(
  "memberships",
  {
    worldId: uuid("world_id")
      .notNull()
      .references(() => worlds.id),
    userId: uuid("user_id")
      .notNull()
      .references(() => users.id),
    displayName: text("display_name").notNull(),
    joinedAt: ms("joined_at").notNull(),
  },
  (t) => [
    primaryKey({ columns: [t.worldId, t.userId] }),
    index("memberships_user_idx").on(t.userId),
  ],
);

/** The append-only input log; replaying it from a snapshot rebuilds a world. */
export const worldInputs = pgTable(
  "world_inputs",
  {
    worldId: uuid("world_id")
      .notNull()
      .references(() => worlds.id),
    seq: integer("seq").notNull(),
    kind: text("kind").notNull(),
    payload: jsonb("payload").notNull(),
    cid: uuid("cid"),
    userId: uuid("user_id"),
    at: ms("at").notNull(),
  },
  (t) => [
    primaryKey({ columns: [t.worldId, t.seq] }),
    // A command id is applied once per user (NULL cids, e.g. ticks, never collide).
    unique("world_inputs_cid_unique").on(t.worldId, t.userId, t.cid),
  ],
);

export const worldSnapshots = pgTable(
  "world_snapshots",
  {
    worldId: uuid("world_id")
      .notNull()
      .references(() => worlds.id),
    seq: integer("seq").notNull(),
    simVersion: text("sim_version").notNull(),
    state: jsonb("state").notNull(),
    at: ms("at").notNull(),
  },
  (t) => [primaryKey({ columns: [t.worldId, t.seq] })],
);

export const jobs = pgTable(
  "jobs",
  {
    id: uuid("id").primaryKey(),
    /** Arrival order: jobs enqueued in the same millisecond still run first-come first-served. */
    seq: serial("seq").notNull(),
    type: text("type").notNull(),
    priority: integer("priority").notNull().default(0),
    worldId: uuid("world_id"),
    /** The player whose card must do it (Fair mode off) or whose share it counts against (on). */
    ownerUserId: text("owner_user_id"),
    /** The company it is for, for the Fair-mode share. */
    companyId: text("company_id"),
    /** The biggest model it loads, in parameters: a card must be able to hold it. */
    minParams: integer("min_params").notNull().default(0),
    payload: jsonb("payload").notNull(),
    status: text("status").notNull(),
    leaseWorker: text("lease_worker"),
    leaseUntil: ms("lease_until"),
    attempts: integer("attempts").notNull().default(0),
    result: jsonb("result"),
    error: text("error"),
    createdAt: ms("created_at").notNull(),
    updatedAt: ms("updated_at").notNull(),
  },
  (t) => [index("jobs_queue_idx").on(t.status, t.priority, t.seq)],
);

export const trainingRuns = pgTable("training_runs", {
  id: uuid("id").primaryKey(),
  worldId: uuid("world_id").notNull(),
  companyId: uuid("company_id").notNull(),
  jobId: uuid("job_id"),
  /** The version this run will become; its checkpoint is uploaded under this id. */
  versionId: uuid("version_id").notNull(),
  status: text("status").notNull(),
  createdAt: ms("created_at").notNull(),
});

export const trainingEvents = pgTable(
  "training_events",
  {
    runId: uuid("run_id").notNull(),
    idx: integer("idx").notNull(),
    step: integer("step").notNull(),
    event: jsonb("event").notNull(),
  },
  (t) => [primaryKey({ columns: [t.runId, t.idx] })],
);

export const lineages = pgTable("lineages", {
  id: uuid("id").primaryKey(),
  worldId: uuid("world_id").notNull(),
  companyId: uuid("company_id").notNull(),
});

export const modelVersions = pgTable("model_versions", {
  id: uuid("id").primaryKey(),
  lineageId: uuid("lineage_id").notNull(),
  worldId: uuid("world_id").notNull(),
  companyId: uuid("company_id").notNull(),
  parentId: uuid("parent_id"),
  runId: uuid("run_id").notNull(),
  blobKey: text("blob_key").notNull(),
  shape: jsonb("shape").notNull(),
  createdAt: ms("created_at").notNull(),
});

export const evalResults = pgTable("eval_results", {
  versionId: uuid("version_id").primaryKey(),
  report: jsonb("report").notNull(),
  fingerprint: jsonb("fingerprint"),
});

export const chatThreads = pgTable("chat_threads", {
  id: uuid("id").primaryKey(),
  worldId: uuid("world_id").notNull(),
  userId: uuid("user_id").notNull(),
  /** A player's version (a uuid) or a rival's (e.g. "rival-1-v2"). */
  versionId: text("version_id").notNull(),
  createdAt: ms("created_at").notNull(),
  /** "chat", or "interview": the press is listening, and answers can be published. */
  kind: text("kind").notNull().default("chat"),
  /** An interview's product (the AI on sale being interviewed). */
  productId: text("product_id"),
});

export const chatMessages = pgTable(
  "chat_messages",
  {
    threadId: uuid("thread_id").notNull(),
    idx: integer("idx").notNull(),
    role: text("role").notNull(),
    text: text("text").notNull(),
    at: ms("at").notNull(),
    /** In an interview, the prompt card a question came from (null for the player's own). */
    cardId: text("card_id"),
  },
  (t) => [primaryKey({ columns: [t.threadId, t.idx] })],
);

/** A published interview: which answers of which thread, and the job that scores them. */
export const interviewRequests = pgTable("interview_requests", {
  id: uuid("id").primaryKey(),
  worldId: uuid("world_id").notNull(),
  companyId: text("company_id").notNull(),
  productId: text("product_id").notNull(),
  threadId: uuid("thread_id").notNull(),
  idxs: jsonb("idxs").notNull(),
  jobId: uuid("job_id"),
  createdAt: ms("created_at").notNull(),
});

/** A company's notebook: the fingerprint of the text it holds now. */
export const notebooks = pgTable("notebooks", {
  companyId: text("company_id").primaryKey(),
  worldId: uuid("world_id").notNull(),
  hash: text("hash").notNull(),
});

/** A player's helper key version: the key itself is derived from the server secret, never stored. */
export const helperKeys = pgTable("helper_keys", {
  userId: uuid("user_id").primaryKey(),
  version: integer("version").notNull().default(0),
  createdAt: ms("created_at").notNull(),
});

/** Notebook texts by their sha256, so a run trains on exactly the text saved when it started. */
export const notebookTexts = pgTable("notebook_texts", {
  hash: text("hash").primaryKey(),
  text: text("text").notNull(),
});

/** One customer-feedback batch per product and key ("launch", "w12"): a restart never asks twice. */
export const feedbackRequests = pgTable(
  "feedback_requests",
  {
    productId: uuid("product_id").notNull(),
    key: text("key").notNull(),
    jobId: uuid("job_id"),
  },
  (t) => [primaryKey({ columns: [t.productId, t.key] })],
);

export const TABLE_NAMES = [
  "users",
  "auth_identities",
  "sessions",
  "worlds",
  "memberships",
  "world_inputs",
  "world_snapshots",
  "jobs",
  "training_runs",
  "training_events",
  "lineages",
  "model_versions",
  "eval_results",
  "chat_threads",
  "chat_messages",
  "feedback_requests",
] as const;

/** One job per battle: a restart never runs a battle twice. */
export const battleRequests = pgTable("battle_requests", {
  battleId: text("battle_id").primaryKey(),
  worldId: uuid("world_id").notNull(),
  /** The battle's job, or side a's in a battle between friends' AIs. */
  jobId: uuid("job_id"),
  /** Side b's job in a battle between friends' AIs (Stage 2). */
  sideBJobId: uuid("side_b_job_id"),
});

/** Playground minigame runs (playtest pack §4): practice, duels and showcases, private to their player. */
export const minigameRuns = pgTable(
  "minigame_runs",
  {
    id: uuid("id").primaryKey(),
    worldId: uuid("world_id").notNull(),
    userId: uuid("user_id").notNull(),
    companyId: text("company_id").notNull(),
    game: text("game").notNull(),
    difficulty: text("difficulty").notNull(),
    settings: jsonb("settings").notNull(),
    versionId: text("version_id").notNull(),
    /** The rival AI of a duel: its pack key, and how the player sees it named. */
    opponentPack: text("opponent_pack"),
    opponentName: text("opponent_name"),
    /** Set when the run is a showcase the world booked. */
    showcaseId: text("showcase_id").unique(),
    seed: integer("seed").notNull(),
    status: text("status").notNull(),
    jobId: uuid("job_id"),
    result: jsonb("result"),
    score: doublePrecision("score"),
    maxScore: doublePrecision("max_score"),
    opponentScore: doublePrecision("opponent_score"),
    createdAt: bigint("created_at", { mode: "number" }).notNull(),
  },
  (t) => [index("minigame_runs_world_user").on(t.worldId, t.userId)],
);

/** Feedback replies customers loved (playtest pack §5): private training text for their company (R2). */
export const lovedReplies = pgTable(
  "loved_replies",
  {
    id: serial("id").primaryKey(),
    worldId: uuid("world_id").notNull(),
    companyId: text("company_id").notNull(),
    question: text("question").notNull(),
    reply: text("reply").notNull(),
    score: doublePrecision("score").notNull(),
    at: bigint("at", { mode: "number" }).notNull(),
  },
  (t) => [index("loved_replies_company").on(t.worldId, t.companyId)],
);

/** Jobs asked for once per thing: a keynote (`keynote:<product>`), a week of app reviews. */
export const gameRequests = pgTable("game_requests", {
  key: text("key").primaryKey(),
  worldId: uuid("world_id").notNull(),
  jobId: uuid("job_id"),
});
