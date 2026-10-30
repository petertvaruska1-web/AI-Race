import {
  bigint,
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
});

export const chatMessages = pgTable(
  "chat_messages",
  {
    threadId: uuid("thread_id").notNull(),
    idx: integer("idx").notNull(),
    role: text("role").notNull(),
    text: text("text").notNull(),
    at: ms("at").notNull(),
  },
  (t) => [primaryKey({ columns: [t.threadId, t.idx] })],
);

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
