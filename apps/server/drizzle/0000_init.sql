CREATE TABLE "auth_identities" (
	"id" uuid PRIMARY KEY NOT NULL,
	"user_id" uuid NOT NULL,
	"kind" text NOT NULL,
	"secret_hash" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "chat_messages" (
	"thread_id" uuid NOT NULL,
	"idx" integer NOT NULL,
	"role" text NOT NULL,
	"text" text NOT NULL,
	"at" bigint NOT NULL,
	CONSTRAINT "chat_messages_thread_id_idx_pk" PRIMARY KEY("thread_id","idx")
);
--> statement-breakpoint
CREATE TABLE "chat_threads" (
	"id" uuid PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"version_id" uuid NOT NULL,
	"created_at" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "eval_results" (
	"version_id" uuid PRIMARY KEY NOT NULL,
	"report" jsonb NOT NULL,
	"fingerprint" jsonb
);
--> statement-breakpoint
CREATE TABLE "jobs" (
	"id" uuid PRIMARY KEY NOT NULL,
	"type" text NOT NULL,
	"priority" integer DEFAULT 0 NOT NULL,
	"world_id" uuid,
	"payload" jsonb NOT NULL,
	"status" text NOT NULL,
	"lease_worker" text,
	"lease_until" bigint,
	"attempts" integer DEFAULT 0 NOT NULL,
	"result" jsonb,
	"error" text,
	"created_at" bigint NOT NULL,
	"updated_at" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "lineages" (
	"id" uuid PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"company_id" uuid NOT NULL
);
--> statement-breakpoint
CREATE TABLE "memberships" (
	"world_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"display_name" text NOT NULL,
	"joined_at" bigint NOT NULL,
	CONSTRAINT "memberships_world_id_user_id_pk" PRIMARY KEY("world_id","user_id")
);
--> statement-breakpoint
CREATE TABLE "model_versions" (
	"id" uuid PRIMARY KEY NOT NULL,
	"lineage_id" uuid NOT NULL,
	"world_id" uuid NOT NULL,
	"company_id" uuid NOT NULL,
	"parent_id" uuid,
	"run_id" uuid NOT NULL,
	"blob_key" text NOT NULL,
	"shape" jsonb NOT NULL,
	"created_at" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"id" uuid PRIMARY KEY NOT NULL,
	"user_id" uuid NOT NULL,
	"token_hash" text NOT NULL,
	"created_at" bigint NOT NULL,
	"last_seen_at" bigint NOT NULL,
	CONSTRAINT "sessions_token_hash_unique" UNIQUE("token_hash")
);
--> statement-breakpoint
CREATE TABLE "training_events" (
	"run_id" uuid NOT NULL,
	"idx" integer NOT NULL,
	"step" integer NOT NULL,
	"event" jsonb NOT NULL,
	CONSTRAINT "training_events_run_id_idx_pk" PRIMARY KEY("run_id","idx")
);
--> statement-breakpoint
CREATE TABLE "training_runs" (
	"id" uuid PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"company_id" uuid NOT NULL,
	"job_id" uuid,
	"status" text NOT NULL,
	"created_at" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" uuid PRIMARY KEY NOT NULL,
	"display_name" text NOT NULL,
	"created_at" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "world_inputs" (
	"world_id" uuid NOT NULL,
	"seq" integer NOT NULL,
	"kind" text NOT NULL,
	"payload" jsonb NOT NULL,
	"cid" uuid,
	"user_id" uuid,
	"at" bigint NOT NULL,
	CONSTRAINT "world_inputs_world_id_seq_pk" PRIMARY KEY("world_id","seq"),
	CONSTRAINT "world_inputs_cid_unique" UNIQUE("world_id","user_id","cid")
);
--> statement-breakpoint
CREATE TABLE "world_snapshots" (
	"world_id" uuid NOT NULL,
	"seq" integer NOT NULL,
	"sim_version" text NOT NULL,
	"state" jsonb NOT NULL,
	"at" bigint NOT NULL,
	CONSTRAINT "world_snapshots_world_id_seq_pk" PRIMARY KEY("world_id","seq")
);
--> statement-breakpoint
CREATE TABLE "worlds" (
	"id" uuid PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"host_user_id" uuid NOT NULL,
	"invite_code" text NOT NULL,
	"balance" text NOT NULL,
	"seed" integer NOT NULL,
	"created_at" bigint NOT NULL,
	"owner_instance" text,
	"lease_until" bigint,
	CONSTRAINT "worlds_invite_code_unique" UNIQUE("invite_code")
);
--> statement-breakpoint
ALTER TABLE "auth_identities" ADD CONSTRAINT "auth_identities_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "memberships" ADD CONSTRAINT "memberships_world_id_worlds_id_fk" FOREIGN KEY ("world_id") REFERENCES "public"."worlds"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "memberships" ADD CONSTRAINT "memberships_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "world_inputs" ADD CONSTRAINT "world_inputs_world_id_worlds_id_fk" FOREIGN KEY ("world_id") REFERENCES "public"."worlds"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "world_snapshots" ADD CONSTRAINT "world_snapshots_world_id_worlds_id_fk" FOREIGN KEY ("world_id") REFERENCES "public"."worlds"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "worlds" ADD CONSTRAINT "worlds_host_user_id_users_id_fk" FOREIGN KEY ("host_user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "jobs_queue_idx" ON "jobs" USING btree ("status","priority","created_at");--> statement-breakpoint
CREATE INDEX "memberships_user_idx" ON "memberships" USING btree ("user_id");