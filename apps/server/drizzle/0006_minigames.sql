CREATE TABLE "minigame_runs" (
	"id" uuid PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"company_id" text NOT NULL,
	"game" text NOT NULL,
	"difficulty" text NOT NULL,
	"settings" jsonb NOT NULL,
	"version_id" text NOT NULL,
	"opponent_pack" text,
	"opponent_name" text,
	"showcase_id" text,
	"seed" integer NOT NULL,
	"status" text NOT NULL,
	"job_id" uuid,
	"result" jsonb,
	"score" double precision,
	"max_score" double precision,
	"opponent_score" double precision,
	"created_at" bigint NOT NULL,
	CONSTRAINT "minigame_runs_showcase_id_unique" UNIQUE("showcase_id")
);
--> statement-breakpoint
CREATE INDEX "minigame_runs_world_user" ON "minigame_runs" USING btree ("world_id","user_id");