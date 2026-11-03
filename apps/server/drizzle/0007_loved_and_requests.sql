CREATE TABLE "game_requests" (
	"key" text PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"job_id" uuid
);
--> statement-breakpoint
CREATE TABLE "loved_replies" (
	"id" serial PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"company_id" text NOT NULL,
	"question" text NOT NULL,
	"reply" text NOT NULL,
	"score" double precision NOT NULL,
	"at" bigint NOT NULL
);
--> statement-breakpoint
CREATE INDEX "loved_replies_company" ON "loved_replies" USING btree ("world_id","company_id");