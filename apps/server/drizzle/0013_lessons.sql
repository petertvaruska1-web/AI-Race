CREATE TABLE "lessons" (
	"id" serial PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"company_id" text NOT NULL,
	"question" text NOT NULL,
	"answer" text NOT NULL,
	"kind" text NOT NULL,
	"source_thread" uuid NOT NULL,
	"source_idx" integer NOT NULL,
	"at" bigint NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX "lessons_source_idx" ON "lessons" USING btree ("source_thread","source_idx","kind");--> statement-breakpoint
CREATE INDEX "lessons_company_idx" ON "lessons" USING btree ("world_id","company_id","id");