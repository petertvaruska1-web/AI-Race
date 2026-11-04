CREATE TABLE "interview_requests" (
	"id" uuid PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"company_id" text NOT NULL,
	"product_id" text NOT NULL,
	"thread_id" uuid NOT NULL,
	"idxs" jsonb NOT NULL,
	"job_id" uuid,
	"created_at" bigint NOT NULL
);
--> statement-breakpoint
CREATE TABLE "notebook_texts" (
	"hash" text PRIMARY KEY NOT NULL,
	"text" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "notebooks" (
	"company_id" text PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"hash" text NOT NULL
);
--> statement-breakpoint
ALTER TABLE "chat_messages" ADD COLUMN "card_id" text;--> statement-breakpoint
ALTER TABLE "chat_threads" ADD COLUMN "kind" text DEFAULT 'chat' NOT NULL;--> statement-breakpoint
ALTER TABLE "chat_threads" ADD COLUMN "product_id" text;