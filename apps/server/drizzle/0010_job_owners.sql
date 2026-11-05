ALTER TABLE "jobs" ADD COLUMN "owner_user_id" text;--> statement-breakpoint
ALTER TABLE "jobs" ADD COLUMN "company_id" text;--> statement-breakpoint
ALTER TABLE "jobs" ADD COLUMN "min_params" integer DEFAULT 0 NOT NULL;