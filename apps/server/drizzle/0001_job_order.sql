DROP INDEX "jobs_queue_idx";--> statement-breakpoint
ALTER TABLE "jobs" ADD COLUMN "seq" serial NOT NULL;--> statement-breakpoint
CREATE INDEX "jobs_queue_idx" ON "jobs" USING btree ("status","priority","seq");