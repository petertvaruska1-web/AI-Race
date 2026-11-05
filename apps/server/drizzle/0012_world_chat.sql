CREATE TABLE "world_chat" (
	"id" serial PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"name" text NOT NULL,
	"text" text NOT NULL,
	"at" bigint NOT NULL
);
--> statement-breakpoint
CREATE INDEX "world_chat_world_idx" ON "world_chat" USING btree ("world_id","id");