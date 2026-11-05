CREATE TABLE "helper_keys" (
	"user_id" uuid PRIMARY KEY NOT NULL,
	"version" integer DEFAULT 0 NOT NULL,
	"created_at" bigint NOT NULL
);
