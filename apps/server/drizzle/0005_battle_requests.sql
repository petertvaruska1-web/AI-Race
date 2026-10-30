CREATE TABLE "battle_requests" (
	"battle_id" text PRIMARY KEY NOT NULL,
	"world_id" uuid NOT NULL,
	"job_id" uuid
);
