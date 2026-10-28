CREATE TABLE "feedback_requests" (
	"product_id" uuid NOT NULL,
	"key" text NOT NULL,
	"job_id" uuid,
	CONSTRAINT "feedback_requests_product_id_key_pk" PRIMARY KEY("product_id","key")
);
