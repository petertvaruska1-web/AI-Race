import { defineConfig } from "drizzle-kit";

// `npm run db:generate -w @airace/server` writes the SQL migrations into ./drizzle (committed).
export default defineConfig({
  dialect: "postgresql",
  schema: "./src/db/schema.ts",
  out: "./drizzle",
});
