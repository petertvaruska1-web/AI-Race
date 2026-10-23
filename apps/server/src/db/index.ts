import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { PGlite } from "@electric-sql/pglite";
import { drizzle, type PgliteDatabase } from "drizzle-orm/pglite";
import { migrate } from "drizzle-orm/pglite/migrator";
import * as schema from "./schema.ts";

export { schema };
export type Db = PgliteDatabase<typeof schema>;

const MIGRATIONS = fileURLToPath(new URL("../../drizzle", import.meta.url));

/** Opens PGlite (on disk, or in memory when `dir` is null) and applies migrations. */
export async function openDb(
  dir: string | null,
): Promise<{ db: Db; client: PGlite; close(): Promise<void> }> {
  if (dir) mkdirSync(dir, { recursive: true });
  const client = dir ? new PGlite(dir) : new PGlite();
  await client.waitReady;
  const db = drizzle({ client, schema });
  await migrate(db, { migrationsFolder: MIGRATIONS });
  return { db, client, close: () => client.close() };
}
