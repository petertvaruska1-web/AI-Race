import type { JobType } from "@airace/protocol";
import { eq } from "drizzle-orm";
import type { AppContext } from "./app.ts";
import { schema } from "./db/index.ts";
import type { EnqueueOptions, JobRow } from "./jobs/broker.ts";

/**
 * Asks for a job once per key: a row is written before the job is queued, so a restart never
 * asks twice. Returns the earlier job when the key was asked for before (null if it is new).
 */
export async function requestOnce(
  ctx: AppContext,
  key: string,
  worldId: string,
  type: JobType,
  payload: unknown,
  routing: EnqueueOptions = {},
): Promise<JobRow | null> {
  const fresh = await ctx.db
    .insert(schema.gameRequests)
    .values({ key, worldId })
    .onConflictDoNothing()
    .returning();
  if (fresh.length > 0) {
    const jobId = await ctx.broker.enqueue(type, payload, worldId, routing);
    await ctx.db.update(schema.gameRequests).set({ jobId }).where(eq(schema.gameRequests.key, key));
    return null;
  }
  const [row] = await ctx.db
    .select()
    .from(schema.gameRequests)
    .where(eq(schema.gameRequests.key, key));
  return row?.jobId ? await ctx.broker.getJob(row.jobId) : null;
}
