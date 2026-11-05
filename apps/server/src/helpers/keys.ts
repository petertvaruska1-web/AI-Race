import { createHmac, timingSafeEqual } from "node:crypto";
import { eq } from "drizzle-orm";
import type { FastifyInstance } from "fastify";
import type { AppContext } from "../app.ts";
import { requireUser, sendError } from "../auth.ts";
import { schema } from "../db/index.ts";

// Personal helper keys (Stage 2): the key a player's helper program presents, so the game
// knows whose graphics card it is. A key is `<userId>.<mac>`, the mac derived from the
// server's secret and the key's version, so nothing secret is stored and the key can be
// shown again whenever the player needs it. Rotating bumps the version: old keys stop.

const mac = (ctx: AppContext, userId: string, version: number) =>
  createHmac("sha256", ctx.opts.secret).update(`helper:${userId}:${version}`).digest("base64url");

async function versionOf(ctx: AppContext, userId: string): Promise<number> {
  const [row] = await ctx.db
    .select({ version: schema.helperKeys.version })
    .from(schema.helperKeys)
    .where(eq(schema.helperKeys.userId, userId));
  if (row) return row.version;
  await ctx.db
    .insert(schema.helperKeys)
    .values({ userId, version: 0, createdAt: ctx.opts.scheduler.now() })
    .onConflictDoNothing();
  return 0;
}

/** The player's current helper key (made on first use). */
export async function helperKeyFor(ctx: AppContext, userId: string): Promise<string> {
  return `${userId}.${mac(ctx, userId, await versionOf(ctx, userId))}`;
}

/** Replaces the player's helper key; the old one stops working. */
export async function rotateHelperKey(ctx: AppContext, userId: string): Promise<string> {
  const version = (await versionOf(ctx, userId)) + 1;
  await ctx.db
    .update(schema.helperKeys)
    .set({ version })
    .where(eq(schema.helperKeys.userId, userId));
  return `${userId}.${mac(ctx, userId, version)}`;
}

/** The player a helper key belongs to, or null for anything else. */
export async function userOfHelperKey(ctx: AppContext, key: string): Promise<string | null> {
  const dot = key.indexOf(".");
  if (dot <= 0) return null;
  const userId = key.slice(0, dot);
  let version: number | undefined;
  try {
    const [row] = await ctx.db
      .select({ version: schema.helperKeys.version })
      .from(schema.helperKeys)
      .where(eq(schema.helperKeys.userId, userId));
    version = row?.version;
  } catch {
    return null; // not a user id
  }
  if (version === undefined) return null;
  const given = Buffer.from(key.slice(dot + 1));
  const want = Buffer.from(mac(ctx, userId, version));
  return given.length === want.length && timingSafeEqual(given, want) ? userId : null;
}

export function registerHelperKeyRoutes(app: FastifyInstance, ctx: AppContext): void {
  app.get("/api/helper/key", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      return { key: await helperKeyFor(ctx, user.userId) };
    } catch (err) {
      return sendError(reply, err);
    }
  });
  app.post("/api/helper/key/rotate", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      return { key: await rotateHelperKey(ctx, user.userId) };
    } catch (err) {
      return sendError(reply, err);
    }
  });
}
