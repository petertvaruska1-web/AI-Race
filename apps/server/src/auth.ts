import { createHash, createHmac, randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { TEXT_LIMITS, zName } from "@airace/protocol";
import { eq } from "drizzle-orm";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { z } from "zod";
import type { AppContext } from "./app.ts";
import { schema } from "./db/index.ts";

export const SESSION_COOKIE = "airace_session";
const WS_TOKEN_MS = 10 * 60 * 1000;
const COOKIE_MAX_AGE_S = 400 * 24 * 60 * 60; // browsers cap cookie lifetimes at 400 days

const GuestBody = z.object({ displayName: zName(...TEXT_LIMITS.displayName) });

export class HttpError extends Error {
  constructor(
    readonly statusCode: number,
    message: string,
  ) {
    super(message);
  }
}

function sha256(s: string): string {
  return createHash("sha256").update(s).digest("hex");
}

function mac(ctx: AppContext, payload: string): Buffer {
  return createHmac("sha256", ctx.opts.secret).update(`ws:${payload}`).digest();
}

/** A short-lived token that opens a WebSocket as `userId` (cookies don't reach every ws client). */
export function issueWsToken(ctx: AppContext, userId: string): string {
  const exp = ctx.opts.scheduler.now() + WS_TOKEN_MS;
  const payload = `${userId}.${exp}`;
  return Buffer.from(`${payload}.${mac(ctx, payload).toString("base64url")}`).toString("base64url");
}

/** The token's user id, or null when it is malformed, forged or expired. */
export function verifyWsToken(ctx: AppContext, token: string): string | null {
  const decoded = Buffer.from(token, "base64url").toString();
  // The decoder ignores stray characters; only the exact encoding counts.
  if (Buffer.from(decoded).toString("base64url") !== token) return null;
  const parts = decoded.split(".");
  if (parts.length !== 3) return null;
  const [userId, exp, given] = parts as [string, string, string];
  const expected = mac(ctx, `${userId}.${exp}`);
  const actual = Buffer.from(given, "base64url");
  if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) return null;
  if (!(Number(exp) > ctx.opts.scheduler.now())) return null;
  return userId;
}

/** The signed-in user, or a 401. */
export async function requireUser(
  ctx: AppContext,
  req: FastifyRequest,
): Promise<{ userId: string; displayName: string }> {
  const token = req.cookies[SESSION_COOKIE];
  if (!token) throw new HttpError(401, "Please sign in again.");
  const rows = await ctx.db
    .select({ userId: schema.sessions.userId, displayName: schema.users.displayName })
    .from(schema.sessions)
    .innerJoin(schema.users, eq(schema.users.id, schema.sessions.userId))
    .where(eq(schema.sessions.tokenHash, sha256(token)))
    .limit(1);
  const row = rows[0];
  if (!row) throw new HttpError(401, "Please sign in again.");
  return row;
}

export function sendError(reply: FastifyReply, err: unknown): FastifyReply {
  if (err instanceof HttpError) return reply.code(err.statusCode).send({ error: err.message });
  throw err;
}

export function registerAuthRoutes(app: FastifyInstance, ctx: AppContext): void {
  const secure = ctx.opts.publicUrl.startsWith("https:");

  app.post("/api/guest", async (req, reply) => {
    const parsed = GuestBody.safeParse(req.body);
    if (!parsed.success) {
      const [min, max] = TEXT_LIMITS.displayName;
      return reply.code(400).send({ error: `Display names are ${min}–${max} characters.` });
    }
    const now = ctx.opts.scheduler.now();
    const userId = randomUUID();
    const token = randomBytes(32).toString("base64url");
    const tokenHash = sha256(token);
    await ctx.db.transaction(async (tx) => {
      await tx
        .insert(schema.users)
        .values({ id: userId, displayName: parsed.data.displayName, createdAt: now });
      // A guest's only credential is its session token; the identity keeps its hash.
      await tx
        .insert(schema.authIdentities)
        .values({ id: randomUUID(), userId, kind: "guest", secretHash: tokenHash });
      await tx
        .insert(schema.sessions)
        .values({ id: randomUUID(), userId, tokenHash, createdAt: now, lastSeenAt: now });
    });
    reply.setCookie(SESSION_COOKIE, token, {
      httpOnly: true,
      sameSite: "lax",
      secure,
      path: "/",
      maxAge: COOKIE_MAX_AGE_S,
    });
    return { userId, displayName: parsed.data.displayName };
  });

  app.get("/api/me", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      return { ...user, wsToken: issueWsToken(ctx, user.userId) };
    } catch (err) {
      return sendError(reply, err);
    }
  });
}
