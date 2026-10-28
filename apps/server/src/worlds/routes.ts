import { randomInt, randomUUID } from "node:crypto";
import { TEXT_LIMITS, zName } from "@airace/protocol";
import { createWorld, type WorldState } from "@airace/sim";
import { eq } from "drizzle-orm";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import type { AppContext } from "../app.ts";
import { requireUser, sendError } from "../auth.ts";
import { SIM_VERSION } from "../config.ts";
import { schema } from "../db/index.ts";
import { isUniqueViolation, OutdatedWorldError } from "./actor.ts";
import { generateInviteCode } from "./inviteCode.ts";

const CreateWorldBody = z.object({ name: zName(...TEXT_LIMITS.worldName) });

/**
 * Writes a new world: its row, the creator's membership and the starting state
 * as snapshot 0. `createWorld` already seats the creator, so replay starts from
 * that snapshot and needs no join input.
 */
export async function createWorldRecord(
  ctx: AppContext,
  p: { name: string; hostUserId: string; hostName: string },
): Promise<{ worldId: string; inviteCode: string }> {
  const now = ctx.opts.scheduler.now();
  for (let attempt = 0; ; attempt++) {
    const worldId = randomUUID();
    const inviteCode = generateInviteCode();
    const seed = randomInt(2 ** 31);
    const state = createWorld({
      worldId,
      name: p.name,
      hostUserId: p.hostUserId,
      hostName: p.hostName,
      seed,
      balance: ctx.opts.balance,
      inviteCode,
    });
    try {
      await ctx.db.transaction(async (tx) => {
        await tx.insert(schema.worlds).values({
          id: worldId,
          name: p.name,
          hostUserId: p.hostUserId,
          inviteCode,
          balance: ctx.opts.balance,
          seed,
          createdAt: now,
        });
        await tx
          .insert(schema.memberships)
          .values({ worldId, userId: p.hostUserId, displayName: p.hostName, joinedAt: now });
        await tx
          .insert(schema.worldSnapshots)
          .values({ worldId, seq: 0, simVersion: SIM_VERSION, state, at: now });
      });
      return { worldId, inviteCode };
    } catch (err) {
      // An invite code already in use: draw another (31^6 codes make this rare).
      if (isUniqueViolation(err) && attempt < 5) continue;
      throw err;
    }
  }
}

export function registerWorldRoutes(app: FastifyInstance, ctx: AppContext): void {
  app.post("/api/worlds", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      const parsed = CreateWorldBody.safeParse(req.body);
      if (!parsed.success) {
        const [min, max] = TEXT_LIMITS.worldName;
        return reply.code(400).send({ error: `World names are ${min}–${max} characters.` });
      }
      const { worldId } = await createWorldRecord(ctx, {
        name: parsed.data.name,
        hostUserId: user.userId,
        hostName: user.displayName,
      });
      return { worldId };
    } catch (err) {
      return sendError(reply, err);
    }
  });

  app.get("/api/worlds", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      const rows = await ctx.db
        .select({ worldId: schema.memberships.worldId, joinedAt: schema.memberships.joinedAt })
        .from(schema.memberships)
        .where(eq(schema.memberships.userId, user.userId))
        .orderBy(schema.memberships.joinedAt);
      const worlds = [];
      for (const row of rows) {
        let s: WorldState;
        try {
          s = (await ctx.worlds.get(row.worldId)).state();
        } catch (err) {
          if (!(err instanceof OutdatedWorldError)) throw err;
          // Made with an older version: listed so the player knows, but it can't be opened.
          const [world] = await ctx.db
            .select({ name: schema.worlds.name })
            .from(schema.worlds)
            .where(eq(schema.worlds.id, row.worldId));
          worlds.push({ id: row.worldId, name: world?.name ?? "", outdated: true });
          continue;
        }
        worlds.push({
          id: s.worldId,
          name: s.name,
          phase: s.phase,
          day: s.day,
          memberCount: Object.keys(s.members).length,
        });
      }
      return { worlds };
    } catch (err) {
      return sendError(reply, err);
    }
  });
}
