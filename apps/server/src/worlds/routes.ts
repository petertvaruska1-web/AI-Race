import { randomInt, randomUUID } from "node:crypto";
import { GoalSpec, NO_GOAL, TEXT_LIMITS, zName } from "@airace/protocol";
import { createWorld, getBalance, type WorldState } from "@airace/sim";
import { eq } from "drizzle-orm";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import type { AppContext } from "../app.ts";
import { requireUser, sendError } from "../auth.ts";
import { SIM_VERSION } from "../config.ts";
import { schema } from "../db/index.ts";
import { isUniqueViolation, OutdatedWorldError } from "./actor.ts";
import { generateInviteCode } from "./inviteCode.ts";

const CreateWorldBody = z.object({
  name: zName(...TEXT_LIMITS.worldName),
  goal: z.unknown().optional(),
  /** Fair mode (Stage 2): chosen at creation and fixed for the world's life. */
  fairMode: z.boolean().optional(),
});
const JoinBody = z.object({ code: z.string().trim().min(1).max(32) });
const NO_SUCH_CODE = "That invite code doesn't match a world.";
const RIVAL_GOALS = ["bestAi", "beatArch"];

/**
 * Writes a new world: its row, the creator's membership and the starting state
 * as snapshot 0. `createWorld` already seats the creator, so replay starts from
 * that snapshot and needs no join input.
 */
export async function createWorldRecord(
  ctx: AppContext,
  p: { name: string; hostUserId: string; hostName: string; goal?: GoalSpec; fairMode?: boolean },
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
      goal: p.goal ?? NO_GOAL,
      fairMode: p.fairMode ?? false,
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
      const goal = GoalSpec.safeParse(parsed.data.goal ?? NO_GOAL);
      if (!goal.success) return reply.code(400).send({ error: "That goal isn't valid." });
      if (RIVAL_GOALS.includes(goal.data.goal.kind) && !ctx.rivals.installed) {
        return reply
          .code(400)
          .send({ error: "Rival goals need the rival AIs, which aren't built yet." });
      }
      const { worldId } = await createWorldRecord(ctx, {
        name: parsed.data.name,
        hostUserId: user.userId,
        hostName: user.displayName,
        goal: goal.data,
        fairMode: parsed.data.fairMode ?? false,
      });
      return { worldId };
    } catch (err) {
      return sendError(reply, err);
    }
  });

  /** A world by its invite code, for the join screen: name, size and Fair mode only (Stage 2). */
  app.get("/api/worlds/invite/:code", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      const code = String((req.params as { code?: string }).code ?? "").toUpperCase();
      const [world] = await ctx.db
        .select({ id: schema.worlds.id })
        .from(schema.worlds)
        .where(eq(schema.worlds.inviteCode, code));
      if (!world) return reply.code(404).send({ error: NO_SUCH_CODE });
      const s = (await ctx.worlds.get(world.id)).state();
      return {
        name: s.name,
        fairMode: s.fairMode === true,
        players: Object.keys(s.members).length,
        maxPlayers: getBalance(s.balance).maxMembers,
        member: !!s.members[user.userId],
        // A member is simply taken to the world.
        ...(s.members[user.userId] ? { worldId: world.id } : {}),
      };
    } catch (err) {
      return sendError(reply, err);
    }
  });

  /**
   * Joins a friend's world by its invite code (Stage 2). Joining again is fine; a running world
   * can be joined (the newcomer's founding money grows with the world's age, sim-7).
   */
  app.post("/api/worlds/join", async (req, reply) => {
    try {
      const user = await requireUser(ctx, req);
      const parsed = JoinBody.safeParse(req.body);
      if (!parsed.success) return reply.code(404).send({ error: NO_SUCH_CODE });
      const code = parsed.data.code.toUpperCase();
      const [world] = await ctx.db
        .select({ id: schema.worlds.id })
        .from(schema.worlds)
        .where(eq(schema.worlds.inviteCode, code));
      if (!world) return reply.code(404).send({ error: NO_SUCH_CODE });
      const actor = await ctx.worlds.get(world.id);
      const s = actor.state();
      if (!s.members[user.userId]) {
        const max = getBalance(s.balance).maxMembers;
        if (Object.keys(s.members).length >= max) {
          return reply.code(409).send({ error: `This world is full (${max} players).` });
        }
        const r = await actor.submit({
          kind: "system",
          event: { type: "memberJoined", userId: user.userId, displayName: user.displayName },
        });
        if (!r.ok) return reply.code(409).send({ error: r.reason });
        // Joining isn't being here: the player is online once their page opens the world
        // (the gateway marks it), so a join alone never keeps the clock running.
        await actor.submit({
          kind: "system",
          event: { type: "presence", userId: user.userId, online: false },
        });
      }
      // Also on a repeat, so a join cut short between the two steps is finished.
      await ctx.db
        .insert(schema.memberships)
        .values({
          worldId: world.id,
          userId: user.userId,
          displayName: user.displayName,
          joinedAt: ctx.opts.scheduler.now(),
        })
        .onConflictDoNothing();
      return { worldId: world.id };
    } catch (err) {
      return sendError(reply, err);
    }
  });

  /** What this computer has installed: the training data and the rival AIs. */
  app.get("/api/setup", async (req, reply) => {
    try {
      await requireUser(ctx, req);
      return { trainingData: ctx.catalog !== null, rivals: ctx.rivals.installed };
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
