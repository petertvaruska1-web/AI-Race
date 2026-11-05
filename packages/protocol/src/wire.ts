import { z } from "zod";
import { Difficulty, DUEL_GAMES, GameId, MinigameSettings } from "./minigames.ts";
import { PatchOp } from "./patch.ts";
import { TEXT_LIMITS, zId, zText } from "./text.ts";
import { WorldView } from "./view.ts";
import { Command } from "./world.ts";

const v = z.literal(1);

/** Stream names, e.g. `chat:<worldId>:<threadId>`. */
export const StreamName = z.string().min(1).max(200);
export const RequestId = z.string().min(1).max(64);

/** Read-only requests; later tasks add kinds. */
export const Query = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("ping") }),
  z.object({ kind: z.literal("catalog") }),
  z.object({ kind: z.literal("runTelemetry"), worldId: zId, runId: zId }),
  z.object({ kind: z.literal("chatThreads"), worldId: zId }),
  z.object({ kind: z.literal("chatThread"), worldId: zId, threadId: zId }),
  z.object({ kind: z.literal("battle"), worldId: zId, battleId: zId }),
  /** The player's minigame runs in a world (newest first) and personal bests. */
  z.object({ kind: z.literal("minigameRuns"), worldId: zId }),
  /** The player's own notebook text ("" when empty). */
  z.object({ kind: z.literal("notebook"), worldId: zId }),
  /** Every player's graphics cards in a world (Stage 2). */
  z.object({ kind: z.literal("cards"), worldId: zId }),
  /** The biggest brain the player's card (or Fair mode's shared cards) can train, and its speed. */
  z.object({ kind: z.literal("trainingLimit"), worldId: zId }),
  /** How often other players tried each of your shared AIs this week (counts only). */
  z.object({ kind: z.literal("versionTries"), worldId: zId }),
  /** The world chat's last 200 messages, oldest first. */
  z.object({ kind: z.literal("worldChat"), worldId: zId }),
  /** Your company's coaching lessons, oldest first (private). */
  z.object({ kind: z.literal("lessons"), worldId: zId }),
]);
export type Query = z.infer<typeof Query>;

export const ClientMessage = z.discriminatedUnion("type", [
  z.object({ v, type: z.literal("hello"), token: z.string().min(1).max(512) }),
  z.object({ v, type: z.literal("watch"), worldId: zId }),
  z.object({ v, type: z.literal("unwatch"), worldId: zId }),
  z.object({ v, type: z.literal("resync"), worldId: zId }),
  z.object({ v, type: z.literal("command"), cid: z.uuid(), worldId: zId, cmd: Command }),
  z.object({
    v,
    type: z.literal("chat"),
    cid: z.uuid(),
    worldId: zId,
    threadId: zId.nullable(),
    versionId: zId,
    text: zText(...TEXT_LIMITS.chatMessage, { allowNewlines: true }),
    /** An interview: the press is listening to the AI on sale as `productId`. */
    interview: z
      .object({ productId: zId, cardId: z.string().min(1).max(80).nullable() })
      .optional(),
  }),
  /** Saves the company's notebook (private training text; empty clears it). */
  z.object({
    v,
    type: z.literal("notebookSave"),
    cid: z.uuid(),
    worldId: zId,
    text: zText(0, 40_000, { allowNewlines: true }),
  }),
  /** Publishes chosen answers of an interview the server holds (1-5 of the AI's messages). */
  z.object({
    v,
    type: z.literal("interviewPublish"),
    cid: z.uuid(),
    worldId: zId,
    threadId: zId,
    idxs: z.array(z.number().int().min(0)).min(1).max(5),
  }),
  z.object({ v, type: z.literal("query"), id: RequestId, q: Query }),
  z.object({ v, type: z.literal("subscribe"), sub: StreamName }),
  z.object({ v, type: z.literal("unsubscribe"), sub: StreamName }),
  /** A practice minigame (free and private), or a duel against a rival product on sale. */
  z
    .object({
      v,
      type: z.literal("minigame"),
      cid: z.uuid(),
      worldId: zId,
      game: GameId,
      difficulty: Difficulty,
      settings: MinigameSettings,
      versionId: zId,
      /** A rival product to duel, or null to play alone. */
      opponent: zId.nullable(),
    })
    .refine((m) => m.opponent === null || (DUEL_GAMES as readonly string[]).includes(m.game), {
      message: "Story Dice can't be a duel.",
    }),
  /** A lesson for your own AI (coaching): like its reply, or write the answer you wished for. */
  z.object({
    v,
    type: z.literal("lesson"),
    cid: z.uuid(),
    worldId: zId,
    threadId: zId,
    /** The AI reply's place in the conversation. */
    idx: z.number().int().min(0),
    kind: z.enum(["liked", "written"]),
    text: z.string().max(2000).optional(),
  }),
  z.object({
    v,
    type: z.literal("lessonDelete"),
    cid: z.uuid(),
    worldId: zId,
    lessonId: z.number().int().min(0),
  }),
  /** A message to the world chat (Stage 2); the server checks its length and words. */
  z.object({
    v,
    type: z.literal("worldChat"),
    cid: z.uuid(),
    worldId: zId,
    text: z.string().max(2000),
  }),
  /** A vote on a story round of a battle between two other players' AIs (Stage 2). */
  z.object({
    v,
    type: z.literal("battleVote"),
    cid: z.uuid(),
    worldId: zId,
    battleId: zId,
    round: z.number().int().min(0).max(9),
    pick: z.enum(["left", "right"]),
  }),
  /** The player is in the Playground (sent every 15 s): the world's customer posts wait. */
  z.object({ v, type: z.literal("inPlayground"), worldId: zId }),
]);
export type ClientMessage = z.infer<typeof ClientMessage>;

export const ServerMessage = z.discriminatedUnion("type", [
  z.object({ v, type: z.literal("welcome"), userId: zId }),
  z.object({
    v,
    type: z.literal("snapshot"),
    worldId: zId,
    version: z.number().int().min(0),
    view: WorldView,
  }),
  z.object({
    v,
    type: z.literal("patch"),
    worldId: zId,
    from: z.number().int().min(0),
    to: z.number().int().min(0),
    ops: z.array(PatchOp),
  }),
  z.object({
    v,
    type: z.literal("ack"),
    cid: z.uuid(),
    duplicate: z.boolean(),
    data: z.unknown().optional(),
  }),
  z.object({ v, type: z.literal("reject"), cid: z.uuid(), reason: z.string() }),
  z.object({ v, type: z.literal("event"), worldId: zId, kind: z.string(), text: z.string() }),
  z.object({ v, type: z.literal("stream"), sub: StreamName, data: z.unknown() }),
  z.object({
    v,
    type: z.literal("queryResult"),
    id: RequestId,
    ok: z.boolean(),
    data: z.unknown().optional(),
    error: z.string().optional(),
  }),
  z.object({ v, type: z.literal("error"), reason: z.string() }),
]);
export type ServerMessage = z.infer<typeof ServerMessage>;
