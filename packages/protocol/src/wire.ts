import { z } from "zod";
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
  }),
  z.object({ v, type: z.literal("query"), id: RequestId, q: Query }),
  z.object({ v, type: z.literal("subscribe"), sub: StreamName }),
  z.object({ v, type: z.literal("unsubscribe"), sub: StreamName }),
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
