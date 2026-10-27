import { randomUUID } from "node:crypto";
import { ClientMessage, type Command, type Query } from "@airace/protocol";
import type { Input } from "@airace/sim";
import websocket from "@fastify/websocket";
import type { FastifyInstance } from "fastify";
import type { RawData, WebSocket } from "ws";
import type { AppContext } from "../app.ts";
import { verifyWsToken } from "../auth.ts";
import { NO_TRAINING_DATA } from "../catalog.ts";
import type { TimerHandle } from "../scheduler.ts";
import type { WorldActor } from "../worlds/actor.ts";
import { WorldWatch } from "./broadcast.ts";
import { Session } from "./session.ts";

const HELLO_TIMEOUT_MS = 5000;
const PRESENCE_GRACE_MS = 5000;
const RATE_WINDOW_MS = 10_000;
const RATE_MAX_COMMANDS = 20;
/** Player messages are small; the socket layer allows more for workers' results. */
const MAX_MESSAGE_BYTES = 64 * 1024;
const MAX_SOCKET_BYTES = 8 * 1024 * 1024;
/** Close code for a socket that never proved who it is. */
export const CLOSE_UNAUTHORIZED = 4401;

export type StreamAuthorizer = (userId: string, sub: string) => Promise<boolean>;
export type QueryHandler = (userId: string, q: Query) => Promise<unknown>;

/** Commands that create something get their id from the server, inside the logged input. */
function needsId(cmd: Command): boolean {
  return cmd.type === "createCompany" || cmd.type === "startTraining";
}

/**
 * The `/ws` endpoint: sign-in by token, per-player world views (snapshot, then
 * coalesced patches), commands with exactly-once acks, presence with a grace
 * period, streams and queries (spec §5.3).
 */
export class Gateway {
  private sessions = new Set<Session>();
  private presence = new Map<string, { sockets: number; grace: TimerHandle | null }>();
  private rate = new Map<string, number[]>();
  private streams = new Map<string, StreamAuthorizer>();
  private subscribers = new Map<string, Set<Session>>();
  private queries = new Map<string, QueryHandler>();
  private closed = false;

  constructor(private readonly ctx: AppContext) {
    this.registerQuery("ping", async () => ({ pong: true }));
  }

  async register(app: FastifyInstance): Promise<void> {
    await app.register(websocket, { options: { maxPayload: MAX_SOCKET_BYTES } });
    app.get("/ws", { websocket: true }, (socket) => this.accept(socket));
  }

  /** Lets clients follow streams named `<prefix>:...` when `authorize` agrees. */
  registerStream(prefix: string, authorize: StreamAuthorizer): void {
    this.streams.set(prefix, authorize);
  }

  registerQuery(kind: string, handler: QueryHandler): void {
    this.queries.set(kind, handler);
  }

  publish(sub: string, data: unknown): void {
    for (const session of this.subscribers.get(sub) ?? []) {
      session.send({ v: 1, type: "stream", sub, data });
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    for (const entry of this.presence.values()) {
      if (entry.grace) this.ctx.opts.scheduler.clearTimeout(entry.grace);
    }
    this.presence.clear();
  }

  private accept(socket: WebSocket): void {
    const session = new Session(socket);
    this.sessions.add(session);
    const sched = this.ctx.opts.scheduler;
    const hello = sched.setTimeout(() => {
      if (!session.userId) socket.close(CLOSE_UNAUTHORIZED, "sign in first");
    }, HELLO_TIMEOUT_MS);
    socket.on("message", (raw: RawData) => {
      session.queue = session.queue
        .then(() => this.onMessage(session, raw, hello))
        .catch((err) => {
          this.ctx.log.error({ err, userId: session.userId }, "ws message failed");
          session.send({ v: 1, type: "error", reason: "Something went wrong. Please try again." });
        });
    });
    socket.on("close", () => {
      sched.clearTimeout(hello);
      this.drop(session);
    });
  }

  private async onMessage(session: Session, raw: RawData, hello: TimerHandle): Promise<void> {
    const bytes = Array.isArray(raw) ? raw.reduce((n, b) => n + b.length, 0) : raw.byteLength;
    if (bytes > MAX_MESSAGE_BYTES) {
      session.socket.close(1009, "message too large");
      return;
    }
    let data: unknown = null;
    try {
      data = JSON.parse(raw.toString());
    } catch {
      // handled as an invalid message below
    }
    const parsed = ClientMessage.safeParse(data);

    if (!session.userId) {
      const userId =
        parsed.success && parsed.data.type === "hello"
          ? verifyWsToken(this.ctx, parsed.data.token)
          : null;
      if (!userId) {
        session.socket.close(CLOSE_UNAUTHORIZED, "sign in first");
        return;
      }
      session.userId = userId;
      this.ctx.opts.scheduler.clearTimeout(hello);
      session.send({ v: 1, type: "welcome", userId });
      return;
    }
    if (!parsed.success) {
      session.send({ v: 1, type: "error", reason: "That message isn't valid." });
      return;
    }
    const userId = session.userId;
    const msg = parsed.data;
    switch (msg.type) {
      case "hello":
        session.send({ v: 1, type: "error", reason: "Already signed in." });
        return;
      case "watch":
        return this.watch(session, userId, msg.worldId);
      case "unwatch":
        this.unwatch(session, userId, msg.worldId);
        return;
      case "resync": {
        const watch = session.watches.get(msg.worldId);
        if (watch) watch.sendSnapshot();
        else return this.watch(session, userId, msg.worldId);
        return;
      }
      case "command":
        return this.command(session, userId, msg.cid, msg.worldId, msg.cmd);
      case "chat":
        session.send({ v: 1, type: "reject", cid: msg.cid, reason: "Chat isn't available yet." });
        return;
      case "subscribe":
        return this.subscribe(session, userId, msg.sub);
      case "unsubscribe":
        this.subscribers.get(msg.sub)?.delete(session);
        session.subs.delete(msg.sub);
        return;
      case "query":
        return this.query(session, userId, msg.id, msg.q);
    }
  }

  /** The world's actor when `userId` is a member of it, else null. */
  private async memberActor(worldId: string, userId: string): Promise<WorldActor | null> {
    let actor: WorldActor;
    try {
      actor = await this.ctx.worlds.get(worldId);
    } catch {
      return null;
    }
    return actor.state().members[userId] ? actor : null;
  }

  private async watch(session: Session, userId: string, worldId: string): Promise<void> {
    const existing = session.watches.get(worldId);
    if (existing) {
      existing.sendSnapshot();
      return;
    }
    const actor = await this.memberActor(worldId, userId);
    if (!actor) {
      session.send({
        v: 1,
        type: "error",
        reason: "That world doesn't exist or you're not in it.",
      });
      return;
    }
    await this.arrive(actor, userId);
    session.watches.set(worldId, new WorldWatch(session, actor, userId, this.ctx.opts.scheduler));
  }

  private unwatch(session: Session, userId: string, worldId: string): void {
    const watch = session.watches.get(worldId);
    if (!watch) return;
    watch.close();
    session.watches.delete(worldId);
    this.leave(worldId, userId);
  }

  /** A player's first socket on a world marks them online; a quick reconnect cancels leaving. */
  private async arrive(actor: WorldActor, userId: string): Promise<void> {
    const key = `${actor.worldId}|${userId}`;
    const entry = this.presence.get(key) ?? { sockets: 0, grace: null };
    entry.sockets += 1;
    if (entry.grace) {
      this.ctx.opts.scheduler.clearTimeout(entry.grace);
      entry.grace = null;
    }
    this.presence.set(key, entry);
    if (!actor.state().members[userId]?.online) {
      const online: Input = { kind: "system", event: { type: "presence", userId, online: true } };
      await actor.submit(online);
    }
  }

  /** When a player's last socket on a world closes, they go offline after a grace period. */
  private leave(worldId: string, userId: string): void {
    const key = `${worldId}|${userId}`;
    const entry = this.presence.get(key);
    if (!entry) return;
    entry.sockets -= 1;
    if (entry.sockets > 0 || this.closed) return;
    entry.grace = this.ctx.opts.scheduler.setTimeout(async () => {
      if (this.presence.get(key) !== entry || entry.sockets > 0) return;
      this.presence.delete(key);
      if (this.closed || !this.ctx.worlds.isLoaded(worldId)) return;
      const actor = await this.ctx.worlds.get(worldId);
      if (!actor.state().members[userId]?.online) return;
      const offline: Input = { kind: "system", event: { type: "presence", userId, online: false } };
      await actor.submit(offline);
    }, PRESENCE_GRACE_MS);
  }

  private async command(
    session: Session,
    userId: string,
    cid: string,
    worldId: string,
    cmd: Command,
  ): Promise<void> {
    const actor = await this.memberActor(worldId, userId);
    if (!actor) {
      session.send({
        v: 1,
        type: "reject",
        cid,
        reason: "That world doesn't exist or you're not in it.",
      });
      return;
    }
    if (cmd.type === "startTraining" && !this.ctx.catalog) {
      session.send({ v: 1, type: "reject", cid, reason: NO_TRAINING_DATA });
      return;
    }
    const now = this.ctx.opts.scheduler.now();
    const key = `${worldId}|${userId}`;
    const recent = (this.rate.get(key) ?? []).filter((t) => t > now - RATE_WINDOW_MS);
    if (recent.length >= RATE_MAX_COMMANDS) {
      this.rate.set(key, recent);
      session.send({
        v: 1,
        type: "reject",
        cid,
        reason: "You're going too fast. Slow down a moment.",
      });
      return;
    }
    recent.push(now);
    this.rate.set(key, recent);
    const input: Input = {
      kind: "command",
      userId,
      cmd,
      ...(needsId(cmd) ? { newId: randomUUID() } : {}),
    };
    const r = await actor.submit(input, { cid, userId });
    if (r.ok) session.send({ v: 1, type: "ack", cid, duplicate: r.duplicate });
    else session.send({ v: 1, type: "reject", cid, reason: r.reason });
  }

  private async subscribe(session: Session, userId: string, sub: string): Promise<void> {
    const authorize = this.streams.get(sub.split(":")[0] ?? "");
    if (!authorize || !(await authorize(userId, sub))) {
      session.send({ v: 1, type: "error", reason: "You can't follow that." });
      return;
    }
    let set = this.subscribers.get(sub);
    if (!set) {
      set = new Set();
      this.subscribers.set(sub, set);
    }
    set.add(session);
    session.subs.add(sub);
  }

  private async query(session: Session, userId: string, id: string, q: Query): Promise<void> {
    const handler = this.queries.get(q.kind);
    if (!handler) {
      session.send({ v: 1, type: "queryResult", id, ok: false, error: "Unknown request." });
      return;
    }
    try {
      session.send({ v: 1, type: "queryResult", id, ok: true, data: await handler(userId, q) });
    } catch (err) {
      const message =
        err instanceof Error && "statusCode" in err ? err.message : "Something went wrong.";
      if (!(err instanceof Error && "statusCode" in err))
        this.ctx.log.error({ err, userId }, "query failed");
      session.send({ v: 1, type: "queryResult", id, ok: false, error: message });
    }
  }

  private drop(session: Session): void {
    this.sessions.delete(session);
    for (const [worldId, watch] of session.watches) {
      watch.close();
      if (session.userId) this.leave(worldId, session.userId);
    }
    session.watches.clear();
    for (const sub of session.subs) this.subscribers.get(sub)?.delete(session);
    session.subs.clear();
  }
}
