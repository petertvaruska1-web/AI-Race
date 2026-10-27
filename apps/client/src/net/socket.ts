import { type ClientMessage, type Command, type Query, ServerMessage } from "@airace/protocol";

/** The parts of a WebSocket the game socket uses (a test passes a fake one). */
export type WsLike = {
  readonly readyState: number;
  send(data: string): void;
  close(): void;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: unknown) => void) | null;
  onerror: ((ev: unknown) => void) | null;
};

export type SocketDeps = {
  url: string;
  open(url: string): WsLike;
  /** A fresh ws token for the signed-in player (they expire after 10 minutes). */
  token(): Promise<string>;
  schedule(fn: () => void, ms: number): unknown;
  cancel(handle: unknown): void;
  uuid(): string;
};

export type SocketStatus = "connecting" | "live" | "offline";
export type CommandResult =
  | { ok: true; duplicate: boolean; data?: unknown }
  | { ok: false; reason: string };
type WorldMessage = Extract<ServerMessage, { type: "snapshot" | "patch" | "event" }>;

/** Waits between reconnects: 0.5, 1, 2, 4, 8, then 10 s. */
export const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000, 10_000];

/**
 * The client's one connection to the server (spec §5.3). It signs in with a
 * ws token, keeps the watched worlds, streams and unanswered commands across
 * reconnects, and resends an unanswered command with the same id, so the
 * server applies it once however often the connection drops (R4: the client
 * only ever sends intents).
 */
export class GameSocket {
  status: SocketStatus = "offline";
  private ws: WsLike | null = null;
  private live = false;
  private stopped = false;
  private attempt = 0;
  private retry: unknown = null;
  private firstWelcome: { resolve: () => void; reject: (e: Error) => void } | null = null;
  private watched = new Set<string>();
  private subs = new Map<string, Set<(data: unknown) => void>>();
  private pending = new Map<string, { msg: ClientMessage; resolve: (r: CommandResult) => void }>();
  private queries = new Map<
    string,
    { msg: ClientMessage; resolve: (data: unknown) => void; reject: (e: Error) => void }
  >();
  private worldListeners = new Set<(m: WorldMessage) => void>();
  private statusListeners = new Set<(s: SocketStatus) => void>();
  private errorListeners = new Set<(reason: string) => void>();

  constructor(private readonly deps: SocketDeps) {}

  connect(): Promise<void> {
    this.stopped = false;
    return new Promise((resolve, reject) => {
      this.firstWelcome = { resolve, reject };
      void this.open();
    });
  }

  close(): void {
    this.stopped = true;
    if (this.retry !== null) this.deps.cancel(this.retry);
    this.ws?.close();
  }

  onWorld(fn: (m: WorldMessage) => void): () => void {
    this.worldListeners.add(fn);
    return () => this.worldListeners.delete(fn);
  }

  onStatus(fn: (s: SocketStatus) => void): () => void {
    this.statusListeners.add(fn);
    return () => this.statusListeners.delete(fn);
  }

  onError(fn: (reason: string) => void): () => void {
    this.errorListeners.add(fn);
    return () => this.errorListeners.delete(fn);
  }

  watch(worldId: string): void {
    this.watched.add(worldId);
    this.send({ v: 1, type: "watch", worldId });
  }

  unwatch(worldId: string): void {
    this.watched.delete(worldId);
    this.send({ v: 1, type: "unwatch", worldId });
  }

  resync(worldId: string): void {
    this.send({ v: 1, type: "resync", worldId });
  }

  command(worldId: string, cmd: Command): Promise<CommandResult> {
    return this.request({ v: 1, type: "command", cid: this.deps.uuid(), worldId, cmd });
  }

  chat(
    worldId: string,
    threadId: string | null,
    versionId: string,
    text: string,
  ): Promise<CommandResult> {
    return this.request({
      v: 1,
      type: "chat",
      cid: this.deps.uuid(),
      worldId,
      threadId,
      versionId,
      text,
    });
  }

  query<T = unknown>(q: Query): Promise<T> {
    const id = this.deps.uuid();
    const msg: ClientMessage = { v: 1, type: "query", id, q };
    return new Promise<T>((resolve, reject) => {
      this.queries.set(id, { msg, resolve: resolve as (d: unknown) => void, reject });
      this.send(msg);
    });
  }

  subscribe(sub: string, onData: (data: unknown) => void): () => void {
    let set = this.subs.get(sub);
    if (!set) {
      set = new Set();
      this.subs.set(sub, set);
      this.send({ v: 1, type: "subscribe", sub });
    }
    set.add(onData);
    return () => {
      const s = this.subs.get(sub);
      if (!s) return;
      s.delete(onData);
      if (s.size === 0) {
        this.subs.delete(sub);
        this.send({ v: 1, type: "unsubscribe", sub });
      }
    };
  }

  // --- connection ----------------------------------------------------------------

  private request(msg: Extract<ClientMessage, { cid: string }>): Promise<CommandResult> {
    return new Promise((resolve) => {
      this.pending.set(msg.cid, { msg, resolve });
      this.send(msg);
    });
  }

  /** Sends now when signed in; otherwise the message waits for the next welcome. */
  private send(msg: ClientMessage): void {
    if (this.live && this.ws) this.ws.send(JSON.stringify(msg));
  }

  private setStatus(s: SocketStatus): void {
    this.status = s;
    for (const fn of this.statusListeners) fn(s);
  }

  private async open(): Promise<void> {
    this.setStatus("connecting");
    let token: string;
    try {
      token = await this.deps.token();
    } catch {
      this.scheduleReconnect();
      return;
    }
    if (this.stopped) return;
    const ws = this.deps.open(this.deps.url);
    this.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ v: 1, type: "hello", token }));
    ws.onmessage = (ev) => this.receive(ev.data);
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.live = false;
      this.setStatus("offline");
      if (!this.stopped) this.scheduleReconnect();
    };
    ws.onerror = () => {};
  }

  private scheduleReconnect(): void {
    const delay =
      RECONNECT_DELAYS_MS[Math.min(this.attempt, RECONNECT_DELAYS_MS.length - 1)] ?? 10_000;
    this.attempt += 1;
    this.setStatus("offline");
    this.retry = this.deps.schedule(() => {
      this.retry = null;
      void this.open();
    }, delay);
  }

  private receive(raw: unknown): void {
    let parsed: ReturnType<typeof ServerMessage.safeParse>;
    try {
      parsed = ServerMessage.safeParse(JSON.parse(String(raw)));
    } catch {
      return;
    }
    if (!parsed.success) {
      console.error("unexpected server message", parsed.error.issues);
      return;
    }
    const msg = parsed.data;
    switch (msg.type) {
      case "welcome":
        this.welcomed();
        return;
      case "snapshot":
      case "patch":
      case "event":
        for (const fn of this.worldListeners) fn(msg);
        return;
      case "ack": {
        const p = this.pending.get(msg.cid);
        this.pending.delete(msg.cid);
        p?.resolve({ ok: true, duplicate: msg.duplicate, data: msg.data });
        return;
      }
      case "reject": {
        const p = this.pending.get(msg.cid);
        this.pending.delete(msg.cid);
        p?.resolve({ ok: false, reason: msg.reason });
        return;
      }
      case "queryResult": {
        const q = this.queries.get(msg.id);
        this.queries.delete(msg.id);
        if (msg.ok) q?.resolve(msg.data);
        else q?.reject(new Error(msg.error ?? "Something went wrong."));
        return;
      }
      case "stream":
        for (const fn of this.subs.get(msg.sub) ?? []) fn(msg.data);
        return;
      case "error":
        for (const fn of this.errorListeners) fn(msg.reason);
        return;
    }
  }

  /** Signed in: restore what this client had asked for, then send what is still unanswered. */
  private welcomed(): void {
    this.live = true;
    this.attempt = 0;
    this.setStatus("live");
    for (const worldId of this.watched) this.send({ v: 1, type: "watch", worldId });
    for (const sub of this.subs.keys()) this.send({ v: 1, type: "subscribe", sub });
    for (const { msg } of this.pending.values()) this.send(msg); // same cid: applied at most once
    for (const { msg } of this.queries.values()) this.send(msg);
    this.firstWelcome?.resolve();
    this.firstWelcome = null;
  }
}

/** The socket for this page, talking to the server that served it. */
export function browserSocket(token: () => Promise<string>): GameSocket {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return new GameSocket({
    url: `${scheme}://${location.host}/ws`,
    open: (url) => new WebSocket(url) as unknown as WsLike,
    token,
    schedule: (fn, ms) => window.setTimeout(fn, ms),
    cancel: (h) => window.clearTimeout(h as number),
    uuid: () => crypto.randomUUID(),
  });
}
