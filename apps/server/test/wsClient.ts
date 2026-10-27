import type { ServerMessage } from "@airace/protocol";
import WebSocket from "ws";
import { api, type TestServer } from "./harness.ts";

type Of<T extends ServerMessage["type"]> = Extract<ServerMessage, { type: T }>;

/** A typed WebSocket client that buffers messages and waits for the ones a test expects. */
export class WsClient {
  private inbox: ServerMessage[] = [];
  private waiters: (() => void)[] = [];
  readonly closed: Promise<{ code: number; reason: string }>;

  private constructor(private readonly ws: WebSocket) {
    ws.on("message", (raw) => {
      this.inbox.push(JSON.parse(raw.toString()) as ServerMessage);
      for (const wake of this.waiters.splice(0)) wake();
    });
    this.closed = new Promise((resolve) => {
      ws.on("close", (code, reason) => resolve({ code, reason: reason.toString() }));
    });
  }

  static open(url: string): Promise<WsClient> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      const client = new WsClient(ws);
      ws.once("open", () => resolve(client));
      ws.once("error", reject);
    });
  }

  send(msg: unknown): void {
    this.ws.send(typeof msg === "string" ? msg : JSON.stringify(msg));
  }

  /** The first unread message of `type` matching `pred`, waiting up to `timeoutMs` of real time. */
  async next<T extends ServerMessage["type"]>(
    type: T,
    pred: (m: Of<T>) => boolean = () => true,
    timeoutMs = 5000,
  ): Promise<Of<T>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const i = this.inbox.findIndex((m) => m.type === type && pred(m as Of<T>));
      if (i >= 0) return this.inbox.splice(i, 1)[0] as Of<T>;
      const left = deadline - Date.now();
      if (left <= 0) throw new Error(`no ${type} message within ${timeoutMs} ms`);
      await new Promise<void>((resolve) => {
        const t = setTimeout(resolve, left);
        this.waiters.push(() => {
          clearTimeout(t);
          resolve();
        });
      });
    }
  }

  /** Unread messages of `type` (for asserting that nothing arrived). */
  unread(type: ServerMessage["type"]): ServerMessage[] {
    return this.inbox.filter((m) => m.type === type);
  }

  close(): void {
    this.ws.close();
  }
}

/** Opens `/ws` as the guest with `cookie` and completes the hello. */
export async function connectAs(s: TestServer, cookie: string): Promise<WsClient> {
  const me = (await (await api(s, "/api/me", { cookie })).json()) as { wsToken: string };
  const client = await WsClient.open(`${s.wsUrl}/ws`);
  client.send({ v: 1, type: "hello", token: me.wsToken });
  await client.next("welcome");
  return client;
}

/** Lets in-flight socket messages reach the server. */
export function settle(ms = 30): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
