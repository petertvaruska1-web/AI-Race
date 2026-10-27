import type { ClientMessage } from "@airace/protocol";
import { GameSocket, type WsLike } from "../src/net/socket.ts";

/** A WebSocket the test drives by hand. */
export class MockWs implements WsLike {
  readyState = 0;
  sent: ClientMessage[] = [];
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  send(data: string): void {
    this.sent.push(JSON.parse(data));
  }
  close(): void {
    this.drop();
  }
  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }
  receive(msg: unknown): void {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }
  drop(): void {
    this.readyState = 3;
    this.onclose?.({});
  }
  sentOf<T extends ClientMessage["type"]>(type: T): Extract<ClientMessage, { type: T }>[] {
    return this.sent.filter((m) => m.type === type) as Extract<ClientMessage, { type: T }>[];
  }
}

/** A GameSocket on mock sockets, with timers the test runs by hand. */
export function mockGame() {
  const sockets: MockWs[] = [];
  const timers: (() => void)[] = [];
  let n = 0;
  const socket = new GameSocket({
    url: "ws://test/ws",
    open: () => {
      const ws = new MockWs();
      sockets.push(ws);
      return ws;
    },
    token: async () => "tok",
    schedule: (fn) => {
      timers.push(fn);
      return timers.length;
    },
    cancel: () => {},
    uuid: () => `00000000-0000-4000-8000-${String(++n).padStart(12, "0")}`,
  });
  const flush = () => new Promise((r) => setTimeout(r, 0));
  return {
    socket,
    sockets,
    /** Opens the latest socket and welcomes it. */
    async welcome(userId = "u1") {
      await flush();
      const ws = sockets.at(-1);
      if (!ws) throw new Error("no socket yet");
      ws.open();
      ws.receive({ v: 1, type: "welcome", userId });
      await flush();
      return ws;
    },
    async runTimers() {
      for (const t of timers.splice(0)) t();
      await flush();
    },
    flush,
  };
}

/** A signed-in mock game: the socket is live and its first mock WebSocket is returned. */
export async function liveGame() {
  const g = mockGame();
  const connected = g.socket.connect();
  const ws = await g.welcome();
  await connected;
  return { ...g, ws };
}
