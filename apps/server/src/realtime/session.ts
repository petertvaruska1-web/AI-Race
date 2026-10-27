import type { ServerMessage } from "@airace/protocol";
import type { WebSocket } from "ws";
import type { WorldWatch } from "./broadcast.ts";

/** One open WebSocket: who it is, the worlds it watches and the streams it follows. */
export class Session {
  userId: string | null = null;
  readonly watches = new Map<string, WorldWatch>();
  readonly subs = new Set<string>();
  /** Messages are handled one at a time, in arrival order. */
  queue: Promise<void> = Promise.resolve();

  constructor(readonly socket: WebSocket) {}

  send(msg: ServerMessage): void {
    if (this.socket.readyState === this.socket.OPEN) this.socket.send(JSON.stringify(msg));
  }
}
