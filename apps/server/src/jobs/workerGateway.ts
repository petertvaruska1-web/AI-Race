import { timingSafeEqual } from "node:crypto";
import { type ServerToWorker, WorkerMessage } from "@airace/protocol";
import type { FastifyInstance } from "fastify";
import type { RawData, WebSocket } from "ws";
import type { AppContext } from "../app.ts";
import type { WorkerConn } from "./broker.ts";

const HELLO_TIMEOUT_MS = 5000;
export const CLOSE_BAD_WORKER = 4401;

function sameSecret(a: string, b: string): boolean {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

/** `/worker`: ML workers dial in, prove the shared token, then lease jobs from the broker. */
export function registerWorkerRoutes(app: FastifyInstance, ctx: AppContext): void {
  app.get("/worker", { websocket: true }, (socket: WebSocket) => {
    let conn: WorkerConn | null = null;
    let queue: Promise<void> = Promise.resolve();
    const sched = ctx.opts.scheduler;
    const hello = sched.setTimeout(() => {
      if (!conn) socket.close(CLOSE_BAD_WORKER, "say whello first");
    }, HELLO_TIMEOUT_MS);
    const send = (msg: ServerToWorker) => {
      if (socket.readyState === socket.OPEN) socket.send(JSON.stringify(msg));
    };

    const onMessage = async (raw: RawData): Promise<void> => {
      let data: unknown = null;
      try {
        data = JSON.parse(raw.toString());
      } catch {
        // invalid, handled below
      }
      const parsed = WorkerMessage.safeParse(data);
      if (!conn) {
        if (
          !parsed.success ||
          parsed.data.type !== "whello" ||
          !sameSecret(parsed.data.token, ctx.opts.workerToken)
        ) {
          socket.close(CLOSE_BAD_WORKER, "unknown worker");
          return;
        }
        sched.clearTimeout(hello);
        conn = {
          workerId: parsed.data.workerId,
          caps: parsed.data.capabilities,
          send,
          close: (code, reason) => socket.close(code, reason),
          leases: new Map(),
        };
        ctx.log.info({ workerId: conn.workerId, caps: conn.caps }, "worker connected");
        send({ v: 1, type: "wwelcome" });
        await ctx.broker.attach(conn, parsed.data.running);
        return;
      }
      if (!parsed.success) {
        ctx.log.warn(
          { workerId: conn.workerId, issues: parsed.error.issues },
          "bad worker message",
        );
        return;
      }
      await ctx.broker.handle(conn, parsed.data);
    };

    socket.on("message", (raw: RawData) => {
      queue = queue
        .then(() => onMessage(raw))
        .catch((err) => ctx.log.error({ err }, "worker message failed"));
    });
    socket.on("close", () => {
      sched.clearTimeout(hello);
      if (conn) {
        ctx.log.info({ workerId: conn.workerId }, "worker disconnected");
        ctx.broker.detach(conn);
      }
    });
  });
}
