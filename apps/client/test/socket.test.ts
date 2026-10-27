import { describe, expect, it } from "vitest";
import { liveGame, mockGame } from "./mockSocket.ts";

describe("game socket", () => {
  it("signs in with a ws token", async () => {
    const g = mockGame();
    const connected = g.socket.connect();
    const ws = await g.welcome();
    await connected;
    expect(ws.sent[0]).toEqual({ v: 1, type: "hello", token: "tok" });
    expect(g.socket.status).toBe("live");
  });

  it("resends an unanswered command once, with the same id, after a reconnect", async () => {
    const g = await liveGame();
    g.socket.watch("w1");
    const result = g.socket.command("w1", { type: "setSpeed", speed: 2 });
    const [sent] = g.ws.sentOf("command");
    expect(sent?.cmd).toEqual({ type: "setSpeed", speed: 2 });

    g.ws.drop();
    expect(g.socket.status).toBe("offline");
    await g.runTimers();
    const second = await g.welcome();
    expect(second.sentOf("command").map((m) => m.cid)).toEqual([sent?.cid]);
    expect(second.sentOf("watch").map((m) => m.worldId)).toEqual(["w1"]);

    second.receive({ v: 1, type: "ack", cid: sent?.cid, duplicate: true });
    expect(await result).toEqual({ ok: true, duplicate: true, data: undefined });
    // Answered: a later reconnect does not send it again.
    second.drop();
    await g.runTimers();
    const third = await g.welcome();
    expect(third.sentOf("command")).toEqual([]);
  });

  it("keeps streams across reconnects and answers queries", async () => {
    const g = await liveGame();
    const seen: unknown[] = [];
    const off = g.socket.subscribe("training:w1:r1", (d) => seen.push(d));
    g.ws.receive({ v: 1, type: "stream", sub: "training:w1:r1", data: { n: 1 } });
    g.ws.drop();
    await g.runTimers();
    const second = await g.welcome();
    expect(second.sentOf("subscribe").map((m) => m.sub)).toEqual(["training:w1:r1"]);
    second.receive({ v: 1, type: "stream", sub: "training:w1:r1", data: { n: 2 } });
    expect(seen).toEqual([{ n: 1 }, { n: 2 }]);
    off();
    expect(second.sentOf("unsubscribe").map((m) => m.sub)).toEqual(["training:w1:r1"]);

    const pong = g.socket.query({ kind: "ping" });
    const [q] = second.sentOf("query");
    second.receive({ v: 1, type: "queryResult", id: q?.id, ok: true, data: { pong: true } });
    expect(await pong).toEqual({ pong: true });
    const missing = g.socket.query({ kind: "catalog" });
    const [, q2] = second.sentOf("query");
    second.receive({
      v: 1,
      type: "queryResult",
      id: q2?.id,
      ok: false,
      error: "The training data is not installed yet.",
    });
    await expect(missing).rejects.toThrow(/not installed/);
  });

  it("waits longer between each failed reconnect, up to 10 s", async () => {
    const delays: number[] = [];
    const g = mockGame();
    // Patch the scheduler to record the delays it is asked for.
    const socket = g.socket as unknown as {
      deps: { schedule: (fn: () => void, ms: number) => unknown };
    };
    const schedule = socket.deps.schedule;
    socket.deps.schedule = (fn, ms) => {
      delays.push(ms);
      return schedule(fn, ms);
    };
    void g.socket.connect();
    for (let i = 0; i < 7; i++) {
      await g.flush();
      g.sockets.at(-1)?.drop();
      await g.runTimers();
    }
    expect(delays).toEqual([500, 1000, 2000, 4000, 8000, 10_000, 10_000]);
  });
});
