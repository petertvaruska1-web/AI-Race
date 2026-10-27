import { diff } from "@airace/protocol";
import { beforeEach, describe, expect, it } from "vitest";
import { bindWorldStore, useWorld } from "../src/store/world.ts";
import { liveGame } from "./mockSocket.ts";
import { apply, command, LOGO, PALETTE, testWorld, viewOf } from "./views.ts";

beforeEach(() => useWorld.getState().open("w1"));

describe("world store", () => {
  it("applies patches in order to give the server's view", async () => {
    const g = await liveGame();
    bindWorldStore(g.socket);
    let s = testWorld();
    const v0 = viewOf(s);
    g.ws.receive({ v: 1, type: "snapshot", worldId: "w1", version: s.seq, view: v0 });
    expect(useWorld.getState().view).toEqual(v0);

    const s0 = s;
    s = apply(
      s,
      command("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }, "c1"),
    );
    g.ws.receive({
      v: 1,
      type: "patch",
      worldId: "w1",
      from: s0.seq,
      to: s.seq,
      ops: diff(v0, viewOf(s)),
    });
    const s1 = s;
    s = apply(s, command("u1", { type: "startWorld" }));
    g.ws.receive({
      v: 1,
      type: "patch",
      worldId: "w1",
      from: s1.seq,
      to: s.seq,
      ops: diff(viewOf(s1), viewOf(s)),
    });

    expect(useWorld.getState().version).toBe(s.seq);
    expect(useWorld.getState().view).toEqual(viewOf(s));
    expect(g.ws.sentOf("resync")).toEqual([]);
  });

  it("asks for a fresh snapshot when a patch was missed", async () => {
    const g = await liveGame();
    bindWorldStore(g.socket);
    g.ws.receive({ v: 1, type: "snapshot", worldId: "w1", version: 5, view: viewOf(testWorld()) });
    g.ws.receive({
      v: 1,
      type: "patch",
      worldId: "w1",
      from: 7,
      to: 8,
      ops: [{ op: "set", path: ["world", "day"], value: 3 }],
    });
    expect(g.ws.sentOf("resync")).toEqual([{ v: 1, type: "resync", worldId: "w1" }]);
    expect(useWorld.getState().version).toBe(5);
    expect(useWorld.getState().view?.world.day).toBe(0);
  });

  it("ignores messages about another world", () => {
    useWorld.getState().onSnapshot({
      v: 1,
      type: "snapshot",
      worldId: "other",
      version: 1,
      view: viewOf(testWorld()),
    });
    expect(useWorld.getState().view).toBeNull();
  });
});
