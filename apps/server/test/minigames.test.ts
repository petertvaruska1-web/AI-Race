import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import { MinigameJobSpec } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { afterEach, describe, expect, it } from "vitest";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import { closeSockets, send, setup, start, toDay, trainVersion, worker } from "./flows.ts";
import type { TestServer } from "./harness.ts";
import { writeFixturePack } from "./rivalPack.ts";
import type { WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

const EVAL = {
  report: {
    suite: "bench-v1",
    scores: { knowledge: { score: 50, raw: 0.5, n: 40 }, pattern: { score: 60, raw: 0.6, n: 40 } },
    overall: 40,
    missing: [],
    items: [],
    seconds: 1,
  },
  traits: { verbosity: 0.4 },
  personalityWords: [],
};

const STEP = (attempt: number, correct: boolean) => ({
  attempt,
  depth: attempt,
  retry: 0,
  itemId: `bridge-${attempt}-0`,
  band: 0,
  typed: false,
  prompt: "Next: red, blue, red, blue,",
  display: { kind: "bridge", family: "cycle", terms: [] },
  options: ["blue", "red"],
  answerIndex: 1,
  choice: correct ? 1 : 0,
  sure: [0.3, 0.7],
  reply: null,
  points: null,
  ruleMet: null,
  correct,
  lives: 3,
});
const SIDE = (score: number) => ({
  steps: Array.from({ length: score }, (_, k) => STEP(k, true)),
  score,
  maxScore: 12,
});

function startWithPack() {
  const pack = writeFixturePack();
  temps.push(pack.dir);
  return start({ rivalsDir: pack.dir });
}

const mine = (o: { actor: Awaited<ReturnType<typeof setup>>["actor"]; userId: string }) => {
  const s = o.actor.state();
  return s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
};

/** A player with a trained, scored AI; `launch` also puts it on sale. */
async function ready(s: TestServer, launch = false) {
  const w: FakeWorker = await worker(s, { jobTypes: ["train", "eval", "minigame"] });
  const o = await setup(s);
  const versionId = await trainVersion(s, w, o);
  const evalLease = await w.nextLease();
  w.done(evalLease.jobId, EVAL);
  await eventually(() => mine(o).versions[versionId]?.scores);
  if (launch) {
    await send(o.ws, o.worldId, {
      type: "launchProduct",
      versionId,
      name: "Spark",
      plans: ["free"],
      countdownDays: 3,
    });
    await toDay(s, o.actor, o.actor.state().day + 4);
  }
  return { ...o, w, versionId };
}

async function play(
  ws: WsClient,
  msg: Record<string, unknown>,
): Promise<{ type: "ack"; runId: string } | { type: "reject"; reason: string }> {
  const cid = randomUUID();
  ws.send({ v: 1, type: "minigame", cid, settings: {}, opponent: null, ...msg });
  return Promise.race([
    ws
      .next("ack", (m) => m.cid === cid)
      .then((m) => ({
        type: "ack" as const,
        runId: (m.data as { runId: string }).runId,
      })),
    ws
      .next("reject", (m) => m.cid === cid)
      .then((m) => ({ type: "reject" as const, reason: m.reason })),
  ]);
}

async function query(ws: WsClient, q: unknown) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q });
  return ws.next("queryResult", (m) => m.id === id);
}

describe("Playground minigames", () => {
  it("play a practice run as a job, stream its steps and keep the personal best", async () => {
    server = await start();
    const o = await ready(server);
    const ack = await play(o.ws, {
      worldId: o.worldId,
      game: "bridge",
      difficulty: "easy",
      versionId: o.versionId,
    });
    if (ack.type !== "ack") throw new Error(ack.reason);
    const runId = ack.runId;
    const lease = await o.w.nextLease();
    expect(lease.jobType).toBe("minigame");
    const spec = MinigameJobSpec.parse(lease.spec);
    expect(spec).toMatchObject({
      runId,
      game: "bridge",
      difficulty: "easy",
      sides: [{ side: "a", versionId: o.versionId }],
    });
    expect(await (await fetch(spec.sides[0]?.modelTarUrl ?? "")).text()).toBe("fake-checkpoint");

    o.ws.send({ v: 1, type: "subscribe", sub: `minigame:${o.worldId}:${runId}` });
    expect((await query(o.ws, { kind: "ping" })).ok).toBe(true);
    o.w.progress(lease.jobId, { kind: "minigameStep", side: "a", ...STEP(0, true) });
    const step = await o.ws.next("stream");
    expect(step).toMatchObject({
      data: { kind: "step", side: "a", step: { attempt: 0, correct: true } },
    });
    o.w.done(lease.jobId, { sides: { a: SIDE(7) } });
    await o.ws.next("stream", (m) => (m as { data: { kind: string } }).data.kind === "done");

    const runs = (await query(o.ws, { kind: "minigameRuns", worldId: o.worldId })) as {
      data: {
        runs: { id: string; score: number; status: string }[];
        bests: Record<string, number>;
      };
    };
    expect(runs.data.runs[0]).toMatchObject({ id: runId, score: 7, status: "done" });
    expect(runs.data.bests["bridge:easy"]).toBe(7);
  });

  it("refuses a second run while one is going, and an AI that isn't yours", async () => {
    server = await start();
    const o = await ready(server);
    const base = { worldId: o.worldId, game: "maze", difficulty: "easy" };
    expect((await play(o.ws, { ...base, versionId: randomUUID() })).type).toBe("reject");
    expect((await play(o.ws, { ...base, versionId: o.versionId })).type).toBe("ack");
    const again = await play(o.ws, { ...base, versionId: o.versionId });
    expect(again).toMatchObject({ type: "reject", reason: "Your last game is still going." });
  });

  it("duels a rival's AI on sale, loaded from the rival pack", async () => {
    server = await startWithPack();
    const o = await ready(server, true);
    await eventually(() => o.actor.state().rivals);
    await toDay(server, o.actor, o.actor.state().day + 30);
    const rival = Object.values(o.actor.state().companies).find(
      (c) => c.rival && Object.values(c.products).some((p) => p.status === "live"),
    );
    const target = rival && Object.values(rival.products).find((p) => p.status === "live");
    if (!target) throw new Error("no rival AI on sale");
    const ack = await play(o.ws, {
      worldId: o.worldId,
      game: "quiz",
      difficulty: "medium",
      versionId: o.versionId,
      opponent: target.id,
    });
    expect(ack.type).toBe("ack");
    const lease = await o.w.nextLease();
    const spec = MinigameJobSpec.parse(lease.spec);
    const b = spec.sides.find((x) => x.side === "b");
    expect(b?.versionId).toBe(rival?.versions[target.versionId]?.packKey);
    expect(await (await fetch(b?.modelTarUrl ?? "")).text()).toBe(`fake-rival-${b?.versionId}`);
  });

  it("plays a showcase the world booked, and the result moves the world", async () => {
    server = await start();
    const o = await ready(server, true);
    await send(o.ws, o.worldId, {
      type: "startShowcase",
      game: "bridge",
      difficulty: "easy",
      versionId: o.versionId,
      settings: {},
    });
    const lease = await o.w.nextLease();
    expect(lease.jobType).toBe("minigame");
    o.w.done(lease.jobId, { sides: { a: SIDE(12) } });
    await eventually(() => mine(o).showcases.find((x) => x.status === "done"));
    expect(mine(o).showcases[0]).toMatchObject({ score: 12, maxScore: 12 });
  });

  it("calls a showcase off when its job fails for good, refunding the fee", async () => {
    server = await start();
    const o = await ready(server, true);
    const cash = mine(o).cash;
    await send(o.ws, o.worldId, {
      type: "startShowcase",
      game: "bridge",
      difficulty: "easy",
      versionId: o.versionId,
      settings: {},
    });
    const lease = await o.w.nextLease();
    o.w.failed(lease.jobId, "the model is gone", false);
    await eventually(() => mine(o).showcases.find((x) => x.status === "calledOff"));
    expect(mine(o).cash).toBe(cash);
  });
});
