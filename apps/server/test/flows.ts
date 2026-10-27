import { randomUUID } from "node:crypto";
import { type TrainingPlan, TrainJobSpec } from "@airace/protocol";
import { BALANCE_TEST, type RunState } from "@airace/sim";
import type { WorldActor } from "../src/worlds/actor.ts";
import { eventually, FakeWorker } from "./fakeWorker.ts";
import { createGuestWorld, SAMPLE_CATALOG, startTestServer, type TestServer } from "./harness.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

// Steps shared by the training and chat tests: a running world with a company,
// commands that must be accepted, and moving game time.

/** Sockets opened by these helpers; `closeSockets` in afterEach closes them. */
export const openSockets: { close(): void }[] = [];

export function closeSockets(): void {
  for (const s of openSockets.splice(0)) s.close();
}

export const DAY_MS = BALANCE_TEST.tickMsAt1x;
export const RESULT = {
  steps: 8,
  final_loss: 3.9,
  heldout_losses: { web: 4.1 },
  status: "completed",
};
export const plan = (): TrainingPlan => ({
  ...BALANCE_TEST.defaultPlan,
  mix: { ...BALANCE_TEST.defaultPlan.mix },
});

export const start = (over = {}) => startTestServer({ catalogPath: SAMPLE_CATALOG, ...over });

export async function worker(s: TestServer, o: Parameters<typeof FakeWorker.connect>[1] = {}) {
  const w = await FakeWorker.connect(s, { workerId: "w", ...o });
  openSockets.push(w);
  return w;
}

export async function owner(s: TestServer, cookie: string, worldId: string): Promise<WsClient> {
  const ws = await connectAs(s, cookie);
  openSockets.push(ws);
  ws.send({ v: 1, type: "watch", worldId });
  await ws.next("snapshot");
  return ws;
}

export async function send(ws: WsClient, worldId: string, cmd: unknown): Promise<void> {
  const cid = randomUUID();
  ws.send({ v: 1, type: "command", cid, worldId, cmd });
  const outcome = await Promise.race([
    ws.next("ack", (m) => m.cid === cid).then(() => null),
    ws.next("reject", (m) => m.cid === cid).then((m) => m.reason),
  ]);
  if (outcome) throw new Error(`rejected: ${outcome}`);
}

/** A running world whose owner has a company and an open socket. */
export async function setup(s: TestServer) {
  const o = await createGuestWorld(s);
  const ws = await owner(s, o.cookie, o.worldId);
  await send(ws, o.worldId, {
    type: "createCompany",
    name: "Acme",
    logo: { shape: "circle", symbol: "spark", initials: "AC" },
    palette: { primary: "#4f8cff", secondary: "#ffb020" },
  });
  await send(ws, o.worldId, { type: "startWorld" });
  const actor = await s.ctx.worlds.get(o.worldId);
  return { ...o, ws, actor };
}

export function runOf(actor: WorldActor, userId: string): RunState {
  const s = actor.state();
  const company = s.companies[s.members[userId]?.companyId ?? ""];
  const run = Object.values(company?.runs ?? {})[0];
  if (!run) throw new Error("no run");
  return run;
}

export function cashOf(actor: WorldActor, userId: string): number {
  const s = actor.state();
  return s.companies[s.members[userId]?.companyId ?? ""]?.cash ?? -1;
}

export async function toDay(s: TestServer, actor: WorldActor, day: number): Promise<void> {
  await s.scheduler.advance((day - actor.state().day) * DAY_MS);
}

/** Trains the owner's next version to completion with fake worker `w`; returns its id. */
export async function trainVersion(
  s: TestServer,
  w: FakeWorker,
  o: { ws: WsClient; worldId: string; userId: string; actor: WorldActor },
): Promise<string> {
  await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
  const lease = await w.nextLease();
  const spec = TrainJobSpec.parse(lease.spec);
  await fetch(spec.outTarUrl, { method: "PUT", body: "fake-checkpoint" });
  w.done(lease.jobId, RESULT);
  const run = runOf(o.actor, o.userId);
  await toDay(s, o.actor, run.endDay);
  const done = await eventually(() => {
    const r = runOf(o.actor, o.userId);
    return r.status === "completed" ? r : null;
  });
  if (!done.versionId) throw new Error("no version");
  return done.versionId;
}
