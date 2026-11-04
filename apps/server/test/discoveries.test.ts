import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EvalJobSpec, TrainJobSpec } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { afterEach, describe, expect, it } from "vitest";
import { eventually, type FakeWorker } from "./fakeWorker.ts";
import {
  closeSockets,
  owner,
  plan,
  RESULT,
  runOf,
  send,
  setup,
  start,
  toDay,
  trainVersion,
  worker,
} from "./flows.ts";
import type { TestServer } from "./harness.ts";
import { settle } from "./wsClient.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

const BREAKDOWN = { "knowledge/topic:animals": [72, 21] as [number, number] };
const FINDING = {
  key: "knowledge/topic:animals",
  kind: "strength",
  about: "facts about animals",
  title: "Surprisingly good at facts about animals",
  category: "knowledge",
  tag: "topic:animals",
  value: 72,
  compared: 30,
  evidence: { prompt: "Which group does the newt belong to?", reply: " amphibians <|end|>" },
};

function evalResult(discoveries: unknown[] = [FINDING]) {
  return {
    report: {
      suite: "bench-v1",
      scores: { language: { score: 41.5, raw: 0.415, n: 40 } },
      overall: 41.5,
      missing: ["creativity"],
      items: [],
      seconds: 1,
    },
    traits: { verbosity: 0.4 },
    personalityWords: ["talkative"],
    signals: { legibleReplies: null, correctFacts: 2, passingPrograms: 0 },
    breakdown: BREAKDOWN,
    discoveries,
  };
}

type Owner = Awaited<ReturnType<typeof setup>>;
const companyOf = (o: Owner) => {
  const s = o.actor.state();
  return s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
};

async function scoredVersion(s: TestServer, w: FakeWorker, o: Owner) {
  const versionId = await trainVersion(s, w, o);
  const lease = await w.nextLease();
  expect(lease.jobType).toBe("eval");
  return { versionId, lease, spec: EvalJobSpec.parse(lease.spec) };
}

describe("evals bring signals and discoveries into the world", () => {
  it("compare a version with its parent and the world's AIs, and earn milestones", async () => {
    server = await start();
    const w = await worker(server);
    const o = await setup(server);
    const first = await scoredVersion(server, w, o);
    expect(first.spec.previous).toBeNull();
    expect(first.spec.peers).toEqual([]);
    // The test balance's thresholds, so a first eval can already find something.
    expect(first.spec.thresholds).toEqual({ minPoints: 1, minZ: 0, minPeers: 1, perEval: 3 });
    w.done(first.lease.jobId, evalResult());
    await eventually(() => companyOf(o).discoveries.length === 1);
    expect(companyOf(o).discoveries[0]).toMatchObject({
      id: `${first.versionId}-d0`,
      evidence: FINDING.evidence,
    });
    expect(companyOf(o).milestones.map((m) => m.id)).toEqual(["firstFact"]);

    // A version grown from the first is compared with it.
    await send(o.ws, o.worldId, {
      type: "startTraining",
      plan: { ...plan(), parentVersionId: first.versionId },
    });
    const train = await w.nextLease();
    await fetch(TrainJobSpec.parse(train.spec).outTarUrl, { method: "PUT", body: "x" });
    w.done(train.jobId, RESULT);
    const growing = Object.values(companyOf(o).runs).find((r) => r.status === "training");
    await toDay(server, o.actor, growing?.endDay ?? 0);
    const second = await w.nextLease();
    expect(second.jobType).toBe("eval");
    const spec = EvalJobSpec.parse(second.spec);
    expect(spec.previous).toEqual({
      label: "v1",
      breakdown: BREAKDOWN,
      traits: { verbosity: 0.4 },
    });
    expect(spec.peers).toEqual([{ breakdown: BREAKDOWN, traits: { verbosity: 0.4 } }]);
  });

  it("an eval with nothing found adds no discoveries", async () => {
    server = await start();
    const w = await worker(server);
    const o = await setup(server);
    const first = await scoredVersion(server, w, o);
    w.done(first.lease.jobId, evalResult([]));
    await eventually(() => companyOf(o).versions[first.versionId]?.scores);
    await settle(50);
    expect(companyOf(o).discoveries).toEqual([]);
  });

  it("findings arrive once, even when the server restarts before the world saw them", async () => {
    const dir = mkdtempSync(join(tmpdir(), "airace-discoveries-"));
    temps.push(dir);
    const over = { dbDir: join(dir, "db"), blobDir: join(dir, "blobs") };
    server = await start(over);
    const w = await worker(server);
    const o = await setup(server);
    const first = await scoredVersion(server, w, o);
    w.done(first.lease.jobId, evalResult());
    await eventually(() => companyOf(o).discoveries.length === 1);
    closeSockets();
    await server.close();

    server = await start(over);
    await worker(server);
    await owner(server, o.cookie, o.worldId);
    const actor = await server.ctx.worlds.get(o.worldId);
    await settle(200);
    const s = actor.state();
    const c = s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
    expect(c.discoveries).toHaveLength(1);
    expect(c.milestones).toHaveLength(1);
  });
});
