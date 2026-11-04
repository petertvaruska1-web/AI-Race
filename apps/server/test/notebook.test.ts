import { createHash, randomUUID } from "node:crypto";
import { Writable } from "node:stream";
import { TrainJobSpec } from "@airace/protocol";
import type { CompanyState } from "@airace/sim";
import { pino } from "pino";
import { afterEach, describe, expect, it } from "vitest";
import { notebookProblem } from "../src/notebook/filter.ts";
import { eventually } from "./fakeWorker.ts";
import { closeSockets, openSockets, plan, send, setup, start, worker } from "./flows.ts";
import { api, createGuest, type TestServer } from "./harness.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  closeSockets();
  await server?.close();
  server = null;
});

const SECRET = "My grandmother's secret recipe: two eggs and a blue moon.";
const sha = (t: string) => createHash("sha256").update(t).digest("hex");

type Owner = Awaited<ReturnType<typeof setup>>;
const companyOf = (o: Owner) => {
  const s = o.actor.state();
  return s.companies[s.members[o.userId]?.companyId ?? ""] as CompanyState;
};

async function save(ws: WsClient, worldId: string, text: string) {
  const cid = randomUUID();
  ws.send({ v: 1, type: "notebookSave", cid, worldId, text });
  return Promise.race([
    ws.next("ack", (m) => m.cid === cid).then(() => ({ ok: true as const })),
    ws.next("reject", (m) => m.cid === cid).then((m) => ({ ok: false as const, reason: m.reason })),
  ]);
}

async function read(ws: WsClient, worldId: string) {
  const id = randomUUID();
  ws.send({ v: 1, type: "query", id, q: { kind: "notebook", worldId } });
  return ws.next("queryResult", (m) => m.id === id);
}

describe("the notebook's filter", () => {
  it("refuses what a notebook can't hold, in plain words", () => {
    expect(notebookProblem("")).toBeNull();
    expect(notebookProblem("A short note about cats.\n\nAnd dogs.")).toBeNull();
    expect(notebookProblem(Array(5_001).fill("word").join(" "))).toBe(
      "Notebooks hold up to 5,000 words.",
    );
    expect(notebookProblem(Array(5_000).fill("word").join(" "))).toBeNull();
    for (const bad of [
      "see https://example.com now",
      "visit www.example.org",
      "mail me at ana@example.com",
      "call +1 555 123 4567",
      "ring (020) 7946-0958 today",
    ]) {
      expect(notebookProblem(bad), bad).toBe(
        "Notebooks can't hold links, email addresses or phone numbers.",
      );
    }
    expect(notebookProblem("Counting: 1 2 3 4 5 6 7. The years 2024-2025.")).toBeNull();
    expect(notebookProblem("hello <|end|> there")).toBe(
      "That text has characters the AI uses for itself.",
    );
    expect(notebookProblem("you are a FUCKING idiot")).toBe(
      "That notebook has words that aren't allowed.",
    );
  });
});

describe("the notebook service", () => {
  it("stores the text by its fingerprint, tells the world only that, and shows it to its owner", async () => {
    server = await start();
    const o = await setup(server);
    expect((await read(o.ws, o.worldId)).data).toEqual({ text: "" });
    expect(await save(o.ws, o.worldId, SECRET)).toEqual({ ok: true });
    await eventually(() => companyOf(o).notebook?.hash === sha(SECRET));
    expect(companyOf(o).notebook?.words).toBe(10);
    expect((await read(o.ws, o.worldId)).data).toEqual({ text: SECRET });
    expect(JSON.stringify(o.actor.state())).not.toContain("grandmother");
    expect(await save(o.ws, o.worldId, "see https://example.com")).toEqual({
      ok: false,
      reason: "Notebooks can't hold links, email addresses or phone numbers.",
    });

    const ben = await createGuest(server, "Ben");
    await o.actor.submit({
      kind: "system",
      event: { type: "memberJoined", userId: ben.userId, displayName: "Ben" },
    });
    const benWs = await connectAs(server, ben.cookie);
    openSockets.push(benWs);
    const theirs = await read(benWs, o.worldId);
    expect(JSON.stringify(theirs)).not.toContain("grandmother");
  });

  it("trains a run on exactly the text saved when it started, and never logs it", async () => {
    server = await start();
    const lines: string[] = [];
    const sink = new Writable({
      write(chunk, _enc, done) {
        lines.push(String(chunk));
        done();
      },
    });
    server.ctx.log = pino({ level: "trace" }, sink);
    const w = await worker(server);
    const o = await setup(server);
    await save(o.ws, o.worldId, `${SECRET}\n\nSecond paragraph.`);
    await eventually(() => companyOf(o).notebook !== null);
    await send(o.ws, o.worldId, {
      type: "startTraining",
      plan: { ...plan(), mix: { ...plan().mix, notebook: 10 } },
    });
    await save(o.ws, o.worldId, "Something new entirely.");
    const lease = await w.nextLease();
    const config = TrainJobSpec.parse(lease.spec).config;
    expect(config.notebook).toEqual([SECRET, "Second paragraph."]);
    expect((config.mixture as Record<string, number>).notebook).toBe(10);
    expect(lines.join("")).not.toContain("grandmother");
    expect(lines.join("")).not.toContain("Something new");
  });

  it("a run without a notebook share gets no notebook", async () => {
    server = await start();
    const w = await worker(server);
    const o = await setup(server);
    await save(o.ws, o.worldId, SECRET);
    await send(o.ws, o.worldId, { type: "startTraining", plan: plan() });
    const config = TrainJobSpec.parse((await w.nextLease()).spec).config;
    expect(config.notebook).toBeUndefined();
    expect((config.mixture as Record<string, number>).notebook).toBeUndefined();
  });
});

describe("goals at world creation", () => {
  it("a world can start with a goal, which its view shows", async () => {
    server = await start();
    const guest = await createGuest(server, "Ana");
    const res = await api(server, "/api/worlds", {
      body: {
        name: "Race to a million",
        goal: { goal: { kind: "customers", target: 1_000_000 }, deadlineYears: 2 },
      },
      cookie: guest.cookie,
    });
    expect(res.status).toBe(200);
    const { worldId } = (await res.json()) as { worldId: string };
    const actor = await server.ctx.worlds.get(worldId);
    expect(actor.state().goal.spec).toEqual({
      goal: { kind: "customers", target: 1_000_000 },
      deadlineYears: 2,
    });
  });

  it("a rival goal needs the rival AIs, and a bad goal is refused", async () => {
    server = await start();
    const guest = await createGuest(server, "Ana");
    const create = (goal: unknown) =>
      api(server as TestServer, "/api/worlds", {
        body: { name: "Goals", goal },
        cookie: guest.cookie,
      });
    const rival = await create({ goal: { kind: "beatArch" }, deadlineYears: null });
    expect(rival.status).toBe(400);
    expect(((await rival.json()) as { error: string }).error).toBe(
      "Rival goals need the rival AIs, which aren't built yet.",
    );
    expect(
      (await create({ goal: { kind: "customers", target: 5 }, deadlineYears: null })).status,
    ).toBe(400);
  });
});
