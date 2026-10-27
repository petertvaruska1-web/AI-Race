import { randomUUID } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { DATASET_IDS } from "@airace/protocol";
import { BALANCE_TEST } from "@airace/sim";
import { afterEach, describe, expect, it } from "vitest";
import { createGuestWorld, startTestServer, type TestServer } from "./harness.ts";
import { connectAs, type WsClient } from "./wsClient.ts";

const SAMPLE = fileURLToPath(
  new URL("../../../packages/protocol/fixtures/catalog.sample.json", import.meta.url),
);

let server: TestServer | null = null;
let ws: WsClient | null = null;
const temps: string[] = [];
afterEach(async () => {
  ws?.close();
  ws = null;
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

/** The sample catalog with its datasets shuffled, as an export might list them. */
function shuffledCatalog(): string {
  const dir = mkdtempSync(join(tmpdir(), "airace-catalog-"));
  temps.push(dir);
  const c = JSON.parse(readFileSync(SAMPLE, "utf8"));
  c.datasets.reverse();
  const path = join(dir, "catalog.json");
  writeFileSync(path, JSON.stringify(c));
  return path;
}

describe("catalog", () => {
  it("answers the catalog query with 8 datasets in order and no reference traits", async () => {
    server = await startTestServer({ catalogPath: shuffledCatalog() });
    const { cookie } = await createGuestWorld(server);
    ws = await connectAs(server, cookie);
    ws.send({ v: 1, type: "query", id: "c", q: { kind: "catalog" } });
    const r = await ws.next("queryResult");
    expect(r.ok).toBe(true);
    const data = r.data as { datasets: { id: string }[]; referenceTraits?: unknown };
    expect(data.datasets.map((d) => d.id)).toEqual([...DATASET_IDS]);
    expect(data.referenceTraits).toBeUndefined();
  });

  it("refuses training while no training data is installed", async () => {
    server = await startTestServer({ catalogPath: null });
    const { cookie, worldId } = await createGuestWorld(server);
    ws = await connectAs(server, cookie);
    ws.send({ v: 1, type: "query", id: "c", q: { kind: "catalog" } });
    expect(await ws.next("queryResult")).toMatchObject({
      ok: false,
      error: "The training data is not installed yet.",
    });
    const cid = randomUUID();
    const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
    ws.send({ v: 1, type: "command", cid, worldId, cmd: { type: "startTraining", plan } });
    expect((await ws.next("reject", (m) => m.cid === cid)).reason).toBe(
      "The training data is not installed yet.",
    );
  });

  it("fails loudly on a catalog file that doesn't parse", async () => {
    const dir = mkdtempSync(join(tmpdir(), "airace-catalog-"));
    temps.push(dir);
    const path = join(dir, "catalog.json");
    writeFileSync(path, JSON.stringify({ version: "catalog-v0" }));
    await expect(startTestServer({ catalogPath: path })).rejects.toThrow();
  });
});
