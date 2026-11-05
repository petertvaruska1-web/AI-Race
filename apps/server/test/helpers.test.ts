import { afterEach, describe, expect, it } from "vitest";
import { schema } from "../src/db/index.ts";
import { eventually, FakeWorker } from "./fakeWorker.ts";
import { api, createGuest, startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

async function keyOf(s: TestServer, cookie: string, rotate = false): Promise<string> {
  const res = rotate
    ? await api(s, "/api/helper/key/rotate", { method: "POST", cookie, body: {} })
    : await api(s, "/api/helper/key", { cookie });
  expect(res.status).toBe(200);
  return ((await res.json()) as { key: string }).key;
}

const owners = (s: TestServer) => s.ctx.broker.connections().map((c) => c.ownerUserId);

describe("helper keys", () => {
  it("are personal, stable, and only work until rotated", async () => {
    server = await startTestServer();
    expect((await api(server, "/api/helper/key")).status).toBe(401);
    const ana = await createGuest(server, "Ana");
    const key = await keyOf(server, ana.cookie);
    expect(await keyOf(server, ana.cookie)).toBe(key);

    const w = await FakeWorker.connect(server, { token: key, workerId: "ana-pc" });
    await eventually(() => owners(server as TestServer).includes(ana.userId));
    w.close();

    const fresh = await keyOf(server, ana.cookie, true);
    expect(fresh).not.toBe(key);
    const old = await FakeWorker.connect(server, { token: key, workerId: "old" });
    expect(await old.closed).toBe(4401);
    const again = await FakeWorker.connect(server, { token: fresh, workerId: "ana-pc-2" });
    await eventually(() => owners(server as TestServer).includes(ana.userId));
    again.close();
  });

  it("the server's own token still connects the house card", async () => {
    server = await startTestServer();
    await FakeWorker.connect(server);
    await eventually(() => owners(server as TestServer).includes(null));
  });

  it("keys are never stored", async () => {
    server = await startTestServer();
    const ana = await createGuest(server, "Ana");
    const key = await keyOf(server, ana.cookie);
    const rows = await server.ctx.db.select().from(schema.helperKeys);
    expect(JSON.stringify(rows)).not.toContain(key.split(".")[1]);
  });
});
