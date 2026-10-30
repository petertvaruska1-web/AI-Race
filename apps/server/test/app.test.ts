import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { eq, sql } from "drizzle-orm";
import { afterEach, describe, expect, it } from "vitest";
import { buildApp } from "../src/app.ts";
import { schema } from "../src/db/index.ts";
import { fakeScheduler } from "../src/scheduler.ts";
import { startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

describe("server scaffold", () => {
  it("answers the health check", async () => {
    server = await startTestServer();
    const res = await fetch(`${server.url}/api/health`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, sim: "sim-4" });
  });

  it("creates every table", async () => {
    server = await startTestServer();
    const rows = await server.ctx.db.execute<{ table_name: string }>(
      sql`select table_name from information_schema.tables where table_schema = 'public'`,
    );
    const names = rows.rows.map((r) => r.table_name);
    for (const table of schema.TABLE_NAMES) expect(names).toContain(table);
  });

  it("keeps data across a restart on the same directory", async () => {
    const dir = mkdtempSync(join(tmpdir(), "airace-db-"));
    const opts = {
      dbDir: join(dir, "db"),
      blobDir: join(dir, "blobs"),
      catalogPath: null,
      secret: "s",
      workerToken: "w",
      balance: "balance-test" as const,
      scheduler: fakeScheduler(),
      publicUrl: "http://127.0.0.1",
      logLevel: "warn",
    };
    const id = "0b9a7c1e-2f4d-4e8a-9c3b-5d6e7f8a9b0c";
    try {
      const first = await buildApp(opts);
      await first.ctx.db.insert(schema.users).values({ id, displayName: "Ana", createdAt: 1 });
      await first.close();

      const second = await buildApp(opts);
      const rows = await second.ctx.db.select().from(schema.users).where(eq(schema.users.id, id));
      await second.close();
      expect(rows.map((r) => r.displayName)).toEqual(["Ana"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("fake scheduler", () => {
  it("runs due timers in time order and lets a timer schedule the next", async () => {
    const s = fakeScheduler(0);
    const seen: number[] = [];
    const tick = () => {
      seen.push(s.now());
      if (seen.length < 3) s.setTimeout(tick, 100);
    };
    s.setTimeout(tick, 100);
    const cancelled = s.setTimeout(() => seen.push(-1), 150);
    s.clearTimeout(cancelled);
    await s.advance(1000);
    expect(seen).toEqual([100, 200, 300]);
    expect(s.now()).toBe(1000);
  });
});
