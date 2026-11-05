import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

const INDEX = "<!doctype html><title>AI Race</title><div id=root></div>";

function builtClient(): string {
  const dir = mkdtempSync(join(tmpdir(), "airace-dist-"));
  temps.push(dir);
  writeFileSync(join(dir, "index.html"), INDEX);
  mkdirSync(join(dir, "assets"));
  writeFileSync(join(dir, "assets", "app.js"), "console.log('game')");
  return dir;
}

describe("the game served from the host's PC", () => {
  it("serves the built web app, its files and the join links, beside the API", async () => {
    server = await startTestServer({ clientDist: builtClient() });
    for (const path of ["/", "/join/ABC123", "/w/some-world"]) {
      const res = await fetch(`${server.url}${path}`);
      expect(res.status).toBe(200);
      expect(await res.text()).toBe(INDEX);
    }
    const js = await fetch(`${server.url}/assets/app.js`);
    expect(js.status).toBe(200);
    expect(js.headers.get("content-type")).toMatch(/javascript/);
    expect(((await (await fetch(`${server.url}/api/health`)).json()) as { ok: boolean }).ok).toBe(
      true,
    );
    expect((await fetch(`${server.url}/api/nothing-here`)).status).toBe(404);
    expect((await fetch(`${server.url}/assets/missing.js`)).status).toBe(404);
  });

  it("without a built web app, the API still runs", async () => {
    server = await startTestServer({ clientDist: null });
    expect((await fetch(`${server.url}/api/health`)).status).toBe(200);
    expect((await fetch(`${server.url}/`)).status).toBe(404);
  });
});
