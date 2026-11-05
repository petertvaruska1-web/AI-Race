import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { api, createGuest, SAMPLE_CATALOG, startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
const temps: string[] = [];
afterEach(async () => {
  await server?.close();
  server = null;
  for (const t of temps.splice(0)) rmSync(t, { recursive: true, force: true });
});

/** A data root with a little of everything a helper needs, plus things it must never get. */
function dataRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "airace-data-"));
  temps.push(root);
  const files: Record<string, string> = {
    "corpus/web/tokens.bin": "0123456789abcdef",
    "corpus/web/info.json": '{"docs": 2}',
    "tokenizer/tok-v1.json": '{"vocab": 4096}',
    "judge/v1/model.safetensors": "judge-weights",
    "novelty/v1/index.bin": "novelty",
    "rivals/v1/pack.json": "{}",
    "helper/ml.zip": "PK-fake-zip",
    "server/secret": "never-shared",
    "server/worker-token": "never-shared",
  };
  for (const [path, body] of Object.entries(files)) {
    mkdirSync(dirname(join(root, path)), { recursive: true });
    writeFileSync(join(root, path), body);
  }
  return root;
}

async function keyOf(s: TestServer, cookie: string): Promise<string> {
  const res = await api(s, "/api/helper/key", { cookie });
  return ((await res.json()) as { key: string }).key;
}

const sha = (b: Buffer | string) => createHash("sha256").update(b).digest("hex");

describe("downloading the helper from the game", () => {
  it("gives a signed-in player their own setup file, and nobody else one", async () => {
    server = await startTestServer({ dataRoot: dataRoot(), catalogPath: SAMPLE_CATALOG });
    expect((await fetch(`${server.url}/helper/setup.cmd`)).status).toBe(401);
    const ana = await createGuest(server, "Ana");
    const res = await api(server, "/helper/setup.cmd", { cookie: ana.cookie });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-disposition")).toMatch(/AI Race Helper setup\.cmd/);
    const text = await res.text();
    expect(text).toContain(await keyOf(server, ana.cookie));
    expect(text).toContain(server.ctx.opts.publicUrl);
    expect(text).toContain("/helper/setup.ps1");
    expect(text.split("\r\n").every((line) => line.length < 8000)).toBe(true); // cmd.exe's limit
    const script = await fetch(`${server.url}/helper/setup.ps1`, {
      headers: { authorization: `Bearer ${await keyOf(server, ana.cookie)}` },
    });
    expect(await script.text()).toContain("airace-helper");
    expect(text.split("\r\n").length).toBeGreaterThan(3); // Windows line ends
  });

  it("lists the game's data with checksums and serves it, ranges included, to a helper key", async () => {
    const root = dataRoot();
    server = await startTestServer({ dataRoot: root, catalogPath: SAMPLE_CATALOG });
    expect((await fetch(`${server.url}/helper/data/manifest.json`)).status).toBe(401);
    const ana = await createGuest(server, "Ana");
    const auth = { authorization: `Bearer ${await keyOf(server, ana.cookie)}` };

    const manifest = (await (
      await fetch(`${server.url}/helper/data/manifest.json`, { headers: auth })
    ).json()) as { dataIdentity: string; files: { path: string; size: number; sha256: string }[] };
    expect(manifest.dataIdentity).toBe("sample-0001");
    expect(manifest.files.map((f) => f.path)).toEqual([
      "corpus/web/info.json",
      "corpus/web/tokens.bin",
      "judge/v1/model.safetensors",
      "novelty/v1/index.bin",
      "rivals/v1/pack.json",
      "tokenizer/tok-v1.json",
    ]);
    for (const f of manifest.files) {
      const bytes = readFileSync(join(root, f.path));
      expect([f.size, f.sha256]).toEqual([bytes.length, sha(bytes)]);
    }

    const part = await fetch(`${server.url}/helper/data/corpus/web/tokens.bin`, {
      headers: { ...auth, range: "bytes=4-9" },
    });
    expect(part.status).toBe(206);
    expect(await part.text()).toBe("456789");
    const zip = await fetch(`${server.url}/helper/ml.zip`, { headers: auth });
    expect(await zip.text()).toBe("PK-fake-zip");

    for (const path of [
      "server/secret",
      "server/worker-token",
      "helper/ml.zip",
      "..%2Fserver%2Fsecret",
    ]) {
      const res = await fetch(`${server.url}/helper/data/${path}`, { headers: auth });
      expect(res.status, path).toBe(404);
    }
    const wrong = { authorization: "Bearer not-a-key" };
    expect((await fetch(`${server.url}/helper/ml.zip`, { headers: wrong })).status).toBe(401);
  });
});
