import { randomBytes } from "node:crypto";
import { readdirSync } from "node:fs";
import { request } from "node:http";
import { join } from "node:path";
import { Readable } from "node:stream";
import { afterEach, describe, expect, it } from "vitest";
import { assertBlobKey, signBlobUrl } from "../src/blobs.ts";
import { startTestServer, type TestServer } from "./harness.ts";
import { settle } from "./wsClient.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

async function readAll(stream: Readable): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const c of stream) chunks.push(c as Buffer);
  return Buffer.concat(chunks);
}

describe("blob store", () => {
  it("round-trips 3 MB through signed PUT and GET URLs", async () => {
    server = await startTestServer();
    const data = randomBytes(3 * 1024 * 1024);
    const put = await fetch(signBlobUrl(server.ctx, "checkpoints/v1/checkpoint.tar", "PUT"), {
      method: "PUT",
      headers: { "content-type": "application/octet-stream" },
      body: data,
    });
    expect(put.status).toBe(204);
    const get = await fetch(signBlobUrl(server.ctx, "checkpoints/v1/checkpoint.tar", "GET"));
    expect(get.status).toBe(200);
    expect(Buffer.from(await get.arrayBuffer()).equals(data)).toBe(true);
    expect(await server.ctx.blobs.exists("checkpoints/v1/checkpoint.tar")).toBe(true);
  });

  it("refuses a URL signed for another method, an expired one and an unsigned one", async () => {
    server = await startTestServer();
    await server.ctx.blobs.put("a/b", Readable.from([Buffer.from("hi")]));
    const getUrl = signBlobUrl(server.ctx, "a/b", "GET", 60_000);
    expect((await fetch(getUrl, { method: "PUT", body: "x" })).status).toBe(403);
    expect((await fetch(`${server.url}/blobs/a/b`)).status).toBe(403);
    expect((await fetch(getUrl.replace("/a/b?", "/a/c?"))).status).toBe(403);
    expect((await fetch(getUrl)).status).toBe(200);
    await server.scheduler.advance(61_000);
    expect((await fetch(getUrl)).status).toBe(403);
    expect((await fetch(signBlobUrl(server.ctx, "a/missing", "GET"))).status).toBe(404);
  });

  it("rejects keys that could leave the store", async () => {
    server = await startTestServer();
    for (const key of ["../x", "a//..", "a/../b", "/abs", "A/upper", "a/./b", ""]) {
      expect(() => assertBlobKey(key), key).toThrow();
      expect(() => signBlobUrl(server?.ctx as never, key, "GET"), key).toThrow();
    }
    await expect(server.ctx.blobs.put("../x", Readable.from([Buffer.from("x")]))).rejects.toThrow();
    // URL parsing folds an encoded ".." away before the request leaves; it never reaches a blob.
    const url = signBlobUrl(server.ctx, "ok", "GET").replace("/blobs/ok?", "/blobs/%2e%2e/ok?");
    expect([403, 404]).toContain((await fetch(url)).status);
  });

  it("leaves nothing at the key when an upload stops halfway", async () => {
    server = await startTestServer();
    const url = new URL(signBlobUrl(server.ctx, "half/checkpoint.tar", "PUT"));
    await new Promise<void>((resolve) => {
      const req = request(url, {
        method: "PUT",
        headers: { "content-type": "application/octet-stream", "content-length": String(4 << 20) },
      });
      req.on("error", () => resolve());
      req.write(randomBytes(1 << 20), () => {
        setTimeout(() => {
          req.destroy();
          resolve();
        }, 50);
      });
    });
    await settle(200);
    expect(await server.ctx.blobs.exists("half/checkpoint.tar")).toBe(false);
    // The upload did reach the server (its folder exists), and its partial file is gone.
    expect(readdirSync(join(server.ctx.opts.blobDir, "half"))).toEqual([]);
  });
});
